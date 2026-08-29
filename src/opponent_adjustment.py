"""Opponent-adjusted team ratings via iterative SRS (Simple Rating System:
rating = average scoring margin + average opponent rating, solved iteratively
to convergence). This is the piece the reference NBA model does NOT have --
its rolling box-score features are raw/unadjusted, relying entirely on ELO to
absorb opponent quality. That's not good enough for CFB: a Group of 5 team's
raw per-game stats aren't comparable to a Power 4 team's without adjusting for
who they actually played.

Computed strictly chronologically (per season, using only games from weeks
already completed) so no game's rating ever reflects a game that hasn't been
played yet -- the same no-leakage discipline as the ELO module.

Opponents we don't track (FCS/non-FBS "buy game" opponents) are treated as a
fixed replacement-level rating of 0.0 (i.e., average) rather than excluded --
a close game against a weak FCS opponent is real signal that a team is
struggling, and dropping those games would throw that away.
"""
from collections import defaultdict

MAX_ITERATIONS = 25
CONVERGENCE_THRESHOLD = 0.01
PRIOR_SEASON_REGRESSION = 0.5  # how much of last season's final SRS carries into week 1


def _solve_srs(games: list[tuple]) -> dict:
    """games: list of (team, opponent, margin) from ONE team's perspective each
    (i.e. each real game contributes two entries, one per side)."""
    teams = {g[0] for g in games} | {g[1] for g in games}
    ratings = {t: 0.0 for t in teams}
    by_team = defaultdict(list)
    for team, opponent, margin in games:
        by_team[team].append((opponent, margin))

    for _ in range(MAX_ITERATIONS):
        max_delta = 0.0
        new_ratings = {}
        for team in teams:
            matchups = by_team[team]
            if not matchups:
                new_ratings[team] = 0.0
                continue
            new_ratings[team] = sum(margin + ratings.get(opp, 0.0) for opp, margin in matchups) / len(matchups)
            max_delta = max(max_delta, abs(new_ratings[team] - ratings[team]))
        ratings = new_ratings
        if max_delta < CONVERGENCE_THRESHOLD:
            break
    return ratings


def compute_weekly_srs(games_rows: list[dict]) -> dict:
    """games_rows: dicts with keys year, week, season_type, home_team, away_team,
    home_points, away_points (regular season only -- postseason excluded from
    SRS since bowl participation itself is a non-random, strength-correlated
    signal that would bias the rating).

    Returns {(year, week, team): srs_entering_that_week}. Week 1 of a season
    uses the prior season's final SRS, regressed 50% toward 0, as its prior
    (mirrors the ELO module's between-season regression, same roster-turnover
    rationale) -- 2021 (our earliest backfilled season) has no prior, so it
    starts at 0 for everyone.
    """
    by_season = defaultdict(list)
    for g in games_rows:
        if g["season_type"] != "regular" or g["home_points"] is None or g["away_points"] is None:
            continue
        by_season[g["year"]].append(g)

    result = {}
    prior_final_srs = {}
    for year in sorted(by_season.keys()):
        season_games = sorted(by_season[year], key=lambda g: g["week"])
        weeks = sorted({g["week"] for g in season_games})

        for week in weeks:
            completed = [g for g in season_games if g["week"] < week]
            pairwise = []
            for g in completed:
                margin = g["home_points"] - g["away_points"]
                pairwise.append((g["home_team"], g["away_team"], margin))
                pairwise.append((g["away_team"], g["home_team"], -margin))

            srs = _solve_srs(pairwise) if pairwise else {}

            teams_this_week = {g["home_team"] for g in season_games if g["week"] == week} | \
                               {g["away_team"] for g in season_games if g["week"] == week}
            for team in teams_this_week:
                if team in srs:
                    result[(year, week, team)] = srs[team]
                elif team in prior_final_srs:
                    result[(year, week, team)] = prior_final_srs[team] * (1 - PRIOR_SEASON_REGRESSION)
                else:
                    result[(year, week, team)] = 0.0

        # Final SRS of the season (using ALL of that season's games) becomes next season's prior.
        all_pairwise = []
        for g in season_games:
            margin = g["home_points"] - g["away_points"]
            all_pairwise.append((g["home_team"], g["away_team"], margin))
            all_pairwise.append((g["away_team"], g["home_team"], -margin))
        prior_final_srs = _solve_srs(all_pairwise)

    return result
