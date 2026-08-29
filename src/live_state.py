"""Computes each team's CURRENT state (as of right now) for live inference --
distinct from build_features.py, which computes state "entering game X" for
historical training rows. Same underlying signals (ELO, SRS, rolling form,
ATS record), but here there's no future game to anchor to: we want each
team's freshest available rating/average given everything completed so far.
"""
from collections import defaultdict

import pandas as pd

from src.elo import CFBElo
from src.opponent_adjustment import PRIOR_SEASON_REGRESSION, _solve_srs

ROLLING_WINDOW = 4
ATS_WINDOW = 5


def compute_current_elo(completed_games: list[dict]) -> dict:
    """completed_games: chronologically-sortable dicts with year, start_date, home_team,
    away_team, home_points, away_points, neutral_site. Returns {team: current_elo}."""
    elo = CFBElo()
    for g in sorted(completed_games, key=lambda g: (g["year"], g["start_date"])):
        elo.maybe_regress_for_new_season(g["year"])
        elo.update(g["home_team"], g["away_team"], g["home_points"], g["away_points"], bool(g["neutral_site"]))
    return dict(elo.ratings)


def compute_current_srs(completed_games: list[dict], current_year: int) -> dict:
    """Current-season SRS where available, else prior season's final SRS regressed
    toward the mean (same rationale as the season-boundary handling in
    opponent_adjustment.compute_weekly_srs)."""
    def pairwise(games):
        rows = []
        for g in games:
            margin = g["home_points"] - g["away_points"]
            rows.append((g["home_team"], g["away_team"], margin))
            rows.append((g["away_team"], g["home_team"], -margin))
        return rows

    current_season_games = [g for g in completed_games
                             if g["year"] == current_year and g["season_type"] == "regular"]
    prior_season_games = [g for g in completed_games
                           if g["year"] == current_year - 1 and g["season_type"] == "regular"]

    current_srs = _solve_srs(pairwise(current_season_games)) if current_season_games else {}
    prior_srs = _solve_srs(pairwise(prior_season_games)) if prior_season_games else {}

    result = dict(current_srs)
    for team, rating in prior_srs.items():
        if team not in result:
            result[team] = rating * (1 - PRIOR_SEASON_REGRESSION)
    return result


def compute_current_rolling_form(long_df: pd.DataFrame, window: int = ROLLING_WINDOW) -> dict:
    """long_df: output of box_score_features.build_long_format + add_derived_rate_stats.
    Returns {team: {stat: trailing_avg}} over each team's last `window` completed games."""
    stat_cols = ["total_yards", "rushing_yards", "net_passing_yards", "yards_per_play",
                 "third_down_pct", "turnover_margin", "point_diff", "win"]
    result = {}
    for team, group in long_df.groupby("team"):
        recent = group.sort_values("start_date").tail(window)
        if recent.empty:
            continue
        result[team] = {stat: pd.to_numeric(recent[stat], errors="coerce").mean() for stat in stat_cols}
    return result


def compute_current_opponent_srs(long_df: pd.DataFrame, srs: dict, window: int = ROLLING_WINDOW) -> dict:
    """Average SRS (current, not historical-at-the-time) of each team's last `window` opponents."""
    result = {}
    for team, group in long_df.groupby("team"):
        recent = group.sort_values("start_date").tail(window)
        opp_ratings = [srs.get(opp, 0.0) for opp in recent["opponent"]]
        if opp_ratings:
            result[team] = sum(opp_ratings) / len(opp_ratings)
    return result


def compute_current_ats_pct(games_with_covers: list[dict], window: int = ATS_WINDOW) -> dict:
    """games_with_covers: dicts with team, start_date, covered (bool|None, None=push/excluded)."""
    by_team = defaultdict(list)
    for row in games_with_covers:
        by_team[row["team"]].append(row)
    result = {}
    for team, rows in by_team.items():
        rows.sort(key=lambda r: r["start_date"])
        decided = [r["covered"] for r in rows if r["covered"] is not None][-window:]
        if decided:
            result[team] = sum(decided) / len(decided)
    return result


def compute_current_rest_days(completed_games: list[dict], team: str, as_of_date, season_year: int,
                               default_days: int = 7) -> int:
    """Days since `team`'s most recent completed game IN season_year. Matches the training-time
    definition (ats_and_situational.compute_rest_days), which resets per year -- a team's first
    game of a new season always defaults to 7 (neutral), never counts back into last year's finale."""
    from datetime import date
    team_games = [g for g in completed_games
                  if g["year"] == season_year and (g["home_team"] == team or g["away_team"] == team)]
    if not team_games:
        return default_days
    last_date = max(date.fromisoformat(g["start_date"][:10]) for g in team_games)
    return (as_of_date - last_date).days
