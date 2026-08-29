"""Against-the-spread record, rest/bye-week, and head-to-head features.

All computed with the same no-leakage discipline as ELO/SRS: a game's features
only ever use data from strictly earlier games (shift-by-one before any
rolling calculation).
"""
from collections import defaultdict
from datetime import date

ATS_ROLLING_WINDOW = 5  # short window: CFB has ~13 games/season, not the NBA's 82
BYE_WEEK_REST_DAYS = 10  # a normal week-to-week gap is ~7 days; 10+ signals a bye


def median_spread_per_game(lines_rows: list[tuple]) -> dict:
    """lines_rows: (game_id, spread) across all providers for a game.
    Returns {game_id: median_spread} -- median is more robust to one outlier
    book than a mean, and doesn't require picking a single 'reference' provider.
    """
    by_game = defaultdict(list)
    for game_id, spread in lines_rows:
        if spread is not None:
            by_game[game_id].append(spread)
    result = {}
    for game_id, spreads in by_game.items():
        spreads.sort()
        n = len(spreads)
        result[game_id] = spreads[n // 2] if n % 2 else (spreads[n // 2 - 1] + spreads[n // 2]) / 2
    return result


def compute_ats_results(games: list[dict], game_spreads: dict) -> list[dict]:
    """games: dicts with id, year, week, season_type, start_date, home_team, away_team,
    home_points, away_points. Returns one row per team per game with whether that
    team covered (True/False/None for push or no line available).
    """
    rows = []
    for g in games:
        spread = game_spreads.get(g["id"])
        if spread is None or g["home_points"] is None:
            continue
        cover_margin = (g["home_points"] - g["away_points"]) + spread
        home_covered = None if cover_margin == 0 else cover_margin > 0
        away_covered = None if home_covered is None else (not home_covered)
        rows.append({"game_id": g["id"], "year": g["year"], "week": g["week"],
                      "season_type": g["season_type"], "start_date": g["start_date"],
                      "team": g["home_team"], "covered": home_covered})
        rows.append({"game_id": g["id"], "year": g["year"], "week": g["week"],
                      "season_type": g["season_type"], "start_date": g["start_date"],
                      "team": g["away_team"], "covered": away_covered})
    return rows


def compute_rolling_ats_pct(ats_rows: list[dict]) -> dict:
    """Returns {(game_id, team): trailing_ats_pct_before_this_game}, using only
    each team's own prior games (any push is excluded from the denominator,
    matching standard ATS record convention)."""
    by_team = defaultdict(list)
    for row in ats_rows:
        by_team[row["team"]].append(row)

    result = {}
    for team, rows in by_team.items():
        rows.sort(key=lambda r: (r["year"], r["start_date"]))
        history = []  # covered/not, excluding pushes
        for row in rows:
            decided = [c for c in history[-ATS_ROLLING_WINDOW:]]
            result[(row["game_id"], team)] = (sum(decided) / len(decided)) if decided else None
            if row["covered"] is not None:
                history.append(row["covered"])
    return result


def _parse_date(iso_str: str) -> date:
    return date.fromisoformat(iso_str[:10])


def compute_rest_days(games: list[dict]) -> dict:
    """Returns {game_id: {'home_rest_days': int, 'away_rest_days': int,
    'home_bye_week': bool, 'away_bye_week': bool}}. First game of a team's
    season gets a default of 7 (a normal week's rest, i.e. no signal either way)."""
    last_played = {}
    result = {}
    games_sorted = sorted(games, key=lambda g: (g["year"], g["start_date"]))
    for g in games_sorted:
        game_date = _parse_date(g["start_date"])
        info = {}
        for side, team in (("home", g["home_team"]), ("away", g["away_team"])):
            key = (g["year"], team)
            rest_days = (game_date - last_played[key]).days if key in last_played else 7
            info[f"{side}_rest_days"] = rest_days
            info[f"{side}_bye_week"] = rest_days >= BYE_WEEK_REST_DAYS
        result[g["id"]] = info
        for side, team in (("home", g["home_team"]), ("away", g["away_team"])):
            last_played[(g["year"], team)] = game_date
    return result


def compute_h2h_features(games: list[dict], n_last: int = 5, min_meetings: int = 2) -> dict:
    """Returns {game_id: {'h2h_home_win_pct': float|None, 'h2h_avg_home_margin': float|None,
    'h2h_meetings': int}} using up to the last n_last meetings between the two teams,
    from ANY prior season, before this game's date. Requires min_meetings prior
    matchups or returns None (too little history to mean anything)."""
    games_sorted = sorted(games, key=lambda g: (g["year"], g["start_date"]))
    history = defaultdict(list)  # frozenset({teamA, teamB}) -> [(date, home_team, margin)]
    result = {}

    for g in games_sorted:
        pair_key = frozenset({g["home_team"], g["away_team"]})
        past = history[pair_key][-n_last:]
        if len(past) < min_meetings:
            result[g["id"]] = {"h2h_home_win_pct": None, "h2h_avg_home_margin": None,
                                "h2h_meetings": len(past)}
        else:
            wins, margins = 0, []
            for _, past_home, margin in past:
                # normalize each past margin to "from the perspective of the CURRENT home team"
                normalized = margin if past_home == g["home_team"] else -margin
                margins.append(normalized)
                if normalized > 0:
                    wins += 1
            result[g["id"]] = {"h2h_home_win_pct": wins / len(past),
                                "h2h_avg_home_margin": sum(margins) / len(margins),
                                "h2h_meetings": len(past)}

        if g["home_points"] is not None:
            history[pair_key].append((g["start_date"], g["home_team"], g["home_points"] - g["away_points"]))

    return result
