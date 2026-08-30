"""Team-level ATS performance in adverse-weather games, plus the "is this game
itself in adverse weather" flag. Same no-leakage discipline as ELO/SRS/ATS: a
team's split only ever uses its own strictly prior games.

Adverse weather is defined as freezing temperature, strong sustained wind, or
meaningful precipitation -- conditions plausibly tied to passing/kicking-game
degradation that a team's overall rating doesn't already capture on its own.
These games are sparse (most games aren't played in adverse weather), so the
rolling window here is longer than the general ATS_ROLLING_WINDOW in
ats_and_situational.py and routinely spans multiple seasons for a given team.

Training-time (build_features.py) uses each game's own actually-recorded weather
(from game_weather, backfilled via Open-Meteo's archive API) as a proxy for what a
pre-game forecast would have shown -- a reasonable simplification since forecasts
close to kickoff are typically accurate for temperature/wind, but a simplification
worth naming rather than treating as ground truth. Live inference (predict_games.py)
uses src.weather_client.fetch_forecast via scripts/pull_weather_forecast.py instead,
which only covers ~16 days out -- games further out than that have no weather row
yet and the feature correctly falls back to "not adverse" (0) until a forecast pull
gets close enough to know.
"""
from collections import defaultdict

ADVERSE_TEMP_F = 40        # cold enough to plausibly affect grip/ball flight, not just literal freezing
ADVERSE_WIND_MPH = 20      # sustained wind strong enough to affect passing/kicking
ADVERSE_PRECIP_IN = 0.1    # meaningful precipitation, not just a trace reading
ADVERSE_WX_ROLLING_WINDOW = 8  # longer than the general 5-game ATS window -- adverse games are rare
# Measured against the 2021-2025 backfill before picking these numbers. The strict
# textbook thresholds (32F/20mph, min 3 prior games) leave only 6 games in 5 seasons
# where BOTH teams have a usable adverse-weather ATS history -- nowhere near enough for
# a model to learn from. Loosening wind to 15mph fixed the sample-size problem (191
# games) but pulled in a lot of merely-breezy warm days as "adverse" (125 of 361
# flagged games were wind-only at that threshold) -- not what bettors mean by bad
# weather. These thresholds (40F/20mph/0.1in, min 2 prior games) land in between: 263
# adverse games, 110 with both teams' history known -- a clear minority of the
# 4,547-game training set (correctly reflecting that most games aren't in meaningfully
# bad weather), enough to be learnable, without diluting the signal with mild days.
MIN_ADVERSE_GAMES = 2      # below this, treat the team's split as unknown rather than trust a noisy small sample


def is_adverse(temperature_f, wind_speed_mph, precipitation_in) -> bool:
    if temperature_f is not None and temperature_f <= ADVERSE_TEMP_F:
        return True
    if wind_speed_mph is not None and wind_speed_mph >= ADVERSE_WIND_MPH:
        return True
    if precipitation_in is not None and precipitation_in >= ADVERSE_PRECIP_IN:
        return True
    return False


def load_weather_by_game(conn) -> dict:
    """Returns {game_id: {'temperature_f', 'wind_speed_mph', 'precipitation_in', 'is_dome'}}."""
    rows = conn.execute(
        "SELECT game_id, is_dome, temperature_f, wind_speed_mph, precipitation_in FROM game_weather"
    ).fetchall()
    return {gid: {"is_dome": bool(dome), "temperature_f": t, "wind_speed_mph": w, "precipitation_in": p}
            for gid, dome, t, w, p in rows}


def was_game_adverse(game_id: int, weather_by_game: dict) -> bool:
    wx = weather_by_game.get(game_id)
    if not wx or wx["is_dome"]:
        return False  # domes are climate-controlled -- never adverse, regardless of outdoor conditions
    return is_adverse(wx["temperature_f"], wx["wind_speed_mph"], wx["precipitation_in"])


def compute_adverse_wx_ats_pct(ats_rows: list[dict], weather_by_game: dict) -> dict:
    """Training-time: returns {(game_id, team): trailing_adverse_wx_ats_pct_before_this_game}.
    ats_rows: same shape as ats_and_situational.compute_ats_results output (needs game_id,
    year, start_date, team, covered)."""
    by_team = defaultdict(list)
    for row in ats_rows:
        by_team[row["team"]].append(row)

    result = {}
    for team, rows in by_team.items():
        rows.sort(key=lambda r: (r["year"], r["start_date"]))
        history = []
        for row in rows:
            decided = history[-ADVERSE_WX_ROLLING_WINDOW:]
            result[(row["game_id"], team)] = (
                sum(decided) / len(decided) if len(decided) >= MIN_ADVERSE_GAMES else None
            )
            if was_game_adverse(row["game_id"], weather_by_game) and row["covered"] is not None:
                history.append(row["covered"])
    return result


def compute_current_adverse_wx_ats_pct(ats_rows: list[dict], weather_by_game: dict) -> dict:
    """Live inference: returns {team: current_adverse_wx_ats_pct} using everything
    completed so far, no game to anchor to (mirrors src/live_state.py's convention)."""
    by_team = defaultdict(list)
    for row in ats_rows:
        by_team[row["team"]].append(row)

    result = {}
    for team, rows in by_team.items():
        rows.sort(key=lambda r: r["start_date"])
        decided = [r["covered"] for r in rows
                   if r["covered"] is not None and was_game_adverse(r["game_id"], weather_by_game)]
        decided = decided[-ADVERSE_WX_ROLLING_WINDOW:]
        if len(decided) >= MIN_ADVERSE_GAMES:
            result[team] = sum(decided) / len(decided)
    return result
