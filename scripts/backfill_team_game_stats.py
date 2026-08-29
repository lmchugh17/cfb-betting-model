"""Backfill per-game team box score stats (yards, turnovers, 3rd down %, etc.) --
needed for rolling-form and Four-Factors-style features. Not pulled by backfill.py
since /games/teams requires a week param (unlike /games, /lines, /stats/player/season).

Only stores stats for games already in our (FBS-filtered) games table.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cfbd_client import CFBDClient
from src.db import get_connection, init_db

DEFAULT_START_YEAR = 2021
DEFAULT_END_YEAR = 2025

SIMPLE_FIELDS = {
    "firstDowns": ("first_downs", int),
    "totalYards": ("total_yards", int),
    "netPassingYards": ("net_passing_yards", int),
    "yardsPerPass": ("yards_per_pass", float),
    "rushingYards": ("rushing_yards", int),
    "rushingAttempts": ("rushing_attempts", int),
    "yardsPerRushAttempt": ("yards_per_rush", float),
    "rushingTDs": ("rushing_tds", int),
    "passingTDs": ("passing_tds", int),
    "turnovers": ("turnovers", int),
    "fumblesLost": ("fumbles_lost", int),
    "totalFumbles": ("total_fumbles", int),
    "fumblesRecovered": ("fumbles_recovered", int),
    "interceptions": ("interceptions", int),
    "passesIntercepted": ("passes_intercepted", int),
    "interceptionYards": ("interception_yards", int),
    "interceptionTDs": ("interception_tds", int),
    "sacks": ("sacks", int),
    "tacklesForLoss": ("tackles_for_loss", float),
    "tackles": ("tackles", int),
    "qbHurries": ("qb_hurries", int),
    "passesDeflected": ("passes_deflected", int),
    "defensiveTDs": ("defensive_tds", int),
    "kickReturns": ("kick_returns", int),
    "kickReturnYards": ("kick_return_yards", int),
    "kickReturnTDs": ("kick_return_tds", int),
    "puntReturns": ("punt_returns", int),
    "puntReturnYards": ("punt_return_yards", int),
    "puntReturnTDs": ("punt_return_tds", int),
    "kickingPoints": ("kicking_points", int),
}

# "made-attempted" or "count-yards" style fields
SPLIT_FIELDS = {
    "completionAttempts": ("completions", "pass_attempts"),
    "thirdDownEff": ("third_down_conversions", "third_down_attempts"),
    "fourthDownEff": ("fourth_down_conversions", "fourth_down_attempts"),
    "totalPenaltiesYards": ("penalties", "penalty_yards"),
}


def parse_stats(stat_list: list) -> dict:
    row = {}
    for s in stat_list:
        category, value = s.get("category"), s.get("stat")
        if value in (None, ""):
            continue
        if category in SIMPLE_FIELDS:
            col, caster = SIMPLE_FIELDS[category]
            try:
                row[col] = caster(value)
            except (ValueError, TypeError):
                pass
        elif category in SPLIT_FIELDS:
            col_a, col_b = SPLIT_FIELDS[category]
            parts = value.split("-")
            if len(parts) == 2:
                try:
                    row[col_a], row[col_b] = int(parts[0]), int(parts[1])
                except ValueError:
                    pass
        elif category == "possessionTime":
            parts = value.split(":")
            if len(parts) == 2:
                try:
                    row["possession_time_seconds"] = int(parts[0]) * 60 + int(parts[1])
                except ValueError:
                    pass
    return row


def upsert_game_teams(conn, games_teams_payload: list, known_game_ids: set) -> int:
    written = 0
    for game in games_teams_payload:
        if game["id"] not in known_game_ids:
            continue  # not one of our FBS games (e.g. FCS-vs-FCS matchup CFBD still lists here)
        for team_entry in game.get("teams", []):
            parsed = parse_stats(team_entry.get("stats", []))
            columns = ["game_id", "team_id", "team", "home_away", "points"] + list(parsed.keys())
            values = [game["id"], team_entry.get("teamId"), team_entry.get("team"),
                      team_entry.get("homeAway"), team_entry.get("points")] + list(parsed.values())
            placeholders = ", ".join("?" * len(values))
            conn.execute(
                f"INSERT OR REPLACE INTO team_game_stats ({', '.join(columns)}) VALUES ({placeholders})",
                values,
            )
            written += 1
    return written


def main():
    start_year = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_START_YEAR
    end_year = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_END_YEAR

    init_db()
    client = CFBDClient()
    conn = get_connection()

    try:
        for year in range(start_year, end_year + 1):
            for season_type in ("regular", "postseason"):
                weeks = [r[0] for r in conn.execute(
                    "SELECT DISTINCT week FROM games WHERE year=? AND season_type=? ORDER BY week",
                    (year, season_type),
                )]
                if not weeks:
                    continue
                known_game_ids = {r[0] for r in conn.execute(
                    "SELECT id FROM games WHERE year=? AND season_type=?", (year, season_type),
                )}
                print(f"{year} {season_type}: pulling {len(weeks)} weeks...")
                total = 0
                for week in weeks:
                    payload = client.games_teams(year, week, season_type)
                    total += upsert_game_teams(conn, payload, known_game_ids)
                    conn.commit()
                print(f"  wrote {total} team-game rows")

        count = conn.execute("SELECT COUNT(*) FROM team_game_stats").fetchone()[0]
        print(f"\nBackfill complete. team_game_stats total rows: {count}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
