"""Backfill per-game team box score stats (yards, turnovers, 3rd down %, etc.) --
needed for rolling-form and Four-Factors-style features. Not pulled by backfill.py
since /games/teams requires a week param (unlike /games, /lines, /stats/player/season).

Only stores stats for games already in our (FBS-filtered) games table.

Usage: .venv/bin/python scripts/backfill_team_game_stats.py [start_year] [end_year] [--all]
Each week costs one CFBD call per season_type, so by default this only re-pulls weeks that
still have completed games without box scores, plus the newest completed week (CFBD fills
those in over several hours). --all re-pulls every week, for a periodic full sweep.
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


def weeks_needing_stats(conn, year: int, season_type: str, all_weeks: list) -> list:
    """Weeks actually worth re-pulling: any week that still has a completed game with no
    box-score rows, plus the newest completed week (CFBD fills those in over several hours).
    Weeks whose games all have stats are final and cost a CFBD call to re-download for nothing --
    on the daily pull that was every week of the season, every day (~16 calls/day by December
    against a 1,000-call month). `--all` forces the old behaviour for a periodic full sweep."""
    rows = conn.execute(
        """SELECT g.week, SUM(t.game_id IS NULL) AS missing
           FROM games g
           LEFT JOIN (SELECT DISTINCT game_id FROM team_game_stats) t ON t.game_id = g.id
           WHERE g.year = ? AND g.season_type = ? AND g.home_points IS NOT NULL
           GROUP BY g.week""",
        (year, season_type),
    ).fetchall()
    if not rows:
        return []
    needed = {week for week, missing in rows if missing}
    needed.add(max(week for week, _ in rows))
    return [w for w in all_weeks if w in needed]


def main():
    args = [a for a in sys.argv[1:] if a != "--all"]
    pull_all = "--all" in sys.argv[1:]
    start_year = int(args[0]) if args else DEFAULT_START_YEAR
    end_year = int(args[1]) if len(args) > 1 else DEFAULT_END_YEAR

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
                if not pull_all:
                    skipped = len(weeks)
                    weeks = weeks_needing_stats(conn, year, season_type, weeks)
                    skipped -= len(weeks)
                    if not weeks:
                        print(f"{year} {season_type}: every week already has box scores, nothing to pull")
                        continue
                    print(f"{year} {season_type}: pulling weeks {weeks} "
                          f"({skipped} complete week(s) skipped, 1 CFBD call each)")
                known_game_ids = {r[0] for r in conn.execute(
                    "SELECT id FROM games WHERE year=? AND season_type=?", (year, season_type),
                )}
                if pull_all:
                    print(f"{year} {season_type}: pulling all {len(weeks)} weeks (--all)...")
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
