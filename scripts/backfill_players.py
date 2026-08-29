"""Backfill player season stats and rosters, filtered to FBS teams.

Both /stats/player/season and /roster return every division (like /games
did) so we filter client-side against the FBS school names already stored
in the teams/team_seasons tables from backfill.py.

Usage: .venv/bin/python scripts/backfill_players.py [start_year] [end_year]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cfbd_client import CFBDClient
from src.db import get_connection, init_db

DEFAULT_START_YEAR = 2021
DEFAULT_END_YEAR = 2025


def fbs_school_names(conn, year: int) -> set:
    rows = conn.execute(
        "SELECT t.school FROM teams t JOIN team_seasons ts ON t.id = ts.team_id WHERE ts.year = ?",
        (year,),
    ).fetchall()
    return {r[0] for r in rows}


def upsert_player_stats(conn, stats: list, year: int, fbs_schools: set):
    kept = 0
    for s in stats:
        if s.get("team") not in fbs_schools:
            continue
        conn.execute(
            """INSERT OR REPLACE INTO player_season_stats
               (player_id, player_name, position, team, conference, year, category, stat_type, stat_value)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                s.get("playerId"), s.get("player"), s.get("position"), s.get("team"),
                s.get("conference"), year, s.get("category"), s.get("statType"),
                float(s["stat"]) if s.get("stat") not in (None, "") else None,
            ),
        )
        kept += 1
    return kept


def upsert_roster(conn, roster: list, year: int, fbs_schools: set):
    kept = 0
    for p in roster:
        if p.get("team") not in fbs_schools:
            continue
        conn.execute(
            """INSERT OR REPLACE INTO roster
               (player_id, year, first_name, last_name, team, position, jersey, height, weight,
                class_year, home_city, home_state)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                p.get("id"), year, p.get("firstName"), p.get("lastName"), p.get("team"),
                p.get("position"), p.get("jersey"), p.get("height"), p.get("weight"),
                p.get("year"), p.get("homeCity"), p.get("homeState"),
            ),
        )
        kept += 1
    return kept


def main():
    start_year = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_START_YEAR
    end_year = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_END_YEAR

    init_db()
    client = CFBDClient()
    conn = get_connection()

    try:
        for year in range(start_year, end_year + 1):
            fbs_schools = fbs_school_names(conn, year)
            if not fbs_schools:
                print(f"Skipping {year}: no FBS teams found in db (run scripts/backfill.py first)")
                continue

            print(f"Pulling player season stats for {year}...")
            stats = client.player_season_stats(year)
            kept = upsert_player_stats(conn, stats, year, fbs_schools)
            conn.commit()
            print(f"  kept {kept}/{len(stats)} rows (FBS only)")

            print(f"Pulling roster for {year}...")
            roster = client.roster(year)
            kept = upsert_roster(conn, roster, year, fbs_schools)
            conn.commit()
            print(f"  kept {kept}/{len(roster)} rows (FBS only)")

        counts = {}
        for table in ("player_season_stats", "roster"):
            counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print("\nBackfill complete. Row counts:")
        for table, count in counts.items():
            print(f"  {table}: {count}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
