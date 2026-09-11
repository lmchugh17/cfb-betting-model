"""Lean scores-only refresh for already-known games: 2 CFBD calls total
(games() for regular + postseason), vs. backfill.py's ~7+ calls per run.

Only updates games already in the DB (never adds/filters new ones -- that's
still backfill.py's job on the standing weekly_data_pull cadence). Exists so
results_refresh.yml can poll frequently, every day, without burning through
CFBD's 1,000-call/month free tier -- see feedback_sports_betting_model_architecture
memory for why the day-specific-checkpoint approach this replaces was fragile
(it assumed "CFB has no Monday games," which is false).

Usage: .venv/bin/python scripts/refresh_scores_only.py [year]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cfbd_client import CFBDClient
from src.db import get_connection, init_db

DEFAULT_YEAR = 2026


def upsert_scores(conn, games: list, year: int, season_type: str, known_ids: set) -> int:
    updated = 0
    for g in games:
        if g["id"] not in known_ids:
            continue  # new games are backfill.py's job, not this script's
        cur = conn.execute(
            """UPDATE games SET home_points = ?, away_points = ?, raw_json = ?
               WHERE id = ? AND year = ? AND season_type = ?""",
            (g.get("homePoints"), g.get("awayPoints"), json.dumps(g), g["id"], year, season_type),
        )
        updated += cur.rowcount
    return updated


def main():
    year = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_YEAR
    init_db()
    client = CFBDClient()
    conn = get_connection()
    try:
        known_ids = {r[0] for r in conn.execute("SELECT id FROM games WHERE year = ?", (year,))}
        for season_type in ("regular", "postseason"):
            games = client.games(year, season_type)
            upsert_scores(conn, games, year, season_type, known_ids)
            conn.commit()
        print("Scores refresh complete.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
