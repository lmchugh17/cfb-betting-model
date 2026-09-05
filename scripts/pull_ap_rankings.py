"""Pulls the AP Top 25 poll for a season and upserts it into ap_rankings.

Display-only (the site shows "No. N" next to a ranked team's name) -- not a trained
model feature. Cheap: one API call per season type (regular/postseason), not per-week,
since CFBD's own /rankings response already bundles every week's polls together.

Usage: .venv/bin/python scripts/pull_ap_rankings.py [year]
Defaults to the current year.
"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cfbd_client import CFBDClient
from src.db import get_connection, init_db


def upsert_ap_rankings(conn, weeks_data: list, year: int, season_type: str) -> int:
    written = 0
    for week_entry in weeks_data:
        week = week_entry["week"]
        ap_poll = next((p for p in week_entry["polls"] if p["poll"] == "AP Top 25"), None)
        if not ap_poll:
            continue
        for r in ap_poll["ranks"]:
            conn.execute(
                """INSERT INTO ap_rankings (year, week, season_type, team, rank)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(year, week, season_type, team) DO UPDATE SET rank=excluded.rank""",
                (year, week, season_type, r["school"], r["rank"]),
            )
            written += 1
    return written


def main():
    year = int(sys.argv[1]) if len(sys.argv) > 1 else date.today().year
    init_db()
    conn = get_connection()
    client = CFBDClient()

    try:
        total = 0
        for season_type in ("regular", "postseason"):
            print(f"Pulling {season_type} AP Top 25 rankings for {year}...")
            weeks_data = client.rankings(year, season_type)
            total += upsert_ap_rankings(conn, weeks_data, year, season_type)
        conn.commit()
        print(f"\nWrote/updated {total} (week, team) ranking rows for {year}.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
