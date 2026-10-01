"""Pulls preseason team context (talent composite + returning production) into
team_talent and returning_production.

Both are season-level, so this is one CFBD call per endpoint per year -- the weekly pull
only needs the current year, and only until the rows exist (CFBD publishes both before
the season starts). Pass a range to backfill history.

Usage: .venv/bin/python scripts/pull_preseason_team_data.py [first_year [last_year]]
Defaults to the current year only. --if-missing skips a year whose rows already exist.
"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cfbd_client import CFBDClient
from src.db import get_connection, init_db

TALENT_FIRST_YEAR = 2015  # CFBD /talent returns nothing before this


def has_rows(conn, table: str, year: int) -> bool:
    return conn.execute(f"SELECT 1 FROM {table} WHERE year = ? LIMIT 1", (year,)).fetchone() is not None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if_missing = "--if-missing" in sys.argv
    first = int(args[0]) if args else date.today().year
    last = int(args[1]) if len(args) > 1 else first
    init_db()
    conn = get_connection()
    client = CFBDClient()
    try:
        for year in range(first, last + 1):
            if year >= TALENT_FIRST_YEAR and not (if_missing and has_rows(conn, "team_talent", year)):
                rows = client.get("/talent", {"year": year})
                conn.executemany(
                    """INSERT INTO team_talent (year, team, talent) VALUES (?, ?, ?)
                       ON CONFLICT(year, team) DO UPDATE SET talent=excluded.talent""",
                    [(year, r["team"], r["talent"]) for r in rows])
                print(f"{year}: {len(rows)} talent rows")
            if not (if_missing and has_rows(conn, "returning_production", year)):
                rows = client.get("/player/returning", {"year": year})
                conn.executemany(
                    """INSERT INTO returning_production (year, team, percent_ppa, percent_passing_ppa,
                           percent_rushing_ppa, percent_receiving_ppa, usage)
                       VALUES (?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(year, team) DO UPDATE SET percent_ppa=excluded.percent_ppa,
                           percent_passing_ppa=excluded.percent_passing_ppa,
                           percent_rushing_ppa=excluded.percent_rushing_ppa,
                           percent_receiving_ppa=excluded.percent_receiving_ppa, usage=excluded.usage""",
                    [(year, r["team"], r["percentPPA"], r["percentPassingPPA"], r["percentRushingPPA"],
                      r["percentReceivingPPA"], r["usage"]) for r in rows])
                print(f"{year}: {len(rows)} returning-production rows")
            conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    main()
