"""One-time backfill of teams, venues, coaches, games, and lines into SQLite.

Usage: .venv/bin/python scripts/backfill.py [start_year] [end_year]
Defaults to the last 5 completed seasons.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cfbd_client import CFBDClient
from src.db import get_connection, init_db

DEFAULT_START_YEAR = 2021
DEFAULT_END_YEAR = 2025


def upsert_teams(conn, teams: list, year: int):
    for t in teams:
        conn.execute(
            "INSERT OR REPLACE INTO teams (id, school, mascot, abbreviation) VALUES (?, ?, ?, ?)",
            (t["id"], t["school"], t.get("mascot"), t.get("abbreviation")),
        )
        loc = t.get("location") or {}
        conn.execute(
            """INSERT OR REPLACE INTO team_seasons (team_id, year, conference, classification, venue_id)
               VALUES (?, ?, ?, ?, ?)""",
            (t["id"], year, t.get("conference"), t.get("classification"), loc.get("id")),
        )


def upsert_venues(conn, venues: list):
    for v in venues:
        conn.execute(
            """INSERT OR REPLACE INTO venues
               (id, name, city, state, latitude, longitude, elevation, capacity, grass, dome)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                v["id"], v.get("name"), v.get("city"), v.get("state"),
                v.get("latitude"), v.get("longitude"), v.get("elevation"),
                v.get("capacity"), int(bool(v.get("grass"))), int(bool(v.get("dome"))),
            ),
        )


def upsert_coaches(conn, coaches: list):
    for c in coaches:
        for s in c.get("seasons", []):
            conn.execute(
                """INSERT OR REPLACE INTO coach_seasons
                   (first_name, last_name, school, year, games, wins, losses, ties,
                    preseason_rank, postseason_rank, srs, sp_overall)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    c.get("firstName"), c.get("lastName"), s.get("school"), s.get("year"),
                    s.get("games"), s.get("wins"), s.get("losses"), s.get("ties"),
                    s.get("preseasonRank"), s.get("postseasonRank"), s.get("srs"), s.get("spOverall"),
                ),
            )


def upsert_games(conn, games: list, year: int, season_type: str):
    # Caller is expected to have already filtered to games involving at least one FBS team.
    for g in games:
        conn.execute(
            """INSERT OR REPLACE INTO games
               (id, year, week, season_type, start_date, neutral_site, conference_game,
                venue_id, venue, home_id, home_team, home_conference, home_points,
                away_id, away_team, away_conference, away_points, raw_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                g["id"], year, g.get("week"), season_type, g.get("startDate"),
                int(bool(g.get("neutralSite"))), int(bool(g.get("conferenceGame"))),
                g.get("venueId"), g.get("venue"),
                g.get("homeId"), g.get("homeTeam"), g.get("homeConference"), g.get("homePoints"),
                g.get("awayId"), g.get("awayTeam"), g.get("awayConference"), g.get("awayPoints"),
                json.dumps(g),
            ),
        )


def upsert_lines(conn, line_entries: list, kept_game_ids: set):
    for entry in line_entries:
        game_id = entry.get("id")
        if game_id not in kept_game_ids:
            continue  # /lines occasionally references game ids /games doesn't return at all; drop to avoid orphans
        for line in entry.get("lines", []):
            provider = line.get("provider")
            if not provider:
                continue
            conn.execute(
                """INSERT OR REPLACE INTO lines
                   (game_id, provider, spread, spread_open, over_under, over_under_open,
                    home_moneyline, away_moneyline, formatted_spread)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    game_id, provider, line.get("spread"), line.get("spreadOpen"),
                    line.get("overUnder"), line.get("overUnderOpen"),
                    line.get("homeMoneyline"), line.get("awayMoneyline"), line.get("formattedSpread"),
                ),
            )


def main():
    start_year = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_START_YEAR
    end_year = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_END_YEAR

    init_db()
    client = CFBDClient()
    conn = get_connection()

    try:
        print("Pulling venues...")
        upsert_venues(conn, client.venues())
        conn.commit()

        print(f"Pulling coaches {start_year}-{end_year}...")
        upsert_coaches(conn, client.coaches(start_year, end_year))
        conn.commit()

        for year in range(start_year, end_year + 1):
            print(f"Pulling teams for {year}...")
            teams = client.teams_fbs(year)
            upsert_teams(conn, teams, year)
            conn.commit()
            fbs_team_ids = {t["id"] for t in teams}

            for season_type in ("regular", "postseason"):
                print(f"Pulling {season_type} games for {year}...")
                games = client.games(year, season_type)
                games = [g for g in games if g.get("homeId") in fbs_team_ids or g.get("awayId") in fbs_team_ids]
                upsert_games(conn, games, year, season_type)
                conn.commit()
                kept_game_ids = {g["id"] for g in games}

                print(f"Pulling {season_type} lines for {year}...")
                lines = client.lines(year, season_type)
                upsert_lines(conn, lines, kept_game_ids)
                conn.commit()

        counts = {}
        for table in ("teams", "team_seasons", "venues", "coach_seasons", "games", "lines"):
            counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print("\nBackfill complete. Row counts:")
        for table, count in counts.items():
            print(f"  {table}: {count}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
