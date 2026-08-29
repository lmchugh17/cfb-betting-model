"""Backfill game_weather for all games currently in the games table.

Domes are marked is_dome=1 with null outdoor metrics (climate controlled --
weather isn't a meaningful feature there). Outdoor games are fetched from
Open-Meteo's historical archive, batched one API call per (venue, year) to
keep call volume low (~700 calls for a 5-season backfill instead of ~4,500).
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import get_connection, init_db
from src.weather_client import fetch_historical, nearest_hour, polite_sleep


def main():
    init_db()
    conn = get_connection()

    try:
        # Dome games: no weather to fetch, just record the flag.
        dome_games = conn.execute(
            """SELECT g.id FROM games g JOIN venues v ON g.venue_id = v.id
               WHERE v.dome = 1 AND g.id NOT IN (SELECT game_id FROM game_weather)"""
        ).fetchall()
        for (game_id,) in dome_games:
            conn.execute(
                "INSERT OR REPLACE INTO game_weather (game_id, is_dome, source) VALUES (?, 1, 'dome')",
                (game_id,),
            )
        conn.commit()
        print(f"Marked {len(dome_games)} dome games (no weather fetch needed).")

        # Outdoor games with known venue coordinates, grouped by (venue, year) to batch calls.
        rows = conn.execute(
            """SELECT g.id, g.venue_id, g.year, g.start_date, v.latitude, v.longitude
               FROM games g JOIN venues v ON g.venue_id = v.id
               WHERE v.dome = 0 AND v.latitude IS NOT NULL AND v.longitude IS NOT NULL
                 AND g.id NOT IN (SELECT game_id FROM game_weather)"""
        ).fetchall()

        groups = defaultdict(list)
        for game_id, venue_id, year, start_date, lat, lon in rows:
            groups[(venue_id, year, lat, lon)].append((game_id, start_date))

        print(f"Fetching weather for {len(groups)} venue-year combos covering {len(rows)} games...")
        fetched = 0
        skipped_no_data = 0
        for (venue_id, year, lat, lon), games in groups.items():
            dates = sorted(d[:10] for _, d in games)
            start_date, end_date = dates[0], dates[-1]
            try:
                hourly = fetch_historical(lat, lon, start_date, end_date)
            except Exception as e:
                print(f"  WARN: venue {venue_id} {year} ({start_date}..{end_date}) failed: {e}")
                continue
            for game_id, kickoff in games:
                weather = nearest_hour(hourly, kickoff)
                if weather is None:
                    skipped_no_data += 1
                    continue
                conn.execute(
                    """INSERT OR REPLACE INTO game_weather
                       (game_id, is_dome, temperature_f, wind_speed_mph, wind_direction_deg,
                        precipitation_in, humidity_pct, source)
                       VALUES (?, 0, ?, ?, ?, ?, ?, 'open-meteo-archive')""",
                    (game_id, weather["temperature_f"], weather["wind_speed_mph"],
                     weather["wind_direction_deg"], weather["precipitation_in"], weather["humidity_pct"]),
                )
                fetched += 1
            conn.commit()
            polite_sleep()

        print(f"Fetched weather for {fetched} games. {skipped_no_data} games had no matching hourly data.")

        total = conn.execute("SELECT COUNT(*) FROM game_weather").fetchone()[0]
        print(f"\ngame_weather total rows: {total}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
