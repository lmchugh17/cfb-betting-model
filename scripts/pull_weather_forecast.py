"""Pulls forecast weather for upcoming games and writes/refreshes their game_weather
rows. Open-Meteo's forecast endpoint only covers ~16 days ahead (FORECAST_HORIZON_DAYS
below), so games further out than that get no row yet -- correctly falls back to "no
weather signal" in src.weather_features.is_adverse until a later pull gets close enough.

Meant to run on the same Tue/Fri/Sat cadence as the odds pull. A forecast taken a few
days out gets more accurate as kickoff approaches, so re-running this each cycle
deliberately overwrites the earlier (less accurate) forecast via game_weather's PK
(game_id only, no timestamp) -- unlike live_odds/injuries, this table wants "latest
known value," not a time series.

Also marks any newly-added dome games (scripts/backfill.py adds new games weekly but
doesn't touch game_weather) -- domes are climate-controlled and never need a forecast,
same convention as scripts/backfill_weather.py's initial historical pass.

Once games in this script's forecast-covered range complete, their weather here is a
forecast, not the actual recorded conditions -- close enough for live prediction, but
scripts/backfill_weather.py should be re-run before any future retrain so completed
games get corrected to their real recorded weather instead.

Usage: .venv/bin/python scripts/pull_weather_forecast.py
"""
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import get_connection, init_db
from src.weather_client import fetch_forecast, nearest_hour, polite_sleep

FORECAST_HORIZON_DAYS = 16  # matches src.weather_client.fetch_forecast's forecast_days cap

# Circuit breaker: if Open-Meteo (or the network path to it, e.g. GitHub Actions' shared
# runner IPs getting throttled -- confirmed 2026-09-02, a full week's ~130 venues hung for
# 13+ minutes there with zero completed requests) is having a bad day, fail fast and let
# next run's forecast pull catch up rather than grinding through every remaining venue at
# the (now shortened, but still nonzero) per-request timeout.
MAX_CONSECUTIVE_FAILURES = 10


def main():
    init_db()
    conn = get_connection()
    try:
        # Dome games added since the last backfill_weather.py/pull_weather_forecast.py run:
        # flag them, no fetch needed, ever.
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
        if dome_games:
            print(f"Marked {len(dome_games)} new dome game(s) (no forecast needed).")

        now = datetime.now(timezone.utc)
        horizon = (now + timedelta(days=FORECAST_HORIZON_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
        now_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")

        rows = conn.execute(
            """SELECT g.id, g.venue_id, g.start_date, v.latitude, v.longitude
               FROM games g JOIN venues v ON g.venue_id = v.id
               WHERE v.dome = 0 AND v.latitude IS NOT NULL AND v.longitude IS NOT NULL
                 AND g.home_points IS NULL AND g.start_date BETWEEN ? AND ?""",
            (now_iso, horizon),
        ).fetchall()

        groups = defaultdict(list)
        for game_id, venue_id, start_date, lat, lon in rows:
            groups[(venue_id, lat, lon)].append((game_id, start_date))

        print(f"Fetching forecast weather for {len(groups)} venue(s) covering {len(rows)} upcoming game(s)...",
              flush=True)
        fetched, skipped, consecutive_failures, remaining_venues = 0, 0, 0, 0
        for i, ((venue_id, lat, lon), games) in enumerate(groups.items(), 1):
            try:
                hourly = fetch_forecast(lat, lon)
                consecutive_failures = 0
            except Exception as e:
                consecutive_failures += 1
                print(f"  [{i}/{len(groups)}] WARN: venue {venue_id} forecast fetch failed: {e}", flush=True)
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    remaining_venues = len(groups) - i
                    print(f"  {consecutive_failures} venues in a row failed -- stopping early "
                          f"({remaining_venues} venue(s) skipped this run, will retry next pull). "
                          "Likely Open-Meteo or the network path to it, not this script.", flush=True)
                    break
                continue
            if i % 20 == 0 or i == len(groups):
                print(f"  [{i}/{len(groups)}] venues fetched so far...", flush=True)
            for game_id, kickoff in games:
                weather = nearest_hour(hourly, kickoff)
                if weather is None:
                    skipped += 1
                    continue
                conn.execute(
                    """INSERT OR REPLACE INTO game_weather
                       (game_id, is_dome, temperature_f, wind_speed_mph, wind_direction_deg,
                        precipitation_in, humidity_pct, source)
                       VALUES (?, 0, ?, ?, ?, ?, ?, 'open-meteo-forecast')""",
                    (game_id, weather["temperature_f"], weather["wind_speed_mph"],
                     weather["wind_direction_deg"], weather["precipitation_in"], weather["humidity_pct"]),
                )
                fetched += 1
            conn.commit()
            polite_sleep()

        print(f"Wrote forecast weather for {fetched} game(s). {skipped} fell outside the forecast's "
              f"hourly range (kickoff right at the {FORECAST_HORIZON_DAYS}-day edge).")
        if remaining_venues:
            # GitHub Actions renders a `::warning::` line as a yellow annotation on the run's
            # summary page, visible without failing the step -- deliberately not a nonzero
            # exit here, since that would abort the job before the "Commit and push" step and
            # throw away every other step's good data over one partial fetch. This just makes
            # a partial run visible instead of reporting an identical green checkmark to a
            # fully-successful one.
            print(f"::warning::pull_weather_forecast.py only fetched some venues before its "
                  f"circuit breaker tripped ({remaining_venues} venue(s) never attempted this run) "
                  f"-- likely Open-Meteo rate-limiting GitHub Actions' shared IPs. Not fatal: "
                  f"game_weather upserts by game_id, so the next scheduled pull will retry the "
                  f"missed venues. If this keeps happening across multiple runs, some upcoming "
                  f"games may be predicted without a weather signal for longer than intended.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
