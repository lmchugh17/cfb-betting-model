"""Live/current injury+suspension snapshot via ESPN's roster endpoint.

Best-effort: college football has no official league-wide injury report like
the NBA's, and ESPN's data here is manually reported per-team by beat writers,
so coverage is inconsistent -- some teams will show nothing even when a real
injury exists. Meant to be run weekly during the season (via the eventual
GitHub Actions cron), appending a new snapshot each time rather than
overwriting, so status changes over the season are visible.

Note: "injury" status here also catches some suspensions since ESPN doesn't
always distinguish them cleanly in this feed (e.g. "Suspension" appears as a
status value on some entries) -- worth a manual scan of results for now
rather than assuming full suspension coverage.
"""
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import get_connection, init_db
from src.espn_client import fetch_roster_with_injuries, polite_sleep

# Same circuit-breaker reasoning as scripts/pull_weather_forecast.py: this loop hits a
# free/unauthenticated API per-team (up to 138 sequential calls), the same shape of risk
# that turned out to hang on GitHub Actions for Open-Meteo. Bail after this many
# consecutive failures rather than grinding through every remaining team at the timeout.
MAX_CONSECUTIVE_FAILURES = 10


def main():
    init_db()
    conn = get_connection()
    scraped_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        teams = conn.execute("SELECT id, school, espn_id FROM teams WHERE espn_id IS NOT NULL").fetchall()
        # Shuffled so a bad run's failures don't always land on the same tail-end teams --
        # unshuffled, a circuit breaker tripping partway through would starve whichever
        # teams happen to sort last every single run, not spread the misses around.
        teams = list(teams)
        random.shuffle(teams)
        print(f"Scraping current injuries for {len(teams)} teams...", flush=True)

        total_players, total_injuries, consecutive_failures, remaining_teams = 0, 0, 0, 0
        for i, (team_id, school, espn_id) in enumerate(teams, 1):
            try:
                players = fetch_roster_with_injuries(espn_id)
                consecutive_failures = 0
            except Exception as e:
                consecutive_failures += 1
                print(f"  [{i}/{len(teams)}] WARN: {school} (espn {espn_id}) failed: {e}", flush=True)
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    remaining_teams = len(teams) - i
                    print(f"  {consecutive_failures} teams in a row failed -- stopping early "
                          f"({remaining_teams} team(s) skipped this run, will retry next pull). "
                          "Likely ESPN or the network path to it, not this script.", flush=True)
                    break
                continue
            if i % 20 == 0 or i == len(teams):
                print(f"  [{i}/{len(teams)}] teams scanned so far...", flush=True)
            total_players += len(players)
            for p in players:
                injuries = p.get("injuries") or []
                if not injuries:
                    continue
                latest = injuries[0]
                conn.execute(
                    """INSERT OR REPLACE INTO injuries
                       (espn_athlete_id, scraped_at, team_id, espn_team_id, player_name,
                        position, status, injury_date)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        p.get("id"), scraped_at, team_id, espn_id, p.get("fullName"),
                        (p.get("position") or {}).get("abbreviation"),
                        latest.get("status"), latest.get("date"),
                    ),
                )
                total_injuries += 1
            polite_sleep()
        conn.commit()

        print(f"\nScanned {total_players} players across {len(teams) - remaining_teams} teams.")
        print(f"Recorded {total_injuries} current injury/suspension entries (snapshot: {scraped_at}).")
        if remaining_teams:
            print(f"::warning::scrape_injuries.py only scanned some teams before its circuit "
                  f"breaker tripped ({remaining_teams} team(s) never attempted this run) -- "
                  f"likely ESPN rate-limiting GitHub Actions' shared IPs. Not fatal: injuries "
                  f"is append-only by scraped_at, so the next scheduled pull will retry the "
                  f"missed teams.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
