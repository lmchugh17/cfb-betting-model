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
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import get_connection, init_db
from src.espn_client import fetch_roster_with_injuries, polite_sleep


def main():
    init_db()
    conn = get_connection()
    scraped_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        teams = conn.execute("SELECT id, school, espn_id FROM teams WHERE espn_id IS NOT NULL").fetchall()
        print(f"Scraping current injuries for {len(teams)} teams...")

        total_players = 0
        total_injuries = 0
        for team_id, school, espn_id in teams:
            try:
                players = fetch_roster_with_injuries(espn_id)
            except Exception as e:
                print(f"  WARN: {school} (espn {espn_id}) failed: {e}")
                continue
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

        print(f"\nScanned {total_players} players across {len(teams)} teams.")
        print(f"Recorded {total_injuries} current injury/suspension entries (snapshot: {scraped_at}).")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
