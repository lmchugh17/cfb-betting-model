"""Pull current NCAAF odds from The Odds API and store as a timestamped snapshot.

Meant to run multiple times a week (Tue/Fri/Sat per the agreed cadence) so line
movement is visible, not just a single closing line. Costs 3 credits per run
(spreads + totals + h2h markets, us region) -- ~135/season at that cadence,
well under the free tier's 500/month, leaving room to share the same account
with an NFL model later.
"""
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import get_connection, init_db
from src.odds_client import OddsAPIClient

# CFBD and The Odds API occasionally use different short names for the same school.
SCHOOL_ALIASES = {
    "Southern Miss": "Southern Mississippi",
    "Sam Houston": "Sam Houston State",
    "Massachusetts": "UMass",
    "App State": "Appalachian State",
}


def normalize(name: str) -> str:
    """Strips diacritics (Hawai'i/San José -> Hawaii/San Jose) and apostrophes for matching."""
    stripped = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return stripped.replace("'", "").replace("’", "")


def build_team_lookup(conn) -> dict:
    """Maps normalized 'School Mascot' (as the Odds API names teams) -> our team id."""
    rows = conn.execute("SELECT id, school, mascot FROM teams").fetchall()
    lookup = {}
    for team_id, school, mascot in rows:
        lookup[normalize(f"{school} {mascot}")] = team_id
        alias = SCHOOL_ALIASES.get(school)
        if alias:
            lookup[normalize(f"{alias} {mascot}")] = team_id
    return lookup


def main():
    init_db()
    client = OddsAPIClient()
    conn = get_connection()
    scraped_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        team_lookup = build_team_lookup(conn)
        games, quota = client.get_odds()
        print(f"Pulled odds for {len(games)} games. Quota remaining: {quota['remaining']} "
              f"(this call cost {quota['last_cost']}).")

        unmatched_teams = set()
        rows_written = 0
        for game in games:
            home_id = team_lookup.get(normalize(game["home_team"]))
            away_id = team_lookup.get(normalize(game["away_team"]))
            if home_id is None:
                unmatched_teams.add(game["home_team"])
            if away_id is None:
                unmatched_teams.add(game["away_team"])

            for bookmaker in game.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    for outcome in market.get("outcomes", []):
                        conn.execute(
                            """INSERT OR REPLACE INTO live_odds
                               (odds_game_id, scraped_at, commence_time, home_team, away_team,
                                home_team_id, away_team_id, bookmaker, market, outcome_name, price, point)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                game["id"], scraped_at, game.get("commence_time"),
                                game["home_team"], game["away_team"], home_id, away_id,
                                bookmaker["key"], market["key"], outcome["name"],
                                outcome.get("price"), outcome.get("point"),
                            ),
                        )
                        rows_written += 1
        conn.commit()

        print(f"Wrote {rows_written} odds rows (snapshot: {scraped_at}).")
        if unmatched_teams:
            print(f"WARN: {len(unmatched_teams)} team names didn't match our teams table "
                  f"(likely non-FBS or name mismatch): {sorted(unmatched_teams)}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
