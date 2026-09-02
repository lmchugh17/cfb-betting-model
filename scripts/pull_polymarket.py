"""Pulls pre-game Polymarket win probabilities for upcoming games and writes/
refreshes their polymarket_odds rows -- a passive accuracy benchmark, not a live
pick source (see src/polymarket_client.py's docstring for the scope decision).

Meant to run on the same Tue/Fri/Sat cadence as the odds pull. Re-running this each
cycle deliberately overwrites the earlier (less accurate, further from kickoff)
probability via polymarket_odds' PK (game_id only, no timestamp) -- same
"latest known value" convention as game_weather/pull_weather_forecast.py.

Usage: .venv/bin/python scripts/pull_polymarket.py
"""
import sys
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import get_connection, init_db
from src.polymarket_client import extract_moneyline, fetch_cfb_events
from src.team_names import build_school_only_lookup, normalize

# Games more than ~2 weeks out rarely have a Polymarket game-level market open yet
# (confirmed empirically 2026-09-02: this week's slate opened ~1 week before kickoff).
HORIZON_DAYS = 16


def main():
    init_db()
    conn = get_connection()
    scraped_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        now = datetime.now(timezone.utc)
        start_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_iso = (now + timedelta(days=HORIZON_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")

        print(f"Fetching Polymarket CFB events ({start_iso} to {end_iso})...")
        try:
            events = fetch_cfb_events(start_iso, end_iso)
        except Exception as e:
            print(f"WARN: Polymarket fetch failed entirely: {e} -- nothing to write this run, "
                  "next scheduled pull will retry.")
            return
        print(f"Got {len(events)} event(s).")

        team_lookup = build_school_only_lookup(conn)
        games = conn.execute(
            """SELECT id, home_team, away_team, start_date FROM games
               WHERE home_points IS NULL AND start_date BETWEEN ? AND ?""",
            (start_iso, end_iso),
        ).fetchall()
        # Match our games to Polymarket events by (home_team_id, away_team_id) pair,
        # not by date/slug string-matching -- more robust to timezone/slug-format quirks.
        games_by_teams = {}
        for game_id, home, away, start_date in games:
            home_id = team_lookup.get(normalize(home))
            away_id = team_lookup.get(normalize(away))
            if home_id is not None and away_id is not None:
                games_by_teams[(home_id, away_id)] = game_id

        matched, unmatched_teams = 0, set()
        for event in events:
            ml = extract_moneyline(event)
            if ml is None:
                continue
            id_a = team_lookup.get(normalize(ml["team_a"]))
            id_b = team_lookup.get(normalize(ml["team_b"]))
            if id_a is None:
                unmatched_teams.add(ml["team_a"])
            if id_b is None:
                unmatched_teams.add(ml["team_b"])
            if id_a is None or id_b is None:
                continue
            # Polymarket's outcome order isn't guaranteed to be home-first -- resolve
            # against our own games table, which knows which team is actually home.
            game_id = games_by_teams.get((id_a, id_b))
            if game_id is not None:
                home_prob, away_prob = ml["prob_a"], ml["prob_b"]
            else:
                game_id = games_by_teams.get((id_b, id_a))
                if game_id is None:
                    continue
                home_prob, away_prob = ml["prob_b"], ml["prob_a"]
            conn.execute(
                """INSERT OR REPLACE INTO polymarket_odds
                   (game_id, scraped_at, polymarket_event_id, home_prob, away_prob)
                   VALUES (?, ?, ?, ?, ?)""",
                (game_id, scraped_at, event.get("id"), home_prob, away_prob),
            )
            matched += 1
        conn.commit()

        print(f"Matched and wrote {matched} game(s) to polymarket_odds.")
        if unmatched_teams:
            print(f"WARN: {len(unmatched_teams)} Polymarket team name(s) didn't match our teams "
                  f"table (likely non-FBS or name mismatch): {sorted(unmatched_teams)}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
