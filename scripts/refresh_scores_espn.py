"""Fill in final scores ESPN already has but CFBD hasn't posted yet.

CFBD stays the source of record: scripts/refresh_scores_only.py still runs first on the
same schedule, and the weekly pull rewrites everything from CFBD later. This only fills
gaps, because CFBD's finals can lag hours behind the game (measured 2026-09-19: Florida
State @ Alabama was Final on ESPN while CFBD still returned completed=false and no points
5.5 hours after kickoff).

Deliberately conservative, since ESPN's site API is undocumented and can change:
- only games already in our DB, matched on both teams and the date,
- only events ESPN marks completed, with both scores present,
- never writes a NULL over an existing score, and never adds a game.

Any network/parse failure is reported and exits 0 -- the CFBD path already ran, and this
step must never fail the results-refresh workflow (a free API blocking GitHub Actions' IPs
is a real failure mode here, confirmed for Open-Meteo on 2026-09-02).

Usage: .venv/bin/python scripts/refresh_scores_espn.py [year] [--days N] [--dry-run]
"""
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import get_connection, init_db
from src.espn_client import fetch_scoreboard
from src.team_names import normalize

DEFAULT_YEAR = 2026
DEFAULT_DAYS = 2  # today and yesterday (UTC), enough to catch late-finishing night games


def team_id_lookup(conn) -> tuple[dict, dict]:
    """ESPN team id -> our team id, plus normalized school name -> our team id as a fallback
    for the one team without a cached espn_id (and for any ESPN id that changes)."""
    by_espn, by_name = {}, {}
    for team_id, school, espn_id in conn.execute("SELECT id, school, espn_id FROM teams"):
        if espn_id:
            by_espn[str(espn_id)] = team_id
        by_name[normalize(school)] = team_id
    return by_espn, by_name


def resolve(competitor: dict, by_espn: dict, by_name: dict):
    team = competitor.get("team", {})
    our_id = by_espn.get(str(team.get("id")))
    if our_id is None:
        our_id = by_name.get(normalize(team.get("location") or ""))
    return our_id


def main():
    args = [a for a in sys.argv[1:]]
    dry_run = "--dry-run" in args
    args = [a for a in args if a != "--dry-run"]
    days = DEFAULT_DAYS
    if "--days" in args:
        i = args.index("--days")
        days = int(args[i + 1])
        del args[i:i + 2]
    year = int(args[0]) if args else DEFAULT_YEAR

    init_db()
    conn = get_connection()
    try:
        by_espn, by_name = team_id_lookup(conn)
        events = []
        for offset in range(days):
            day = (date.today() - timedelta(days=offset)).strftime("%Y%m%d")
            try:
                events.extend(fetch_scoreboard(day))
            except Exception as exc:  # network error, throttling, changed payload
                print(f"WARN: ESPN scoreboard unavailable for {day} ({exc}) -- leaving CFBD's scores alone")
        updated, matched, unmatched = 0, 0, []
        for event in events:
            try:
                comp = event["competitions"][0]
                if not comp.get("status", event.get("status", {})).get("type", {}).get("completed"):
                    continue
                sides = {c.get("homeAway"): c for c in comp.get("competitors", [])}
                home, away = sides.get("home"), sides.get("away")
                if not home or not away:
                    continue
                home_pts, away_pts = home.get("score"), away.get("score")
                if home_pts is None or away_pts is None:
                    continue
                home_id, away_id = resolve(home, by_espn, by_name), resolve(away, by_espn, by_name)
                kickoff = event.get("date", "")[:10]  # ESPN returns UTC, same as games.start_date
                if not kickoff or (home_id is None and away_id is None):
                    unmatched.append(event.get("shortName", "?"))
                    continue
                # One side is enough when the other is an FCS/non-FBS team we don't track (very
                # common early season: UNI @ Iowa). A team plays at most once on a given date, so
                # requiring exactly one matching row keeps this unambiguous.
                clauses, params = ["year = ?"], [year]
                if home_id is not None:
                    clauses.append("home_id = ?"); params.append(home_id)
                if away_id is not None:
                    clauses.append("away_id = ?"); params.append(away_id)
                clauses.append("DATE(start_date) BETWEEN DATE(?, '-1 day') AND DATE(?, '+1 day')")
                params += [kickoff, kickoff]
                rows = conn.execute(
                    f"SELECT id, home_points, away_points FROM games WHERE {' AND '.join(clauses)}",
                    params).fetchall()
                if len(rows) != 1:
                    unmatched.append(event.get("shortName", "?"))
                    continue
                row = rows[0]
                matched += 1
                game_id, have_home, have_away = row
                new_home, new_away = int(home_pts), int(away_pts)
                if (have_home, have_away) == (new_home, new_away):
                    continue
                print(f"{'would update' if dry_run else 'updating'} {event.get('shortName')}: "
                      f"{have_away}-{have_home} -> {new_away}-{new_home}")
                if not dry_run:
                    conn.execute("UPDATE games SET home_points = ?, away_points = ? WHERE id = ?",
                                 (new_home, new_away, game_id))
                updated += 1
            except Exception as exc:
                print(f"WARN: skipped an ESPN event ({exc})")
        if not dry_run:
            conn.commit()
        print(f"ESPN fill-in{' (dry run)' if dry_run else ''}: {len(events)} events, {matched} matched our games, "
              f"{updated} score(s) written, {len(unmatched)} unmatched (non-FBS or not in our schedule)")
        if unmatched:
            print("  unmatched sample: " + ", ".join(unmatched[:8]))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
