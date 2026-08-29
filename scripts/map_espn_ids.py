"""One-time (re-runnable) mapping of our FBS teams to ESPN team IDs, via ESPN's
search API. Only re-resolves teams missing an espn_id, so it's safe to re-run.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import get_connection, init_db
from src.espn_client import find_team_espn_id, polite_sleep


def main():
    init_db()
    conn = get_connection()

    try:
        teams = conn.execute("SELECT id, school, mascot FROM teams WHERE espn_id IS NULL").fetchall()
        print(f"Resolving ESPN ids for {len(teams)} teams without one...")

        unresolved = []
        for team_id, school, mascot in teams:
            espn_id = find_team_espn_id(school, mascot)
            if espn_id:
                conn.execute("UPDATE teams SET espn_id = ? WHERE id = ?", (espn_id, team_id))
            else:
                unresolved.append(school)
            polite_sleep()
        conn.commit()

        resolved_count = conn.execute("SELECT COUNT(*) FROM teams WHERE espn_id IS NOT NULL").fetchone()[0]
        total = conn.execute("SELECT COUNT(*) FROM teams").fetchone()[0]
        print(f"\nResolved: {resolved_count}/{total} teams have an espn_id.")
        if unresolved:
            print(f"Could not resolve ({len(unresolved)}), needs manual review: {unresolved}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
