"""Pull every FBS team's primary and alternate colors from CFBD into data/team_colors.json,
for the site's team-color theme picker (scripts/build_site.py). One API call; colors rarely
change, so this is run by hand when needed, not on the scheduled data pull.

Usage: .venv/bin/python scripts/pull_team_colors.py [year]
"""
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cfbd_client import CFBDClient

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "team_colors.json"


def main():
    year = int(sys.argv[1]) if len(sys.argv) > 1 else date.today().year
    teams = CFBDClient().teams_fbs(year)
    colors = {
        t["school"]: {"mascot": t.get("mascot"), "color": t.get("color"), "alternate_color": t.get("alternateColor")}
        for t in teams if t.get("color")
    }
    OUTPUT_PATH.write_text(json.dumps(dict(sorted(colors.items())), indent=1) + "\n")
    print(f"Wrote {len(colors)} FBS teams' colors ({year}) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
