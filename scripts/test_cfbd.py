"""Smoke test: confirms the CFBD API key works before building the real pipeline."""
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("CFBD_API_KEY")
BASE_URL = "https://api.collegefootballdata.com"


def main():
    if not API_KEY:
        sys.exit("CFBD_API_KEY not set. Copy .env.example to .env and add your key from https://collegefootballdata.com/key")

    headers = {"Authorization": f"Bearer {API_KEY}"}

    resp = requests.get(f"{BASE_URL}/teams/fbs", params={"year": 2025}, headers=headers)
    resp.raise_for_status()
    teams = resp.json()
    print(f"Pulled {len(teams)} FBS teams for 2025.")
    print("Sample:", teams[0]["school"], "-", teams[0].get("conference"))

    resp = requests.get(f"{BASE_URL}/games", params={"year": 2025, "week": 1, "seasonType": "regular"}, headers=headers)
    resp.raise_for_status()
    games = resp.json()
    print(f"Pulled {len(games)} games for 2025 week 1.")
    if games:
        g = games[0]
        print("Sample:", g.get("awayTeam"), "@", g.get("homeTeam"), "-", g.get("startDate"))


if __name__ == "__main__":
    main()
