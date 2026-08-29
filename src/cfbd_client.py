"""Thin wrapper around the CollegeFootballData.com API with basic retry.

CFBD's gateway occasionally returns a transient 502 under load (observed
empirically, not documented) -- retry a couple of times before giving up.
"""
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.collegefootballdata.com"
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2
REQUEST_TIMEOUT_SECONDS = 90  # some endpoints (e.g. /stats/player/season) return 100k+ rows


class CFBDClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("CFBD_API_KEY")
        if not self.api_key:
            raise RuntimeError("CFBD_API_KEY not set (check .env)")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def get(self, path: str, params: dict | None = None) -> list | dict:
        url = f"{BASE_URL}{path}"
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            except requests.exceptions.RequestException as e:
                last_error = e
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                continue
            if resp.status_code < 500:
                resp.raise_for_status()
                return resp.json()
            last_error = requests.exceptions.HTTPError(
                f"{resp.status_code} error for {url} (attempt {attempt}/{MAX_RETRIES})"
            )
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
        raise last_error

    def teams_fbs(self, year: int) -> list:
        return self.get("/teams/fbs", {"year": year})

    def venues(self) -> list:
        return self.get("/venues")

    def coaches(self, min_year: int, max_year: int) -> list:
        return self.get("/coaches", {"minYear": min_year, "maxYear": max_year})

    def games(self, year: int, season_type: str = "regular") -> list:
        # No classification filter here: without it, /games returns every division
        # (FCS/D2/D3/NAIA included), so backfill.py filters client-side to games
        # involving at least one FBS team (confirmed equivalent to classification=fbs).
        return self.get("/games", {"year": year, "seasonType": season_type})

    def lines(self, year: int, season_type: str = "regular") -> list:
        return self.get("/lines", {"year": year, "seasonType": season_type})

    def player_season_stats(self, year: int, season_type: str = "regular") -> list:
        # Includes every division, same as /games -- filter client-side to FBS teams.
        return self.get("/stats/player/season", {"year": year, "seasonType": season_type})

    def roster(self, year: int) -> list:
        # Also includes every division -- filter client-side to FBS teams.
        return self.get("/roster", {"year": year})

    def games_teams(self, year: int, week: int, season_type: str = "regular") -> list:
        # Unlike /games, /lines, /stats/player/season -- this endpoint requires a week param.
        return self.get("/games/teams", {"year": year, "week": week, "seasonType": season_type})
