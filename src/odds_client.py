"""The Odds API client. Free tier: 500 credits/month, cost = markets x regions per call."""
import os

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.the-odds-api.com/v4"


class OddsAPIClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("ODDS_API_KEY")
        if not self.api_key:
            raise RuntimeError("ODDS_API_KEY not set (check .env)")

    def get_odds(self, sport: str = "americanfootball_ncaaf", regions: str = "us",
                 markets: str = "spreads,totals,h2h") -> tuple[list, dict]:
        """Returns (games, quota_info). quota_info has 'remaining' and 'used' from response headers."""
        resp = requests.get(
            f"{BASE_URL}/sports/{sport}/odds",
            params={"apiKey": self.api_key, "regions": regions, "markets": markets,
                    "oddsFormat": "american", "dateFormat": "iso"},
            timeout=30,
        )
        resp.raise_for_status()
        quota = {
            "remaining": resp.headers.get("x-requests-remaining"),
            "used": resp.headers.get("x-requests-used"),
            "last_cost": resp.headers.get("x-requests-last"),
        }
        return resp.json(), quota
