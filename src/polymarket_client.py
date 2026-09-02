"""Polymarket's public Gamma API (no auth needed for read-only market data).

Scope, confirmed with the user (2026-09-02): this is a PASSIVE POST-GAME ACCURACY
CHECK, not live edge detection. The original NBA reference model used Polymarket
the other way -- pulling it live, pre-game, to compute edge = model_prob - poly_prob
and generate betting recommendations (its own BettingOpportunityDetector class,
confirmed by reading nba_betting_model_v5.py directly). This project's spread and
moneyline picks already cover that "generate an actionable pick" role; Polymarket
here only needs to answer, after the fact, whether the model's pre-game win
probability was closer to the truth than Polymarket's crowd-sourced one was -- same
shape as the model-vs-market-spread margin-accuracy chart already shipped, just for
win probability instead of point margin, and Polymarket instead of a sportsbook.

Endpoints confirmed empirically (2026-09-02), not from docs -- the public docs are
thin on exact response fields:
- GET /events?tag_slug=cfb&closed=false&end_date_min=...&end_date_max=...&limit=100&offset=N
  Lists CFB game events (one event per game) whose market resolves (endDate) in that
  window. Hard page cap of 100 regardless of a larger `limit`, needs offset pagination.
- Each event embeds a `markets` array. The base moneyline market is the one whose
  `question` exactly equals the event's own `title` ("UMass vs. Rutgers") -- other
  markets on the same event (spread, over/under, 2H moneyline) have different
  question text and would give the wrong number if picked up by mistake.
- That market's `outcomes` (JSON-encoded list of two team names) and `outcomePrices`
  (JSON-encoded list of two probability strings, index-matched to outcomes) are the
  actual win probabilities, e.g. outcomes=["UMass","Rutgers"], outcomePrices=["0.0295","0.9705"].
"""
import json

import requests

BASE_URL = "https://gamma-api.polymarket.com"
PAGE_SIZE = 100
REQUEST_TIMEOUT_S = 10  # same reasoning as src/weather_client.py -- fail fast, not 30s


def fetch_cfb_events(start_iso: str, end_iso: str) -> list[dict]:
    """Returns raw CFB game events whose market resolves (endDate) within
    [start_iso, end_iso). Paginated internally."""
    events = []
    offset = 0
    while True:
        resp = requests.get(
            f"{BASE_URL}/events",
            params={
                "tag_slug": "cfb", "closed": "false", "limit": PAGE_SIZE, "offset": offset,
                "end_date_min": start_iso, "end_date_max": end_iso,
            },
            timeout=REQUEST_TIMEOUT_S,
        )
        resp.raise_for_status()
        page = resp.json()
        events.extend(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return events


def extract_moneyline(event: dict) -> dict | None:
    """Returns {'team_a', 'team_b', 'prob_a', 'prob_b'} for the event's base
    moneyline market, or None if it's missing/malformed. team_a/prob_a and
    team_b/prob_b are index-matched pairs, not home/away -- caller resolves that."""
    for m in event.get("markets", []):
        if m.get("question") != event.get("title"):
            continue
        try:
            outcomes = json.loads(m["outcomes"])
            prices = [float(p) for p in json.loads(m["outcomePrices"])]
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            return None
        if len(outcomes) != 2 or len(prices) != 2:
            return None
        return {"team_a": outcomes[0], "team_b": outcomes[1], "prob_a": prices[0], "prob_b": prices[1]}
    return None
