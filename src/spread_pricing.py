"""Real per-book spread pricing from live_odds, when available -- CFBD's `lines`
table (the source of market_spread used for training and the edge calc) only has
the spread NUMBER, never its price. live_odds does, via scripts/pull_odds.py's
Odds API pull, but per-book price genuinely varies (seen -110/-105/-101/-115/-111
across books on the same side of the same game) -- worth using instead of the
standing -110-both-sides assumption whenever it's actually available.
"""
from collections import defaultdict

from src.team_names import build_team_lookup, normalize


def load_latest_spread_prices(conn) -> dict:
    """Returns {team_id: (median_price, book_count)} using only the most recent
    live_odds pull (scraped_at) so pricing reflects the current market rather than
    blending stale and fresh snapshots across the week. Keyed purely by team_id --
    within one snapshot each team appears in at most one currently-listed game, so
    no need to also key on the opponent."""
    latest = conn.execute("SELECT MAX(scraped_at) FROM live_odds").fetchone()[0]
    if latest is None:
        return {}
    rows = conn.execute(
        """SELECT outcome_name, price FROM live_odds
           WHERE market = 'spreads' AND scraped_at = ? AND price IS NOT NULL""",
        (latest,),
    ).fetchall()

    team_lookup = build_team_lookup(conn)
    by_team = defaultdict(list)
    for outcome_name, price in rows:
        team_id = team_lookup.get(normalize(outcome_name))
        if team_id is not None:
            by_team[team_id].append(price)

    result = {}
    for team_id, prices in by_team.items():
        prices.sort()
        n = len(prices)
        median = prices[n // 2] if n % 2 else (prices[n // 2 - 1] + prices[n // 2]) / 2
        result[team_id] = (round(median), n)
    return result


def get_spread_price(spread_prices: dict, team_id: int | None) -> tuple[int | None, int]:
    """Returns (median_price, book_count) for team_id, or (None, 0) if unavailable --
    caller falls back to the assumed price (ASSUMED_SPREAD_ODDS_AMERICAN)."""
    if team_id is None:
        return None, 0
    return spread_prices.get(team_id, (None, 0))
