"""Bet sizing for the spread pick -- replaced the original sizing on 2026-09-19.

The original cover probability, Phi(edge / regressor RMSE), treated the model's disagreement
with the market as if the market carried no information of its own, so it overstated the
chance of covering: on the 2025 holdout, picks it rated 59% / 71% / 85% covered 51% / 59% / 64%.
The $500 paper bankroll sized off it fell to $6.98 over the first 144 bets of 2026.

cover_probability() is instead a logistic fit of "did the pick cover" on |edge| over the 2025
holdout's 917 decided spread picks (scripts/train_model.py's HOLDOUT_YEAR). The pick is always
the side the edge points to, so only the size of the edge matters. Refit the two constants
whenever the model is retrained -- they describe this model's edges, not edges in general.

kelly_fraction() is 25% fractional Kelly on that probability, capped at MAX_BET_FRACTION, and
zero (no bet) whenever the calibrated probability doesn't clear the price's break-even. Replayed
over 2026's first 145 decided picks, this ended at $318 from $500 (vs $6.98 for the old sizing).

**Blowout no-bet rule, added 2026-09-22**: the margin regressor systematically under-predicts
how much the better team wins by in extreme mismatches (a known shrinkage bias), which makes it
pick the underdog in ~2/3 of games where |market_spread| >= BLOWOUT_SPREAD_THRESHOLD, every
season. That's fine when underdogs in those mismatches tend to cover (2025 weeks 1-3: Vegas
favorites covered only 52.9% of their own big spreads, and our 14+ edge picks -- almost all
underdog, in this bucket -- went 65.6%). It stops being fine when the market regime shifts: 2026
weeks 1-3 same bucket, Vegas favorites are covering 58.7% of their own spreads (a market-wide
shift, not a modeling error -- likely thinner backup depth in the NIL/transfer-portal era means
blowouts don't taper off in garbage time the way they used to), and because our shrinkage bug
means we're concentrated almost entirely on the underdog side of exactly these games, that same
14+ edge bucket flipped to 37.5%. Non-blowout games (|market_spread| < BLOWOUT_SPREAD_THRESHOLD)
barely moved between the two seasons (50.9% -> 53.8% favorite-covers), so this is specific to
the extreme-mismatch bucket, not a general market shift -- and a retrain that folded all 260
completed 2026 games into training didn't change the regressor's shrinkage behavior at all
(same ~4-7 point gap between predicted and market margin on 2025 holdout blowouts, before and
after), so this isn't a staleness problem a retrain fixes either. Until this resolves (more weeks
of data confirming or reversing the 2026 shift, or a structural fix to the regressor itself),
picks in this bucket are forced to "no bet" rather than sized by a confidence signal that's
currently pointing the wrong way for exactly this game type.
"""
import math

CALIBRATION_INTERCEPT = -0.113335
CALIBRATION_SLOPE = 0.042266
KELLY_MULTIPLIER = 0.25
MAX_BET_FRACTION = 0.02
# The paper bankroll restarted at $500 with this sizing: only picks whose game kicks off at or
# after this UTC time count toward it. No trailing "Z" on purpose, so a plain string comparison
# against games.start_date ('2026-09-20T00:00:00.000Z') includes a game starting exactly then.
BANKROLL_RESTART_UTC = "2026-09-20T00:00:00"
# See the blowout no-bet rule above. 21 (not the older, informal "28+" figure from an earlier
# session) is what this session's own weeks 1-3 analysis actually separates cleanly on.
BLOWOUT_SPREAD_THRESHOLD = 21.0
BLOWOUT_NO_BET_REASON = (
    "extreme-mismatch pick (market spread {spread:.1f}): the model's margin regressor "
    "under-predicts blowouts and takes the underdog in most of these, which lost its edge when "
    "2026 favorites started covering big spreads more often than 2025's did (market-wide shift, "
    "still being confirmed) -- staked at zero until this resolves, not because of low confidence"
)


def american_odds_to_net_decimal(odds: int) -> float:
    return 100 / abs(odds) if odds < 0 else odds / 100


def break_even(odds: int) -> float:
    b = american_odds_to_net_decimal(odds)
    return 1 / (1 + b)


def cover_probability(edge: float | None) -> float | None:
    """Calibrated probability that the spread pick covers, from the size of its edge."""
    if edge is None:
        return None
    z = CALIBRATION_INTERCEPT + CALIBRATION_SLOPE * abs(edge)
    return 1 / (1 + math.exp(-z))


def kelly_fraction(p_cover: float | None, odds: int) -> float | None:
    """Fraction of bankroll to stake: 25% Kelly, capped, 0 when p_cover is below break-even."""
    if p_cover is None:
        return None
    b = american_odds_to_net_decimal(odds)
    full = p_cover - (1 - p_cover) / b
    return min(max(0.0, full) * KELLY_MULTIPLIER, MAX_BET_FRACTION)


def is_blowout_underdog_pick(edge: float | None, market_spread: float | None) -> bool:
    """True when the pick is the underdog in an extreme mismatch -- the specific pattern the
    blowout no-bet rule targets (see module docstring). edge > 0 means the model favors the home
    team more than the market does, so it picks home; market_spread < 0 means home is the market
    favorite. The pick is the underdog exactly when those two disagree on which side is which."""
    if edge is None or market_spread is None or abs(market_spread) < BLOWOUT_SPREAD_THRESHOLD:
        return False
    pick_home = edge > 0
    favorite_home = market_spread < 0
    return pick_home != favorite_home


def size_bet(edge: float | None, odds: int, market_spread: float | None = None) -> tuple[float | None, float | None]:
    p = cover_probability(edge)
    if is_blowout_underdog_pick(edge, market_spread):
        return p, 0.0
    return p, kelly_fraction(p, odds)
