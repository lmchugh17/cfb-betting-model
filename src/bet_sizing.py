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


def size_bet(edge: float | None, odds: int) -> tuple[float | None, float | None]:
    p = cover_probability(edge)
    return p, kelly_fraction(p, odds)
