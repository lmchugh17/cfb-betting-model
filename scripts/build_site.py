"""Renders the static dashboard (docs/index.html) from the predictions table +
prediction_results view. Self-contained HTML/CSS, no build step, no external
assets -- deployable straight to GitHub Pages via a plain git push (see the
scheduled Claude Code session, task 11).
"""
import json
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import get_connection, init_db

EASTERN = ZoneInfo("America/New_York")  # handles EDT/EST correctly across the DST transition

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "docs" / "index.html"

STARTING_BANKROLL = 500.0
# Matches scripts/predict_games.py's FULL_SEASON_WINDOW (max of src.live_state's
# ROLLING_WINDOW=4 and ATS_WINDOW=5) -- below this many current-season games played, recent
# form/ATS%/opponent-SRS stats are quietly blending in games from a prior season.
FULL_SEASON_WINDOW = 5
# Fallback only -- used when a pick has no measured spread_price (src.spread_pricing had
# no per-book data for that game). Matches scripts/predict_games.py's own fallback.
ASSUMED_SPREAD_ODDS_AMERICAN = -110
# True ATS breakeven at standard -110 vig (need to win 110/210 = 52.4% of decided bets just
# to break even) -- the meaningful reference line for the ATS chart, not 50%. A spread market
# is deliberately set so both sides are close to a coin flip by design, so 50% ATS is actually
# a losing record once the vig is paid; unlike the moneyline chart's true-50% coin-flip line.
ATS_BREAKEVEN = 110 / 210
# Matches scripts/predict_games.py's LOW_SAMPLE_GAME_THRESHOLD -- see that file's comment for
# how this was measured against the real distribution of tracked games per team.
LOW_SAMPLE_GAME_THRESHOLD = 20

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "extra_trees": "Extra Trees",
}


def _net_decimal_odds(odds: int) -> float:
    return 100 / abs(odds) if odds < 0 else odds / 100


def fetch_upcoming(conn) -> list[dict]:
    rows = conn.execute("""
        SELECT p.game_id, p.year, p.week, p.start_date, p.home_team, p.away_team,
               p.predicted_margin, p.win_prob_home, p.market_spread, p.pick_team,
               p.edge, p.confidence_tier, p.highlights_json, p.tldr, p.bullets_json,
               p.model_breakdown_json, p.cover_probability, p.kelly_fraction,
               p.moneyline_pick, p.moneyline_win_prob, p.moneyline_confidence_tier,
               p.spread_price, p.spread_price_source, p.spread_price_book_count,
               p.min_current_season_games, p.low_sample_team, p.low_sample_team_games
        FROM predictions p
        JOIN games g ON p.game_id = g.id
        WHERE g.home_points IS NULL
        ORDER BY p.start_date
    """).fetchall()
    cols = ["game_id", "year", "week", "start_date", "home_team", "away_team",
            "predicted_margin", "win_prob_home", "market_spread", "pick_team",
            "edge", "confidence_tier", "highlights_json", "tldr", "bullets_json",
            "model_breakdown_json", "cover_probability", "kelly_fraction",
            "moneyline_pick", "moneyline_win_prob", "moneyline_confidence_tier",
            "spread_price", "spread_price_source", "spread_price_book_count",
            "min_current_season_games", "low_sample_team", "low_sample_team_games"]
    return [dict(zip(cols, r)) for r in rows]


def fetch_results(conn) -> list[dict]:
    rows = conn.execute("""
        SELECT game_id, year, week, start_date, home_team, away_team, predicted_margin,
               win_prob_home, market_spread, actual_margin, home_points, away_points, pick_team,
               edge, confidence_tier, highlights_json, tldr, bullets_json,
               ats_pick_won_straight_up, pick_covered, model_breakdown_json, cover_probability, kelly_fraction,
               moneyline_pick, moneyline_win_prob, moneyline_confidence_tier, moneyline_pick_won,
               spread_price, spread_price_source, spread_price_book_count, min_current_season_games,
               low_sample_team, low_sample_team_games
        FROM prediction_results ORDER BY start_date DESC
    """).fetchall()
    cols = ["game_id", "year", "week", "start_date", "home_team", "away_team", "predicted_margin",
            "win_prob_home", "market_spread", "actual_margin", "home_points", "away_points", "pick_team",
            "edge", "confidence_tier", "highlights_json", "tldr", "bullets_json",
            "ats_pick_won_straight_up", "pick_covered", "model_breakdown_json", "cover_probability", "kelly_fraction",
            "moneyline_pick", "moneyline_win_prob", "moneyline_confidence_tier", "moneyline_pick_won",
            "spread_price", "spread_price_source", "spread_price_book_count", "min_current_season_games",
            "low_sample_team", "low_sample_team_games"]
    return [dict(zip(cols, r)) for r in rows]


def _week_zero_cutoff(dates: list[str]) -> str | None:
    """CFBD bundles a small slate of early season-opener games (what broadcasters/fans
    already call 'Week 0') into the same week=1 as the following week's main slate --
    confirmed 2026-09-02: week 1 spans Aug 29 - Sep 7, but games only actually happen on
    Aug 29-30 and Sep 3-7, a clean 3-day gap in between. Finds that gap generically (the
    largest gap between distinct game dates) rather than hardcoding a calendar date, so it
    still works correctly in a season where the split falls on a different date. Returns
    the first date that should stay in week 1, or None if week 1 has no real gap to split."""
    if len(dates) < 2:
        return None
    parsed = sorted(date.fromisoformat(d) for d in set(dates))
    gaps = [(parsed[i + 1] - parsed[i]).days for i in range(len(parsed) - 1)]
    max_gap = max(gaps)
    if max_gap < 2:
        return None
    return parsed[gaps.index(max_gap) + 1].isoformat()


def fetch_weekly_performance(conn) -> list[dict]:
    """One row per (year, display_week) with that week's moneyline record, ATS record,
    and a same-week 'always take the market favorite' straight-up baseline -- lets the
    season-long summary tiles be checked against week-by-week form, and the model's own
    moneyline record against a trivial baseline that needs no model at all, instead of
    only a single cumulative number. display_week splits CFBD's week=1 into a true week 0
    (see _week_zero_cutoff) when that week actually bundles two slates."""
    rows = conn.execute("""
        SELECT year, week, DATE(start_date), home_team, away_team, market_spread,
               actual_winner, moneyline_pick_won, pick_covered, pick_team, actual_margin,
               ABS(predicted_margin - actual_margin) AS margin_error
        FROM prediction_results
    """).fetchall()

    # The week-0/week-1 gap only shows up once you look at the WHOLE week-1 span, not just
    # completed games -- early in the season prediction_results (completed only) might only
    # contain the small week-0 slate so far, with no visible gap yet. Pull all week=1 dates
    # from `predictions` (upcoming + completed) instead, purely to find the split point.
    week1_dates_by_year = defaultdict(list)
    for year, week, d in conn.execute("SELECT year, week, DATE(start_date) FROM predictions WHERE week = 1"):
        week1_dates_by_year[year].append(d)
    cutoffs = {year: _week_zero_cutoff(dates) for year, dates in week1_dates_by_year.items()}

    buckets = defaultdict(lambda: {"n": 0, "ml_wins": 0, "ml_decided": 0, "ats_wins": 0,
                                    "ats_losses": 0, "ats_pushes": 0, "fav_wins": 0, "fav_decided": 0,
                                    "margin_errors": [], "market_margin_errors": []})
    for year, week, d, home, away, spread, winner, ml_won, covered, pick_team, actual_margin, margin_err in rows:
        display_week = week
        if week == 1 and cutoffs.get(year) and d < cutoffs[year]:
            display_week = 0
        b = buckets[(year, display_week)]
        b["n"] += 1
        if ml_won is not None:
            b["ml_decided"] += 1
            b["ml_wins"] += ml_won
        if covered == 1:
            b["ats_wins"] += 1
        elif covered == 0:
            b["ats_losses"] += 1
        elif covered is None and pick_team is not None:
            b["ats_pushes"] += 1
        if spread is not None and spread != 0:
            favorite = home if spread < 0 else away
            b["fav_decided"] += 1
            b["fav_wins"] += int(winner == favorite)
        if margin_err is not None:
            b["margin_errors"].append(margin_err)
        # Market's own implied margin is -market_spread (home-signed, same convention as
        # predicted_margin) -- its error against the actual result is the same "how far off
        # was this number" measure as the model's own margin_error, just for Vegas's number
        # instead of ours. Lets the site show whether the model is closing the gap on the
        # market's own accuracy over time, week by week (see render_margin_accuracy_chart).
        if spread is not None and actual_margin is not None:
            b["market_margin_errors"].append(abs(-spread - actual_margin))

    result = []
    for (year, week), b in buckets.items():
        avg_err = sum(b["margin_errors"]) / len(b["margin_errors"]) if b["margin_errors"] else None
        market_avg_err = (sum(b["market_margin_errors"]) / len(b["market_margin_errors"])
                           if b["market_margin_errors"] else None)
        result.append({"year": year, "week": week, "n": b["n"], "ml_wins": b["ml_wins"],
                        "ml_decided": b["ml_decided"], "ats_wins": b["ats_wins"], "ats_losses": b["ats_losses"],
                        "ats_pushes": b["ats_pushes"], "fav_wins": b["fav_wins"], "fav_decided": b["fav_decided"],
                        "avg_err": avg_err, "market_avg_err": market_avg_err})
    result.sort(key=lambda r: (r["year"], r["week"]), reverse=True)
    return result


def compute_current_bankroll(conn) -> float:
    """Chronological paper-bankroll replay: starts at STARTING_BANKROLL and compounds
    through every settled (non-push) pick in date order using its own kelly_fraction and
    its own spread_price (real per-book price when that pick had one, else the standing
    -110 assumption) -- each pick's payout uses the SAME price its Kelly fraction was
    actually sized against, not a blanket -110 for every historical bet regardless of what
    was really available at the time. Upcoming picks size their recommended wager off this
    running total rather than the fixed starting amount, same as the NBA reference site."""
    rows = conn.execute("""
        SELECT kelly_fraction, pick_covered, spread_price FROM prediction_results
        WHERE kelly_fraction IS NOT NULL AND pick_covered IS NOT NULL
        ORDER BY start_date ASC
    """).fetchall()
    bankroll = STARTING_BANKROLL
    for kelly_fraction, covered, spread_price in rows:
        b = _net_decimal_odds(spread_price if spread_price is not None else ASSUMED_SPREAD_ODDS_AMERICAN)
        wager = bankroll * kelly_fraction
        bankroll += wager * b if covered else -wager
    return bankroll


def fetch_summary(conn) -> dict:
    row = conn.execute("""
        SELECT COUNT(*),
               SUM(CASE WHEN moneyline_pick_won = 1 THEN 1 ELSE 0 END),
               SUM(CASE WHEN moneyline_pick IS NOT NULL THEN 1 ELSE 0 END),
               SUM(CASE WHEN pick_covered = 1 THEN 1 ELSE 0 END),
               SUM(CASE WHEN pick_covered = 0 THEN 1 ELSE 0 END),
               SUM(CASE WHEN pick_covered IS NULL AND pick_team IS NOT NULL THEN 1 ELSE 0 END),
               AVG(ABS(predicted_margin - actual_margin))
        FROM prediction_results
    """).fetchone()
    n, ml_wins, ml_decided, ats_wins, ats_losses, ats_pushes, avg_err = row
    ml_decided = ml_decided or 0
    return {
        "n": n or 0, "ml_wins": ml_wins or 0, "ml_losses": ml_decided - (ml_wins or 0), "ml_decided": ml_decided,
        "ats_wins": ats_wins or 0, "ats_losses": ats_losses or 0, "ats_pushes": ats_pushes or 0,
        "avg_err": avg_err,
    }


def fmt_kickoff(iso_str: str) -> str:
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00")).astimezone(EASTERN)
    return dt.strftime(f"%a %b %-d, %-I:%M %p {dt.tzname()}")


def tier_badge(tier: str | None, low_data: bool = False) -> str:
    if not tier:
        return ""
    # A dagger, not an asterisk -- the price line already uses * for a different footnote
    # (assumed vs. measured spread price); reusing the same mark for two different caveats
    # on one card would make it ambiguous which one applies.
    mark = "&dagger;" if low_data else ""
    return f'<span class="tier tier-{tier}">{tier.upper()} Confidence{mark}</span>'


def render_model_breakdown(breakdown_json: str | None) -> str:
    if not breakdown_json:
        return ""
    breakdown = json.loads(breakdown_json)
    rows_html = "".join(
        f'<div class="model-row"><span class="model-name">{MODEL_LABELS.get(name, name)}</span>'
        f'<span class="model-prob">{prob:.0%}</span></div>'
        for name, prob in breakdown.items()
    )
    return (
        '<details class="model-breakdown"><summary>Per-model breakdown (home win probability)</summary>'
        f'{rows_html}</details>'
    )


def render_wager_line(p: dict, bankroll: float | None) -> str:
    if bankroll is None or not p.get("kelly_fraction") or p.get("cover_probability") is None:
        return ""
    wager = bankroll * p["kelly_fraction"]
    return (
        f'<div class="wager-line">Recommended wager: <strong>${wager:.2f}</strong> '
        f'({p["kelly_fraction"]:.1%} of ${bankroll:.2f} paper bankroll, 25% Kelly) '
        f'&middot; cover probability {p["cover_probability"]:.0%}</div>'
    )


def render_pick_card(p: dict, result: dict | None = None, bankroll: float | None = None) -> str:
    highlights = json.loads(p["highlights_json"]) if p.get("highlights_json") else []
    bullets = json.loads(p["bullets_json"]) if p.get("bullets_json") else []
    tldr = p.get("tldr")

    market_line = "no line"
    if p["market_spread"] is not None:
        fav = p["home_team"] if p["market_spread"] < 0 else p["away_team"]
        market_line = f"{fav} by {abs(p['market_spread']):.1f}"

    low_data = (p.get("min_current_season_games") is not None
                and p["min_current_season_games"] < FULL_SEASON_WINDOW)

    ml_html = ""
    if p.get("moneyline_pick"):
        ml_html = (
            f'<div class="pick-line">Moneyline pick: <strong>{p["moneyline_pick"]}</strong> '
            f'({p["moneyline_win_prob"]:.0%} win prob) '
            f'{tier_badge(p["moneyline_confidence_tier"], low_data)}</div>'
        )
    pick_html = ""
    if p.get("pick_team"):
        # The pick's own spread, signed from ITS perspective (not always the home-signed
        # market_spread) -- e.g. the underdog pick shows a positive number (points received),
        # the favorite pick shows negative (points given). Odds shown are the real per-book
        # median price when a recent live_odds pull has one; otherwise the standing -110
        # assumption, marked with a footnote asterisk so it's clear which one it is.
        pick_spread_html = ""
        if p["market_spread"] is not None:
            pick_spread = p["market_spread"] if p["pick_team"] == p["home_team"] else -p["market_spread"]
            price = p.get("spread_price") if p.get("spread_price") is not None else ASSUMED_SPREAD_ODDS_AMERICAN
            is_measured = p.get("spread_price_source") == "measured"
            price_label = f"{price}" + ("" if is_measured else "*")
            pick_spread_html = f' {pick_spread:+.1f} <span class="odds">({price_label})</span>'
        pick_html = (
            f'<div class="pick-line">Spread pick: <strong>{p["pick_team"]}{pick_spread_html}</strong> '
            f'{tier_badge(p["confidence_tier"], low_data)}</div>'
        )
    wager_html = render_wager_line(p, bankroll)
    model_breakdown_html = render_model_breakdown(p.get("model_breakdown_json"))

    # Dedicated, always-current callout -- deliberately NOT folded into the bullets/tldr text,
    # since those are only regenerated when a game is re-predicted and can go stale (bullets_json
    # takes precedence over highlights_json below once written). This ties directly to the edge,
    # same reasoning as scripts/predict_games.py's version of this sentence.
    low_sample_html = ""
    if (p.get("low_sample_team") and p.get("low_sample_team_games") is not None
            and p["low_sample_team_games"] < LOW_SAMPLE_GAME_THRESHOLD and p.get("edge") is not None):
        low_sample_html = (
            f'<div class="callout">{p["low_sample_team"]} has only {p["low_sample_team_games"]:.0f} '
            f'tracked games in our database (a recent FBS entrant, or a rarely-tracked crossover '
            f'opponent -- our data only covers games involving an FBS team) -- its recent-form stats '
            f'are a small, possibly unrepresentative sample, likely contributing to the '
            f'{abs(p["edge"]):.1f}-point gap between the model\'s margin and the market\'s line.</div>'
        )

    body_bullets = bullets or highlights
    bullets_html = "".join(f"<li>{b}</li>" for b in body_bullets)

    result_html = ""
    if result:
        ml_result = {1: "correct", 0: "incorrect"}.get(result["moneyline_pick_won"], "n/a")
        cover = {1: "covered", 0: "did not cover", None: "push"}[result["pick_covered"]]
        result_html = (
            f'<div class="result-line">Final: {result["home_team"]} {result["home_points"]:.0f} - '
            f'{result["away_points"]:.0f} {result["away_team"]} &mdash; moneyline pick was {ml_result}, '
            f'spread pick {cover} the spread</div>'
        )

    return f"""
    <div class="card">
      <div class="matchup">{p["away_team"]} @ {p["home_team"]}</div>
      <div class="kickoff">{fmt_kickoff(p["start_date"])}</div>
      <div class="prediction-line">Model: {p["home_team"]} by {p["predicted_margin"]:+.1f} ({p["win_prob_home"]:.0%} win prob) &middot; Market: {market_line}</div>
      {ml_html}
      {pick_html}
      {low_sample_html}
      {wager_html}
      {f'<div class="tldr">{tldr}</div>' if tldr else ""}
      <ul class="bullets">{bullets_html}</ul>
      {model_breakdown_html}
      {result_html}
    </div>"""


def render_stat_tile(label: str, value: str) -> str:
    return f'<div class="stat-tile"><div class="stat-value">{value}</div><div class="stat-label">{label}</div></div>'


def _week_label(w: dict) -> str:
    return "Postseason" if w.get("week") is None else f"Week {w['week']}"


def render_weekly_table(weekly: list[dict]) -> str:
    if not weekly:
        return '<p class="empty">No completed weeks tracked yet.</p>'
    rows_html = ""
    for w in weekly:
        ml_pct = f"{w['ml_wins']}-{w['ml_decided'] - w['ml_wins']} ({w['ml_wins']/w['ml_decided']:.0%})" if w["ml_decided"] else "n/a"
        ats_pct = f"{w['ats_wins']}-{w['ats_losses']}-{w['ats_pushes']}"
        fav_pct = (f"{w['fav_wins']}-{w['fav_decided'] - w['fav_wins']} ({w['fav_wins']/w['fav_decided']:.0%})"
                   if w["fav_decided"] else "n/a")
        avg_err = f"{w['avg_err']:.1f}" if w["avg_err"] is not None else "n/a"
        rows_html += (
            f"<tr><td>{w['year']} {_week_label(w)}</td><td>{w['n']}</td>"
            f"<td>{ml_pct}</td><td>{fav_pct}</td><td>{ats_pct}</td><td>{avg_err} pts</td></tr>"
        )
    return f"""<div class="table-wrap"><table class="weekly-table">
      <thead><tr><th>Week</th><th>Games</th><th>Moneyline</th><th>Favorite Baseline</th><th>ATS</th><th>Avg. Error</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table></div>"""


def render_bar_value(pct: float, min_pct_for_inside: float = 0.15) -> str:
    """The bar's own %-value label, shared by every weekly bar chart. Sits just inside the
    top of the bar (dark text directly on the bar's own green/red fill) when there's enough
    height for it to read cleanly -- floats just above a very short bar instead, where
    "inside" would look cramped. Anchored the same way as any other track-relative marker
    (bottom:{pct}%), never a fixed spot at the top of the whole column regardless of height.
    Placing it inside the bar (rather than floating above it) is what actually fixes bar
    values colliding with a reference/baseline line drawn near the same height -- floating
    labels compete for the same space above the bar; a label inside the bar's own colored
    area never can, regardless of where any line on the chart happens to sit."""
    variant = "bar-value-inside" if pct >= min_pct_for_inside else "bar-value-above"
    return f'<div class="bar-value {variant}" style="bottom: {pct * 100:.1f}%">{pct:.0%}</div>'


def render_weekly_win_pct_chart(weekly: list[dict]) -> str:
    """Moneyline win% per week as a simple CSS bar chart -- oldest week first (left to
    right) so it reads as a trend, opposite of the table's newest-first order. Bar height
    is win% of vertical space. No plain 50% coin-flip line here on purpose -- it's not the
    meaningful threshold for straight-up picks (even blindly picking the favorite clears
    50% easily, our own favorite-baseline data shows 75%), so a bare 50% line next to the
    baseline marker was redundant and confusing. The favorite-baseline marker on each bar
    is the real comparison: a bar that doesn't clear its OWN week's baseline is a genuine
    red flag, not just a bad week."""
    decided = [w for w in weekly if w["ml_decided"]]
    if not decided:
        return '<p class="empty">No completed weeks tracked yet.</p>'
    bars_html = ""
    for w in reversed(decided):  # weekly is newest-first; chart wants oldest-first
        pct = w["ml_wins"] / w["ml_decided"]
        css_class = "above" if pct >= 0.5 else "below"
        baseline_html = ""
        if w["fav_decided"]:
            fav_pct = w["fav_wins"] / w["fav_decided"]
            baseline_html = (
                f'<div class="baseline-marker" style="bottom: {fav_pct * 100:.1f}%" '
                f'title="Favorite baseline: {fav_pct:.0%}">'
                f'<span class="baseline-label">{fav_pct:.0%}</span></div>'
            )
        bars_html += (
            '<div class="bar-col">'
            f'<div class="bar-track">{render_bar_value(pct)}<div class="bar-fill {css_class}" style="height: {pct * 100:.1f}%"></div>'
            f'{baseline_html}</div>'
            f'<div class="bar-label">{w["year"]} {_week_label(w)}</div>'
            "</div>"
        )
    return (f'<div class="bar-chart">{bars_html}</div>'
            '<div class="bar-legend"><span class="legend-swatch legend-model"></span>Model moneyline win%'
            '<span class="legend-swatch legend-baseline"></span>Favorite baseline (same week)</div>')


def render_ats_win_pct_chart(weekly: list[dict]) -> str:
    """Against-the-spread win% per week -- same bar-chart shape as the moneyline chart,
    but keeps a reference line at ATS_BREAKEVEN (52.4%), unlike the moneyline chart's
    dropped 50% line: this one is a real wagering threshold (the actual break-even rate
    against standard -110 vig), not a redundant coin-flip marker, so it earns its place.
    Styled as a dashed neutral line (not the solid amber "baseline strategy" color) to keep
    the visual language consistent: dashed = a threshold to clear, solid amber = a real
    competing strategy's own results (used on the moneyline chart)."""
    decided = [w for w in weekly if (w["ats_wins"] + w["ats_losses"])]
    if not decided:
        return '<p class="empty">No completed weeks tracked yet.</p>'
    bars_html = ""
    for w in reversed(decided):
        n = w["ats_wins"] + w["ats_losses"]
        pct = w["ats_wins"] / n
        css_class = "above" if pct >= ATS_BREAKEVEN else "below"
        bars_html += (
            '<div class="bar-col">'
            f'<div class="bar-track">{render_bar_value(pct)}<div class="bar-fill {css_class}" style="height: {pct * 100:.1f}%"></div>'
            f'<div class="ref-line-marker" style="bottom: {ATS_BREAKEVEN * 100:.1f}%"></div></div>'
            f'<div class="bar-label">{w["year"]} {_week_label(w)}</div>'
            "</div>"
        )
    # The threshold's own text label is a fixed corner annotation, deliberately NOT tied to
    # any bar's height -- a per-bar label at the line's exact position collided with that
    # bar's own value label whenever a week's result happened to land close to the
    # threshold (a common, not edge-case, scenario -- results cluster near breakeven).
    return (f'<div class="bar-chart"><div class="ref-line-corner-label">Breakeven {ATS_BREAKEVEN:.1%}</div>{bars_html}</div>'
            f'<div class="bar-legend"><span class="legend-swatch legend-model"></span>ATS win% '
            f'<span class="legend-swatch legend-ref"></span>Breakeven at -110 ({ATS_BREAKEVEN:.1%})</div>')


def render_margin_accuracy_chart(weekly: list[dict]) -> str:
    """Model's avg margin error vs. the market's own avg margin error, per week -- paired
    bars on a shared points scale (lower is better for both). Unlike the win% charts there's
    no natural 0-100% cap, so each bar's height is relative to the largest error seen across
    all weeks shown, not a fixed scale. Same comparison as the 2025 holdout eval (regressor
    MAE 12.87 vs. the market's own 11.90, see scripts/train_model.py) -- not shown elsewhere
    on the live site itself, this is the first place that comparison is tracked in-season."""
    decided = [w for w in weekly if w["avg_err"] is not None or w["market_avg_err"] is not None]
    if not decided:
        return '<p class="empty">No completed weeks tracked yet.</p>'
    all_errs = [w["avg_err"] for w in decided if w["avg_err"] is not None]
    all_errs += [w["market_avg_err"] for w in decided if w["market_avg_err"] is not None]
    max_err = max(all_errs) if all_errs else 1
    bars_html = ""
    for w in reversed(decided):
        model_err, market_err = w["avg_err"], w["market_avg_err"]
        model_pct = (model_err / max_err * 100) if model_err is not None else 0
        market_pct = (market_err / max_err * 100) if market_err is not None else 0
        model_label = f"{model_err:.1f}" if model_err is not None else "n/a"
        market_label = f"{market_err:.1f}" if market_err is not None else "n/a"
        bars_html += (
            '<div class="pair-col">'
            f'<div class="pair-values"><span class="pair-model-text">{model_label}</span> vs '
            f'<span class="pair-market-text">{market_label}</span></div>'
            '<div class="pair-tracks">'
            f'<div class="bar-track pair-track"><div class="bar-fill pair-model-fill" style="height: {model_pct:.1f}%"></div></div>'
            f'<div class="bar-track pair-track"><div class="bar-fill pair-market-fill" style="height: {market_pct:.1f}%"></div></div>'
            "</div>"
            f'<div class="bar-label">{w["year"]} {_week_label(w)}</div>'
            "</div>"
        )
    return (f'<div class="bar-chart">{bars_html}</div>'
            '<div class="bar-legend"><span class="legend-swatch legend-model"></span>Model avg error (pts)'
            '<span class="legend-swatch legend-market-sw"></span>Market avg error (pts) &middot; lower is better</div>')


def build_html(upcoming: list[dict], results: list[dict], summary: dict, bankroll: float,
                weekly: list[dict]) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    ml_pct = f"{summary['ml_wins']}/{summary['ml_decided']}" if summary["ml_decided"] else "0/0"
    ats_pct = (f"{summary['ats_wins']}-{summary['ats_losses']}-{summary['ats_pushes']}"
               if summary["n"] else "0-0-0")
    avg_err = f"{summary['avg_err']:.1f} pts" if summary["avg_err"] is not None else "n/a"

    stat_tiles = "".join([
        render_stat_tile("Picks Tracked", str(summary["n"])),
        render_stat_tile("Moneyline (Straight-Up)", ml_pct),
        render_stat_tile("Against the Spread", ats_pct),
        render_stat_tile("Avg. Margin Error", avg_err),
        render_stat_tile("Paper Bankroll", f"${bankroll:.2f}"),
    ])

    upcoming_html = "".join(render_pick_card(p, bankroll=bankroll) for p in upcoming) or '<p class="empty">No upcoming games with picks right now.</p>'
    results_html = "".join(render_pick_card(r, result=r) for r in results) or '<p class="empty">No completed games yet.</p>'
    weekly_html = render_weekly_table(weekly)
    ml_chart_html = render_weekly_win_pct_chart(weekly)
    ats_chart_html = render_ats_win_pct_chart(weekly)
    margin_chart_html = render_margin_accuracy_chart(weekly)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CFB Betting Model</title>
<style>
  :root {{
    --bg: #0f1115; --card: #171a21; --border: #262b36; --text: #e8eaed;
    --text-dim: #9aa1ac; --accent: #4f8cff; --green: #3ddc84; --red: #ff6161; --amber: #ffb84f;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 0 1rem 4rem; background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  .wrap {{ max-width: 900px; margin: 0 auto; }}
  header {{ padding: 2.5rem 0 1rem; }}
  h1 {{ font-size: 1.6rem; margin: 0 0 0.25rem; }}
  .tagline {{ color: var(--text-dim); font-size: 0.9rem; margin: 0; }}
  .stats-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0.75rem; margin: 1.5rem 0 2.5rem; }}
  .stat-tile {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 1rem; text-align: center; }}
  .stat-value {{ font-size: 1.4rem; font-weight: 600; }}
  .stat-label {{ font-size: 0.75rem; color: var(--text-dim); margin-top: 0.25rem; text-transform: uppercase; letter-spacing: 0.03em; }}
  h2 {{ font-size: 1.1rem; border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; margin: 2.5rem 0 1rem; }}
  h3 {{ font-size: 0.85rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.03em; margin: 1.5rem 0 0.6rem; }}
  h3:first-of-type {{ margin-top: 0.5rem; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 1.25rem; margin-bottom: 1rem; }}
  .matchup {{ font-size: 1.05rem; font-weight: 600; }}
  .kickoff {{ color: var(--text-dim); font-size: 0.8rem; margin-bottom: 0.6rem; }}
  .prediction-line {{ font-size: 0.9rem; margin-bottom: 0.4rem; }}
  .pick-line {{ font-size: 0.9rem; margin-bottom: 0.6rem; }}
  .odds {{ font-weight: 400; color: var(--text-dim); }}
  .tier {{ font-size: 0.7rem; padding: 0.1rem 0.5rem; border-radius: 999px; margin-left: 0.4rem; white-space: nowrap; display: inline-block; }}
  .tier-high {{ background: rgba(61,220,132,0.15); color: var(--green); }}
  .tier-medium {{ background: rgba(255,184,79,0.15); color: var(--amber); }}
  .tier-low {{ background: rgba(154,161,172,0.15); color: var(--text-dim); }}
  .wager-line {{ font-size: 0.85rem; color: var(--text-dim); margin-bottom: 0.6rem; }}
  .callout {{ font-size: 0.85rem; color: var(--amber); background: rgba(255,184,79,0.1); border: 1px solid rgba(255,184,79,0.25); border-radius: 8px; padding: 0.6rem 0.8rem; margin-bottom: 0.6rem; }}
  .tldr {{ font-style: italic; color: var(--text-dim); font-size: 0.88rem; margin-bottom: 0.6rem; }}
  .bullets {{ margin: 0; padding-left: 1.1rem; font-size: 0.85rem; color: var(--text-dim); }}
  .bullets li {{ margin-bottom: 0.25rem; }}
  .model-breakdown {{ margin-top: 0.7rem; font-size: 0.8rem; color: var(--text-dim); }}
  .model-breakdown summary {{ cursor: pointer; }}
  .model-row {{ display: flex; justify-content: space-between; padding: 0.2rem 0 0.2rem 1rem; }}
  .result-line {{ margin-top: 0.7rem; padding-top: 0.6rem; border-top: 1px solid var(--border); font-size: 0.85rem; }}
  .empty {{ color: var(--text-dim); font-size: 0.9rem; }}
  .bar-chart {{ position: relative; display: flex; align-items: flex-end; gap: 0.75rem; background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 1.25rem 1rem 1rem; margin-bottom: 1rem; overflow-x: auto; }}
  .ref-line-marker {{ position: absolute; left: -3px; right: -3px; height: 0; border-top: 1px dashed var(--text-dim); }}
  .ref-line-corner-label {{ position: absolute; top: 0.5rem; left: 1rem; font-size: 0.65rem; color: var(--text-dim); }}
  .bar-col {{ display: flex; flex-direction: column; align-items: center; flex: 0 0 auto; width: 52px; }}
  .bar-track {{ position: relative; width: 32px; height: 120px; background: rgba(255,255,255,0.05); border-radius: 4px; display: flex; align-items: flex-end; margin-top: 1.4rem; }}
  .bar-fill {{ position: relative; width: 100%; border-radius: 3px 3px 0 0; }}
  .bar-fill.above {{ background: var(--green); }}
  .bar-fill.below {{ background: var(--red); }}
  .bar-value {{ position: absolute; left: 50%; font-size: 0.72rem; white-space: nowrap; z-index: 1; }}
  .bar-value-inside {{ transform: translate(-50%, 0.35rem); color: #fff; font-weight: 700; text-shadow: 0 1px 2px rgba(0,0,0,0.5); }}
  .bar-value-above {{ transform: translate(-50%, -100%); margin-bottom: 0.25rem; color: var(--text); background: var(--card); padding: 0 3px; border-radius: 3px; }}
  .baseline-marker {{ position: absolute; left: -3px; right: -3px; height: 2px; background: var(--amber); }}
  .baseline-label {{ position: absolute; right: 2px; top: -0.75em; font-size: 0.6rem; color: var(--amber); white-space: nowrap; background: var(--card); padding: 0 2px; border-radius: 2px; }}
  .bar-label {{ font-size: 0.68rem; color: var(--text-dim); margin-top: 0.4rem; text-align: center; white-space: nowrap; }}
  .bar-legend {{ display: flex; align-items: center; gap: 0.4rem; font-size: 0.75rem; color: var(--text-dim); margin: 0.6rem 0 1rem; flex-wrap: wrap; }}
  .legend-swatch {{ display: inline-block; width: 12px; height: 12px; border-radius: 2px; margin-left: 0.6rem; }}
  .legend-swatch:first-child {{ margin-left: 0; }}
  .legend-model {{ background: var(--green); }}
  .legend-baseline {{ background: var(--amber); height: 2px; width: 12px; border-radius: 0; align-self: center; }}
  .legend-ref {{ background: none; border-top: 1px dashed var(--text-dim); height: 0; width: 12px; border-radius: 0; align-self: center; }}
  .legend-market-sw {{ background: var(--accent); }}
  .pair-col {{ display: flex; flex-direction: column; align-items: center; flex: 0 0 auto; width: 76px; }}
  .pair-values {{ font-size: 0.68rem; color: var(--text-dim); margin-bottom: 0.3rem; white-space: nowrap; }}
  .pair-model-text {{ color: var(--green); }}
  .pair-market-text {{ color: var(--accent); }}
  .pair-tracks {{ display: flex; gap: 4px; }}
  .pair-track {{ width: 22px; }}
  .pair-model-fill {{ background: var(--green); }}
  .pair-market-fill {{ background: var(--accent); }}
  .table-wrap {{ overflow-x: auto; margin-bottom: 1rem; }}
  .weekly-table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; background: var(--card); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }}
  .weekly-table th, .weekly-table td {{ padding: 0.6rem 0.9rem; text-align: left; white-space: nowrap; }}
  .weekly-table th {{ color: var(--text-dim); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.03em; border-bottom: 1px solid var(--border); }}
  .weekly-table tbody tr:not(:last-child) td {{ border-bottom: 1px solid var(--border); }}
  footer {{ margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid var(--border); color: var(--text-dim); font-size: 0.78rem; line-height: 1.5; }}
  footer .footnote {{ font-size: 0.72rem; opacity: 0.8; }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>CFB Betting Model</h1>
    <p class="tagline">Opponent-adjusted college football predictions vs. the market. Personal research project, not financial advice.</p>
  </header>

  <div class="stats-row">{stat_tiles}</div>

  <h2>Weekly Trends</h2>
  <h3>Moneyline Win %</h3>
  {ml_chart_html}
  <h3>Against the Spread Win %</h3>
  {ats_chart_html}
  <h3>Margin Accuracy: Model vs. Market</h3>
  {margin_chart_html}

  <h2>This Week's Picks</h2>
  {upcoming_html}

  <h2>Recent Results</h2>
  {results_html}

  <h2>Weekly Performance</h2>
  {weekly_html}

  <footer>
    <p><strong>Methodology:</strong> 5-model stacked ensemble (logistic regression, random forest,
    XGBoost, LightGBM, extra trees) for win probability, plus a separate XGBoost regressor for
    predicted margin. Features: ELO ratings, opponent-adjusted SRS ratings (iterative strength-of-schedule
    solve), rolling 4-game team form, rolling ATS record, rest/bye-week, head-to-head history.
    Trained on 2021-2024 FBS seasons, held out all of 2025 for evaluation (75.5% straight-up accuracy,
    appropriately trailing the market's own 76.8% -- landing behind the market, not matching or beating it,
    is the healthy sign of a real model rather than a leakage bug).</p>
    <p><strong>ELO</strong>, referenced throughout the picks below, is a rating system (originally from chess)
    where every team starts at 1500 and gains or loses points after each game based on the result and how
    surprising it was -- beating a stronger team gains more than beating a weaker one, and the size of the
    swing scales with a K-factor (40 here, larger than a typical NBA ELO setup since a ~13-game CFB season
    gives each result more information to react to). A team's ELO rating converts directly into a win
    probability against any opponent -- that conversion, not the raw rating number, is what actually drives
    the model. Home teams get a flat +65 rating bonus before that calculation, an unvalidated placeholder
    (see <code>src/elo.py</code>) rather than a measured home-field value, same caveat as the confidence
    tiers above.</p>
    <p>Two separate picks are shown per game: a <strong>moneyline pick</strong> (whichever team the model
    gives &gt;50% win probability -- who wins outright, no spread) and a <strong>spread pick</strong> (which
    side has value against the market line). These are often different teams, deliberately -- a big
    underdog can be the correct spread pick (expected to lose, just by less than the market thinks) while
    still being the wrong moneyline pick. Market spread is deliberately excluded from model training and
    used only to compute the spread pick's edge. Both confidence tiers are first-pass, not statistically
    calibrated: spread tiers are based on |edge| in points, moneyline tiers on the picked side's win
    probability (&ge;75% high, &ge;60% medium, below that low).</p>
    <p>Model and market numbers can differ by a lot, especially on lopsided games -- that's expected, not
    a sign something's wrong. The model never sees the market line during training, so its number is an
    independent estimate from team-strength/form ratings alone; the market line reflects betting activity
    and each sportsbook's own risk-balancing, which isn't the same goal as pinning down the single most
    likely margin. Checked against the 2025 holdout: the model's spread picks covered at about the same
    rate on games with 28+ point market spreads (53.3%) as on more typical ones (53.2%) -- a wide gap on
    a lopsided game doesn't by itself mean the model is less reliable there.</p>
    <p>The Weekly Trends charts track three things separately: moneyline win% against that same week's
    "always take the market favorite" baseline (a plain 50% coin-flip line isn't the meaningful reference
    for straight-up picks -- even blindly picking favorites clears 50% easily), ATS win% against the real
    breakeven at standard -110 vig (52.4%, not 50% -- a spread market is deliberately set so both sides are
    close to a coin flip, so 50% ATS is actually a losing record once the vig is paid), and <strong>margin
    error</strong> -- the model's average margin error against the market's own average margin error on the
    same games. Margin error for one game is simply |predicted margin &minus; actual final margin|, in points
    (e.g. picking a team to win by 10 when they win by 14 is a 4-point error); averaging that across every
    graded game gives the "Avg. Margin Error" stat above and each week's bar in the chart below. Lower is
    better for both the model's own number and the market's -- the chart's real question is whether the
    model is closing the gap on the market's own accuracy over time, not just whether picks are winning.</p>
    <p>Recommended wager is a paper amount only, sized with 25% fractional Kelly against a running
    $500 starting bankroll that compounds through settled picks. Cover probability treats the margin
    model's prediction error as normally distributed around its point estimate, using its own measured
    RMSE on the 2025 holdout. The spread pick's price is the real median price across sportsbooks when
    a recent odds pull has one for that side.* Nothing here is real money or a recommendation to place
    a real bet.</p>
    <p class="footnote">* No per-book pricing was available for this game (too far out for the ~2-week
    odds board, or a name-matching miss) -- falls back to the standard -110-both-sides assumption
    instead of a measured value.</p>
    <p class="footnote">&dagger; The less-experienced team in this matchup has played fewer than 5
    games so far this season -- recent-form, ATS%, and opponent-SRS-averaging stats (each computed
    over a trailing window) are including games from a prior season to fill that window, not just
    the current one. ELO and SRS handle this season boundary explicitly (regressed toward the mean);
    these rolling stats don't yet, so treat the confidence tier with extra caution early in the season.</p>
    <p>Generated {generated_at}.</p>
  </footer>
</div>
</body>
</html>"""


def main():
    init_db()
    conn = get_connection()
    upcoming = fetch_upcoming(conn)
    results = fetch_results(conn)
    summary = fetch_summary(conn)
    bankroll = compute_current_bankroll(conn)
    weekly = fetch_weekly_performance(conn)

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(build_html(upcoming, results, summary, bankroll, weekly))
    print(f"Wrote {OUTPUT_PATH} ({len(upcoming)} upcoming, {len(results)} completed, "
          f"{len(weekly)} week(s) tracked, bankroll ${bankroll:.2f})")


if __name__ == "__main__":
    main()
