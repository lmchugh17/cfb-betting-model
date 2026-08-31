"""Renders the static dashboard (docs/index.html) from the predictions table +
prediction_results view. Self-contained HTML/CSS, no build step, no external
assets -- deployable straight to GitHub Pages via a plain git push (see the
scheduled Claude Code session, task 11).
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import get_connection, init_db

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "docs" / "index.html"

STARTING_BANKROLL = 500.0
# Matches scripts/predict_games.py's own assumption (no per-book spread juice data available).
ASSUMED_SPREAD_ODDS_AMERICAN = -110

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
               p.moneyline_pick, p.moneyline_win_prob, p.moneyline_confidence_tier
        FROM predictions p
        JOIN games g ON p.game_id = g.id
        WHERE g.home_points IS NULL
        ORDER BY p.start_date
    """).fetchall()
    cols = ["game_id", "year", "week", "start_date", "home_team", "away_team",
            "predicted_margin", "win_prob_home", "market_spread", "pick_team",
            "edge", "confidence_tier", "highlights_json", "tldr", "bullets_json",
            "model_breakdown_json", "cover_probability", "kelly_fraction",
            "moneyline_pick", "moneyline_win_prob", "moneyline_confidence_tier"]
    return [dict(zip(cols, r)) for r in rows]


def fetch_results(conn) -> list[dict]:
    rows = conn.execute("""
        SELECT game_id, year, week, start_date, home_team, away_team, predicted_margin,
               win_prob_home, market_spread, actual_margin, home_points, away_points, pick_team,
               edge, confidence_tier, highlights_json, tldr, bullets_json,
               ats_pick_won_straight_up, pick_covered, model_breakdown_json, cover_probability, kelly_fraction,
               moneyline_pick, moneyline_win_prob, moneyline_confidence_tier, moneyline_pick_won
        FROM prediction_results ORDER BY start_date DESC
    """).fetchall()
    cols = ["game_id", "year", "week", "start_date", "home_team", "away_team", "predicted_margin",
            "win_prob_home", "market_spread", "actual_margin", "home_points", "away_points", "pick_team",
            "edge", "confidence_tier", "highlights_json", "tldr", "bullets_json",
            "ats_pick_won_straight_up", "pick_covered", "model_breakdown_json", "cover_probability", "kelly_fraction",
            "moneyline_pick", "moneyline_win_prob", "moneyline_confidence_tier", "moneyline_pick_won"]
    return [dict(zip(cols, r)) for r in rows]


def fetch_weekly_performance(conn) -> list[dict]:
    """One row per (year, week) with that week's moneyline and ATS record --
    lets the season-long summary tiles be checked against week-by-week form
    instead of only a single cumulative number."""
    rows = conn.execute("""
        SELECT year, week, COUNT(*),
               SUM(CASE WHEN moneyline_pick_won = 1 THEN 1 ELSE 0 END),
               SUM(CASE WHEN moneyline_pick IS NOT NULL THEN 1 ELSE 0 END),
               SUM(CASE WHEN pick_covered = 1 THEN 1 ELSE 0 END),
               SUM(CASE WHEN pick_covered = 0 THEN 1 ELSE 0 END),
               SUM(CASE WHEN pick_covered IS NULL AND pick_team IS NOT NULL THEN 1 ELSE 0 END),
               AVG(ABS(predicted_margin - actual_margin))
        FROM prediction_results
        GROUP BY year, week
        ORDER BY year DESC, week DESC
    """).fetchall()
    cols = ["year", "week", "n", "ml_wins", "ml_decided", "ats_wins", "ats_losses", "ats_pushes", "avg_err"]
    return [dict(zip(cols, r)) for r in rows]


def compute_current_bankroll(conn) -> float:
    """Chronological paper-bankroll replay: starts at STARTING_BANKROLL and compounds
    through every settled (non-push) pick in date order using its own kelly_fraction and
    the -110 payout assumption. Upcoming picks size their recommended wager off this
    running total rather than the fixed starting amount, same as the NBA reference site."""
    rows = conn.execute("""
        SELECT kelly_fraction, pick_covered FROM prediction_results
        WHERE kelly_fraction IS NOT NULL AND pick_covered IS NOT NULL
        ORDER BY start_date ASC
    """).fetchall()
    bankroll = STARTING_BANKROLL
    b = _net_decimal_odds(ASSUMED_SPREAD_ODDS_AMERICAN)
    for kelly_fraction, covered in rows:
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
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    return dt.strftime("%a %b %-d, %-I:%M %p UTC")


def tier_badge(tier: str | None) -> str:
    if not tier:
        return ""
    return f'<span class="tier tier-{tier}">{tier.upper()} Confidence</span>'


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

    ml_html = ""
    if p.get("moneyline_pick"):
        ml_html = (
            f'<div class="pick-line">Moneyline pick: <strong>{p["moneyline_pick"]}</strong> '
            f'({p["moneyline_win_prob"]:.0%} win prob) {tier_badge(p["moneyline_confidence_tier"])}</div>'
        )
    pick_html = ""
    if p.get("pick_team"):
        # The pick's own spread, signed from ITS perspective (not always the home-signed
        # market_spread) -- e.g. the underdog pick shows a positive number (points received),
        # the favorite pick shows negative (points given). Odds shown are the standing
        # -110-both-sides assumption (see ASSUMED_SPREAD_ODDS_AMERICAN), not a measured price.
        pick_spread_html = ""
        if p["market_spread"] is not None:
            pick_spread = p["market_spread"] if p["pick_team"] == p["home_team"] else -p["market_spread"]
            pick_spread_html = f' {pick_spread:+.1f} <span class="odds">({ASSUMED_SPREAD_ODDS_AMERICAN})</span>'
        pick_html = (
            f'<div class="pick-line">Spread pick: <strong>{p["pick_team"]}{pick_spread_html}</strong> '
            f'{tier_badge(p["confidence_tier"])}</div>'
        )
    wager_html = render_wager_line(p, bankroll)
    model_breakdown_html = render_model_breakdown(p.get("model_breakdown_json"))

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
      {wager_html}
      {f'<div class="tldr">{tldr}</div>' if tldr else ""}
      <ul class="bullets">{bullets_html}</ul>
      {model_breakdown_html}
      {result_html}
    </div>"""


def render_stat_tile(label: str, value: str) -> str:
    return f'<div class="stat-tile"><div class="stat-value">{value}</div><div class="stat-label">{label}</div></div>'


def render_weekly_table(weekly: list[dict]) -> str:
    if not weekly:
        return '<p class="empty">No completed weeks tracked yet.</p>'
    rows_html = ""
    for w in weekly:
        ml_pct = f"{w['ml_wins']}-{w['ml_decided'] - w['ml_wins']} ({w['ml_wins']/w['ml_decided']:.0%})" if w["ml_decided"] else "n/a"
        ats_pct = f"{w['ats_wins']}-{w['ats_losses']}-{w['ats_pushes']}"
        avg_err = f"{w['avg_err']:.1f}" if w["avg_err"] is not None else "n/a"
        season_type_label = "Postseason" if w.get("week") is None else f"Week {w['week']}"
        rows_html += (
            f"<tr><td>{w['year']} {season_type_label}</td><td>{w['n']}</td>"
            f"<td>{ml_pct}</td><td>{ats_pct}</td><td>{avg_err} pts</td></tr>"
        )
    return f"""<div class="table-wrap"><table class="weekly-table">
      <thead><tr><th>Week</th><th>Games</th><th>Moneyline</th><th>ATS</th><th>Avg. Error</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table></div>"""


def render_weekly_win_pct_chart(weekly: list[dict]) -> str:
    """Moneyline win% per week as a simple CSS bar chart -- oldest week first (left to
    right) so it reads as a trend, opposite of the table's newest-first order. Bar height
    is win% of vertical space; a dashed line at the 50% mark gives a coin-flip reference."""
    decided = [w for w in weekly if w["ml_decided"]]
    if not decided:
        return '<p class="empty">No completed weeks tracked yet.</p>'
    bars_html = ""
    for w in reversed(decided):  # weekly is newest-first; chart wants oldest-first
        pct = w["ml_wins"] / w["ml_decided"]
        label = "Postseason" if w.get("week") is None else f"Wk {w['week']}"
        css_class = "above" if pct >= 0.5 else "below"
        bars_html += (
            '<div class="bar-col">'
            f'<div class="bar-value">{pct:.0%}</div>'
            f'<div class="bar-track"><div class="bar-fill {css_class}" style="height: {pct * 100:.1f}%"></div></div>'
            f'<div class="bar-label">{w["year"]} {label}</div>'
            "</div>"
        )
    return f'<div class="bar-chart"><div class="bar-ref-line"></div>{bars_html}</div>'


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
    weekly_bar_html = render_weekly_win_pct_chart(weekly)

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
  .tldr {{ font-style: italic; color: var(--text-dim); font-size: 0.88rem; margin-bottom: 0.6rem; }}
  .bullets {{ margin: 0; padding-left: 1.1rem; font-size: 0.85rem; color: var(--text-dim); }}
  .bullets li {{ margin-bottom: 0.25rem; }}
  .model-breakdown {{ margin-top: 0.7rem; font-size: 0.8rem; color: var(--text-dim); }}
  .model-breakdown summary {{ cursor: pointer; }}
  .model-row {{ display: flex; justify-content: space-between; padding: 0.2rem 0 0.2rem 1rem; }}
  .result-line {{ margin-top: 0.7rem; padding-top: 0.6rem; border-top: 1px solid var(--border); font-size: 0.85rem; }}
  .empty {{ color: var(--text-dim); font-size: 0.9rem; }}
  .bar-chart {{ position: relative; display: flex; align-items: flex-end; gap: 0.75rem; background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 1.25rem 1rem 1rem; margin-bottom: 1rem; overflow-x: auto; }}
  .bar-ref-line {{ position: absolute; left: 1rem; right: 1rem; bottom: calc(1rem + 60px); border-top: 1px dashed var(--border); }}
  .bar-col {{ display: flex; flex-direction: column; align-items: center; flex: 0 0 auto; width: 52px; }}
  .bar-value {{ font-size: 0.72rem; color: var(--text-dim); margin-bottom: 0.3rem; }}
  .bar-track {{ width: 32px; height: 120px; background: rgba(255,255,255,0.05); border-radius: 4px; display: flex; align-items: flex-end; overflow: hidden; }}
  .bar-fill {{ width: 100%; border-radius: 3px 3px 0 0; }}
  .bar-fill.above {{ background: var(--green); }}
  .bar-fill.below {{ background: var(--red); }}
  .bar-label {{ font-size: 0.68rem; color: var(--text-dim); margin-top: 0.4rem; text-align: center; white-space: nowrap; }}
  .table-wrap {{ overflow-x: auto; margin-bottom: 1rem; }}
  .weekly-table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; background: var(--card); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }}
  .weekly-table th, .weekly-table td {{ padding: 0.6rem 0.9rem; text-align: left; white-space: nowrap; }}
  .weekly-table th {{ color: var(--text-dim); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.03em; border-bottom: 1px solid var(--border); }}
  .weekly-table tbody tr:not(:last-child) td {{ border-bottom: 1px solid var(--border); }}
  footer {{ margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid var(--border); color: var(--text-dim); font-size: 0.78rem; line-height: 1.5; }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>CFB Betting Model</h1>
    <p class="tagline">Opponent-adjusted college football predictions vs. the market. Personal research project, not financial advice.</p>
  </header>

  <div class="stats-row">{stat_tiles}</div>

  <h2>This Week's Picks</h2>
  {upcoming_html}

  <h2>Weekly Performance</h2>
  {weekly_bar_html}
  {weekly_html}

  <h2>Recent Results</h2>
  {results_html}

  <footer>
    <p><strong>Methodology:</strong> 5-model stacked ensemble (logistic regression, random forest,
    XGBoost, LightGBM, extra trees) for win probability, plus a separate XGBoost regressor for
    predicted margin. Features: ELO ratings, opponent-adjusted SRS ratings (iterative strength-of-schedule
    solve), rolling 4-game team form, rolling ATS record, rest/bye-week, head-to-head history.
    Trained on 2021-2024 FBS seasons, held out all of 2025 for evaluation (75.5% straight-up accuracy,
    appropriately trailing the market's own 76.8% -- landing behind the market, not matching or beating it,
    is the healthy sign of a real model rather than a leakage bug).</p>
    <p>Two separate picks are shown per game: a <strong>moneyline pick</strong> (whichever team the model
    gives &gt;50% win probability -- who wins outright, no spread) and a <strong>spread pick</strong> (which
    side has value against the market line). These are often different teams, deliberately -- a big
    underdog can be the correct spread pick (expected to lose, just by less than the market thinks) while
    still being the wrong moneyline pick. Market spread is deliberately excluded from model training and
    used only to compute the spread pick's edge. Both confidence tiers are first-pass, not statistically
    calibrated: spread tiers are based on |edge| in points, moneyline tiers on the picked side's win
    probability (&ge;75% high, &ge;60% medium, below that low).</p>
    <p>Recommended wager is a paper amount only, sized with 25% fractional Kelly against a running
    $500 starting bankroll that compounds through settled picks. Cover probability treats the margin
    model's prediction error as normally distributed around its point estimate, using its own measured
    RMSE on the 2025 holdout. Assumes standard -110 pricing on both sides since per-book spread juice
    isn't tracked -- a simplification, not a measured value. Nothing here is real money or a
    recommendation to place a real bet.</p>
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
