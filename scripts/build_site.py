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


def fetch_upcoming(conn) -> list[dict]:
    rows = conn.execute("""
        SELECT p.game_id, p.year, p.week, p.start_date, p.home_team, p.away_team,
               p.predicted_margin, p.win_prob_home, p.market_spread, p.pick_team,
               p.edge, p.confidence_tier, p.highlights_json, p.tldr, p.bullets_json
        FROM predictions p
        JOIN games g ON p.game_id = g.id
        WHERE g.home_points IS NULL
        ORDER BY p.start_date
    """).fetchall()
    cols = ["game_id", "year", "week", "start_date", "home_team", "away_team",
            "predicted_margin", "win_prob_home", "market_spread", "pick_team",
            "edge", "confidence_tier", "highlights_json", "tldr", "bullets_json"]
    return [dict(zip(cols, r)) for r in rows]


def fetch_results(conn) -> list[dict]:
    rows = conn.execute("""
        SELECT game_id, year, week, start_date, home_team, away_team, predicted_margin,
               win_prob_home, market_spread, actual_margin, home_points, away_points, pick_team,
               edge, confidence_tier, highlights_json, tldr, bullets_json,
               pick_won_straight_up, pick_covered
        FROM prediction_results ORDER BY start_date DESC
    """).fetchall()
    cols = ["game_id", "year", "week", "start_date", "home_team", "away_team", "predicted_margin",
            "win_prob_home", "market_spread", "actual_margin", "home_points", "away_points", "pick_team",
            "edge", "confidence_tier", "highlights_json", "tldr", "bullets_json",
            "pick_won_straight_up", "pick_covered"]
    return [dict(zip(cols, r)) for r in rows]


def fetch_summary(conn) -> dict:
    row = conn.execute("""
        SELECT COUNT(*),
               SUM(pick_won_straight_up),
               SUM(CASE WHEN pick_covered = 1 THEN 1 ELSE 0 END),
               SUM(CASE WHEN pick_covered = 0 THEN 1 ELSE 0 END),
               SUM(CASE WHEN pick_covered IS NULL AND pick_team IS NOT NULL THEN 1 ELSE 0 END),
               AVG(ABS(predicted_margin - actual_margin))
        FROM prediction_results
    """).fetchone()
    n, su_wins, ats_wins, ats_losses, ats_pushes, avg_err = row
    return {
        "n": n or 0, "su_wins": su_wins or 0, "su_losses": (n or 0) - (su_wins or 0),
        "ats_wins": ats_wins or 0, "ats_losses": ats_losses or 0, "ats_pushes": ats_pushes or 0,
        "avg_err": avg_err,
    }


def fmt_kickoff(iso_str: str) -> str:
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    return dt.strftime("%a %b %-d, %-I:%M %p UTC")


def tier_badge(tier: str | None) -> str:
    if not tier:
        return ""
    return f'<span class="tier tier-{tier}">{tier}</span>'


def render_pick_card(p: dict, result: dict | None = None) -> str:
    highlights = json.loads(p["highlights_json"]) if p.get("highlights_json") else []
    bullets = json.loads(p["bullets_json"]) if p.get("bullets_json") else []
    tldr = p.get("tldr")

    market_line = "no line"
    if p["market_spread"] is not None:
        fav = p["home_team"] if p["market_spread"] < 0 else p["away_team"]
        market_line = f"{fav} by {abs(p['market_spread']):.1f}"

    pick_html = f'<div class="pick-line">Pick: <strong>{p["pick_team"]}</strong> {tier_badge(p["confidence_tier"])}</div>' if p.get("pick_team") else ""

    body_bullets = bullets or highlights
    bullets_html = "".join(f"<li>{b}</li>" for b in body_bullets)

    result_html = ""
    if result:
        su = "correct" if result["pick_won_straight_up"] else "incorrect"
        cover = {1: "covered", 0: "did not cover", None: "push"}[result["pick_covered"]]
        result_html = (
            f'<div class="result-line">Final: {result["home_team"]} {result["home_points"]:.0f} - '
            f'{result["away_points"]:.0f} {result["away_team"]} &mdash; pick was {su} straight-up, {cover} the spread</div>'
        )

    return f"""
    <div class="card">
      <div class="matchup">{p["away_team"]} @ {p["home_team"]}</div>
      <div class="kickoff">{fmt_kickoff(p["start_date"])}</div>
      <div class="prediction-line">Model: {p["home_team"]} by {p["predicted_margin"]:+.1f} ({p["win_prob_home"]:.0%} win prob) &middot; Market: {market_line}</div>
      {pick_html}
      {f'<div class="tldr">{tldr}</div>' if tldr else ""}
      <ul class="bullets">{bullets_html}</ul>
      {result_html}
    </div>"""


def render_stat_tile(label: str, value: str) -> str:
    return f'<div class="stat-tile"><div class="stat-value">{value}</div><div class="stat-label">{label}</div></div>'


def build_html(upcoming: list[dict], results: list[dict], summary: dict) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    su_pct = f"{summary['su_wins']}/{summary['n']}" if summary["n"] else "0/0"
    ats_pct = (f"{summary['ats_wins']}-{summary['ats_losses']}-{summary['ats_pushes']}"
               if summary["n"] else "0-0-0")
    avg_err = f"{summary['avg_err']:.1f} pts" if summary["avg_err"] is not None else "n/a"

    stat_tiles = "".join([
        render_stat_tile("Picks Tracked", str(summary["n"])),
        render_stat_tile("Straight-Up", su_pct),
        render_stat_tile("Against the Spread", ats_pct),
        render_stat_tile("Avg. Margin Error", avg_err),
    ])

    upcoming_html = "".join(render_pick_card(p) for p in upcoming) or '<p class="empty">No upcoming games with picks right now.</p>'
    results_html = "".join(render_pick_card(r, result=r) for r in results) or '<p class="empty">No completed games yet.</p>'

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
  .tier {{ font-size: 0.7rem; text-transform: uppercase; padding: 0.1rem 0.5rem; border-radius: 999px; margin-left: 0.4rem; }}
  .tier-high {{ background: rgba(61,220,132,0.15); color: var(--green); }}
  .tier-medium {{ background: rgba(255,184,79,0.15); color: var(--amber); }}
  .tier-low {{ background: rgba(154,161,172,0.15); color: var(--text-dim); }}
  .tldr {{ font-style: italic; color: var(--text-dim); font-size: 0.88rem; margin-bottom: 0.6rem; }}
  .bullets {{ margin: 0; padding-left: 1.1rem; font-size: 0.85rem; color: var(--text-dim); }}
  .bullets li {{ margin-bottom: 0.25rem; }}
  .result-line {{ margin-top: 0.7rem; padding-top: 0.6rem; border-top: 1px solid var(--border); font-size: 0.85rem; }}
  .empty {{ color: var(--text-dim); font-size: 0.9rem; }}
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
    <p>Market spread is deliberately excluded from model training and used only to compute the edge shown
    here. Confidence tiers are based on |edge| in points and are not statistically calibrated yet --
    treat them as a rough first pass, not a validated signal.</p>
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

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(build_html(upcoming, results, summary))
    print(f"Wrote {OUTPUT_PATH} ({len(upcoming)} upcoming, {len(results)} completed)")


if __name__ == "__main__":
    main()
