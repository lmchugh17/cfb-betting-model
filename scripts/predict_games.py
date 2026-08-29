"""Predicts specific upcoming games using current team state (ELO, SRS, rolling
form, ATS record, rest, H2H) computed as of right now, not "entering game X"
like the training pipeline. Loads the trained model bundle, prints a
prediction + grounded explanation facts for each requested game, and persists
the pick to the predictions table so the site can show a real track record.

Usage: .venv/bin/python scripts/predict_games.py <game_id> [<game_id> ...]
"""
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ats_and_situational import (compute_ats_results, compute_h2h_features,
                                      median_spread_per_game)
from src.box_score_features import add_derived_rate_stats, build_long_format
from src.db import get_connection, init_db
from src.elo import HOME_ADVANTAGE_ELO, CFBElo
from src.explain import build_feature_highlights, get_shap_contributions
from src.live_state import (compute_current_ats_pct, compute_current_elo,
                             compute_current_opponent_srs, compute_current_rest_days,
                             compute_current_rolling_form, compute_current_srs)
from src.model import FEATURE_COLUMNS

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "cfb_model.pkl"

# Placeholder thresholds on |edge| in points -- not statistically calibrated,
# just a first-pass tiering to distinguish "the model barely disagrees with the
# market" from "the model disagrees a lot". Worth revisiting once enough real
# picks have accumulated to see which tier actually performs best.
def confidence_tier(edge: float | None) -> str | None:
    if edge is None:
        return None
    abs_edge = abs(edge)
    if abs_edge >= 7:
        return "high"
    if abs_edge >= 3:
        return "medium"
    return "low"


def load_all_games(conn) -> list[dict]:
    rows = conn.execute("""
        SELECT id, year, week, season_type, start_date, neutral_site,
               home_team, away_team, home_points, away_points
        FROM games
    """).fetchall()
    cols = ["id", "year", "week", "season_type", "start_date", "neutral_site",
            "home_team", "away_team", "home_points", "away_points"]
    return [dict(zip(cols, r)) for r in rows]


def main():
    target_ids = [int(x) for x in sys.argv[1:]]
    if not target_ids:
        sys.exit("Usage: predict_games.py <game_id> [<game_id> ...]")

    init_db()
    bundle = joblib.load(MODEL_PATH)
    conn = get_connection()

    all_games = load_all_games(conn)
    targets = [g for g in all_games if g["id"] in target_ids]
    if len(targets) != len(target_ids):
        found = {g["id"] for g in targets}
        sys.exit(f"Game id(s) not found: {set(target_ids) - found}")

    # Exclude target games from "completed" state even if already played -- this keeps the
    # script usable as an honest backtest (predicting a game using only what was knowable
    # beforehand) as well as for genuinely upcoming games.
    completed = [g for g in all_games if g["home_points"] is not None and g["id"] not in target_ids]

    current_year = max(g["year"] for g in targets)

    print("Computing current ELO...")
    current_elo = compute_current_elo(completed)

    print("Computing current SRS...")
    current_srs = compute_current_srs(completed, current_year)

    print("Computing current rolling form...")
    completed_games_df = pd.DataFrame(completed)
    team_stats_df = pd.read_sql("SELECT * FROM team_game_stats", conn)
    long_df = build_long_format(completed_games_df, team_stats_df)
    long_df = add_derived_rate_stats(long_df)
    current_form = compute_current_rolling_form(long_df)
    current_opp_srs = compute_current_opponent_srs(long_df, current_srs)

    print("Computing current ATS record...")
    lines_rows = conn.execute("SELECT game_id, spread FROM lines").fetchall()
    spreads = median_spread_per_game(lines_rows)
    ats_rows = compute_ats_results(completed, spreads)
    current_ats = compute_current_ats_pct(ats_rows)

    print("Computing head-to-head...")
    h2h = compute_h2h_features(all_games)

    elo_model = CFBElo()  # only used for its expected_score() static method

    records = []
    for g in targets:
        home, away, neutral = g["home_team"], g["away_team"], bool(g["neutral_site"])
        elo_home, elo_away = current_elo.get(home, 1500.0), current_elo.get(away, 1500.0)
        home_bonus = 0 if neutral else HOME_ADVANTAGE_ELO
        elo_expected_home = elo_model.expected_score(elo_home + home_bonus, elo_away)

        srs_home, srs_away = current_srs.get(home, 0.0), current_srs.get(away, 0.0)

        game_date = date.fromisoformat(g["start_date"][:10])
        home_rest = compute_current_rest_days(completed, home, game_date, g["year"])
        away_rest = compute_current_rest_days(completed, away, game_date, g["year"])

        h2h_f = h2h.get(g["id"], {})

        record = {
            "game_id": g["id"], "home_team": home, "away_team": away,
            "elo_home": elo_home, "elo_away": elo_away, "elo_diff": elo_home - elo_away,
            "elo_expected_home": elo_expected_home,
            "srs_home": srs_home, "srs_away": srs_away, "srs_diff": srs_home - srs_away,
            "home_ats_pct": current_ats.get(home), "away_ats_pct": current_ats.get(away),
            "home_rest_days": home_rest, "away_rest_days": away_rest,
            "home_bye_week": home_rest >= 10, "away_bye_week": away_rest >= 10,
            "h2h_home_win_pct": h2h_f.get("h2h_home_win_pct"),
            "h2h_avg_home_margin": h2h_f.get("h2h_avg_home_margin"),
            "h2h_meetings": h2h_f.get("h2h_meetings"),
            "market_spread": spreads.get(g["id"]),
        }
        for stat in ("total_yards", "rushing_yards", "net_passing_yards", "yards_per_play",
                     "third_down_pct", "turnover_margin", "point_diff", "win"):
            home_val = current_form.get(home, {}).get(stat)
            away_val = current_form.get(away, {}).get(stat)
            record[f"home_avg_{stat}"] = home_val
            record[f"away_avg_{stat}"] = away_val
            record[f"diff_avg_{stat}"] = (home_val - away_val) if (home_val is not None and away_val is not None) else None
        record["home_avg_opponent_srs"] = current_opp_srs.get(home)
        record["away_avg_opponent_srs"] = current_opp_srs.get(away)
        if record["home_avg_opponent_srs"] is not None and record["away_avg_opponent_srs"] is not None:
            record["diff_avg_opponent_srs"] = record["home_avg_opponent_srs"] - record["away_avg_opponent_srs"]
        else:
            record["diff_avg_opponent_srs"] = None
        records.append(record)

    df = pd.DataFrame(records)
    X = df[FEATURE_COLUMNS].copy()
    for col in ("home_bye_week", "away_bye_week"):
        X[col] = X[col].astype(float)
    X = X.fillna(bundle["feature_medians"]).astype(float)

    win_probs = bundle["ensemble"].predict_proba(X)
    margins = bundle["regressor"].predict(X)
    predicted_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for i, row in df.iterrows():
        g = next(g for g in targets if g["id"] == row["game_id"])
        print(f"\n{'=' * 70}")
        print(f"{row['away_team']} @ {row['home_team']}")
        print(f"{'=' * 70}")
        print(f"Predicted: {row['home_team']} by {margins[i]:+.1f} (home win prob {win_probs[i]:.0%})")

        edge, pick_team = None, None
        if pd.notna(row["market_spread"]):
            fav = row["home_team"] if row["market_spread"] < 0 else row["away_team"]
            print(f"Market: {fav} favored by {abs(row['market_spread']):.1f}")
            edge = float(margins[i] - (-row["market_spread"]))
            pick_team = row["home_team"] if edge > 0 else row["away_team"]
            print(f"Edge: model favors {row['home_team']} by {edge:+.1f} vs. the market line -> pick {pick_team}")
        else:
            print("Market: no line available")

        contribs = get_shap_contributions(bundle["regressor"], X.iloc[[i]], FEATURE_COLUMNS)
        highlights = build_feature_highlights(row.to_dict(), contribs, row["home_team"], row["away_team"])
        print("\nTop factors:")
        for h in highlights:
            print(f"  - {h}")

        conn.execute(
            """INSERT OR REPLACE INTO predictions
               (game_id, predicted_at, year, week, season_type, start_date, home_team, away_team,
                predicted_margin, win_prob_home, market_spread, pick_team, edge, confidence_tier,
                highlights_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                int(row["game_id"]), predicted_at, g["year"], g["week"], g["season_type"], g["start_date"],
                row["home_team"], row["away_team"], float(margins[i]), float(win_probs[i]),
                float(row["market_spread"]) if pd.notna(row["market_spread"]) else None,
                pick_team, edge, confidence_tier(edge), json.dumps(highlights),
            ),
        )
    conn.commit()
    print(f"\nSaved {len(df)} prediction(s) to the predictions table.")


if __name__ == "__main__":
    main()
