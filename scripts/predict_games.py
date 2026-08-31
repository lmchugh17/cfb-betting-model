"""Predicts specific upcoming games using current team state (ELO, SRS, rolling
form, ATS record, rest, H2H) computed as of right now, not "entering game X"
like the training pipeline. Loads the trained model bundle, prints a
prediction + grounded explanation facts for each requested game, and persists
the pick to the predictions table so the site can show a real track record.

Usage: .venv/bin/python scripts/predict_games.py <game_id> [<game_id> ...]
"""
import json
import math
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
from src.live_state import (ATS_WINDOW, ROLLING_WINDOW, compute_current_ats_pct,
                             compute_current_elo, compute_current_opponent_srs,
                             compute_current_rest_days, compute_current_rolling_form,
                             compute_current_srs)
from src.model import FEATURE_COLUMNS
from src.spread_pricing import get_spread_price, load_latest_spread_prices
from src.weather_features import (compute_current_adverse_wx_ats_pct, load_weather_by_game,
                                   was_game_adverse)

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


# Same "not statistically calibrated, first-pass" caveat as confidence_tier above, but on
# a separate scale: win probability and point-edge aren't on the same footing (a small
# favorite can carry a big point edge on the spread while still being close to a coin
# flip to win outright), so the moneyline pick needs its own tiering, not confidence_tier's
# edge-based thresholds reused.
def moneyline_confidence_tier(win_prob_picked_side: float | None) -> str | None:
    if win_prob_picked_side is None:
        return None
    if win_prob_picked_side >= 0.75:
        return "high"
    if win_prob_picked_side >= 0.60:
        return "medium"
    return "low"


KELLY_FRACTION_CAP = 0.25  # 25% fractional Kelly, matches the NBA reference model's own cap
# CFBD's /lines (the source of market_spread) gives the spread number but not its price --
# src.spread_pricing looks up the real per-book median price from live_odds when a recent
# pull has it. This is the fallback for when it doesn't (game too far out for the ~2-week
# odds board, or a team-name match miss) -- a stated assumption, not a measured value.
ASSUMED_SPREAD_ODDS_AMERICAN = -110


def _american_odds_to_net_decimal(odds: int) -> float:
    return 100 / abs(odds) if odds < 0 else odds / 100


def _normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def cover_probability_and_kelly(edge: float | None, is_home_pick: bool, regressor_rmse: float,
                                 spread_odds_american: int = ASSUMED_SPREAD_ODDS_AMERICAN
                                 ) -> tuple[float | None, float | None]:
    """cover_probability is for the PICKED side specifically, not always the home side --
    edge is defined as home's edge over the market, so picking the away side needs 1 minus
    the home cover probability. Treats the regressor's residuals as approximately
    Normal(0, rmse) around its point estimate -- rmse is the model's own measured error on
    the 2025 holdout (see scripts/train_model.py), not an assumed number. kelly_fraction is
    the recommended fraction of bankroll to wager, already capped at KELLY_FRACTION_CAP; the
    site converts this to a dollar amount against the running paper bankroll at display time.
    spread_odds_american defaults to the standing assumption but should be the real measured
    per-book price (src.spread_pricing) when the caller has one -- makes the Kelly math
    reflect the actual price being bet, not just the number of points."""
    if edge is None:
        return None, None
    p_home_covers = _normal_cdf(edge / regressor_rmse)
    p_cover = p_home_covers if is_home_pick else (1 - p_home_covers)
    b = _american_odds_to_net_decimal(spread_odds_american)
    kelly_full = p_cover - (1 - p_cover) / b
    kelly = max(0.0, kelly_full) * KELLY_FRACTION_CAP
    return p_cover, kelly


# The larger of the two rolling windows compute_current_rolling_form/compute_current_ats_pct
# actually use (src.live_state) -- a team needs at least this many CURRENT-season games for
# neither window to be quietly blending in games from a prior season. ELO and SRS already
# handle the season boundary explicitly (regress toward the mean, see src.elo/opponent_adjustment);
# rolling form, ATS%, and opponent-SRS averaging don't -- they just take the trailing N completed
# games regardless of year, which is exactly last season's form at the start of a new one.
FULL_SEASON_WINDOW = max(ROLLING_WINDOW, ATS_WINDOW)


def count_current_season_games(completed: list[dict], year: int) -> dict:
    """Returns {team: games played in `year` so far}, from the same `completed` list used
    for every other current-state computation."""
    counts: dict[str, int] = {}
    for g in completed:
        if g["year"] != year:
            continue
        counts[g["home_team"]] = counts.get(g["home_team"], 0) + 1
        counts[g["away_team"]] = counts.get(g["away_team"], 0) + 1
    return counts


def load_all_games(conn) -> list[dict]:
    rows = conn.execute("""
        SELECT id, year, week, season_type, start_date, neutral_site,
               home_id, home_team, away_id, away_team, home_points, away_points
        FROM games
    """).fetchall()
    cols = ["id", "year", "week", "season_type", "start_date", "neutral_site",
            "home_id", "home_team", "away_id", "away_team", "home_points", "away_points"]
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
    games_played_this_season = count_current_season_games(completed, current_year)

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

    print("Computing adverse-weather ATS splits...")
    weather_by_game = load_weather_by_game(conn)
    current_adverse_wx_ats = compute_current_adverse_wx_ats_pct(ats_rows, weather_by_game)

    print("Loading latest per-book spread pricing...")
    spread_prices = load_latest_spread_prices(conn)

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
            "game_id": g["id"], "home_id": g["home_id"], "away_id": g["away_id"],
            "min_current_season_games": min(games_played_this_season.get(home, 0),
                                             games_played_this_season.get(away, 0)),
            "home_team": home, "away_team": away,
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

        wx = weather_by_game.get(g["id"], {})
        record["is_adverse_weather"] = int(was_game_adverse(g["id"], weather_by_game))
        record["weather_temperature_f"] = wx.get("temperature_f")
        record["weather_wind_speed_mph"] = wx.get("wind_speed_mph")
        record["weather_precipitation_in"] = wx.get("precipitation_in")
        home_wx_pct = current_adverse_wx_ats.get(home)
        away_wx_pct = current_adverse_wx_ats.get(away)
        record["home_adverse_wx_ats_pct"] = home_wx_pct
        record["away_adverse_wx_ats_pct"] = away_wx_pct
        # Zero (not missing) when this game isn't itself in adverse weather, or either
        # team lacks enough adverse-weather history yet -- matches build_features.py.
        record["adverse_wx_ats_edge"] = (
            (home_wx_pct - away_wx_pct)
            if record["is_adverse_weather"] and home_wx_pct is not None and away_wx_pct is not None
            else 0.0
        )
        records.append(record)

    df = pd.DataFrame(records)
    X = df[FEATURE_COLUMNS].copy()
    for col in ("home_bye_week", "away_bye_week", "is_adverse_weather"):
        X[col] = X[col].astype(float)
    X = X.fillna(bundle["feature_medians"]).astype(float)

    detailed = bundle["ensemble"].predict_proba_detailed(X)
    win_probs = detailed["final"]
    margins = bundle["regressor"].predict(X)
    regressor_rmse = bundle["regressor_metrics"]["rmse"]
    predicted_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for i, row in df.iterrows():
        g = next(g for g in targets if g["id"] == row["game_id"])
        print(f"\n{'=' * 70}")
        print(f"{row['away_team']} @ {row['home_team']}")
        print(f"{'=' * 70}")
        print(f"Predicted: {row['home_team']} by {margins[i]:+.1f} (home win prob {win_probs[i]:.0%})")
        if row["min_current_season_games"] < FULL_SEASON_WINDOW:
            print(f"NOTE: recent-form/ATS%/opponent-SRS stats include prior-season games -- "
                  f"the less-experienced team has only played {row['min_current_season_games']} "
                  f"game(s) so far this season (need {FULL_SEASON_WINDOW} for a fully current window).")
        print("Per-model win probability (home team):")
        for name, probs in detailed["base"].items():
            print(f"  {name}: {probs[i]:.0%}")

        moneyline_pick = row["home_team"] if win_probs[i] > 0.5 else row["away_team"]
        moneyline_win_prob = float(win_probs[i] if moneyline_pick == row["home_team"] else 1 - win_probs[i])
        ml_tier = moneyline_confidence_tier(moneyline_win_prob)
        print(f"Moneyline pick: {moneyline_pick} ({moneyline_win_prob:.0%} win prob, {ml_tier} confidence)")

        edge, pick_team = None, None
        cover_prob, kelly = None, None
        spread_price, spread_price_source, spread_price_book_count = None, None, 0
        if pd.notna(row["market_spread"]):
            fav = row["home_team"] if row["market_spread"] < 0 else row["away_team"]
            print(f"Market: {fav} favored by {abs(row['market_spread']):.1f}")
            edge = float(margins[i] - (-row["market_spread"]))
            pick_team = row["home_team"] if edge > 0 else row["away_team"]
            print(f"Edge: model favors {row['home_team']} by {edge:+.1f} vs. the market line -> pick {pick_team}")

            pick_team_id = row["home_id"] if pick_team == row["home_team"] else row["away_id"]
            measured_price, spread_price_book_count = get_spread_price(spread_prices, pick_team_id)
            if measured_price is not None:
                spread_price, spread_price_source = measured_price, "measured"
                print(f"Spread price: {spread_price} (median across {spread_price_book_count} book(s))")
            else:
                spread_price, spread_price_source = ASSUMED_SPREAD_ODDS_AMERICAN, "assumed"
                print(f"Spread price: {spread_price} (assumed -- no per-book pricing available for this game)")

            cover_prob, kelly = cover_probability_and_kelly(
                edge, pick_team == row["home_team"], regressor_rmse, spread_price)
            print(f"Cover probability: {cover_prob:.0%} -> {kelly:.1%} of bankroll recommended (25% Kelly)")
        else:
            print("Market: no line available")

        contribs = get_shap_contributions(bundle["regressor"], X.iloc[[i]], FEATURE_COLUMNS)
        highlights = build_feature_highlights(row.to_dict(), contribs, row["home_team"], row["away_team"])
        print("\nTop factors:")
        for h in highlights:
            print(f"  - {h}")

        # Plain INSERT OR REPLACE would delete+reinsert the row, wiping tldr/bullets_json
        # (written separately, once, by whoever explains the pick) back to NULL every time
        # this game gets re-predicted later in the week with fresher odds. ON CONFLICT DO
        # UPDATE only touches the columns this script owns.
        model_breakdown = {name: float(probs[i]) for name, probs in detailed["base"].items()}

        conn.execute(
            """INSERT INTO predictions
               (game_id, predicted_at, year, week, season_type, start_date, home_team, away_team,
                predicted_margin, win_prob_home, market_spread, pick_team, edge, confidence_tier,
                highlights_json, model_breakdown_json, cover_probability, kelly_fraction,
                moneyline_pick, moneyline_win_prob, moneyline_confidence_tier,
                spread_price, spread_price_source, spread_price_book_count, min_current_season_games)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(game_id) DO UPDATE SET
                   predicted_at=excluded.predicted_at, year=excluded.year, week=excluded.week,
                   season_type=excluded.season_type, start_date=excluded.start_date,
                   home_team=excluded.home_team, away_team=excluded.away_team,
                   predicted_margin=excluded.predicted_margin, win_prob_home=excluded.win_prob_home,
                   market_spread=excluded.market_spread, pick_team=excluded.pick_team,
                   edge=excluded.edge, confidence_tier=excluded.confidence_tier,
                   highlights_json=excluded.highlights_json, model_breakdown_json=excluded.model_breakdown_json,
                   cover_probability=excluded.cover_probability, kelly_fraction=excluded.kelly_fraction,
                   moneyline_pick=excluded.moneyline_pick, moneyline_win_prob=excluded.moneyline_win_prob,
                   moneyline_confidence_tier=excluded.moneyline_confidence_tier,
                   spread_price=excluded.spread_price, spread_price_source=excluded.spread_price_source,
                   spread_price_book_count=excluded.spread_price_book_count,
                   min_current_season_games=excluded.min_current_season_games""",
            (
                int(row["game_id"]), predicted_at, g["year"], g["week"], g["season_type"], g["start_date"],
                row["home_team"], row["away_team"], float(margins[i]), float(win_probs[i]),
                float(row["market_spread"]) if pd.notna(row["market_spread"]) else None,
                pick_team, edge, confidence_tier(edge), json.dumps(highlights),
                json.dumps(model_breakdown), cover_prob, kelly,
                moneyline_pick, moneyline_win_prob, ml_tier,
                spread_price, spread_price_source, spread_price_book_count,
                int(row["min_current_season_games"]),
            ),
        )
    conn.commit()
    print(f"\nSaved {len(df)} prediction(s) to the predictions table.")


if __name__ == "__main__":
    main()
