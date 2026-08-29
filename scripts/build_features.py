"""Assembles the final per-game feature table from everything backfilled so far:
ELO, opponent-adjusted SRS, rolling box-score form, ATS record, rest/bye-week,
and head-to-head history. Writes to the `game_features` table (one row per
completed FBS game, 2021-2025) for model training (task 8).

Only completed games (both scores present) are included -- this is a training
table, not a prediction input. Predicting an upcoming week's games will reuse
these same feature functions but isn't built yet (that happens alongside the
live weekly pipeline in a later task).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ats_and_situational import (compute_ats_results, compute_h2h_features,
                                      compute_rest_days, compute_rolling_ats_pct,
                                      median_spread_per_game)
from src.box_score_features import (add_derived_rate_stats, assemble_game_features,
                                     build_long_format, compute_rolling_form)
from src.db import get_connection, init_db
from src.elo import CFBElo
from src.opponent_adjustment import compute_weekly_srs


def load_games(conn) -> list[dict]:
    rows = conn.execute("""
        SELECT id, year, week, season_type, start_date, neutral_site,
               home_team, away_team, home_points, away_points
        FROM games WHERE home_points IS NOT NULL AND away_points IS NOT NULL
    """).fetchall()
    cols = ["id", "year", "week", "season_type", "start_date", "neutral_site",
            "home_team", "away_team", "home_points", "away_points"]
    return [dict(zip(cols, r)) for r in rows]


def compute_elo_features(games: list[dict]) -> dict:
    games_sorted = sorted(games, key=lambda g: (g["year"], g["start_date"]))
    elo = CFBElo()
    result = {}
    for g in games_sorted:
        elo.maybe_regress_for_new_season(g["year"])
        result[g["id"]] = elo.pre_game_features(g["home_team"], g["away_team"], bool(g["neutral_site"]))
        elo.update(g["home_team"], g["away_team"], g["home_points"], g["away_points"], bool(g["neutral_site"]))
    return result


def main():
    init_db()
    conn = get_connection()

    try:
        games = load_games(conn)
        print(f"Building features for {len(games)} completed games...")
        games_df = pd.DataFrame(games)

        print("Computing ELO...")
        elo_features = compute_elo_features(games)

        print("Computing opponent-adjusted SRS...")
        srs_lookup = compute_weekly_srs(games)

        print("Computing rolling box-score form...")
        team_stats_df = pd.read_sql("SELECT * FROM team_game_stats", conn)
        long_df = build_long_format(games_df, team_stats_df)
        long_df = add_derived_rate_stats(long_df)
        rolling_df = compute_rolling_form(long_df, srs_lookup)
        box_score_features = assemble_game_features(rolling_df)

        print("Computing ATS record...")
        lines_rows = conn.execute("SELECT game_id, spread FROM lines").fetchall()
        spreads = median_spread_per_game(lines_rows)
        ats_rows = compute_ats_results(games, spreads)
        rolling_ats = compute_rolling_ats_pct(ats_rows)

        print("Computing rest/bye-week...")
        rest = compute_rest_days(games)

        print("Computing head-to-head...")
        h2h = compute_h2h_features(games)

        print("Assembling final table...")
        records = []
        for g in games:
            gid = g["id"]
            elo_f = elo_features.get(gid, {})
            rest_f = rest.get(gid, {})
            h2h_f = h2h.get(gid, {})
            record = {
                "game_id": gid, "year": g["year"], "week": g["week"], "season_type": g["season_type"],
                "home_team": g["home_team"], "away_team": g["away_team"],
                "home_points": g["home_points"], "away_points": g["away_points"],
                "home_margin": g["home_points"] - g["away_points"],
                "home_win": int(g["home_points"] > g["away_points"]),
                "market_spread": spreads.get(gid),
                "elo_home": elo_f.get("elo_home"), "elo_away": elo_f.get("elo_away"),
                "elo_diff": elo_f.get("elo_diff"), "elo_expected_home": elo_f.get("elo_expected_home"),
                "srs_home": srs_lookup.get((g["year"], g["week"], g["home_team"])),
                "srs_away": srs_lookup.get((g["year"], g["week"], g["away_team"])),
                "home_ats_pct": rolling_ats.get((gid, g["home_team"])),
                "away_ats_pct": rolling_ats.get((gid, g["away_team"])),
                "home_rest_days": rest_f.get("home_rest_days"), "away_rest_days": rest_f.get("away_rest_days"),
                "home_bye_week": rest_f.get("home_bye_week"), "away_bye_week": rest_f.get("away_bye_week"),
                "h2h_home_win_pct": h2h_f.get("h2h_home_win_pct"),
                "h2h_avg_home_margin": h2h_f.get("h2h_avg_home_margin"),
                "h2h_meetings": h2h_f.get("h2h_meetings"),
            }
            if record["srs_home"] is not None and record["srs_away"] is not None:
                record["srs_diff"] = record["srs_home"] - record["srs_away"]
            records.append(record)

        final_df = pd.DataFrame(records).merge(box_score_features, on="game_id", how="left")

        final_df.to_sql("game_features", conn, if_exists="replace", index=False)
        conn.commit()

        print(f"\nWrote {len(final_df)} rows to game_features ({len(final_df.columns)} columns).")
        non_null_pct = final_df.notna().mean().sort_values()
        print("\nColumns with the most missing data (worth knowing before modeling):")
        print(non_null_pct.head(10))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
