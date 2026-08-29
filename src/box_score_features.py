"""Rolling team-form features from per-game box scores, in the spirit of the
reference NBA model's Four-Factors rolling stats -- but with two CFB-specific
departures:

1. Window=4 games, min_periods=2 (vs the NBA model's window=10, min_periods=5).
   A 13-game CFB season can't spare a 5-game burn-in -- that's over a third
   of the season sitting out with NaN features.

2. Opponent-adjustment: rather than hand-building a bespoke SOS-adjustment for
   every single box-score column (yards, 3rd down%, etc. -- a large modeling
   effort on its own), each team's rolling opponent SRS rating (see
   opponent_adjustment.py) over the same window is included as its own
   feature alongside the raw rolling stats. This lets the actual model
   (task 8) learn how much to discount a raw stat based on who it was earned
   against, rather than us guessing a hand-tuned adjustment formula. The
   *true* opponent-adjusted signal in this feature set is the SRS margin
   rating itself and ELO -- these box-score rolling stats are raw performance
   plus a schedule-strength control, not independently opponent-adjusted.
"""
import pandas as pd

ROLLING_WINDOW = 4
MIN_PERIODS = 2

RATE_STAT_SOURCE_COLUMNS = [
    "total_yards", "rushing_yards", "net_passing_yards", "turnovers",
    "first_downs", "penalty_yards", "sacks", "tackles_for_loss",
]


def build_long_format(games_df: pd.DataFrame, team_stats_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, game), with the team's own stats plus opponent's
    stats (OPP_ prefix), matching the NBA model's team-perspective reshape."""
    merged = team_stats_df.merge(
        games_df[["id", "year", "week", "season_type", "start_date"]],
        left_on="game_id", right_on="id", how="inner",
    )

    stat_cols = [c for c in team_stats_df.columns if c not in ("game_id", "team_id", "team", "home_away", "points")]

    home = merged[merged["home_away"] == "home"].copy()
    away = merged[merged["home_away"] == "away"].copy()
    paired = home.merge(away, on="game_id", suffixes=("_home", "_away"))

    rows = []
    for _, r in paired.iterrows():
        for side, opp_side in (("home", "away"), ("away", "home")):
            row = {
                "game_id": r["game_id"], "year": r[f"year_{side}"], "week": r[f"week_{side}"],
                "season_type": r[f"season_type_{side}"], "start_date": r[f"start_date_{side}"],
                "team": r[f"team_{side}"], "opponent": r[f"team_{opp_side}"],
                "is_home": side == "home",
                "points_for": r[f"points_{side}"], "points_against": r[f"points_{opp_side}"],
            }
            row["win"] = int(row["points_for"] > row["points_against"]) if pd.notna(row["points_for"]) else None
            for col in stat_cols:
                row[col] = r.get(f"{col}_{side}")
                row[f"opp_{col}"] = r.get(f"{col}_{opp_side}")
            rows.append(row)

    return pd.DataFrame(rows)


def add_derived_rate_stats(long_df: pd.DataFrame) -> pd.DataFrame:
    df = long_df.copy()
    plays = (df["rushing_attempts"].fillna(0) + df["pass_attempts"].fillna(0)).replace(0, pd.NA)
    df["yards_per_play"] = df["total_yards"] / plays
    df["third_down_pct"] = df["third_down_conversions"] / df["third_down_attempts"].replace(0, pd.NA)
    df["turnover_margin"] = df["opp_turnovers"].fillna(0) - df["turnovers"].fillna(0)
    df["point_diff"] = df["points_for"] - df["points_against"]
    return df


def compute_rolling_form(long_df: pd.DataFrame, srs_lookup: dict) -> pd.DataFrame:
    """srs_lookup: {(year, week, team): srs_rating} from opponent_adjustment.compute_weekly_srs."""
    df = long_df.sort_values(["team", "year", "start_date"]).copy()

    roll_cols = ["total_yards", "rushing_yards", "net_passing_yards", "yards_per_play",
                 "third_down_pct", "turnover_margin", "point_diff", "win"]
    roll_cols = [c for c in roll_cols if c in df.columns]

    for col in roll_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # SQLite round-trip can leave object/NA dtype

    for col in roll_cols:
        df[f"avg_{col}"] = (
            df.groupby("team")[col]
            .transform(lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=MIN_PERIODS).mean())
        )

    df["opponent_srs"] = df.apply(lambda r: srs_lookup.get((r["year"], r["week"], r["opponent"])), axis=1)
    df["avg_opponent_srs"] = (
        df.groupby("team")["opponent_srs"]
        .transform(lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=MIN_PERIODS).mean())
    )

    return df


def assemble_game_features(rolling_df: pd.DataFrame) -> pd.DataFrame:
    """Pivots the per-team rolling form back to one row per game with home_/away_/diff_ columns,
    matching the NBA model's feature-assembly pattern (diff_ features flagged as strongest there)."""
    feature_cols = [c for c in rolling_df.columns if c.startswith("avg_") or c == "avg_opponent_srs"]

    home = rolling_df[rolling_df["is_home"]][["game_id"] + feature_cols].set_index("game_id")
    away = rolling_df[~rolling_df["is_home"]][["game_id"] + feature_cols].set_index("game_id")

    home = home.add_prefix("home_")
    away = away.add_prefix("away_")
    joined = home.join(away, how="inner")

    for col in feature_cols:
        joined[f"diff_{col}"] = joined[f"home_{col}"] - joined[f"away_{col}"]

    return joined.reset_index()
