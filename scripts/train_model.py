"""Trains the win-probability ensemble and margin regressor on game_features,
evaluates on a held-out season, and saves the model bundle.

Split: train on everything before HOLDOUT_YEAR (2011-2024 as of the 2026-09-03
historical backfill; originally just 2021-2024), hold out all of 2025 as the
test set. A full-season holdout (not a row-count 80/20 split) avoids cutting a
season in half and keeps the evaluation honest -- the model never sees any
2025 result during training, mirroring how it would actually be used (predict
a season it hasn't seen yet). Any completed 2026 rows already in game_features
are excluded from both train and test on purpose -- the current season is
still in progress and tracked separately via the live predictions/reconcile
pipeline, not folded into this holdout-eval cycle.

Pre-2021 (pre-NIL) rows are down-weighted relative to 2021+ during training
(see src.model.compute_nil_era_sample_weight) -- more historical games to
learn from, without letting a roster-construction era the sport has moved on
from carry equal say to today's NIL/transfer-portal reality.
"""
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import get_connection
from src.model import (FEATURE_COLUMNS, StackingEnsemble, compute_nil_era_sample_weight,
                        evaluate_classifier, evaluate_regressor, get_calibration_curve,
                        prepare_matrix, train_margin_regressor)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
HOLDOUT_YEAR = 2025


def main():
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM game_features ORDER BY year, week", conn)
    conn.close()

    train_df = df[df["year"] < HOLDOUT_YEAR].reset_index(drop=True)
    test_df = df[df["year"] == HOLDOUT_YEAR].reset_index(drop=True)
    print(f"Train: {len(train_df)} games ({train_df['year'].min()}-{HOLDOUT_YEAR - 1}). "
          f"Test: {len(test_df)} games ({HOLDOUT_YEAR}).")

    sample_weight = compute_nil_era_sample_weight(train_df["year"])
    n_pre_nil = int((train_df["year"] < 2021).sum())
    print(f"Sample weighting: {n_pre_nil} pre-NIL rows (<2021) at weight 0.5, "
          f"{len(train_df) - n_pre_nil} NIL-era rows (2021+) at weight 1.0.")

    X_train, medians = prepare_matrix(train_df)
    X_test = test_df[FEATURE_COLUMNS].copy()
    for col in ("home_bye_week", "away_bye_week", "is_adverse_weather"):
        X_test[col] = X_test[col].astype(float)
    X_test = X_test.fillna(medians)

    y_train_class, y_test_class = train_df["home_win"], test_df["home_win"]
    y_train_margin, y_test_margin = train_df["home_margin"], test_df["home_margin"]

    print("\nTraining 5-model stacking ensemble (win probability)...")
    ensemble = StackingEnsemble()
    ensemble.fit(X_train, y_train_class, sample_weight=sample_weight)
    ensemble.feature_medians = medians

    print("Training margin regressor...")
    regressor = train_margin_regressor(X_train, y_train_margin, sample_weight=sample_weight)

    print("\n=== Classifier evaluation (2025 holdout) ===")
    class_metrics = evaluate_classifier(ensemble, X_test, y_test_class)
    for k, v in class_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\n=== Regressor evaluation (2025 holdout) ===")
    reg_metrics = evaluate_regressor(regressor, X_test, y_test_margin)
    for k, v in reg_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\n=== Calibration curve (predicted vs actual win %, 8 bins) ===")
    prob_true, prob_pred = get_calibration_curve(ensemble, X_test, y_test_class)
    for pt, pp in zip(prob_true, prob_pred):
        print(f"  predicted {pp:.3f} -> actual {pt:.3f}")

    print("\n=== Naive baseline comparison: always predict home team wins ===")
    naive_acc = (y_test_class == 1).mean()
    print(f"  home-team-always-wins accuracy: {naive_acc:.4f} (our model: {class_metrics['accuracy']:.4f})")

    MODEL_DIR.mkdir(exist_ok=True)
    bundle = {
        "ensemble": ensemble, "regressor": regressor, "feature_columns": FEATURE_COLUMNS,
        "feature_medians": medians, "train_years": (int(train_df["year"].min()), HOLDOUT_YEAR - 1),
        "holdout_year": HOLDOUT_YEAR, "classifier_metrics": class_metrics, "regressor_metrics": reg_metrics,
        "nil_era_sample_weighting": True,
    }
    joblib.dump(bundle, MODEL_DIR / "cfb_model.pkl", compress=3)
    print(f"\nSaved model bundle to {MODEL_DIR / 'cfb_model.pkl'}")


if __name__ == "__main__":
    main()
