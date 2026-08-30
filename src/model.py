"""Prediction model: a 5-model stacked ensemble for win probability (same
architecture as the reference NBA model -- LogisticRegression, RandomForest,
XGBoost, LightGBM, ExtraTrees, blended by a logistic-regression meta-learner
trained on out-of-fold predictions), plus a separate XGBoost regressor for
predicted margin.

Two targets, not one, because CFB betting is spread-centric in a way the NBA
reference model (built for a win-probability market, Polymarket) wasn't:
- Classifier -> P(home win), for confidence tiers in the write-up.
- Regressor -> predicted point margin, compared directly against market_spread
  to compute an edge (this is the number that actually matters for spread bets).
  The NBA model's own docstring calls its margin projection a "display-only
  heuristic, not trained" -- we train ours properly instead since margin IS
  the primary decision signal for spread betting, not an afterthought.

market_spread is deliberately EXCLUDED from both models' training features,
same principle the reference model used for its own market data: training on
the market creates circular dependency (a model trained on the market can't
be evaluated as beating it). market_spread is only ever used post-hoc to
compute edge = predicted_margin - (-market_spread).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier

FEATURE_COLUMNS = [
    "elo_home", "elo_away", "elo_diff", "elo_expected_home",
    "srs_home", "srs_away", "srs_diff",
    "home_ats_pct", "away_ats_pct",
    "home_rest_days", "away_rest_days", "home_bye_week", "away_bye_week",
    "h2h_home_win_pct", "h2h_avg_home_margin", "h2h_meetings",
    "home_avg_total_yards", "away_avg_total_yards", "diff_avg_total_yards",
    "home_avg_rushing_yards", "away_avg_rushing_yards", "diff_avg_rushing_yards",
    "home_avg_net_passing_yards", "away_avg_net_passing_yards", "diff_avg_net_passing_yards",
    "home_avg_yards_per_play", "away_avg_yards_per_play", "diff_avg_yards_per_play",
    "home_avg_third_down_pct", "away_avg_third_down_pct", "diff_avg_third_down_pct",
    "home_avg_turnover_margin", "away_avg_turnover_margin", "diff_avg_turnover_margin",
    "home_avg_point_diff", "away_avg_point_diff", "diff_avg_point_diff",
    "home_avg_win", "away_avg_win", "diff_avg_win",
    "home_avg_opponent_srs", "away_avg_opponent_srs", "diff_avg_opponent_srs",
    "is_adverse_weather", "adverse_wx_ats_edge",
]

N_SPLITS = 3  # TimeSeriesSplit folds for OOF stacking -- kept small since each CFB season is short


def prepare_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_COLUMNS].copy()
    for col in ("home_bye_week", "away_bye_week", "is_adverse_weather"):
        X[col] = X[col].astype(float)
    medians = X.median()
    X = X.fillna(medians)
    return X, medians


def _base_classifiers() -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, C=0.1),
        "random_forest": RandomForestClassifier(random_state=42, max_depth=8,
                                                  min_samples_leaf=10, n_estimators=200),
        "xgboost": XGBClassifier(random_state=42, eval_metric="logloss", subsample=0.8,
                                  colsample_bytree=0.8, learning_rate=0.05, max_depth=4, n_estimators=200),
        "lightgbm": LGBMClassifier(random_state=42, verbose=-1, subsample=0.8, colsample_bytree=0.8,
                                    n_estimators=200, num_leaves=15, learning_rate=0.05, min_child_samples=10),
        "extra_trees": ExtraTreesClassifier(random_state=42, n_jobs=-1, max_depth=12,
                                             min_samples_leaf=5, n_estimators=400),
    }


@dataclass
class StackingEnsemble:
    base_models: dict = field(default_factory=dict)
    meta_model: LogisticRegression = None
    feature_medians: pd.Series = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.base_models = {}
        oof_preds = np.zeros((len(X), len(_base_classifiers())))
        tscv = TimeSeriesSplit(n_splits=N_SPLITS)

        for i, (name, model) in enumerate(_base_classifiers().items()):
            pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
            fold_preds = np.full(len(X), np.nan)
            for train_idx, val_idx in tscv.split(X):
                pipeline_fold = Pipeline([("scaler", StandardScaler()), ("model", model.__class__(**model.get_params()))])
                pipeline_fold.fit(X.iloc[train_idx], y.iloc[train_idx])
                fold_preds[val_idx] = pipeline_fold.predict_proba(X.iloc[val_idx])[:, 1]
            oof_preds[:, i] = fold_preds

            pipeline.fit(X, y)  # refit on full training set for inference-time use
            self.base_models[name] = pipeline

        valid_rows = ~np.isnan(oof_preds).any(axis=1)
        self.meta_model = LogisticRegression(C=0.1, max_iter=1000)
        self.meta_model.fit(oof_preds[valid_rows], y.iloc[valid_rows])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        base_preds = np.column_stack([m.predict_proba(X)[:, 1] for m in self.base_models.values()])
        return self.meta_model.predict_proba(base_preds)[:, 1]

    def predict_proba_detailed(self, X: pd.DataFrame) -> dict:
        """Same as predict_proba, but also returns each base model's own win probability
        before blending -- for showing users what each of the 5 models individually predicted,
        not just the final ensemble output."""
        base_probs = {name: m.predict_proba(X)[:, 1] for name, m in self.base_models.items()}
        base_matrix = np.column_stack(list(base_probs.values()))
        final = self.meta_model.predict_proba(base_matrix)[:, 1]
        return {"base": base_probs, "final": final}


def train_margin_regressor(X: pd.DataFrame, y: pd.Series) -> XGBRegressor:
    model = XGBRegressor(random_state=42, n_estimators=300, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8)
    model.fit(X, y)
    return model


def evaluate_classifier(ensemble: StackingEnsemble, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    proba = ensemble.predict_proba(X_test)
    preds = (proba > 0.5).astype(int)
    return {
        "accuracy": (preds == y_test.values).mean(),
        "log_loss": log_loss(y_test, proba),
        "brier_score": brier_score_loss(y_test, proba),
        "auc": roc_auc_score(y_test, proba),
    }


def evaluate_regressor(model: XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    preds = model.predict(X_test)
    return {
        "mae": mean_absolute_error(y_test, preds),
        "rmse": float(np.sqrt(np.mean((preds - y_test.values) ** 2))),
    }


def get_calibration_curve(ensemble: StackingEnsemble, X_test: pd.DataFrame, y_test: pd.Series, n_bins: int = 8):
    proba = ensemble.predict_proba(X_test)
    return calibration_curve(y_test, proba, n_bins=n_bins, strategy="quantile")
