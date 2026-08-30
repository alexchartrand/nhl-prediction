"""Baseline forward/defense point-prediction models.

ElasticNet (regularized linear) vs LightGBM (gradient boosting), trained
separately per position group on year-N stats -> year-N+1 fantasy points,
per CLAUDE.md's modeling approach.

Validation: the most recent season transition (2024-25 features -> the
already-known 2025-26 results) is held out as the test set rather than
included in any fold, matching CLAUDE.md's "train on earlier years, test on
most recent" guidance. Within the remaining training rows, CV is grouped by
player_id (GroupKFold) so a player's rows from different season-pairs never
span both sides of a fold -- a plain random split would let the model see
a near-duplicate of a validation player's skill level via his other years.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loading
import scoring
from features import FEATURE_COLS, MIN_GP, load_scored_seasons

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
HOLDOUT_TARGET_SEASON = "2025_2026"
N_SPLITS = 5


def build_dataset() -> pd.DataFrame:
    df = load_scored_seasons()
    return loading.make_training_pairs(
        df, feature_cols=FEATURE_COLS + ["Player", "pos_group"], min_feature_gp=MIN_GP
    )


def make_elasticnet() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
            cv=5,
            max_iter=10000,
            alphas=50,
        )),
    ])


def make_lightgbm() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=0,
        verbose=-1,
    )


def cv_predictions(make_model, X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int = N_SPLITS) -> np.ndarray:
    """Out-of-fold predictions from a fresh model per fold, grouped by player."""
    preds = np.full(len(y), np.nan)
    for train_idx, val_idx in GroupKFold(n_splits=n_splits).split(X, y, groups):
        model = make_model()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds[val_idx] = model.predict(X.iloc[val_idx])
    return preds


def eval_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "Spearman": spearmanr(y_true, y_pred).correlation,
    }


def train_position(pairs: pd.DataFrame, pos_group: str) -> dict:
    sub = pairs[pairs["pos_group"] == pos_group].reset_index(drop=True)
    train = sub[sub["target_season"] != HOLDOUT_TARGET_SEASON].reset_index(drop=True)
    test = sub[sub["target_season"] == HOLDOUT_TARGET_SEASON].reset_index(drop=True)

    X_train, y_train = train[FEATURE_COLS], train[scoring.TARGET]
    X_test, y_test = test[FEATURE_COLS], test[scoring.TARGET]
    groups = train["player_id"]

    en_cv_pred = cv_predictions(make_elasticnet, X_train, y_train, groups)
    gbm_cv_pred = cv_predictions(make_lightgbm, X_train, y_train, groups)

    en = make_elasticnet().fit(X_train, y_train)
    gbm = make_lightgbm().fit(X_train, y_train)
    en_test_pred = en.predict(X_test)
    gbm_test_pred = gbm.predict(X_test)

    n_players = groups.nunique()
    print(f"\n=== {pos_group} === train={len(train)} rows / {n_players} players, holdout={len(test)} rows")
    print(f"{'model':<12}{'split':<12}{'MAE':>8}{'R2':>8}{'Spearman':>10}")
    for name, y_cv_pred, y_te_pred in (
        ("ElasticNet", en_cv_pred, en_test_pred),
        ("LightGBM", gbm_cv_pred, gbm_test_pred),
    ):
        cv_m = eval_metrics(y_train, y_cv_pred)
        te_m = eval_metrics(y_test, y_te_pred)
        print(f"{name:<12}{'CV (train)':<12}{cv_m['MAE']:>8.2f}{cv_m['R2']:>8.3f}{cv_m['Spearman']:>10.3f}")
        print(f"{name:<12}{'holdout':<12}{te_m['MAE']:>8.2f}{te_m['R2']:>8.3f}{te_m['Spearman']:>10.3f}")

    en_coefs = (
        pd.Series(en.named_steps["model"].coef_, index=FEATURE_COLS)
        .sort_values(key=abs, ascending=False)
    )
    print(f"\n{pos_group} ElasticNet top coefficients (standardized):")
    print(en_coefs.head(8).round(2).to_string())

    gbm_imp = pd.Series(gbm.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print(f"\n{pos_group} LightGBM top feature importances:")
    print(gbm_imp.head(8).to_string())

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(en, MODELS_DIR / f"{pos_group.lower()}_elasticnet.joblib")
    joblib.dump(gbm, MODELS_DIR / f"{pos_group.lower()}_lightgbm.joblib")

    return {"elasticnet": en, "lightgbm": gbm}


if __name__ == "__main__":
    pairs = build_dataset()
    for pos in ("F", "D"):
        train_position(pairs, pos)
