"""Model factories, imbalance handling and Optuna search spaces (Stage 4).
xgboost / lightgbm are imported lazily so the rest of the pipeline works without them."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.features import FEATURES

SEED = 42
STRATEGIES = ["none", "class_weight", "undersample", "oversample"]
# non-negative, heavy-tailed columns get log1p for the linear model (trend features are -1..1, is_uk is 0/1)
LOG_COLS = [c for c in FEATURES if c not in ("order_trend", "spend_trend", "is_uk")]

DEFAULTS = {
    "lr": {"C": 1.0},
    "xgb": dict(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
                colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0),
    "lgbm": dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=40,
                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0),
}


def resample(X: pd.DataFrame, y, strategy: str, seed: int = SEED):
    """Random under/over-sampling of TRAIN only. Other strategies pass through unchanged."""
    y = np.asarray(y)
    if strategy not in ("undersample", "oversample"):
        return X, y
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    minority, majority = (pos, neg) if len(pos) < len(neg) else (neg, pos)
    if strategy == "undersample":
        idx = np.concatenate([minority, rng.choice(majority, len(minority), replace=False)])
    else:
        extra = rng.choice(minority, len(majority) - len(minority), replace=True)
        idx = np.concatenate([majority, minority, extra])
    rng.shuffle(idx)
    return X.iloc[idx], y[idx]


def make_lr(params, class_weight, y):
    pre = ColumnTransformer(
        [("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), LOG_COLS)],
        remainder="passthrough")
    return Pipeline([
        ("log", pre),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),   # single-order customers have NaN gaps
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=params["C"], max_iter=2000,
                                   class_weight="balanced" if class_weight else None)),
    ])


def make_xgb(params, class_weight, y):
    from xgboost import XGBClassifier
    y = np.asarray(y)
    spw = float((y == 0).sum() / max((y == 1).sum(), 1)) if class_weight else 1.0
    return XGBClassifier(**params, scale_pos_weight=spw, tree_method="hist", eval_metric="logloss",
                         random_state=SEED, n_jobs=-1)


def make_lgbm(params, class_weight, y):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(**params, subsample_freq=1, class_weight="balanced" if class_weight else None,
                          random_state=SEED, verbose=-1)


FACTORY = {"lr": make_lr, "xgb": make_xgb, "lgbm": make_lgbm}


def fit_model(name: str, params: dict, strategy: str, X: pd.DataFrame, y):
    Xr, yr = resample(X, y, strategy)
    model = FACTORY[name](params, strategy == "class_weight", yr)
    model.fit(Xr, yr)
    return model


def space_xgb(t):
    return dict(n_estimators=t.suggest_int("n_estimators", 100, 600),
                learning_rate=t.suggest_float("learning_rate", 0.02, 0.2, log=True),
                max_depth=t.suggest_int("max_depth", 2, 7),
                min_child_weight=t.suggest_int("min_child_weight", 1, 30),
                subsample=t.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
                reg_lambda=t.suggest_float("reg_lambda", 1e-2, 20.0, log=True))


def space_lgbm(t):
    return dict(n_estimators=t.suggest_int("n_estimators", 100, 600),
                learning_rate=t.suggest_float("learning_rate", 0.02, 0.2, log=True),
                num_leaves=t.suggest_int("num_leaves", 4, 48),
                min_child_samples=t.suggest_int("min_child_samples", 10, 150),
                subsample=t.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
                reg_lambda=t.suggest_float("reg_lambda", 1e-2, 20.0, log=True))


SPACES = {"xgb": space_xgb, "lgbm": space_lgbm}
