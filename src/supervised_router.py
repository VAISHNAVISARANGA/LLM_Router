"""Supervised contextual router trained on RouterBench reward labels.

Outcomes are labels only: model inputs are prompt features, never per-model
quality/cost/reward columns. Fit on train; use validation for model/lambda
selection; evaluate the held-out test split once at the end.
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import ExtraTreesRegressor
from .env import reward_matrix


class ModelCostPredictor:
    """Predict per-model costs using training prompts and their observed costs."""

    def __init__(self, random_state=42):
        self.model = ExtraTreesRegressor(
            n_estimators=250,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X, costs):
        X = np.asarray(X, dtype=np.float32)
        costs = np.asarray(costs, dtype=np.float32)
        if (
            X.ndim != 2
            or costs.ndim != 2
            or len(X) != len(costs)
            or not len(X)
        ):
            raise ValueError("X and costs must be non-empty aligned 2D arrays.")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(costs)) or np.any(costs < 0):
            raise ValueError("X and costs must be finite; costs must be non-negative.")
        self.model.fit(X, costs)
        return self

    def predict(self, X):
        if not hasattr(self.model, "estimators_"):
            raise RuntimeError("Call fit before predicting costs.")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or not np.all(np.isfinite(X)):
            raise ValueError("X must be a finite 2D array.")
        return np.maximum(self.model.predict(X), 0.0)


def quality_gain_targets(quality, cost, cost_scale):
    """Adjacent upgrade gains for each same prompt: quality and normalized cost."""
    quality, cost = np.asarray(quality, float), np.asarray(cost, float)
    if quality.shape != cost.shape or quality.ndim != 2 or quality.shape[1] < 2:
        raise ValueError("quality and cost must align and contain >=2 models")
    q_gain = quality[:, 1:] - quality[:, :-1]
    c_gain = cost[:, 1:] - cost[:, :-1]
    c_gain_norm = c_gain / float(cost_scale)
    return q_gain, c_gain, c_gain_norm


def worthwhile_upgrade(q_gain, c_gain_norm, lam):
    """Upgrade only where the extra quality exceeds its lambda-weighted cost."""
    return np.asarray(q_gain) - float(lam) * np.asarray(c_gain_norm) > 0


class SupervisedRewardRouter:
    """Predict each candidate model's reward; choose the highest predicted reward."""
    def __init__(self, estimator="extra_trees", n_estimators=250, min_samples_leaf=2,
                 max_features=1.0, random_state=42):
        self.estimator_name = estimator
        self.params = dict(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                           max_features=max_features, random_state=random_state,
                           n_jobs=-1)
        self.model = None
        self.n_actions = None

    def _new_model(self):
        cls = ExtraTreesRegressor if self.estimator_name == "extra_trees" else RandomForestRegressor
        return MultiOutputRegressor(cls(**self.params), n_jobs=1)

    def fit(self, X, reward_labels):
        X, y = np.asarray(X, dtype=np.float32), np.asarray(reward_labels, dtype=np.float32)
        if X.ndim != 2 or y.ndim != 2 or len(X) != len(y):
            raise ValueError("X and reward labels must be aligned 2D arrays")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("X and reward labels must be finite")
        self.n_actions = y.shape[1]
        self.model = self._new_model().fit(X, y)
        return self

    def predict_rewards(self, X):
        if self.model is None:
            raise RuntimeError("Call fit before prediction")
        return np.asarray(self.model.predict(np.asarray(X, dtype=np.float32)))

    def predict(self, X):
        return self.predict_rewards(X).argmax(axis=1)


def tune_lambda_and_router(X_train, quality_train, cost_train, X_val, quality_val, cost_val,
                           cost_scale, lambdas, estimator_configs=None,
                           difficulty_train=None, difficulty_val=None):
    """Choose lambda and estimator settings using validation reward only."""
    estimator_configs = estimator_configs or [
        {"estimator": "extra_trees", "n_estimators": 150, "min_samples_leaf": 2},
        {"estimator": "extra_trees", "n_estimators": 300, "min_samples_leaf": 5},
        {"estimator": "random_forest", "n_estimators": 200, "min_samples_leaf": 2},
    ]
    best, records = None, []
    for lam in lambdas:
        train_rewards = reward_matrix(
            quality_train,
            cost_train,
            lam,
            cost_scale,
            difficulty=difficulty_train,
        )
        val_rewards = reward_matrix(
            quality_val,
            cost_val,
            lam,
            cost_scale,
            difficulty=difficulty_val,
        )
        for cfg in estimator_configs:
            router = SupervisedRewardRouter(**cfg).fit(X_train, train_rewards)
            actions = router.predict(X_val)
            score = float(val_rewards[np.arange(len(actions)), actions].mean())
            row = {"lambda": float(lam), "validation_reward": score, **cfg}
            records.append(row)
            if best is None or score > best["validation_reward"]:
                best = {**row, "router": router}
    return best, records
