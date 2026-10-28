"""Evaluation helpers and fair reference policies."""
import numpy as np
from .env import reward_matrix


def dual_budget_actions(
    action_scores,
    predicted_costs,
    observed_costs,
    cost_scale,
    budget_per_1k,
    learning_rate,
    initial_lambda=0.0,
):
    """Choose actions with predicted costs and update the dual price post-action."""
    action_scores = np.asarray(action_scores, dtype=np.float64)
    predicted_costs = np.asarray(predicted_costs, dtype=np.float64)
    observed_costs = np.asarray(observed_costs, dtype=np.float64)
    if (
        action_scores.ndim != 2
        or predicted_costs.shape != action_scores.shape
        or observed_costs.shape != action_scores.shape
        or not len(action_scores)
    ):
        raise ValueError("Scores and cost matrices must be non-empty and have matching 2D shapes.")
    if (
        not np.all(np.isfinite(action_scores))
        or not np.all(np.isfinite(predicted_costs))
        or not np.all(np.isfinite(observed_costs))
        or np.any(predicted_costs < 0)
        or np.any(observed_costs < 0)
    ):
        raise ValueError("Scores and costs must be finite; costs must be non-negative.")
    if not np.isfinite(cost_scale) or cost_scale <= 0:
        raise ValueError("cost_scale must be finite and positive.")
    if not np.isfinite(budget_per_1k) or budget_per_1k <= 0:
        raise ValueError("budget_per_1k must be finite and positive.")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("dual learning rate must be finite and positive.")
    if not np.isfinite(initial_lambda) or initial_lambda < 0:
        raise ValueError("initial lambda must be finite and non-negative.")

    predicted_normalized_costs = predicted_costs / cost_scale
    budget_per_prompt_normalized = budget_per_1k / 1000.0 / cost_scale
    actions = np.empty(len(action_scores), dtype=int)
    lambdas = np.empty(len(action_scores), dtype=np.float64)
    dual_lambda = float(initial_lambda)
    for row in range(len(action_scores)):
        lambdas[row] = dual_lambda
        actions[row] = int(
            np.argmax(
                action_scores[row]
                - dual_lambda * predicted_normalized_costs[row]
            )
        )
        observed_normalized_cost = observed_costs[row, actions[row]] / cost_scale
        dual_lambda = max(
            0.0,
            dual_lambda
            + learning_rate
            * (observed_normalized_cost - budget_per_prompt_normalized),
        )
    return actions, lambdas, dual_lambda


def evaluate_actions(actions, data, idx, lam, cost_scale, rewards=None):
    actions, idx = np.asarray(actions, dtype=int), np.asarray(idx, dtype=int)
    if actions.ndim != 1 or len(actions) != len(idx) or not len(idx):
        raise ValueError("actions and idx must be non-empty 1D arrays of equal length")
    n_actions = data.quality.shape[1]
    if np.any(actions < 0) or np.any(actions >= n_actions):
        raise ValueError("Action index outside available model pool")
    q = data.quality[idx, actions]
    c = data.cost[idx, actions]
    if rewards is None:
        rewards = reward_matrix(data.quality, data.cost, lam, cost_scale)
    else:
        rewards = np.asarray(rewards, dtype=np.float64)
        if rewards.shape != data.quality.shape or not np.all(np.isfinite(rewards)):
            raise ValueError("Precomputed rewards must match the full dataset shape.")
    out = {
        "quality": float(q.mean()),
        "cost_per_1k_usd": float(c.mean() * 1000.0),
        "reward": float(rewards[idx, actions].mean()),
        "quality_std": float(q.std()),
        "cost_std_usd": float(c.std()),
    }
    for a, model in enumerate(data.models):
        safe = str(model).replace("/", "_").replace(" ", "_")
        out[f"frac_{safe}"] = float((actions == a).mean())
    out["small_frac"] = float((actions == 0).mean())
    out["medium_frac"] = float((actions == 1).mean()) if n_actions > 1 else 0.0
    out["large_frac"] = float((actions == n_actions - 1).mean())
    return out


def baseline_actions(data, idx, train_idx, lam, cost_scale, seed=0, rewards=None):
    """All baselines choose actions from prompt-only information, except oracle.

    Oracle is an explicitly labelled upper bound and uses outcomes only to
    choose the best action for each evaluated row; never use it as a feature.
    """
    idx, train_idx = np.asarray(idx, dtype=int), np.asarray(train_idx, dtype=int)
    n_actions, n = data.quality.shape[1], len(idx)
    if n_actions < 2:
        raise ValueError("At least two candidate models are required")
    rng = np.random.default_rng(seed)
    lengths = np.asarray([len(p) for p in data.prompts])
    thresholds = np.quantile(lengths[train_idx], [1/3, 2/3])
    length_policy = np.clip(np.digitize(lengths[idx], thresholds), 0, n_actions - 1)
    if rewards is None:
        rewards = reward_matrix(data.quality, data.cost, lam, cost_scale)
    else:
        rewards = np.asarray(rewards, dtype=np.float64)
        if rewards.shape != data.quality.shape or not np.all(np.isfinite(rewards)):
            raise ValueError("Precomputed rewards must match the full dataset shape.")
    split_rewards = rewards[idx]
    result = {
        "always_small": np.zeros(n, dtype=int),
        "always_medium": np.full(n, min(1, n_actions - 1), dtype=int),
        "always_large": np.full(n, n_actions - 1, dtype=int),
        "random": rng.integers(0, n_actions, size=n),
        "length_based": length_policy,
        "oracle": split_rewards.argmax(axis=1),
    }
    return result
