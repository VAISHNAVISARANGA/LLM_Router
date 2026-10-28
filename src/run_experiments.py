"""Step 2: train every agent, evaluate on the test split, make tables + figures.

    python -m src.run_experiments                 # full run
    python -m src.run_experiments --quick         # tiny run to check everything works
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from .features import (
    EmbeddingFeaturizer,
    HandcraftedFeaturizer,
    HybridFeaturizer,
    embed_texts,
)
from .difficulty import DifficultyEstimator
from . import plots
from .config import SEED
from .data import load_data
from .env import reward_matrix
from .evaluate import baseline_actions, dual_budget_actions, evaluate_actions
from .train import action_scores, train_agent
from .supervised_router import ModelCostPredictor
CONFIGS = [
    ("q", "handcrafted"),
    ("q", "embedding"),
    ("q", "hybrid"),
    ("dqn", "handcrafted"),
    ("dqn", "embedding"),
    ("dqn", "hybrid"),
]
DEFAULT_DUAL_LEARNING_RATES = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]

def available_configs(has_embeddings=True):
    configs = CONFIGS if has_embeddings else [
        key for key in CONFIGS if key[1] == "handcrafted"
    ]
    try:
        import torch  # noqa: F401
        return configs
    except Exception:
        print("!! PyTorch not usable -> skipping DQN configurations (pip install torch)")
        return [c for c in configs if c[0] == "q"]


def cfg_name(key):
    return plots.LABELS[key]


def _make_features(featurizer, prompts, embeddings=None):
    discrete, continuous = featurizer.transform(prompts, emb=embeddings)
    return {
        "disc": discrete,
        "cont": continuous,
        "n_states": featurizer.n_states,
        "dim": continuous.shape[1],
    }


def display_model_names(models):
    labels = []
    for index, model in enumerate(models):
        if index == 0:
            env_name = "SMALL_MODEL_NAME"
        elif index == len(models) - 1:
            env_name = "LARGE_MODEL_NAME"
        else:
            env_name = "MEDIUM_MODEL_NAME" if len(models) == 3 else f"MEDIUM_{index}_MODEL_NAME"
        labels.append(os.getenv(env_name) or model)
    return labels


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed.joblib")
    ap.add_argument(
        "--budget-per-1k",
        type=float,
        default=0.50,
        help="Hard average cost ceiling in USD per 1,000 prompts (default: 0.50).",
    )
    ap.add_argument(
        "--dual-learning-rates",
        type=float,
        nargs="+",
        default=DEFAULT_DUAL_LEARNING_RATES,
        help="Step sizes evaluated on validation data for the dual-price update.",
    )
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--episode-len", type=int, default=100)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--n-clusters", type=int, default=32)
    ap.add_argument("--out", default="results")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--quick", action="store_true", help="1 seed and 100 training episodes")
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.episodes = 1, 100
    if not np.isfinite(args.budget_per_1k) or args.budget_per_1k <= 0:
        ap.error("--budget-per-1k must be finite and positive")
    if not args.dual_learning_rates or any(
        not np.isfinite(rate) or rate <= 0
        for rate in args.dual_learning_rates
    ):
        ap.error("--dual-learning-rates must contain finite, positive values")
    fig_dir = os.path.join(args.out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)
    
# ---------------- data + features ----------------
    try:
        data = load_data(args.data, require_embeddings=False)
    except (FileNotFoundError, ValueError) as exc:
        ap.error(str(exc))
    tr, va, te = data.idx("train"), data.idx("val"), data.idx("test")
    print(f"Loaded {data.source} data: train={len(tr)} val={len(va)} test={len(te)}")

    global CONFIGS
    CONFIGS = available_configs(has_embeddings=data.embeddings is not None)
    if data.embeddings is None:
        print("No embeddings found; running handcrafted configurations only.")

    # Fit prompt-only state representations using training prompts.
    hand = HandcraftedFeaturizer().fit([data.prompts[i] for i in tr])
    featurizers = {"handcrafted": hand}
    if data.embeddings is not None:
        emb = EmbeddingFeaturizer(
            data.embedder, args.n_clusters, SEED
        ).fit(emb=data.embeddings[tr])
        hybrid = HybridFeaturizer(
            embedder=data.embedder,
            n_clusters=args.n_clusters,
            seed=SEED,
        ).fit(
            prompts=[data.prompts[i] for i in tr],
            emb=data.embeddings[tr],
        )
        featurizers.update({"embedding": emb, "hybrid": hybrid})
    feats = {
        name: _make_features(
            featurizer,
            data.prompts,
            data.embeddings,
        )
        for name, featurizer in featurizers.items()
    }

    # Keep one reward matrix for training, calibration, evaluation, and baselines.
    cost_scale = float(np.quantile(data.cost[tr, -1], 0.99))
    A = data.quality.shape[1]

    print(f"Cost normalization scale: ${cost_scale:.6f}")

    difficulty_embedder = data.embedder or "hash"
    difficulty_estimator = DifficultyEstimator(embedder=difficulty_embedder).fit(
        [data.prompts[i] for i in tr],
        data.quality[tr],
        embeddings=data.embeddings[tr] if data.embeddings is not None else None,
    )
    difficulty = difficulty_estimator.predict(
        data.prompts,
        embeddings=(
            data.embeddings
            if data.embeddings is not None
            else embed_texts(data.prompts, difficulty_embedder)
        ),
    )
    print(
        f"Estimated difficulty range: "
        f"{difficulty.min():.2f} to {difficulty.max():.2f}"
    )
    R = reward_matrix(data.quality, data.cost, 0.0, cost_scale, difficulty=difficulty)
    cost_predictors = {}
    for name, feature_data in feats.items():
        cost_predictors[name] = ModelCostPredictor(random_state=SEED).fit(
            feature_data["cont"][tr],
            data.cost[tr],
        )
    histories = {k: [] for k in CONFIGS}
    results = {k: [] for k in CONFIGS}
    best = {}
    candidates = []
    for key in CONFIGS:
        kind, fname = key
        for seed in range(args.seeds):
            print(f"[training] {cfg_name(key)} seed={seed} ...", flush=True)
            agent, hist = train_agent(
                kind,
                feats[fname],
                R,
                tr,
                seed,
                args.episodes,
                args.episode_len,
                args.gamma,
            )
            validation_scores = action_scores(agent, feats[fname], va)
            validation_predicted_costs = cost_predictors[fname].predict(
                feats[fname]["cont"][va]
            )
            for dual_learning_rate in args.dual_learning_rates:
                (
                    validation_actions,
                    validation_lambdas,
                    validation_final_lambda,
                ) = dual_budget_actions(
                    validation_scores,
                    validation_predicted_costs,
                    data.cost[va],
                    cost_scale,
                    args.budget_per_1k,
                    dual_learning_rate,
                )
                validation_metrics = evaluate_actions(
                    validation_actions,
                    data,
                    va,
                    0.0,
                    cost_scale,
                    rewards=R,
                )
                validation_cost = validation_metrics["cost_per_1k_usd"]
                candidates.append({
                    "config": key,
                    "agent": agent,
                    "seed": seed,
                    "history": hist,
                    "dual_learning_rate": dual_learning_rate,
                    "validation_reward": validation_metrics["reward"],
                    "validation_quality": validation_metrics["quality"],
                    "validation_cost_per_1k": validation_cost,
                    "validation_final_lambda": validation_final_lambda,
                    "validation_mean_lambda": float(validation_lambdas.mean()),
                    "feasible": validation_cost <= args.budget_per_1k,
                })

    feasible_candidates = [candidate for candidate in candidates if candidate["feasible"]]
    selection_pool = feasible_candidates if feasible_candidates else candidates
    if feasible_candidates:
        selected_candidate = max(
            selection_pool,
            key=lambda row: (
                row["validation_quality"],
                -row["validation_cost_per_1k"],
            ),
        )
    else:
        selected_candidate = min(
            selection_pool,
            key=lambda row: (
                row["validation_cost_per_1k"],
                -row["validation_quality"],
            ),
        )
    best_key = selected_candidate["config"]
    best = {}
    sweep = {key: [] for key in CONFIGS}
    sweep_rows = []
    agent_calibrations = {}
    for key in CONFIGS:
        config_candidates = [row for row in candidates if row["config"] == key]
        feasible_for_config = [row for row in config_candidates if row["feasible"]]
        pool = feasible_for_config if feasible_for_config else config_candidates
        chosen = (
            max(pool, key=lambda row: (row["validation_quality"], -row["validation_cost_per_1k"]))
            if feasible_for_config
            else min(pool, key=lambda row: (row["validation_cost_per_1k"], -row["validation_quality"]))
        )
        best[key] = (
            chosen["validation_quality"],
            chosen["agent"],
            chosen["seed"],
            chosen["history"],
            chosen["validation_reward"],
            chosen["validation_final_lambda"],
        )
        agent_calibrations[f"{key[0]}_{key[1]}"] = {
            "seed": chosen["seed"],
            "validation_quality": chosen["validation_quality"],
            "validation_final_dual_lambda": chosen["validation_final_lambda"],
            "dual_learning_rate": chosen["dual_learning_rate"],
        }
        histories[key].append(chosen["history"])
        test_scores = action_scores(chosen["agent"], feats[key[1]], te)
        test_predicted_costs = cost_predictors[key[1]].predict(
            feats[key[1]]["cont"][te]
        )
        test_actions, test_lambdas, test_final_lambda = dual_budget_actions(
            test_scores,
            test_predicted_costs,
            data.cost[te],
            cost_scale,
            args.budget_per_1k,
            chosen["dual_learning_rate"],
        )
        test_metrics = evaluate_actions(
            test_actions,
            data,
            te,
            0.0,
            cost_scale,
            rewards=R,
        )
        test_metrics.update({
            "dual_final_lambda": test_final_lambda,
            "dual_mean_lambda": float(test_lambdas.mean()),
            "validation_reward": chosen["validation_reward"],
            "validation_quality": chosen["validation_quality"],
            "validation_cost_per_1k_usd": chosen["validation_cost_per_1k"],
            "validation_final_lambda": chosen["validation_final_lambda"],
            "dual_learning_rate": chosen["dual_learning_rate"],
            "within_budget": chosen["feasible"],
        })
        results[key].append(test_metrics)
        sweep[key].append((test_metrics["cost_per_1k_usd"], test_metrics["quality"]))
        sweep_rows.append({
            "method": cfg_name(key),
            "seed": chosen["seed"],
            "dual_learning_rate": chosen["dual_learning_rate"],
            "selected_for_deployment": chosen is selected_candidate,
            "within_budget": chosen["feasible"],
            "validation_quality": chosen["validation_quality"],
            "validation_cost_per_1k_usd": chosen["validation_cost_per_1k"],
            "validation_final_dual_lambda": chosen["validation_final_lambda"],
            "test_quality": test_metrics["quality"],
            "test_cost_per_1k_usd": test_metrics["cost_per_1k_usd"],
            "test_reward": test_metrics["reward"],
            "test_final_dual_lambda": test_final_lambda,
        })

    pd.DataFrame([
        {
            "method": cfg_name(row["config"]),
            "seed": row["seed"],
            "validation_quality": row["validation_quality"],
            "validation_cost_per_1k_usd": row["validation_cost_per_1k"],
            "within_budget": row["feasible"],
            "selected": row is selected_candidate,
            "validation_mean_dual_lambda": row["validation_mean_lambda"],
            "validation_final_dual_lambda": row["validation_final_lambda"],
            "dual_learning_rate": row["dual_learning_rate"],
        }
        for row in candidates
    ]).to_csv(os.path.join(args.out, "budget_selection_candidates.csv"), index=False)
    if not feasible_candidates:
        print(
            f"WARNING: no validation candidate met the ${args.budget_per_1k:.4f}/1k "
            "budget; selecting the lowest-cost candidate."
        )
    print(
        f"Selected by validation constraint: {cfg_name(best_key)}, "
        f"quality={selected_candidate['validation_quality']:.4f}, "
        f"cost=${selected_candidate['validation_cost_per_1k']:.4f}/1k "
        f"(budget=${args.budget_per_1k:.4f}/1k)"
    )
    base_acts = baseline_actions(
        data,
        te,
        tr,
        0.0,
        cost_scale,
        SEED,
        rewards=R,
    )

    base_res = {
        name: evaluate_actions(actions, data, te, 0.0, cost_scale, rewards=R)
        for name, actions in base_acts.items()
    }

    rows = []
    large = base_res["always_large"]
    for name, r in base_res.items():
        rows.append({
        "method": name,
        **r,
        "reward_std": np.nan,
        "quality_std": np.nan,
        "cost_std": np.nan,
    })
    for key, rs in results.items():
        metric_keys = list(rs[0])
        m = {name: float(np.mean([r[name] for r in rs])) for name in metric_keys}
        rows.append({"method": cfg_name(key), **m,
                     "validation_reward_mean": float(np.mean([r["validation_reward"] for r in rs])),
                     "validation_final_dual_lambda": float(rs[0]["validation_final_lambda"]),
                     "validation_quality_mean": float(np.mean([r["validation_quality"] for r in rs])),
                     "validation_cost_per_1k_usd_mean": float(np.mean([r["validation_cost_per_1k_usd"] for r in rs])),
                     "within_budget": bool(rs[0]["within_budget"]),
                     "reward_std": float(np.std([r["reward"] for r in rs])),
                     "quality_std": float(np.std([r["quality"] for r in rs])),
                     "cost_std": float(np.std([r["cost_per_1k_usd"] for r in rs]))})
    table = pd.DataFrame(rows)
    table["cost_saving_vs_large_%"] = 100 * (1 - table["cost_per_1k_usd"] / large["cost_per_1k_usd"])
    table["quality_retained_%"] = 100 * table["quality"] / large["quality"]
    table.to_csv(os.path.join(args.out, "metrics_main.csv"), index=False)
    print(
        "\n=== Test-set results (dynamic dual-price controller, "
        f"{args.seeds} seeds) ==="
    )
    print(table.round(4).to_string(index=False))

    # Report the selected checkpoint for each configuration.
    pd.DataFrame(sweep_rows).to_csv(os.path.join(args.out, "metrics_sweep.csv"), index=False)

    # ---------------- figures ----------------
    plots.training_curves({k: np.stack(v) for k, v in histories.items()}, fig_dir)
    plots.pareto(sweep, {n: (r["cost_per_1k_usd"], r["quality"]) for n, r in base_res.items()}, fig_dir)
    bars = [(n, r["reward"], 0.0) for n, r in base_res.items() if n != "random"]
    bars += [(cfg_name(k), float(np.mean([r["reward"] for r in rs])),
              float(np.std([r["reward"] for r in rs]))) for k, rs in results.items()]
    plots.reward_bars(bars, fig_dir)

    
    # Compare the validation-selected agent with baselines on the held-out test split.

    (
        best_val_quality,
        best_agent,
        best_seed,
        _,
        best_val_reward,
        validation_final_lambda,
    ) = best[best_key]
    best_dual_learning_rate = agent_calibrations[
        f"{best_key[0]}_{best_key[1]}"
    ]["dual_learning_rate"]

    kind, feature_name = best_key

    agent_scores = action_scores(best_agent, feats[feature_name], te)
    test_predicted_costs = cost_predictors[feature_name].predict(
        feats[feature_name]["cont"][te]
    )
    agent_actions, test_lambdas, test_final_lambda = dual_budget_actions(
        agent_scores,
        test_predicted_costs,
        data.cost[te],
        cost_scale,
        args.budget_per_1k,
        best_dual_learning_rate,
    )

    agent_metrics = evaluate_actions(
        agent_actions, data, te, 0.0, cost_scale, rewards=R
    )
    agent_metrics["dual_final_lambda"] = test_final_lambda
    agent_metrics["dual_mean_lambda"] = float(test_lambdas.mean())
    agent_metrics["within_budget"] = (
        agent_metrics["cost_per_1k_usd"] <= args.budget_per_1k
    )

    comparison_rows = [{
        "method": f"Selected agent: {cfg_name(best_key)}",
        "validation_reward": best_val_reward,
        "validation_quality": best_val_quality,
        "validation_cost_per_1k_usd": selected_candidate["validation_cost_per_1k"],
        "validation_dual_final_lambda": validation_final_lambda,
        "test_dual_final_lambda": test_final_lambda,
        "selected_seed": best_seed,
        "validation_final_dual_lambda": validation_final_lambda,
        **agent_metrics,
    }]

    frequency_rows = []
    oracle_reward = base_res["oracle"]["reward"]
    diagnostic_rows = []

    # Evaluate each baseline on the SAME test split
    for name, actions in base_acts.items():
        metrics = evaluate_actions(
            actions, data, te, 0.0, cost_scale, rewards=R
        )
        display_name = "oracle_upper_bound" if name == "oracle" else name

        comparison_rows.append({
            "method": display_name,
            "validation_reward": np.nan,
            "selected_seed": np.nan,
            "validation_final_dual_lambda": np.nan,
            "oracle_note": (
                "Uses each test query's outcomes; upper bound, not attainable "
                "by a prompt-only router."
                if name == "oracle" else ""
            ),
            **metrics,
        })

        diagnostic = {
            "method": display_name,
            "mean_reward": metrics["reward"],
            "oracle_reward": oracle_reward,
            "regret_vs_oracle": oracle_reward - metrics["reward"],
            "validation_final_dual_lambda": np.nan,
            "oracle_note": (
                "Per-test-query outcomes; upper bound, not attainable by a "
                "prompt-only router."
                if name == "oracle" else ""
            ),
        }
        for action, model in enumerate(data.models):
            diagnostic[f"action_{action}_frequency"] = float(
                np.mean(actions == action)
            )
            frequency_rows.append({
                "method": display_name,
                "model": model,
                "selection_pct": 100 * np.mean(actions == action),
            })
        diagnostic_rows.append(diagnostic)

    # Add the selected agent's model selection frequencies
    selected_method = f"Selected agent: {cfg_name(best_key)}"
    selected_diagnostic = {
        "method": selected_method,
        "mean_reward": agent_metrics["reward"],
        "oracle_reward": oracle_reward,
        "regret_vs_oracle": oracle_reward - agent_metrics["reward"],
        "validation_final_dual_lambda": validation_final_lambda,
        "oracle_note": "",
    }
    for action, model in enumerate(data.models):
        selected_diagnostic[f"action_{action}_frequency"] = float(
            np.mean(agent_actions == action)
        )
        frequency_rows.append({
            "method": selected_method,
            "model": model,
            "selection_pct": 100 * np.mean(agent_actions == action),
        })
    diagnostic_rows.append(selected_diagnostic)

    comparison = pd.DataFrame(comparison_rows)
    frequencies = pd.DataFrame(frequency_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)

    comparison.to_csv(
        os.path.join(args.out, "selected_agent_vs_baselines.csv"),
        index=False,
    )
    frequencies.to_csv(
        os.path.join(args.out, "selected_agent_frequencies.csv"),
        index=False,
    )
    diagnostics.to_csv(
        os.path.join(args.out, "routing_diagnostics.csv"),
        index=False,
    )

    margin_rows = []
    for split_name, split_idx in (("train", tr), ("validation", va), ("test", te)):
        split_rewards = R[split_idx]
        margin_rows.append({
            "split": split_name,
            "prompt_count": len(split_idx),
            "medium_minus_small_reward": (
                float(np.mean(split_rewards[:, 1] - split_rewards[:, 0]))
                if A >= 2 else np.nan
            ),
            "large_minus_medium_reward": (
                float(np.mean(split_rewards[:, -1] - split_rewards[:, -2]))
                if A >= 3 else np.nan
            ),
            "large_minus_small_reward": (
                float(np.mean(split_rewards[:, -1] - split_rewards[:, 0]))
                if A >= 2 else np.nan
            ),
        })
    pd.DataFrame(margin_rows).to_csv(
        os.path.join(args.out, "reward_margin_diagnostics.csv"),
        index=False,
    )

    print(f"\nSelected configuration: {cfg_name(best_key)}")
    print(f"Selected seed: {best_seed}")
    print(f"Validation quality: {best_val_quality:.4f}")
    print(
        f"Validation cost: ${selected_candidate['validation_cost_per_1k']:.4f}/1k "
        f"(budget ${args.budget_per_1k:.4f}/1k)"
    )
    print(
        f"Dual lambda: validation ended at {validation_final_lambda:.4f}; "
        f"test ended at {test_final_lambda:.4f}"
    )
    print(f"Validation quality reward (diagnostic): {best_val_reward:.4f}")
    print("\n=== Selected agent vs baselines: TEST SET ===")
    print(comparison.round(4).to_string(index=False))

    print("\n=== Selection frequencies (%) ===")
    print(
        frequencies.pivot(
            index="method",
            columns="model",
            values="selection_pct",
        ).round(2).to_string()
    )
    print("\nSaved:")
    print(f"  {args.out}/routing_diagnostics.csv")
    print(f"  {args.out}/reward_margin_diagnostics.csv")
    print(f"  {args.out}/budget_selection_candidates.csv")
    # ---------------- save trained agents + featurizers for the demo ----------------
    for key, (_, agent, seed, _, _, validation_dual_lambda) in best.items():
        ext = "npz" if key[0] == "q" else "pt"
        agent.save(os.path.join(args.models_dir, f"{key[0]}_{key[1]}.{ext}"))
    for name, f in featurizers.items():
        joblib.dump(f, os.path.join(args.models_dir, f"{name}_featurizer.joblib"))
        joblib.dump(
            cost_predictors[name],
            os.path.join(args.models_dir, f"cost_predictor_{name}.joblib"),
        )
    with open(os.path.join(args.models_dir, "meta.json"), "w") as fh:
        json.dump({
            "models": data.models,
            "selection_budget_per_1k_usd": args.budget_per_1k,
            "selection_objective": "maximize validation quality subject to average validation cost budget",
            "dual_learning_rate": best_dual_learning_rate,
            "dual_update": "lambda = max(0, lambda + eta * (observed_normalized_cost - budget_per_prompt_normalized))",
            "cost_scale": cost_scale,
            "source": data.source,
            "best_config": list(best_key),
            "agent_calibrations": agent_calibrations,
            "reward_objective": "quality / estimated_difficulty - dynamic_lambda * normalized_cost",
            "difficulty_source": "DifficultyEstimator predictions in [1, 5]",
            "selection_feasible": bool(selected_candidate["feasible"]),
            "selected_agent": {
                "config": list(best_key),
                "seed": best_seed,
                "validation_quality": best_val_quality,
                "validation_cost_per_1k_usd": selected_candidate["validation_cost_per_1k"],
                "dual_learning_rate": best_dual_learning_rate,
            },
        }, fh, indent=2)
    with open(os.path.join(args.out, "experiment.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "selection_budget_per_1k_usd": args.budget_per_1k,
            "selection_objective": "maximize validation quality subject to average validation cost budget",
            "dual_learning_rate": best_dual_learning_rate,
            "selected_configuration": list(best_key),
            "selected_seed": best_seed,
            "validation_quality": best_val_quality,
            "validation_cost_per_1k_usd": selected_candidate["validation_cost_per_1k"],
            "dual_learning_rate": best_dual_learning_rate,
            "validation_final_dual_lambda": validation_final_lambda,
            "test_final_dual_lambda": test_final_lambda,
            "selection_feasible": bool(selected_candidate["feasible"]),
            "reward_objective": "quality / estimated_difficulty - dynamic_lambda * normalized_cost",
            "difficulty_source": "DifficultyEstimator predictions in [1, 5]",
            "oracle_note": (
                "Upper bound using per-test-query outcomes; not attainable by "
                "a prompt-only router."
            ),
        }, fh, indent=2)
    print(f"\nBest configuration under validation budget: {cfg_name(best_key)}")
    print(f"Figures -> {fig_dir} | tables -> {args.out} | agents -> {args.models_dir}")
if __name__ == "__main__":
    main()
