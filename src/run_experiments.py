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
    calculate_difficulty_scores,
)
from . import plots
from .config import DEFAULT_LAMBDA, LAMBDA_SWEEP, SEED
from .data import load_data
from .env import reward_matrix
from .evaluate import baseline_actions, evaluate_actions
from .train import (
    calibrate_final_action_penalty,
    policy_actions,
    train_agent,
)
CONFIGS = [
    ("q", "handcrafted"),
    ("q", "embedding"),
    ("q", "hybrid"),
    ("dqn", "handcrafted"),
    ("dqn", "embedding"),
    ("dqn", "hybrid"),
]

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
        "--lam",
        type=float,
        default=DEFAULT_LAMBDA,
        help=(
            "Reward cost-penalty weight for the trained/deployed model "
            f"(default: {DEFAULT_LAMBDA:g})."
        ),
    )
    ap.add_argument(
        "--lams",
        type=float,
        nargs="+",
        default=LAMBDA_SWEEP,
        help=(
            "Lambda values for the validation/test trade-off report. "
            "Different lambda rewards are different objectives and are not "
            "ranked against each other to choose the deployment lambda."
        ),
    )
    ap.add_argument(
        "--gpt-penalties",
        type=float,
        nargs="+",
        default=[0.0, 0.005, 0.01, 0.02, 0.05, 0.1],
        help="Candidate penalties for the final action, selected on validation reward.",
    )
    ap.add_argument(
        "--difficulty-strength",
        type=float,
        default=1.0,
        help=(
            "Strength of prompt-structure difficulty weighting on quality "
            "(0 disables it; default 1 scales quality by 1x to 2x)."
        ),
    )
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--episode-len", type=int, default=100)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--n-clusters", type=int, default=32)
    ap.add_argument("--out", default="results")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--quick", action="store_true", help="1 seed, 100 episodes, 3 lambdas")
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.episodes, args.lams = 1, 100, [0.0, args.lam, 4.0]
    if not args.gpt_penalties or any(
        not np.isfinite(penalty) or penalty < 0
        for penalty in args.gpt_penalties
    ):
        ap.error("--gpt-penalties must contain finite, non-negative values")
    if not np.isfinite(args.lam) or args.lam < 0:
        ap.error("--lam must be finite and non-negative")
    if not args.lams or any(
        not np.isfinite(lam) or lam < 0 for lam in args.lams
    ):
        ap.error("--lams must contain finite, non-negative values")
    if not np.isfinite(args.difficulty_strength) or args.difficulty_strength < 0:
        ap.error("--difficulty-strength must be finite and non-negative")

    fig_dir = os.path.join(args.out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)

    # ---------------- data + features (fit on TRAIN only) ----------------
    
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

    # This score uses prompt structure only; it does not inspect RouterBench outcomes.
    difficulty = calculate_difficulty_scores(data.prompts)
    print(
        f"Prompt-structure difficulty range: "
        f"{difficulty.min():.2f} to {difficulty.max():.2f}"
    )
    R = reward_matrix(
        data.quality,
        data.cost,
        args.lam,
        cost_scale,
        difficulty=difficulty,
        difficulty_strength=args.difficulty_strength,
    )

    histories = {k: [] for k in CONFIGS}
    results = {k: [] for k in CONFIGS}
    best = {}   # best calibrated seed by validation reward, per config
    calibration_rows = []

    for key in CONFIGS:
        kind, fname = key
        for seed in range(args.seeds):
            print(f"[main] {cfg_name(key)} seed {seed} ...", flush=True)
            agent, hist = train_agent(kind, feats[fname], R, tr, seed, args.episodes,
                                      args.episode_len, args.gamma)
            histories[key].append(hist)
            penalty, validation_reward = calibrate_final_action_penalty(
                agent, feats[fname], va, R[va], args.gpt_penalties
            )
            validation_actions = policy_actions(
                agent, feats[fname], va, final_action_penalty=penalty
            )
            test_actions = policy_actions(
                agent, feats[fname], te, final_action_penalty=penalty
            )
            va_res = evaluate_actions(
                validation_actions, data, va, args.lam, cost_scale, rewards=R
            )
            te_res = evaluate_actions(
                test_actions, data, te, args.lam, cost_scale, rewards=R
            )
            te_res["gpt_penalty"] = penalty
            te_res["validation_reward"] = validation_reward
            results[key].append(te_res)
            calibration_rows.append({
                "lambda": args.lam,
                "configuration": cfg_name(key),
                "seed": seed,
                "validation_selected_penalty": penalty,
                "validation_reward": va_res["reward"],
                "test_reward": te_res["reward"],
            })
            if key not in best or validation_reward > best[key][0]:
                best[key] = (validation_reward, agent, seed, penalty)

    base_acts = baseline_actions(data, te, tr, args.lam, cost_scale, SEED, rewards=R)
    base_res = {
        name: evaluate_actions(actions, data, te, args.lam, cost_scale, rewards=R)
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
        metric_keys = [
            key for key in rs[0]
            if key not in ("validation_reward", "gpt_penalty")
        ]
        m = {name: float(np.mean([r[name] for r in rs])) for name in metric_keys}
        rows.append({"method": cfg_name(key), **m,
                     "validation_reward_mean": float(np.mean([r["validation_reward"] for r in rs])),
                     "gpt_penalty_mean": float(np.mean([r["gpt_penalty"] for r in rs])),
                     "gpt_penalty_std": float(np.std([r["gpt_penalty"] for r in rs])),
                     "reward_std": float(np.std([r["reward"] for r in rs])),
                     "quality_std": float(np.std([r["quality"] for r in rs])),
                     "cost_std": float(np.std([r["cost_per_1k_usd"] for r in rs]))})
    table = pd.DataFrame(rows)
    table["cost_saving_vs_large_%"] = 100 * (1 - table["cost_per_1k_usd"] / large["cost_per_1k_usd"])
    table["quality_retained_%"] = 100 * table["quality"] / large["quality"]
    table.to_csv(os.path.join(args.out, "metrics_main.csv"), index=False)
    print(
        f"\n=== Test-set results (deployment lambda={args.lam}, "
        f"{args.seeds} seeds) ==="
    )
    print(table.round(4).to_string(index=False))

    # ---------------- lambda sweep (Pareto front) ----------------
    sweep = {k: [] for k in CONFIGS}
    sweep_rows = []
    for lam in args.lams:
        R_l = reward_matrix(
            data.quality,
            data.cost,
            lam,
            cost_scale,
            difficulty=difficulty,
            difficulty_strength=args.difficulty_strength,
        )
        for key in CONFIGS:
            kind, fname = key
            print(f"[sweep] lambda={lam} {cfg_name(key)} ...", flush=True)
            agent, _ = train_agent(kind, feats[fname], R_l, tr, 0, args.episodes,
                                   args.episode_len, args.gamma)
            penalty, validation_reward = calibrate_final_action_penalty(
                agent, feats[fname], va, R_l[va], args.gpt_penalties
            )
            validation_actions = policy_actions(
                agent, feats[fname], va, final_action_penalty=penalty
            )
            test_actions = policy_actions(
                agent, feats[fname], te, final_action_penalty=penalty
            )
            validation_metrics = evaluate_actions(
                validation_actions, data, va, lam, cost_scale, rewards=R_l
            )
            test_metrics = evaluate_actions(
                test_actions, data, te, lam, cost_scale, rewards=R_l
            )
            sweep[key].append(
                (test_metrics["cost_per_1k_usd"], test_metrics["quality"])
            )
            sweep_rows.append({
                "lambda": lam,
                "method": cfg_name(key),
                "validation_selected_penalty": penalty,
                **{
                    f"validation_{name}": value
                    for name, value in validation_metrics.items()
                },
                **{f"test_{name}": value for name, value in test_metrics.items()},
            })
            calibration_rows.append({
                "lambda": lam,
                "configuration": cfg_name(key),
                "seed": 0,
                "validation_selected_penalty": penalty,
                "validation_reward": validation_reward,
                "test_reward": test_metrics["reward"],
            })
    pd.DataFrame(sweep_rows).to_csv(os.path.join(args.out, "metrics_sweep.csv"), index=False)
    pd.DataFrame(calibration_rows).to_csv(
        os.path.join(args.out, "gpt_penalty_calibration.csv"),
        index=False,
    )

    # ---------------- figures ----------------
    plots.training_curves({k: np.stack(v) for k, v in histories.items()}, fig_dir)
    plots.pareto(sweep, {n: (r["cost_per_1k_usd"], r["quality"]) for n, r in base_res.items()}, fig_dir)
    bars = [(n, r["reward"], 0.0) for n, r in base_res.items() if n != "random"]
    bars += [(cfg_name(k), float(np.mean([r["reward"] for r in rs])),
              float(np.std([r["reward"] for r in rs]))) for k, rs in results.items()]
    plots.reward_bars(bars, fig_dir)

    
    # Select the configuration using validation reward only
    # ---------------------------------------------------------
# Select ONE configuration using validation reward.
# Compare that selected agent with all baselines on test.
# ---------------------------------------------------------

    best_key = max(best, key=lambda k: best[k][0])
    best_val_reward, best_agent, best_seed, best_penalty = best[best_key]

    kind, feature_name = best_key

    # Predictions from the single validation-selected checkpoint
    agent_actions = policy_actions(
        best_agent,
        feats[feature_name],
        te,
        final_action_penalty=best_penalty,
    )

    # Evaluate the selected agent on test
    agent_metrics = evaluate_actions(
        agent_actions, data, te, args.lam, cost_scale, rewards=R
    )

    comparison_rows = [{
        "method": f"Selected agent: {cfg_name(best_key)}",
        "validation_reward": best_val_reward,
        "selected_seed": best_seed,
        "validation_selected_gpt_penalty": best_penalty,
        **agent_metrics,
    }]

    frequency_rows = []
    oracle_reward = base_res["oracle"]["reward"]
    diagnostic_rows = []

    # Evaluate each baseline on the SAME test split
    for name, actions in base_acts.items():
        metrics = evaluate_actions(
            actions, data, te, args.lam, cost_scale, rewards=R
        )
        display_name = "oracle_upper_bound" if name == "oracle" else name

        comparison_rows.append({
            "method": display_name,
            "validation_reward": np.nan,
            "selected_seed": np.nan,
            "validation_selected_gpt_penalty": np.nan,
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
            "validation_selected_gpt_penalty": np.nan,
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
        "validation_selected_gpt_penalty": best_penalty,
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
    print(f"Best validation reward: {best_val_reward:.4f}")
    print(f"Validation-selected final-action penalty: {best_penalty:g}")

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
    print(f"  {args.out}/gpt_penalty_calibration.csv")
    # ---------------- save trained agents + featurizers for the demo ----------------
    agent_calibrations = {}
    for key, (_, agent, seed, penalty) in best.items():
        ext = "npz" if key[0] == "q" else "pt"
        agent.save(os.path.join(args.models_dir, f"{key[0]}_{key[1]}.{ext}"))
        agent_calibrations[f"{key[0]}_{key[1]}"] = {
            "seed": seed,
            "gpt_penalty": penalty,
        }
    for name, f in featurizers.items():
        joblib.dump(f, os.path.join(args.models_dir, f"{name}_featurizer.joblib"))
    with open(os.path.join(args.models_dir, "meta.json"), "w") as fh:
        json.dump({
            "models": data.models,
            "lambda": args.lam,
            "lambda_selection": (
                "Fixed deployment objective chosen from validation "
                "quality/cost trade-offs; lambda sweep is diagnostic."
            ),
            "cost_scale": cost_scale,
            "source": data.source,
            "best_config": list(best_key),
            "gpt_penalty": best_penalty,
            "agent_calibrations": agent_calibrations,
            "gpt_penalty_candidates": args.gpt_penalties,
            "reward_objective": (
                "quality * (1 + difficulty_strength * (structural_difficulty - 1)) "
                "- lambda * normalized_cost"
            ),
            "difficulty_source": "prompt-only structural proxy in [1, 2]",
            "difficulty_strength": args.difficulty_strength,
        }, fh, indent=2)
    with open(os.path.join(args.out, "experiment.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "lambda": args.lam,
            "lambda_selection": (
                "Fixed deployment objective chosen from validation "
                "quality/cost trade-offs; lambda sweep is diagnostic."
            ),
            "selected_configuration": list(best_key),
            "selected_seed": best_seed,
            "validation_selected_gpt_penalty": best_penalty,
            "gpt_penalty_candidates": args.gpt_penalties,
            "reward_objective": (
                "quality * (1 + difficulty_strength * (structural_difficulty - 1)) "
                "- lambda * normalized_cost"
            ),
            "difficulty_source": "prompt-only structural proxy in [1, 2]",
            "difficulty_strength": args.difficulty_strength,
            "oracle_note": (
                "Upper bound using per-test-query outcomes; not attainable by "
                "a prompt-only router."
            ),
        }, fh, indent=2)
    print(f"\nBest configuration on validation reward: {cfg_name(best_key)}")
    print(f"Figures -> {fig_dir} | tables -> {args.out} | agents -> {args.models_dir}")
if __name__ == "__main__":
    main()
