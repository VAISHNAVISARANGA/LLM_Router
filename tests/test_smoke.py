"""Quick end-to-end checks using small, deterministic in-memory fixtures."""
import json
import pandas as pd
import pytest
import numpy as np
import joblib
import sys

from src import run_experiments
from src.data import RouterData, assign_splits, load_data, load_routerbench, save_data
from src.config import DEFAULT_LAMBDA, MODEL_POOL
from src.demo import tier_labels
from src.env import RoutingEnv, reward_matrix
from src.evaluate import baseline_actions, evaluate_actions
from src.features import (
    EmbeddingFeaturizer,
    HandcraftedFeaturizer,
    HybridFeaturizer,
    calculate_difficulty_scores,
    embed_texts,
)
from src import plots
from src.run_experiments import (
    CONFIGS,
    _make_features,
    available_configs,
    cfg_name,
)
from src.router import LLMRouter
from src.agents.qlearning import QLearningAgent
from src.train import (
    action_scores,
    calibrate_final_action_penalty,
    policy_actions,
    train_agent,
)


def test_experiment_configs_have_plot_labels_and_styles():
    assert set(CONFIGS) <= plots.LABELS.keys()
    assert set(CONFIGS) <= plots.STYLE.keys()
    assert cfg_name(("q", "hybrid")) == "Q-learn + hybrid"


def test_hybrid_features_are_built_for_q_learning_and_dqn():
    data, tr, _, _, _ = _setup(120)
    featurizer = HybridFeaturizer(embedder="hash", n_clusters=4, seed=1).fit(
        prompts=[data.prompts[i] for i in tr],
        emb=data.embeddings[tr],
    )

    features = _make_features(
        featurizer,
        data.prompts,
        data.embeddings,
    )

    assert features["disc"].shape == (len(data.prompts),)
    assert features["disc"].min() >= 0
    assert features["disc"].max() < features["n_states"]
    assert features["cont"].shape == (
        len(data.prompts),
        featurizer.dim,
    )
    assert features["dim"] == features["cont"].shape[1]


def _setup(n=2000):
    difficulty = np.linspace(0.0, 1.0, n)
    prompts = []
    tasks = []
    for i, score in enumerate(difficulty):
        padding = " detail" * int(score * 20)
        if i % 3 == 0:
            prompts.append(f"Solve step by step: {i % 97} + 2 = ?{padding}")
            tasks.append("math")
        elif i % 3 == 1:
            prompts.append(f"Explain why this claim is true: example {i}{padding}")
            tasks.append("reasoning")
        else:
            prompts.append(
                f"Write Python code:\n```python\nprint({i % 10})\n```{padding}"
            )
            tasks.append("code")

    q_small = np.clip(0.95 - 0.85 * difficulty, 0.05, 1.0)
    q_medium = np.clip(0.96 - 0.50 * difficulty, 0.05, 1.0)
    q_large = np.clip(0.97 - 0.20 * difficulty, 0.05, 1.0)
    token_count = np.array([len(prompt.split()) for prompt in prompts])
    total_tokens = token_count * 1.3 + 80 + 400 * difficulty
    prices = np.array([0.2e-6, 2e-6, 30e-6])
    costs = total_tokens[:, None] * prices[None, :]

    data = RouterData(
        prompts=prompts,
        tasks=np.asarray(tasks),
        quality=np.stack([q_small, q_medium, q_large], axis=1),
        cost=costs,
        models=["test-small", "test-medium", "test-large"],
        split=assign_splits(n, seed=1),
        source="test-fixture",
    )
    data.embeddings = embed_texts(data.prompts, "hash")
    data.embedder = "hash"
    tr, te = data.idx("train"), data.idx("test")
    hand = HandcraftedFeaturizer().fit([data.prompts[i] for i in tr])
    d, c = hand.transform(data.prompts)
    feats = {"disc": d, "cont": c, "n_states": hand.n_states, "dim": hand.dim}
    scale = float(np.quantile(data.cost[tr, -1], 0.99))
    return data, tr, te, feats, scale


def test_state_ids_in_range():
    data, tr, te, feats, _ = _setup()
    assert feats["disc"].min() >= 0 and feats["disc"].max() < feats["n_states"]
    assert data.quality.shape[1] == data.cost.shape[1] == 3
    assert data.models[1] == "test-medium"
    assert np.all(data.cost[:, 0] < data.cost[:, 1])
    assert np.all(data.cost[:, 1] < data.cost[:, 2])


def test_reward_matrix_uses_quality_minus_normalized_cost():
    quality = np.array([[0.5, 0.8], [0.6, 0.9]])
    cost = np.array([[0.0, 0.5], [0.25, 1.0]])

    rewards = reward_matrix(quality, cost, lam=0.5, cost_scale=1.0)
    np.testing.assert_allclose(rewards, quality - 0.5 * cost)


def test_reward_matrix_weights_quality_with_prompt_structural_difficulty():
    quality = np.array([[0.5, 0.8], [0.5, 0.8]])
    cost = np.array([[0.1, 0.2], [0.1, 0.2]])
    difficulty = np.array([1.0, 2.0])
    lam = 0.5

    rewards = reward_matrix(
        quality,
        cost,
        lam,
        cost_scale=1.0,
        difficulty=difficulty,
    )
    expected = quality * np.array([[1.0], [2.0]]) - lam * cost
    np.testing.assert_allclose(rewards, expected)

    unweighted = reward_matrix(
        quality,
        cost,
        lam,
        cost_scale=1.0,
        difficulty=difficulty,
        difficulty_strength=0.0,
    )
    np.testing.assert_allclose(unweighted, quality - lam * cost)

    prompt_scores = calculate_difficulty_scores([
        "Hi?",
        "Derive the equation $x^2 + y^2 = z^2$.\n```python\nprint(1)\n```",
    ])
    assert 1.0 <= prompt_scores[0] <= 2.0
    assert prompt_scores[1] > prompt_scores[0]


def test_reward_is_shared_by_training_environment_baselines_and_evaluation():
    data, tr, te, feats, scale = _setup(120)
    lam = 0.7
    rewards = reward_matrix(data.quality, data.cost, lam, scale)
    env = RoutingEnv(rewards, feats["disc"], feats["cont"], tr, episode_len=1, seed=3)
    env.reset()
    _, observed_reward, _, info = env.step(1)
    row = info["query_index"]
    assert observed_reward == rewards[row, 1]

    actions = np.zeros(len(te), dtype=int)
    metrics = evaluate_actions(actions, data, te, lam, scale, rewards=rewards)
    assert metrics["reward"] == pytest.approx(rewards[te, 0].mean())

    baselines = baseline_actions(data, te, tr, lam, scale, rewards=rewards)
    fixed_metrics = evaluate_actions(
        baselines["always_small"], data, te, lam, scale, rewards=rewards
    )
    assert fixed_metrics["reward"] == metrics["reward"]
    oracle_metrics = evaluate_actions(
        baselines["oracle"], data, te, lam, scale, rewards=rewards
    )
    np.testing.assert_allclose(
        rewards[te].max(axis=1).mean(),
        oracle_metrics["reward"],
    )


def test_final_action_penalty_changes_only_last_action():
    from src.train import adjusted_action_scores, actions_from_scores

    scores = np.array([[3.0, 2.0, 2.5], [1.0, 3.0, 2.5]])
    adjusted = adjusted_action_scores(scores, final_action_penalty=1.0)
    np.testing.assert_array_equal(adjusted[:, :-1], scores[:, :-1])
    np.testing.assert_allclose(adjusted[:, -1], scores[:, -1] - 1.0)
    np.testing.assert_array_equal(actions_from_scores(scores, 1.0), [0, 1])


def test_validation_penalty_calibration_never_reads_test_rewards():
    agent = QLearningAgent(n_states=4, n_actions=3)
    agent.Q[1] = [0.0, 0.9, 1.0]
    agent.Q[2] = [0.0, 0.8, 0.9]
    feats = {
        "disc": np.arange(4),
        "cont": np.zeros((4, 1), dtype=np.float32),
    }
    rewards = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.9, 0.1],
        [0.0, 0.8, 0.2],
        [0.0, 0.0, 1.0],
    ])
    validation_idx = np.array([1, 2])
    penalty, validation_reward = calibrate_final_action_penalty(
        agent, feats, validation_idx, rewards[validation_idx], [0.0, 0.2]
    )
    assert penalty == 0.2
    assert validation_reward == pytest.approx(0.85)

    rewards[3] = [1.0, 0.0, 0.0]
    changed_test_penalty, changed_validation_reward = calibrate_final_action_penalty(
        agent, feats, validation_idx, rewards[validation_idx], [0.0, 0.2]
    )
    assert changed_test_penalty == penalty
    assert changed_validation_reward == validation_reward


def test_experiment_calibrates_only_on_validation_and_writes_diagnostics(
    tmp_path, monkeypatch
):
    data, _, _, _, _ = _setup(120)
    data.embeddings = None
    monkeypatch.setattr(run_experiments, "CONFIGS", run_experiments.CONFIGS)
    validation_idx = data.idx("val")
    calibration_indices = []
    training_reward_matrices = []
    original_calibrate = run_experiments.calibrate_final_action_penalty
    original_train_agent = run_experiments.train_agent

    def capture_training_rewards(kind, feats, rewards, *args, **kwargs):
        training_reward_matrices.append(np.asarray(rewards).copy())
        return original_train_agent(kind, feats, rewards, *args, **kwargs)

    def capture_calibration(agent, feats, idx, validation_rewards, candidates):
        calibration_indices.append(np.asarray(idx).copy())
        assert len(validation_rewards) == len(validation_idx)
        return original_calibrate(
            agent, feats, idx, validation_rewards, candidates
        )

    monkeypatch.setattr(
        run_experiments,
        "load_data",
        lambda path, require_embeddings: data,
    )
    monkeypatch.setattr(
        run_experiments,
        "available_configs",
        lambda has_embeddings: [("q", "handcrafted")],
    )
    monkeypatch.setattr(
        run_experiments,
        "calibrate_final_action_penalty",
        capture_calibration,
    )
    monkeypatch.setattr(
        run_experiments,
        "train_agent",
        capture_training_rewards,
    )
    monkeypatch.setattr(run_experiments.plots, "training_curves", lambda *args: None)
    monkeypatch.setattr(run_experiments.plots, "pareto", lambda *args: None)
    monkeypatch.setattr(run_experiments.plots, "reward_bars", lambda *args: None)

    out_dir = tmp_path / "results"
    models_dir = tmp_path / "models"
    monkeypatch.setattr(sys, "argv", [
        "run_experiments",
        "--episodes", "1",
        "--episode-len", "2",
        "--seeds", "1",
        "--lam", "0.5",
        "--lams", "0.5",
        "--gpt-penalties", "0", "0.2",
        "--data", "fixture.joblib",
        "--out", str(out_dir),
        "--models-dir", str(models_dir),
    ])

    run_experiments.main()

    assert len(calibration_indices) == 2  # main run and lambda sweep
    assert len(training_reward_matrices) == 2
    expected_difficulty = calculate_difficulty_scores(data.prompts)
    expected_rewards = reward_matrix(
        data.quality,
        data.cost,
        lam=0.5,
        cost_scale=float(np.quantile(data.cost[data.idx("train"), -1], 0.99)),
        difficulty=expected_difficulty,
    )
    for training_rewards in training_reward_matrices:
        np.testing.assert_allclose(training_rewards, expected_rewards)
    assert all(
        np.array_equal(indices, validation_idx)
        for indices in calibration_indices
    )
    calibration = pd.read_csv(out_dir / "gpt_penalty_calibration.csv")
    assert set(calibration["validation_selected_penalty"]) <= {0.0, 0.2}
    diagnostics = pd.read_csv(out_dir / "routing_diagnostics.csv")
    assert {"mean_reward", "oracle_reward", "regret_vs_oracle",
            "validation_selected_gpt_penalty", "action_2_frequency"} <= set(
                diagnostics.columns
            )
    oracle = diagnostics.loc[
        diagnostics["method"] == "oracle_upper_bound"
    ].iloc[0]
    assert "not attainable" in oracle["oracle_note"]
    assert diagnostics["mean_reward"].notna().all()
    assert diagnostics["action_2_frequency"].between(0.0, 1.0).all()
    margins = pd.read_csv(out_dir / "reward_margin_diagnostics.csv")
    assert set(margins["split"]) == {"train", "validation", "test"}
    metadata = json.loads((models_dir / "meta.json").read_text(encoding="utf-8"))
    assert metadata["gpt_penalty"] in {0.0, 0.2}
    assert metadata["lambda"] == 0.5
    assert not (models_dir / "difficulty_estimator.joblib").exists()


def test_handcrafted_features_and_router_do_not_require_embeddings_or_difficulty(tmp_path):
    data, tr, _, _, _ = _setup(120)
    data.source = "routerbench"
    data.embeddings = None
    dataset_path = tmp_path / "handcrafted.joblib"
    save_data(data, dataset_path)
    loaded = load_data(dataset_path, require_embeddings=False)
    assert loaded.embeddings is None
    configs = available_configs(has_embeddings=False)
    assert configs
    assert all(feature_name == "handcrafted" for _, feature_name in configs)

    featurizer = HandcraftedFeaturizer().fit([loaded.prompts[i] for i in tr])
    agent = QLearningAgent(featurizer.n_states, 3)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    joblib.dump(featurizer, models_dir / "handcrafted_featurizer.joblib")
    agent.save(models_dir / "q_handcrafted.npz")
    (models_dir / "meta.json").write_text(
        '{"models":["small","medium","large"],"gpt_penalty":0.0}',
        encoding="utf-8",
    )

    router = LLMRouter(models_dir=models_dir, agent="q", state="handcrafted")
    action, scores = router.route("Explain a short example.")
    assert 0 <= action < 3
    assert scores.shape == (3,)
    assert not (models_dir / "difficulty_estimator.joblib").exists()


def test_load_data_requires_an_existing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Processed dataset not found"):
        load_data(tmp_path / "missing.joblib")


def test_load_data_rejects_non_routerbench_data(tmp_path):
    import joblib

    data, *_ = _setup(30)
    path = tmp_path / "not_routerbench.joblib"
    joblib.dump(data, path)
    with pytest.raises(ValueError, match="only real RouterBench data is accepted"):
        load_data(path)


def test_load_data_rejects_invalid_real_dataset_values(tmp_path):
    import joblib

    data, *_ = _setup(30)
    data.source = "routerbench"
    data.quality[0, 0] = np.nan
    path = tmp_path / "invalid_routerbench.joblib"
    joblib.dump(data, path)
    with pytest.raises(ValueError, match="invalid quality values"):
        load_data(path)


def test_load_routerbench_requires_existing_pickle(tmp_path):
    with pytest.raises(FileNotFoundError, match="RouterBench pickle not found"):
        load_routerbench(
            tmp_path / "missing.pkl",
            ["small-model", "large-model"],
        )


def test_load_routerbench_rejects_missing_model_columns(tmp_path):
    path = tmp_path / "incomplete.pkl"
    pd.DataFrame({"prompt": ["A real prompt"]}).to_pickle(path)
    with pytest.raises(KeyError, match="Columns not found"):
        load_routerbench(
            path,
            ["small-model", "large-model"],
        )


def test_load_routerbench_loads_real_columns_without_fabricating_embeddings(tmp_path):
    path = tmp_path / "routerbench.pkl"
    pd.DataFrame({
        "prompt": [f"Question {i}?" for i in range(100)],
        "eval_name": ["test-task"] * 100,
        "small-model": np.linspace(0.1, 0.9, 100),
        "large-model": np.linspace(0.3, 1.0, 100),
        "small-model|total_cost": np.full(100, 0.001),
        "large-model|total_cost": np.full(100, 0.01),
    }).to_pickle(path)

    data = load_routerbench(path, ["small-model", "large-model"])

    assert data.source == "routerbench"
    assert len(data.prompts) == 100
    assert data.embeddings is None
    assert set(data.split) == {"train", "val", "test"}


def test_demo_tier_labels_preserve_two_model_routers():
    assert tier_labels(2) == ["SMALL", "LARGE"]
    assert tier_labels(3) == ["SMALL", "MEDIUM", "LARGE"]


def test_default_model_pool_matches_routerbench_columns():
    assert DEFAULT_LAMBDA == 1.0
    assert MODEL_POOL == [
        "mistralai/mistral-7b-chat",
        "mistralai/mixtral-8x7b-chat",
        "gpt-4-1106-preview",
    ]


def test_qlearning_beats_random_and_never_worse_than_best_fixed_by_much():
    data, tr, te, feats, scale = _setup()
    lam = 1.0
    R = reward_matrix(data.quality, data.cost, lam, scale)
    agent, hist = train_agent("q", feats, R, tr, seed=0, episodes=300, episode_len=100)
    scores = action_scores(agent, feats, te)
    assert scores.shape == (len(te), data.quality.shape[1])
    got = evaluate_actions(
        policy_actions(agent, feats, te), data, te, lam, scale, rewards=R
    )["reward"]
    base = baseline_actions(data, te, tr, lam, scale)
    rnd = evaluate_actions(base["random"], data, te, lam, scale)["reward"]
    orc = evaluate_actions(base["oracle"], data, te, lam, scale)["reward"]
    assert got > rnd
    assert got <= orc + 1e-9


def test_dqn_runs():
    import pytest
    pytest.importorskip("torch")
    data, tr, te, feats, scale = _setup(800)
    R = reward_matrix(data.quality, data.cost, 1.0, scale)
    agent, hist = train_agent("dqn", feats, R, tr, seed=0, episodes=30, episode_len=50)
    assert len(hist) == 30 and np.isfinite(hist).all()
    assert action_scores(agent, feats, te).shape == (
        len(te),
        data.quality.shape[1],
    )
