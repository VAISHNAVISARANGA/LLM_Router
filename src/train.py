"""Training and policy helpers for Q-learning, DQN, and LinUCB."""
import numpy as np

from .env import RoutingEnv
from .agents.qlearning import QLearningAgent
from .agents.dqn import DQNAgent
from .agents.linucb import LinUCBAgent


def pick(kind, obs):
    return obs[0] if kind == "q" else obs[1]


def train_agent(kind, feats, R, train_idx, seed=0, episodes=500,
                episode_len=100, gamma=0.0, eps_decay_frac=0.6):
    """Train with feedback from the selected action only."""
    env = RoutingEnv(R, feats["disc"], feats["cont"], train_idx,
                     episode_len=episode_len, seed=seed)
    n_actions = R.shape[1]
    decay_steps = int(eps_decay_frac * episodes * episode_len)
    if kind == "q":
        agent = QLearningAgent(feats["n_states"], n_actions, gamma=gamma,
                               eps_decay_steps=decay_steps, seed=seed)
    elif kind == "dqn":
        agent = DQNAgent(feats["dim"], n_actions, gamma=gamma,
                         eps_decay_steps=decay_steps, seed=seed)
    elif kind == "linucb":
        agent = LinUCBAgent(feats["dim"], n_actions, alpha=1.0, ridge=1.0, seed=seed)
    else:
        raise ValueError(f"Unknown agent kind: {kind}")
    history = []
    for _ in range(episodes):
        obs, total, done = env.reset(), 0.0, False
        while not done:
            state = pick(kind, obs)
            action = agent.act(state)
            obs2, reward, done, _ = env.step(action)
            agent.observe(state, action, reward, pick(kind, obs2), done)
            total += reward
            obs = obs2
        history.append(total / episode_len)
    return agent, history


def action_scores(agent, feats, idx):
    """Return one row of action scores/Q-values for each requested example."""
    idx = np.asarray(idx, dtype=int)
    X = feats["disc"][idx] if agent.kind == "q" else feats["cont"][idx]
    scores = np.asarray(agent.q_values(X), dtype=np.float64)
    if scores.ndim == 1 and len(idx) == 1:
        scores = scores[None, :]
    if scores.ndim != 2 or scores.shape[0] != len(idx):
        raise ValueError("Agent action scores must have shape (examples, actions).")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Agent returned non-finite action scores.")
    return scores


def adjusted_action_scores(scores, final_action_penalty=0.0):
    """Subtract a non-negative penalty from only the final action's scores."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim not in (1, 2) or scores.shape[-1] < 1:
        raise ValueError("Action scores must be a non-empty 1D or 2D array.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Action scores must be finite.")
    if not np.isfinite(final_action_penalty) or final_action_penalty < 0:
        raise ValueError("Final-action penalty must be finite and non-negative.")
    adjusted = scores.copy()
    adjusted[..., -1] -= final_action_penalty
    return adjusted


def actions_from_scores(scores, final_action_penalty=0.0):
    """Choose greedy actions after penalizing only the final action."""
    return np.argmax(
        adjusted_action_scores(scores, final_action_penalty),
        axis=-1,
    )


def select_final_action_penalty(validation_scores, validation_rewards, candidates):
    """Choose a final-action penalty by mean reward on validation examples only."""
    validation_scores = np.asarray(validation_scores, dtype=np.float64)
    validation_rewards = np.asarray(validation_rewards, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.float64)
    if (
        validation_scores.ndim != 2
        or validation_rewards.shape != validation_scores.shape
        or validation_scores.shape[0] == 0
    ):
        raise ValueError("Validation scores and rewards must be matching non-empty matrices.")
    if candidates.ndim != 1 or candidates.size == 0:
        raise ValueError("At least one candidate final-action penalty is required.")
    if not np.all(np.isfinite(candidates)) or np.any(candidates < 0):
        raise ValueError("Candidate penalties must be finite and non-negative.")
    if not np.all(np.isfinite(validation_rewards)):
        raise ValueError("Validation rewards must be finite.")

    row_indices = np.arange(validation_scores.shape[0])
    best_penalty = float(candidates[0])
    best_reward = -np.inf
    for penalty in candidates:
        actions = actions_from_scores(validation_scores, float(penalty))
        mean_reward = float(validation_rewards[row_indices, actions].mean())
        if mean_reward > best_reward:
            best_penalty, best_reward = float(penalty), mean_reward
    return best_penalty, best_reward


def calibrate_final_action_penalty(
    agent, feats, validation_idx, validation_rewards, candidates
):
    """Calibrate using only scores and rewards for the supplied validation rows."""
    validation_idx = np.asarray(validation_idx, dtype=int)
    validation_rewards = np.asarray(validation_rewards, dtype=np.float64)
    if (
        validation_idx.ndim != 1
        or validation_rewards.ndim != 2
        or len(validation_idx) != len(validation_rewards)
        or not len(validation_idx)
        or np.any(validation_idx < 0)
    ):
        raise ValueError("Validation indices and reward rows must be non-empty and aligned.")
    return select_final_action_penalty(
        action_scores(agent, feats, validation_idx),
        validation_rewards,
        candidates,
    )


def policy_actions(agent, feats, idx, final_action_penalty=0.0):
    """Return greedy actions; the optional penalty preserves prior behavior at zero."""
    return actions_from_scores(
        action_scores(agent, feats, idx),
        final_action_penalty,
    )
