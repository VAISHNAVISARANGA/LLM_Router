"""The routing environment (Gym-style, but dependency-free).

One episode = a stream of `episode_len` user queries.
At every step the agent
    sees   : the features of the current query               (state)
    picks  : which LLM answers it (0 = cheapest, last = most expensive) (action)
    gets   : reward = quality / difficulty - lambda * normalised_cost (reward)

Because every query was pre-answered by every model (RouterBench), the
environment is a fast table lookup: R[query, model].

Note: the next query does not depend on the action taken, so routing is
a contextual-bandit-style MDP (gamma = 0 by default).  We still use the full
Q-learning / DQN update rules, with gamma exposed as a hyper-parameter.
"""
import numpy as np
def normalise_cost(cost, scale):
    """Map raw USD cost to ~[0, 1]; `scale` = 99th percentile of the large model's cost."""
    return np.clip(cost / scale, 0.0, 1.0)



def reward_matrix(
    quality,
    cost,
    lam,
    cost_scale,
    difficulty=None,
):
    """Return quality divided by difficulty, minus lambda-weighted normalized cost."""
    quality = np.asarray(quality, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    if quality.shape != cost.shape or quality.ndim != 2:
        raise ValueError("quality and cost must have the same 2D shape.")
    if not np.all(np.isfinite(quality)) or np.any((quality < 0) | (quality > 1)):
        raise ValueError("quality must be finite and in [0, 1].")
    if not np.isfinite(lam) or lam < 0:
        raise ValueError("lambda must be finite and non-negative.")
    cost_norm = normalise_cost(cost, cost_scale)

    if difficulty is None:
        difficulty = np.ones(quality.shape[0], dtype=np.float64)
    else:
        difficulty = np.asarray(difficulty, dtype=np.float64).reshape(-1)

    if len(difficulty) != quality.shape[0]:
        raise ValueError("One difficulty score is required per prompt.")
    if not np.all(np.isfinite(difficulty)):
        raise ValueError("Difficulty scores must be finite.")

    if np.any((difficulty < 1.0) | (difficulty > 5.0)):
        raise ValueError("Difficulty scores must be in [1, 5].")

    return quality / difficulty[:, None] - lam * cost_norm

class RoutingEnv:
    def __init__(self, R, disc, cont, pool_idx, episode_len=100, seed=0):
        self.R = R                    # (N, A) rewards for every (query, model)
        self.disc = disc              # (N,)   discrete state ids
        self.cont = cont              # (N, d) continuous feature vectors
        self.pool = np.asarray(pool_idx)
        self.L = episode_len
        self.rng = np.random.default_rng(seed)
        self.n_actions = R.shape[1]

    def _obs(self, i):
        return self.disc[i], self.cont[i]

    def reset(self):
        replace = len(self.pool) < self.L
        self.order = self.rng.choice(self.pool, size=self.L, replace=replace)
        self.t = 0
        return self._obs(self.order[0])

    def step(self, action):
        i = self.order[self.t]
        reward = float(self.R[i, action])
        self.t += 1
        done = self.t >= self.L
        next_obs = self._obs(self.order[self.t]) if not done else self._obs(i)
        return next_obs, reward, done, {"query_index": int(i)}
