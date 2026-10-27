"""Tabular Q-learning (Watkins, 1989) with epsilon-greedy exploration."""
import numpy as np


class QLearningAgent:
    kind = "q"

    def __init__(self, n_states, n_actions, gamma=0.0, alpha_min=0.02,
                 eps_start=1.0, eps_end=0.05, eps_decay_steps=20000, seed=0):
        self.S, self.A = n_states, n_actions
        self.gamma = gamma
        self.alpha_min = alpha_min
        self.eps_start, self.eps_end, self.eps_decay_steps = eps_start, eps_end, eps_decay_steps
        self.Q = np.zeros((n_states, n_actions))
        self.N = np.zeros((n_states, n_actions))   # visit counts -> learning rate
        self.t = 0
        self.rng = np.random.default_rng(seed)

    def epsilon(self):
        frac = min(1.0, self.t / max(self.eps_decay_steps, 1))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, s, greedy=False):
        if not greedy:
            explore = self.rng.random() < self.epsilon()
            self.t += 1
            if explore:
                return int(self.rng.integers(self.A))
        return int(np.argmax(self.Q[s]))

    def observe(self, s, a, r, s2, done):
        """Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))."""
        self.N[s, a] += 1
        alpha = max(self.alpha_min, 1.0 / self.N[s, a])   # sample-average, floored
        target = r + (0.0 if done else self.gamma * self.Q[s2].max())
        self.Q[s, a] += alpha * (target - self.Q[s, a])

    # --- inference helpers -------------------------------------------------
    def predict(self, states):
        return np.argmax(self.Q[np.asarray(states)], axis=1)

    def q_values(self, s):
        return self.Q[s]

    def save(self, path):
        np.savez(path, Q=self.Q, N=self.N, gamma=self.gamma)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        agent = cls(d["Q"].shape[0], d["Q"].shape[1], gamma=float(d["gamma"]))
        agent.Q, agent.N = d["Q"], d["N"]
        return agent
