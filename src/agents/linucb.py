"""Online LinUCB contextual-bandit agent."""
import numpy as np


class LinUCBAgent:
    kind = "linucb"

    def __init__(self, dim, n_actions, alpha=1.0, ridge=1.0, seed=0):
        if dim < 1 or n_actions < 2:
            raise ValueError("LinUCB requires dim >= 1 and at least two actions")
        self.dim, self.A, self.alpha, self.ridge = int(dim), int(n_actions), float(alpha), float(ridge)
        self.rng = np.random.default_rng(seed)
        self.A_inv = np.stack([np.eye(self.dim) / self.ridge for _ in range(self.A)])
        self.b = np.zeros((self.A, self.dim))
        self.t = 0

    def _x(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.dim:
            raise ValueError(f"Expected context dim {self.dim}, got {x.size}")
        return np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

    def scores(self, x):
        x = self._x(x)
        result = np.empty(self.A)
        for a in range(self.A):
            theta = self.A_inv[a] @ self.b[a]
            mean = float(theta @ x)
            uncertainty = float(np.sqrt(max(x @ self.A_inv[a] @ x, 0.0)))
            result[a] = mean + self.alpha * uncertainty
        return result

    def act(self, x, greedy=False):
        action = int(np.argmax(self.scores(x)))
        self.t += 1
        return action

    def observe(self, x, action, reward, next_x=None, done=True):
        x, action = self._x(x), int(action)
        inv = self.A_inv[action]
        Ax = inv @ x
        denom = 1.0 + float(x @ Ax)
        self.A_inv[action] = inv - np.outer(Ax, Ax) / max(denom, 1e-12)
        self.b[action] += float(reward) * x

    def predict(self, contexts):
        X = np.asarray(contexts, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.asarray([int(np.argmax(self.scores(x))) for x in X], dtype=int)

    def q_values(self, x):
        return self.scores(x)

    def save(self, path):
        np.savez(path, A_inv=self.A_inv, b=self.b, dim=self.dim,
                 n_actions=self.A, alpha=self.alpha, ridge=self.ridge, t=self.t)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        obj = cls(int(d["dim"]), int(d["n_actions"]),
                  alpha=float(d["alpha"]), ridge=float(d["ridge"]))
        obj.A_inv, obj.b, obj.t = d["A_inv"], d["b"], int(d["t"])
        return obj
