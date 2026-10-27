"""Deep Q-Network (Mnih et al., 2015): MLP Q-function, replay buffer, target network."""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim, n_actions, hidden):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, n_actions),
    )


class DQNAgent:
    kind = "dqn"

    def __init__(self, in_dim, n_actions, hidden=64, lr=1e-3, gamma=0.0,
                 buffer_size=50000, batch_size=64, target_every=500, learn_every=2,
                 warmup=500, eps_start=1.0, eps_end=0.05, eps_decay_steps=20000, seed=0):
        torch.manual_seed(seed)
        self.in_dim, self.A, self.hidden = in_dim, n_actions, hidden
        self.gamma, self.batch_size = gamma, batch_size
        self.target_every, self.learn_every, self.warmup = target_every, learn_every, warmup
        self.eps_start, self.eps_end, self.eps_decay_steps = eps_start, eps_end, eps_decay_steps
        self.rng = np.random.default_rng(seed)

        self.q = _mlp(in_dim, n_actions, hidden)
        self.target = copy.deepcopy(self.q)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)

        # replay buffer (ring buffer of numpy arrays)
        self.cap, self.size, self.ptr = buffer_size, 0, 0
        self.bS = np.zeros((buffer_size, in_dim), dtype=np.float32)
        self.bS2 = np.zeros((buffer_size, in_dim), dtype=np.float32)
        self.bA = np.zeros(buffer_size, dtype=np.int64)
        self.bR = np.zeros(buffer_size, dtype=np.float32)
        self.bD = np.zeros(buffer_size, dtype=np.float32)
        self.t = 0

    def epsilon(self):
        frac = min(1.0, self.t / max(self.eps_decay_steps, 1))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, x, greedy=False):
        if not greedy:
            explore = self.rng.random() < self.epsilon()
            self.t += 1
            if explore:
                return int(self.rng.integers(self.A))
        return int(np.argmax(self.q_values(x)))

    def q_values(self, x):
        x = np.asarray(x, dtype=np.float32)
        is_single = x.ndim == 1
        inputs = torch.as_tensor(x)
        if is_single:
            inputs = inputs.unsqueeze(0)
        with torch.no_grad():
            values = self.q(inputs)
        return values[0].numpy() if is_single else values.numpy()

    def observe(self, s, a, r, s2, done):
        i = self.ptr
        self.bS[i], self.bA[i], self.bR[i], self.bS2[i], self.bD[i] = s, a, r, s2, float(done)
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)
        if self.t % self.learn_every == 0:
            self._learn()
        if self.t % self.target_every == 0:
            self.target.load_state_dict(self.q.state_dict())

    def _learn(self):
        if self.size < max(self.warmup, self.batch_size):
            return
        idx = self.rng.integers(0, self.size, self.batch_size)
        s = torch.from_numpy(self.bS[idx]); s2 = torch.from_numpy(self.bS2[idx])
        a = torch.from_numpy(self.bA[idx]); r = torch.from_numpy(self.bR[idx])
        d = torch.from_numpy(self.bD[idx])
        q_sa = self.q(s).gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            y = r + self.gamma * (1.0 - d) * self.target(s2).max(1).values
        loss = F.smooth_l1_loss(q_sa, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    # --- inference helpers -------------------------------------------------
    def predict(self, X):
        with torch.no_grad():
            return self.q(torch.as_tensor(np.asarray(X, dtype=np.float32))).argmax(1).numpy()

    def save(self, path):
        torch.save({"state": self.q.state_dict(), "in_dim": self.in_dim,
                    "A": self.A, "hidden": self.hidden, "gamma": self.gamma}, path)

    @classmethod
    def load(cls, path):
        ck = torch.load(path, map_location="cpu")
        agent = cls(ck["in_dim"], ck["A"], hidden=ck["hidden"], gamma=ck["gamma"])
        agent.q.load_state_dict(ck["state"])
        return agent
