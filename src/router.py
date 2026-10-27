"""Load a trained agent and use it to route a single new prompt."""
import json
import os

import joblib
import numpy as np

from .train import actions_from_scores


class LLMRouter:
    def __init__(self, models_dir="models", agent="q", state="handcrafted"):
        with open(os.path.join(models_dir, "meta.json")) as fh:
            self.meta = json.load(fh)
        self.featurizer = joblib.load(os.path.join(models_dir, f"{state}_featurizer.joblib"))
        if agent == "q":
            from .agents.qlearning import QLearningAgent
            self.agent = QLearningAgent.load(os.path.join(models_dir, f"q_{state}.npz"))
        else:
            from .agents.dqn import DQNAgent
            self.agent = DQNAgent.load(os.path.join(models_dir, f"dqn_{state}.pt"))
        self.kind = agent
        calibration = self.meta.get("agent_calibrations", {}).get(
            f"{agent}_{state}", {}
        )
        self.final_action_penalty = float(
            calibration.get("gpt_penalty", self.meta.get("gpt_penalty", 0.0))
        )
        if not np.isfinite(self.final_action_penalty) or self.final_action_penalty < 0:
            raise ValueError("Checkpoint final-action penalty must be finite and non-negative.")

    def route(self, prompt):
        """Return (action_index, learned_action_scores)."""
        disc, cont = self.featurizer.transform([prompt])

        if self.kind == "q":
            x = int(disc[0])
        else:
            x = cont[0]

        scores = np.asarray(self.agent.q_values(x), dtype=float).reshape(-1)

        if len(scores) != len(self.meta["models"]):
            raise ValueError("Checkpoint action count does not match model metadata.")
        if not np.all(np.isfinite(scores)):
            raise ValueError("Agent returned non-finite action scores.")

        action = int(actions_from_scores(scores, self.final_action_penalty))
        adjusted_scores = scores.copy()
        adjusted_scores[-1] -= self.final_action_penalty
        return action, adjusted_scores