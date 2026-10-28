"""Load a trained agent and use it to route a single new prompt."""
import json
import os

import joblib
import numpy as np


class LLMRouter:
    def __init__(self, models_dir="models", agent=None, state=None):
        with open(os.path.join(models_dir, "meta.json")) as fh:
            self.meta = json.load(fh)
        selected_agent, selected_state = self.meta.get("best_config", ["q", "handcrafted"])
        agent = selected_agent if agent is None else agent
        state = selected_state if state is None else state
        self.agent_name = agent
        self.state_name = state
        self.featurizer = joblib.load(os.path.join(models_dir, f"{state}_featurizer.joblib"))
        self.cost_predictor = joblib.load(
            os.path.join(models_dir, f"cost_predictor_{state}.joblib")
        )
        if agent == "q":
            from .agents.qlearning import QLearningAgent
            self.agent = QLearningAgent.load(os.path.join(models_dir, f"q_{state}.npz"))
        else:
            from .agents.dqn import DQNAgent
            self.agent = DQNAgent.load(os.path.join(models_dir, f"dqn_{state}.pt"))
        self.kind = agent
        self.cost_scale = float(self.meta["cost_scale"])
        self.budget_per_prompt = (
            float(self.meta["selection_budget_per_1k_usd"]) / 1000.0
        )
        calibration = self.meta.get("agent_calibrations", {}).get(
            f"{agent}_{state}", {}
        )
        self.dual_learning_rate = float(
            calibration.get(
                "dual_learning_rate",
                self.meta.get("dual_learning_rate", float("nan")),
            )
        )
        if (
            not np.isfinite(self.cost_scale)
            or self.cost_scale <= 0
            or not np.isfinite(self.budget_per_prompt)
            or self.budget_per_prompt <= 0
            or not np.isfinite(self.dual_learning_rate)
            or self.dual_learning_rate <= 0
        ):
            raise ValueError("Checkpoint dual-controller metadata is invalid.")
        self.dual_lambda = 0.0
        self._pending_action = None

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

        predicted_costs = np.asarray(
            self.cost_predictor.predict(cont), dtype=float
        ).reshape(-1)
        if (
            len(predicted_costs) != len(scores)
            or not np.all(np.isfinite(predicted_costs))
            or np.any(predicted_costs < 0)
        ):
            raise ValueError("Cost predictor returned invalid per-model estimates.")
        adjusted_scores = scores - (
            self.dual_lambda * predicted_costs / self.cost_scale
        )
        action = int(np.argmax(adjusted_scores))
        self._pending_action = action
        return action, adjusted_scores

    def observe_cost(self, actual_cost):
        """Update the dual price after the selected model's observed cost."""
        if self._pending_action is None:
            raise RuntimeError("Call route before reporting the selected model's cost.")
        if actual_cost is not None:
            actual_cost = float(actual_cost)
            if not np.isfinite(actual_cost) or actual_cost < 0:
                raise ValueError("Observed cost must be finite and non-negative.")
            budget_normalized = self.budget_per_prompt / self.cost_scale
            observed_normalized = actual_cost / self.cost_scale
            self.dual_lambda = max(
                0.0,
                self.dual_lambda
                + self.dual_learning_rate
                * (observed_normalized - budget_normalized),
            )
        self._pending_action = None
        return self.dual_lambda