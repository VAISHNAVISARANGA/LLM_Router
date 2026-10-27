"""
PLUG-IN POINTS  --  this is the only file you need to edit to connect your own RL agent.

The UI (llm_router_dashboard.py) calls exactly three things:

    TIERS                       -> static description of the model pool
    route_prompt(prompt)        -> RoutingDecision   (RL agent: scores + prompt features)
    call_model(tier, prompt)    -> ModelResult       (one LLM API call)

Everything here is synchronous; the UI runs it in a worker thread.
By default the file runs in MOCK mode (no keys, no trained agent needed) so you can
see the dashboard working immediately.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------
# 1. Model pool (action i == TIERS[i])
# --------------------------------------------------------------------------
TIERS = [
    {"tier": "SMALL",  "color": "blue",   "hex": "#60a5fa", "params": "~2.6B",
     "model_id": os.getenv("SMALL_MODEL_NAME") or "liquid/lfm-2.5-2.6b:free",  "env": "SMALL"},
    {"tier": "MEDIUM", "color": "violet", "hex": "#8b5cf6", "params": "~14B",
     "model_id": os.getenv("MEDIUM_MODEL_NAME") or "nvidia/nemotron-3.5-lightning:free", "env": "MEDIUM"},
    {"tier": "LARGE",  "color": "green",  "hex": "#34d399", "params": "~550B",
     "model_id": os.getenv("LARGE_MODEL_NAME") or "nvidia/nemotron-3-ultra-550b-a55b:free", "env": "LARGE"},
]

API_KEY = os.getenv("ROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")
API_BASE_URL = os.getenv("ROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
MOCK_MODE = not API_KEY          # no key -> everything below returns fake data


def is_connected() -> bool:
    return bool(API_KEY)


# --------------------------------------------------------------------------
# 2. RL agent hook
# --------------------------------------------------------------------------
@dataclass
class RoutingDecision:
    scores: list[float]                     # learned action values, len == len(TIERS)
    analysis: dict[str, str] = field(default_factory=dict)   # e.g. {"Length": "Medium", "Math": "Yes"}

    @property
    def action(self) -> int:
        return max(range(len(self.scores)), key=lambda i: self.scores[i])


def _analyse(prompt: str) -> dict[str, str]:
    """Human-readable prompt analysis shown in the 'Prompt Analysis' card."""
    low = prompt.lower()
    n = len(prompt)
    yn = lambda b: "Yes" if b else "No"
    return {
        "Length": "Short" if n < 60 else "Medium" if n < 250 else "Long",
        "Math": yn(any(k in low for k in ("derive", "solve", "calculate", "equation", "prove", "integral"))),
        "Reasoning": yn(any(k in low for k in ("explain", "why", "prove", "derive", "compare", "justify"))),
        "Code": yn(any(k in low for k in ("def ", "python", "function", "```", "code"))),
        "Choices": yn("\nA." in prompt or "(A)" in prompt),
    }


def route_prompt(prompt: str) -> RoutingDecision:
    """TODO: replace the mock with your trained agent.

    --- Option A: use the agent from the llm-router-rl repo --------------------
        from src.router import LLMRouter            # pip install -e ../llm-router-rl  (or add to PYTHONPATH)
        _ROUTER = LLMRouter("../llm-router-rl/models", agent="q", state="handcrafted")
        action, q_values = _ROUTER.route(prompt)
        return RoutingDecision(scores=[float(v) for v in q_values], analysis=_analyse(prompt))
        # Keep RouterBench column IDs in src/config.py aligned with the offline data.
        # Live provider model IDs are configured separately with *_MODEL_NAME in .env.
    ---------------------------------------------------------------------------
    """
    analysis = _analyse(prompt)
    # MOCK: deterministic fake Q-values that grow with prompt "difficulty"
    hard = sum(v == "Yes" for k, v in analysis.items() if k in ("Math", "Reasoning", "Code")) / 3
    jitter = int(hashlib.md5(prompt.encode()).hexdigest()[:4], 16) / 65535 * 0.1
    scores = [0.0, round(-0.08 + 0.45 * hard + jitter, 5), round(-0.95 + 1.5 * hard, 5)]
    return RoutingDecision(scores=scores, analysis=analysis)


# --------------------------------------------------------------------------
# 3. LLM call hook
# --------------------------------------------------------------------------
@dataclass
class ModelResult:
    text: str = ""
    error: str = ""
    tokens: int = 0
    cost: float = 0.0


def call_model(tier: int, prompt: str, max_tokens: int = 400) -> ModelResult:
    """Call one model through OpenRouter. Never raises: errors become `ModelResult.error`
    so the UI can show the red 'API Error' card."""
    t = TIERS[tier]
    if MOCK_MODE:
        return ModelResult(
            text=(f"*(mock answer from **{t['tier']}**)*\n\nThe Bellman optimality equation gives\n\n"
                  r"$$V^*(s)=\max_{a}\Big[R(s,a)+\gamma\sum_{s'}P(s'|s,a)V^*(s')\Big]$$" "\n\n"
                  "```python\nq_target = r + gamma * q_next.max(dim=1).values\n```"),
            tokens=120 + 60 * tier, cost=0.0)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
        r = client.chat.completions.create(
            model=t["model_id"], max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}])
        u = r.usage
        p_in = float(os.getenv(f"{t['env']}_PRICE_IN", 0) or 0)
        p_out = float(os.getenv(f"{t['env']}_PRICE_OUT", 0) or 0)
        cost = (u.prompt_tokens * p_in + u.completion_tokens * p_out) / 1e6
        return ModelResult(text=r.choices[0].message.content or "", tokens=u.total_tokens, cost=cost)
    except Exception as e:                      # noqa: BLE001  (surface any provider error in the UI)
        return ModelResult(error=str(e)[:400])
