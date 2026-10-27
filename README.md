# LLM Router: Reinforcement Learning for Adaptive Model Selection

An RL agent that reads a user query and decides **which LLM should answer it** — a cheap
small model, a medium model, or an expensive large one — so we keep answer quality high while cutting API cost.

| RL element | Our design |
|---|---|
| **State** | Features of the incoming query (two variants, see below) |
| **Action** | Discrete: `0 = small`, `1 = medium`, `2 = large` (ordered by cost) |
| **Reward** | `difficulty_weight · quality − λ · normalised_cost` |
| **Algorithms** | Tabular **Q-learning** and **DQN** |
| **Data** | [RouterBench](https://huggingface.co/datasets/withmartian/routerbench) (real queries, real per-model quality scores and costs) |

> **Team:** _fill in names_  |  **Course:** AI Assignment 2 (Mini Project)

---

## 1. How it works (in plain words)

1. A query arrives, e.g. *"Solve step by step: 3x + 7 = 31"*.
2. We turn it into a **state** (is it long? does it contain digits, code, "prove/explain"?…).
3. The agent picks an **action**: send it to the small, medium, or large model.
4. We observe the **reward**: how good the answer was, minus a penalty for how expensive it was.
5. Q-learning / DQN updates its value estimates so that, over time, easy queries go to the cheap
   model and only hard ones go to a more capable, expensive model.

`λ` controls the trade-off: `λ = 0` means "quality only" (always pick the best model);
large `λ` means "cost matters a lot". The deployment default remains `λ = 1`;
rerun the validation sweep after changing the reward formula before finalizing a
deployment setting. The sweep reports the trade-off but does not choose a
deployment lambda by comparing raw rewards across different lambdas: each lambda
defines a different objective.

### RL formulation

* **State space S.** Two alternatives (our ablation):
  * *Handcrafted* — 11 numbers: log-chars, log-words, digit ratio, math-symbol ratio, has-code,
    #question-marks, reasoning keywords, math keywords, multiple-choice format, avg word length, #lines.
    Tabular agent sees them binned into **192 discrete states**; DQN sees the 11-d standardised vector.
  * *Embedding* — 384-d MiniLM sentence embedding. Tabular agent sees the **K-means cluster id (K=32)**;
    DQN sees the full 384-d vector.
* **Action space A.** `{0: small, 1: medium, 2: large}` (tiers are ordered cheap to expensive in `MODEL_POOL`).
* **Reward R.** `r = quality(q, a) · [1 + α · (d(q) − 1)] − λ · min(1, cost(q, a) / c_scale)`,
  where `quality ∈ [0,1]` is the RouterBench score, `d(q) ∈ [1,2]` is a prompt-only structural
  complexity proxy, `α` is `--difficulty-strength` (default 1), `cost` is USD cost, and `c_scale`
  is the 99th percentile of the large model's cost on the training split. This proxy measures
  structure such as length, code, and math notation; it is not a ground-truth difficulty label.
  At `α = 0`, the original quality-cost objective is recovered.
* **Environment.** An episode is a stream of 100 queries sampled from the training split. Since
  RouterBench already contains every model's answer to every query, a step is a table lookup.
* **Honest note.** The next query does not depend on the chosen model, so this is a one-step decision
  problem (a contextual bandit — a special case of an MDP). We keep the full Q-learning / DQN update
  with `γ` as a hyper-parameter (default `γ = 0`).
* **Baselines.** always-small, always-large, random, a query-length heuristic, and an *oracle*
  upper bound that uses each test query's outcomes. The oracle is not directly attainable by a
  prompt-only router.

---

## 2. Repository layout

```
src/
  config.py          constants (model pool, λ sweep, seeds)
  data.py            RouterBench loader and train/val/test split
  features.py        handcrafted + embedding featurizers (the "state")
  env.py             RoutingEnv + reward function
  agents/
    qlearning.py     tabular Q-learning
    dqn.py           DQN (replay buffer + target network)
  train.py           training loop shared by both agents
  evaluate.py        metrics + baseline policies
  plots.py           figures for the report
  prepare_data.py    step 1: build processed dataset + embeddings
  run_experiments.py step 2: train, evaluate, tables, figures, save agents
  router.py          load a trained agent and route one prompt
  demo.py            live demo with real LLM APIs
tests/test_smoke.py  end-to-end and invalid-data checks using local fixtures
report/report.tex    2-page two-column report (Overleaf-ready)
```

---

## 3. Installation

```bash
git clone <your-repo-url> && cd llm-router-rl
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Python 3.10+ recommended. A GPU is **not** needed (everything runs on CPU).

## 4. Get the data (RouterBench, free)

```bash
python - <<'PY'
from huggingface_hub import hf_hub_download
p = hf_hub_download(repo_id="withmartian/routerbench", filename="routerbench_0shot.pkl",
                    repo_type="dataset", local_dir="data")
print(p)
PY
```
(or download `routerbench_0shot.pkl` manually from the dataset page and put it in `data/`).

Check the RouterBench model column names once — `src/config.py` uses
`mistralai/mistral-7b-chat` (small), `mistralai/mixtral-8x7b-chat` (medium), and
`gpt-4-1106-preview` (large). If the loader reports
missing columns it prints the available ones; edit `MODEL_POOL` accordingly.

These are offline benchmark IDs and must match the RouterBench score/cost columns.
The live API model IDs are configured separately in `.env` with
`SMALL_MODEL_NAME`, `MEDIUM_MODEL_NAME`, and `LARGE_MODEL_NAME`; changing those does
not change which models the trained RL agent learned from.

## 5. Reproduce all results

```bash
# 1) build processed dataset + MiniLM embeddings for embedding/hybrid experiments
python -m src.prepare_data --pkl data/routerbench_0shot.pkl

# For handcrafted-only experiments, skip embedding download and inference:
python -m src.prepare_data --pkl data/routerbench_0shot.pkl --no-embeddings

# 2) train 4 agents x 3 seeds, lambda sweep, baselines, figures  (roughly 15-30 min on CPU; DQN dominates the time)
python -m src.run_experiments

# Override the deployment reward objective explicitly:
python -m src.run_experiments --lam 1.0

# optional: tiny run to verify the setup
python -m src.run_experiments --quick
```
GPT/final-action penalties are selected only on validation reward and applied to test
predictions. Rewards use a prompt-only structural complexity multiplier; set
`--difficulty-strength 0` to disable it and recover `quality - lambda * normalized_cost`.
When embeddings are absent, experiments automatically run handcrafted configurations
only. Outputs include `results/gpt_penalty_calibration.csv`,
`results/routing_diagnostics.csv`, `results/reward_margin_diagnostics.csv`,
`results/metrics_main.csv`, `results/metrics_sweep.csv`,
`results/figures/*.pdf|png`, and trained agents in `models/`.
All randomness is seeded (`src/config.py`), so the numbers reproduce.

Tests use small local fixtures and do not generate or substitute synthetic training
data. Experiment commands require a valid processed RouterBench dataset; if it is
missing or invalid, the loader reports an error instead of generating replacement data.

## 6. Live demo (real API calls)

```bash
cp .env.example .env     # then fill in your API key, base URL, three model names and their prices
python -m src.demo --dry-run                 # routing decisions only (no key needed)
python -m src.demo --compare                 # real calls; shows cost saved vs always-large
python -m src.demo --query "Prove that sqrt(2) is irrational" --agent dqn --state embedding
```
Any OpenAI-compatible provider works (OpenAI, Groq, Together AI, OpenRouter, …).
Prices are in USD per 1M tokens, from your provider's pricing page.

## 7. Results

_Paste the table from `results/metrics_main.csv` here after running (test split, mean over 3 seeds)._

## 8. Originality & references

All code in `src/` was written from scratch for this project. External resources used as *data/models only*:
RouterBench (Hu et al., 2024), `all-MiniLM-L6-v2` (Wang et al., 2020). Algorithms follow
Watkins & Dayan (1992) and Mnih et al. (2015).

## 9. Limitations

* Quality labels come from RouterBench's automatic scoring, not human judgements.
* Offline training on logged outcomes; the live demo only checks that the router transfers to
  real API calls and does not re-measure quality with a judge.
* The decision is one-step; sequential effects (budgets, conversation context) are future work.
#   L L M _ R o u t e r  
 