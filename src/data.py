"""Data layer.

Everything the RL environment needs is a table with one row per user query:

    prompt text | task name | quality of each LLM (0..1) | cost of each LLM (USD)

RouterBench (Hu et al., 2024) ships exactly this: every query was already sent
to ~11 LLMs and each answer was scored.  So "calling a model" during training
is just a table lookup -- fast, free and perfectly reproducible.
"""
import os
import pickle
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from .config import SPLIT_FRACTIONS


@dataclass
class RouterData:
    prompts: list            # list[str], length N
    tasks: np.ndarray        # (N,) task / benchmark name for each query
    quality: np.ndarray      # (N, A) score in [0, 1]; column j = model j
    cost: np.ndarray         # (N, A) USD cost of answering the query with model j
    models: list             # list[str], length A (cheap -> expensive)
    split: np.ndarray        # (N,) 'train' | 'val' | 'test'
    embeddings: np.ndarray = None   # (N, D) sentence embeddings (optional)
    embedder: str = ""              # which embedder produced them
    source: str = ""                # 'routerbench'

    def idx(self, name):
        """Indices of the rows belonging to a split."""
        return np.where(self.split == name)[0]


# --------------------------------------------------------------------------
# splits / io
# --------------------------------------------------------------------------
def assign_splits(n, seed, fractions=SPLIT_FRACTIONS):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_tr = int(fractions[0] * n)
    n_va = int(fractions[1] * n)
    split = np.empty(n, dtype=object)
    split[perm[:n_tr]] = "train"
    split[perm[n_tr:n_tr + n_va]] = "val"
    split[perm[n_tr + n_va:]] = "test"
    return split.astype(str)


def save_data(data, path):
    _validate_router_data(data, path, require_embeddings=False)
    joblib.dump(data, path)


def load_data(path, require_embeddings=True):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Processed dataset not found: {path}. "
            "Prepare real RouterBench data first with "
            "`python -m src.prepare_data --pkl <path-to-routerbench_0shot.pkl>`."
        )

    try:
        data = joblib.load(path)
    except (OSError, EOFError, ValueError, pickle.UnpicklingError,
            ImportError, AttributeError) as exc:
        raise ValueError(
            f"Could not load processed dataset '{path}': {exc}"
        ) from exc

    if not isinstance(data, RouterData):
        raise ValueError(
            f"Processed dataset '{path}' is not a RouterData object. "
            "Rebuild it from a valid RouterBench pickle with src.prepare_data."
        )
    _validate_router_data(data, path, require_embeddings=require_embeddings)
    return data


def _validate_router_data(data, context, require_embeddings=True):
    if data.source != "routerbench":
        raise ValueError(
            f"Dataset '{context}' has source '{data.source or 'unspecified'}'; "
            "only real RouterBench data is accepted. Rebuild it with "
            "`python -m src.prepare_data --pkl <path-to-routerbench_0shot.pkl>`."
        )

    n_prompts = len(data.prompts)
    if n_prompts == 0:
        raise ValueError(f"Dataset '{context}' contains no prompts.")
    if any(not isinstance(prompt, str) or not prompt.strip() for prompt in data.prompts):
        raise ValueError(f"Dataset '{context}' contains missing or empty prompts.")
    if not data.models or len(set(data.models)) != len(data.models):
        raise ValueError(f"Dataset '{context}' must list unique model names.")

    quality = np.asarray(data.quality)
    cost = np.asarray(data.cost)
    tasks = np.asarray(data.tasks)
    split = np.asarray(data.split)
    expected_shape = (n_prompts, len(data.models))
    if quality.shape != expected_shape:
        raise ValueError(
            f"Dataset '{context}' quality shape is {quality.shape}; "
            f"expected {expected_shape}."
        )
    if cost.shape != expected_shape:
        raise ValueError(
            f"Dataset '{context}' cost shape is {cost.shape}; "
            f"expected {expected_shape}."
        )
    if tasks.shape != (n_prompts,) or split.shape != (n_prompts,):
        raise ValueError(
            f"Dataset '{context}' must have one task and split label per prompt."
        )
    if not np.issubdtype(quality.dtype, np.number):
        raise ValueError(f"Dataset '{context}' quality scores must be numeric.")
    if not np.issubdtype(cost.dtype, np.number):
        raise ValueError(f"Dataset '{context}' costs must be numeric.")
    if not np.all(np.isfinite(quality)) or np.any((quality < 0) | (quality > 1)):
        raise ValueError(
            f"Dataset '{context}' contains invalid quality values; expected finite scores in [0, 1]."
        )
    if not np.all(np.isfinite(cost)) or np.any(cost < 0):
        raise ValueError(
            f"Dataset '{context}' contains invalid costs; expected finite non-negative values."
        )
    if not np.all(np.isin(split, ("train", "val", "test"))):
        raise ValueError(
            f"Dataset '{context}' has unrecognized split labels; expected train, val, and test."
        )
    for name in ("train", "val", "test"):
        if not np.any(split == name):
            raise ValueError(f"Dataset '{context}' has no rows in the '{name}' split.")
    if data.embeddings is None:
        if require_embeddings:
            raise ValueError(
                f"Dataset '{context}' has no prompt embeddings. Rebuild it with "
                "`python -m src.prepare_data --pkl <path-to-routerbench_0shot.pkl>`."
            )
        return
    embeddings = np.asarray(data.embeddings)
    if (
        embeddings.ndim != 2
        or embeddings.shape[0] != n_prompts
        or embeddings.shape[1] != 384
        or not np.all(np.isfinite(embeddings))
    ):
        raise ValueError(
            f"Dataset '{context}' must contain one finite 384-dimensional "
            "embedding vector per prompt."
        )


# --------------------------------------------------------------------------
# RouterBench
# --------------------------------------------------------------------------
def _to_text(p):
    if isinstance(p, (list, tuple, np.ndarray)):
        return "\n".join(str(x) for x in p)
    return str(p)


def load_routerbench(pkl_path, models, max_samples=None, seed=0):
    """Read RouterBench's `routerbench_0shot.pkl` and keep the chosen models.

    RouterBench columns (per the dataset card):
        prompt, eval_name, <model>            -> quality score of that model
        <model>|total_cost                    -> USD cost of that model's answer
    """
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"RouterBench pickle not found: {pkl_path}")
    if len(models) < 2 or len(set(models)) != len(models):
        raise ValueError("Specify at least two unique RouterBench model columns.")
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive when specified.")

    try:
        df = pd.read_pickle(pkl_path)
    except (OSError, EOFError, ValueError, pickle.UnpicklingError,
            ImportError, AttributeError) as exc:
        raise ValueError(
            f"Could not read RouterBench pickle '{pkl_path}': {exc}"
        ) from exc
    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            f"RouterBench file '{pkl_path}' did not contain a pandas DataFrame."
        )

    need = ["prompt"] + list(models) + [f"{m}|total_cost" for m in models]
    missing = [c for c in need if c not in df.columns]
    if missing:
        score_cols = sorted(c for c in df.columns if "|" not in c)
        raise KeyError(
            f"Columns not found in the pickle: {missing}\n"
            f"Columns that look like model score columns: {score_cols}\n"
            "Edit MODEL_POOL in src/config.py (or pass --models) to match."
        )

    df = df.dropna(subset=need).reset_index(drop=True)
    if df.empty:
        raise ValueError(
            f"RouterBench file '{pkl_path}' has no rows with complete prompts, "
            "quality scores, and costs for the requested models."
        )
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)

    try:
        quality = df[list(models)].to_numpy(dtype=np.float64)
        cost = df[[f"{m}|total_cost" for m in models]].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"RouterBench file '{pkl_path}' has non-numeric quality or cost "
            "values for the requested models."
        ) from exc
    if not np.all(np.isfinite(quality)) or np.any((quality < 0) | (quality > 1)):
        raise ValueError(
            f"RouterBench file '{pkl_path}' has non-finite quality scores or "
            "scores outside [0, 1] for the requested models."
        )
    if not np.all(np.isfinite(cost)) or np.any(cost < 0):
        raise ValueError(
            f"RouterBench file '{pkl_path}' has non-finite or negative costs "
            "for the requested models."
        )
    tasks = (df["eval_name"].astype(str).to_numpy()
             if "eval_name" in df.columns else np.array(["unknown"] * len(df)))
    prompts = [_to_text(p) for p in df["prompt"]]
    if any(not prompt.strip() for prompt in prompts):
        raise ValueError(
            f"RouterBench file '{pkl_path}' contains empty prompts after filtering."
        )

    data = RouterData(
        prompts=prompts, tasks=tasks, quality=quality, cost=cost,
        models=list(models), split=assign_splits(len(df), seed), source="routerbench",
    )
    _validate_router_data(data, pkl_path, require_embeddings=False)
    return data
