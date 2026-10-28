"""Step 1: build the processed dataset (queries, per-model quality/cost, embeddings).

Example
    python -m src.prepare_data --pkl data/routerbench_0shot.pkl
"""
import argparse
import os

import numpy as np

from .config import MODEL_POOL, SEED
from .data import load_routerbench, save_data
from .features import embed_texts


def summarise(data):
    print(f"\nSource: {data.source} | queries: {len(data.prompts)} | "
          f"train/val/test = {[int((data.split == s).sum()) for s in ('train', 'val', 'test')]}")
    for j, m in enumerate(data.models):
        print(f"  {m:35s} mean quality {data.quality[:, j].mean():.3f} | "
              f"mean cost ${data.cost[:, j].mean() * 1000:.4f} per 1k queries")
    small_ok = (data.quality[:, 0] >= data.quality[:, -1]).mean()
    print(f"  small model is at least as good as large on {small_ok:.1%} of queries "
          "(this is the headroom a router can exploit)")
    vals, cnt = np.unique(data.tasks, return_counts=True)
    print("  tasks:", dict(zip(vals.tolist(), cnt.tolist())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, help="path to the real routerbench_0shot.pkl")
    ap.add_argument("--models", nargs="+", default=MODEL_POOL, help="cheap -> expensive")
    ap.add_argument("--max-samples", type=int, default=40000)
    ap.add_argument("--embedder", choices=["minilm", "hash"], default="minilm")
    ap.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Prepare data for handcrafted routing without downloading an embedder.",
    )
    ap.add_argument("--out", default="data/processed.joblib")
    args = ap.parse_args()

    try:
        data = load_routerbench(args.pkl, args.models, args.max_samples, SEED)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        ap.error(str(exc))

    if not args.no_embeddings:
        print(f"Embedding {len(data.prompts)} prompts with '{args.embedder}' ...")
        data.embeddings = embed_texts(data.prompts, args.embedder)
        data.embedder = args.embedder

    summarise(data)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_data(data, args.out)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
