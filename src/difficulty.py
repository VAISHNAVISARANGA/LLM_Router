
"""Learn prompt difficulty from existing RouterBench training data.

Labels are derived from model-quality gaps:
    gap = best_model_quality - small_model_quality
    difficulty = 1 + 4 * gap

This is a routing-oriented difficulty proxy, not ground-truth difficulty.
Only training-split outcomes may be used to fit the estimator.
"""
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import normalize

from .features import embed_texts


class DifficultyEstimator:
    """KNN regression over sentence embeddings."""

    def __init__(self, k=5, embedder="minilm"):
        self.k = k
        self.embedder = embedder
        self.model = None
        self.train_embeddings = None
        self.train_scores = None

    @staticmethod
    def make_labels(quality):
        """Convert quality outcomes into difficulty labels in [1, 5].

        quality has shape (n_prompts, n_models), ordered cheapest first.
        """
        quality = np.asarray(quality, dtype=np.float32)

        if quality.ndim != 2 or quality.shape[1] < 2:
            raise ValueError("quality must have shape (N, at least 2 models).")
        if not np.all(np.isfinite(quality)):
            raise ValueError("Quality scores must be finite.")

        # Larger gap means SMALL loses more quality relative to the best model.
        gap = np.max(quality, axis=1) - quality[:, 0]
        gap = np.clip(gap, 0.0, 1.0)

        return (1.0 + 4.0 * gap).astype(np.float32)

    def fit(self, prompts, quality, embeddings=None):
        """Fit using TRAIN prompts and their quality outcomes only."""
        prompts = list(prompts)
        scores = self.make_labels(quality)

        if len(prompts) != len(scores) or len(prompts) == 0:
            raise ValueError("Provide one prompt per quality row.")

        if embeddings is None:
            embeddings = embed_texts(prompts, self.embedder)

        X = normalize(np.asarray(embeddings, dtype=np.float32))

        if X.ndim != 2 or X.shape[0] != len(prompts):
            raise ValueError("Embeddings must have one row per prompt.")
        if not np.all(np.isfinite(X)):
            raise ValueError("Embeddings must be finite.")

        self.train_embeddings = X
        self.train_scores = scores

        self.model = KNeighborsRegressor(
            n_neighbors=min(self.k, len(prompts)),
            weights="distance",
            metric="cosine",
        )
        self.model.fit(X, scores)
        return self

    def predict(self, prompts, embeddings=None):
        """Predict difficulty without accessing prompt quality outcomes."""
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")

        prompts = list(prompts)
        if not prompts:
            return np.empty(0, dtype=np.float32)

        if embeddings is None:
            embeddings = embed_texts(prompts, self.embedder)

        X = normalize(np.asarray(embeddings, dtype=np.float32))

        if X.ndim != 2 or X.shape[0] != len(prompts):
            raise ValueError("Embeddings must have one row per prompt.")
        if X.shape[1] != self.train_embeddings.shape[1]:
            raise ValueError("Embedding dimension differs from training data.")

        scores = self.model.predict(X)
        return np.clip(scores, 1.0, 5.0).astype(np.float32)
