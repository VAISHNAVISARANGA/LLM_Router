"""Turning a query string into the RL *state*.

Two alternative state representations (this is our ablation):

1. HandcraftedFeaturizer -- 11 interpretable numbers (length, digits, code, ...).
   * tabular view : 192 discrete states (binned features)   -> Q-learning
   * vector view  : 11 standardised floats                    -> DQN

2. EmbeddingFeaturizer -- a sentence embedding (MiniLM, 384-d).
   * tabular view : K-means cluster id (K = 32 states)        -> Q-learning
   * vector view  : the raw 384-d embedding                   -> DQN

Both expose `fit(...)` (learn bin edges / clusters from TRAIN data only) and
`transform(...)` which returns `(discrete_state_ids, continuous_vectors)`.
"""
import re
import zlib

import numpy as np
from sklearn.cluster import KMeans

import re


def extract_structural_features(prompt: str) -> np.ndarray:
    """
    Extract inexpensive structural proxies for prompt difficulty.

    These features describe the prompt's structure, not its true difficulty.
    No external API or additional model is required.
    """
    prompt = str(prompt or "")

    # Basic size
    char_count = len(prompt)
    word_count = len(prompt.split())
    line_count = len(prompt.splitlines())

    # Code structure
    code_block_count = len(
        re.findall(r"```[\s\S]*?```|~~~[\s\S]*?~~~", prompt)
    )
    inline_code_count = len(re.findall(r"`[^`\n]+`", prompt))

    # Mathematical structure
    math_block_count = len(
        re.findall(r"\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]", prompt)
    )
    inline_math_count = len(
        re.findall(r"(?<!\$)\$(?!\$)[^$\n]+\$(?!\$)", prompt)
    )
    math_symbol_count = len(
        re.findall(
            r"\\(?:frac|sum|int|sqrt|prod|lim|begin|end|"
            r"alpha|beta|theta|lambda|infty)\b"
            r"|[=<>±×÷∑∫√πΣΔλθ]",
            prompt,
        )
    )

    # Other structural indicators
    has_code = int(code_block_count > 0 or inline_code_count > 0)
    has_math = int(
        math_block_count > 0
        or inline_math_count > 0
        or math_symbol_count > 0
    )

    return np.array(
        [
            word_count,
            char_count,
            line_count,
            code_block_count,
            inline_code_count,
            math_block_count,
            inline_math_count,
            math_symbol_count,
            has_code,
            has_math,
        ],
        dtype=np.float32,
    )      

class HybridFeaturizer:
    """Combine handcrafted features and MiniLM semantic embeddings."""

    name = "hybrid"

    def __init__(self, embedder="minilm", n_clusters=32, seed=0):
        self.hand = HandcraftedFeaturizer()
        self.embedding = EmbeddingFeaturizer(
            embedder=embedder,
            n_clusters=n_clusters,
            seed=seed,
        )

    @property
    def n_states(self):
        # Each handcrafted state is split by its embedding cluster.
        return self.hand.n_states * self.embedding.n_states

    @property
    def dim(self):
        # 11 standardized handcrafted features + 384 embedding values.
        return self.hand.dim + self.embedding.dim

    def fit(self, prompts=None, emb=None):
        self.hand.fit(prompts=prompts)
        self.embedding.fit(prompts=prompts, emb=emb)
        return self

    def transform(self, prompts=None, emb=None):
        hand_disc, hand_cont = self.hand.transform(prompts=prompts)
        embedding_disc, embedding_cont = self.embedding.transform(
            prompts=prompts, emb=emb
        )

        # Discrete representation for tabular Q-learning.
        hybrid_disc = (
            hand_disc * self.embedding.n_states + embedding_disc
        )

        # Continuous representation for DQN.
        hybrid_cont = np.concatenate(
            [hand_cont, embedding_cont], axis=1
        ).astype(np.float32)

        return hybrid_disc.astype(int), hybrid_cont


def calculate_difficulty_scores(prompts):
    """
    Compute a simple structural difficulty proxy for each prompt.

    Returns scores from 1.0 to 2.0.
    Higher scores indicate more structural complexity,
    not necessarily genuinely harder questions.
    """
    structural = np.vstack([
        extract_structural_features(prompt)
        for prompt in prompts
    ])

    # Feature indices from extract_structural_features():
    # 0: word count
    # 3: code block count
    # 5: math block count
    # 7: math symbol count
    # 8: has code
    # 9: has math

    word_count = structural[:, 0]
    code_blocks = structural[:, 3]
    math_blocks = structural[:, 5]
    math_symbols = structural[:, 7]
    has_code = structural[:, 8]
    has_math = structural[:, 9]

    # Bound each component so long prompts don't dominate.
    length_score = np.clip(
        np.log1p(word_count) / np.log1p(1000), 0.0, 1.0
    )
    code_score = np.clip(code_blocks / 3.0, 0.0, 1.0)
    math_score = np.clip(
        (math_blocks + math_symbols / 5.0) / 3.0,
        0.0,
        1.0,
    )

    complexity = (
        0.30 * length_score
        + 0.25 * has_code
        + 0.25 * has_math
        + 0.05 * code_score
        + 0.05 * math_score
    )

    return 1.0 + complexity
# --------------------------------------------------------------------------
# Handcrafted features
# --------------------------------------------------------------------------
_CODE_RE = re.compile(r"```|\bdef \w+\(|\bimport \w+|\breturn\b|\bclass \w+|#include|public static|=>|\bprint\(")
_CHOICE_RE = re.compile(r"(^|\n)\s*\(?[A-D][\.\):]\s")
_REASON_KW = ("explain", "why", "prove", "derive", "analyze", "analyse", "compare",
              "justify", "step by step", "reason", "evaluate", "discuss")
_MATH_KW = ("calculate", "solve", "equation", "integral", "derivative", "probability",
            "how many", "sum of", "compute", "simplify", "find the value")

FEATURE_NAMES = [
    "log_chars", "log_words", "digit_ratio", "math_symbol_ratio", "has_code",
    "n_question_marks", "has_reasoning_kw", "has_math_kw", "has_choices",
    "avg_word_len", "log_lines",
]


def raw_features(text):
    """11 cheap, interpretable signals of how 'hard' a query might be."""
    low = text.lower()
    n_chars = max(len(text), 1)
    words = text.split()
    n_words = max(len(words), 1)
    digits = sum(c.isdigit() for c in text)
    math_syms = sum(c in "+-*/=^<>%" for c in text)
    return np.array([
        np.log1p(n_chars),
        np.log1p(n_words),
        digits / n_chars,
        math_syms / n_chars,
        float(bool(_CODE_RE.search(text))),
        float(min(text.count("?"), 5)),
        float(any(k in low for k in _REASON_KW)),
        float(any(k in low for k in _MATH_KW)),
        float(bool(_CHOICE_RE.search(text))),
        sum(len(w) for w in words) / n_words,
        np.log1p(text.count("\n")),
    ], dtype=np.float32)


class HandcraftedFeaturizer:
    name = "handcrafted"
    N_STATES = 4 * 3 * 2 * 2 * 2 * 2     # length x digits x code x reasoning x math x choices
    dim = len(FEATURE_NAMES)

    @property
    def n_states(self):
        return self.N_STATES

    @staticmethod
    def _raw(prompts):
        return np.stack([raw_features(p) for p in prompts])

    def fit(self, prompts=None, emb=None):
        X = self._raw(prompts)
        self.len_edges = np.quantile(X[:, 0], [0.25, 0.5, 0.75])   # 4 length bins
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-6
        return self

    def transform(self, prompts=None, emb=None):
        X = self._raw(prompts)
        length = np.searchsorted(self.len_edges, X[:, 0])           # 0..3
        digits = np.digitize(X[:, 2], [0.01, 0.06])                  # 0..2
        code = (X[:, 4] > 0).astype(int)
        reason = (X[:, 6] > 0).astype(int)
        math = (X[:, 7] > 0).astype(int)
        choice = (X[:, 8] > 0).astype(int)
        disc = ((((length * 3 + digits) * 2 + code) * 2 + reason) * 2 + math) * 2 + choice
        cont = ((X - self.mu) / self.sd).astype(np.float32)
        return disc.astype(int), cont


# --------------------------------------------------------------------------
# Embedding features
# --------------------------------------------------------------------------
EMBED_DIM = 384
_ST_MODEL = {}


def _hash_embed(text, dim=EMBED_DIM):
    """Cheap offline stand-in for MiniLM (hashing trick). For smoke tests only."""
    v = np.zeros(dim, dtype=np.float32)
    for tok in re.findall(r"\w+", text.lower()):
        h = zlib.crc32(tok.encode())
        sign = 1.0 if zlib.crc32((tok + "#").encode()) % 2 == 0 else -1.0
        v[h % dim] += sign
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def embed_texts(texts, kind="minilm", batch_size=64):
    """Return an (N, 384) float32 array of L2-normalised embeddings."""
    if kind == "hash":
        return np.stack([_hash_embed(t) for t in texts]).astype(np.float32)
    if kind == "minilm":
        from sentence_transformers import SentenceTransformer
        if "m" not in _ST_MODEL:
            _ST_MODEL["m"] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            _ST_MODEL["m"].max_seq_length = 256
        out = _ST_MODEL["m"].encode(
            list(texts), batch_size=batch_size, normalize_embeddings=True,
            show_progress_bar=len(texts) > 200)
        return np.asarray(out, dtype=np.float32)
    raise ValueError(f"unknown embedder '{kind}'")


class EmbeddingFeaturizer:
    name = "embedding"
    dim = EMBED_DIM

    def __init__(self, embedder="minilm", n_clusters=32, seed=0):
        self.embedder = embedder
        self.n_clusters = n_clusters
        self.seed = seed

    @property
    def n_states(self):
        return self.n_clusters

    def fit(self, prompts=None, emb=None):
        if emb is None:
            emb = embed_texts(prompts, self.embedder)
        self.km = KMeans(self.n_clusters, n_init=10, random_state=self.seed).fit(
            np.asarray(emb, dtype=np.float32))
        return self

    def transform(self, prompts=None, emb=None):
        if emb is None:
            emb = embed_texts(prompts, self.embedder)
        emb = np.asarray(emb, dtype=np.float32)
        return self.km.predict(emb).astype(int), emb
