"""Article encoders.

`ArticleEncoder` is an interface. Downstream code only calls `.vec()` / `.vecs()` and
never learns which implementation produced the vectors, so swapping the TF-IDF baseline
for a sentence-transformer is a config change with no downstream edits.

All encoders return L2-normalised vectors, which makes cosine similarity a plain dot
product everywhere else in the codebase.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from newspush.config import Config

log = logging.getLogger(__name__)

TFIDF_KINDS = ("tfidf-svd", "tfidf", "svd")
SENTENCE_TRANSFORMER_KINDS = ("sentence-transformer", "st", "llm")


def article_text(news: pd.DataFrame) -> pd.Series:
    """The text an encoder sees. Concatenated rather than joined, since MIND leaves
    a meaningful share of abstracts empty and the title must carry those alone."""
    title = news["title"].fillna("").astype(str)
    abstract = news["abstract"].fillna("").astype(str)
    return (title + ". " + abstract).str.strip()


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """L2-normalise, leaving all-zero rows as zeros rather than dividing by zero."""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return np.divide(x, norm, out=np.zeros_like(x, dtype=float), where=norm > 0)


class ArticleEncoder(ABC):
    """Maps a news_id to a dense, L2-normalised vector."""

    def __init__(self) -> None:
        self._index: dict[str, int] = {}
        self._matrix: np.ndarray | None = None

    @abstractmethod
    def fit(self, news: pd.DataFrame) -> "ArticleEncoder":
        """Fit on the article catalogue, populating the index and matrix."""

    def dim(self) -> int:
        return int(self.matrix.shape[1])

    def vec(self, news_id: str) -> np.ndarray:
        """Vector for one article. Unknown ids get zeros, so they contribute nothing to
        a profile and score zero against every candidate."""
        index = self._index.get(news_id)
        if index is None:
            return np.zeros(self.dim(), dtype=float)
        return self.matrix[index]

    def vecs(self, news_ids: list[str]) -> np.ndarray:
        """Vectorised `vec` over many ids."""
        out = np.zeros((len(news_ids), self.dim()), dtype=float)
        for i, news_id in enumerate(news_ids):
            index = self._index.get(news_id)
            if index is not None:
                out[i] = self.matrix[index]
        return out

    def known(self, news_id: str) -> bool:
        return news_id in self._index

    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            raise RuntimeError("encoder is not fitted")
        return self._matrix

    @property
    def news_ids(self) -> list[str]:
        return list(self._index.keys())


class TfidfSvdEncoder(ArticleEncoder):
    """TF-IDF over title and abstract, reduced by truncated SVD.

    The default encoder: deterministic, CPU-only, no extra dependencies, and its SVD
    components stay linear in the TF-IDF terms, which is what lets `explain.why` name
    the words behind a recommendation.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.target_dim = int(cfg.require("encoder.dim"))

        ngram_low, ngram_high = cfg.require("encoder.ngram_range")
        self.vectorizer = TfidfVectorizer(
            max_features=int(cfg.require("encoder.max_features")),
            min_df=int(cfg.require("encoder.min_df")),
            ngram_range=(int(ngram_low), int(ngram_high)),
            stop_words="english",
            sublinear_tf=True,
        )
        self.svd: TruncatedSVD | None = None

    def fit(self, news: pd.DataFrame) -> "TfidfSvdEncoder":
        try:
            tfidf = self.vectorizer.fit_transform(article_text(news))
        except ValueError as exc:
            # An empty vocabulary (too few articles, or every term pruned by min_df)
            # surfaces from sklearn as an opaque message. Say what actually went wrong.
            raise ValueError(
                f"catalogue too small to encode: {len(news)} articles left no vocabulary "
                f"after min_df={self.vectorizer.min_df} pruning"
            ) from exc

        # SVD cannot produce more components than min(n_docs, n_terms) - 1.
        n_components = min(self.target_dim, min(tfidf.shape) - 1)
        if n_components < 1:
            raise ValueError(
                f"catalogue too small to encode: TF-IDF matrix is {tfidf.shape}; "
                "need at least 2 articles and 2 vocabulary terms"
            )
        if n_components < self.target_dim:
            log.warning(
                "encoder.dim=%d exceeds what this catalogue supports; using %d components",
                self.target_dim,
                n_components,
            )

        self.svd = TruncatedSVD(n_components=n_components, random_state=self.cfg.seed)
        self._matrix = l2_normalize(self.svd.fit_transform(tfidf).astype(float))
        self._index = {news_id: i for i, news_id in enumerate(news["news_id"].astype(str))}

        log.info(
            "TfidfSvdEncoder: %d articles, %d terms -> %d dims (%.1f%% variance retained)",
            tfidf.shape[0],
            tfidf.shape[1],
            n_components,
            100.0 * float(self.svd.explained_variance_ratio_.sum()),
        )
        return self

    def top_terms(self, news_id: str, k: int = 8) -> list[tuple[str, float]]:
        """The TF-IDF terms that dominate an article's vector, by projecting its SVD
        representation back into term space."""
        if self.svd is None:
            raise RuntimeError("encoder is not fitted")

        index = self._index.get(news_id)
        if index is None:
            return []

        term_weights = self.svd.components_.T @ self.matrix[index]
        vocabulary = self.vectorizer.get_feature_names_out()
        top = np.argsort(term_weights)[::-1][:k]
        return [(str(vocabulary[i]), float(term_weights[i])) for i in top if term_weights[i] > 0]


class SentenceTransformerEncoder(ArticleEncoder):
    """Pretrained sentence embeddings. Optional dependency, import-guarded."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.model_name = str(cfg.require("encoder.st_model"))

    def fit(self, news: pd.DataFrame) -> "SentenceTransformerEncoder":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "encoder.kind='sentence-transformer' requires the optional dependency: "
                "pip install sentence-transformers"
            ) from exc

        model = SentenceTransformer(self.model_name)
        texts = article_text(news).tolist()
        embeddings = model.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)

        self._matrix = l2_normalize(np.asarray(embeddings, dtype=float))
        self._index = {news_id: i for i, news_id in enumerate(news["news_id"].astype(str))}

        log.info(
            "SentenceTransformerEncoder(%s): %d articles -> %d dims",
            self.model_name,
            len(texts),
            self.dim(),
        )
        return self


def build_encoder(cfg: Config) -> ArticleEncoder:
    """Construct the encoder named by `encoder.kind`.

    A missing optional dependency degrades to TF-IDF rather than failing the run, but
    the substitution is logged and `encoder_name` records what actually ran.
    """
    kind = str(cfg.get("encoder.kind", "tfidf-svd")).lower()

    if kind in TFIDF_KINDS:
        return TfidfSvdEncoder(cfg)

    if kind in SENTENCE_TRANSFORMER_KINDS:
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            log.warning(
                "encoder.kind=%r requested but sentence-transformers is not installed; "
                "falling back to TF-IDF and SVD",
                kind,
            )
            return TfidfSvdEncoder(cfg)
        return SentenceTransformerEncoder(cfg)

    raise ValueError(f"unknown encoder.kind: {kind!r}")


def encoder_name(encoder: ArticleEncoder) -> str:
    """Identify the encoder that actually ran, for metrics.json."""
    if isinstance(encoder, SentenceTransformerEncoder):
        return f"sentence-transformer:{encoder.model_name}"
    return "tfidf-svd"
