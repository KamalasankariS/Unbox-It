"""Content selection: which article goes to which reader.

Scores an impression's candidates by cosine(user_profile, article) and reports the
metrics the MIND benchmark is scored on: AUC, MRR, nDCG@5 and nDCG@10, computed per
impression and averaged.

Impressions whose candidates share a single label are skipped, since there is no
ranking to get right and AUC is undefined. The count of skipped impressions is
reported alongside the metrics.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np

from newspush.config import Config
from newspush.data.schema import MindData
from newspush.features.text import ArticleEncoder
from newspush.features.users import UserProfiles

log = logging.getLogger(__name__)

POPULARITY_CLICK_PRIOR = 1.0
POPULARITY_IMPRESSION_PRIOR = 10.0


@dataclass
class RankingMetrics:
    auc: float
    mrr: float
    ndcg_at_5: float
    ndcg_at_10: float
    n_impressions_evaluated: int
    n_impressions_skipped: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC for one impression via the Mann-Whitney rank-sum identity.

    Equivalent to sklearn's roc_auc_score but far faster on the small arrays here, and
    it is called once per impression. Tied scores get averaged ranks, so a constant
    scorer lands at 0.5.
    """
    n_positive = int(labels.sum())
    n_negative = len(labels) - n_positive
    if n_positive == 0 or n_negative == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)

    sorted_scores = scores[order]
    start = 0
    while start < len(sorted_scores):
        end = start
        while end + 1 < len(sorted_scores) and sorted_scores[end + 1] == sorted_scores[start]:
            end += 1
        if end > start:
            ranks[order[start : end + 1]] = np.mean(ranks[order[start : end + 1]])
        start = end + 1

    positive_rank_sum = ranks[labels == 1].sum()
    return float((positive_rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative))


def mrr_score(labels: np.ndarray, scores: np.ndarray) -> float:
    """Reciprocal rank of the first relevant item."""
    ranked = labels[np.argsort(-scores, kind="mergesort")]
    hits = np.flatnonzero(ranked == 1)
    if hits.size == 0:
        return 0.0
    return float(1.0 / (hits[0] + 1))


def dcg(relevances: np.ndarray, k: int) -> float:
    top = relevances[:k]
    discounts = np.log2(np.arange(2, len(top) + 2))
    return float(np.sum((2**top - 1) / discounts))


def ndcg_score(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    """nDCG@k with binary relevance."""
    actual = dcg(labels[np.argsort(-scores, kind="mergesort")].astype(float), k)
    ideal = dcg(np.sort(labels)[::-1].astype(float), k)
    if ideal == 0:
        return 0.0
    return float(actual / ideal)


class ContentRanker:
    """Cosine-similarity ranker over article embeddings."""

    def __init__(self, encoder: ArticleEncoder, profiles: UserProfiles) -> None:
        self.encoder = encoder
        self.profiles = profiles

    def score_candidates(self, user_id: str, candidates: list[str]) -> np.ndarray:
        """Score one impression's candidates. Both sides are L2-normalised, so the dot
        product is the cosine."""
        profile = self.profiles.get(user_id)
        if not np.any(profile):
            # Cold reader: no basis to rank on. Zeros score this impression as a coin
            # flip rather than inventing an ordering.
            return np.zeros(len(candidates), dtype=float)
        return self.encoder.vecs(candidates) @ profile

    def recommend_for_user(
        self,
        user_id: str,
        top_n: int = 5,
        exclude_history: bool = True,
        candidate_pool: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Top-N articles for a reader from the catalogue. This is the serving path."""
        profile = self.profiles.get_or_global(user_id)
        pool = candidate_pool if candidate_pool is not None else self.encoder.news_ids

        if exclude_history:
            seen = set(self.profiles.history_of(user_id))
            pool = [news_id for news_id in pool if news_id not in seen]

        if not pool:
            return []

        scores = self.encoder.vecs(pool) @ profile
        top_n = min(top_n, len(pool))

        top = np.argpartition(-scores, top_n - 1)[:top_n]
        top = top[np.argsort(-scores[top], kind="mergesort")]
        return [(pool[i], float(scores[i])) for i in top]


def evaluate(ranker: ContentRanker, data: MindData, cfg: Config) -> RankingMetrics:
    """Evaluate the ranker on a split with the standard MIND metrics."""
    return _evaluate_scorer(
        scorer=lambda impression: ranker.score_candidates(impression.user_id, impression.candidates),
        data=data,
        cfg=cfg,
        label="content ranker",
    )


def popularity_baseline(data: MindData) -> dict[str, float]:
    """Smoothed CTR per article: the control policy, and the bar personalisation must clear."""
    clicks: dict[str, int] = {}
    impressions: dict[str, int] = {}

    for impression in data.impressions():
        for news_id, label in zip(impression.candidates, impression.labels):
            impressions[news_id] = impressions.get(news_id, 0) + 1
            clicks[news_id] = clicks.get(news_id, 0) + label

    return {
        news_id: (clicks[news_id] + POPULARITY_CLICK_PRIOR)
        / (impressions[news_id] + POPULARITY_IMPRESSION_PRIOR)
        for news_id in impressions
    }


def evaluate_popularity(
    data: MindData,
    popularity: dict[str, float],
    cfg: Config,
) -> RankingMetrics:
    """Same metrics under a global popularity ranking, as a reference point."""
    return _evaluate_scorer(
        scorer=lambda impression: np.array(
            [popularity.get(news_id, 0.0) for news_id in impression.candidates], dtype=float
        ),
        data=data,
        cfg=cfg,
        label="popularity baseline",
    )


def _evaluate_scorer(scorer, data: MindData, cfg: Config, label: str) -> RankingMetrics:
    max_impressions = int(cfg.get("content_selection.max_eval_impressions", 0) or 0)

    auc: list[float] = []
    mrr: list[float] = []
    ndcg_5: list[float] = []
    ndcg_10: list[float] = []
    skipped = 0

    for impression in data.impressions():
        if max_impressions and len(auc) >= max_impressions:
            break

        labels = np.asarray(impression.labels, dtype=int)
        if labels.sum() in (0, len(labels)):
            skipped += 1
            continue

        scores = scorer(impression)
        auc.append(auc_score(labels, scores))
        mrr.append(mrr_score(labels, scores))
        ndcg_5.append(ndcg_score(labels, scores, 5))
        ndcg_10.append(ndcg_score(labels, scores, 10))

    if not auc:
        raise ValueError("no evaluable impressions: every impression had a single label class")

    metrics = RankingMetrics(
        auc=float(np.nanmean(auc)),
        mrr=float(np.mean(mrr)),
        ndcg_at_5=float(np.mean(ndcg_5)),
        ndcg_at_10=float(np.mean(ndcg_10)),
        n_impressions_evaluated=len(auc),
        n_impressions_skipped=skipped,
    )

    log.info(
        "%s [%s / %s]: AUC=%.4f MRR=%.4f nDCG@5=%.4f nDCG@10=%.4f (n=%d, skipped=%d)",
        label,
        data.split,
        data.data_source,
        metrics.auc,
        metrics.mrr,
        metrics.ndcg_at_5,
        metrics.ndcg_at_10,
        metrics.n_impressions_evaluated,
        metrics.n_impressions_skipped,
    )
    return metrics
