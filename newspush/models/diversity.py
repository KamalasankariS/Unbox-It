"""Editorial guardrails: diversity, desk caps, and editor pins.

A pure relevance ranker, pointed at a reader who clicked three basketball stories, will
send five more. Each is individually the best answer to "what will this reader open",
and the email as a whole is one no newsroom would send.

Three layers let editorial judgement outrank the model:

    MMR         greedily trade an article's relevance against its similarity to what is
                already in the send:

                    score(a) = lambda * relevance(a) - (1 - lambda) * max_sim(a, selected)

                lambda = 1 is pure relevance, lambda = 0 is pure novelty. It lives in
                config because that trade-off is a product decision, not an ML one.
    Desk caps   a hard limit on articles per category. MMR discourages redundancy;
                this forbids it, so editors get a guarantee rather than a tendency.
    Pins        an editor can force an article into the send, or boost it without
                overriding the ranker entirely.

Pins are placed first, then MMR fills the remaining slots subject to the caps.
"""

from __future__ import annotations

import logging

import numpy as np

from newspush.config import Config
from newspush.features.text import ArticleEncoder

log = logging.getLogger(__name__)


def mmr_select(
    candidates: list[str],
    relevance: dict[str, float],
    encoder: ArticleEncoder,
    k: int,
    lambda_: float = 0.7,
    news_category: dict[str, str] | None = None,
    max_per_category: int | None = None,
    pinned: list[str] | None = None,
) -> list[str]:
    """Select at most `k` articles: pins first, then MMR under the desk caps."""
    if k <= 0:
        return []

    selected: list[str] = []
    category_counts: dict[str, int] = {}

    def category_of(news_id: str) -> str | None:
        return news_category.get(news_id) if news_category else None

    def within_cap(news_id: str) -> bool:
        if max_per_category is None or news_category is None:
            return True
        category = category_of(news_id)
        if category is None:
            return True
        return category_counts.get(category, 0) < max_per_category

    def take(news_id: str) -> None:
        selected.append(news_id)
        category = category_of(news_id)
        if category is not None:
            category_counts[category] = category_counts.get(category, 0) + 1

    # Pins are unconditional, but they still consume desk-cap budget.
    for news_id in pinned or []:
        if len(selected) >= k:
            break
        if news_id not in selected:
            take(news_id)

    remaining = [news_id for news_id in candidates if news_id not in selected]
    if not remaining:
        return selected[:k]

    vectors = {news_id: encoder.vec(news_id) for news_id in remaining}
    selected_vectors = [encoder.vec(news_id) for news_id in selected]

    while len(selected) < k and remaining:
        best_id, best_score = None, -np.inf

        for news_id in remaining:
            if not within_cap(news_id):
                continue

            # Vectors are L2-normalised, so a dot product is the cosine similarity.
            redundancy = (
                max(float(vectors[news_id] @ other) for other in selected_vectors)
                if selected_vectors
                else 0.0
            )
            score = lambda_ * relevance.get(news_id, 0.0) - (1.0 - lambda_) * redundancy

            if score > best_score:
                best_id, best_score = news_id, score

        if best_id is None:
            # Every remaining candidate is blocked by a desk cap. A short email beats
            # one that breaks the guarantee.
            log.debug("MMR stopped at %d of %d: all candidates blocked by desk caps", len(selected), k)
            break

        take(best_id)
        selected_vectors.append(vectors[best_id])
        remaining.remove(best_id)

    return selected


def apply_editorial_boost(
    relevance: dict[str, float],
    boosted: dict[str, float] | None = None,
) -> dict[str, float]:
    """Tilt the ranker toward editor-flagged stories.

    Multiplicative rather than additive, so a boosted article that is genuinely wrong
    for a reader still loses to one that is right. An editor gets a thumb on the scale,
    not a replacement for it.
    """
    if not boosted:
        return relevance
    return {news_id: score * boosted.get(news_id, 1.0) for news_id, score in relevance.items()}


def diversity_score(selected: list[str], encoder: ArticleEncoder) -> float:
    """Mean pairwise cosine distance within a send. 0 is identical, 1 is unrelated."""
    if len(selected) < 2:
        return 0.0

    vectors = np.stack([encoder.vec(news_id) for news_id in selected])
    similarities = vectors @ vectors.T
    upper = np.triu_indices(len(selected), k=1)
    return float(1.0 - np.mean(similarities[upper]))


def category_spread(selected: list[str], news_category: dict[str, str]) -> int:
    """How many distinct desks a send draws from."""
    return len({news_category.get(news_id) for news_id in selected if news_category.get(news_id)})


def build_editorial_send(
    cfg: Config,
    candidates: list[str],
    relevance: dict[str, float],
    encoder: ArticleEncoder,
    news_category: dict[str, str],
    k: int,
    pinned: list[str] | None = None,
    boosted: dict[str, float] | None = None,
) -> tuple[list[str], dict[str, float]]:
    """Apply the full guardrail stack and report what it cost.

    `relevance_retained` is the price tag: the share of the pure-relevance send's total
    score that survives the guardrails.
    """
    selected = mmr_select(
        candidates=candidates,
        relevance=apply_editorial_boost(relevance, boosted),
        encoder=encoder,
        k=k,
        lambda_=float(cfg.get("diversity.mmr_lambda", 0.7)),
        news_category=news_category,
        max_per_category=int(cfg.get("diversity.max_per_category", 2)),
        pinned=pinned,
    )

    greedy = sorted(candidates, key=lambda news_id: relevance.get(news_id, 0.0), reverse=True)[:k]
    greedy_total = sum(relevance.get(news_id, 0.0) for news_id in greedy)
    selected_total = sum(relevance.get(news_id, 0.0) for news_id in selected)

    stats = {
        "diversity": diversity_score(selected, encoder),
        "greedy_diversity": diversity_score(greedy, encoder),
        "category_spread": float(category_spread(selected, news_category)),
        "greedy_category_spread": float(category_spread(greedy, news_category)),
        "relevance_retained": float(selected_total / greedy_total) if greedy_total > 0 else 1.0,
    }
    return selected, stats
