"""Explanations: why this article for this reader.

A recommender that cannot say why is hard to debug, hard for an editor to trust, and
impossible to defend when it gets something wrong. The TF-IDF encoder makes this cheap:
its SVD components stay linear in the term basis, so an article's vector can be
projected back onto the words that produced it.

An explanation has three parts:
    the terms the reader's profile and the article agree on,
    the desks the reader reads, and where this article sits among them,
    the history articles most similar to this one, as concrete evidence.

Term-level attribution requires the TF-IDF encoder. Under a sentence-transformer there
is no term basis to project onto, so the explanation degrades to the history and desk
evidence and says so rather than inventing an attribution.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from newspush.features.text import ArticleEncoder, TfidfSvdEncoder
from newspush.features.users import UserProfiles

log = logging.getLogger(__name__)

DEFAULT_TOP_TERMS = 6
DEFAULT_TOP_HISTORY = 3


@dataclass
class Explanation:
    user_id: str
    news_id: str
    title: str
    category: str
    score: float
    shared_terms: list[tuple[str, float]]
    top_history_matches: list[tuple[str, str, float]]
    reader_top_categories: list[tuple[str, int]]
    is_cold_start: bool
    summary: str
    term_attribution_available: bool

    def to_dict(self) -> dict:
        return asdict(self)


def explain(
    user_id: str,
    news_id: str,
    encoder: ArticleEncoder,
    profiles: UserProfiles,
    news: pd.DataFrame,
    top_terms: int = DEFAULT_TOP_TERMS,
    top_history: int = DEFAULT_TOP_HISTORY,
) -> Explanation:
    """Explain why `news_id` was recommended to `user_id`."""
    catalogue = news.set_index("news_id")
    if news_id not in catalogue.index:
        raise KeyError(f"unknown news_id: {news_id!r}")

    article = catalogue.loc[news_id]
    profile = profiles.get_or_global(user_id)
    is_cold = profiles.is_cold(user_id)

    score = float(encoder.vec(news_id) @ profile)
    history = profiles.history_of(user_id)

    shared_terms = _shared_terms(encoder, profiles, user_id, news_id, top_terms)
    history_matches = _closest_history(encoder, history, news_id, catalogue, top_history)
    reader_categories = _reader_categories(history, catalogue)

    explanation = Explanation(
        user_id=user_id,
        news_id=news_id,
        title=str(article["title"]),
        category=str(article["category"]),
        score=score,
        shared_terms=shared_terms,
        top_history_matches=history_matches,
        reader_top_categories=reader_categories,
        is_cold_start=is_cold,
        term_attribution_available=isinstance(encoder, TfidfSvdEncoder),
        summary="",
    )
    explanation.summary = _summarize(explanation)
    return explanation


def _shared_terms(
    encoder: ArticleEncoder,
    profiles: UserProfiles,
    user_id: str,
    news_id: str,
    k: int,
) -> list[tuple[str, float]]:
    """Terms that both the article and the reader's history weight heavily.

    The article's terms are weighted by how often they appear across what the reader has
    already clicked, so a term only surfaces if it is doing work on both sides.
    """
    if not isinstance(encoder, TfidfSvdEncoder):
        return []

    article_terms = dict(encoder.top_terms(news_id, k=25))
    if not article_terms:
        return []

    history = profiles.history_of(user_id)
    if not history:
        return sorted(article_terms.items(), key=lambda item: -item[1])[:k]

    history_weights: dict[str, float] = {}
    for past_id in history[-50:]:
        for term, weight in encoder.top_terms(past_id, k=15):
            history_weights[term] = history_weights.get(term, 0.0) + weight

    shared = {
        term: weight * history_weights[term]
        for term, weight in article_terms.items()
        if term in history_weights
    }
    if not shared:
        return sorted(article_terms.items(), key=lambda item: -item[1])[:k]

    return sorted(shared.items(), key=lambda item: -item[1])[:k]


def _closest_history(
    encoder: ArticleEncoder,
    history: list[str],
    news_id: str,
    catalogue: pd.DataFrame,
    k: int,
) -> list[tuple[str, str, float]]:
    """The reader's past articles most similar to this one: (news_id, title, similarity)."""
    if not history:
        return []

    unique_history = list(dict.fromkeys(history))[-200:]
    article_vector = encoder.vec(news_id)
    similarities = encoder.vecs(unique_history) @ article_vector

    k = min(k, len(unique_history))
    top = np.argsort(-similarities)[:k]

    matches = []
    for index in top:
        past_id = unique_history[index]
        title = str(catalogue.loc[past_id, "title"]) if past_id in catalogue.index else "(unknown article)"
        matches.append((past_id, title, float(similarities[index])))

    return matches


def _reader_categories(history: list[str], catalogue: pd.DataFrame) -> list[tuple[str, int]]:
    """The desks the reader clicks most, with counts."""
    counts: dict[str, int] = {}

    for news_id in history:
        if news_id in catalogue.index:
            category = str(catalogue.loc[news_id, "category"])
            counts[category] = counts.get(category, 0) + 1

    return sorted(counts.items(), key=lambda item: -item[1])[:5]


def _summarize(explanation: Explanation) -> str:
    """A one-paragraph explanation an editor could read without knowing the model."""
    if explanation.is_cold_start:
        return (
            f"We know nothing about {explanation.user_id} yet, so this recommendation falls back "
            f"to what the average reader engages with. '{explanation.title}' is a "
            f"{explanation.category} story; personalisation begins once they click something."
        )

    parts = [f"'{explanation.title}' is a {explanation.category} story."]

    if explanation.reader_top_categories:
        desks = ", ".join(
            f"{category} ({count})" for category, count in explanation.reader_top_categories[:3]
        )
        parts.append(f"This reader mostly clicks {desks}.")

    if explanation.shared_terms:
        terms = ", ".join(term for term, _ in explanation.shared_terms[:4])
        parts.append(f"It shares the themes {terms} with what they have read before.")
    elif not explanation.term_attribution_available:
        parts.append(
            "Term-level attribution is unavailable under the current encoder, so the evidence "
            "below is drawn from reading history rather than from individual words."
        )

    if explanation.top_history_matches:
        closest_title, similarity = explanation.top_history_matches[0][1], explanation.top_history_matches[0][2]
        parts.append(f"Their closest previous read is '{closest_title}' (similarity {similarity:.2f}).")

    parts.append(f"Overall match score: {explanation.score:.3f}.")
    return " ".join(parts)
