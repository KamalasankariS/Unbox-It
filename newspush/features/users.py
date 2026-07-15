"""User profiles built from click history.

    profile(u) = l2_normalize(mean(vec(a) for a in clicked_articles(u)))

The centroid of a reader's clicked articles, which makes cosine(profile, article) a
relevance score in [-1, 1]. Needs no per-user training and updates incrementally as
new clicks arrive, which is what a daily send cycle requires.

Readers with no click history get a zero profile; callers handle that explicitly
rather than pretending to personalise. See `is_cold` and `get_or_global`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from newspush.data.schema import MindData
from newspush.features.text import ArticleEncoder, l2_normalize

log = logging.getLogger(__name__)


@dataclass
class UserProfiles:
    """Profile vectors plus the click history each was built from."""

    vectors: dict[str, np.ndarray]
    history: dict[str, list[str]]
    dim: int
    global_profile: np.ndarray = field(repr=False, default=None)  # type: ignore[assignment]

    def get(self, user_id: str) -> np.ndarray:
        """This reader's profile, or zeros if we know nothing about them."""
        vector = self.vectors.get(user_id)
        return vector if vector is not None else np.zeros(self.dim, dtype=float)

    def get_or_global(self, user_id: str) -> np.ndarray:
        """Profile, backing off to the population centroid for cold readers.

        Used in serving, where a cold reader still has to receive an email.
        """
        vector = self.vectors.get(user_id)
        if vector is None or not np.any(vector):
            return self.global_profile
        return vector

    def is_cold(self, user_id: str) -> bool:
        vector = self.vectors.get(user_id)
        return vector is None or not np.any(vector)

    def history_of(self, user_id: str) -> list[str]:
        return self.history.get(user_id, [])

    def __len__(self) -> int:
        return len(self.vectors)


def build_profiles(
    data: MindData,
    encoder: ArticleEncoder,
    include_clicked_impressions: bool = True,
    extra_history: MindData | None = None,
) -> UserProfiles:
    """Build a profile per user from their reading history.

    Args:
        data: the split to learn from. Must be train when the profiles will score dev:
            folding in this split's clicked impressions is what would leak dev labels.
        include_clicked_impressions: also fold in articles clicked within `data`'s
            impressions, not just the `history` field.
        extra_history: a second split (typically dev) whose per-user `history` field is
            folded in, but whose impression clicks are NOT. A user's history is their
            past by construction, so using the dev history to score dev candidates is
            the intended MIND setup, not leakage — and it is essential here, since MIND
            splits by time and ~88% of dev users never appear in train.
    """
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    history: dict[str, list[str]] = {}

    def add(user_id: str, news_id: str) -> None:
        if not encoder.known(news_id):
            return
        vector = encoder.vec(news_id)
        if user_id in sums:
            sums[user_id] += vector
            counts[user_id] += 1
        else:
            sums[user_id] = vector.copy()
            counts[user_id] = 1
        history.setdefault(user_id, []).append(news_id)

    # MIND repeats a user's history on every one of their rows, so fold it in once per
    # user. Otherwise a frequent reader's profile collapses onto their own history.
    history_seen: set[str] = set()

    def fold_history(source: MindData) -> None:
        for impression in source.impressions():
            if impression.user_id in history_seen:
                continue
            history_seen.add(impression.user_id)
            for news_id in impression.history:
                add(impression.user_id, news_id)

    fold_history(data)
    if extra_history is not None:
        fold_history(extra_history)

    if include_clicked_impressions:
        for impression in data.impressions():
            for news_id, label in zip(impression.candidates, impression.labels):
                if label == 1:
                    add(impression.user_id, news_id)

    vectors = {user: l2_normalize(total / counts[user]) for user, total in sums.items()}

    if vectors:
        global_profile = l2_normalize(np.mean(np.stack(list(vectors.values())), axis=0))
    else:
        global_profile = np.zeros(encoder.dim(), dtype=float)

    log.info(
        "built %d user profiles (dim=%d); mean history length %.1f",
        len(vectors),
        encoder.dim(),
        float(np.mean([len(h) for h in history.values()])) if history else 0.0,
    )

    return UserProfiles(
        vectors=vectors,
        history=history,
        dim=encoder.dim(),
        global_profile=global_profile,
    )


def profile_topic_entropy(
    user_id: str,
    profiles: UserProfiles,
    news_category: dict[str, str],
) -> float:
    """Shannon entropy of the desk mix in a reader's history.

    Low entropy is a specialist, high is a generalist. The two respond differently to
    a recommendation, so the propensity model takes it as a feature.
    """
    history = profiles.history_of(user_id)
    if not history:
        return 0.0

    categories: dict[str, int] = {}
    for news_id in history:
        category = news_category.get(news_id)
        if category is not None:
            categories[category] = categories.get(category, 0) + 1

    if not categories:
        return 0.0

    counts = np.array(list(categories.values()), dtype=float)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log(probabilities + 1e-12)))
