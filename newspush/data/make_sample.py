"""MIND-format sample generator: the labelled fallback when real MIND is absent.

The data is simulated. Everything computed from it is tagged
`data_source="mind-format-sample"` so it can never be mistaken for a real result.

Two signals are planted so the models have something genuine to recover: a latent
per-user topic preference (carried by the article text, not by a leaked topic id) and
a per-user preferred open hour. They are additive in the click logit, so content
selection and send-time optimisation each have an independent signal to find.

    click ~ Bernoulli(sigmoid(base + w_topic * affinity + w_hour * hour_match + noise))
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from newspush.config import Config
from newspush.data.schema import BEHAVIOR_COLUMNS, NEWS_COLUMNS, SAMPLE, MindData

TOPICS: list[str] = [
    "news",
    "sports",
    "finance",
    "travel",
    "lifestyle",
    "health",
    "foodanddrink",
    "weather",
    "autos",
    "tv",
    "movies",
    "music",
]

TOPIC_VOCAB: dict[str, list[str]] = {
    "news": ["senate", "policy", "election", "governor", "ruling", "congress", "vote", "bill", "court", "campaign"],
    "sports": ["quarterback", "playoff", "rebound", "coach", "roster", "inning", "goal", "tournament", "trade", "season"],
    "finance": ["earnings", "markets", "inflation", "shares", "dividend", "revenue", "nasdaq", "investors", "bond", "rate"],
    "travel": ["itinerary", "airfare", "resort", "passport", "cruise", "layover", "hostel", "destination", "flight", "island"],
    "lifestyle": ["wardrobe", "decor", "minimalist", "routine", "budgeting", "declutter", "wellness", "habit", "trend", "style"],
    "health": ["symptoms", "vaccine", "cardiology", "nutrition", "clinical", "diagnosis", "therapy", "patients", "study", "risk"],
    "foodanddrink": ["recipe", "roasted", "skillet", "sourdough", "marinade", "pastry", "brunch", "espresso", "chef", "flavor"],
    "weather": ["forecast", "blizzard", "hurricane", "humidity", "tornado", "snowfall", "advisory", "storm", "temperature", "warning"],
    "autos": ["horsepower", "sedan", "hybrid", "mileage", "chassis", "torque", "recall", "electric", "suv", "engine"],
    "tv": ["episode", "finale", "showrunner", "sitcom", "renewed", "streaming", "cast", "series", "premiere", "drama"],
    "movies": ["blockbuster", "screenplay", "boxoffice", "director", "sequel", "trailer", "oscar", "cinema", "role", "studio"],
    "music": ["album", "tour", "single", "guitarist", "billboard", "remix", "vocals", "concert", "label", "band"],
}

# Mixed into every article so the topic vocabularies are not perfectly separable.
FILLER = ["the", "a", "new", "report", "says", "after", "before", "why", "how", "what", "this", "week", "year", "first"]

SUBCATEGORY_SUFFIXES = ["daily", "weekly", "analysis", "opinion", "live"]

TRAIN_FRACTION = 0.8
HISTORY_PRIOR = 0.4  # Dirichlet concentration; low values give peaky user preferences


def generate(cfg: Config, seed: int | None = None) -> tuple[MindData, MindData]:
    """Generate a simulated (train, dev) pair in the real MIND schema.

    Users, articles and their latent preferences are shared across the splits, as they
    are in real MIND, so a profile learned on train remains meaningful on dev.
    """
    rng = np.random.default_rng(cfg.seed if seed is None else seed)

    n_users = int(cfg.require("sample.n_users"))
    n_news = int(cfg.require("sample.n_news"))
    n_impressions = int(cfg.require("sample.n_impressions"))
    n_candidates = int(cfg.require("sample.candidates_per_impression"))
    history_low, history_high = cfg.require("sample.history_len_range")
    weight_topic = float(cfg.require("sample.topic_affinity_weight"))
    weight_hour = float(cfg.require("sample.hour_effect_weight"))
    base_logit = float(cfg.require("sample.base_click_logit"))

    news, article_topic = _make_news(rng, n_news)
    n_topics = len(TOPICS)

    user_preference = rng.dirichlet(alpha=np.full(n_topics, HISTORY_PRIOR), size=n_users)
    user_hour = rng.integers(0, 24, size=n_users)

    # Centred so an average-affinity article does not shift the base click rate.
    affinity = user_preference * n_topics - 1.0

    histories = _make_histories(
        rng, user_preference, article_topic, n_users, n_topics, int(history_low), int(history_high)
    )

    users = rng.integers(0, n_users, size=n_impressions)
    hours = rng.integers(0, 24, size=n_impressions)
    candidates = rng.integers(0, n_news, size=(n_impressions, n_candidates))

    topic_term = affinity[users[:, None], article_topic[candidates]]
    hour_term = _hour_match(hours, user_hour[users])[:, None]
    noise = rng.normal(0.0, 0.5, size=(n_impressions, n_candidates))

    click_prob = _sigmoid(base_logit + weight_topic * topic_term + weight_hour * hour_term + noise)
    clicked = (rng.random(size=click_prob.shape) < click_prob).astype(int)

    behaviors = _make_behaviors(rng, users, hours, candidates, clicked, histories)

    cut = int(len(behaviors) * TRAIN_FRACTION)
    train = MindData(
        news=news,
        behaviors=behaviors.iloc[:cut].reset_index(drop=True),
        data_source=SAMPLE,
        split="train",
    )
    dev = MindData(
        news=news.copy(),
        behaviors=behaviors.iloc[cut:].reset_index(drop=True),
        data_source=SAMPLE,
        split="dev",
    )
    return train, dev


def _make_news(rng: np.random.Generator, n_news: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Build the article catalogue. Returns (news frame, topic index per article)."""
    topic_index = rng.integers(0, len(TOPICS), size=n_news)
    rows = []

    for i in range(n_news):
        topic = TOPICS[topic_index[i]]
        vocab = TOPIC_VOCAB[topic]

        title_words = list(rng.choice(vocab, size=5)) + list(rng.choice(FILLER, size=3))
        abstract_words = list(rng.choice(vocab, size=14)) + list(rng.choice(FILLER, size=10))
        rng.shuffle(title_words)
        rng.shuffle(abstract_words)

        rows.append(
            {
                "news_id": f"N{i:06d}",
                "category": topic,
                "subcategory": f"{topic}-{SUBCATEGORY_SUFFIXES[i % len(SUBCATEGORY_SUFFIXES)]}",
                "title": " ".join(title_words).capitalize(),
                "abstract": " ".join(abstract_words).capitalize() + ".",
                "url": f"https://example.invalid/{topic}/N{i:06d}",
                "title_entities": "[]",
                "abstract_entities": "[]",
            }
        )

    return pd.DataFrame(rows, columns=NEWS_COLUMNS), topic_index


def _make_histories(
    rng: np.random.Generator,
    user_preference: np.ndarray,
    article_topic: np.ndarray,
    n_users: int,
    n_topics: int,
    history_low: int,
    history_high: int,
) -> list[list[str]]:
    """Draw each user's click history from their own topic preferences."""
    articles_by_topic = {topic: np.flatnonzero(article_topic == topic) for topic in range(n_topics)}
    histories: list[list[str]] = []

    for user in range(n_users):
        length = int(rng.integers(history_low, history_high + 1))
        topics = rng.choice(n_topics, size=length, p=user_preference[user])
        history = [
            f"N{int(rng.choice(articles_by_topic[int(topic)])):06d}"
            for topic in topics
            if articles_by_topic[int(topic)].size > 0
        ]
        histories.append(history)

    return histories


def _make_behaviors(
    rng: np.random.Generator,
    users: np.ndarray,
    hours: np.ndarray,
    candidates: np.ndarray,
    clicked: np.ndarray,
    histories: list[list[str]],
) -> pd.DataFrame:
    """Assemble behaviors rows in MIND's exact string format."""
    base_day = pd.Timestamp("2019-11-11")
    rows = []

    for i in range(len(users)):
        user = int(users[i])
        timestamp = base_day + pd.Timedelta(
            days=int(i % 7), hours=int(hours[i]), minutes=int(rng.integers(0, 60))
        )
        impressions = " ".join(
            f"N{int(candidates[i, j]):06d}-{int(clicked[i, j])}" for j in range(candidates.shape[1])
        )
        rows.append(
            {
                "impression_id": str(i + 1),
                "user_id": f"U{user:06d}",
                "time": timestamp.strftime("%m/%d/%Y %I:%M:%S %p"),
                "history": " ".join(histories[user]),
                "impressions": impressions,
            }
        )

    return pd.DataFrame(rows, columns=BEHAVIOR_COLUMNS)


def _hour_match(hour: np.ndarray, preferred_hour: np.ndarray) -> np.ndarray:
    """Circular closeness in [-1, 1], so 23:00 and 01:00 are adjacent."""
    return np.cos(2.0 * np.pi * (np.asarray(hour) - np.asarray(preferred_hour)) / 24.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
