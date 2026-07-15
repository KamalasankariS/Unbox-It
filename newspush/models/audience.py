"""Audience creation: given an article, which readers should receive it.

Both audience creation and content selection come from one model of
P(click | user, article, hour). Ranking it across articles for a fixed reader gives
content selection; ranking it across readers for a fixed article gives the audience.
Training one model rather than two keeps the two views consistent.

Reported on dev with ROC-AUC and precision@k over the selected audience.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from newspush.config import Config
from newspush.data.schema import MindData
from newspush.features.text import ArticleEncoder
from newspush.features.users import UserProfiles, profile_topic_entropy

log = logging.getLogger(__name__)

# Column layout of the design matrix, ahead of the category one-hot block.
FEATURE_COSINE = 0
FEATURE_HISTORY_LENGTH = 1
FEATURE_TOPIC_ENTROPY = 2
FEATURE_HOUR = 3
FEATURE_POPULARITY = 4
N_DENSE_FEATURES = 5

DEFAULT_EVAL_ROWS = 200_000
DEFAULT_SCORING_HOUR = 12


@dataclass
class AudienceMetrics:
    roc_auc: float
    precision_at_k: float
    audience_k: int
    n_train_rows: int
    n_eval_rows: int
    base_click_rate: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class PropensityModel:
    """P(click | user, article, hour), and the audience selector built on it."""

    def __init__(self, cfg: Config, encoder: ArticleEncoder, profiles: UserProfiles) -> None:
        self.cfg = cfg
        self.encoder = encoder
        self.profiles = profiles

        self.model: LogisticRegression | HistGradientBoostingClassifier | None = None
        self.scaler: StandardScaler | None = None

        self.categories: list[str] = []
        self._category_index: dict[str, int] = {}
        self._news_category: dict[str, str] = {}
        self._popularity: dict[str, float] = {}
        self._entropy_cache: dict[str, float] = {}

    def fit_catalogue(self, news: pd.DataFrame, popularity: dict[str, float]) -> None:
        """Register the article catalogue: category vocabulary and popularity priors."""
        self._news_category = dict(zip(news["news_id"].astype(str), news["category"].astype(str)))
        self.categories = sorted(set(self._news_category.values()))
        self._category_index = {category: i for i, category in enumerate(self.categories)}
        self._popularity = popularity

    def featurize(self, user_ids: list[str], news_ids: list[str], hours: list[int]) -> np.ndarray:
        """Design matrix for a batch of (user, article, hour) triples."""
        n_rows = len(user_ids)
        features = np.zeros((n_rows, N_DENSE_FEATURES + len(self.categories)), dtype=float)

        profiles = np.stack([self.profiles.get(user) for user in user_ids])
        articles = self.encoder.vecs(news_ids)
        features[:, FEATURE_COSINE] = np.einsum("ij,ij->i", profiles, articles)

        for i, (user, news_id, hour) in enumerate(zip(user_ids, news_ids, hours)):
            features[i, FEATURE_HISTORY_LENGTH] = np.log1p(len(self.profiles.history_of(user)))
            features[i, FEATURE_TOPIC_ENTROPY] = self._topic_entropy(user)
            features[i, FEATURE_HOUR] = hour
            features[i, FEATURE_POPULARITY] = self._popularity.get(news_id, 0.0)

            category = self._news_category.get(news_id)
            if category in self._category_index:
                features[i, N_DENSE_FEATURES + self._category_index[category]] = 1.0

        return features

    def build_dataset(self, data: MindData, max_rows: int) -> tuple[np.ndarray, np.ndarray]:
        """Flatten impressions into (features, label) rows, capped at `max_rows`.

        The cap is a runtime lever, not a modelling one: MIND-small train holds roughly
        5M candidate pairs and the model converges on far fewer. It is recorded in
        metrics.json.
        """
        user_ids: list[str] = []
        news_ids: list[str] = []
        hours: list[int] = []
        labels: list[int] = []

        for impression in data.impressions():
            for news_id, label in zip(impression.candidates, impression.labels):
                user_ids.append(impression.user_id)
                news_ids.append(news_id)
                hours.append(impression.hour)
                labels.append(label)

            if max_rows and len(labels) >= max_rows:
                break

        return self.featurize(user_ids, news_ids, hours), np.asarray(labels, dtype=int)

    def fit(self, data: MindData) -> "PropensityModel":
        features, labels = self.build_dataset(data, int(self.cfg.get("audience.max_train_rows", 0) or 0))

        if len(np.unique(labels)) < 2:
            raise ValueError("training data has a single class; cannot fit a propensity model")

        kind = str(self.cfg.get("audience.model", "logistic")).lower()

        if kind == "gbdt":
            self.scaler = None
            self.model = HistGradientBoostingClassifier(
                random_state=self.cfg.seed,
                max_iter=200,
                learning_rate=0.1,
            ).fit(features, labels)
        else:
            # Scaling matters here: hour spans 0-23 while cosine spans [-1, 1], and an
            # unscaled L2 penalty would regularise the content signal away.
            self.scaler = StandardScaler().fit(features)
            self.model = LogisticRegression(
                C=float(self.cfg.get("audience.c", 1.0)),
                max_iter=1000,
                random_state=self.cfg.seed,
                class_weight="balanced",  # MIND is ~4% positive
            ).fit(self.scaler.transform(features), labels)

        log.info(
            "propensity model (%s) fitted on %d rows, %.2f%% positive",
            kind,
            len(labels),
            100.0 * float(labels.mean()),
        )
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("propensity model is not fitted")
        scaled = self.scaler.transform(features) if self.scaler is not None else features
        return self.model.predict_proba(scaled)[:, 1]

    def score(self, user_ids: list[str], news_ids: list[str], hours: list[int]) -> np.ndarray:
        return self.predict_proba(self.featurize(user_ids, news_ids, hours))

    def evaluate(
        self,
        data: MindData,
        max_rows: int = DEFAULT_EVAL_ROWS,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        features, labels = self.build_dataset(data, max_rows)
        predictions = self.predict_proba(features)
        auc = float(roc_auc_score(labels, predictions)) if len(np.unique(labels)) > 1 else float("nan")
        return auc, labels, predictions

    def build_audience(
        self,
        news_id: str,
        k: int,
        candidate_users: list[str] | None = None,
        hour: int | None = None,
        send_hours: dict[str, int] | None = None,
    ) -> list[tuple[str, float]]:
        """Top-k readers by predicted propensity: the campaign audience.

        Passing `send_hours` scores each reader at the hour they would actually be
        emailed, so audience and send-time resolve as one decision.
        """
        users = candidate_users if candidate_users is not None else list(self.profiles.vectors.keys())
        if not users:
            return []

        if send_hours is not None:
            hours = [send_hours.get(user, DEFAULT_SCORING_HOUR) for user in users]
        else:
            hours = [DEFAULT_SCORING_HOUR if hour is None else hour] * len(users)

        scores = self.score(users, [news_id] * len(users), hours)
        k = min(k, len(users))

        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="mergesort")]
        return [(users[i], float(scores[i])) for i in top]

    def _topic_entropy(self, user_id: str) -> float:
        if user_id not in self._entropy_cache:
            self._entropy_cache[user_id] = profile_topic_entropy(
                user_id, self.profiles, self._news_category
            )
        return self._entropy_cache[user_id]


def precision_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Share of the top-k scored rows that were genuinely clicked.

    ROC-AUC ranks the whole population; this describes the slice we would actually send
    to, which is the operationally relevant question.
    """
    k = min(k, len(scores))
    if k == 0:
        return 0.0
    top = np.argpartition(-scores, k - 1)[:k]
    return float(labels[top].mean())


def train_and_evaluate(
    cfg: Config,
    encoder: ArticleEncoder,
    profiles: UserProfiles,
    train: MindData,
    dev: MindData,
    popularity: dict[str, float],
    catalogue: pd.DataFrame | None = None,
) -> tuple[PropensityModel, AudienceMetrics]:
    """Fit on train, evaluate on dev.

    `catalogue` should be the union of both splits' articles: dev candidates that never
    appear in train still need a category one-hot, or they silently score as an unknown
    desk.
    """
    model = PropensityModel(cfg, encoder, profiles)
    model.fit_catalogue(train.news if catalogue is None else catalogue, popularity)
    model.fit(train)

    auc, labels, predictions = model.evaluate(dev)
    k = int(cfg.require("audience.audience_k"))

    metrics = AudienceMetrics(
        roc_auc=auc,
        precision_at_k=precision_at_k(labels, predictions, k),
        audience_k=k,
        n_train_rows=int(cfg.get("audience.max_train_rows", 0) or 0),
        n_eval_rows=int(len(labels)),
        base_click_rate=float(labels.mean()),
    )

    lift = metrics.precision_at_k / metrics.base_click_rate if metrics.base_click_rate > 0 else float("nan")
    log.info(
        "audience [%s / %s]: ROC-AUC=%.4f precision@%d=%.4f (base rate %.4f, %.1fx lift)",
        dev.split,
        dev.data_source,
        metrics.roc_auc,
        k,
        metrics.precision_at_k,
        metrics.base_click_rate,
        lift,
    )
    return model, metrics
