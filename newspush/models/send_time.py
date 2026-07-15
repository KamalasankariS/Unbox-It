"""Send-time optimisation: which hour to email each reader.

Per-user hourly data is thin, so the per-user rate is shrunk toward the global hourly
curve with a Beta prior:

    rate(u, h) = (clicks(u, h) + s * global_rate(h)) / (impressions(u, h) + s)

Evidence washes the prior out as it accumulates; a reader with none inherits the
population's rhythm. Readers below `min_events_for_personal` are not personalised at
all, which keeps a single 3am click from defining a 3am reader.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from newspush.config import Config
from newspush.data.schema import MindData

log = logging.getLogger(__name__)


@dataclass
class SendTimeMetrics:
    best_hour_rate: float
    baseline_rate: float
    absolute_uplift: float
    relative_uplift: float
    n_users_personalised: int
    n_users_global_fallback: int
    global_best_hour: int
    n_eval_impressions: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class SendTimeModel:
    """Per-user best send hour, shrunk toward the global hourly curve."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.n_hours = int(cfg.get("send_time.n_hours", 24))
        self.prior_alpha = float(cfg.get("send_time.prior_alpha", 1.0))
        self.prior_strength = float(cfg.get("send_time.prior_strength", 20.0))
        self.min_events = int(cfg.get("send_time.min_events_for_personal", 5))

        self.global_rate_by_hour = np.zeros(self.n_hours)
        self.global_best_hour = 12

        self._best_hour: dict[str, int] = {}
        self._user_rates: dict[str, np.ndarray] = {}
        self._n_personalised = 0

    def fit(self, user_hour_counts: pd.DataFrame) -> "SendTimeModel":
        """Fit from a (user_id, hour, impressions, clicks) rollup.

        Consuming the SQL aggregate rather than raw events is what lets this scale to
        MIND-large unchanged.
        """
        required = {"user_id", "hour", "impressions", "clicks"}
        missing = required - set(user_hour_counts.columns)
        if missing:
            raise ValueError(f"user_hour_counts missing columns: {sorted(missing)}")

        counts = user_hour_counts.copy()
        counts["hour"] = counts["hour"].astype(int)

        self._fit_global_curve(counts)

        for user_id, group in counts.groupby("user_id"):
            impressions, clicks = self._to_hour_arrays(group)

            if impressions.sum() < self.min_events:
                self._best_hour[str(user_id)] = self.global_best_hour
                continue

            shrunk = (clicks + self.prior_strength * self.global_rate_by_hour) / (
                impressions + self.prior_strength
            )
            self._user_rates[str(user_id)] = shrunk
            self._best_hour[str(user_id)] = int(np.argmax(shrunk))
            self._n_personalised += 1

        log.info(
            "send-time model: %d users personalised, %d on the global fallback; global best hour %d",
            self._n_personalised,
            len(self._best_hour) - self._n_personalised,
            self.global_best_hour,
        )
        return self

    def best_hour(self, user_id: str) -> int:
        """Predicted best hour, falling back to the global best for unknown readers."""
        return self._best_hour.get(user_id, self.global_best_hour)

    def best_hours(self, user_ids: list[str]) -> dict[str, int]:
        return {user_id: self.best_hour(user_id) for user_id in user_ids}

    def rate_by_hour(self, user_id: str) -> np.ndarray:
        """The reader's smoothed engagement curve, or the global one if not personalised."""
        return self._user_rates.get(user_id, self.global_rate_by_hour)

    def is_personalised(self, user_id: str) -> bool:
        """True if this reader had enough evidence for their own curve."""
        return user_id in self._user_rates

    @property
    def n_personalised(self) -> int:
        return self._n_personalised

    @property
    def n_fallback(self) -> int:
        return len(self._best_hour) - self._n_personalised

    def _fit_global_curve(self, counts: pd.DataFrame) -> None:
        by_hour = counts.groupby("hour")[["impressions", "clicks"]].sum()
        impressions, clicks = self._to_hour_arrays(by_hour.reset_index())

        total_impressions = impressions.sum()
        overall_rate = float(clicks.sum() / total_impressions) if total_impressions > 0 else 0.0

        if overall_rate <= 0:
            self.global_rate_by_hour = np.zeros(self.n_hours)
            self.global_best_hour = 12
            return

        # Laplace smoothing, so an hour with three impressions cannot define the curve.
        pseudo_impressions = self.prior_alpha / overall_rate
        self.global_rate_by_hour = (clicks + self.prior_alpha) / (impressions + pseudo_impressions)
        self.global_best_hour = int(np.argmax(self.global_rate_by_hour))

    def _to_hour_arrays(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        impressions = np.zeros(self.n_hours)
        clicks = np.zeros(self.n_hours)

        for row in frame.itertuples(index=False):
            hour = int(row.hour)
            if 0 <= hour < self.n_hours:
                impressions[hour] = float(row.impressions)
                clicks[hour] = float(row.clicks)

        return impressions, clicks


def evaluate(model: SendTimeModel, dev: MindData) -> SendTimeMetrics:
    """Open-rate uplift on held-out data.

    A MIND impression cannot be re-sent at a different hour, so this compares the click
    rate of impressions that happened to arrive in a reader's predicted best hour
    against the same population's rate at other hours. It is an observational estimate,
    not a randomised one.
    """
    hit_clicks = hit_impressions = 0
    other_clicks = other_impressions = 0
    n_evaluated = 0

    for impression in dev.impressions():
        n_evaluated += 1
        clicks = int(sum(impression.labels))
        shown = len(impression.labels)

        if impression.hour == model.best_hour(impression.user_id):
            hit_clicks += clicks
            hit_impressions += shown
        else:
            other_clicks += clicks
            other_impressions += shown

    best_rate = hit_clicks / hit_impressions if hit_impressions else 0.0
    baseline_rate = other_clicks / other_impressions if other_impressions else 0.0
    absolute = best_rate - baseline_rate

    metrics = SendTimeMetrics(
        best_hour_rate=float(best_rate),
        baseline_rate=float(baseline_rate),
        absolute_uplift=float(absolute),
        relative_uplift=float(absolute / baseline_rate) if baseline_rate > 0 else float("nan"),
        n_users_personalised=model.n_personalised,
        n_users_global_fallback=model.n_fallback,
        global_best_hour=model.global_best_hour,
        n_eval_impressions=n_evaluated,
    )

    log.info(
        "send-time [%s / %s]: best-hour CTR=%.4f vs other-hours CTR=%.4f (absolute %+.4f)",
        dev.split,
        dev.data_source,
        metrics.best_hour_rate,
        metrics.baseline_rate,
        metrics.absolute_uplift,
    )
    return metrics
