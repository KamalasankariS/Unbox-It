"""Fatigue-aware targeting: unsubscribe risk and frequency capping.

MIND contains impressions and clicks but no unsubscribes, so this module does not claim
to predict unsubscribes from data. Instead it states an explicit parametric risk model,
keeps its parameters in config, and labels it as assumed wherever it surfaces:

    risk(u) = base * exp(slope * (sends - 1)) * (2 - engagement_percentile(u))

The engagement percentile is estimated from MIND, so the model's ordering of who is at
risk is data-grounded even though its absolute scale is not. `fit_engagement` is where
real unsubscribe logs would replace the assumption.

What is fully measurable is the cost of capping: how many sends are suppressed and how
much predicted engagement that gives up. Those are the reported metrics.

The reason the module exists at all: a system that optimises opens with no notion of
fatigue will learn to send more email indefinitely, and will look excellent until the
list burns down.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from newspush.config import Config

log = logging.getLogger(__name__)

CTR_SHRINKAGE_PRIOR = 10.0  # pseudo-impressions, so a 1-for-1 reader is not top-ranked
MEDIAN_PERCENTILE = 0.5


@dataclass
class FatigueMetrics:
    n_sends_proposed: int
    n_sends_suppressed: int
    suppression_rate: float
    users_at_cap: int
    users_above_risk_threshold: int
    mean_risk_before: float
    mean_risk_after: float
    engagement_retained: float
    max_emails_per_week: int
    risk_model: str

    def to_dict(self) -> dict:
        return asdict(self)


class FatigueModel:
    """Unsubscribe-risk scoring and frequency capping over a proposed send plan."""

    RISK_MODEL = (
        "assumed parametric: base * exp(slope * (sends - 1)) * (2 - engagement_percentile); "
        "not fitted to unsubscribe labels, which MIND does not contain"
    )

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.max_per_week = int(cfg.require("fatigue.max_emails_per_week"))
        self.base_rate = float(cfg.require("fatigue.unsubscribe_base_rate"))
        self.slope = float(cfg.require("fatigue.fatigue_slope"))
        self.risk_threshold = float(cfg.require("fatigue.risk_threshold"))
        self._engagement_percentile: dict[str, float] = {}

    def fit_engagement(self, user_stats: pd.DataFrame) -> "FatigueModel":
        """Estimate each reader's engagement percentile from MIND.

        Args:
            user_stats: (user_id, impressions, clicks) rollup.

        Raw CTR is shrunk toward the population rate before ranking, so a reader with
        one impression and one click does not land at the 100th percentile.
        """
        required = {"user_id", "impressions", "clicks"}
        missing = required - set(user_stats.columns)
        if missing:
            raise ValueError(f"user_stats missing columns: {sorted(missing)}")

        stats = user_stats.copy()
        total_impressions = float(stats["impressions"].sum())
        global_rate = float(stats["clicks"].sum()) / total_impressions if total_impressions > 0 else 0.0

        shrunk_ctr = (stats["clicks"] + CTR_SHRINKAGE_PRIOR * global_rate) / (
            stats["impressions"] + CTR_SHRINKAGE_PRIOR
        )
        percentiles = shrunk_ctr.rank(pct=True)

        self._engagement_percentile = dict(zip(stats["user_id"].astype(str), percentiles.astype(float)))

        log.info(
            "fatigue: engagement percentiles for %d users (population CTR %.4f)",
            len(self._engagement_percentile),
            global_rate,
        )
        return self

    def engagement_percentile(self, user_id: str) -> float:
        """0 is least engaged, 1 is most. Unknown readers sit at the median."""
        return self._engagement_percentile.get(user_id, MEDIAN_PERCENTILE)

    def unsubscribe_risk(self, user_id: str, sends_this_week: int) -> float:
        """Unsubscribe probability under the assumed risk model."""
        excess_sends = max(0, sends_this_week - 1)
        engagement_factor = 2.0 - self.engagement_percentile(user_id)
        risk = self.base_rate * np.exp(self.slope * excess_sends) * engagement_factor
        return float(min(risk, 1.0))

    def should_send(self, user_id: str, sends_this_week: int) -> bool:
        """Two independent guardrails: a hard volume cap and a risk ceiling.

        The hard cap does not depend on believing the risk model. Even if the parametric
        form is wrong, nobody receives more than `max_emails_per_week`.
        """
        if sends_this_week >= self.max_per_week:
            return False
        return self.unsubscribe_risk(user_id, sends_this_week + 1) <= self.risk_threshold

    def apply_cap(self, send_plan: pd.DataFrame) -> tuple[pd.DataFrame, FatigueMetrics]:
        """Filter a proposed send plan down to what is safe to send.

        Args:
            send_plan: (user_id, news_id, score), where score is the propensity of an
                open. Sorted by descending score within each reader, so capping keeps
                their best sends and drops their marginal ones.
        """
        required = {"user_id", "news_id", "score"}
        missing = required - set(send_plan.columns)
        if missing:
            raise ValueError(f"send_plan missing columns: {sorted(missing)}")
        if send_plan.empty:
            raise ValueError("send_plan is empty")

        plan = send_plan.sort_values(["user_id", "score"], ascending=[True, False]).reset_index(drop=True)

        keep = np.zeros(len(plan), dtype=bool)
        sends_by_user: dict[str, int] = {}
        users_at_cap: set[str] = set()
        users_over_risk: set[str] = set()

        for i, row in enumerate(plan.itertuples(index=False)):
            user = str(row.user_id)
            sent = sends_by_user.get(user, 0)

            if self.should_send(user, sent):
                keep[i] = True
                sends_by_user[user] = sent + 1
            elif sent >= self.max_per_week:
                users_at_cap.add(user)
            else:
                users_over_risk.add(user)

        capped = plan[keep].reset_index(drop=True)

        proposed_counts = plan.groupby("user_id").size()
        risk_before = [self.unsubscribe_risk(str(user), int(n)) for user, n in proposed_counts.items()]
        risk_after = [self.unsubscribe_risk(user, n) for user, n in sends_by_user.items()]

        proposed_opens = float(plan["score"].sum())
        kept_opens = float(capped["score"].sum())

        metrics = FatigueMetrics(
            n_sends_proposed=len(plan),
            n_sends_suppressed=int((~keep).sum()),
            suppression_rate=float((~keep).mean()),
            users_at_cap=len(users_at_cap),
            users_above_risk_threshold=len(users_over_risk),
            mean_risk_before=float(np.mean(risk_before)) if risk_before else 0.0,
            mean_risk_after=float(np.mean(risk_after)) if risk_after else 0.0,
            engagement_retained=kept_opens / proposed_opens if proposed_opens > 0 else 0.0,
            max_emails_per_week=self.max_per_week,
            risk_model=self.RISK_MODEL,
        )

        log.info(
            "fatigue cap: suppressed %d of %d sends (%.1f%%), retaining %.1f%% of expected opens; "
            "mean modelled risk %.4f -> %.4f",
            metrics.n_sends_suppressed,
            metrics.n_sends_proposed,
            100.0 * metrics.suppression_rate,
            100.0 * metrics.engagement_retained,
            metrics.mean_risk_before,
            metrics.mean_risk_after,
        )
        return capped, metrics
