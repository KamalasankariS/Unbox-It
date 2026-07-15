"""A/B testing: two-proportion z-test, and a simulated campaign that uses it.

`ab_test` is pure statistics with no dependency on the rest of the project and would
run unchanged against a real campaign's open counts. `simulate_campaign` runs the
experiment we cannot run for real, drawing opens from the `ResponseSimulator`.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats

from newspush.config import Config
from newspush.experiment.simulator import ResponseSimulator
from newspush.features.users import UserProfiles
from newspush.models.content_selection import ContentRanker
from newspush.models.send_time import SendTimeModel

log = logging.getLogger(__name__)

CONTROL_HOUR = 12


@dataclass
class ABResult:
    control_n: int
    control_successes: int
    control_rate: float
    treatment_n: int
    treatment_successes: int
    treatment_rate: float
    absolute_lift: float
    relative_lift: float
    z_statistic: float
    p_value: float
    ci_low: float
    ci_high: float
    significant: bool
    alpha: float

    def to_dict(self) -> dict:
        return asdict(self)


def ab_test(
    control_successes: int,
    control_n: int,
    treatment_successes: int,
    treatment_n: int,
    alpha: float = 0.05,
) -> ABResult:
    """Two-proportion z-test with a confidence interval on the difference in rates.

    The test statistic uses the pooled proportion, which is the right variance estimate
    under the null of equal rates. The interval uses the unpooled standard error, which
    is right when estimating a difference that is not assumed to be zero. Using one for
    both is the classic bug, and produces a p-value and an interval that disagree.
    """
    if control_n <= 0 or treatment_n <= 0:
        raise ValueError("both arms need at least one observation")
    if not 0 <= control_successes <= control_n or not 0 <= treatment_successes <= treatment_n:
        raise ValueError("successes must lie within [0, n]")

    control_rate = control_successes / control_n
    treatment_rate = treatment_successes / treatment_n
    difference = treatment_rate - control_rate

    pooled = (control_successes + treatment_successes) / (control_n + treatment_n)
    pooled_se = math.sqrt(pooled * (1.0 - pooled) * (1.0 / control_n + 1.0 / treatment_n))

    if pooled_se == 0:
        z_statistic, p_value = 0.0, 1.0
    else:
        z_statistic = difference / pooled_se
        p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(z_statistic))))

    unpooled_se = math.sqrt(
        control_rate * (1.0 - control_rate) / control_n
        + treatment_rate * (1.0 - treatment_rate) / treatment_n
    )
    critical_value = float(stats.norm.ppf(1.0 - alpha / 2.0))

    return ABResult(
        control_n=control_n,
        control_successes=control_successes,
        control_rate=control_rate,
        treatment_n=treatment_n,
        treatment_successes=treatment_successes,
        treatment_rate=treatment_rate,
        absolute_lift=difference,
        relative_lift=(difference / control_rate) if control_rate > 0 else float("nan"),
        z_statistic=float(z_statistic),
        p_value=p_value,
        ci_low=float(difference - critical_value * unpooled_se),
        ci_high=float(difference + critical_value * unpooled_se),
        significant=bool(p_value < alpha),
        alpha=alpha,
    )


@dataclass
class SimulatedCampaign:
    """An A/B result plus the provenance needed to read it correctly."""

    result: ABResult
    control_policy: str
    treatment_policy: str
    emails_per_user: int
    environment: str

    def to_dict(self) -> dict:
        return {
            "control_policy": self.control_policy,
            "treatment_policy": self.treatment_policy,
            "emails_per_user": self.emails_per_user,
            "environment": self.environment,
            **self.result.to_dict(),
        }


def simulate_campaign(
    cfg: Config,
    simulator: ResponseSimulator,
    ranker: ContentRanker,
    profiles: UserProfiles,
    popularity: dict[str, float],
    send_time: SendTimeModel | None = None,
    candidate_pool: list[str] | None = None,
) -> SimulatedCampaign:
    """Simulate a campaign: popularity (control) against personalisation (treatment).

    Readers are randomly assigned to disjoint arms, which is what makes the z-test on
    the outcome valid. Passing `send_time` additionally sends the treatment arm at each
    reader's predicted best hour, so the measured lift covers content and timing
    together; the pipeline reports it both ways to separate the two.
    """
    rng = np.random.default_rng(cfg.seed)
    emails_per_user = int(cfg.require("ab_test.emails_per_user"))
    alpha = float(cfg.get("ab_test.alpha", 0.05))

    users = list(profiles.vectors.keys())
    n_per_arm = int(cfg.require("ab_test.n_users_per_arm"))

    if len(users) < 2 * n_per_arm:
        n_per_arm = len(users) // 2
        log.warning("only %d users available; shrinking each arm to %d", len(users), n_per_arm)
    if n_per_arm < 1:
        raise ValueError("not enough users to run an A/B simulation")

    assigned = rng.choice(len(users), size=2 * n_per_arm, replace=False)
    control_users = [users[i] for i in assigned[:n_per_arm]]
    treatment_users = [users[i] for i in assigned[n_per_arm:]]

    pool = candidate_pool if candidate_pool is not None else list(popularity.keys())
    fixed_hour = int(send_time.global_best_hour) if send_time is not None else CONTROL_HOUR

    most_popular = sorted(pool, key=lambda news_id: popularity.get(news_id, 0.0), reverse=True)
    most_popular = most_popular[:emails_per_user]

    control_sends = [
        (user, news_id, fixed_hour) for user in control_users for news_id in most_popular
    ]

    treatment_sends: list[tuple[str, str, int]] = []
    for user in treatment_users:
        hour = int(send_time.best_hour(user)) if send_time is not None else fixed_hour
        recommendations = ranker.recommend_for_user(
            user, top_n=emails_per_user, exclude_history=True, candidate_pool=pool
        )
        treatment_sends.extend((user, news_id, hour) for news_id, _ in recommendations)

    if not control_sends or not treatment_sends:
        raise ValueError("an arm produced no sends; check the candidate pool")

    # Independent draw streams, so the two arms are not correlated through shared state.
    simulator.reseed(cfg.seed + 1)
    control_opens = simulator.sample_clicks(*_unzip_sends(control_sends))
    simulator.reseed(cfg.seed + 2)
    treatment_opens = simulator.sample_clicks(*_unzip_sends(treatment_sends))

    result = ab_test(
        control_successes=int(control_opens.sum()),
        control_n=len(control_opens),
        treatment_successes=int(treatment_opens.sum()),
        treatment_n=len(treatment_opens),
        alpha=alpha,
    )

    treatment_policy = "personalised-content" + ("+send-time" if send_time is not None else "")
    log.info(
        "A/B (simulated): control=%.4f (n=%d) vs %s=%.4f (n=%d), lift %+.4f (%+.1f%%), "
        "p=%.3g, 95%% CI [%+.4f, %+.4f], %s",
        result.control_rate,
        result.control_n,
        treatment_policy,
        result.treatment_rate,
        result.treatment_n,
        result.absolute_lift,
        100.0 * result.relative_lift,
        result.p_value,
        result.ci_low,
        result.ci_high,
        "significant" if result.significant else "not significant",
    )

    return SimulatedCampaign(
        result=result,
        control_policy="popularity top-N at the global best hour",
        treatment_policy=treatment_policy,
        emails_per_user=emails_per_user,
        environment="ResponseSimulator fitted on held-out dev; not a live test",
    )


def _unzip_sends(sends: list[tuple[str, str, int]]) -> tuple[list[str], list[str], list[int]]:
    users, news_ids, hours = zip(*sends)
    return list(users), list(news_ids), list(hours)
