"""The response simulator: a counterfactual environment fitted on logged data.

MIND records what readers did when shown the articles they were actually shown. It
cannot say what they would have done under a different article at a different hour,
which is exactly what an A/B test needs to know. So a response model is fitted on
held-out data and treated as an oracle P(click | user, article, hour) that can be
queried for any counterfactual.

Every A/B, off-policy and bandit number in this project is measured inside this
simulator and is labelled as such. The simulator is calibrated on real MIND clicks, so
the ordering it induces over policies is informative, but its absolute lifts inherit
its own model error and are not live-test results.

The oracle trains on dev; the policies that compete inside it train on train. If a
competing policy were the oracle itself, its winning would be a tautology.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from newspush.config import Config
from newspush.data.schema import MindData
from newspush.features.text import ArticleEncoder
from newspush.features.users import UserProfiles
from newspush.models.audience import PropensityModel

log = logging.getLogger(__name__)


class ResponseSimulator:
    """Oracle P(click | user, article, hour), fitted on a held-out split."""

    def __init__(self, cfg: Config, oracle: PropensityModel) -> None:
        self.cfg = cfg
        self.oracle = oracle
        self.rng = np.random.default_rng(cfg.seed)

    @classmethod
    def fit(
        cls,
        cfg: Config,
        encoder: ArticleEncoder,
        profiles: UserProfiles,
        holdout: MindData,
        popularity: dict[str, float],
        catalogue: pd.DataFrame | None = None,
    ) -> "ResponseSimulator":
        """Fit the oracle on a split the competing policies did not train on."""
        oracle = PropensityModel(cfg, encoder, profiles)
        oracle.fit_catalogue(holdout.news if catalogue is None else catalogue, popularity)
        oracle.fit(holdout)

        log.info(
            "response simulator fitted on split=%s (%s); all A/B and bandit results are "
            "simulated under this environment, not measured on live traffic",
            holdout.split,
            holdout.data_source,
        )
        return cls(cfg, oracle)

    def click_prob(self, user_ids: list[str], news_ids: list[str], hours: list[int]) -> np.ndarray:
        """Click probability for arbitrary counterfactual triples."""
        return self.oracle.score(user_ids, news_ids, hours)

    def sample_clicks(self, user_ids: list[str], news_ids: list[str], hours: list[int]) -> np.ndarray:
        """Draw Bernoulli opens from the oracle."""
        probabilities = self.click_prob(user_ids, news_ids, hours)
        return (self.rng.random(size=probabilities.shape) < probabilities).astype(int)

    def reseed(self, seed: int) -> None:
        """Reset the draw stream, so each experiment arm gets an independent one."""
        self.rng = np.random.default_rng(seed)
