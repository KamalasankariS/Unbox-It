"""Contextual bandit with Thompson sampling: adaptive content selection.

An A/B test is a one-shot decision procedure. A daily email programme is not: the arms
are every desk you could lead with, the best arm differs per reader, the answer moves
with the news cycle, and every day spent serving a known-bad arm is lost engagement. A
contextual bandit keeps exploring, but explores in proportion to its uncertainty, so it
stops paying to relearn what it already knows.

Linear Thompson sampling. Reward is linear in the context, r ~ N(x'theta_a, sigma^2),
with a Gaussian prior over each arm's theta_a, which makes the posterior conjugate:

    A_a = X_a'X_a / sigma^2 + I / tau^2
    b_a = X_a'r_a / sigma^2
    theta_a | data ~ N(A_a^-1 b_a, A_a^-1)

To act, draw one theta per arm from its posterior and play the arm with the highest
x'theta. An arm gets played when its posterior says it could be the best, so exploration
falls out of the uncertainty rather than being bolted on with an epsilon.

Arms are desks, not individual articles: posteriors need a stable arm set to mean
anything, and articles churn daily while desks do not. The content ranker still picks
the story within the chosen desk.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np

from newspush.config import Config


log = logging.getLogger(__name__)


REGRET_CHECKPOINTS = (0.1, 0.25, 0.5, 0.75, 1.0)


@dataclass
class BanditMetrics:
    thompson_reward: float
    epsilon_greedy_reward: float
    static_reward: float
    random_reward: float
    thompson_regret: float
    epsilon_greedy_regret: float
    random_regret: float
    n_rounds: int
    n_arms: int
    lift_vs_epsilon_greedy: float
    lift_vs_random: float
    regret_curve: list[dict[str, float]]

    def to_dict(self) -> dict:
        return asdict(self)


class LinearThompsonSampling:
    """Linear contextual bandit with a conjugate Gaussian posterior per arm."""

    def __init__(self, n_arms: int, context_dim: int, cfg: Config, seed: int | None = None) -> None:
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.prior_variance = float(cfg.get("bandit.prior_variance", 1.0))
        self.noise_variance = float(cfg.get("bandit.noise_variance", 0.25))
        self.rng = np.random.default_rng(cfg.seed if seed is None else seed)

        # Starting at the prior precision leaves an unplayed arm with a wide posterior,
        # which is what gets it explored.
        self.precision = np.stack([np.eye(context_dim) / self.prior_variance for _ in range(n_arms)])
        self.weighted_rewards = np.zeros((n_arms, context_dim))

        self._covariance = np.stack([np.eye(context_dim) * self.prior_variance for _ in range(n_arms)])
        self._stale = np.zeros(n_arms, dtype=bool)

    def select_arm(self, context: np.ndarray) -> int:
        """Sample one theta per arm from its posterior; play the best-scoring arm."""
        best_arm, best_score = 0, -np.inf

        for arm in range(self.n_arms):
            mean, covariance = self._posterior(arm)
            theta = self.rng.multivariate_normal(mean, covariance, method="cholesky")
            score = float(context @ theta)
            if score > best_score:
                best_arm, best_score = arm, score

        return best_arm

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        """Exact conjugate posterior update: no gradient step, no learning rate."""
        self.precision[arm] += np.outer(context, context) / self.noise_variance
        self.weighted_rewards[arm] += context * reward / self.noise_variance
        self._stale[arm] = True

    def _posterior(self, arm: int) -> tuple[np.ndarray, np.ndarray]:
        """(mean, covariance) of theta_a, inverting lazily after that arm updates.

        The covariance is copied out. `self._covariance[arm]` is a view into a 3-D
        array, so handing it back directly would let a caller's reference mutate the
        next time this arm is updated.
        """
        if self._stale[arm]:
            self._covariance[arm] = np.linalg.inv(self.precision[arm])
            self._stale[arm] = False

        covariance = self._covariance[arm].copy()
        return covariance @ self.weighted_rewards[arm], covariance


class EpsilonGreedy:
    """Baseline: per-arm linear regression, exploring uniformly with probability epsilon.

    It explores as hard on the final round as on the first, long after it has learned
    which arms are bad. The regret gap against Thompson sampling is the size of that tax.
    """

    def __init__(self, n_arms: int, context_dim: int, cfg: Config, seed: int | None = None) -> None:
        self.n_arms = n_arms
        self.epsilon = float(cfg.get("bandit.epsilon", 0.1))
        self.rng = np.random.default_rng((cfg.seed if seed is None else seed) + 1)

        self.precision = np.stack([np.eye(context_dim) for _ in range(n_arms)])
        self.weighted_rewards = np.zeros((n_arms, context_dim))

    def select_arm(self, context: np.ndarray) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_arms))

        scores = [
            float(context @ np.linalg.solve(self.precision[arm], self.weighted_rewards[arm]))
            for arm in range(self.n_arms)
        ]
        return int(np.argmax(scores))

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        self.precision[arm] += np.outer(context, context)
        self.weighted_rewards[arm] += context * reward


class StaticGreedy:
    """Context-free baseline: always play the arm with the best running average.

    This is what "send everyone the best-performing desk" looks like. Optimistic
    initialisation makes it try each arm once before settling.
    """

    def __init__(self, n_arms: int) -> None:
        self.counts = np.zeros(n_arms)
        self.totals = np.zeros(n_arms)
        self.n_arms = n_arms

    def select_arm(self) -> int:
        means = np.divide(
            self.totals, self.counts, out=np.full(self.n_arms, np.inf), where=self.counts > 0
        )
        return int(np.argmax(means))

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        self.totals[arm] += reward


def run_bandit_experiment(
    cfg: Config,
    contexts: np.ndarray,
    arm_rewards: np.ndarray,
    seed: int | None = None,
) -> BanditMetrics:
    """Run Thompson sampling against three baselines on one reward stream.

    Args:
        contexts: (n_rounds, context_dim).
        arm_rewards: (n_rounds, n_arms) true expected reward of every arm each round.
            The learners never see this. They observe only the realised Bernoulli reward
            of the single arm they chose, which is the whole difficulty of the bandit
            setting. It is used to draw that reward and to compute regret in hindsight.
    """
    n_rounds, n_arms = arm_rewards.shape
    rng = np.random.default_rng((cfg.seed if seed is None else seed) + 99)

    thompson = LinearThompsonSampling(n_arms, contexts.shape[1], cfg, seed)
    epsilon_greedy = EpsilonGreedy(n_arms, contexts.shape[1], cfg, seed)
    static = StaticGreedy(n_arms)

    rewards = {"thompson": 0.0, "epsilon_greedy": 0.0, "static": 0.0, "random": 0.0}
    regrets = {"thompson": 0.0, "epsilon_greedy": 0.0, "random": 0.0}

    # Thompson sampling is an asymptotic method: it pays its exploration cost up front
    # and recoups it as its regret flattens while epsilon-greedy's keeps growing at a
    # constant rate. A single end-of-run number hides that, so record the curve and let
    # the crossover (or its absence) be visible.
    checkpoints = {max(1, int(fraction * n_rounds)) for fraction in REGRET_CHECKPOINTS}
    regret_curve: list[dict[str, float]] = []

    for round_index in range(n_rounds):
        context = contexts[round_index]
        true_rewards = arm_rewards[round_index]
        best_possible = float(true_rewards.max())

        arm = thompson.select_arm(context)
        reward = float(rng.random() < true_rewards[arm])
        thompson.update(arm, context, reward)
        rewards["thompson"] += reward
        regrets["thompson"] += best_possible - float(true_rewards[arm])

        arm = epsilon_greedy.select_arm(context)
        reward = float(rng.random() < true_rewards[arm])
        epsilon_greedy.update(arm, context, reward)
        rewards["epsilon_greedy"] += reward
        regrets["epsilon_greedy"] += best_possible - float(true_rewards[arm])

        arm = static.select_arm()
        reward = float(rng.random() < true_rewards[arm])
        static.update(arm, reward)
        rewards["static"] += reward

        arm = int(rng.integers(0, n_arms))
        reward = float(rng.random() < true_rewards[arm])
        rewards["random"] += reward
        regrets["random"] += best_possible - float(true_rewards[arm])

        if round_index + 1 in checkpoints:
            regret_curve.append(
                {
                    "round": round_index + 1,
                    "thompson_regret": regrets["thompson"],
                    "epsilon_greedy_regret": regrets["epsilon_greedy"],
                    "random_regret": regrets["random"],
                }
            )

    metrics = BanditMetrics(
        thompson_reward=rewards["thompson"],
        epsilon_greedy_reward=rewards["epsilon_greedy"],
        static_reward=rewards["static"],
        random_reward=rewards["random"],
        thompson_regret=regrets["thompson"],
        epsilon_greedy_regret=regrets["epsilon_greedy"],
        random_regret=regrets["random"],
        n_rounds=n_rounds,
        n_arms=n_arms,
        lift_vs_epsilon_greedy=_relative_lift(rewards["thompson"], rewards["epsilon_greedy"]),
        lift_vs_random=_relative_lift(rewards["thompson"], rewards["random"]),
        regret_curve=regret_curve,
    )

    log.info(
        "bandit (%d rounds, %d arms): cumulative reward TS=%.0f eps-greedy=%.0f static=%.0f random=%.0f | "
        "regret TS=%.0f eps-greedy=%.0f random=%.0f",
        n_rounds,
        n_arms,
        metrics.thompson_reward,
        metrics.epsilon_greedy_reward,
        metrics.static_reward,
        metrics.random_reward,
        metrics.thompson_regret,
        metrics.epsilon_greedy_regret,
        metrics.random_regret,
    )
    return metrics


def _relative_lift(value: float, baseline: float) -> float:
    return (value - baseline) / baseline if baseline > 0 else float("nan")
