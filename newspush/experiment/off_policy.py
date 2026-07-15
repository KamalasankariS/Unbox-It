"""Off-policy evaluation: estimate a new send policy from logged data, without sending.

An email A/B test is expensive in a way an on-site test is not: you cannot un-send an
email, the feedback loop is days long, and a bad arm costs unsubscribes rather than
clicks. Off-policy evaluation answers "is this policy better" from data logged under a
different one, before anyone's inbox is involved.

Three estimators, in increasing order of how much they should be trusted:

    IPS     unbiased, high variance. A rare logged action that the target policy loves
            produces an enormous weight, and one lucky click dominates the estimate.
    SNIPS   normalises by the realised weight mass. Slightly biased, far lower variance,
            and robust to the weights being systematically off-scale.
    DR      importance-weights only the residual of a learned reward model, so it is
            consistent if either the propensities or the reward model are right.

Because the environment is a simulator, the true value of the target policy is
computable here, which is what makes this module evidence rather than decoration: each
estimator can be scored against a truth it was never shown.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.linear_model import Ridge

from newspush.config import Config
from newspush.experiment.simulator import ResponseSimulator
from newspush.features.text import ArticleEncoder
from newspush.features.users import UserProfiles

log = logging.getLogger(__name__)

Z_95 = 1.96
MIN_PROPENSITY = 1e-12


@dataclass
class OPEEstimate:
    estimator: str
    value: float
    std_error: float
    ci_low: float
    ci_high: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OPEReport:
    ips: OPEEstimate
    snips: OPEEstimate
    dr: OPEEstimate
    logged_policy_value: float
    target_true_value: float
    n_logged: int
    effective_sample_size: float
    clip_weight: float
    logging_epsilon: float

    def to_dict(self) -> dict:
        return {
            "ips": self.ips.to_dict(),
            "snips": self.snips.to_dict(),
            "dr": self.dr.to_dict(),
            "logged_policy_value": self.logged_policy_value,
            "target_true_value": self.target_true_value,
            "n_logged": self.n_logged,
            "effective_sample_size": self.effective_sample_size,
            "clip_weight": self.clip_weight,
            "logging_epsilon": self.logging_epsilon,
            "abs_error": {
                "ips": abs(self.ips.value - self.target_true_value),
                "snips": abs(self.snips.value - self.target_true_value),
                "dr": abs(self.dr.value - self.target_true_value),
            },
        }


def importance_weights(
    target_probs: np.ndarray,
    logging_probs: np.ndarray,
    clip: float = np.inf,
) -> np.ndarray:
    return np.clip(target_probs / np.maximum(logging_probs, MIN_PROPENSITY), 0.0, clip)


def ips(
    rewards: np.ndarray,
    target_probs: np.ndarray,
    logging_probs: np.ndarray,
    clip: float = np.inf,
) -> OPEEstimate:
    """Inverse propensity scoring."""
    values = importance_weights(target_probs, logging_probs, clip) * rewards
    return _estimate("IPS", values)


def snips(
    rewards: np.ndarray,
    target_probs: np.ndarray,
    logging_probs: np.ndarray,
    clip: float = np.inf,
) -> OPEEstimate:
    """Self-normalised IPS: divide by the realised weight mass rather than by n."""
    weights = importance_weights(target_probs, logging_probs, clip)
    weight_mass = float(weights.sum())

    if weight_mass == 0:
        return OPEEstimate("SNIPS", 0.0, 0.0, 0.0, 0.0)

    value = float((weights * rewards).sum() / weight_mass)

    # Delta-method standard error for a ratio estimator.
    residual = weights * (rewards - value)
    std_error = float(np.sqrt((residual**2).sum()) / weight_mass) if len(weights) > 1 else 0.0

    return OPEEstimate("SNIPS", value, std_error, value - Z_95 * std_error, value + Z_95 * std_error)


def doubly_robust(
    rewards: np.ndarray,
    target_probs: np.ndarray,
    logging_probs: np.ndarray,
    q_logged_action: np.ndarray,
    q_target_action: np.ndarray,
    clip: float = np.inf,
) -> OPEEstimate:
    """Model-based baseline plus an importance-weighted correction on its residual."""
    weights = importance_weights(target_probs, logging_probs, clip)
    values = q_target_action + weights * (rewards - q_logged_action)
    return _estimate("DR", values)


def effective_sample_size(
    target_probs: np.ndarray,
    logging_probs: np.ndarray,
    clip: float = np.inf,
) -> float:
    """Kish effective sample size of the importance weights.

    The health check to report next to any estimate: if 50,000 logged events yield an
    ESS of 40, the estimate rests on 40 events' worth of information.
    """
    weights = importance_weights(target_probs, logging_probs, clip)
    sum_of_squares = float((weights**2).sum())
    return float((weights.sum() ** 2) / sum_of_squares) if sum_of_squares > 0 else 0.0


def run_off_policy_evaluation(
    cfg: Config,
    simulator: ResponseSimulator,
    encoder: ArticleEncoder,
    profiles: UserProfiles,
    popularity: dict[str, float],
    candidate_pool: list[str],
    n_contexts: int = 5000,
    n_actions: int = 10,
) -> OPEReport:
    """Log data under an epsilon-greedy policy, evaluate the personalised policy off it.

        context  a reader, a slate of candidate articles, and a send hour
        action   which article of the slate to email
        reward   a simulated open
        logging  epsilon-greedy on popularity, so its propensities are known exactly.
                 A deterministic logging policy would make off-policy evaluation
                 impossible, which is the reason production loggers are stochastic.
        target   deterministic argmax of cosine(profile, article): the ranker this
                 project proposes to ship.
    """
    rng = np.random.default_rng(cfg.seed + 7)
    epsilon = float(cfg.require("off_policy.logging_epsilon"))
    clip = float(cfg.require("off_policy.clip_weight"))

    users = list(profiles.vectors.keys())
    if not users:
        raise ValueError("no user profiles available for off-policy evaluation")

    pool = np.array(candidate_pool)
    n_contexts = min(n_contexts, len(users) * 4)
    n_actions = min(n_actions, len(pool))

    context_users = [users[i] for i in rng.integers(0, len(users), size=n_contexts)]
    context_hours = rng.integers(0, 24, size=n_contexts).tolist()
    slates = np.stack([rng.choice(len(pool), size=n_actions, replace=False) for _ in range(n_contexts)])

    popularity_scores = np.array([popularity.get(news_id, 0.0) for news_id in pool])

    logged_actions = np.zeros(n_contexts, dtype=int)
    logging_probs = np.zeros(n_contexts)
    target_actions = np.zeros(n_contexts, dtype=int)
    target_probs = np.zeros(n_contexts)

    for i in range(n_contexts):
        slate = slates[i]

        # Epsilon-greedy over popularity: P(greedy) = 1 - eps + eps/K, P(other) = eps/K.
        greedy_action = int(np.argmax(popularity_scores[slate]))
        action_probs = np.full(n_actions, epsilon / n_actions)
        action_probs[greedy_action] += 1.0 - epsilon

        action = int(rng.choice(n_actions, p=action_probs))
        logged_actions[i] = action
        logging_probs[i] = action_probs[action]

        # The target policy is deterministic, so its action probability is 0 or 1.
        scores = encoder.vecs(pool[slate].tolist()) @ profiles.get(context_users[i])
        target_action = int(np.argmax(scores))
        target_actions[i] = target_action
        target_probs[i] = 1.0 if action == target_action else 0.0

    logged_news = [str(pool[slates[i, logged_actions[i]]]) for i in range(n_contexts)]
    target_news = [str(pool[slates[i, target_actions[i]]]) for i in range(n_contexts)]

    simulator.reseed(cfg.seed + 11)
    rewards = simulator.sample_clicks(context_users, logged_news, context_hours).astype(float)

    # The reward model for DR is fitted on the logged data alone. Using the oracle here
    # would assume away the very thing DR is meant to demonstrate.
    reward_model = _fit_reward_model(cfg, encoder, profiles, context_users, logged_news, context_hours, rewards)
    q_logged = _predict_reward(reward_model, encoder, profiles, context_users, logged_news, context_hours)
    q_target = _predict_reward(reward_model, encoder, profiles, context_users, target_news, context_hours)

    ips_estimate = ips(rewards, target_probs, logging_probs, clip)
    snips_estimate = snips(rewards, target_probs, logging_probs, clip)
    dr_estimate = doubly_robust(rewards, target_probs, logging_probs, q_logged, q_target, clip)

    # Available only inside a simulator: the value the estimators are trying to recover.
    true_value = float(simulator.click_prob(context_users, target_news, context_hours).mean())

    report = OPEReport(
        ips=ips_estimate,
        snips=snips_estimate,
        dr=dr_estimate,
        logged_policy_value=float(rewards.mean()),
        target_true_value=true_value,
        n_logged=n_contexts,
        effective_sample_size=effective_sample_size(target_probs, logging_probs, clip),
        clip_weight=clip,
        logging_epsilon=epsilon,
    )

    log.info(
        "off-policy (n=%d, ESS=%.0f): logged=%.4f | IPS=%.4f SNIPS=%.4f DR=%.4f | true=%.4f "
        "(abs error: IPS %.4f, SNIPS %.4f, DR %.4f)",
        n_contexts,
        report.effective_sample_size,
        report.logged_policy_value,
        ips_estimate.value,
        snips_estimate.value,
        dr_estimate.value,
        true_value,
        abs(ips_estimate.value - true_value),
        abs(snips_estimate.value - true_value),
        abs(dr_estimate.value - true_value),
    )
    return report


def _estimate(name: str, values: np.ndarray) -> OPEEstimate:
    value = float(values.mean())
    std_error = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return OPEEstimate(name, value, std_error, value - Z_95 * std_error, value + Z_95 * std_error)


def _reward_features(
    encoder: ArticleEncoder,
    profiles: UserProfiles,
    users: list[str],
    news_ids: list[str],
    hours: list[int],
) -> np.ndarray:
    """Context-action features for the DR reward model."""
    user_vectors = np.stack([profiles.get(user) for user in users])
    article_vectors = encoder.vecs(news_ids)

    cosine = np.einsum("ij,ij->i", user_vectors, article_vectors)
    history_length = np.array([np.log1p(len(profiles.history_of(user))) for user in users])
    hour = np.asarray(hours, dtype=float)

    # Hour enters as (sin, cos): 23:00 and 00:00 are adjacent on a clock but maximally
    # distant as integers, and a linear model cannot recover that on its own.
    return np.column_stack(
        [cosine, history_length, np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)]
    )


def _fit_reward_model(
    cfg: Config,
    encoder: ArticleEncoder,
    profiles: UserProfiles,
    users: list[str],
    news_ids: list[str],
    hours: list[int],
    rewards: np.ndarray,
) -> Ridge:
    features = _reward_features(encoder, profiles, users, news_ids, hours)
    return Ridge(alpha=1.0, random_state=cfg.seed).fit(features, rewards)


def _predict_reward(
    model: Ridge,
    encoder: ArticleEncoder,
    profiles: UserProfiles,
    users: list[str],
    news_ids: list[str],
    hours: list[int],
) -> np.ndarray:
    features = _reward_features(encoder, profiles, users, news_ids, hours)
    return np.clip(model.predict(features), 0.0, 1.0)
