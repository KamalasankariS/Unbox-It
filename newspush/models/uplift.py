"""Uplift / CATE modelling: email the persuadable, not the already-convinced.

The propensity model ranks readers by P(engage | emailed). Targeting the top of that
ranking sends email to readers who were going to engage anyway, and takes credit for a
visit that was already coming. What matters is the incremental effect of the send:

    uplift(u) = P(engage | u, emailed) - P(engage | u, not emailed)

which separates the persuadables (respond only if emailed) from the sure things
(respond either way), the lost causes (respond to nothing) and the sleeping dogs
(the send actively drives them away). Propensity ranking happily targets sure things
and can even target sleeping dogs; uplift ranking does not.

Method is a T-learner: fit one outcome model per arm and difference their predictions.
Simple and transparent, at the cost of each model fitting its own arm's noise so the
difference is noisier than either part, which is why the reported metrics are AUUC and
Qini rather than a point estimate.

MIND has no randomised email holdout, so the treatment and control arms are constructed
inside the ResponseSimulator. The model and metrics would run unchanged on a real
campaign's holdout; the data they are demonstrated on is simulated, and is tagged so.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from newspush.config import Config

log = logging.getLogger(__name__)

MIN_ROWS_PER_ARM = 10
DEFAULT_BINS = 20
TOP_FRACTION = 0.3

# AUUC normalises by the ATE, so it is only meaningful when the ATE is distinguishable
# from zero. Require it to exceed this many standard errors. See auuc_score.
MIN_ATE_Z_FOR_AUUC = 2.0


@dataclass
class UpliftMetrics:
    auuc: float
    qini: float
    uplift_at_top_30pct: float
    uplift_at_bottom_30pct: float
    overall_ate: float
    n_treated: int
    n_control: int
    n_persuadable_est: int
    n_sleeping_dogs_est: int
    data_note: str

    def to_dict(self) -> dict:
        return asdict(self)


class TLearner:
    """Two-model uplift estimator: mu_1(x) - mu_0(x)."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.model_treated: HistGradientBoostingClassifier | None = None
        self.model_control: HistGradientBoostingClassifier | None = None

    def fit(self, features: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> "TLearner":
        """Fit one outcome model per arm.

        Args:
            treatment: 1 if emailed, 0 if held out.
            outcome: 1 if engaged.
        """
        treated = treatment == 1
        control = ~treated

        if treated.sum() < MIN_ROWS_PER_ARM or control.sum() < MIN_ROWS_PER_ARM:
            raise ValueError(
                f"need at least {MIN_ROWS_PER_ARM} rows per arm to fit a T-learner "
                f"(got {int(treated.sum())} treated, {int(control.sum())} control)"
            )

        for name, mask in (("treated", treated), ("control", control)):
            if len(np.unique(outcome[mask])) < 2:
                raise ValueError(f"the {name} arm has a single outcome class")

        self.model_treated = HistGradientBoostingClassifier(
            random_state=self.cfg.seed, max_iter=150, learning_rate=0.1
        ).fit(features[treated], outcome[treated])

        self.model_control = HistGradientBoostingClassifier(
            random_state=self.cfg.seed + 1, max_iter=150, learning_rate=0.1
        ).fit(features[control], outcome[control])

        log.info(
            "T-learner fitted: %d treated (%.1f%% engaged), %d control (%.1f%% engaged)",
            int(treated.sum()),
            100.0 * float(outcome[treated].mean()),
            int(control.sum()),
            100.0 * float(outcome[control].mean()),
        )
        return self

    def predict_uplift(self, features: np.ndarray) -> np.ndarray:
        """CATE per row. Positive means emailing this reader helps; negative means it hurts."""
        if self.model_treated is None or self.model_control is None:
            raise RuntimeError("T-learner is not fitted")

        treated_response = self.model_treated.predict_proba(features)[:, 1]
        control_response = self.model_control.predict_proba(features)[:, 1]
        return treated_response - control_response


def uplift_curve(
    uplift_scores: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    n_bins: int = DEFAULT_BINS,
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative incremental response as we descend the uplift ranking.

    Uplift has no per-row error metric: for anyone who was emailed, the counterfactual
    outcome is unobservable. What can be measured is a group effect, so this compares
    the response of treated and control members within each top-k slice.
    """
    order = np.argsort(-uplift_scores, kind="mergesort")
    ranked_treatment = treatment[order]
    ranked_outcome = outcome[order]
    n_rows = len(order)

    fractions = np.linspace(1.0 / n_bins, 1.0, n_bins)
    gains = np.zeros(n_bins)

    for i, fraction in enumerate(fractions):
        k = max(1, int(round(fraction * n_rows)))
        treated = ranked_treatment[:k] == 1
        control = ~treated

        if not treated.any() or not control.any():
            continue

        treated_rate = float(ranked_outcome[:k][treated].mean())
        control_rate = float(ranked_outcome[:k][control].mean())

        # Scaled to incremental responses per targeted reader, so the curve is
        # comparable across different treated/control split ratios.
        gains[i] = (treated_rate - control_rate) * k

    return fractions, gains


def auuc_score(
    uplift_scores: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    n_bins: int = DEFAULT_BINS,
) -> float:
    """Area under the uplift curve, normalised by random targeting. Above 1 beats random.

    Undefined when the average treatment effect is not distinguishable from zero,
    because the random-targeting baseline it normalises against is then also zero and
    the ratio is dominated by sampling noise. This is not an edge case to paper over: a
    campaign whose persuadables and sleeping dogs cancel out has no average effect and
    can still have a perfectly good ranking. For that case read Qini, which is an
    absolute gain over random rather than a ratio.
    """
    ate, standard_error = _ate_with_standard_error(treatment, outcome)

    if not np.isfinite(ate):
        return float("nan")
    if standard_error > 0 and abs(ate) < MIN_ATE_Z_FOR_AUUC * standard_error:
        return float("nan")

    fractions, gains = uplift_curve(uplift_scores, treatment, outcome, n_bins)
    random_area = 0.5 * ate * len(uplift_scores)
    if abs(random_area) < 1e-12:
        return float("nan")

    return float(_trapezoid(gains, fractions) / random_area)


def qini_score(
    uplift_scores: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    n_bins: int = DEFAULT_BINS,
) -> float:
    """Area between the uplift curve and the random-targeting line.

    The same information as AUUC, expressed as an absolute gain over random.
    """
    fractions, gains = uplift_curve(uplift_scores, treatment, outcome, n_bins)
    ate = _average_treatment_effect(treatment, outcome)

    if not np.isfinite(ate):
        return float("nan")

    random_line = ate * fractions * len(uplift_scores)
    return float(_trapezoid(gains - random_line, fractions))


def evaluate_uplift(
    model: TLearner,
    features: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    data_note: str,
) -> UpliftMetrics:
    scores = model.predict_uplift(features)
    order = np.argsort(-scores, kind="mergesort")
    top_k = max(1, int(TOP_FRACTION * len(scores)))

    def observed_uplift(indices: np.ndarray) -> float:
        return _average_treatment_effect(treatment[indices], outcome[indices])

    metrics = UpliftMetrics(
        auuc=auuc_score(scores, treatment, outcome),
        qini=qini_score(scores, treatment, outcome),
        uplift_at_top_30pct=observed_uplift(order[:top_k]),
        uplift_at_bottom_30pct=observed_uplift(order[-top_k:]),
        overall_ate=_average_treatment_effect(treatment, outcome),
        n_treated=int(treatment.sum()),
        n_control=int((1 - treatment).sum()),
        n_persuadable_est=int((scores > 0).sum()),
        n_sleeping_dogs_est=int((scores < 0).sum()),
        data_note=data_note,
    )

    log.info(
        "uplift: AUUC=%.3f Qini=%.2f | observed uplift top-30%%=%.4f vs bottom-30%%=%.4f (ATE=%.4f) | "
        "%d persuadable, %d sleeping dogs",
        metrics.auuc,
        metrics.qini,
        metrics.uplift_at_top_30pct,
        metrics.uplift_at_bottom_30pct,
        metrics.overall_ate,
        metrics.n_persuadable_est,
        metrics.n_sleeping_dogs_est,
    )
    return metrics


def build_simulated_trial(
    cfg: Config,
    click_prob: np.ndarray,
    engagement_percentile: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Construct the randomised email trial that MIND does not contain.

    The treated arm's response comes from the simulator. The control arm's does not
    exist anywhere in the data, so it is generated from a stated model of organic
    behaviour: an engaged reader finds a story whether or not we email them, and an
    irrelevant send can displace the visit they would have made anyway.

        p_control = organic
        p_treated = organic + p_email * (1 - organic) - annoyance * organic * (1 - match)

    where organic rises with the reader's engagement percentile and `match` is the
    email's relevance. This yields all four uplift segments: persuadables (low organic,
    good match), sure things (high organic), lost causes (low both), and sleeping dogs
    (high organic, poor match, so the send displaces more than it creates).

    This is circular by construction, as any simulator must be: it demonstrates that
    the T-learner and the Qini/AUUC machinery recover a heterogeneous effect that was
    injected. It does NOT show that MIND readers have this uplift structure. Returns
    (treatment, outcome, note) with the note recorded in metrics.json.
    """
    organic_base = float(cfg.require("uplift.organic_base_rate"))
    annoyance = float(cfg.require("uplift.annoyance_weight"))
    treated_fraction = float(cfg.require("uplift.treatment_fraction"))

    organic = organic_base * engagement_percentile

    peak = float(click_prob.max())
    match_quality = click_prob / peak if peak > 0 else np.zeros_like(click_prob)

    p_treated = np.clip(
        organic + click_prob * (1.0 - organic) - annoyance * organic * (1.0 - match_quality),
        0.0,
        1.0,
    )
    p_control = np.clip(organic, 0.0, 1.0)

    treatment = (rng.random(size=len(click_prob)) < treated_fraction).astype(int)
    response_prob = np.where(treatment == 1, p_treated, p_control)
    outcome = (rng.random(size=len(click_prob)) < response_prob).astype(int)

    note = (
        "simulated randomised trial: the treated arm's response comes from the "
        "ResponseSimulator, the control arm from an assumed organic-engagement model "
        f"(organic_base_rate={organic_base}, annoyance_weight={annoyance}). MIND contains "
        "no email holdout, so no control arm exists in the data."
    )
    return treatment, outcome, note


def _average_treatment_effect(treatment: np.ndarray, outcome: np.ndarray) -> float:
    return _ate_with_standard_error(treatment, outcome)[0]


def _ate_with_standard_error(treatment: np.ndarray, outcome: np.ndarray) -> tuple[float, float]:
    """(ATE, standard error). Returns NaN for both if either arm is empty."""
    treated = treatment == 1
    control = ~treated

    n_treated = int(treated.sum())
    n_control = int(control.sum())
    if n_treated == 0 or n_control == 0:
        return float("nan"), float("nan")

    treated_rate = float(outcome[treated].mean())
    control_rate = float(outcome[control].mean())

    variance = (
        treated_rate * (1.0 - treated_rate) / n_treated
        + control_rate * (1.0 - control_rate) / n_control
    )
    return treated_rate - control_rate, float(np.sqrt(max(variance, 0.0)))


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    # np.trapz was renamed to np.trapezoid in numpy 2.0 and removed in later 2.x.
    # Access trapz only when trapezoid is absent, so the removed name is never touched
    # on a modern numpy.
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))
