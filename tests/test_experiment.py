"""Experiments: the A/B test, the simulator, and off-policy evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from newspush.experiment.ab_test import ab_test, simulate_campaign
from newspush.experiment.off_policy import (
    doubly_robust,
    effective_sample_size,
    ips,
    run_off_policy_evaluation,
    snips,
)
from newspush.experiment.simulator import ResponseSimulator


@pytest.fixture(scope="module")
def simulator(cfg, encoder, profiles, dev, popularity):
    return ResponseSimulator.fit(cfg, encoder, profiles, dev, popularity)


@pytest.fixture(scope="module")
def campaign_pool(popularity, encoder):
    known = [n for n in popularity if encoder.known(n)]
    return sorted(known, key=lambda n: popularity[n], reverse=True)[:60]


class TestABTest:
    def test_detects_a_real_difference(self):
        result = ab_test(
            control_successes=100, control_n=1000, treatment_successes=200, treatment_n=1000
        )

        assert result.control_rate == 0.1
        assert result.treatment_rate == 0.2
        assert result.absolute_lift == pytest.approx(0.1)
        assert result.relative_lift == pytest.approx(1.0)
        assert result.p_value < 0.001
        assert result.significant

    def test_identical_arms_are_not_significant(self):
        result = ab_test(100, 1000, 100, 1000)

        assert result.absolute_lift == 0.0
        assert result.p_value == pytest.approx(1.0)
        assert not result.significant

    def test_the_interval_brackets_the_observed_lift(self):
        result = ab_test(100, 1000, 130, 1000)
        assert result.ci_low < result.absolute_lift < result.ci_high

    def test_a_significant_result_excludes_zero(self):
        """The p-value and the interval must agree. They disagree if the pooled SE is
        mistakenly used for both."""
        result = ab_test(100, 5000, 200, 5000)

        assert result.significant
        assert result.ci_low > 0

    def test_a_null_result_includes_zero(self):
        result = ab_test(100, 1000, 105, 1000)

        assert not result.significant
        assert result.ci_low < 0 < result.ci_high

    def test_detects_a_negative_lift(self):
        result = ab_test(200, 1000, 100, 1000)

        assert result.absolute_lift < 0
        assert result.significant

    def test_degenerate_arms_do_not_divide_by_zero(self):
        result = ab_test(0, 100, 0, 100)

        assert result.z_statistic == 0.0
        assert result.p_value == 1.0
        assert not result.significant

    def test_larger_samples_narrow_the_interval(self):
        small = ab_test(10, 100, 20, 100)
        large = ab_test(1000, 10_000, 2000, 10_000)

        assert (large.ci_high - large.ci_low) < (small.ci_high - small.ci_low)

    @pytest.mark.parametrize(
        ("cs", "cn", "ts", "tn"),
        [(10, 0, 10, 100), (10, 100, 10, 0), (200, 100, 10, 100), (-1, 100, 10, 100)],
    )
    def test_rejects_impossible_counts(self, cs, cn, ts, tn):
        with pytest.raises(ValueError):
            ab_test(cs, cn, ts, tn)


class TestSimulator:
    def test_returns_probabilities(self, simulator, profiles, campaign_pool):
        users = list(profiles.vectors)[:10]
        news = campaign_pool[:10]

        probabilities = simulator.click_prob(users, news, [12] * 10)
        assert ((probabilities >= 0) & (probabilities <= 1)).all()

    def test_sampled_clicks_are_binary(self, simulator, profiles, campaign_pool):
        users = list(profiles.vectors)[:20]
        clicks = simulator.sample_clicks(users, campaign_pool[:20], [12] * 20)

        assert set(np.unique(clicks)) <= {0, 1}

    def test_reseeding_makes_draws_reproducible(self, simulator, profiles, campaign_pool):
        users = list(profiles.vectors)[:30]
        news = campaign_pool[:30]

        simulator.reseed(7)
        first = simulator.sample_clicks(users, news, [9] * 30)
        simulator.reseed(7)
        second = simulator.sample_clicks(users, news, [9] * 30)

        assert np.array_equal(first, second)

    def test_a_better_match_gets_a_higher_probability(self, simulator, ranker, profiles, campaign_pool):
        """The environment has to reward relevance, or no policy comparison inside it
        means anything."""
        user_id = next(iter(profiles.vectors))
        ranked = ranker.recommend_for_user(user_id, top_n=len(campaign_pool), candidate_pool=campaign_pool)

        best, worst = ranked[0][0], ranked[-1][0]
        probabilities = simulator.click_prob([user_id, user_id], [best, worst], [12, 12])

        assert probabilities[0] > probabilities[1]


class TestSimulatedCampaign:
    def test_personalisation_beats_popularity(self, cfg, simulator, ranker, profiles, popularity, campaign_pool):
        campaign = simulate_campaign(
            cfg, simulator, ranker, profiles, popularity, candidate_pool=campaign_pool
        )

        assert campaign.result.treatment_rate > campaign.result.control_rate
        assert campaign.result.significant

    def test_records_that_it_is_simulated(self, cfg, simulator, ranker, profiles, popularity, campaign_pool):
        """Provenance must travel with the number."""
        campaign = simulate_campaign(
            cfg, simulator, ranker, profiles, popularity, candidate_pool=campaign_pool
        )
        assert "not a live test" in campaign.environment.lower()

    def test_send_time_appears_in_the_treatment_name(
        self, cfg, simulator, ranker, profiles, popularity, campaign_pool, send_time_model
    ):
        campaign = simulate_campaign(
            cfg,
            simulator,
            ranker,
            profiles,
            popularity,
            send_time=send_time_model,
            candidate_pool=campaign_pool,
        )
        assert "send-time" in campaign.treatment_policy


class TestOffPolicyEstimators:
    def test_ips_is_unbiased_when_the_policies_agree(self):
        """If the target equals the logging policy, the weights are 1 and the estimate
        is just the mean reward."""
        rewards = np.array([1.0, 0.0, 1.0, 1.0])
        probabilities = np.array([0.5, 0.5, 0.5, 0.5])

        assert ips(rewards, probabilities, probabilities).value == pytest.approx(rewards.mean())

    def test_ips_upweights_rare_logged_actions(self):
        rewards = np.array([1.0, 1.0])
        target = np.array([1.0, 1.0])
        logging = np.array([0.1, 1.0])

        # The action logged with probability 0.1 counts for ten.
        assert ips(rewards, target, logging).value == pytest.approx((10.0 + 1.0) / 2)

    def test_clipping_bounds_the_weights(self):
        rewards = np.array([1.0, 1.0])
        target = np.array([1.0, 1.0])
        logging = np.array([0.001, 1.0])

        unclipped = ips(rewards, target, logging).value
        clipped = ips(rewards, target, logging, clip=5.0).value

        assert clipped < unclipped
        assert clipped == pytest.approx((5.0 + 1.0) / 2)

    def test_snips_is_less_variable_than_ips(self):
        """The reason SNIPS exists: self-normalisation stops one enormous weight from
        running away with the estimate."""
        rng = np.random.default_rng(0)
        n = 4000

        logging = rng.uniform(0.05, 1.0, size=n)
        target = (rng.random(n) < 0.5).astype(float)
        rewards = (rng.random(n) < 0.3).astype(float)

        ips_values = []
        snips_values = []
        for start in range(0, n, 400):
            chunk = slice(start, start + 400)
            ips_values.append(ips(rewards[chunk], target[chunk], logging[chunk]).value)
            snips_values.append(snips(rewards[chunk], target[chunk], logging[chunk]).value)

        assert np.std(snips_values) < np.std(ips_values)

    def test_snips_handles_zero_weight_mass(self):
        estimate = snips(np.array([1.0]), np.array([0.0]), np.array([0.5]))
        assert estimate.value == 0.0

    def test_doubly_robust_survives_a_wrong_reward_model(self):
        """DR's guarantee: correct propensities rescue a bad reward model."""
        rng = np.random.default_rng(1)
        n = 3000

        logging = np.full(n, 0.5)
        target = (rng.random(n) < 0.5).astype(float)
        true_value = 0.4
        rewards = (rng.random(n) < true_value).astype(float)

        garbage_q = np.full(n, 0.9)
        estimate = doubly_robust(rewards, target, logging, garbage_q, garbage_q)

        # Target and logging agree half the time, so the target's value is the base rate.
        assert abs(estimate.value - true_value) < 0.05

    def test_effective_sample_size_falls_with_skewed_weights(self):
        n = 1000
        uniform = effective_sample_size(np.ones(n), np.full(n, 0.5))

        skewed_logging = np.full(n, 0.5)
        skewed_target = np.zeros(n)
        skewed_target[0] = 1.0
        skewed = effective_sample_size(skewed_target, skewed_logging)

        assert uniform == pytest.approx(n)
        assert skewed < 5

    def test_estimates_carry_intervals(self):
        rewards = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        probabilities = np.full(5, 0.5)

        estimate = ips(rewards, probabilities, probabilities)
        assert estimate.ci_low < estimate.value < estimate.ci_high
        assert estimate.std_error > 0


class TestOffPolicyEvaluation:
    @pytest.fixture(scope="class")
    def report(self, cfg, simulator, encoder, profiles, popularity, campaign_pool):
        return run_off_policy_evaluation(
            cfg,
            simulator,
            encoder,
            profiles,
            popularity,
            candidate_pool=campaign_pool,
            n_contexts=1500,
            n_actions=6,
        )

    def test_recovers_the_true_policy_value(self, report):
        """The claim the module exists to support: the estimators find a value they were
        never shown. DR and SNIPS are the ones that should land close."""
        assert abs(report.dr.value - report.target_true_value) < 0.1
        assert abs(report.snips.value - report.target_true_value) < 0.1

    def test_doubly_robust_beats_plain_ips(self, report):
        dr_error = abs(report.dr.value - report.target_true_value)
        ips_error = abs(report.ips.value - report.target_true_value)
        assert dr_error <= ips_error

    def test_the_target_policy_beats_the_logging_policy(self, report):
        assert report.target_true_value > report.logged_policy_value

    def test_reports_effective_sample_size(self, report):
        assert 0 < report.effective_sample_size <= report.n_logged
