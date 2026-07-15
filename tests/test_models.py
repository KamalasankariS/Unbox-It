"""Models: content selection, audience, send-time, fatigue, diversity, bandit, uplift."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from newspush.models import content_selection as cs
from newspush.models.audience import PropensityModel, precision_at_k
from newspush.models.bandit import EpsilonGreedy, LinearThompsonSampling, run_bandit_experiment
from newspush.models.diversity import (
    apply_editorial_boost,
    build_editorial_send,
    category_spread,
    diversity_score,
    mmr_select,
)
from newspush.models.fatigue import FatigueModel
from newspush.models.send_time import SendTimeModel
from newspush.models.uplift import TLearner, auuc_score, evaluate_uplift, qini_score


class TestRankingMetrics:
    def test_perfect_ranking_scores_one(self):
        labels = np.array([0, 1, 0, 1])
        scores = np.array([0.1, 0.9, 0.2, 0.8])

        assert cs.auc_score(labels, scores) == 1.0
        assert cs.mrr_score(labels, scores) == 1.0
        assert cs.ndcg_score(labels, scores, 5) == 1.0

    def test_worst_ranking_scores_zero_auc(self):
        labels = np.array([0, 1])
        scores = np.array([1.0, 0.0])
        assert cs.auc_score(labels, scores) == 0.0

    def test_a_constant_scorer_is_a_coin_flip(self):
        """Ties must get averaged ranks. Without that a constant scorer would read as
        a perfect one, and a broken model would look flawless."""
        labels = np.array([0, 1, 0, 1])
        assert cs.auc_score(labels, np.ones(4)) == 0.5

    def test_auc_is_undefined_without_both_classes(self):
        assert np.isnan(cs.auc_score(np.array([1, 1]), np.array([0.5, 0.9])))
        assert np.isnan(cs.auc_score(np.array([0, 0]), np.array([0.5, 0.9])))

    def test_mrr_reflects_the_first_hit(self):
        labels = np.array([0, 0, 1])
        scores = np.array([0.9, 0.8, 0.7])
        assert cs.mrr_score(labels, scores) == pytest.approx(1 / 3)

    def test_mrr_is_zero_when_nothing_is_relevant(self):
        assert cs.mrr_score(np.array([0, 0]), np.array([0.9, 0.1])) == 0.0

    def test_ndcg_rewards_ranking_the_hit_higher(self):
        labels = np.array([1, 0, 0, 0])
        good = cs.ndcg_score(labels, np.array([0.9, 0.1, 0.1, 0.1]), 5)
        bad = cs.ndcg_score(labels, np.array([0.1, 0.9, 0.8, 0.7]), 5)

        assert good == 1.0
        assert bad < good

    def test_ndcg_matches_sklearn(self):
        rng = np.random.default_rng(0)
        from sklearn.metrics import ndcg_score as sk_ndcg

        for _ in range(20):
            labels = rng.integers(0, 2, size=8)
            if labels.sum() in (0, len(labels)):
                continue
            scores = rng.random(8)
            assert cs.ndcg_score(labels, scores, 5) == pytest.approx(
                sk_ndcg([labels], [scores], k=5), abs=1e-9
            )

    def test_auc_matches_sklearn(self):
        rng = np.random.default_rng(1)
        from sklearn.metrics import roc_auc_score

        for _ in range(20):
            labels = rng.integers(0, 2, size=10)
            if labels.sum() in (0, len(labels)):
                continue
            scores = rng.random(10)
            assert cs.auc_score(labels, scores) == pytest.approx(roc_auc_score(labels, scores))


class TestContentRanker:
    def test_beats_the_popularity_baseline(self, ranker, dev, cfg, popularity):
        """The headline claim of the project. If personalisation cannot beat 'send
        everyone the most popular story', it is not worth deploying."""
        personalised = cs.evaluate(ranker, dev, cfg)
        baseline = cs.evaluate_popularity(dev, popularity, cfg)

        assert personalised.auc > baseline.auc
        assert personalised.auc > 0.5

    def test_reports_how_many_impressions_counted(self, ranker, dev, cfg):
        metrics = cs.evaluate(ranker, dev, cfg)

        assert metrics.n_impressions_evaluated > 0
        assert metrics.n_impressions_skipped >= 0

    def test_recommend_excludes_history(self, ranker, profiles):
        user_id = next(iter(profiles.vectors))
        history = set(profiles.history_of(user_id))

        recommended = {n for n, _ in ranker.recommend_for_user(user_id, top_n=10)}
        assert not (recommended & history)

    def test_recommend_returns_ranked_scores(self, ranker, profiles):
        user_id = next(iter(profiles.vectors))
        recommendations = ranker.recommend_for_user(user_id, top_n=5)

        assert len(recommendations) == 5
        scores = [score for _, score in recommendations]
        assert scores == sorted(scores, reverse=True)

    def test_cold_readers_score_flat(self, ranker):
        scores = ranker.score_candidates("NOT_A_USER", ["N000001", "N000002"])
        assert np.allclose(scores, 0.0)

    def test_cold_readers_still_get_recommendations(self, ranker):
        """Serving cannot return nothing: a cold reader still has to receive an email."""
        assert ranker.recommend_for_user("NOT_A_USER", top_n=3)

    def test_an_empty_pool_returns_nothing(self, ranker, profiles):
        user_id = next(iter(profiles.vectors))
        assert ranker.recommend_for_user(user_id, top_n=5, candidate_pool=[]) == []


class TestPropensityModel:
    def test_beats_chance_on_held_out_data(self, propensity, dev):
        auc, labels, predictions = propensity.evaluate(dev, max_rows=5000)

        assert auc > 0.5
        assert len(labels) == len(predictions)
        assert ((predictions >= 0) & (predictions <= 1)).all()

    def test_audience_is_ranked_and_sized(self, propensity, profiles, train):
        news_id = str(train.news["news_id"].iloc[0])
        audience = propensity.build_audience(news_id, k=10)

        assert len(audience) == 10
        scores = [score for _, score in audience]
        assert scores == sorted(scores, reverse=True)

    def test_audience_respects_the_candidate_pool(self, propensity, profiles, train):
        news_id = str(train.news["news_id"].iloc[0])
        pool = list(profiles.vectors)[:5]

        audience = propensity.build_audience(news_id, k=3, candidate_users=pool)
        assert {user for user, _ in audience} <= set(pool)

    def test_audience_k_is_clamped_to_the_population(self, propensity, train):
        news_id = str(train.news["news_id"].iloc[0])
        audience = propensity.build_audience(news_id, k=10**6)

        assert len(audience) <= len(propensity.profiles.vectors)

    def test_send_hours_change_the_scoring(self, propensity, profiles, train):
        news_id = str(train.news["news_id"].iloc[0])
        users = list(profiles.vectors)[:20]

        morning = propensity.build_audience(news_id, k=5, candidate_users=users, hour=3)
        per_user = propensity.build_audience(
            news_id, k=5, candidate_users=users, send_hours={u: 20 for u in users}
        )
        assert [s for _, s in morning] != [s for _, s in per_user]

    def test_unfitted_model_raises(self, cfg, encoder, profiles):
        with pytest.raises(RuntimeError, match="not fitted"):
            PropensityModel(cfg, encoder, profiles).predict_proba(np.zeros((1, 5)))


class TestPrecisionAtK:
    def test_picks_out_the_top_scored(self):
        labels = np.array([0, 1, 1, 0])
        scores = np.array([0.1, 0.9, 0.8, 0.2])
        assert precision_at_k(labels, scores, 2) == 1.0

    def test_k_larger_than_n_is_clamped(self):
        labels = np.array([1, 0])
        assert precision_at_k(labels, np.array([0.9, 0.1]), 10) == 0.5

    def test_empty_input(self):
        assert precision_at_k(np.array([]), np.array([]), 5) == 0.0


class TestSendTime:
    def test_recovers_a_planted_preferred_hour(self, cfg):
        """A reader who only ever engages at 07:00 must be scheduled for 07:00."""
        rows = []
        for hour in range(24):
            clicks = 40 if hour == 7 else 1
            rows.append(
                {"user_id": "U_MORNING", "hour": hour, "impressions": 50, "clicks": clicks}
            )
        # A population that skews late, so the prior pulls the other way and the test
        # proves the personal signal actually wins.
        for hour in range(24):
            rows.append(
                {
                    "user_id": "U_CROWD",
                    "hour": hour,
                    "impressions": 500,
                    "clicks": 100 if hour == 20 else 5,
                }
            )

        model = SendTimeModel(cfg).fit(pd.DataFrame(rows))
        assert model.best_hour("U_MORNING") == 7

    def test_thin_evidence_falls_back_to_the_global_curve(self, cfg):
        rows = [{"user_id": "U_QUIET", "hour": 3, "impressions": 1, "clicks": 1}]
        rows += [
            {"user_id": "U_CROWD", "hour": h, "impressions": 500, "clicks": 100 if h == 20 else 5}
            for h in range(24)
        ]

        model = SendTimeModel(cfg).fit(pd.DataFrame(rows))

        # One 3am click must not make someone a 3am reader.
        assert model.best_hour("U_QUIET") == model.global_best_hour
        assert not model.is_personalised("U_QUIET")

    def test_unknown_readers_get_the_global_best_hour(self, send_time_model):
        assert send_time_model.best_hour("NOT_A_USER") == send_time_model.global_best_hour

    def test_best_hours_are_valid_clock_hours(self, send_time_model, profiles):
        hours = send_time_model.best_hours(list(profiles.vectors)[:20])
        assert all(0 <= hour <= 23 for hour in hours.values())

    def test_rate_curve_covers_the_clock(self, send_time_model, profiles):
        user_id = next(iter(profiles.vectors))
        assert len(send_time_model.rate_by_hour(user_id)) == 24

    def test_requires_the_expected_columns(self, cfg):
        with pytest.raises(ValueError, match="missing columns"):
            SendTimeModel(cfg).fit(pd.DataFrame({"user_id": ["U1"]}))

    def test_survives_a_split_with_no_clicks(self, cfg):
        rows = [{"user_id": "U1", "hour": h, "impressions": 10, "clicks": 0} for h in range(24)]
        model = SendTimeModel(cfg).fit(pd.DataFrame(rows))

        assert 0 <= model.best_hour("U1") <= 23


class TestFatigue:
    def test_risk_rises_with_send_volume(self, fatigue_model, profiles):
        user_id = next(iter(profiles.vectors))
        risks = [fatigue_model.unsubscribe_risk(user_id, n) for n in range(1, 6)]

        assert risks == sorted(risks)
        assert all(0 <= r <= 1 for r in risks)

    def test_disengaged_readers_carry_more_risk(self, cfg):
        model = FatigueModel(cfg).fit_engagement(
            pd.DataFrame(
                [
                    {"user_id": "U_ENGAGED", "impressions": 100, "clicks": 60},
                    {"user_id": "U_DORMANT", "impressions": 100, "clicks": 0},
                ]
            )
        )
        assert model.unsubscribe_risk("U_DORMANT", 3) > model.unsubscribe_risk("U_ENGAGED", 3)

    def test_the_hard_cap_holds_regardless_of_risk(self, cfg):
        """The cap must not depend on believing the risk model."""
        model = FatigueModel(cfg).fit_engagement(
            pd.DataFrame([{"user_id": "U1", "impressions": 100, "clicks": 100}])
        )
        cap = model.max_per_week
        assert not model.should_send("U1", cap)
        assert not model.should_send("U1", cap + 5)

    def test_capping_reduces_sends_and_reports_the_cost(self, fatigue_model, profiles):
        users = list(profiles.vectors)[:10]
        plan = pd.DataFrame(
            [
                {"user_id": u, "news_id": f"N{i:06d}", "score": 1.0 - 0.05 * i}
                for u in users
                for i in range(8)
            ]
        )
        capped, metrics = fatigue_model.apply_cap(plan)

        assert len(capped) < len(plan)
        assert metrics.n_sends_suppressed == len(plan) - len(capped)
        assert 0 <= metrics.engagement_retained <= 1
        # Capping keeps a reader's best sends, so it must retain more engagement than
        # the share of sends it keeps.
        assert metrics.engagement_retained >= 1.0 - metrics.suppression_rate

    def test_capping_keeps_the_highest_scoring_sends(self, fatigue_model):
        plan = pd.DataFrame(
            [
                {"user_id": "U1", "news_id": "N_BEST", "score": 0.9},
                {"user_id": "U1", "news_id": "N_WORST", "score": 0.1},
            ]
        )
        capped, _ = fatigue_model.apply_cap(plan)
        assert capped.iloc[0]["news_id"] == "N_BEST"

    def test_nobody_exceeds_the_cap(self, fatigue_model, profiles):
        users = list(profiles.vectors)[:5]
        plan = pd.DataFrame(
            [
                {"user_id": u, "news_id": f"N{i:06d}", "score": 0.5}
                for u in users
                for i in range(20)
            ]
        )
        capped, _ = fatigue_model.apply_cap(plan)

        counts = capped.groupby("user_id").size()
        assert (counts <= fatigue_model.max_per_week).all()

    def test_rejects_an_empty_plan(self, fatigue_model):
        with pytest.raises(ValueError, match="empty"):
            fatigue_model.apply_cap(pd.DataFrame(columns=["user_id", "news_id", "score"]))

    def test_unknown_readers_sit_at_the_median(self, fatigue_model):
        assert fatigue_model.engagement_percentile("NOT_A_USER") == 0.5


class TestDiversity:
    def test_mmr_returns_at_most_k(self, encoder, train):
        candidates = list(train.news["news_id"])[:20]
        relevance = {n: 1.0 - 0.01 * i for i, n in enumerate(candidates)}

        assert len(mmr_select(candidates, relevance, encoder, k=5)) == 5

    def test_pure_relevance_matches_a_greedy_sort(self, encoder, train):
        candidates = list(train.news["news_id"])[:20]
        relevance = {n: 1.0 - 0.01 * i for i, n in enumerate(candidates)}

        selected = mmr_select(candidates, relevance, encoder, k=5, lambda_=1.0)
        expected = sorted(candidates, key=lambda n: -relevance[n])[:5]
        assert selected == expected

    def test_diversifying_raises_diversity(self, encoder, train):
        candidates = list(train.news["news_id"])[:40]
        relevance = {n: 1.0 - 0.001 * i for i, n in enumerate(candidates)}

        greedy = mmr_select(candidates, relevance, encoder, k=5, lambda_=1.0)
        diverse = mmr_select(candidates, relevance, encoder, k=5, lambda_=0.3)

        assert diversity_score(diverse, encoder) >= diversity_score(greedy, encoder)

    def test_the_desk_cap_is_never_exceeded(self, encoder, train):
        candidates = list(train.news["news_id"])[:60]
        relevance = {n: 1.0 for n in candidates}
        category = dict(zip(train.news["news_id"].astype(str), train.news["category"].astype(str)))

        selected = mmr_select(
            candidates, relevance, encoder, k=10, news_category=category, max_per_category=2
        )
        counts: dict[str, int] = {}
        for news_id in selected:
            desk = category[news_id]
            counts[desk] = counts.get(desk, 0) + 1

        assert all(count <= 2 for count in counts.values())

    def test_pins_always_appear(self, encoder, train):
        candidates = list(train.news["news_id"])[:20]
        relevance = {n: 0.0 for n in candidates}
        pinned = [candidates[15]]

        selected = mmr_select(candidates, relevance, encoder, k=3, pinned=pinned)
        assert pinned[0] in selected

    def test_a_boost_is_multiplicative(self):
        boosted = apply_editorial_boost({"N1": 0.5, "N2": 0.5}, {"N1": 2.0})

        assert boosted["N1"] == 1.0
        assert boosted["N2"] == 0.5

    def test_no_boost_is_a_no_op(self):
        relevance = {"N1": 0.5}
        assert apply_editorial_boost(relevance, None) == relevance

    def test_diversity_of_a_single_article_is_zero(self, encoder, train):
        assert diversity_score([str(train.news["news_id"].iloc[0])], encoder) == 0.0

    def test_category_spread_counts_distinct_desks(self):
        assert category_spread(["N1", "N2", "N3"], {"N1": "a", "N2": "a", "N3": "b"}) == 2

    def test_the_guardrails_report_their_own_cost(self, cfg, encoder, train):
        candidates = list(train.news["news_id"])[:40]
        relevance = {n: 1.0 - 0.001 * i for i, n in enumerate(candidates)}
        category = dict(zip(train.news["news_id"].astype(str), train.news["category"].astype(str)))

        selected, stats = build_editorial_send(
            cfg, candidates, relevance, encoder, category, k=5
        )

        assert len(selected) <= 5
        assert 0 <= stats["relevance_retained"] <= 1.0
        assert stats["category_spread"] >= 1


class TestBandit:
    def test_thompson_sampling_beats_random(self, cfg):
        rng = np.random.default_rng(0)
        n_rounds, n_arms, dim = 600, 4, 5

        contexts = rng.normal(size=(n_rounds, dim))
        contexts[:, -1] = 1.0

        theta = rng.normal(size=(n_arms, dim))
        arm_rewards = 1.0 / (1.0 + np.exp(-(contexts @ theta.T)))

        metrics = run_bandit_experiment(cfg, contexts, arm_rewards)

        assert metrics.thompson_reward > metrics.random_reward
        assert metrics.thompson_regret < metrics.random_regret

    def test_regret_curve_is_recorded_and_monotonic(self, cfg):
        rng = np.random.default_rng(1)
        contexts = rng.normal(size=(400, 4))
        arm_rewards = rng.random(size=(400, 3))

        metrics = run_bandit_experiment(cfg, contexts, arm_rewards)

        assert metrics.regret_curve
        regrets = [point["thompson_regret"] for point in metrics.regret_curve]
        assert regrets == sorted(regrets)  # cumulative regret can only grow

    def test_thompson_sampling_finds_a_dominant_arm(self, cfg):
        """With one clearly best arm, it should be played far more than its share."""
        rng = np.random.default_rng(2)
        n_rounds, n_arms, dim = 800, 4, 3

        contexts = np.ones((n_rounds, dim))
        arm_rewards = np.tile(np.array([0.05, 0.05, 0.9, 0.05]), (n_rounds, 1))

        agent = LinearThompsonSampling(n_arms, dim, cfg)
        plays = np.zeros(n_arms)

        for t in range(n_rounds):
            arm = agent.select_arm(contexts[t])
            reward = float(rng.random() < arm_rewards[t, arm])
            agent.update(arm, contexts[t], reward)
            plays[arm] += 1

        assert plays.argmax() == 2
        assert plays[2] / n_rounds > 0.5

    def test_epsilon_greedy_explores(self, cfg):
        agent = EpsilonGreedy(3, 2, cfg)
        arms = {agent.select_arm(np.array([1.0, 0.0])) for _ in range(200)}
        assert len(arms) > 1

    def test_posterior_narrows_with_evidence(self, cfg):
        agent = LinearThompsonSampling(2, 3, cfg)
        context = np.array([1.0, 0.0, 0.0])

        _, before = agent._posterior(0)
        for _ in range(50):
            agent.update(0, context, 1.0)
        _, after = agent._posterior(0)

        # Observing the same context 50 times must shrink the uncertainty along it.
        assert after[0, 0] < before[0, 0]


class TestUplift:
    @staticmethod
    def _trial(rng, n=1200, mean_effect=0.15):
        """A trial with a heterogeneous effect: the email helps readers with a high
        first feature and hurts those with a low one (the sleeping dogs), while helping
        on average, which is what a working campaign looks like."""
        features = rng.normal(size=(n, 3))
        treatment = (rng.random(n) < 0.5).astype(int)

        base = 0.3
        effect = mean_effect + 0.35 * np.tanh(features[:, 0])
        probability = np.clip(base + treatment * effect, 0.01, 0.99)
        outcome = (rng.random(n) < probability).astype(int)

        return features, treatment, outcome

    def test_recovers_a_heterogeneous_effect(self, cfg, rng):
        features, treatment, outcome = self._trial(rng)
        model = TLearner(cfg).fit(features, treatment, outcome)

        metrics = evaluate_uplift(model, features, treatment, outcome, data_note="test")

        # The whole point: the top of the ranking must show more incremental response
        # than the bottom, or the model has learned nothing causal.
        assert metrics.uplift_at_top_30pct > metrics.uplift_at_bottom_30pct
        assert metrics.n_persuadable_est > 0

    def test_beats_random_targeting(self, cfg, rng):
        features, treatment, outcome = self._trial(rng)
        model = TLearner(cfg).fit(features, treatment, outcome)
        scores = model.predict_uplift(features)

        assert auuc_score(scores, treatment, outcome) > 1.0
        assert qini_score(scores, treatment, outcome) > 0.0

    def test_a_random_ranking_does_not_beat_random(self, cfg, rng):
        """Guards the metric itself: a meaningless ranking must not score well, or
        AUUC would flatter any model at all."""
        features, treatment, outcome = self._trial(rng)
        noise = rng.normal(size=len(treatment))

        assert qini_score(noise, treatment, outcome) < qini_score(
            TLearner(cfg).fit(features, treatment, outcome).predict_uplift(features),
            treatment,
            outcome,
        )

    def test_finds_sleeping_dogs(self, cfg, rng):
        """Readers the email drives away must be identifiable, since not emailing them
        is the entire operational payoff of uplift modelling."""
        features, treatment, outcome = self._trial(rng)
        model = TLearner(cfg).fit(features, treatment, outcome)

        scores = model.predict_uplift(features)
        assert (scores < 0).any()

        # Sleeping dogs should sit at the low end of the planted effect.
        assert features[scores < 0, 0].mean() < features[scores > 0, 0].mean()

    def test_auuc_is_undefined_when_the_average_effect_vanishes(self, cfg, rng):
        """A campaign whose gains and losses cancel has no random baseline to normalise
        against. AUUC must say so rather than return a huge meaningless ratio; Qini
        still works."""
        features, treatment, outcome = self._trial(rng, mean_effect=0.0)
        model = TLearner(cfg).fit(features, treatment, outcome)
        scores = model.predict_uplift(features)

        assert np.isnan(auuc_score(scores, treatment, outcome))
        assert np.isfinite(qini_score(scores, treatment, outcome))

    def test_requires_both_arms(self, cfg, rng):
        features = rng.normal(size=(50, 3))
        treatment = np.ones(50, dtype=int)
        outcome = (rng.random(50) < 0.5).astype(int)

        with pytest.raises(ValueError, match="per arm"):
            TLearner(cfg).fit(features, treatment, outcome)

    def test_unfitted_model_raises(self, cfg):
        with pytest.raises(RuntimeError, match="not fitted"):
            TLearner(cfg).predict_uplift(np.zeros((2, 3)))
