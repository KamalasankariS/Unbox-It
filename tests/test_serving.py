"""Config, explanations, the FastAPI service, the batch scorer, and the pipeline."""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from newspush.artifacts import Artifacts, artifacts_path
from newspush.config import Config, load_config
from newspush.explain import why
from newspush.features.text import encoder_name
from newspush.serving import batch
from newspush.serving.api import create_app


@pytest.fixture(scope="module")
def artifacts(cfg, encoder, profiles, propensity, send_time_model, fatigue_model, train, popularity):
    return Artifacts(
        encoder=encoder,
        profiles=profiles,
        propensity=propensity,
        send_time=send_time_model,
        fatigue=fatigue_model,
        news=train.news,
        popularity=popularity,
        run_id="test-run",
        data_source=train.data_source,
        encoder_name=encoder_name(encoder),
    )


@pytest.fixture(scope="module")
def client(artifacts, cfg):
    return TestClient(create_app(artifacts=artifacts, cfg=cfg))


@pytest.fixture(scope="module")
def known_user(profiles):
    return next(iter(profiles.vectors))


class TestConfig:
    def test_loads_the_repo_config(self):
        cfg = load_config()

        assert cfg.seed == 42
        assert cfg.hash

    def test_dotted_access(self, cfg):
        assert isinstance(cfg.get("encoder.dim"), int)
        assert cfg.get("nope.nothing", "default") == "default"

    def test_require_raises_on_a_missing_key(self, cfg):
        with pytest.raises(KeyError, match="missing required config key"):
            cfg.require("does.not.exist")

    def test_the_hash_tracks_the_content(self, tmp_path):
        first = tmp_path / "a.yaml"
        second = tmp_path / "b.yaml"
        first.write_text("seed: 1\n", encoding="utf-8")
        second.write_text("seed: 2\n", encoding="utf-8")

        assert Config.load(first).hash != Config.load(second).hash

    def test_the_hash_is_stable_for_identical_content(self, tmp_path):
        first = tmp_path / "a.yaml"
        second = tmp_path / "b.yaml"
        first.write_text("seed: 1\n", encoding="utf-8")
        second.write_text("seed: 1\n", encoding="utf-8")

        assert Config.load(first).hash == Config.load(second).hash

    def test_rejects_a_non_mapping(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("- just\n- a list\n", encoding="utf-8")

        with pytest.raises(ValueError, match="mapping"):
            Config.load(path)


class TestExplain:
    def test_explains_a_known_pairing(self, encoder, profiles, train, known_user):
        news_id = str(train.news["news_id"].iloc[0])
        explanation = why.explain(known_user, news_id, encoder, profiles, train.news)

        assert explanation.user_id == known_user
        assert explanation.news_id == news_id
        assert explanation.title
        assert explanation.summary
        assert explanation.term_attribution_available

    def test_surfaces_the_readers_desks(self, encoder, profiles, train, known_user):
        news_id = str(train.news["news_id"].iloc[0])
        explanation = why.explain(known_user, news_id, encoder, profiles, train.news)

        assert explanation.reader_top_categories
        assert explanation.top_history_matches

    def test_a_cold_reader_is_labelled_as_such(self, encoder, profiles, train):
        news_id = str(train.news["news_id"].iloc[0])
        explanation = why.explain("NOT_A_USER", news_id, encoder, profiles, train.news)

        assert explanation.is_cold_start
        assert "know nothing" in explanation.summary.lower()

    def test_an_unknown_article_raises(self, encoder, profiles, train, known_user):
        with pytest.raises(KeyError):
            why.explain(known_user, "NOT_AN_ARTICLE", encoder, profiles, train.news)

    def test_a_recommended_article_scores_above_a_random_one(
        self, encoder, profiles, train, ranker, known_user
    ):
        """The explanation's score must agree with the ranking it explains."""
        top_article = ranker.recommend_for_user(known_user, top_n=1)[0][0]
        random_article = str(train.news["news_id"].iloc[-1])

        top = why.explain(known_user, top_article, encoder, profiles, train.news)
        other = why.explain(known_user, random_article, encoder, profiles, train.news)

        assert top.score >= other.score


class TestArtifacts:
    def test_roundtrips_through_disk(self, artifacts, tmp_path, known_user):
        path = artifacts.save(tmp_path / "artifacts.pkl")
        loaded = Artifacts.load(path)

        assert loaded.run_id == artifacts.run_id
        assert loaded.data_source == artifacts.data_source
        assert len(loaded.profiles) == len(artifacts.profiles)

        # The derived indices are dropped on pickling and must be rebuilt on load.
        assert loaded.ranker.recommend_for_user(known_user, top_n=3)
        assert loaded.news_category

    def test_missing_artifacts_raise_a_useful_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="make run"):
            Artifacts.load(tmp_path / "absent.pkl")


class TestAPI:
    def test_health_reports_the_data_source(self, client):
        response = client.get("/health")
        body = response.json()

        assert response.status_code == 200
        assert body["status"] == "ok"
        assert body["data_source"] == "mind-format-sample"
        assert body["n_users"] > 0

    def test_recommend_returns_ranked_articles(self, client, known_user):
        response = client.get("/recommend", params={"user_id": known_user, "k": 5})
        body = response.json()

        assert response.status_code == 200
        assert len(body["recommendations"]) <= 5
        assert [r["rank"] for r in body["recommendations"]] == list(
            range(1, len(body["recommendations"]) + 1)
        )
        assert 0 <= body["send_hour"] <= 23

    def test_recommend_never_repeats_a_read_article(self, client, profiles, known_user):
        response = client.get("/recommend", params={"user_id": known_user, "k": 5})
        recommended = {r["news_id"] for r in response.json()["recommendations"]}

        assert not (recommended & set(profiles.history_of(known_user)))

    def test_recommend_applies_the_desk_cap(self, client, known_user, cfg):
        response = client.get(
            "/recommend", params={"user_id": known_user, "k": 5, "diversify": True}
        )
        categories = [r["category"] for r in response.json()["recommendations"]]

        cap = int(cfg.get("diversity.max_per_category"))
        for category in set(categories):
            assert categories.count(category) <= cap

    def test_diversification_can_be_turned_off(self, client, known_user):
        response = client.get(
            "/recommend", params={"user_id": known_user, "k": 5, "diversify": False}
        )
        assert response.status_code == 200
        assert response.json()["guardrails"] == {}

    def test_recommend_serves_a_cold_reader(self, client):
        response = client.get("/recommend", params={"user_id": "NOT_A_USER", "k": 3})
        body = response.json()

        assert response.status_code == 200
        assert body["cold_start"] is True
        assert body["recommendations"]

    def test_recommend_validates_k(self, client, known_user):
        assert client.get("/recommend", params={"user_id": known_user, "k": 0}).status_code == 422
        assert client.get("/recommend", params={"user_id": known_user, "k": 999}).status_code == 422

    def test_audience_returns_ranked_readers(self, client, train):
        news_id = str(train.news["news_id"].iloc[0])
        response = client.get("/audience", params={"news_id": news_id, "k": 10})
        body = response.json()

        assert response.status_code == 200
        assert len(body["audience"]) <= 10

        propensities = [entry["propensity"] for entry in body["audience"]]
        assert propensities == sorted(propensities, reverse=True)

    def test_audience_rejects_an_unknown_article(self, client):
        assert client.get("/audience", params={"news_id": "NOPE"}).status_code == 404

    def test_send_time_returns_the_full_curve(self, client, known_user):
        response = client.get("/send-time", params={"user_id": known_user})
        body = response.json()

        assert response.status_code == 200
        assert 0 <= body["best_hour"] <= 23
        assert len(body["engagement_by_hour"]) == 24

    def test_send_time_serves_an_unknown_reader(self, client):
        response = client.get("/send-time", params={"user_id": "NOT_A_USER"})
        body = response.json()

        assert response.status_code == 200
        assert body["personalised"] is False
        assert body["best_hour"] == body["global_best_hour"]

    def test_why_explains_a_recommendation(self, client, known_user):
        recommendations = client.get(
            "/recommend", params={"user_id": known_user, "k": 1}
        ).json()["recommendations"]
        news_id = recommendations[0]["news_id"]

        response = client.get("/why", params={"user_id": known_user, "news_id": news_id})
        body = response.json()

        assert response.status_code == 200
        assert body["summary"]
        assert body["news_id"] == news_id

    def test_why_rejects_an_unknown_article(self, client, known_user):
        response = client.get("/why", params={"user_id": known_user, "news_id": "NOPE"})
        assert response.status_code == 404

    def test_every_response_carries_its_data_source(self, client, known_user, train):
        news_id = str(train.news["news_id"].iloc[0])
        endpoints = [
            ("/recommend", {"user_id": known_user}),
            ("/audience", {"news_id": news_id, "k": 5}),
            ("/send-time", {"user_id": known_user}),
            ("/why", {"user_id": known_user, "news_id": news_id}),
        ]

        for path, params in endpoints:
            body = client.get(path, params=params).json()
            assert body["data_source"] == "mind-format-sample", path

    def test_health_without_artifacts_says_so(self, tmp_path, cfg):
        raw = dict(cfg.raw)
        raw["paths"] = {**raw["paths"], "runs_dir": str(tmp_path / "empty")}
        empty_cfg = Config(raw=raw, hash="test", source_path=cfg.source_path)

        client = TestClient(create_app(cfg=empty_cfg))
        assert client.get("/health").json()["status"] == "no_artifacts"


class TestBatch:
    def test_builds_a_send_plan(self, cfg, artifacts):
        plan = batch.build_send_plan(cfg, artifacts, max_users=10)

        assert set(plan.columns) == {"user_id", "news_id", "rank", "score", "send_hour"}
        assert plan["user_id"].nunique() <= 10
        assert plan["score"].between(0, 1).all()
        assert plan["send_hour"].between(0, 23).all()

    def test_never_recommends_a_read_article(self, cfg, artifacts):
        plan = batch.build_send_plan(cfg, artifacts, max_users=10)

        for user_id, group in plan.groupby("user_id"):
            history = set(artifacts.profiles.history_of(str(user_id)))
            assert not (set(group["news_id"]) & history)

    def test_capping_is_applied_and_persisted(self, cfg, artifacts):
        capped, metrics = batch.run(cfg, artifacts, max_users=10, write=True)

        assert len(capped) > 0
        assert metrics.n_sends_proposed >= len(capped)

        counts = capped.groupby("user_id").size()
        assert (counts <= artifacts.fatigue.max_per_week).all()

        # Ranks are renumbered after capping, so a reader's ranks have no gaps.
        for _, group in capped.groupby("user_id"):
            assert list(group["rank"]) == list(range(1, len(group) + 1))

    def test_written_rows_are_readable(self, cfg, artifacts):
        from newspush.data import db

        batch.run(cfg, artifacts, max_users=10, write=True)

        conn = db.connect(cfg.path("paths.db_path"))
        try:
            user_id = str(
                batch.build_send_plan(cfg, artifacts, max_users=10)["user_id"].iloc[0]
            )
            rows = db.read_campaign_recommendations(conn, user_id, run_id=artifacts.run_id)
            assert not rows.empty
        finally:
            conn.close()


class TestPipeline:
    @pytest.fixture(scope="class")
    def metrics(self, cfg):
        from newspush import pipeline

        return pipeline.run(cfg)

    def test_writes_metrics_with_full_provenance(self, cfg, metrics):
        run_dir = cfg.path("paths.runs_dir") / metrics["run_id"]
        path = run_dir / "metrics.json"

        assert path.is_file()

        written = json.loads(path.read_text(encoding="utf-8"))
        assert written["data_source"] == "mind-format-sample"
        assert written["config_hash"] == cfg.hash
        assert written["seed"] == cfg.seed
        assert written["encoder"] == "tfidf-svd"

    def test_labels_which_results_are_simulated(self, metrics):
        """The honesty rule, enforced. Anything produced inside the simulator must be
        listed as such, and the simulated keys must be the ones that are named."""
        provenance = metrics["provenance"]

        assert set(provenance["measured_in_simulator"]) == {
            "ab_test_simulated",
            "off_policy_simulated",
            "bandit_simulated",
            "uplift_simulated",
        }
        for key in provenance["measured_in_simulator"]:
            assert key in metrics

        for key in provenance["measured_on_logged_data"]:
            assert key in metrics

    def test_reports_every_headline_metric(self, metrics):
        personalised = metrics["content_selection"]["personalised"]

        for key in ("auc", "mrr", "ndcg_at_5", "ndcg_at_10"):
            assert 0.0 <= personalised[key] <= 1.0

        assert 0.0 <= metrics["audience"]["roc_auc"] <= 1.0
        assert metrics["send_time"]["n_eval_impressions"] > 0
        assert metrics["campaign"]["rows_written"] > 0

    def test_personalisation_beats_popularity(self, metrics):
        personalised = metrics["content_selection"]["personalised"]["auc"]
        baseline = metrics["content_selection"]["popularity_baseline"]["auc"]

        assert personalised > baseline

    def test_writes_loadable_artifacts(self, cfg, metrics):
        loaded = Artifacts.load(artifacts_path(cfg))
        assert loaded.run_id == metrics["run_id"]

    def test_is_deterministic(self, cfg, metrics):
        """Same seed, same config, same numbers. Without this, nothing else in the
        metrics file can be trusted."""
        from newspush import pipeline

        again = pipeline.run(cfg, skip_batch=True)

        first = metrics["content_selection"]["personalised"]
        second = again["content_selection"]["personalised"]

        assert first["auc"] == pytest.approx(second["auc"])
        assert first["ndcg_at_10"] == pytest.approx(second["ndcg_at_10"])
        assert metrics["audience"]["roc_auc"] == pytest.approx(again["audience"]["roc_auc"])
        assert np.isclose(
            metrics["off_policy_simulated"]["dr"]["value"],
            again["off_policy_simulated"]["dr"]["value"],
        )
