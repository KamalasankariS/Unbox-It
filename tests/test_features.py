"""Feature layer: the article encoder and user profiles."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from newspush.features.text import (
    TfidfSvdEncoder,
    article_text,
    build_encoder,
    encoder_name,
    l2_normalize,
)
from newspush.features.users import build_profiles, profile_topic_entropy


class TestNormalisation:
    def test_unit_norm(self):
        normalised = l2_normalize(np.array([[3.0, 4.0]]))
        assert np.allclose(np.linalg.norm(normalised, axis=1), 1.0)

    def test_zero_rows_survive_without_dividing_by_zero(self):
        normalised = l2_normalize(np.array([[0.0, 0.0], [3.0, 4.0]]))

        assert np.allclose(normalised[0], 0.0)
        assert np.isfinite(normalised).all()


class TestArticleText:
    def test_combines_title_and_abstract(self):
        news = pd.DataFrame({"title": ["Title"], "abstract": ["Abstract"]})
        assert article_text(news).iloc[0] == "Title. Abstract"

    def test_survives_an_empty_abstract(self):
        news = pd.DataFrame({"title": ["Title"], "abstract": [""]})
        assert "Title" in article_text(news).iloc[0]


class TestEncoder:
    def test_vectors_are_normalised(self, encoder):
        norms = np.linalg.norm(encoder.matrix, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_unknown_articles_get_zero_vectors(self, encoder):
        assert not encoder.known("NOT_AN_ARTICLE")
        assert np.allclose(encoder.vec("NOT_AN_ARTICLE"), 0.0)

    def test_vecs_matches_vec(self, encoder, train):
        ids = list(train.news["news_id"])[:5]
        stacked = encoder.vecs(ids)

        for i, news_id in enumerate(ids):
            assert np.allclose(stacked[i], encoder.vec(news_id))

    def test_vecs_handles_unknown_ids(self, encoder, train):
        known = str(train.news["news_id"].iloc[0])
        stacked = encoder.vecs([known, "MISSING"])

        assert np.allclose(stacked[1], 0.0)
        assert not np.allclose(stacked[0], 0.0)

    def test_same_topic_articles_are_closer_than_different_topics(self, encoder, train):
        """The whole system rests on this. If article vectors do not carry topic
        structure, every downstream number is noise."""
        by_category = train.news.groupby("category")["news_id"].apply(list)
        categories = [c for c, ids in by_category.items() if len(ids) >= 4][:4]
        assert len(categories) >= 2

        same, different = [], []
        for i, category in enumerate(categories):
            ids = by_category[category][:4]
            vectors = encoder.vecs(ids)
            same.extend(
                float(vectors[a] @ vectors[b])
                for a in range(len(ids))
                for b in range(a + 1, len(ids))
            )
            other_ids = by_category[categories[(i + 1) % len(categories)]][:4]
            other_vectors = encoder.vecs(other_ids)
            different.extend(float(v @ w) for v in vectors for w in other_vectors)

        assert np.mean(same) > np.mean(different)

    def test_is_deterministic(self, cfg, train):
        first = build_encoder(cfg).fit(train.news)
        second = build_encoder(cfg).fit(train.news)
        assert np.allclose(first.matrix, second.matrix)

    def test_rejects_a_catalogue_that_is_too_small(self, cfg):
        news = pd.DataFrame(
            {
                "news_id": ["N1"],
                "category": ["news"],
                "subcategory": ["x"],
                "title": ["word"],
                "abstract": [""],
                "url": [""],
                "title_entities": ["[]"],
                "abstract_entities": ["[]"],
            }
        )
        with pytest.raises(ValueError, match="too small"):
            build_encoder(cfg).fit(news)

    def test_dim_is_capped_by_the_catalogue(self, cfg, train):
        """Asking for more SVD components than the vocabulary supports must degrade
        gracefully, not raise."""
        small = train.news.head(10)
        encoder = build_encoder(cfg).fit(small)
        assert encoder.dim() <= 10

    def test_unfitted_encoder_raises(self, cfg):
        with pytest.raises(RuntimeError, match="not fitted"):
            TfidfSvdEncoder(cfg).dim()

    def test_top_terms_are_ranked(self, encoder, train):
        news_id = str(train.news["news_id"].iloc[0])
        terms = encoder.top_terms(news_id, k=5)

        assert terms
        weights = [weight for _, weight in terms]
        assert weights == sorted(weights, reverse=True)

    def test_top_terms_of_an_unknown_article_is_empty(self, encoder):
        assert encoder.top_terms("MISSING") == []


class TestEncoderFactory:
    def test_builds_the_tfidf_encoder_by_default(self, cfg):
        assert isinstance(build_encoder(cfg), TfidfSvdEncoder)
        assert encoder_name(build_encoder(cfg)) == "tfidf-svd"

    def test_falls_back_when_sentence_transformers_is_missing(self, cfg, monkeypatch):
        """The optional dependency must not be able to break a base install."""
        import builtins

        real_import = builtins.__import__

        def fail_on_sentence_transformers(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fail_on_sentence_transformers)

        raw = dict(cfg.raw)
        raw["encoder"] = {**raw["encoder"], "kind": "sentence-transformer"}
        from newspush.config import Config

        encoder = build_encoder(Config(raw=raw, hash="test", source_path=cfg.source_path))
        assert isinstance(encoder, TfidfSvdEncoder)

    def test_rejects_an_unknown_kind(self, cfg):
        from newspush.config import Config

        raw = dict(cfg.raw)
        raw["encoder"] = {**raw["encoder"], "kind": "word2vec"}

        with pytest.raises(ValueError, match="unknown encoder.kind"):
            build_encoder(Config(raw=raw, hash="test", source_path=cfg.source_path))


class TestUserProfiles:
    def test_profiles_are_normalised(self, profiles):
        for vector in profiles.vectors.values():
            assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-6)

    def test_unknown_readers_are_cold(self, profiles):
        assert profiles.is_cold("NOT_A_USER")
        assert np.allclose(profiles.get("NOT_A_USER"), 0.0)

    def test_cold_readers_fall_back_to_the_population_centroid(self, profiles):
        fallback = profiles.get_or_global("NOT_A_USER")

        assert np.any(fallback)
        assert np.allclose(fallback, profiles.global_profile)

    def test_history_is_recorded(self, profiles, train):
        user_id = train.behaviors["user_id"].iloc[0]
        assert profiles.history_of(user_id)

    def test_history_is_not_double_counted(self, train, encoder):
        """MIND repeats a reader's history on every row. Folding it in once per row
        would let a frequent reader's profile collapse onto their own history."""
        profiles = build_profiles(train, encoder, include_clicked_impressions=False)

        user_id = train.behaviors["user_id"].iloc[0]
        first_row_history = train.behaviors[
            train.behaviors["user_id"] == user_id
        ]["history"].iloc[0].split()

        known = [n for n in first_row_history if encoder.known(n)]
        assert len(profiles.history_of(user_id)) == len(known)

    def test_profile_points_toward_the_readers_own_history(self, profiles, encoder):
        """A profile must be closer to what the reader actually clicked than to a
        random article, or personalisation is not happening."""
        user_id = next(iter(profiles.vectors))
        profile = profiles.get(user_id)
        history = profiles.history_of(user_id)

        history_similarity = float(np.mean(encoder.vecs(history) @ profile))
        catalogue_similarity = float(np.mean(encoder.matrix @ profile))

        assert history_similarity > catalogue_similarity

    def test_empty_data_yields_no_profiles(self, train, encoder):
        from newspush.data.schema import SAMPLE, MindData

        empty = MindData(train.news, train.behaviors.head(0), SAMPLE, "train")
        profiles = build_profiles(empty, encoder)

        assert len(profiles) == 0
        assert np.allclose(profiles.global_profile, 0.0)


class TestTopicEntropy:
    def test_a_specialist_scores_below_a_generalist(self, profiles, train):
        category = dict(zip(train.news["news_id"].astype(str), train.news["category"].astype(str)))
        entropies = [
            profile_topic_entropy(user_id, profiles, category)
            for user_id in list(profiles.vectors)[:40]
        ]

        assert all(e >= 0 for e in entropies)
        assert max(entropies) > min(entropies)

    def test_an_unknown_reader_has_zero_entropy(self, profiles):
        assert profile_topic_entropy("NOT_A_USER", profiles, {}) == 0.0
