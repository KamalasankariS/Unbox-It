"""Data layer: schema, MIND parsing, the sample generator, and the SQL store."""

from __future__ import annotations

import pandas as pd
import pytest

from newspush.data import db, make_sample, mind_loader
from newspush.data.acquire import acquire, real_mind_available
from newspush.data.schema import (
    BEHAVIOR_COLUMNS,
    NEWS_COLUMNS,
    REAL,
    SAMPLE,
    MindData,
    parse_hour,
    parse_impression,
)


class TestParsing:
    def test_parses_candidates_and_labels(self):
        impression = parse_impression("1", "U1", "11/11/2019 9:05:58 AM", "N1 N2", "N3-1 N4-0")

        assert impression is not None
        assert impression.candidates == ["N3", "N4"]
        assert impression.labels == [1, 0]
        assert impression.history == ["N1", "N2"]
        assert impression.hour == 9

    def test_empty_impressions_yield_none(self):
        assert parse_impression("1", "U1", "11/11/2019 9:05:58 AM", "", "") is None

    def test_malformed_tokens_are_skipped_not_fatal(self):
        impression = parse_impression("1", "U1", "11/11/2019 9:05:58 AM", "", "N1-1 garbage N2-9 N3-0")

        assert impression is not None
        assert impression.candidates == ["N1", "N3"]
        assert impression.labels == [1, 0]

    def test_news_id_containing_a_hyphen_splits_on_the_last_one(self):
        impression = parse_impression("1", "U1", "11/11/2019 9:05:58 AM", "", "N-123-1")

        assert impression is not None
        assert impression.candidates == ["N-123"]
        assert impression.labels == [1]

    @pytest.mark.parametrize(
        ("timestamp", "expected"),
        [
            ("11/11/2019 9:05:58 AM", 9),
            ("11/11/2019 11:00:00 PM", 23),
            ("11/11/2019 12:00:00 AM", 0),
            ("not a timestamp", 12),
        ],
    )
    def test_parse_hour(self, timestamp, expected):
        assert parse_hour(timestamp) == expected


class TestSchema:
    def test_rejects_an_unknown_data_source(self, train):
        with pytest.raises(ValueError, match="data_source"):
            MindData(train.news, train.behaviors, data_source="made-up", split="train")

    def test_rejects_missing_columns(self, train):
        with pytest.raises(ValueError, match="missing columns"):
            MindData(train.news.drop(columns=["title"]), train.behaviors, SAMPLE, "train")


class TestSampleGenerator:
    def test_matches_the_real_mind_schema(self, train, dev):
        assert list(train.news.columns) == NEWS_COLUMNS
        assert list(train.behaviors.columns) == BEHAVIOR_COLUMNS
        assert train.data_source == SAMPLE
        assert train.is_simulated and dev.is_simulated

    def test_is_deterministic_under_a_fixed_seed(self, cfg):
        first, _ = make_sample.generate(cfg)
        second, _ = make_sample.generate(cfg)
        pd.testing.assert_frame_equal(first.behaviors, second.behaviors)

    def test_splits_share_users_and_articles(self, train, dev):
        train_users = set(train.behaviors["user_id"])
        dev_users = set(dev.behaviors["user_id"])

        # Profiles learned on train are only meaningful on dev if the populations overlap.
        assert train_users & dev_users
        assert set(train.news["news_id"]) == set(dev.news["news_id"])

    def test_plants_a_recoverable_topic_signal(self, train):
        """A user's clicks should concentrate in their preferred desks, or there is
        nothing for the content ranker to learn and the suite proves nothing."""
        category = dict(zip(train.news["news_id"], train.news["category"]))

        clicked_categories: dict[str, dict[str, int]] = {}
        for impression in train.impressions():
            for news_id, label in zip(impression.candidates, impression.labels):
                if label == 1:
                    counts = clicked_categories.setdefault(impression.user_id, {})
                    desk = category[news_id]
                    counts[desk] = counts.get(desk, 0) + 1

        concentrations = [
            max(counts.values()) / sum(counts.values())
            for counts in clicked_categories.values()
            if sum(counts.values()) >= 5
        ]
        assert concentrations, "no user clicked enough to measure concentration"

        n_desks = train.news["category"].nunique()
        mean_concentration = sum(concentrations) / len(concentrations)
        assert mean_concentration > 2.0 / n_desks

    def test_click_rate_is_plausible(self, train):
        clicks = sum(sum(i.labels) for i in train.impressions())
        shown = sum(len(i.labels) for i in train.impressions())
        assert 0.02 < clicks / shown < 0.5


class TestAcquire:
    def test_falls_back_to_the_sample_when_real_mind_is_absent(self, cfg):
        assert not real_mind_available(cfg)

        train, dev = acquire(cfg)
        assert train.data_source == SAMPLE
        assert dev.data_source == SAMPLE

    def test_real_loader_requires_both_tsvs(self, tmp_path):
        assert not mind_loader.split_available(tmp_path)

        (tmp_path / "news.tsv").write_text("", encoding="utf-8")
        assert not mind_loader.split_available(tmp_path)

        (tmp_path / "behaviors.tsv").write_text("", encoding="utf-8")
        assert mind_loader.split_available(tmp_path)

    def test_real_loader_reads_mind_format(self, tmp_path):
        (tmp_path / "news.tsv").write_text(
            "N1\tsports\tnfl\tA title\tAn abstract\thttps://x\t[]\t[]\n"
            'N2\tnews\tpolitics\tQuoted "title" here\t\thttps://y\t[]\t[]\n',
            encoding="utf-8",
        )
        (tmp_path / "behaviors.tsv").write_text(
            "1\tU1\t11/11/2019 9:05:58 AM\tN1\tN2-1 N1-0\n", encoding="utf-8"
        )

        data = mind_loader.load_split(tmp_path, split="train")

        assert data.data_source == REAL
        assert len(data.news) == 2
        # Unescaped quotes are routine in MIND titles and must not break the reader.
        assert data.news.iloc[1]["title"] == 'Quoted "title" here'

        impressions = list(data.impressions())
        assert impressions[0].candidates == ["N2", "N1"]
        assert impressions[0].labels == [1, 0]

    def test_real_loader_raises_when_absent(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            mind_loader.load_split(tmp_path / "nope", split="train")


class TestEventStore:
    def test_one_row_per_candidate(self, conn, train):
        expected = sum(len(i.candidates) for i in train.impressions())
        actual = conn.execute("SELECT COUNT(*) FROM events WHERE split = 'train'").fetchone()[0]
        assert actual == expected

    def test_reloading_replaces_rather_than_duplicates(self, cfg, train, tmp_path):
        connection = db.connect(tmp_path / "events.db")
        try:
            first = db.load_events(connection, train)
            second = db.load_events(connection, train)

            total = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            assert first == second == total
        finally:
            connection.close()

    def test_ctr_by_hour_covers_the_clock(self, conn):
        frame = db.ctr_by_hour(conn, "train")

        assert set(frame.columns) == {"hour", "impressions", "clicks", "ctr"}
        assert frame["hour"].between(0, 23).all()
        assert frame["ctr"].between(0, 1).all()

    def test_top_users_are_ordered_by_engagement(self, conn):
        frame = db.top_users(conn, "train", min_impressions=1, limit=10)

        assert len(frame) <= 10
        assert frame["clicks"].is_monotonic_decreasing

    def test_ctr_by_category_joins_the_catalogue(self, conn, train):
        frame = db.ctr_by_category(conn, train.news, "train")

        assert set(frame["category"]) <= set(train.news["category"])
        assert frame["ctr"].between(0, 1).all()

    def test_user_hour_counts_never_exceed_impressions(self, conn):
        frame = db.user_hour_counts(conn, "train")
        assert (frame["clicks"] <= frame["impressions"]).all()

    def test_campaign_recommendations_roundtrip(self, cfg, tmp_path):
        connection = db.connect(tmp_path / "campaign.db")
        try:
            plan = pd.DataFrame(
                [
                    {"user_id": "U1", "news_id": "N1", "rank": 1, "score": 0.9, "send_hour": 8},
                    {"user_id": "U1", "news_id": "N2", "rank": 2, "score": 0.5, "send_hour": 8},
                ]
            )
            written = db.write_campaign_recommendations(connection, plan, run_id="r1")
            assert written == 2

            read_back = db.read_campaign_recommendations(connection, "U1", run_id="r1")
            assert list(read_back["news_id"]) == ["N1", "N2"]

            # Re-writing the same run must replace it, not append a second copy.
            db.write_campaign_recommendations(connection, plan, run_id="r1")
            assert len(db.read_campaign_recommendations(connection, "U1", run_id="r1")) == 2
        finally:
            connection.close()

    def test_campaign_recommendations_validates_columns(self, tmp_path):
        connection = db.connect(tmp_path / "bad.db")
        try:
            with pytest.raises(ValueError, match="missing columns"):
                db.write_campaign_recommendations(connection, pd.DataFrame({"user_id": ["U1"]}), "r1")
        finally:
            connection.close()
