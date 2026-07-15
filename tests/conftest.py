"""Shared fixtures.

Every test runs against the simulated sample, never against real MIND: the suite must
pass on a fresh clone with no dataset downloaded. The sample is deliberately small so
the whole suite stays fast.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from newspush.config import Config, load_config
from newspush.data import db, make_sample
from newspush.features.text import build_encoder
from newspush.features.users import build_profiles
from newspush.models import audience, content_selection, send_time
from newspush.models.content_selection import ContentRanker
from newspush.models.fatigue import FatigueModel

SMALL_OVERRIDES = {
    "sample": {
        "n_users": 120,
        "n_news": 150,
        "n_impressions": 900,
        "candidates_per_impression": 8,
        "history_len_range": [3, 12],
        "topic_affinity_weight": 2.2,
        "hour_effect_weight": 1.1,
        "base_click_logit": -2.0,
    },
    "encoder": {"dim": 32},
    "content_selection": {"max_eval_impressions": 200},
    "audience": {"max_train_rows": 5000, "audience_k": 20},
    "ab_test": {"n_users_per_arm": 40, "emails_per_user": 3},
    "bandit": {"n_rounds": 200, "n_arms": 6, "context_dim": 6},
    "uplift": {"n_users": 120},
    "serving": {"batch_top_n": 3, "batch_max_users": 40},
}


def _deep_merge(base: dict, overrides: dict) -> dict:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@pytest.fixture(scope="session")
def cfg(tmp_path_factory) -> Config:
    """A small, fast config, isolated to a temp directory."""
    base = load_config().raw
    raw = _deep_merge(base, SMALL_OVERRIDES)

    workdir = tmp_path_factory.mktemp("newspush")
    raw["paths"] = {
        "data_dir": str(workdir / "data"),
        # Point at paths that cannot exist, so the tests always take the sample branch
        # even on a machine where real MIND happens to be downloaded.
        "mind_train": str(workdir / "absent_train"),
        "mind_dev": str(workdir / "absent_dev"),
        "runs_dir": str(workdir / "runs"),
        "db_path": str(workdir / "runs" / "events.db"),
    }

    config_path = workdir / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return Config.load(config_path)


@pytest.fixture(scope="session")
def data(cfg: Config):
    return make_sample.generate(cfg)


@pytest.fixture(scope="session")
def train(data):
    return data[0]


@pytest.fixture(scope="session")
def dev(data):
    return data[1]


@pytest.fixture(scope="session")
def encoder(cfg: Config, train):
    return build_encoder(cfg).fit(train.news)


@pytest.fixture(scope="session")
def profiles(train, encoder):
    return build_profiles(train, encoder)


@pytest.fixture(scope="session")
def popularity(train):
    return content_selection.popularity_baseline(train)


@pytest.fixture(scope="session")
def ranker(encoder, profiles) -> ContentRanker:
    return ContentRanker(encoder, profiles)


@pytest.fixture(scope="session")
def conn(cfg: Config, train, dev):
    connection = db.connect(cfg.path("paths.db_path"))
    db.load_events(connection, train)
    db.load_events(connection, dev)
    yield connection
    connection.close()


@pytest.fixture(scope="session")
def send_time_model(cfg: Config, conn) -> send_time.SendTimeModel:
    return send_time.SendTimeModel(cfg).fit(db.user_hour_counts(conn, "train"))


@pytest.fixture(scope="session")
def fatigue_model(cfg: Config, conn) -> FatigueModel:
    return FatigueModel(cfg).fit_engagement(db.user_stats(conn, "train"))


@pytest.fixture(scope="session")
def propensity(cfg: Config, encoder, profiles, train, popularity) -> audience.PropensityModel:
    model = audience.PropensityModel(cfg, encoder, profiles)
    model.fit_catalogue(train.news, popularity)
    model.fit(train)
    return model


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
