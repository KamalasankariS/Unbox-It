"""End-to-end pipeline: acquire, train, evaluate, serve, and record.

One deterministic run. Writes runs/<run_id>/metrics.json carrying every number in the
README together with the provenance needed to read it: the data source, the config
hash, the encoder that actually ran, and which results came from the simulator rather
than from logged data.

Stages:
    1  acquire            real MIND if present, otherwise the labelled sample
    2  encode             article vectors over the combined catalogue
    3  profile            reader vectors from train click history
    4  content selection  ranker and popularity baseline, evaluated on dev
    5  SQL                event store, CTR by hour and by desk
    6  send-time          fit on train, evaluate uplift on dev
    7  audience           propensity model, ROC-AUC and precision@k on dev
    8  simulate           response oracle fitted on held-out dev
    9  A/B                popularity vs personalisation, content-only and with send-time
    10 off-policy         IPS, SNIPS and DR against the simulator's ground truth
    11 bandit             Thompson sampling vs epsilon-greedy, static and random
    12 uplift             T-learner on a constructed trial arm
    13 batch              campaign_recommendations table, under the fatigue cap

Profiles are learned on train and every metric is reported on dev. The response oracle
is fitted on dev precisely so it is not the model any competing policy was trained on.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from newspush.artifacts import Artifacts, artifacts_path
from newspush.config import Config, load_config
from newspush.data import db
from newspush.data.acquire import acquire
from newspush.data.schema import MindData
from newspush.experiment import ab_test, off_policy
from newspush.experiment.simulator import ResponseSimulator
from newspush.features.text import ArticleEncoder, build_encoder, encoder_name
from newspush.features.users import UserProfiles, build_profiles
from newspush.models import audience, bandit, content_selection, send_time, uplift
from newspush.models.content_selection import ContentRanker
from newspush.models.fatigue import FatigueModel
from newspush.serving import batch

log = logging.getLogger(__name__)

# The A/B, off-policy and bandit stages need a tractable article pool. The full MIND
# catalogue is 50k+ articles, and an email campaign chooses from an editorial shortlist,
# not from everything ever published.
CAMPAIGN_POOL_SIZE = 500
OFF_POLICY_CONTEXTS = 5000
OFF_POLICY_ACTIONS = 10


def run(cfg: Config, skip_batch: bool = False) -> dict[str, Any]:
    np.random.seed(cfg.seed)

    run_id = _make_run_id(cfg)
    log.info("run_id=%s  config_hash=%s  seed=%d", run_id, cfg.hash, cfg.seed)

    train, dev = acquire(cfg)
    encoder, catalogue = _fit_encoder(cfg, train, dev)

    # Profiles fold in dev users' history field (their past clicks) but never dev's
    # impression labels. MIND splits by time, so most dev users are absent from train
    # and would otherwise score cold. See build_profiles.
    profiles = build_profiles(train, encoder, extra_history=dev)

    popularity = content_selection.popularity_baseline(train)
    ranker = ContentRanker(encoder, profiles)

    ranking_metrics = content_selection.evaluate(ranker, dev, cfg)
    popularity_metrics = content_selection.evaluate_popularity(dev, popularity, cfg)

    conn = db.connect(cfg.path("paths.db_path"))
    try:
        db.load_events(conn, train)
        db.load_events(conn, dev)
        sql_analytics = _sql_analytics(conn, catalogue)

        send_time_model = send_time.SendTimeModel(cfg).fit(db.user_hour_counts(conn, "train"))
        fatigue_model = FatigueModel(cfg).fit_engagement(db.user_stats(conn, "train"))
    finally:
        conn.close()

    send_time_metrics = send_time.evaluate(send_time_model, dev)

    propensity, audience_metrics = audience.train_and_evaluate(
        cfg, encoder, profiles, train, dev, popularity, catalogue=catalogue
    )

    simulator = ResponseSimulator.fit(cfg, encoder, profiles, dev, popularity, catalogue=catalogue)
    campaign_pool = _campaign_pool(popularity, encoder, CAMPAIGN_POOL_SIZE)

    content_only = ab_test.simulate_campaign(
        cfg, simulator, ranker, profiles, popularity, send_time=None, candidate_pool=campaign_pool
    )
    with_send_time = ab_test.simulate_campaign(
        cfg,
        simulator,
        ranker,
        profiles,
        popularity,
        send_time=send_time_model,
        candidate_pool=campaign_pool,
    )

    ope_report = off_policy.run_off_policy_evaluation(
        cfg,
        simulator,
        encoder,
        profiles,
        popularity,
        candidate_pool=campaign_pool,
        n_contexts=OFF_POLICY_CONTEXTS,
        n_actions=OFF_POLICY_ACTIONS,
    )

    bandit_metrics = _run_bandit(cfg, simulator, profiles, catalogue, popularity)
    uplift_metrics = _run_uplift(cfg, simulator, propensity, profiles, fatigue_model, campaign_pool)

    artifacts = Artifacts(
        encoder=encoder,
        profiles=profiles,
        propensity=propensity,
        send_time=send_time_model,
        fatigue=fatigue_model,
        news=catalogue,
        popularity=popularity,
        run_id=run_id,
        data_source=train.data_source,
        encoder_name=encoder_name(encoder),
    )
    artifacts.save(artifacts_path(cfg))

    fatigue_metrics = None
    n_campaign_rows = 0
    if not skip_batch:
        send_plan, fatigue_metrics = batch.run(cfg, artifacts)
        n_campaign_rows = len(send_plan)

    metrics = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_source": train.data_source,
        "config_hash": cfg.hash,
        "seed": cfg.seed,
        "encoder": encoder_name(encoder),
        "dataset": {
            "train_impressions": int(len(train.behaviors)),
            "dev_impressions": int(len(dev.behaviors)),
            "articles": int(len(artifacts.news)),
            "users_profiled": len(profiles),
        },
        "content_selection": {
            "personalised": ranking_metrics.to_dict(),
            "popularity_baseline": popularity_metrics.to_dict(),
        },
        "audience": audience_metrics.to_dict(),
        "send_time": send_time_metrics.to_dict(),
        "ab_test_simulated": {
            "content_only": content_only.to_dict(),
            "content_and_send_time": with_send_time.to_dict(),
        },
        "off_policy_simulated": ope_report.to_dict(),
        "bandit_simulated": bandit_metrics.to_dict(),
        "uplift_simulated": uplift_metrics.to_dict(),
        "campaign": {
            "rows_written": n_campaign_rows,
            "fatigue": fatigue_metrics.to_dict() if fatigue_metrics else None,
        },
        "sql_analytics": sql_analytics,
        "provenance": _provenance(train),
    }

    _write_metrics(cfg, run_id, metrics)
    _log_summary(metrics)
    return metrics


def _fit_encoder(cfg: Config, train: MindData, dev: MindData) -> tuple[ArticleEncoder, pd.DataFrame]:
    """Fit on the union of both catalogues, and return that catalogue.

    Dev contains articles that never appear in train, and an article's text is known at
    send time. Encoding it is not label leakage; failing to encode it would score every
    unseen dev candidate as a zero vector.
    """
    catalogue = (
        pd.concat([train.news, dev.news], ignore_index=True)
        .drop_duplicates(subset="news_id", keep="first")
        .reset_index(drop=True)
    )
    log.info("article catalogue: %d unique articles across train and dev", len(catalogue))

    return build_encoder(cfg).fit(catalogue), catalogue


def _campaign_pool(
    popularity: dict[str, float],
    encoder: ArticleEncoder,
    size: int,
) -> list[str]:
    """The editorial shortlist a campaign chooses from: the most popular known articles."""
    known = [news_id for news_id in popularity if encoder.known(news_id)]
    return sorted(known, key=lambda news_id: popularity[news_id], reverse=True)[:size]


def _run_bandit(
    cfg: Config,
    simulator: ResponseSimulator,
    profiles: UserProfiles,
    catalogue: pd.DataFrame,
    popularity: dict[str, float],
) -> bandit.BanditMetrics:
    """Set up the desk-level bandit and run it against its baselines.

    Arms are desks. Each desk is represented by its most popular article, and the true
    expected reward of an arm is the simulator's click probability for that reader,
    that article and that hour.
    """
    rng = np.random.default_rng(cfg.seed + 3)
    n_rounds = int(cfg.require("bandit.n_rounds"))
    context_dim = int(cfg.require("bandit.context_dim"))

    representatives = _desk_representatives(catalogue, popularity, int(cfg.require("bandit.n_arms")))
    if len(representatives) < 2:
        raise ValueError("need at least two desks with articles to run the bandit")

    users = list(profiles.vectors)
    round_users = [users[i] for i in rng.integers(0, len(users), size=n_rounds)]
    round_hours = rng.integers(0, 24, size=n_rounds).tolist()

    contexts = _bandit_contexts(cfg, profiles, round_users, context_dim)

    arm_rewards = np.column_stack(
        [
            simulator.click_prob(round_users, [news_id] * n_rounds, round_hours)
            for news_id in representatives.values()
        ]
    )

    log.info("bandit arms (desk -> representative article): %s", dict(representatives))
    return bandit.run_bandit_experiment(cfg, contexts, arm_rewards)


def _desk_representatives(
    news: pd.DataFrame,
    popularity: dict[str, float],
    n_arms: int,
) -> dict[str, str]:
    """The most popular article per desk, for the top-N desks by article count."""
    catalogue = news.copy()
    catalogue["popularity"] = catalogue["news_id"].map(popularity).fillna(0.0)

    top_desks = catalogue["category"].value_counts().head(n_arms).index
    representatives: dict[str, str] = {}

    for desk in top_desks:
        desk_articles = catalogue[catalogue["category"] == desk]
        best = desk_articles.loc[desk_articles["popularity"].idxmax()]
        representatives[str(desk)] = str(best["news_id"])

    return representatives


def _bandit_contexts(
    cfg: Config,
    profiles: UserProfiles,
    users: list[str],
    context_dim: int,
) -> np.ndarray:
    """Reduce reader profiles to the bandit's context dimension, plus a bias term.

    The bandit's per-arm posterior is O(d^2), so it wants a compact context. PCA over
    the profile matrix keeps the directions readers actually differ along.
    """
    profile_matrix = np.stack([profiles.get(user) for user in users])
    n_components = min(context_dim - 1, profile_matrix.shape[1], len(users))

    reduced = PCA(n_components=n_components, random_state=cfg.seed).fit_transform(profile_matrix)

    # Standardise, so no single component dominates the linear model's scale.
    spread = reduced.std(axis=0)
    reduced = reduced / np.where(spread > 0, spread, 1.0)

    bias = np.ones((len(users), 1))
    return np.hstack([reduced, bias])


def _run_uplift(
    cfg: Config,
    simulator: ResponseSimulator,
    propensity: audience.PropensityModel,
    profiles: UserProfiles,
    fatigue_model: FatigueModel,
    campaign_pool: list[str],
) -> uplift.UpliftMetrics:
    """Run the T-learner on the constructed trial. See uplift.build_simulated_trial."""
    rng = np.random.default_rng(cfg.seed + 5)
    n_users = int(cfg.require("uplift.n_users"))

    users = list(profiles.vectors)
    n_users = min(n_users, len(users))

    sampled = [users[i] for i in rng.choice(len(users), size=n_users, replace=False)]
    articles = [campaign_pool[i] for i in rng.integers(0, len(campaign_pool), size=n_users)]
    hours = rng.integers(0, 24, size=n_users).tolist()

    click_prob = simulator.click_prob(sampled, articles, hours)
    engagement = np.array([fatigue_model.engagement_percentile(user) for user in sampled])

    treatment, outcome, note = uplift.build_simulated_trial(cfg, click_prob, engagement, rng)

    # The uplift model sees the reader and the offer, never the simulator's probability.
    features = np.column_stack(
        [propensity.featurize(sampled, articles, hours), engagement],
    )

    model = uplift.TLearner(cfg).fit(features, treatment, outcome)
    return uplift.evaluate_uplift(model, features, treatment, outcome, data_note=note)


def _sql_analytics(conn, catalogue: pd.DataFrame) -> dict[str, Any]:
    ctr_hour = db.ctr_by_hour(conn, "train")
    ctr_category = db.ctr_by_category(conn, catalogue, "train")
    engaged = db.top_users(conn, "train", min_impressions=5, limit=10)

    best_hour = ctr_hour.loc[ctr_hour["ctr"].idxmax()]
    return {
        "ctr_by_hour": ctr_hour.to_dict(orient="records"),
        "ctr_by_category": ctr_category.head(10).to_dict(orient="records"),
        "top_users": engaged.to_dict(orient="records"),
        "peak_hour": int(best_hour["hour"]),
        "peak_hour_ctr": float(best_hour["ctr"]),
        "total_events": int(ctr_hour["impressions"].sum()),
    }


def _provenance(train: MindData) -> dict[str, Any]:
    """What each family of numbers is actually measured on. Read this before quoting any."""
    return {
        "data_source": train.data_source,
        "measured_on_logged_data": [
            "content_selection",
            "audience",
            "send_time",
            "sql_analytics",
        ],
        "measured_in_simulator": [
            "ab_test_simulated",
            "off_policy_simulated",
            "bandit_simulated",
            "uplift_simulated",
        ],
        "simulator_note": (
            "A/B, off-policy, bandit and uplift results are produced inside a response "
            "oracle fitted on held-out dev impressions, because a logged dataset cannot "
            "reveal what a reader would have done under a policy that was never run. They "
            "are not live-test results."
        ),
        "send_time_note": (
            "Send-time uplift is observational: it compares impressions that happened to "
            "arrive in a reader's predicted best hour against the same population at other "
            "hours. It is not a randomised measurement."
        ),
    }


def _make_run_id(cfg: Config) -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{cfg.hash}"


def _write_metrics(cfg: Config, run_id: str, metrics: dict[str, Any]) -> Path:
    run_dir = cfg.path("paths.runs_dir") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    path = run_dir / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2, default=_json_default), encoding="utf-8")

    log.info("wrote %s", path)
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


def _log_summary(metrics: dict[str, Any]) -> None:
    personalised = metrics["content_selection"]["personalised"]
    baseline = metrics["content_selection"]["popularity_baseline"]
    ab = metrics["ab_test_simulated"]["content_and_send_time"]
    ope = metrics["off_policy_simulated"]

    log.info("=" * 78)
    log.info("run %s  (data_source=%s)", metrics["run_id"], metrics["data_source"])
    log.info("=" * 78)
    log.info(
        "content selection   AUC %.4f  MRR %.4f  nDCG@5 %.4f  nDCG@10 %.4f  (n=%d)",
        personalised["auc"],
        personalised["mrr"],
        personalised["ndcg_at_5"],
        personalised["ndcg_at_10"],
        personalised["n_impressions_evaluated"],
    )
    log.info(
        "popularity baseline AUC %.4f  MRR %.4f  nDCG@5 %.4f  nDCG@10 %.4f",
        baseline["auc"],
        baseline["mrr"],
        baseline["ndcg_at_5"],
        baseline["ndcg_at_10"],
    )
    log.info("audience            ROC-AUC %.4f  precision@%d %.4f",
             metrics["audience"]["roc_auc"],
             metrics["audience"]["audience_k"],
             metrics["audience"]["precision_at_k"])
    log.info("send-time           best-hour CTR %.4f vs %.4f (absolute %+.4f)",
             metrics["send_time"]["best_hour_rate"],
             metrics["send_time"]["baseline_rate"],
             metrics["send_time"]["absolute_uplift"])
    log.info("A/B (simulated)     lift %+.1f%%  p=%.3g  95%% CI [%+.4f, %+.4f]",
             100.0 * ab["relative_lift"], ab["p_value"], ab["ci_low"], ab["ci_high"])
    log.info("off-policy (sim)    DR %.4f vs true %.4f  (ESS %.0f)",
             ope["dr"]["value"], ope["target_true_value"], ope["effective_sample_size"])
    log.info("=" * 78)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Run the NewsPush pipeline end to end.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-batch", action="store_true", help="Skip the campaign scorer")
    args = parser.parse_args(argv)

    run(load_config(args.config), skip_batch=args.skip_batch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
