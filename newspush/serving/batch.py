"""Batch scorer: build the campaign send plan and write it to SQL.

The `campaign_recommendations` table is the data product. One row per (reader, article,
rank) with the hour to send it, produced by running the full decision stack in the order
a real campaign would:

    1. content selection      rank the catalogue for each reader
    2. editorial guardrails   MMR and per-desk caps over the shortlist
    3. propensity scoring     score each surviving send, so the plan is calibrated
    4. send-time              attach each reader's best hour
    5. fatigue capping        drop the sends that would over-mail someone

Capping runs last on purpose: it needs the calibrated scores from step 3 to know which
of a reader's sends are worth keeping.
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from newspush.artifacts import Artifacts, artifacts_path
from newspush.config import Config, load_config
from newspush.data import db
from newspush.models.diversity import build_editorial_send
from newspush.models.fatigue import FatigueMetrics

log = logging.getLogger(__name__)

SHORTLIST_SIZE = 200


def build_send_plan(cfg: Config, artifacts: Artifacts, max_users: int | None = None) -> pd.DataFrame:
    """Score every reader and return the uncapped send plan."""
    top_n = int(cfg.require("serving.batch_top_n"))
    limit = max_users if max_users is not None else int(cfg.require("serving.batch_max_users"))

    users = list(artifacts.profiles.vectors)[:limit]
    if not users:
        raise ValueError("no user profiles available to score")

    rows: list[dict] = []

    for user_id in users:
        shortlist = artifacts.ranker.recommend_for_user(
            user_id, top_n=SHORTLIST_SIZE, exclude_history=True
        )
        if not shortlist:
            continue

        relevance = dict(shortlist)
        selected, _ = build_editorial_send(
            cfg=cfg,
            candidates=list(relevance),
            relevance=relevance,
            encoder=artifacts.encoder,
            news_category=artifacts.news_category,
            k=top_n,
        )
        if not selected:
            continue

        send_hour = artifacts.send_time.best_hour(user_id)
        propensities = artifacts.propensity.score(
            [user_id] * len(selected), selected, [send_hour] * len(selected)
        )

        for rank, (news_id, score) in enumerate(zip(selected, propensities), start=1):
            rows.append(
                {
                    "user_id": user_id,
                    "news_id": news_id,
                    "rank": rank,
                    "score": float(score),
                    "send_hour": send_hour,
                }
            )

    plan = pd.DataFrame(rows)
    log.info("send plan: %d sends across %d readers", len(plan), plan["user_id"].nunique())
    return plan


def run(
    cfg: Config,
    artifacts: Artifacts,
    max_users: int | None = None,
    write: bool = True,
) -> tuple[pd.DataFrame, FatigueMetrics]:
    """Build the plan, apply the fatigue cap, and persist the result."""
    plan = build_send_plan(cfg, artifacts, max_users=max_users)
    capped, fatigue_metrics = artifacts.fatigue.apply_cap(plan)

    # Ranks are re-numbered after capping, so the stored plan has no gaps.
    capped["rank"] = capped.groupby("user_id").cumcount() + 1

    if write:
        conn = db.connect(cfg.path("paths.db_path"))
        try:
            written = db.write_campaign_recommendations(conn, capped, artifacts.run_id)
            log.info(
                "wrote %d rows to campaign_recommendations (run_id=%s)", written, artifacts.run_id
            )
        finally:
            conn.close()

    return capped, fatigue_metrics


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Score all readers and write the campaign table.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-users", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    try:
        artifacts = Artifacts.load(artifacts_path(cfg))
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1

    capped, metrics = run(cfg, artifacts, max_users=args.max_users)

    log.info(
        "campaign ready: %d sends after capping (%.1f%% suppressed), data_source=%s",
        len(capped),
        100.0 * metrics.suppression_rate,
        artifacts.data_source,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
