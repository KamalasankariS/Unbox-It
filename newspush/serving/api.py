"""FastAPI service.

    GET /health      liveness, plus the data source the models were trained on
    GET /recommend   top-k articles for a reader, under the editorial guardrails
    GET /audience    top-k readers for an article
    GET /send-time   a reader's predicted best send hour
    GET /why         why an article was recommended to a reader

`create_app(artifacts)` takes an explicit bundle, which is what the tests use. The
module-level `app` loads the bundle the pipeline wrote, lazily, so importing this
module never triggers training.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query

from newspush.artifacts import Artifacts, artifacts_path
from newspush.config import Config, load_config
from newspush.explain import why
from newspush.models.diversity import build_editorial_send

log = logging.getLogger(__name__)

MAX_RECOMMENDATIONS = 50
MAX_AUDIENCE = 5000

# MMR is quadratic in the candidate count and the tail of a 50k-article catalogue is
# never competitive, so the guardrails run over a shortlist rather than the whole thing.
SHORTLIST_SIZE = 400


class ServingState:
    """Holds the artifact bundle, loading it on first use."""

    def __init__(self, cfg: Config | None = None, artifacts: Artifacts | None = None) -> None:
        self.cfg = cfg or load_config()
        self._artifacts = artifacts

    @property
    def artifacts(self) -> Artifacts:
        if self._artifacts is None:
            try:
                self._artifacts = Artifacts.load(artifacts_path(self.cfg))
            except FileNotFoundError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        return self._artifacts

    @property
    def loaded(self) -> bool:
        return self._artifacts is not None or artifacts_path(self.cfg).is_file()


def create_app(artifacts: Artifacts | None = None, cfg: Config | None = None) -> FastAPI:
    state = ServingState(cfg=cfg, artifacts=artifacts)

    app = FastAPI(
        title="NewsPush",
        description="Email-targeting recommender on the MIND news dataset.",
        version="0.1.0",
    )

    def get_state() -> ServingState:
        return state

    @app.get("/health")
    def health(state: ServingState = Depends(get_state)) -> dict[str, Any]:
        if not state.loaded:
            return {"status": "no_artifacts", "detail": "run the pipeline first: make run"}

        bundle = state.artifacts
        return {
            "status": "ok",
            "run_id": bundle.run_id,
            "data_source": bundle.data_source,
            "encoder": bundle.encoder_name,
            "n_users": len(bundle.profiles),
            "n_articles": len(bundle.news),
        }

    @app.get("/recommend")
    def recommend(
        user_id: str = Query(..., description="Reader to recommend for"),
        k: int = Query(5, ge=1, le=MAX_RECOMMENDATIONS),
        diversify: bool = Query(True, description="Apply MMR and the per-desk cap"),
        state: ServingState = Depends(get_state),
    ) -> dict[str, Any]:
        bundle = state.artifacts

        shortlist = bundle.ranker.recommend_for_user(
            user_id, top_n=SHORTLIST_SIZE, exclude_history=True
        )
        if not shortlist:
            raise HTTPException(status_code=404, detail=f"no candidates available for {user_id!r}")

        relevance = dict(shortlist)

        if diversify:
            selected, guardrails = build_editorial_send(
                cfg=state.cfg,
                candidates=list(relevance),
                relevance=relevance,
                encoder=bundle.encoder,
                news_category=bundle.news_category,
                k=k,
            )
        else:
            selected = [news_id for news_id, _ in shortlist[:k]]
            guardrails = {}

        return {
            "user_id": user_id,
            "cold_start": bundle.profiles.is_cold(user_id),
            "send_hour": bundle.send_time.best_hour(user_id),
            "data_source": bundle.data_source,
            "guardrails": guardrails,
            "recommendations": [
                _article_payload(bundle, news_id, rank, relevance.get(news_id, 0.0))
                for rank, news_id in enumerate(selected, start=1)
            ],
        }

    @app.get("/audience")
    def audience(
        news_id: str = Query(..., description="Article to build an audience for"),
        k: int = Query(100, ge=1, le=MAX_AUDIENCE),
        state: ServingState = Depends(get_state),
    ) -> dict[str, Any]:
        bundle = state.artifacts

        if not bundle.has_article(news_id):
            raise HTTPException(status_code=404, detail=f"unknown news_id: {news_id!r}")

        users = list(bundle.profiles.vectors)
        selected = bundle.propensity.build_audience(
            news_id=news_id,
            k=k,
            candidate_users=users,
            send_hours=bundle.send_time.best_hours(users),
        )
        article = bundle.article(news_id)

        return {
            "news_id": news_id,
            "title": str(article["title"]),
            "category": str(article["category"]),
            "audience_size": len(selected),
            "data_source": bundle.data_source,
            "audience": [
                {
                    "user_id": user_id,
                    "propensity": score,
                    "send_hour": bundle.send_time.best_hour(user_id),
                }
                for user_id, score in selected
            ],
        }

    @app.get("/send-time")
    def send_time(
        user_id: str = Query(..., description="Reader to schedule"),
        state: ServingState = Depends(get_state),
    ) -> dict[str, Any]:
        bundle = state.artifacts
        rates = bundle.send_time.rate_by_hour(user_id)

        return {
            "user_id": user_id,
            "best_hour": bundle.send_time.best_hour(user_id),
            "global_best_hour": bundle.send_time.global_best_hour,
            "personalised": bundle.send_time.is_personalised(user_id),
            "data_source": bundle.data_source,
            "engagement_by_hour": [
                {"hour": hour, "rate": float(rate)} for hour, rate in enumerate(rates)
            ],
        }

    @app.get("/why")
    def explain_recommendation(
        user_id: str = Query(..., description="Reader"),
        news_id: str = Query(..., description="Article that was recommended"),
        state: ServingState = Depends(get_state),
    ) -> dict[str, Any]:
        bundle = state.artifacts

        try:
            explanation = why.explain(
                user_id=user_id,
                news_id=news_id,
                encoder=bundle.encoder,
                profiles=bundle.profiles,
                news=bundle.news,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"unknown news_id: {news_id!r}") from exc

        return {"data_source": bundle.data_source, **explanation.to_dict()}

    return app


def _article_payload(bundle: Artifacts, news_id: str, rank: int, score: float) -> dict[str, Any]:
    article = bundle.article(news_id)
    return {
        "rank": rank,
        "news_id": news_id,
        "title": str(article["title"]),
        "category": str(article["category"]),
        "score": float(score),
    }


app = create_app()
