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
from fastapi.responses import HTMLResponse

from newspush.artifacts import Artifacts, artifacts_path
from newspush.config import Config, load_config
from newspush.explain import why
from newspush.models.diversity import build_editorial_send

log = logging.getLogger(__name__)

MAX_RECOMMENDATIONS = 50
MAX_AUDIENCE = 5000

# Interactive single-page UI served at "/". Self-contained (inline CSS/JS, no external
# assets, no build step) and calls the same-origin API endpoints. Not an f-string: the
# CSS/JS braces are literal.
LANDING_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Unbox It: The Inbox Recommender</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #ffffff; --fg: #17181c; --muted: #6b7280; --card: #f6f7f9; --border: #e2e5ea;
    --accent: #2563eb; --bar-bg: #e2e5ea; --chip: #e8eefc; --chip-fg: #1e40af;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0f1115; --fg: #e6e8eb; --muted: #9aa2ad; --card: #1a1d23; --border: #2a2e37;
      --accent: #5b9bff; --bar-bg: #2a2e37; --chip: #1e2a44; --chip-fg: #9dc0ff;
    }
  }
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--fg); margin: 0; line-height: 1.55;
  }
  .wrap { max-width: 720px; margin: 0 auto; padding: 3rem 1.25rem 4rem; }
  h1 { font-size: 1.8rem; margin: 0 0 0.2rem; }
  .tagline { color: var(--muted); margin: 0 0 1.5rem; }
  .controls { display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; }
  label { font-weight: 600; }
  input {
    font: inherit; padding: 0.5rem 0.6rem; border: 1px solid var(--border); border-radius: 8px;
    background: var(--bg); color: var(--fg); width: 8rem;
  }
  button {
    font: inherit; font-weight: 600; padding: 0.5rem 0.9rem; border-radius: 8px; cursor: pointer;
    border: 1px solid var(--accent); background: var(--accent); color: #fff;
  }
  button.secondary { background: transparent; color: var(--accent); }
  .hint { color: var(--muted); font-size: 0.85rem; margin: 0.4rem 0 1.25rem; }
  .meta { color: var(--muted); font-size: 0.9rem; margin: 0.5rem 0 1rem; min-height: 1.2rem; }
  .card {
    border: 1px solid var(--border); background: var(--card); border-radius: 12px;
    padding: 0.9rem 1rem; margin-bottom: 0.75rem;
  }
  .card-top { display: flex; align-items: baseline; gap: 0.6rem; }
  .rank { font-weight: 700; color: var(--muted); }
  .title { font-weight: 600; flex: 1; }
  .chip {
    font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.03em;
    background: var(--chip); color: var(--chip-fg); padding: 0.15rem 0.5rem; border-radius: 999px;
  }
  .bar { height: 6px; background: var(--bar-bg); border-radius: 999px; margin: 0.6rem 0 0.3rem; overflow: hidden; }
  .bar > span { display: block; height: 100%; background: var(--accent); }
  .score { color: var(--muted); font-size: 0.8rem; }
  .why-btn {
    margin-top: 0.6rem; background: transparent; color: var(--accent); border-color: var(--border);
    padding: 0.3rem 0.7rem; font-size: 0.85rem;
  }
  .why {
    margin-top: 0.6rem; font-size: 0.9rem; border-left: 3px solid var(--accent);
    padding-left: 0.75rem; display: none;
  }
  .why.show { display: block; }
  .status { color: var(--muted); margin: 1rem 0; }
  footer { margin-top: 2rem; color: var(--muted); font-size: 0.85rem; border-top: 1px solid var(--border); padding-top: 1rem; }
  footer a { color: var(--accent); }
</style>
</head>
<body>
<div class="wrap">
  <h1>Unbox It</h1>
  <p class="tagline">Right Story, Right Reader, Right Time: an email-targeting recommender on the MIND dataset.</p>

  <div class="controls">
    <label for="uid">Reader</label>
    <input id="uid" value="U000001" spellcheck="false" autocomplete="off">
    <button id="go">Recommend</button>
    <button id="rand" class="secondary">Random reader</button>
  </div>
  <p class="hint">Any reader from U000000 to U000799. Each has a different reading history, so the picks change.</p>

  <div class="meta" id="meta"></div>
  <div id="results"></div>

  <footer>
    Running the simulated MIND-format sample, so it uses no real user data.
    <a href="/docs">API docs</a> &middot; <a href="https://github.com/KamalasankariS/Unbox-It">Source on GitHub</a>
  </footer>
</div>

<script>
const byId = (id) => document.getElementById(id);
const results = byId('results');
const meta = byId('meta');

async function recommend() {
  const uid = byId('uid').value.trim();
  if (!uid) return;
  meta.textContent = '';
  results.innerHTML = '<p class="status">Loading. If the free server was asleep, the first request can take up to a minute.</p>';
  try {
    const res = await fetch('/recommend?user_id=' + encodeURIComponent(uid) + '&k=5');
    if (!res.ok) {
      results.innerHTML = '<p class="status">No recommendations for "' + uid + '". Try a reader between U000000 and U000799.</p>';
      return;
    }
    const data = await res.json();
    renderMeta(data);
    renderCards(uid, data.recommendations || []);
  } catch (err) {
    results.innerHTML = '<p class="status">Could not reach the API. Try again in a moment.</p>';
  }
}

function renderMeta(data) {
  const known = data.cold_start ? 'new reader, no history yet' : 'known reader';
  const hour = String(data.send_hour).padStart(2, '0');
  meta.textContent = 'Reader ' + data.user_id + ' : ' + known + ' : best send hour ' + hour + ':00';
}

function renderCards(uid, recs) {
  results.innerHTML = '';
  if (!recs.length) {
    results.innerHTML = '<p class="status">No articles to show for this reader.</p>';
    return;
  }
  for (const r of recs) {
    const card = document.createElement('div');
    card.className = 'card';

    const top = document.createElement('div');
    top.className = 'card-top';
    const rank = document.createElement('span'); rank.className = 'rank'; rank.textContent = '#' + r.rank;
    const title = document.createElement('span'); title.className = 'title'; title.textContent = r.title;
    const chip = document.createElement('span'); chip.className = 'chip'; chip.textContent = r.category;
    top.append(rank, title, chip);

    const bar = document.createElement('div'); bar.className = 'bar';
    const fill = document.createElement('span');
    fill.style.width = (Math.max(0, Math.min(1, r.score)) * 100).toFixed(0) + '%';
    bar.append(fill);

    const score = document.createElement('div');
    score.className = 'score';
    score.textContent = 'match score ' + r.score.toFixed(3);

    const whyBtn = document.createElement('button');
    whyBtn.className = 'why-btn';
    whyBtn.textContent = 'Why this article?';
    const whyPanel = document.createElement('div');
    whyPanel.className = 'why';
    whyBtn.addEventListener('click', () => explain(uid, r.news_id, whyPanel));

    card.append(top, bar, score, whyBtn, whyPanel);
    results.append(card);
  }
}

async function explain(uid, newsId, panel) {
  if (panel.classList.contains('show')) { panel.classList.remove('show'); return; }
  panel.textContent = 'Loading explanation...';
  panel.classList.add('show');
  try {
    const res = await fetch('/why?user_id=' + encodeURIComponent(uid) + '&news_id=' + encodeURIComponent(newsId));
    panel.textContent = res.ok ? (await res.json()).summary : 'No explanation available.';
  } catch (err) {
    panel.textContent = 'Could not load the explanation.';
  }
}

byId('go').addEventListener('click', recommend);
byId('rand').addEventListener('click', () => {
  byId('uid').value = 'U' + String(Math.floor(Math.random() * 800)).padStart(6, '0');
  recommend();
});
byId('uid').addEventListener('keydown', (e) => { if (e.key === 'Enter') recommend(); });

recommend();
</script>
</body>
</html>"""

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
        title="Unbox It: The Inbox Recommender",
        description="Email-targeting recommender on the MIND news dataset.",
        version="0.1.0",
    )

    def get_state() -> ServingState:
        return state

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def root() -> str:
        return LANDING_PAGE

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
