# Unbox It: The Inbox Recommender

*Right Story, Right Reader, Right Time: an email-targeting recommender on the MIND dataset.*

[![CI](https://github.com/KamalasankariS/Unbox-It/actions/workflows/ci.yml/badge.svg)](https://github.com/KamalasankariS/Unbox-It/actions/workflows/ci.yml)
[![CD](https://github.com/KamalasankariS/Unbox-It/actions/workflows/cd.yml/badge.svg)](https://github.com/KamalasankariS/Unbox-It/actions/workflows/cd.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-185%20passing-brightgreen.svg)](tests)
[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen.svg)](https://unbox-it.onrender.com/docs)

**Live demo:** [unbox-it.onrender.com/docs](https://unbox-it.onrender.com/docs) (interactive API, running the simulated sample). First load may take ~50s while the free instance wakes.

[![Unbox It demo](docs/demo.gif)](https://unbox-it.onrender.com/docs)

Unbox It decides **what** article to send **which** reader, **when**, and measures the
impact with **A/B testing and off-policy evaluation**. It is trained and evaluated on
the real, public [MIND](https://msnews.github.io/) news dataset (Microsoft News), and
served behind a FastAPI application and a batch pipeline.

It is a research/portfolio project on a public dataset, not production traffic. Every
reported number records the data source it came from, and results produced inside the
response simulator are labelled as simulated rather than presented as live measurements.

> Built content selection, audience creation, and send-time optimisation on the MIND
> news dataset, reporting AUC/MRR/nDCG and open-rate uplift; validated impact with A/B
> testing and off-policy evaluation (IPS/SNIPS/Doubly-Robust); served recommendations
> via FastAPI and a batch pipeline.

## Try it in 3 steps

No dataset, config, or GPU needed. It runs on a built-in simulated sample, so a fresh
clone works immediately.

```bash
git clone https://github.com/KamalasankariS/Unbox-It.git && cd Unbox-It
pip install -r requirements.txt
python -m newspush.pipeline        # trains, evaluates, writes runs/<id>/metrics.json
```

Then serve the API and open http://localhost:8000/docs:

```bash
python -m uvicorn newspush.serving.api:app --port 8000
```

Prefer not to install anything? Use the [live demo](https://unbox-it.onrender.com/docs).
For the complete guide (every command, endpoint, config option, real-MIND setup, and
troubleshooting) see the **[user manual](docs/USAGE.md)**.

The pipeline runs with or without the dataset: if MIND is not present it falls back to a
clearly-labelled simulated sample. `make test` runs the full suite offline. The `make`
shortcuts (`setup`, `data`, `run`, `test`, `api`, `batch`) are documented in the manual.

## What problem this solves

"Grow email engagement without burning subscribers" is a business goal, not an ML
problem. Unbox It turns it into concrete, measurable objectives:

| Business question | ML formulation | Metric |
|---|---|---|
| Which story leads a reader's email? | rank candidate articles by predicted relevance | AUC, MRR, nDCG@5/10 |
| Who should receive a given story? | model P(click \| reader, article, hour), rank readers | ROC-AUC, precision@k |
| When should we send it? | estimate each reader's engagement by hour | open-rate uplift |
| Is the new policy actually better? | A/B test + off-policy evaluation | lift, p-value, CI, IPS/SNIPS/DR |
| Are we over-mailing anyone? | frequency capping + unsubscribe-risk model | sends suppressed, engagement retained |

## How it maps to the recommender-systems skill set

| Requirement | Where it lives |
|---|---|
| ML applied to email (timing, content, audience) | `models/send_time.py`, `models/content_selection.py`, `models/audience.py` |
| Recommendation with NLP / LLM embeddings | `features/text.py`: TF-IDF+SVD baseline, swappable for a sentence-transformer |
| Ambiguous business question to ML problem | this README's problem framing; concrete objectives and metrics |
| Data products behind APIs / batch | `serving/api.py` (FastAPI), `serving/batch.py` writing the `campaign_recommendations` table |
| A/B testing and experimentation | `experiment/ab_test.py`, plus off-policy evaluation in `experiment/off_policy.py` |
| Robustness and reproducibility | single seed, config hash in every `metrics.json`, deterministic pipeline, full test suite |
| SQL over large datasets | `data/db.py`: SQLite event store (~5M rows) with SQL analytics |
| Communicating complex ML | the `/why` endpoint and `explain/why.py`, per-recommendation explanations |
| Respecting editorial judgement | `models/diversity.py`: MMR diversity, per-desk caps, editor pins |

## Architecture

```
                MIND TSVs  ──or──  simulated sample
                          │
                    data/acquire.py            single MindData schema either way
                          │
        ┌─────────────────┼──────────────────────────────┐
        │                 │                               │
  features/text.py   features/users.py               data/db.py
  article encoder    reader profiles                 SQLite events + SQL analytics
        │                 │                               │
        └────────┬────────┘                               │
                 │                                        │
     ┌───────────┼────────────┬───────────────┐          │
     │           │            │               │          │
 content_    audience.py   send_time.py    diversity.py  │
 selection    propensity   best send hour  MMR + guardrails
     │           │            │               │          │
     └───────────┴─────┬──────┴───────────────┘          │
                       │                                  │
              experiment/simulator.py  ◄──────────────────┘
              response oracle (held-out dev)
                       │
     ┌─────────────────┼─────────────────┬────────────────┐
     │                 │                 │                │
  ab_test.py      off_policy.py      bandit.py        uplift.py
  z-test + sim    IPS/SNIPS/DR       Thompson smp.    T-learner / CATE
     │                 │                 │                │
     └─────────────────┴────────┬────────┴────────────────┘
                                │
                          pipeline.py
              runs/<run_id>/metrics.json  +  artifacts.pkl
                                │
                    ┌───────────┴───────────┐
                serving/api.py        serving/batch.py
                FastAPI endpoints     campaign_recommendations
```

## Results

All figures below come from a single `make run` on **real MIND-small** (`data_source:
real-MIND`, `seed: 42`), regenerable with `python scripts/readme_numbers.py`. The run
covers 156,965 train and 73,152 dev impressions, 65,238 articles, and 92,827 reader
profiles; the SQL store holds 8.58M events.

### Content selection (logged dev)

Personalised cosine ranking against a popularity baseline, on the standard MIND metrics.

| Policy | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| Personalised (this system) | **0.5616** | 0.2970 | 0.2737 | 0.3370 |
| Popularity baseline | 0.5505 | 0.3078 | 0.2846 | 0.3375 |

Personalisation beats popularity on AUC; popularity is fractionally ahead on the rank
metrics. This is the honest shape of content-only ranking on MIND, a real but modest
signal, and it is why the audience model below combines content with context rather than
relying on cosine alone.

### Audience and send-time (logged dev)

| Metric | Value |
|---|---|
| Audience propensity ROC-AUC | **0.603** |
| Audience precision@500 vs base click rate | **0.200 vs 0.041 (4.9x lift)** |
| Send-time open-rate uplift (best hour vs others) | **+2.7%** (0.0416 vs 0.0405) |
| Readers with a personalised send hour | 46,920 |
| Peak engagement hour (SQL analytics) | 08:00 |

The propensity model (cosine plus history length, topic entropy, hour, desk, and article
popularity) reaches ROC-AUC 0.603, and the 500 readers it ranks highest for a campaign
click at 4.9x the population base rate. Send-time uplift is **observational**: MIND
impressions cannot be re-sent at a different hour, so it compares impressions that landed
in a reader's predicted best hour against the same population at other hours.

### A/B, off-policy, bandit, uplift (simulated)

Measured inside the response simulator (see [below](#what-is-real-and-what-is-simulated)),
not on live traffic.

| Experiment | Result |
|---|---|
| **Off-policy: DR vs true policy value** | **DR 0.755, SNIPS 0.725 vs true 0.737**; IPS 0.300 (ESS 138) |
| Off-policy: target vs logged policy | 0.737 vs 0.724, personalised policy edges the logger |
| A/B: personalised content vs popularity | -1.9%, p=0.024, 95% CI [-0.030, -0.002] |
| Bandit: regret at 12k rounds (TS / e-greedy / random) | 634 / 555 / 2378 |
| Uplift: AUUC / Qini | 1.59 / 952 (top-30% uplift 0.999 vs bottom-30% -0.362) |

**Off-policy evaluation is the centerpiece, and it works exactly as the theory predicts.**
Doubly-robust and self-normalised IPS recover the target policy's true value to within 0.02
and 0.01, while plain IPS is off by 0.44, the textbook consequence of an effective sample
size of 138 out of 5,000 logged events. This is the whole argument for the method: it
measures a small policy difference from logged data that a naive estimator gets badly
wrong.

**Two results are honestly negative, and reported rather than hidden.** The simulated A/B
shows personalised *content selection* slightly *below* a pure-popularity control (-1.9%),
and the contextual bandit does not beat a static best-desk policy. Both reflect a genuine
property of MIND: popularity is a famously strong CTR baseline, and TF-IDF profiles carry
weak per-reader desk signal. The A/B control also directly optimises the simulator oracle's
popularity feature, which makes it a deliberately hard baseline. The off-policy comparison,
where the personalised policy does edge the logged one (0.737 vs 0.724), is the more
reliable read, because its estimators are built to measure exactly these small differences.
Chasing a positive A/B by re-tuning the experiment would violate the honesty rules this
project is built on.

## Data

Primary data is MIND-small. The dataset's original Microsoft blob URLs now return HTTP
409 (`PublicAccessNotPermitted`): public access was disabled on the storage account, so
they are dead for everyone, not only for restricted networks. `make data` therefore pulls
from the HuggingFace re-host maintained by the recommenders-team project, which is
byte-faithful to the official release (MIND-small dev is 42,416 articles and 73,152
impressions, the published statistics). Override the source with `MIND_TRAIN_URL` /
`MIND_DEV_URL`, or drop the extracted TSVs into `data/MINDsmall_train/` and
`data/MINDsmall_dev/` by hand.

If the dataset is absent, the pipeline generates a MIND-format sample with real learnable
structure (a latent per-reader topic preference carried by the article text, and a
per-reader preferred open hour). It is labelled `data_source="mind-format-sample"`
everywhere it appears. The real loader and the sample generator return the identical
in-memory schema, so the entire pipeline runs unchanged on either.

## What is real and what is simulated

This is the project's central honesty distinction.

**Measured on logged MIND data:** content selection, audience, send-time, and the SQL
analytics. These are computed directly from recorded impressions and clicks.

**Measured inside a response simulator:** A/B testing, off-policy evaluation, the
bandit, and uplift. A logged dataset records only what readers did when shown what they
were actually shown; it cannot reveal what they would have done under a policy that was
never run. So a response model is fitted on held-out dev and used as an oracle
P(click | reader, article, hour) for counterfactual queries. The simulator is calibrated
on real MIND clicks, so the ordering it induces over policies is informative, but its
absolute numbers inherit its own model error and are **not** live-test results.

Two guards keep the simulator honest:
- the oracle is fitted on **dev**, while the policies competing inside it are trained on
  **train**, so if a policy were the oracle itself, its winning would be a tautology;
- off-policy evaluation reports the effective sample size and validates its estimators
  against a ground-truth value the simulator can compute but never shows them.

Uplift additionally requires a control ("not emailed") arm, which MIND does not contain;
that arm is constructed from a stated organic-engagement model whose parameters live in
`config.yaml`. The fatigue model's unsubscribe risk is likewise an explicit parametric
assumption, not a fit to unsubscribe labels, and MIND has none. Both are labelled as
assumptions wherever they surface.

`metrics.json` carries a `provenance` block that lists exactly which result families are
logged and which are simulated.

## Extra-mile modules

Beyond the core content/audience/send-time/A-B stack:

- **Off-policy evaluation** (`experiment/off_policy.py`): IPS, self-normalised IPS, and
  doubly-robust estimators, validated against the simulator's ground truth.
- **Contextual bandit** (`models/bandit.py`): linear Thompson sampling over desks,
  benchmarked against epsilon-greedy, static-greedy, and random on cumulative regret.
- **Uplift / CATE** (`models/uplift.py`): a T-learner that targets persuadable readers
  rather than those who would engage anyway, scored with AUUC and Qini.
- **Fatigue-aware capping** (`models/fatigue.py`): an unsubscribe-risk signal and a hard
  frequency cap, reporting the engagement/volume trade-off.
- **Editorial guardrails** (`models/diversity.py`): MMR diversity, per-desk caps, and
  editor pins/boosts, so editorial judgement can outrank the model.
- **Explainability** (`explain/why.py`): per-recommendation explanations naming the
  shared themes, the reader's desks, and their closest prior reads.

## API

```
GET /health                       liveness + the data source the models were trained on
GET /recommend?user_id=&k=        top-k articles for a reader, under the guardrails
GET /audience?news_id=&k=         top-k readers for an article
GET /send-time?user_id=           a reader's predicted best send hour + hourly curve
GET /why?user_id=&news_id=        why an article was recommended to a reader
```

```bash
make api
curl "localhost:8000/recommend?user_id=U13740&k=5"
curl "localhost:8000/why?user_id=U13740&news_id=N55528"
```

## Continuous integration and deployment

**CI** (`.github/workflows/ci.yml`): every push and pull request runs the full pytest
suite on Python 3.10, 3.11, and 3.12. The suite is offline and deterministic, using the
simulated MIND-format sample, so it needs no dataset download and no network.

**CD** (`.github/workflows/cd.yml`): once CI passes on `main`, a Docker image is built and
published to the GitHub Container Registry at `ghcr.io/kamalasankaris/unbox-it:latest`.
The image trains its serving models from the sample at build time, so it boots without a
dataset and reports `data_source: "mind-format-sample"` on `/health`.

Run the published image locally:

```bash
docker run -p 8000:8000 ghcr.io/kamalasankaris/unbox-it:latest
curl "localhost:8000/recommend?user_id=U000001&k=5"
```

Deploy to a live URL two ways:

- **Render** (recommended): connect this repo as a Blueprint at render.com; `render.yaml`
  makes Render build the Dockerfile and redeploy on every push. Optionally add a
  `RENDER_DEPLOY_HOOK_URL` repository secret to have the CD workflow ping Render after each
  successful build.
- **Hugging Face Spaces**: `.github/workflows/deploy-hf.yml` mirrors the repo to a Docker
  Space on every green build, giving a second public demo at
  `huggingface.co/spaces/Kamalasankari/Unbox-It`. It uses the Space config in
  `deploy/hf/README.md` and stays dormant until you add an `HF_TOKEN` repository secret.
- **Any container host** (Fly.io, Cloud Run): pull the GHCR image above, or build the
  Dockerfile directly.

## Reproducibility

- One seed in `config.yaml`; no hidden randomness. Reruns are bit-for-bit identical (a
  test asserts this).
- Every run writes `runs/<run_id>/metrics.json` with the run id, timestamp, data source,
  config hash, encoder, and all metrics.
- Every module has pytest coverage; `make test` is CI-friendly and offline.
- The optional `sentence-transformers` encoder is import-guarded, so the base project
  runs on scikit-learn alone.

## Project layout

```
newspush/
  config.py                 config loading + hash
  data/
    schema.py               MindData: the shared in-memory schema
    mind_loader.py          real MIND TSV loader
    make_sample.py          MIND-format sample generator (fallback)
    acquire.py              real if present, else sample
    download.py             dataset fetch (the only networked code)
    db.py                   SQLite event store + SQL analytics
  features/
    text.py                 article encoder (TF-IDF+SVD; pluggable ST/LLM)
    users.py                reader profiles from click history
  models/
    content_selection.py    ranker + MIND metrics
    audience.py             propensity model + audience selection
    send_time.py            per-reader best send hour
    diversity.py            MMR + editorial guardrails
    fatigue.py              frequency capping + unsubscribe risk
    bandit.py               Thompson-sampling contextual bandit
    uplift.py               T-learner / CATE
  experiment/
    simulator.py            the response oracle
    ab_test.py              z-test + simulation harness
    off_policy.py           IPS / SNIPS / DR
  explain/why.py            per-recommendation explanations
  serving/
    api.py                  FastAPI service
    batch.py                campaign scorer
  artifacts.py              trained-state bundle shared by pipeline and serving
  pipeline.py               end-to-end run
tests/                      pytest for every module
scripts/readme_numbers.py   print README figures from a run
```

## Feedback

If you ran this, tried the demo, or found it through a resume or portfolio, I would
genuinely like to hear how it went, positive or critical. Open a
[Feedback issue](https://github.com/KamalasankariS/Unbox-It/issues/new?template=feedback.yml)
(no field is required) or start a Discussion. Bug reports and questions are welcome too.

If it was useful, a star helps other people find it.

## License

MIT (see `LICENSE`). MIND is used under Microsoft's dataset terms and is not
redistributed here.
