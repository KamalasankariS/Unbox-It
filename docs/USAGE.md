# User manual

Everything you need to run Unbox It locally, use its API, and read its results. If you
just want to try the deployed version, open the [live demo](https://unbox-it.onrender.com/docs)
instead.

## Requirements

- Python 3.10, 3.11, or 3.12
- About 2 GB of free RAM for the sample; ~4 GB and a few minutes for the full real-MIND run
- No GPU, no network at run time (the dataset download is a separate, optional step)

## Install and run in 3 steps

```bash
git clone https://github.com/KamalasankariS/Unbox-It.git
cd Unbox-It
pip install -r requirements.txt
```

Then run the whole thing:

```bash
python -m newspush.pipeline
```

That is the entire setup. The pipeline trains every model, evaluates them, and writes
`runs/<run_id>/metrics.json`. With no dataset present it uses a built-in simulated
sample, so this works on a fresh clone with nothing else installed.

To serve the API afterwards:

```bash
python -m uvicorn newspush.serving.api:app --port 8000
```

Open http://localhost:8000/docs.

## The Makefile shortcuts

| Command | What it does |
|---|---|
| `make setup` | Install dependencies |
| `make data` | Download the real MIND-small dataset (optional; see below) |
| `make run` | Run the end-to-end pipeline |
| `make test` | Run the full test suite (offline, deterministic) |
| `make api` | Serve the FastAPI app on port 8000 |
| `make batch` | Score all readers and write the campaign_recommendations table |

## Using the API

Five read-only endpoints. Examples assume the app is on `localhost:8000`.

| Endpoint | Purpose | Example |
|---|---|---|
| `GET /health` | Liveness and the data source in use | `/health` |
| `GET /recommend` | Top-k articles for a reader | `/recommend?user_id=U000001&k=5` |
| `GET /audience` | Top-k readers for an article | `/audience?news_id=N000027&k=10` |
| `GET /send-time` | A reader's predicted best send hour | `/send-time?user_id=U000001` |
| `GET /why` | Why an article was recommended to a reader | `/why?user_id=U000001&news_id=N000027` |

```bash
curl "localhost:8000/recommend?user_id=U000001&k=5"
curl "localhost:8000/why?user_id=U000001&news_id=N000027"
```

On the simulated sample, reader IDs run `U000000` to `U000799` and article IDs run
`N000000` to `N000599`. On real MIND, use the IDs from the dataset (for example `U80234`,
`N55528`).

## Configuration

Everything that affects a result lives in `config.yaml`, and a hash of that file is
recorded in every `metrics.json`, so a run is reproducible from (code commit, config).
The knobs you are most likely to touch:

| Key | Meaning |
|---|---|
| `seed` | The single random seed for the whole pipeline |
| `encoder.kind` | `tfidf-svd` (default) or `sentence-transformer` (needs the optional dependency) |
| `encoder.dim` | Article embedding size |
| `content_selection.max_eval_impressions` | Cap on dev impressions scored, for runtime |
| `audience.max_train_rows` | Cap on training rows for the propensity model |
| `bandit.n_rounds` | Bandit horizon |
| `serving.batch_max_users` | How many readers the batch scorer writes |

Point the pipeline at a different config with `--config`:

```bash
python -m newspush.pipeline --config my_config.yaml
```

## Real MIND vs the sample

By default, if the dataset is not on disk the pipeline uses a clearly-labelled simulated
sample so it always runs. To use the real data:

```bash
make data
```

This downloads MIND-small into `data/MINDsmall_train` and `data/MINDsmall_dev`. The
pipeline then uses it automatically, with no code change. If the download fails (the
original Microsoft URLs are dead; `make data` uses a mirror), set `MIND_TRAIN_URL` and
`MIND_DEV_URL`, or drop the extracted `news.tsv` and `behaviors.tsv` files into those two
folders by hand.

Every run records `data_source` as `real-MIND` or `mind-format-sample` in its
`metrics.json`, so a number is never mistaken for the wrong data source.

## Reading the results

`runs/<run_id>/metrics.json` contains everything, including a `provenance` block that
lists which result families are measured on logged data and which are simulated. Print
the headline numbers with:

```bash
python scripts/readme_numbers.py
```

Content selection, audience, and send-time are measured on the logged MIND data. A/B
testing, off-policy evaluation, the bandit, and uplift are measured inside a response
simulator (a model fitted on held-out data), because a logged dataset cannot reveal what
a reader would have done under a policy that was never run. The README section
"What is real and what is simulated" explains this in full.

## Deploying it yourself

The repo ships a `Dockerfile` that trains the sample models at build time and serves the
API, so any container host works:

```bash
docker build -t unbox-it .
docker run -p 8000:8000 unbox-it
```

`render.yaml` deploys it to Render as a Blueprint. The GitHub Actions workflows publish a
container image to the GitHub Container Registry on every green build and, if you add the
relevant secret, mirror the repo to a Hugging Face Space.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `/health` returns `no_artifacts` | Run `python -m newspush.pipeline` first; serving loads what the pipeline wrote |
| Pipeline says `mind-format-sample` but you wanted real data | Run `make data`, or check the TSVs are under `data/MINDsmall_train` and `data/MINDsmall_dev` |
| `sentence-transformers` import error | It is optional; leave `encoder.kind: tfidf-svd`, or `pip install sentence-transformers` |
| Tests are slow | They train small models; a full run is under a minute on a laptop |

## Tests

```bash
make test
```

The suite is offline and deterministic (it uses the sample, never the network), and it is
the same suite CI runs on Python 3.10, 3.11, and 3.12.
