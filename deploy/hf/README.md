---
title: Unbox It
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# Unbox It: The Inbox Recommender

An email-targeting recommender on the MIND news dataset: content selection, audience
creation, send-time optimisation, and A/B plus off-policy evaluation, served behind
FastAPI.

This Space runs the API on the simulated MIND-format sample (confirmed on `/health`), so
it boots without a dataset download. The full project, the real-MIND results, and the
source live on GitHub:

https://github.com/KamalasankariS/Unbox-It

Endpoints: open `/docs` for the interactive API, or call `/health`, `/recommend`,
`/audience`, `/send-time`, and `/why` directly.
