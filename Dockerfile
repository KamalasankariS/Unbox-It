# Self-contained image serving the Unbox It API.
#
# Serving artifacts are trained at build time from the deterministic MIND-format sample,
# so the image needs no dataset download and starts fast. The live API reports
# data_source="mind-format-sample" on /health, matching exactly what it serves.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Train encoder, profiles, propensity, send-time, and fatigue models into runs/artifacts.pkl.
# --skip-batch drops the campaign table the live endpoints do not need.
RUN python -m newspush.pipeline --skip-batch

EXPOSE 8000

# Render and most PaaS hosts inject $PORT; default to 8000 for local `docker run`.
CMD ["sh", "-c", "uvicorn newspush.serving.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
