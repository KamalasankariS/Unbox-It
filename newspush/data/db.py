"""SQLite event store and SQL analytics over the impression log.

One impression with K candidates becomes K event rows, which is the grain both the
send-time model and the campaign analytics need. MIND-small produces roughly 5M rows.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

from newspush.data.schema import MindData

log = logging.getLogger(__name__)

INSERT_BATCH_SIZE = 50_000

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id      INTEGER PRIMARY KEY,
    impression_id TEXT    NOT NULL,
    user_id       TEXT    NOT NULL,
    news_id       TEXT    NOT NULL,
    clicked       INTEGER NOT NULL CHECK (clicked IN (0, 1)),
    hour          INTEGER NOT NULL CHECK (hour BETWEEN 0 AND 23),
    split         TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS campaign_recommendations (
    user_id    TEXT    NOT NULL,
    news_id    TEXT    NOT NULL,
    rank       INTEGER NOT NULL,
    score      REAL    NOT NULL,
    send_hour  INTEGER NOT NULL,
    run_id     TEXT    NOT NULL,
    PRIMARY KEY (run_id, user_id, rank)
);
CREATE INDEX IF NOT EXISTS idx_recommendations_user ON campaign_recommendations(user_id);
"""

# Built after the bulk load, not before: maintaining four indexes per row would slow the
# ~9M-row insert by an order of magnitude.
EVENT_INDEXES = {
    "idx_events_user": "user_id",
    "idx_events_hour": "hour",
    "idx_events_news": "news_id",
    "idx_events_split": "split",
}

CTR_BY_HOUR_SQL = """
SELECT hour,
       COUNT(*)                              AS impressions,
       SUM(clicked)                          AS clicks,
       CAST(SUM(clicked) AS REAL) / COUNT(*) AS ctr
FROM events
WHERE split = ?
GROUP BY hour
ORDER BY hour
"""

TOP_USERS_SQL = """
SELECT user_id,
       COUNT(*)                              AS impressions,
       SUM(clicked)                          AS clicks,
       CAST(SUM(clicked) AS REAL) / COUNT(*) AS ctr
FROM events
WHERE split = ?
GROUP BY user_id
HAVING COUNT(*) >= ?
ORDER BY clicks DESC, ctr DESC
LIMIT ?
"""

# Aggregate per article first (uses idx_events_news, yields ~one row per article),
# then join that small result to the catalogue. Joining the catalogue directly against
# the full ~9M-row events table makes SQLite scan the catalogue once per event, which
# is catastrophically slow; this keeps the join tiny.
CTR_BY_CATEGORY_SQL = """
WITH per_news AS (
    SELECT news_id,
           COUNT(*)     AS impressions,
           SUM(clicked) AS clicks
    FROM events
    WHERE split = ?
    GROUP BY news_id
)
SELECT n.category,
       SUM(p.impressions)                              AS impressions,
       SUM(p.clicks)                                   AS clicks,
       CAST(SUM(p.clicks) AS REAL) / SUM(p.impressions) AS ctr
FROM per_news p
JOIN news n ON n.news_id = p.news_id
GROUP BY n.category
ORDER BY impressions DESC
"""

USER_HOUR_COUNTS_SQL = """
SELECT user_id,
       hour,
       COUNT(*)     AS impressions,
       SUM(clicked) AS clicks
FROM events
WHERE split = ?
GROUP BY user_id, hour
"""

USER_STATS_SQL = """
SELECT user_id,
       COUNT(*)     AS impressions,
       SUM(clicked) AS clicks
FROM events
WHERE split = ?
GROUP BY user_id
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)

    # This is a derived, regenerated-every-run store, so durability is not a concern;
    # these pragmas roughly halve the ~9M-row bulk-load time.
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA journal_mode = MEMORY")

    conn.executescript(SCHEMA)
    _ensure_event_indexes(conn)
    return conn


def _ensure_event_indexes(conn: sqlite3.Connection) -> None:
    for name, column in EVENT_INDEXES.items():
        conn.execute(f"CREATE INDEX IF NOT EXISTS {name} ON events({column})")


def _drop_event_indexes(conn: sqlite3.Connection) -> None:
    for name in EVENT_INDEXES:
        conn.execute(f"DROP INDEX IF EXISTS {name}")


def load_events(conn: sqlite3.Connection, data: MindData, replace: bool = True) -> int:
    """Flatten a split's impressions into the events table. Returns rows written.

    Indexes are dropped for the insert and rebuilt once at the end. Maintaining them
    row-by-row would dominate the load time on a multi-million-row split.
    """
    if replace:
        conn.execute("DELETE FROM events WHERE split = ?", (data.split,))

    _drop_event_indexes(conn)

    written = 0
    batch: list[tuple[str, str, str, int, int, str]] = []
    cursor = conn.cursor()

    def flush() -> None:
        nonlocal written, batch
        if not batch:
            return
        cursor.executemany(
            "INSERT INTO events (impression_id, user_id, news_id, clicked, hour, split) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            batch,
        )
        written += len(batch)
        batch = []

    for impression in data.impressions():
        for news_id, label in zip(impression.candidates, impression.labels):
            batch.append(
                (impression.impression_id, impression.user_id, news_id, label, impression.hour, data.split)
            )
        if len(batch) >= INSERT_BATCH_SIZE:
            flush()

    flush()
    _ensure_event_indexes(conn)
    conn.commit()
    log.info("wrote %d events for split=%s", written, data.split)
    return written


def ctr_by_hour(conn: sqlite3.Connection, split: str = "train") -> pd.DataFrame:
    return pd.read_sql_query(CTR_BY_HOUR_SQL, conn, params=(split,))


def top_users(
    conn: sqlite3.Connection,
    split: str = "train",
    min_impressions: int = 5,
    limit: int = 20,
) -> pd.DataFrame:
    return pd.read_sql_query(TOP_USERS_SQL, conn, params=(split, min_impressions, limit))


def ctr_by_category(conn: sqlite3.Connection, news: pd.DataFrame, split: str = "train") -> pd.DataFrame:
    """CTR per desk. Registers the catalogue as a table so the join happens in SQL."""
    news[["news_id", "category"]].to_sql("news", conn, if_exists="replace", index=False)
    # Index the registered catalogue so the join probes it instead of scanning it.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_news_tmp ON news(news_id)")
    return pd.read_sql_query(CTR_BY_CATEGORY_SQL, conn, params=(split,))


def user_hour_counts(conn: sqlite3.Connection, split: str = "train") -> pd.DataFrame:
    """Per-user, per-hour rollup: the input to the send-time model."""
    return pd.read_sql_query(USER_HOUR_COUNTS_SQL, conn, params=(split,))


def user_stats(conn: sqlite3.Connection, split: str = "train") -> pd.DataFrame:
    """Per-user impression and click totals: the input to the fatigue model."""
    return pd.read_sql_query(USER_STATS_SQL, conn, params=(split,))


def write_campaign_recommendations(
    conn: sqlite3.Connection,
    recommendations: pd.DataFrame,
    run_id: str,
) -> int:
    """Persist the batch scorer's output. This table is the data product."""
    required = {"user_id", "news_id", "rank", "score", "send_hour"}
    missing = required - set(recommendations.columns)
    if missing:
        raise ValueError(f"recommendations missing columns: {sorted(missing)}")

    rows = recommendations[["user_id", "news_id", "rank", "score", "send_hour"]].copy()
    rows["run_id"] = run_id

    with closing(conn.cursor()) as cursor:
        cursor.execute("DELETE FROM campaign_recommendations WHERE run_id = ?", (run_id,))
        cursor.executemany(
            "INSERT INTO campaign_recommendations (user_id, news_id, rank, score, send_hour, run_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows.itertuples(index=False, name=None),
        )
    conn.commit()
    return len(rows)


def read_campaign_recommendations(
    conn: sqlite3.Connection,
    user_id: str,
    run_id: str | None = None,
) -> pd.DataFrame:
    """Read back one user's campaign recommendations, latest run by default."""
    if run_id is None:
        return pd.read_sql_query(
            """
            SELECT * FROM campaign_recommendations
            WHERE user_id = ?
              AND run_id = (SELECT run_id FROM campaign_recommendations ORDER BY rowid DESC LIMIT 1)
            ORDER BY rank
            """,
            conn,
            params=(user_id,),
        )
    return pd.read_sql_query(
        "SELECT * FROM campaign_recommendations WHERE user_id = ? AND run_id = ? ORDER BY rank",
        conn,
        params=(user_id, run_id),
    )
