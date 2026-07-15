"""In-memory schema shared by the real MIND loader and the sample generator.

Everything downstream consumes `MindData` and nothing else, so swapping real data for
the simulated sample is a no-op for the rest of the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd

NEWS_COLUMNS = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]

BEHAVIOR_COLUMNS = [
    "impression_id",
    "user_id",
    "time",
    "history",
    "impressions",
]

REAL = "real-MIND"
SAMPLE = "mind-format-sample"


@dataclass(frozen=True)
class Impression:
    """One parsed row of behaviors.tsv."""

    impression_id: str
    user_id: str
    hour: int
    history: list[str]
    candidates: list[str]
    labels: list[int]


@dataclass(frozen=True)
class MindData:
    """A MIND split: the article catalogue plus the impression log.

    `data_source` is REAL or SAMPLE and is propagated into metrics.json, so every
    reported number carries the provenance of the data it came from.
    """

    news: pd.DataFrame
    behaviors: pd.DataFrame
    data_source: str
    split: str

    def __post_init__(self) -> None:
        missing_news = set(NEWS_COLUMNS) - set(self.news.columns)
        if missing_news:
            raise ValueError(f"news frame missing columns: {sorted(missing_news)}")
        missing_behaviors = set(BEHAVIOR_COLUMNS) - set(self.behaviors.columns)
        if missing_behaviors:
            raise ValueError(f"behaviors frame missing columns: {sorted(missing_behaviors)}")
        if self.data_source not in (REAL, SAMPLE):
            raise ValueError(f"data_source must be {REAL!r} or {SAMPLE!r}, got {self.data_source!r}")

    @property
    def is_simulated(self) -> bool:
        return self.data_source == SAMPLE

    def impressions(self) -> Iterator[Impression]:
        """Yield parsed impressions, skipping malformed rows."""
        for row in self.behaviors.itertuples(index=False):
            parsed = parse_impression(
                impression_id=str(row.impression_id),
                user_id=str(row.user_id),
                time=str(row.time),
                history=str(row.history) if pd.notna(row.history) else "",
                impressions=str(row.impressions) if pd.notna(row.impressions) else "",
            )
            if parsed is not None:
                yield parsed


def parse_hour(time_str: str) -> int:
    """Hour of day from a MIND timestamp, defaulting to midday when unparseable."""
    timestamp = pd.to_datetime(time_str, errors="coerce")
    if pd.isna(timestamp):
        return 12
    return int(timestamp.hour)


def parse_impression(
    impression_id: str,
    user_id: str,
    time: str,
    history: str,
    impressions: str,
) -> Impression | None:
    """Parse one behaviors row. Returns None if it carries no usable candidates."""
    candidates: list[str] = []
    labels: list[int] = []

    for token in impressions.split():
        news_id, separator, label = token.rpartition("-")
        if not separator or label not in ("0", "1"):
            continue
        candidates.append(news_id)
        labels.append(int(label))

    if not candidates:
        return None

    return Impression(
        impression_id=impression_id,
        user_id=user_id,
        hour=parse_hour(time),
        history=history.split() if history.strip() else [],
        candidates=candidates,
        labels=labels,
    )
