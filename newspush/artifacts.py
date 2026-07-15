"""Trained state shared between the pipeline, the batch scorer and the API.

The pipeline trains once and writes a single artifact bundle. Serving loads it rather
than retraining, so the API answers from exactly the models the reported metrics
describe. `data_source` travels with the bundle for the same reason it travels with the
metrics: a served recommendation should be able to say what it was trained on.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from newspush.config import Config
from newspush.features.text import ArticleEncoder
from newspush.features.users import UserProfiles
from newspush.models.audience import PropensityModel
from newspush.models.content_selection import ContentRanker
from newspush.models.fatigue import FatigueModel
from newspush.models.send_time import SendTimeModel

log = logging.getLogger(__name__)

ARTIFACTS_FILENAME = "artifacts.pkl"


@dataclass
class Artifacts:
    """Everything serving needs, and nothing it does not."""

    encoder: ArticleEncoder
    profiles: UserProfiles
    propensity: PropensityModel
    send_time: SendTimeModel
    fatigue: FatigueModel
    news: pd.DataFrame
    popularity: dict[str, float]
    run_id: str
    data_source: str
    encoder_name: str

    def __post_init__(self) -> None:
        self._catalogue = self.news.set_index("news_id")
        self._news_category = dict(
            zip(self.news["news_id"].astype(str), self.news["category"].astype(str))
        )
        self._ranker = ContentRanker(self.encoder, self.profiles)

    def __setstate__(self, state: dict) -> None:
        # Pickling drops the derived indices, so rebuild them on load.
        self.__dict__.update(state)
        self.__post_init__()

    def __getstate__(self) -> dict:
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("_catalogue", "_news_category", "_ranker")
        }

    @property
    def ranker(self) -> ContentRanker:
        return self._ranker

    @property
    def news_category(self) -> dict[str, str]:
        return self._news_category

    def article(self, news_id: str) -> pd.Series:
        """Catalogue row for one article. Raises KeyError if unknown."""
        return self._catalogue.loc[news_id]

    def has_article(self, news_id: str) -> bool:
        return news_id in self._catalogue.index

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        with destination.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

        log.info("wrote artifacts to %s (%.1f MB)", destination, destination.stat().st_size / 1e6)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "Artifacts":
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(
                f"no artifacts at {source}. Run the pipeline first: make run"
            )

        with source.open("rb") as handle:
            artifacts = pickle.load(handle)

        if not isinstance(artifacts, cls):
            raise TypeError(f"{source} does not contain an Artifacts bundle")

        log.info("loaded artifacts from run %s (data_source=%s)", artifacts.run_id, artifacts.data_source)
        return artifacts


def artifacts_path(cfg: Config) -> Path:
    """Artifacts live beside the runs, but are not per-run: they are large and the
    latest bundle is the only one serving ever wants."""
    return cfg.path("paths.runs_dir") / ARTIFACTS_FILENAME
