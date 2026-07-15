"""Data acquisition: real MIND if present, otherwise the labelled sample.

The single place that decides which data source is in play. Downstream modules receive
a `MindData` and never branch on its origin.
"""

from __future__ import annotations

import logging

from newspush.config import Config
from newspush.data import make_sample, mind_loader
from newspush.data.schema import MindData

log = logging.getLogger(__name__)


def real_mind_available(cfg: Config) -> bool:
    """True only if both real splits are present; a partial download is not usable."""
    return mind_loader.split_available(cfg.path("paths.mind_train")) and mind_loader.split_available(
        cfg.path("paths.mind_dev")
    )


def acquire(cfg: Config) -> tuple[MindData, MindData]:
    """Return (train, dev), preferring real MIND."""
    if real_mind_available(cfg):
        train = mind_loader.load_split(cfg.path("paths.mind_train"), split="train")
        dev = mind_loader.load_split(cfg.path("paths.mind_dev"), split="dev")
        log.info(
            "loaded real MIND: train=%d impressions / %d articles, dev=%d impressions / %d articles",
            len(train.behaviors),
            len(train.news),
            len(dev.behaviors),
            len(dev.news),
        )
        return train, dev

    log.warning(
        "real MIND not found at %s or %s; falling back to the simulated sample. Every metric "
        "from this run is tagged data_source='mind-format-sample' and is not a result on real data.",
        cfg.path("paths.mind_train"),
        cfg.path("paths.mind_dev"),
    )
    return make_sample.generate(cfg)
