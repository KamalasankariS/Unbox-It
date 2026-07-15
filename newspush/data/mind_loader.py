"""Loader for the real MIND TSV files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from newspush.data.schema import BEHAVIOR_COLUMNS, NEWS_COLUMNS, REAL, MindData


def split_available(split_dir: str | Path) -> bool:
    """True if both MIND TSVs are present in `split_dir`."""
    directory = Path(split_dir)
    return (directory / "news.tsv").is_file() and (directory / "behaviors.tsv").is_file()


def load_split(split_dir: str | Path, split: str) -> MindData:
    """Read one MIND split directory. Raises if the TSVs are absent."""
    directory = Path(split_dir)
    news_path = directory / "news.tsv"
    behaviors_path = directory / "behaviors.tsv"

    if not news_path.is_file() or not behaviors_path.is_file():
        raise FileNotFoundError(f"expected news.tsv and behaviors.tsv under {directory}")

    read_options = {
        "sep": "\t",
        "header": None,
        "dtype": str,
        "quoting": 3,  # QUOTE_NONE: MIND titles contain unescaped double quotes
        "na_filter": False,
    }
    news = pd.read_csv(news_path, names=NEWS_COLUMNS, **read_options)
    behaviors = pd.read_csv(behaviors_path, names=BEHAVIOR_COLUMNS, **read_options)

    news = news.drop_duplicates(subset="news_id", keep="first").reset_index(drop=True)

    return MindData(news=news, behaviors=behaviors, data_source=REAL, split=split)
