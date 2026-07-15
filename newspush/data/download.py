"""Fetch the real MIND-small dataset.

The only networked code in the project. `make data` calls it; the pipeline never does.

The original Microsoft blob endpoints (mind201910small.blob.core.windows.net) now
return HTTP 409 PublicAccessNotPermitted for everyone, so this pulls from the
HuggingFace re-host maintained by the recommenders-team project. Override with
MIND_TRAIN_URL / MIND_DEV_URL to use a different mirror.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

from newspush.config import Config, load_config

log = logging.getLogger(__name__)

HF_BASE = "https://huggingface.co/datasets/Recommenders/MIND/resolve/main"
TRAIN_URL = os.environ.get("MIND_TRAIN_URL", f"{HF_BASE}/MINDsmall_train.zip")
DEV_URL = os.environ.get("MIND_DEV_URL", f"{HF_BASE}/MINDsmall_dev.zip")

CHUNK_SIZE = 1 << 20


def fetch(cfg: Config, keep_archives: bool = False) -> None:
    """Download and extract both MIND-small splits, skipping any already present."""
    data_dir = cfg.path("paths.data_dir")
    targets = [
        (TRAIN_URL, data_dir / "MINDsmall_train.zip", cfg.path("paths.mind_train")),
        (DEV_URL, data_dir / "MINDsmall_dev.zip", cfg.path("paths.mind_dev")),
    ]

    for url, archive, out_dir in targets:
        if (out_dir / "news.tsv").is_file() and (out_dir / "behaviors.tsv").is_file():
            log.info("%s already present, skipping", out_dir)
            continue

        _download(url, archive)
        _extract(archive, out_dir)

        if not keep_archives:
            archive.unlink(missing_ok=True)


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s", url)

    request = urllib.request.Request(url, headers={"User-Agent": "newspush/0.1"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as handle:  # noqa: S310
        while chunk := response.read(CHUNK_SIZE):
            handle.write(chunk)

    log.info("downloaded %.1f MB to %s", destination.stat().st_size / 1e6, destination)


def _extract(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as zf:
        corrupt_entry = zf.testzip()
        if corrupt_entry is not None:
            raise RuntimeError(f"{archive} is corrupt (first bad entry: {corrupt_entry})")
        zf.extractall(out_dir)

    log.info("extracted %s to %s", archive.name, out_dir)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Download the MIND-small dataset.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--keep-archives", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    try:
        fetch(cfg, keep_archives=args.keep_archives)
    except Exception as exc:
        log.error("MIND download failed: %s", exc)
        log.error(
            "the pipeline still runs without it, on the simulated sample. To use real data, "
            "place news.tsv and behaviors.tsv under %s and %s.",
            cfg.path("paths.mind_train"),
            cfg.path("paths.mind_dev"),
        )
        return 1

    log.info("MIND-small ready under %s", cfg.path("paths.data_dir"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
