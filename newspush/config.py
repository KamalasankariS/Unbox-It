"""Configuration loading.

The whole project is driven by a single YAML file so that a run is reproducible
from (code commit, config file). `Config.hash` is recorded in every metrics.json
so a result can always be traced back to the settings that produced it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@dataclass(frozen=True)
class Config:
    """Immutable view over config.yaml with dotted-path access."""

    raw: dict[str, Any]
    hash: str
    source_path: Path

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        p = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        text = p.read_text(encoding="utf-8")
        raw = yaml.safe_load(text)
        if not isinstance(raw, dict):
            raise ValueError(f"config at {p} must be a YAML mapping, got {type(raw)!r}")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        return cls(raw=raw, hash=digest, source_path=p)

    def get(self, dotted: str, default: Any = None) -> Any:
        """Fetch a nested value, e.g. cfg.get("encoder.dim")."""
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        """Like `get`, but raises if the key is missing (no silent defaults)."""
        sentinel = object()
        value = self.get(dotted, sentinel)
        if value is sentinel:
            raise KeyError(f"missing required config key: {dotted!r} in {self.source_path}")
        return value

    @property
    def seed(self) -> int:
        return int(self.require("seed"))

    def path(self, dotted: str) -> Path:
        """Resolve a `paths.*` entry relative to the repo root."""
        return self.source_path.parent / str(self.require(dotted))


def load_config(path: str | Path | None = None) -> Config:
    return Config.load(path)
