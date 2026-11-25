"""Configs: TOML files read with the standard library."""

from __future__ import annotations

import tomllib
from pathlib import Path

from .world import World


def load(path) -> dict:
    with open(Path(path), "rb") as f:
        return tomllib.load(f)


def world_from(cfg: dict, **override) -> World:
    """The [world] table as a World, with optional overrides (for example kappa in a sweep)."""
    w = dict(cfg.get("world", {}))
    w.update(override)
    return World(**w)
