"""YAML config loader with env var interpolation."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


def _interpolate_env(value: Any) -> Any:
    """Recursively resolve ${ENV_VAR} patterns in config values."""
    if isinstance(value, str):
        return re.sub(
            r"\$\{([^}]+)\}",
            lambda m: os.environ.get(m.group(1), ""),
            value,
        )
    if isinstance(value, dict):
        return {k: _interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env(v) for v in value]
    return value


def load_config(
    path: str | Path | None = None,
    profile: str | None = None,
) -> dict[str, Any]:
    """Load and resolve llmgate YAML config.

    Supports flat configs and multi-profile configs with defaults.
    """
    if path is None:
        path = Path.cwd() / "llmgate.yaml"
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    raw = _interpolate_env(raw)

    # Multi-profile config
    if "profiles" in raw:
        profile_name = profile or raw.get("active_profile")
        if not profile_name or profile_name not in raw["profiles"]:
            raise ValueError(
                f"Profile '{profile_name}' not found. "
                f"Available: {list(raw['profiles'].keys())}"
            )
        defaults = raw.get("defaults", {})
        resolved = {**defaults, **raw["profiles"][profile_name]}
    else:
        # Flat config
        resolved = raw

    if "provider" not in resolved:
        raise ValueError("Config must specify a 'provider' field")

    return resolved
