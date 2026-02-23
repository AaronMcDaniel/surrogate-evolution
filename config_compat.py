"""
Compatibility helpers for transitioning from legacy single-dataset config
to modular database/surrogate-set/test-matrix config.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import toml


LEGACY_DATABASE_NAME = "legacy_aot"
LEGACY_SURROGATE_SET_NAME = "legacy_default"


def load_config_with_compat(config_path: str) -> dict[str, Any]:
    """
    Load TOML config and inject phase-1 compatibility defaults for:
    - databases
    - surrogate_sets
    - test_matrix

    This does not modify legacy behavior. Existing legacy sections (pipeline,
    model, codec, surrogate, data) remain the source of truth.
    """
    cfg = toml.load(config_path)
    return normalize_config_with_compat(cfg)


def normalize_config_with_compat(config: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(config)

    if "databases" not in out or not out["databases"]:
        out["databases"] = {
            LEGACY_DATABASE_NAME: {
                "enabled": True,
                "dataset_kind": "aot_legacy",
                "source": "legacy_sections",
            }
        }

    if "surrogate_sets" not in out or not out["surrogate_sets"]:
        out["surrogate_sets"] = {
            LEGACY_SURROGATE_SET_NAME: {
                "enabled": True,
                "source": "legacy_surrogate_section",
            }
        }

    if "test_matrix" not in out or not out["test_matrix"]:
        out["test_matrix"] = [
            {
                "enabled": True,
                "task": "evolution_surrogate",
                "database": LEGACY_DATABASE_NAME,
                "surrogate_set": LEGACY_SURROGATE_SET_NAME,
            }
        ]

    return out