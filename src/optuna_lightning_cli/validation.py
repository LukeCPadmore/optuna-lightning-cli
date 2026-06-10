"""Validation helpers for checking Lightning and Optuna config compatibility."""

from __future__ import annotations

from typing import Any

from optuna_lightning_cli.config import (
    OptunaConfig,
    TrainingConfig,
    normalize_training_config,
)
from optuna_lightning_cli.instantiate import instantiate_training


def validate_configs(
    training_config: TrainingConfig, optuna_config: OptunaConfig
) -> None:
    """Validate that a training config and Optuna config are compatible.

    The check resolves every Optuna search-space path against the raw training
    config and instantiates the Lightning objects without running training.

    Args:
        training_config: Raw Lightning training config loaded from YAML.
        optuna_config: Validated Optuna tuning configuration.

    Raises:
        KeyError: If any search-space path does not exist in the training
            config.
        ValueError: If Lightning instantiation fails.
    """

    for path in optuna_config.search_space:
        resolve_dotted_path(training_config, path)

    instantiate_training(normalize_training_config(training_config))


def resolve_dotted_path(config: dict[str, Any], dotted_path: str) -> Any:
    """Resolve a dotted path against a nested mapping.

    Args:
        config: Nested mapping to traverse.
        dotted_path: Dot-separated path, for example ``"model.init_args.lr"``.

    Returns:
        The value stored at the requested path.

    Raises:
        ValueError: If the dotted path is malformed.
        KeyError: If the path does not exist in ``config``.
    """

    parts = dotted_path.split(".")
    if any(not part for part in parts):
        raise ValueError(f"invalid dotted path: {dotted_path}")

    cursor: Any = config
    for part in parts:
        if not isinstance(cursor, dict) or part not in cursor:
            raise KeyError(f"search_space path '{dotted_path}' does not exist")
        cursor = cursor[part]
    return cursor
