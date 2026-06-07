"""Helpers for inspecting persisted Optuna studies."""

from __future__ import annotations

from collections import Counter
from datetime import timedelta
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

import optuna
from optuna.trial import TrialState

from optuna_lightning_cli.config import OptunaConfig


def list_study_summaries(storage: str) -> list[optuna.study.StudySummary]:
    """Return all study summaries stored at the given Optuna storage URL.

    Args:
        storage: Optuna storage URL, for example ``sqlite:///optuna.db``.

    Returns:
        A list of Optuna study summaries available in the storage backend.
    """

    return optuna.get_all_study_summaries(storage=storage)


def load_study(study_name: str, storage: str) -> optuna.Study:
    """Load a single Optuna study from persistent storage.

    Args:
        study_name: Name of the study to load.
        storage: Optuna storage URL.

    Returns:
        The loaded Optuna study.
    """

    return optuna.load_study(study_name=study_name, storage=storage)


def state_counts(study: optuna.Study) -> Counter[str]:
    """Count trials by state for a study.

    Args:
        study: Loaded Optuna study.

    Returns:
        A counter keyed by lower-case Optuna trial state names.
    """

    return Counter(_state_name(trial.state) for trial in study.trials)


def serialize_config(value: Any) -> Any:
    """Convert dataclasses and enums into plain Python values for YAML output.

    Args:
        value: Arbitrary nested value from a config dataclass or mapping.

    Returns:
        A YAML-friendly structure containing only plain Python data types.
    """

    if is_dataclass(value):
        return serialize_config(asdict(value))
    if isinstance(value, dict):
        return {key: serialize_config(child) for key, child in value.items()}
    if isinstance(value, list):
        return [serialize_config(child) for child in value]
    if isinstance(value, Enum):
        return value.name
    return value


def optuna_config_to_dict(config: OptunaConfig) -> dict[str, Any]:
    """Serialize a validated Optuna config to a plain dictionary.

    Args:
        config: Validated Optuna config instance.

    Returns:
        A plain dictionary representation suitable for YAML dumping.
    """

    return serialize_config(config)


def best_value(summary: optuna.study.StudySummary) -> str:
    """Return a printable best value for a study summary.

    Args:
        summary: Optuna study summary.

    Returns:
        The best trial value as a string, or ``"-"`` when unavailable.
    """

    if summary.best_trial is None:
        return "-"
    return str(summary.best_trial.value)


def trial_params_string(trial: optuna.trial.FrozenTrial) -> str:
    """Render a trial's parameters as a compact string.

    Args:
        trial: Completed or running Optuna trial.

    Returns:
        A comma-separated ``name=value`` string, or ``"-"`` if there are no
        parameters.
    """

    if not trial.params:
        return "-"
    return ", ".join(f"{name}={trial.params[name]}" for name in sorted(trial.params))


def trial_duration_string(trial: optuna.trial.FrozenTrial) -> str:
    """Render a trial duration as a printable string.

    Args:
        trial: Completed or running Optuna trial.

    Returns:
        A printable duration string, or ``"-"`` if the trial has no duration.
    """

    if trial.duration is None:
        return "-"
    if isinstance(trial.duration, timedelta):
        return str(trial.duration)
    return str(trial.duration)


def _state_name(state: TrialState) -> str:
    return state.name.lower()
