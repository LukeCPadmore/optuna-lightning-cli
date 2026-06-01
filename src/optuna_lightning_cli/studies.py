from __future__ import annotations

from collections import Counter
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

import optuna
from optuna.trial import TrialState

from optuna_lightning_cli.config import OptunaConfig


def list_study_summaries(storage: str) -> list[optuna.study.StudySummary]:
    return optuna.get_all_study_summaries(storage=storage)


def load_study(study_name: str, storage: str) -> optuna.Study:
    return optuna.load_study(study_name=study_name, storage=storage)


def state_counts(study: optuna.Study) -> Counter[str]:
    return Counter(_state_name(trial.state) for trial in study.trials)


def serialize_config(value: Any) -> Any:
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
    return serialize_config(config)


def best_value(summary: optuna.study.StudySummary) -> str:
    if summary.best_trial is None:
        return "-"
    return str(summary.best_trial.value)


def _state_name(state: TrialState) -> str:
    return state.name.lower()
