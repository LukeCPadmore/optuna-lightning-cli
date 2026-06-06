from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import optuna
from optuna_integration import PyTorchLightningPruningCallback
import yaml

from optuna_lightning_cli.config import (
    OptunaConfig,
    SearchSpaceItem,
    TrainingConfig,
    normalize_training_config,
)
from optuna_lightning_cli.instantiate import instantiate_training, instantiate_untyped


def run_study(
    training_config: TrainingConfig, optuna_config: OptunaConfig
) -> optuna.Study:
    study = optuna.create_study(
        direction=optuna_config.study.direction,
        study_name=optuna_config.study.study_name,
        storage=optuna_config.study.storage,
        load_if_exists=optuna_config.study.load_if_exists,
        sampler=_optional_untyped(optuna_config.study.sampler),
        pruner=_optional_untyped(optuna_config.study.pruner),
    )
    study.optimize(
        lambda trial: objective(trial, training_config, optuna_config),
        n_trials=optuna_config.n_trials,
    )
    return study


def save_best_config(
    training_config: TrainingConfig,
    study: optuna.Study,
    path: Path,
) -> None:
    best_config = best_training_config(training_config, study)
    normalized = normalize_training_config(best_config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(normalized, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


def best_training_config(
    training_config: TrainingConfig,
    study: optuna.Study,
) -> TrainingConfig:
    best_config = deepcopy(training_config)
    for path, value in study.best_params.items():
        patch_config_value(best_config, path, value)
    return best_config


def objective(
    trial: optuna.Trial,
    training_config: TrainingConfig,
    optuna_config: OptunaConfig,
) -> float:
    trial_config = deepcopy(training_config)

    for path, distribution in optuna_config.search_space.items():
        patch_config_value(
            trial_config,
            path,
            sample_value(trial, path, distribution),
        )

    trainer, model, datamodule = instantiate_training(
        normalize_training_config(trial_config)
    )

    pruning_callback = None
    if optuna_config.objective.enable_pruning:
        pruning_callback = PyTorchLightningPruningCallback(
            trial,
            monitor=optuna_config.objective.metric,
        )
        trainer.callbacks.append(pruning_callback)

    trainer.fit(model=model, datamodule=datamodule)
    if pruning_callback is not None:
        pruning_callback.check_pruned()
    return metric_to_float(
        trainer.callback_metrics.get(optuna_config.objective.metric),
        optuna_config.objective.metric,
    )


def sample_value(
    trial: optuna.Trial,
    name: str,
    distribution: SearchSpaceItem,
) -> Any:
    if distribution.type == "float":
        return trial.suggest_float(
            name,
            float(distribution.low),
            float(distribution.high),
            log=distribution.log,
            step=distribution.step,
        )
    if distribution.type == "int":
        return trial.suggest_int(
            name,
            int(distribution.low),
            int(distribution.high),
            log=distribution.log,
            step=distribution.step,
        )
    if distribution.type == "categorical":
        return trial.suggest_categorical(name, distribution.choices)
    raise ValueError(f"unsupported distribution type: {distribution.type}")


def patch_config_value(config: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    if any(not part for part in parts):
        raise ValueError(f"invalid dotted path: {dotted_path}")

    cursor: Any = config
    for part in parts[:-1]:
        if not isinstance(cursor, dict) or part not in cursor:
            raise KeyError(f"cannot patch missing config path: {dotted_path}")
        cursor = cursor[part]
    if not isinstance(cursor, dict):
        raise KeyError(f"cannot patch non-mapping config path: {dotted_path}")
    cursor[parts[-1]] = value


def metric_to_float(value: Any, metric_name: str) -> float:
    if value is None:
        raise ValueError(f"trainer.callback_metrics does not contain '{metric_name}'")
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        value = value.item()
    if not isinstance(value, (int, float)):
        raise ValueError(f"metric '{metric_name}' must be scalar, got {type(value)!r}")
    return float(value)


def _optional_untyped(spec):
    if spec is None:
        return None
    return instantiate_untyped(spec)
