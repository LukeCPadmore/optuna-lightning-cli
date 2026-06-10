from __future__ import annotations

from copy import deepcopy
from typing import Any

import optuna

MLFLOW_PARENT_RUN_ID = "mlflow.parentRunId"
MLFLOW_PARENT_RUN_ID_ATTR = "mlflow_parent_run_id"
OPTUNA_STUDY_NAME_TAG = "optuna.study_name"
OPTUNA_TRIAL_NUMBER_TAG = "optuna.trial_number"


def ensure_mlflow_parent_run_id(
    study: optuna.Study,
    training_config: dict[str, Any],
) -> str | None:
    logger = _mlflow_logger_config(training_config)
    if logger is None:
        return None
    parent_run_id = study.user_attrs.get(MLFLOW_PARENT_RUN_ID_ATTR)
    if parent_run_id:
        return str(parent_run_id)

    try:
        from mlflow.tracking import MlflowClient
    except ModuleNotFoundError as exc:
        raise ValueError(
            "MLflow is required when using the MLFlowLogger. Install it with `pip install mlflow`."
        ) from exc
    client = MlflowClient(
        tracking_uri=logger.get("tracking_uri"),
    )
    experiment_name = logger.get("experiment_name", "lightning_logs")
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            name=experiment_name,
            artifact_location=logger.get("artifact_location"),
        )
    else:
        experiment_id = experiment.experiment_id

    parent_run = client.create_run(
        experiment_id=experiment_id,
        tags={"mlflow.runName": _display_study_name(study) or "optuna-study"},
    )
    study.set_user_attr(MLFLOW_PARENT_RUN_ID_ATTR, parent_run.info.run_id)
    return parent_run.info.run_id


def with_mlflow_trial_logger(
    training_config: dict[str, Any],
    study: optuna.Study,
    trial: optuna.Trial,
    parent_run_id: str | None,
) -> dict[str, Any]:
    config = deepcopy(training_config)
    logger = _mlflow_logger_config(config)
    if logger is None:
        return config

    study_name = _display_study_name(study)
    if study_name:
        logger["run_name"] = f"{study_name}-trial-{trial.number}"
    else:
        logger["run_name"] = f"trial-{trial.number}"

    tags = logger.get("tags") or {}
    if not isinstance(tags, dict):
        tags = {}
    tags.update(
        {
            OPTUNA_TRIAL_NUMBER_TAG: str(trial.number),
        }
    )
    if parent_run_id:
        tags[MLFLOW_PARENT_RUN_ID] = parent_run_id
    if study_name:
        tags[OPTUNA_STUDY_NAME_TAG] = study_name
    logger["tags"] = tags
    return config


def _display_study_name(study: optuna.Study) -> str | None:
    study_name = study.study_name
    if not study_name or study_name.startswith("no-name-"):
        return None
    return study_name


def _mlflow_logger_config(config: dict[str, Any]) -> dict[str, Any] | None:
    trainer = config.get("trainer")
    if not isinstance(trainer, dict):
        return None
    logger = trainer.get("logger")
    if not isinstance(logger, dict):
        return None
    class_path = logger.get("class_path")
    if class_path not in {
        "lightning.pytorch.loggers.MLFlowLogger",
        "lightning.pytorch.loggers.mlflow.MLFlowLogger",
    }:
        return None
    return logger
