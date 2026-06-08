import sys
from types import ModuleType
from unittest.mock import Mock, patch

import optuna

from optuna_lightning_cli.mlflow import (
    MLFLOW_PARENT_RUN_ID_ATTR,
    ensure_mlflow_parent_run_id,
    with_mlflow_trial_logger,
)


def test_with_mlflow_trial_logger_sets_run_name_and_tags():
    training = {
        "trainer": {
            "logger": {
                "class_path": "lightning.pytorch.loggers.MLFlowLogger",
                "experiment_name": "example-exp",
                "tags": {"existing": "tag", "optuna.trial_number": "old"},
            }
        },
        "model": {"class_path": "tests.tiny_lightning.TinyModel"},
    }
    study = optuna.create_study(study_name="example", direction="minimize")
    trial = study.ask()

    patched = with_mlflow_trial_logger(training, study, trial, "parent-123")

    logger = patched["trainer"]["logger"]
    assert logger["run_name"] == f"example-trial-{trial.number}"
    assert logger["tags"]["existing"] == "tag"
    assert logger["tags"]["mlflow.parentRunId"] == "parent-123"
    assert logger["tags"]["optuna.study_name"] == "example"
    assert logger["tags"]["optuna.trial_number"] == str(trial.number)
    assert "run_name" not in training["trainer"]["logger"]


def test_with_mlflow_trial_logger_uses_trial_name_without_study_name():
    training = {
        "trainer": {
            "logger": {
                "class_path": "lightning.pytorch.loggers.MLFlowLogger",
            }
        },
        "model": {"class_path": "tests.tiny_lightning.TinyModel"},
    }
    study = optuna.create_study(direction="minimize")
    trial = study.ask()

    patched = with_mlflow_trial_logger(training, study, trial, "parent-123")

    logger = patched["trainer"]["logger"]
    assert logger["run_name"] == f"trial-{trial.number}"
    assert logger["tags"]["mlflow.parentRunId"] == "parent-123"
    assert "optuna.study_name" not in logger["tags"]


def test_with_mlflow_trial_logger_leaves_non_mlflow_logger_unchanged():
    training = {
        "trainer": {
            "logger": {
                "class_path": "lightning.pytorch.loggers.TensorBoardLogger",
                "save_dir": "logs",
            }
        },
        "model": {"class_path": "tests.tiny_lightning.TinyModel"},
    }
    study = optuna.create_study(study_name="example", direction="minimize")
    trial = study.ask()

    patched = with_mlflow_trial_logger(training, study, trial, "parent-123")

    assert patched == training


def test_ensure_mlflow_parent_run_id_reuses_study_attr():
    study = optuna.create_study(study_name="example", direction="minimize")
    study.set_user_attr(MLFLOW_PARENT_RUN_ID_ATTR, "existing-parent")
    training = {
        "trainer": {
            "logger": {
                "class_path": "lightning.pytorch.loggers.MLFlowLogger",
            }
        },
        "model": {"class_path": "tests.tiny_lightning.TinyModel"},
    }

    assert ensure_mlflow_parent_run_id(study, training) == "existing-parent"


def test_ensure_mlflow_parent_run_id_creates_parent_run():
    study = optuna.create_study(study_name="example", direction="minimize")
    training = {
        "trainer": {
            "logger": {
                "class_path": "lightning.pytorch.loggers.MLFlowLogger",
                "experiment_name": "example-exp",
                "tracking_uri": "file:./mlruns",
            }
        },
        "model": {"class_path": "tests.tiny_lightning.TinyModel"},
    }
    experiment = Mock(experiment_id="exp-1")
    run = Mock()
    run.info.run_id = "parent-123"

    mlflow_module = ModuleType("mlflow")
    tracking_module = ModuleType("mlflow.tracking")
    client_cls = Mock()
    client = client_cls.return_value
    client.get_experiment_by_name.return_value = experiment
    client.create_run.return_value = run
    tracking_module.MlflowClient = client_cls
    mlflow_module.tracking = tracking_module

    with patch.dict(
        sys.modules,
        {
            "mlflow": mlflow_module,
            "mlflow.tracking": tracking_module,
        },
    ):
        parent_run_id = ensure_mlflow_parent_run_id(study, training)

    assert parent_run_id == "parent-123"
    assert study.user_attrs[MLFLOW_PARENT_RUN_ID_ATTR] == "parent-123"
    client_cls.assert_called_once_with(tracking_uri="file:./mlruns")
    client.get_experiment_by_name.assert_called_once_with("example-exp")
    client.create_run.assert_called_once_with(
        experiment_id="exp-1",
        tags={"mlflow.runName": "example"},
    )
