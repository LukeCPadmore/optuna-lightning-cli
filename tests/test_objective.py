import optuna
import pytest
import torch

from optuna_lightning_cli.config import (
    ObjectiveConfig,
    OptunaConfig,
    SearchSpaceItem,
)
from optuna_lightning_cli.objective import (
    best_training_config,
    metric_to_float,
    objective,
    patch_config_value,
    save_best_config,
)


def test_patch_config_value_sets_nested_value():
    cfg = {"model": {"lr": 0.1}}

    patch_config_value(cfg, "model.lr", 0.01)

    assert cfg["model"]["lr"] == 0.01


def test_patch_config_value_allows_new_leaf_key():
    cfg = {"model": {}}

    patch_config_value(cfg, "model.lr", 0.01)

    assert cfg["model"]["lr"] == 0.01


def test_metric_to_float_accepts_tensor_scalar():
    assert metric_to_float(torch.tensor(1.25), "val_loss") == pytest.approx(1.25)


def test_best_training_config_patches_best_params():
    training = {"model": {"class_path": "tests.tiny_lightning.TinyModel", "lr": 0.1}}
    study = optuna.create_study(direction="minimize")
    study.enqueue_trial({"model.lr": 0.01})
    study.optimize(
        lambda trial: trial.suggest_float("model.lr", 0.001, 0.1), n_trials=1
    )

    best_config = best_training_config(training, study)

    assert best_config["model"]["lr"] == 0.01
    assert training["model"]["lr"] == 0.1


def test_save_best_config_writes_normalized_lightning_cli_yaml(tmp_path):
    training = {
        "trainer": {"max_epochs": 1},
        "model": {"class_path": "tests.tiny_lightning.TinyModel", "lr": 0.1},
    }
    study = optuna.create_study(direction="minimize")
    study.enqueue_trial({"model.lr": 0.01})
    study.optimize(
        lambda trial: trial.suggest_float("model.lr", 0.001, 0.1), n_trials=1
    )
    path = tmp_path / "best_config.yaml"

    save_best_config(training, study, path)

    text = path.read_text(encoding="utf-8")
    assert "init_args:" in text
    assert "lr: 0.01" in text


def test_objective_smoke():
    training = {
        "trainer": {
            "max_epochs": 1,
            "accelerator": "cpu",
            "devices": 1,
            "logger": False,
            "enable_checkpointing": False,
            "enable_progress_bar": False,
            "limit_train_batches": 1,
            "limit_val_batches": 1,
            "num_sanity_val_steps": 0,
        },
        "model": {
            "class_path": "tests.tiny_lightning.TinyModel",
            "lr": 0.01,
        },
        "data": {
            "class_path": "tests.tiny_lightning.TinyDataModule",
            "batch_size": 2,
        },
    }
    optuna_cfg = OptunaConfig(
        objective=ObjectiveConfig(metric="val_loss", enable_pruning=True),
        n_trials=1,
        search_space={
            "model.lr": SearchSpaceItem(
                type="float",
                low=0.001,
                high=0.01,
            )
        },
    )
    study = optuna.create_study(direction="minimize")
    trial = study.ask()

    value = objective(trial, training, optuna_cfg)

    assert isinstance(value, float)
