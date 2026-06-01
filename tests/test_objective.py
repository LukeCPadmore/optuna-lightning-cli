import optuna
import pytest
import torch

from optuna_lightning_cli.config import (
    ObjectiveConfig,
    OptunaConfig,
    SearchSpaceItem,
)
from optuna_lightning_cli.objective import (
    metric_to_float,
    objective,
    patch_config_value,
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
