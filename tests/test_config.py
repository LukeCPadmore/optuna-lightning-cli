from pathlib import Path

import pytest

from optuna_lightning_cli.config import load_optuna_config, load_training_config


def test_load_training_config(tmp_path: Path):
    path = tmp_path / "training.yaml"
    path.write_text(
        """
trainer:
  max_epochs: 1
model:
  class_path: tests.tiny_lightning.TinyModel
  lr: 0.01
""",
        encoding="utf-8",
    )

    cfg = load_training_config(path)

    assert cfg["trainer"]["max_epochs"] == 1
    assert cfg["model"]["lr"] == 0.01
    assert "data" not in cfg


def test_load_optuna_config(tmp_path: Path):
    path = tmp_path / "optuna.yaml"
    path.write_text(
        """
objective:
  metric: val_loss
n_trials: 2
search_space:
  model:
    lr:
      type: float
      low: 0.001
      high: 0.1
      log: true
""",
        encoding="utf-8",
    )

    cfg = load_optuna_config(path)

    assert cfg.objective.metric == "val_loss"
    assert cfg.n_trials == 2
    assert cfg.search_space["model.lr"].log is True


def test_invalid_search_space_fails(tmp_path: Path):
    path = tmp_path / "optuna.yaml"
    path.write_text(
        """
objective:
  metric: val_loss
search_space:
  model:
    lr:
      type: float
      low: 1.0
      high: 0.1
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="low must be less than high"):
        load_optuna_config(path)
