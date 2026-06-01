from pathlib import Path

import optuna
from typer.testing import CliRunner

from optuna_lightning_cli.cli import app


runner = CliRunner()


def test_print_config_outputs_validated_sections(tmp_path: Path):
    training = tmp_path / "training.yaml"
    optuna_config = tmp_path / "optuna.yaml"
    training.write_text(
        """
trainer:
  max_epochs: 1
model:
  class_path: tests.tiny_lightning.TinyModel
  lr: 0.01
""",
        encoding="utf-8",
    )
    optuna_config.write_text(
        """
objective:
  metric: val_loss
search_space:
  model:
    lr:
      type: float
      low: 0.001
      high: 0.1
""",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "print-config",
            "--training-config",
            str(training),
            "--optuna-config",
            str(optuna_config),
        ],
    )

    assert result.exit_code == 0
    assert "lightning_cli" in result.output
    assert "init_args" in result.output
    assert "model.lr" in result.output


def test_studies_list_outputs_persisted_study(tmp_path: Path):
    storage = f"sqlite:///{tmp_path / 'optuna.db'}"
    study = optuna.create_study(
        study_name="example",
        direction="minimize",
        storage=storage,
    )
    study.optimize(lambda trial: trial.suggest_float("x", 0.0, 1.0), n_trials=1)

    result = runner.invoke(app, ["studies", "list", "--storage", storage])

    assert result.exit_code == 0
    assert "example" in result.output
    assert "minimize" in result.output


def test_studies_show_outputs_best_trial(tmp_path: Path):
    storage = f"sqlite:///{tmp_path / 'optuna.db'}"
    study = optuna.create_study(
        study_name="example",
        direction="minimize",
        storage=storage,
    )
    study.optimize(lambda trial: trial.suggest_float("x", 0.0, 1.0), n_trials=1)

    result = runner.invoke(
        app,
        ["studies", "show", "--storage", storage, "--study-name", "example"],
    )

    assert result.exit_code == 0
    assert "Study: example" in result.output
    assert "Best Trial Params" in result.output
    assert "complete" in result.output


def test_studies_show_missing_study_fails(tmp_path: Path):
    storage = f"sqlite:///{tmp_path / 'optuna.db'}"
    optuna.create_study(
        study_name="example",
        direction="minimize",
        storage=storage,
    )

    result = runner.invoke(
        app,
        ["studies", "show", "--storage", storage, "--study-name", "missing"],
    )

    assert result.exit_code != 0
    assert "was not found" in result.output
