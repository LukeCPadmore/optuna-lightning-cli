from pathlib import Path
from unittest.mock import Mock, patch

import optuna
from typer.testing import CliRunner
import yaml

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
    assert "Lightning Base Config" in result.output
    assert "Optuna Config" in result.output
    assert "init_args" in result.output
    assert "model.lr" in result.output


def test_validate_accepts_compatible_configs(tmp_path: Path):
    training = tmp_path / "training.yaml"
    optuna_config = tmp_path / "optuna.yaml"
    training.write_text(
        """
trainer:
  max_epochs: 1
  logger: false
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
            "validate",
            "--training-config",
            str(training),
            "--optuna-config",
            str(optuna_config),
        ],
    )

    assert result.exit_code == 0
    assert "Validation passed" in result.output


def test_validate_fails_on_missing_model_class_path(tmp_path: Path):
    training = tmp_path / "training.yaml"
    optuna_config = tmp_path / "optuna.yaml"
    training.write_text(
        """
trainer:
  max_epochs: 1
model:
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
            "validate",
            "--training-config",
            str(training),
            "--optuna-config",
            str(optuna_config),
        ],
    )

    assert result.exit_code != 0
    assert "class_path" in result.output


def test_validate_fails_on_invalid_search_space_path(tmp_path: Path):
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
    hidden_size:
      type: categorical
      choices: [32, 64]
""",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "validate",
            "--training-config",
            str(training),
            "--optuna-config",
            str(optuna_config),
        ],
    )

    assert result.exit_code != 0
    assert "does not exist" in result.output


def test_validate_fails_on_invalid_logger_shape(tmp_path: Path):
    training = tmp_path / "training.yaml"
    optuna_config = tmp_path / "optuna.yaml"
    training.write_text(
        """
trainer:
  max_epochs: 1
  logger:
    foo: bar
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
            "validate",
            "--training-config",
            str(training),
            "--optuna-config",
            str(optuna_config),
        ],
    )

    assert result.exit_code != 0
    assert "logger" in result.output


def test_validate_fails_on_invalid_optuna_bounds(tmp_path: Path):
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
      low: 1.0
      high: 0.1
""",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "validate",
            "--training-config",
            str(training),
            "--optuna-config",
            str(optuna_config),
        ],
    )

    assert result.exit_code != 0
    assert "low must be less than high" in result.output


def test_tune_writes_best_config_to_current_working_directory(
    tmp_path: Path, monkeypatch
):
    training = tmp_path / "training.yaml"
    optuna_config = tmp_path / "optuna.yaml"
    training.write_text(
        """
trainer:
  max_epochs: 1
model:
  class_path: tests.tiny_lightning.TinyModel
  lr: 0.1
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
    study = Mock()
    study.best_trial.number = 0
    study.best_value = 0.5
    study.best_params = {"model.lr": 0.01}

    with patch("optuna_lightning_cli.cli.run_study", return_value=study):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(
            app,
            [
                "tune",
                "--training-config",
                str(training),
                "--optuna-config",
                str(optuna_config),
            ],
            catch_exceptions=False,
        )

    assert result.exit_code == 0
    output_path = tmp_path / "best_config.yaml"
    assert output_path.exists()
    saved = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert saved["model"]["init_args"]["lr"] == 0.01
    assert "best config" in result.output


def test_tune_best_config_out_overrides_default_and_config_path(tmp_path: Path):
    training = tmp_path / "training.yaml"
    optuna_config = tmp_path / "optuna.yaml"
    configured_output = tmp_path / "configured.yaml"
    cli_output = tmp_path / "cli.yaml"
    training.write_text(
        """
trainer:
  max_epochs: 1
model:
  class_path: tests.tiny_lightning.TinyModel
  lr: 0.1
""",
        encoding="utf-8",
    )
    optuna_config.write_text(
        f"""
objective:
  metric: val_loss
output:
  best_config_path: {configured_output}
search_space:
  model:
    lr:
      type: float
      low: 0.001
      high: 0.1
""",
        encoding="utf-8",
    )
    study = Mock()
    study.best_trial.number = 0
    study.best_value = 0.5
    study.best_params = {"model.lr": 0.02}

    with patch("optuna_lightning_cli.cli.run_study", return_value=study):
        result = runner.invoke(
            app,
            [
                "tune",
                "--training-config",
                str(training),
                "--optuna-config",
                str(optuna_config),
                "--best-config-out",
                str(cli_output),
            ],
            catch_exceptions=False,
        )

    assert result.exit_code == 0
    assert cli_output.exists()
    assert not configured_output.exists()
    saved = yaml.safe_load(cli_output.read_text(encoding="utf-8"))
    assert saved["model"]["init_args"]["lr"] == 0.02


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


def test_studies_trials_uses_defaults_from_optuna_config(tmp_path: Path, monkeypatch):
    examples_optuna = (
        Path(__file__).resolve().parents[1] / "examples" / "mnist-optuna.yaml"
    )
    monkeypatch.chdir(tmp_path)
    storage = "sqlite:///optuna.db"
    study = optuna.create_study(
        study_name="mnist",
        direction="maximize",
        storage=storage,
    )
    study.optimize(
        lambda trial: trial.suggest_float("model.lr", 0.001, 0.1),
        n_trials=1,
    )
    result = runner.invoke(
        app,
        [
            "studies",
            "trials",
            "--optuna-config",
            str(examples_optuna),
        ],
    )

    assert result.exit_code == 0
    assert "Trials: mnist" in result.output
    assert "Objective [val_acc, maximize]" in result.output
    assert "model.lr=" in result.output
    assert "0" in result.output


def test_studies_trials_cli_overrides_config_values(tmp_path: Path):
    storage = f"sqlite:///{tmp_path / 'override.db'}"
    study = optuna.create_study(
        study_name="actual",
        direction="minimize",
        storage=storage,
    )
    study.optimize(lambda trial: trial.suggest_float("x", 0.0, 1.0), n_trials=1)

    optuna_config = tmp_path / "optuna.yaml"
    optuna_config.write_text(
        """
study:
  study_name: ignored
  storage: sqlite:///ignored.db
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
            "studies",
            "trials",
            "--optuna-config",
            str(optuna_config),
            "--storage",
            storage,
            "--study-name",
            "actual",
        ],
    )

    assert result.exit_code == 0
    assert "Trials: actual" in result.output
    assert "Objective [val_loss, minimize]" in result.output
    assert "x=" in result.output
    assert "ignored" not in result.output


def test_studies_trials_fails_when_storage_and_study_name_are_missing():
    result = runner.invoke(app, ["studies", "trials"])

    assert result.exit_code != 0
    assert "storage must be provided" in result.output


def test_studies_trials_renders_trial_rows(tmp_path: Path):
    storage = f"sqlite:///{tmp_path / 'optuna.db'}"
    study = optuna.create_study(
        study_name="example",
        direction="minimize",
        storage=storage,
    )
    study.optimize(lambda trial: trial.suggest_float("x", 0.0, 1.0), n_trials=1)

    result = runner.invoke(
        app,
        [
            "studies",
            "trials",
            "--storage",
            storage,
            "--study-name",
            "example",
        ],
    )

    assert result.exit_code == 0
    assert "Trials: example" in result.output
    assert "Objective [minimize]" in result.output
    assert "complete" in result.output
    assert "x=" in result.output
