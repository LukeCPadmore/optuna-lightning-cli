from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table
import yaml

from optuna_lightning_cli.config import (
    load_optuna_config,
    load_training_config,
    normalize_training_config,
)
from optuna_lightning_cli.objective import run_study, save_best_config
from optuna_lightning_cli.studies import (
    best_value,
    list_study_summaries,
    load_study,
    optuna_config_to_dict,
    trial_duration_string,
    trial_params_string,
    serialize_config,
    state_counts,
)
from optuna_lightning_cli.validation import validate_configs


app = typer.Typer(help="Tune Lightning modules with Optuna.")
studies_app = typer.Typer(help="Query persisted Optuna studies.")
app.add_typer(studies_app, name="studies")
console = Console()


@app.callback()
def main() -> None:
    """Tune Lightning modules with Optuna."""


@app.command()
def tune(
    training_config: Annotated[
        Path,
        typer.Option(
            "--training-config",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to the base Lightning training YAML config.",
        ),
    ],
    optuna_config: Annotated[
        Path,
        typer.Option(
            "--optuna-config",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to the Optuna tuning YAML config.",
        ),
    ],
    best_config_out: Annotated[
        Path | None,
        typer.Option(
            "--best-config-out",
            file_okay=True,
            dir_okay=False,
            writable=True,
            help="Path to write the best normalized LightningCLI config.",
        ),
    ] = None,
) -> None:
    training = load_training_config(training_config)
    optuna_cfg = load_optuna_config(optuna_config)

    study = run_study(training, optuna_cfg)
    output_path = _best_config_path(best_config_out, optuna_cfg.output.best_config_path)
    save_best_config(training, study, output_path)

    table = Table(title="Best Trial")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("number", str(study.best_trial.number))
    table.add_row("value", str(study.best_value))
    table.add_row("params", str(study.best_params))
    table.add_row("best config", str(output_path))
    console.print(table)


@app.command("print-config")
def print_config(
    training_config: Annotated[
        Path,
        typer.Option(
            "--training-config",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to the base Lightning training YAML config.",
        ),
    ],
    optuna_config: Annotated[
        Path,
        typer.Option(
            "--optuna-config",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to the Optuna tuning YAML config.",
        ),
    ],
) -> None:
    training = load_training_config(training_config)
    optuna_cfg = load_optuna_config(optuna_config)
    payload = {
        "training": {
            "user": training,
            "lightning_cli": normalize_training_config(training),
        },
        "optuna": optuna_config_to_dict(optuna_cfg),
    }
    _print_yaml(payload)


@app.command()
def validate(
    training_config: Annotated[
        Path,
        typer.Option(
            "--training-config",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to the base Lightning training YAML config.",
        ),
    ],
    optuna_config: Annotated[
        Path,
        typer.Option(
            "--optuna-config",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to the Optuna tuning YAML config.",
        ),
    ],
) -> None:
    try:
        training = load_training_config(training_config)
        optuna_cfg = load_optuna_config(optuna_config)
        validate_configs(training, optuna_cfg)
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print("Validation passed")


@studies_app.command("list")
def list_studies(
    storage: Annotated[
        str,
        typer.Option(
            "--storage", help="Optuna storage URL, for example sqlite:///optuna.db."
        ),
    ],
) -> None:
    summaries = list_study_summaries(storage)
    table = Table(title="Optuna Studies")
    table.add_column("Name")
    table.add_column("Direction")
    table.add_column("Trials", justify="right")
    table.add_column("Best Value")
    for summary in summaries:
        table.add_row(
            summary.study_name,
            ", ".join(direction.name.lower() for direction in summary.directions),
            str(summary.n_trials),
            best_value(summary),
        )
    console.print(table)


@studies_app.command("show")
def show_study(
    storage: Annotated[
        str,
        typer.Option(
            "--storage", help="Optuna storage URL, for example sqlite:///optuna.db."
        ),
    ],
    study_name: Annotated[
        str,
        typer.Option("--study-name", help="Name of the study to inspect."),
    ],
) -> None:
    try:
        study = load_study(study_name=study_name, storage=storage)
    except KeyError as exc:
        raise typer.BadParameter(
            f"Study '{study_name}' was not found in storage '{storage}'."
        ) from exc

    counts = state_counts(study)
    payload = {
        "study_name": study.study_name,
        "directions": [direction.name.lower() for direction in study.directions],
        "n_trials": len(study.trials),
        "states": dict(sorted(counts.items())),
    }
    try:
        best_trial = study.best_trial
    except ValueError:
        best_trial = None
    summary = Table(title=f"Study: {study.study_name}")
    summary.add_column("Field")
    summary.add_column("Value")
    summary.add_row("directions", ", ".join(payload["directions"]))
    summary.add_row("trials", str(payload["n_trials"]))
    if best_trial is not None:
        summary.add_row("best trial", str(best_trial.number))
        summary.add_row("best value", str(best_trial.value))
    else:
        summary.add_row("best trial", "-")
        summary.add_row("best value", "-")
    console.print(summary)

    states = Table(title="Trial States")
    states.add_column("State")
    states.add_column("Count", justify="right")
    for state, count in payload["states"].items():
        states.add_row(state, str(count))
    console.print(states)

    if best_trial is not None:
        params = Table(title="Best Trial Params")
        params.add_column("Parameter")
        params.add_column("Value")
        for name, value in best_trial.params.items():
            params.add_row(name, str(value))
        console.print(params)


@studies_app.command("trials")
def list_trials(
    optuna_config: Annotated[
        Path | None,
        typer.Option(
            "--optuna-config",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to the Optuna tuning YAML config.",
        ),
    ] = None,
    storage: Annotated[
        str | None,
        typer.Option(
            "--storage",
            help="Optuna storage URL, for example sqlite:///optuna.db.",
        ),
    ] = None,
    study_name: Annotated[
        str | None,
        typer.Option(
            "--study-name",
            help="Name of the study to inspect.",
        ),
    ] = None,
) -> None:
    optuna_cfg = load_optuna_config(optuna_config) if optuna_config else None
    resolved_storage = storage or (optuna_cfg.study.storage if optuna_cfg else None)
    resolved_study_name = study_name or (
        optuna_cfg.study.study_name if optuna_cfg else None
    )

    if not resolved_storage:
        raise typer.BadParameter(
            "storage must be provided via --storage or optuna config study.storage"
        )
    if not resolved_study_name:
        raise typer.BadParameter(
            "study-name must be provided via --study-name or optuna config "
            "study.study_name"
        )

    try:
        study = load_study(study_name=resolved_study_name, storage=resolved_storage)
    except KeyError as exc:
        raise typer.BadParameter(
            f"Study '{resolved_study_name}' was not found in storage "
            f"'{resolved_storage}'."
        ) from exc

    table = Table(title=f"Trials: {study.study_name}")
    table.add_column("Trial", justify="right")
    table.add_column("State")
    table.add_column("Value", justify="right")
    table.add_column("Duration")
    table.add_column("Params")
    for trial in study.trials:
        table.add_row(
            str(trial.number),
            trial.state.name.lower(),
            "-" if trial.value is None else str(trial.value),
            trial_duration_string(trial),
            trial_params_string(trial),
        )
    console.print(table)


def _print_yaml(payload) -> None:
    text = yaml.safe_dump(
        serialize_config(payload),
        sort_keys=False,
        default_flow_style=False,
    )
    console.print(Syntax(text, "yaml"))


def _best_config_path(cli_path: Path | None, config_path: str | None) -> Path:
    if cli_path is not None:
        return cli_path
    if config_path:
        return Path(config_path)
    return Path.cwd() / "best_config.yaml"


if __name__ == "__main__":
    app()
