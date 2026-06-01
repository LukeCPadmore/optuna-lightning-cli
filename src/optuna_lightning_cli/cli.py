from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from optuna_lightning_cli.config import load_optuna_config, load_training_config
from optuna_lightning_cli.objective import run_study


app = typer.Typer(help="Tune Lightning modules with Optuna.")
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
) -> None:
    training = load_training_config(training_config)
    optuna_cfg = load_optuna_config(optuna_config)

    study = run_study(training, optuna_cfg)

    table = Table(title="Best Trial")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("number", str(study.best_trial.number))
    table.add_row("value", str(study.best_value))
    table.add_row("params", str(study.best_params))
    console.print(table)


if __name__ == "__main__":
    app()
