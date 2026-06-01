from __future__ import annotations

import sys
from importlib import import_module
from typing import Any

from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.cli import LightningCLI

from optuna_lightning_cli.config import ObjectConfig


def instantiate_training(
    config: dict[str, Any],
) -> tuple[Trainer, LightningModule, LightningDataModule | None]:
    argv = sys.argv
    sys.argv = [argv[0]]
    try:
        cli = LightningCLI(
            args=config,
            run=False,
            subclass_mode_model=True,
            subclass_mode_data=True,
            save_config_callback=None,
            seed_everything_default=False,
        )
    finally:
        sys.argv = argv
    return cli.trainer, cli.model, cli.datamodule


def instantiate_untyped(spec: ObjectConfig) -> Any:
    module_name, class_name = spec.class_path.rsplit(".", 1)
    cls = getattr(import_module(module_name), class_name)
    return cls(**spec.init_args)
