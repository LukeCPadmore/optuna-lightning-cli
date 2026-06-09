# optuna-lightning-cli

[![CI](https://github.com/LukeCPadmore/optuna-lightning-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/LukeCPadmore/optuna-lightning-cli/actions/workflows/ci.yml)
[![Docs](https://github.com/LukeCPadmore/optuna-lightning-cli/actions/workflows/docs.yml/badge.svg)](https://github.com/LukeCPadmore/optuna-lightning-cli/actions/workflows/docs.yml)
[![Publish](https://github.com/LukeCPadmore/optuna-lightning-cli/actions/workflows/publish.yml/badge.svg)](https://github.com/LukeCPadmore/optuna-lightning-cli/actions/workflows/publish.yml)
[![PyPI](https://img.shields.io/pypi/v/optuna-lightning-cli.svg)](https://pypi.org/project/optuna-lightning-cli/)
[![Python](https://img.shields.io/pypi/pyversions/optuna-lightning-cli.svg)](https://pypi.org/project/optuna-lightning-cli/)
[![Docs site](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://lukecpadmore.github.io/optuna-lightning-cli/)

A small Typer CLI for tuning `lightning.pytorch` modules with Optuna.

Install the MNIST example extra and run the sample HPO flow:

```bash
pip install -e ".[examples]"

optuna-lightning tune \
  --training-config examples/mnist-training.yaml \
  --optuna-config examples/mnist-optuna.yaml
```

`MnistClassifier` owns the training loop (`training_step`,
`validation_step`, and `configure_optimizers`). `MnistDataModule` owns the
MNIST download and data loaders. The CLI validates the config pair, patches
Optuna samples into the flat training config, and hands everything to
`Trainer.fit()`.

`print-config` renders the normalized Lightning config under a `Lightning
Base Config` header and the Optuna config under an `Optuna Config` header.

The CLI also includes config printing, validation, and persisted study
inspection:

```bash
optuna-lightning print-config \
  --training-config examples/mnist-training.yaml \
  --optuna-config examples/mnist-optuna.yaml

optuna-lightning validate \
  --training-config examples/mnist-training.yaml \
  --optuna-config examples/mnist-optuna.yaml

optuna-lightning studies list --storage sqlite:///optuna.db
optuna-lightning studies show \
  --storage sqlite:///optuna.db \
  --study-name mnist
optuna-lightning studies trials \
  --optuna-config examples/mnist-optuna.yaml
```

The example Optuna config stores studies in `./optuna.db` relative to the
directory where the command is run, so `studies trials --optuna-config
examples/mnist-optuna.yaml` can inspect persisted trials later. The trials
table labels the value column as `Objective [val_acc, maximize]` for the MNIST
example, and storage-only inspection still shows the study direction in the
column header.
