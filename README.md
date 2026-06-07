# optuna-lightning-cli

[![CI](https://github.com/LukeCPadmore/optuna-lightning-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/LukeCPadmore/optuna-lightning-cli/actions/workflows/ci.yml)
[![Docs](https://github.com/LukeCPadmore/optuna-lightning-cli/actions/workflows/docs.yml/badge.svg)](https://github.com/LukeCPadmore/optuna-lightning-cli/actions/workflows/docs.yml)
[![Publish](https://github.com/LukeCPadmore/optuna-lightning-cli/actions/workflows/publish.yml/badge.svg)](https://github.com/LukeCPadmore/optuna-lightning-cli/actions/workflows/publish.yml)
[![PyPI](https://img.shields.io/pypi/v/optuna-lightning-cli.svg)](https://pypi.org/project/optuna-lightning-cli/)
[![Python](https://img.shields.io/pypi/pyversions/optuna-lightning-cli.svg)](https://pypi.org/project/optuna-lightning-cli/)
[![Docs site](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://lukecpadmore.github.io/optuna-lightning-cli/)

A small Typer CLI for tuning `lightning.pytorch` modules with Optuna.

```bash
optuna-lightning tune \
  --training-config examples/training.yaml \
  --optuna-config examples/optuna.yaml
```

The training config defines the base Lightning objects. The Optuna config
defines the study, objective metric, trial count, and search space. The CLI
also includes config printing, validation, and persisted study inspection:

```bash
optuna-lightning print-config \
  --training-config examples/training.yaml \
  --optuna-config examples/optuna.yaml

optuna-lightning validate \
  --training-config examples/training.yaml \
  --optuna-config examples/optuna.yaml

optuna-lightning studies list --storage sqlite:///optuna.db
optuna-lightning studies show \
  --storage sqlite:///optuna.db \
  --study-name example
optuna-lightning studies trials \
  --optuna-config examples/optuna.yaml
```

`examples/training.yaml` uses LightningCLI-style sections with flat constructor
arguments:

```yaml
trainer:
  max_epochs: 3

model:
  class_path: my_project.models.MyLightningModule
  lr: 0.001

data:
  class_path: my_project.data.MyDataModule
  batch_size: 32
```

`examples/optuna.yaml` nests search-space entries under the training config
section they patch:

```yaml
study:
  study_name: example
  storage: sqlite:///optuna.db
search_space:
  model:
    lr:
      type: float
      low: 0.0001
      high: 0.1
      log: true
    hidden_size:
      type: categorical
      choices: [32, 64, 128]
```

The example Optuna config stores studies in `./optuna.db` relative to the
directory where the command is run, so `studies trials --optuna-config
examples/optuna.yaml` can inspect persisted trials later.
