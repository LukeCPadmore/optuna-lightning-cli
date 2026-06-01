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
defines the study, objective metric, trial count, and search space.

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
