# Usage

## Tune

Tune a Lightning training config with a separate Optuna config:

```bash
optuna-lightning tune \
  --training-config examples/mnist-training.yaml \
  --optuna-config examples/mnist-optuna.yaml
```

Install the example extra first if you want to run the MNIST sample:

```bash
pip install -e ".[examples]"
```

`examples/mnist-training.yaml` describes the base Lightning objects, while
`examples/mnist-optuna.yaml` defines the study, objective metric, trial count,
and search space. The study uses SQLite storage, so trials persist in
`./optuna.db` relative to the directory where the command is run.

## Print Config

Preview the normalized LightningCLI-compatible config and the validated Optuna
config without running trials. The output is split into a `Lightning Base
Config` section and an `Optuna Config` section:

```bash
optuna-lightning print-config \
  --training-config examples/mnist-training.yaml \
  --optuna-config examples/mnist-optuna.yaml
```

## Validate

Dry-run the config stack to catch structural incompatibilities before tuning.
This parses and validates both YAML files, normalizes the Lightning config,
and instantiates the Lightning objects without training or creating an Optuna
study:

```bash
optuna-lightning validate \
  --training-config examples/mnist-training.yaml \
  --optuna-config examples/mnist-optuna.yaml
```

## How The Example Maps To Lightning

In the MNIST example, the model owns the training loop:

- `MnistClassifier.training_step` computes the loss.
- `MnistClassifier.validation_step` logs validation loss and accuracy.
- `MnistClassifier.configure_optimizers` creates the optimizer.

The datamodule owns data access:

- `MnistDataModule.prepare_data` downloads MNIST if needed.
- `MnistDataModule.setup` creates the train and validation datasets.
- `MnistDataModule.train_dataloader` and `val_dataloader` return loaders.

`optuna-lightning-cli` only wires the pieces together: it validates the
training and Optuna configs, patches sampled parameters into the training
config, and calls `Trainer.fit(...)`.

## Studies List

List persisted Optuna studies from a storage backend:

```bash
optuna-lightning studies list --storage sqlite:///optuna.db
```

## Studies Show

Inspect study metadata, state counts, and the best trial summary:

```bash
optuna-lightning studies show \
  --storage sqlite:///optuna.db \
  --study-name mnist
```

## Studies Trials

Inspect every trial in a persisted study as a Rich table. When
`--optuna-config` is provided, `study.storage` and `study.study_name` are used
as defaults unless overridden on the command line. The value column is labeled
with the objective metric and optimization direction when available. If you
only pass `--storage` and `--study-name`, the direction still appears on the
column header:

```bash
optuna-lightning studies trials \
  --optuna-config examples/mnist-optuna.yaml

optuna-lightning studies trials \
  --storage sqlite:///optuna.db \
  --study-name mnist
```

The example Optuna config stores studies in `./optuna.db` relative to the
current working directory, so the first command can inspect trials that were
created by a previous `tune` run in the same directory.
