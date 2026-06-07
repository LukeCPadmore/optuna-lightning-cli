# Usage

## Tune

Tune a Lightning training config with a separate Optuna config:

```bash
optuna-lightning tune \
  --training-config examples/training.yaml \
  --optuna-config examples/optuna.yaml
```

`examples/training.yaml` describes the base Lightning objects, while
`examples/optuna.yaml` defines the study, objective metric, trial count, and
search space.

## Print Config

Preview the normalized LightningCLI-compatible config and the validated Optuna
config without running trials:

```bash
optuna-lightning print-config \
  --training-config examples/training.yaml \
  --optuna-config examples/optuna.yaml
```

## Validate

Dry-run the config stack to catch structural incompatibilities before tuning.
This parses and validates both YAML files, normalizes the Lightning config,
and instantiates the Lightning objects without training or creating an Optuna
study:

```bash
optuna-lightning validate \
  --training-config examples/training.yaml \
  --optuna-config examples/optuna.yaml
```

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
  --study-name example
```

## Studies Trials

Inspect every trial in a persisted study as a Rich table. When
`--optuna-config` is provided, `study.storage` and `study.study_name` are used
as defaults unless overridden on the command line:

```bash
optuna-lightning studies trials \
  --optuna-config examples/optuna.yaml

optuna-lightning studies trials \
  --storage sqlite:///optuna.db \
  --study-name example
```

The example Optuna config stores studies in `./optuna.db` relative to the
current working directory, so the first command can inspect trials that were
created by a previous `tune` run in the same directory.
