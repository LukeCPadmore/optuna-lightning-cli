"""Config loading and normalization for Lightning and Optuna YAML files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

from jsonargparse import ArgumentParser
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.cli import LightningArgumentParser


Direction = Literal["minimize", "maximize"]
DistributionType = Literal["float", "int", "categorical"]


@dataclass
class ObjectConfig:
    """Generic class-path configuration for instantiating helper objects.

    Attributes:
        class_path: Fully qualified import path of the class to construct.
        init_args: Keyword arguments passed to the constructor.
    """

    class_path: str
    init_args: dict[str, Any] = field(default_factory=dict)


TrainingConfig: TypeAlias = dict[str, Any]


@dataclass
class StudyConfig:
    """Optuna study configuration.

    Attributes:
        direction: Optimization direction, either ``"minimize"`` or
            ``"maximize"``.
        study_name: Optional persisted study name.
        storage: Optuna storage URL used to persist and reload the study.
        load_if_exists: Reuse an existing study when the name already exists.
        sampler: Optional sampler object configuration.
        pruner: Optional pruner object configuration.
    """

    direction: Direction = "minimize"
    study_name: str | None = None
    storage: str | None = None
    load_if_exists: bool = True
    sampler: ObjectConfig | None = None
    pruner: ObjectConfig | None = None


@dataclass
class ObjectiveConfig:
    """Objective metric settings used by Optuna.

    Attributes:
        metric: Lightning metric name read from ``trainer.callback_metrics``.
        enable_pruning: Whether to attach the Optuna pruning callback.
    """

    metric: str
    enable_pruning: bool = True


@dataclass
class OutputConfig:
    """Output locations for derived artifacts.

    Attributes:
        best_config_path: Optional path where the best normalized config is
            written after tuning.
    """

    best_config_path: str | None = None


@dataclass
class SearchSpaceItem:
    """A single Optuna search-space entry.

    Attributes:
        type: Distribution type, one of ``float``, ``int``, or ``categorical``.
        low: Lower bound for numeric distributions.
        high: Upper bound for numeric distributions.
        choices: Allowed values for categorical distributions.
        log: Use logarithmic sampling for numeric distributions.
        step: Optional step size for numeric distributions.
    """

    type: DistributionType
    low: int | float | None = None
    high: int | float | None = None
    choices: list[Any] | None = None
    log: bool = False
    step: int | float | None = None


@dataclass
class OptunaConfig:
    """Validated Optuna tuning configuration.

    Attributes:
        objective: Objective metric settings.
        search_space: Mapping from dotted training-config paths to Optuna
            distributions.
        study: Study-level configuration.
        output: Optional output locations.
        n_trials: Number of trials to run.
    """

    objective: ObjectiveConfig
    search_space: dict[str, SearchSpaceItem]
    study: StudyConfig = field(default_factory=StudyConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    n_trials: int = 10


def load_training_config(path: Path) -> TrainingConfig:
    """Load and validate a LightningCLI-style training config from YAML.

    Args:
        path: Path to a YAML file containing the base training config.

    Returns:
        The raw training config dictionary as loaded from YAML.

    Raises:
        ValueError: If the file is not a mapping or fails Lightning validation.
    """

    data = _load_yaml_dict(path)
    normalized = normalize_training_config(data)
    _training_parser().parse_object(normalized)
    return data


def normalize_training_config(config: TrainingConfig) -> dict[str, Any]:
    """Convert a flat training config into LightningCLI-compatible YAML.

    Args:
        config: Raw training config dictionary.

    Returns:
        A dict with ``trainer``, ``model``, and optional ``data`` sections in
        LightningCLI format.
    """

    normalized = {
        "trainer": config.get("trainer") or {},
        "model": _flat_object_config(config, "model"),
    }
    data = config.get("data", config.get("datamodule"))
    if data is not None:
        normalized["data"] = _flat_object_config({"data": data}, "data")
    return normalized


def _training_parser() -> LightningArgumentParser:
    parser = LightningArgumentParser(exit_on_error=False)
    parser.add_lightning_class_args(Trainer, "trainer")
    parser.add_subclass_arguments(LightningModule, "model", required=True)
    parser.add_subclass_arguments(LightningDataModule, "data", required=False)
    return parser


def load_optuna_config(path: Path) -> OptunaConfig:
    """Load, normalize, and validate an Optuna tuning config from YAML.

    Args:
        path: Path to a YAML file containing the Optuna tuning config.

    Returns:
        A validated :class:`OptunaConfig` instance.

    Raises:
        ValueError: If the config is missing required sections or contains an
            invalid search space definition.
    """

    data = _load_yaml_dict(path)
    if "objective" not in data:
        raise ValueError("optuna config requires an 'objective' section")
    if "search_space" not in data:
        raise ValueError("optuna config requires a 'search_space' section")

    study_data = data.get("study") or {}
    objective_data = data["objective"] or {}
    output_data = data.get("output") or {}
    metric = objective_data.get("metric")
    if not isinstance(metric, str) or not metric:
        raise ValueError("objective.metric must be a non-empty string")
    direction = study_data.get("direction", "minimize")
    if direction not in {"minimize", "maximize"}:
        raise ValueError("study.direction must be 'minimize' or 'maximize'")
    cfg = OptunaConfig(
        study=StudyConfig(
            direction=cast(Direction, direction),
            study_name=study_data.get("study_name"),
            storage=study_data.get("storage"),
            load_if_exists=study_data.get("load_if_exists", True),
            sampler=_optional_object_config(study_data, "sampler"),
            pruner=_optional_object_config(study_data, "pruner"),
        ),
        objective=ObjectiveConfig(
            metric=metric,
            enable_pruning=objective_data.get("enable_pruning", True),
        ),
        output=OutputConfig(
            best_config_path=output_data.get("best_config_path"),
        ),
        n_trials=data.get("n_trials", 10),
        search_space={
            path: _search_space_item(path, raw)
            for path, raw in _normalize_search_space(
                data.get("search_space") or {}
            ).items()
        },
    )
    _validate_optuna_config(cfg)
    return cfg


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    parser = ArgumentParser(exit_on_error=False)
    parsed = parser.parse_path(str(path), _skip_validation=True)
    data = parsed.as_dict()
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _normalize_search_space(raw: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}

    def visit(prefix: list[str], value: Any) -> None:
        if isinstance(value, dict) and "type" not in value:
            for key, child in value.items():
                visit([*prefix, key], child)
            return
        normalized[".".join(prefix)] = value

    for key, value in raw.items():
        visit([key], value)
    return normalized


def _object_config(data: dict[str, Any], key: str) -> ObjectConfig:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"'{key}' must be a mapping with class_path/init_args")
    class_path = value.get("class_path")
    if not isinstance(class_path, str) or not class_path:
        raise ValueError(f"'{key}.class_path' must be a non-empty string")
    init_args = value.get("init_args") or {}
    if not isinstance(init_args, dict):
        raise ValueError(f"'{key}.init_args' must be a mapping")
    return ObjectConfig(class_path=class_path, init_args=init_args)


def _flat_object_config(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(
            f"'{key}' must be a mapping with class_path and constructor args"
        )
    class_path = value.get("class_path")
    if not isinstance(class_path, str) or not class_path:
        raise ValueError(f"'{key}.class_path' must be a non-empty string")
    if "init_args" in value:
        raise ValueError(
            f"'{key}.init_args' is not supported; put constructor args directly under '{key}'"
        )
    init_args = {k: v for k, v in value.items() if k != "class_path"}
    return {"class_path": class_path, "init_args": init_args}


def _optional_object_config(data: dict[str, Any], key: str) -> ObjectConfig | None:
    if key not in data or data[key] is None:
        return None
    return _object_config(data, key)


def _search_space_item(path: str, raw: Any) -> SearchSpaceItem:
    if not isinstance(raw, dict):
        raise ValueError(f"search_space.{path} must be a mapping")
    distribution_type = raw.get("type")
    if distribution_type not in {"float", "int", "categorical"}:
        raise ValueError(
            f"search_space.{path}.type must be one of: float, int, categorical"
        )
    return SearchSpaceItem(
        type=cast(DistributionType, distribution_type),
        low=raw.get("low"),
        high=raw.get("high"),
        choices=raw.get("choices"),
        log=raw.get("log", False),
        step=raw.get("step"),
    )


def _validate_optuna_config(cfg: OptunaConfig) -> None:
    if cfg.study.direction not in {"minimize", "maximize"}:
        raise ValueError("study.direction must be 'minimize' or 'maximize'")
    if not isinstance(cfg.objective.metric, str) or not cfg.objective.metric:
        raise ValueError("objective.metric must be a non-empty string")
    if not isinstance(cfg.n_trials, int) or cfg.n_trials < 1:
        raise ValueError("n_trials must be a positive integer")
    if not cfg.search_space:
        raise ValueError("search_space must contain at least one parameter")
    for path, item in cfg.search_space.items():
        if not path:
            raise ValueError("search_space keys must be non-empty dotted paths")
        if item.type == "float":
            _validate_numeric(path, item, (int, float))
        elif item.type == "int":
            _validate_numeric(path, item, (int,))
        elif item.type == "categorical":
            if not isinstance(item.choices, list) or not item.choices:
                raise ValueError(
                    f"search_space.{path}.choices must be a non-empty list"
                )
        else:
            raise ValueError(
                f"search_space.{path}.type must be one of: float, int, categorical"
            )


def _validate_numeric(
    path: str, item: SearchSpaceItem, expected_type: tuple[type, ...]
) -> None:
    low = item.low
    high = item.high
    if not isinstance(low, expected_type) or not isinstance(high, expected_type):
        raise ValueError(f"search_space.{path}.low/high must match type '{item.type}'")
    numeric_low = cast(int | float, low)
    numeric_high = cast(int | float, high)
    if numeric_low >= numeric_high:
        raise ValueError(f"search_space.{path}.low must be less than high")
    if item.log and numeric_low <= 0:
        raise ValueError(f"search_space.{path}.low must be positive when log=true")
