"""Config-driven model registry for ElderCare inference workers.

Replaces hardcoded model creation in InferenceEngine._create_workers().
Supports per-zone model selection, versioning, and hot-swapping.
"""

import importlib
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Type

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    """Specification for a single model worker."""
    name: str
    worker_class_path: str
    config: dict[str, Any]
    checkpoint: Optional[str] = None
    version: str = "1.0"
    enabled: bool = True
    zones: list[str] = field(default_factory=list)

    @property
    def worker_class_name(self) -> str:
        return self.worker_class_path.rsplit(".", 1)[-1]

    @property
    def module_path(self) -> str:
        return self.worker_class_path.rsplit(".", 1)[0]


class ModelRegistry:
    """Registry of model specs loaded from YAML config.

    Supports per-zone model assignment and dynamic loading of worker classes.
    """

    def __init__(self, config_path: str = "configs/models.yaml") -> None:
        self._config_path = config_path
        self._specs: dict[str, ModelSpec] = {}
        self._class_cache: dict[str, Type] = {}
        self._load()

    def _load(self) -> None:
        try:
            with open(self._config_path, "r") as f:
                config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(f"Model config not found at {self._config_path}")
            return

        registry = config.get("registry", self._build_default_registry(config))
        for entry in registry:
            spec = ModelSpec(
                name=entry.get("name", "unknown"),
                worker_class_path=entry.get("worker_class", ""),
                config=entry.get("config", {}),
                checkpoint=entry.get("checkpoint"),
                version=entry.get("version", "1.0"),
                enabled=entry.get("enabled", True),
                zones=entry.get("zones", []),
            )
            if spec.enabled and spec.worker_class_path:
                self._specs[spec.name] = spec
                logger.info(f"Registered model: {spec.name} v{spec.version} -> {spec.worker_class_path}")

    @staticmethod
    def _build_default_registry(config: dict) -> list[dict]:
        """Build default registry from flat config if registry section missing."""
        return [
            {
                "name": "fall_detection",
                "worker_class": "pipeline.inference_engine.FallDetectionWorker",
                "config": {
                    "sample_rate": 50.0, "window_size": 100, "n_subcarriers": 52,
                    "confidence_threshold": 0.85, "confirmation_window_seconds": 3.0,
                    "inactivity_threshold": 0.15,
                },
                "checkpoint": config.get("fall_detection", {}).get("checkpoint"),
            },
            {
                "name": "vital_signs",
                "worker_class": "pipeline.inference_engine.VitalSignsWorker",
                "config": {
                    "sample_rate": 50.0, "n_subcarriers": 52,
                    "fft_window_seconds": 30.0, "update_interval_seconds": 5.0,
                },
            },
            {
                "name": "sleep",
                "worker_class": "pipeline.inference_engine.SleepWorker",
                "config": {"sample_rate": 50.0, "epoch_duration_minutes": 1},
                "checkpoint": config.get("sleep", {}).get("checkpoint"),
            },
            {
                "name": "activity",
                "worker_class": "pipeline.inference_engine.ActivityWorker",
                "config": {
                    "sample_rate": 50.0, "window_seconds": 30.0,
                    "threshold_active": 0.5, "threshold_still": 0.15,
                    "inactivity_timeout_seconds": 7200.0,
                    "daytime_start_hour": 6, "daytime_end_hour": 22,
                    "recovery_timeout_seconds": 30.0,
                },
            },
        ]

    def resolve_worker_class(self, spec: ModelSpec) -> Optional[Type]:
        """Dynamically import and return the worker class."""
        if spec.worker_class_path in self._class_cache:
            return self._class_cache[spec.worker_class_path]

        try:
            module_path = spec.module_path
            class_name = spec.worker_class_name
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            self._class_cache[spec.worker_class_path] = cls
            return cls
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to load worker class {spec.worker_class_path}: {e}")
            return None

    def get_specs_for_zone(self, zone_id: str) -> list[ModelSpec]:
        """Return all model specs applicable to a zone.

        If a spec has no zones list (empty), it applies to all zones.
        Otherwise only to listed zones.
        """
        result: list[ModelSpec] = []
        for spec in self._specs.values():
            if not spec.zones or zone_id in spec.zones:
                result.append(spec)
        return result

    def get_all_specs(self) -> list[ModelSpec]:
        return list(self._specs.values())

    def get_spec(self, name: str) -> Optional[ModelSpec]:
        return self._specs.get(name)

    def reload(self) -> None:
        """Hot-reload config file."""
        self._specs.clear()
        self._load()
        logger.info(f"Model registry reloaded: {len(self._specs)} models")
