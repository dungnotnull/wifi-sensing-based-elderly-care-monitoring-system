"""
Model performance tracking for the dashboard.

Tracks per-model confusion matrix, precision, recall, and F1 scores
from shadow mode labeled events. Provides data to the /api/model-performance
dashboard endpoint.
"""

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PerClassMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    count: int = 0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


@dataclass
class ModelPerformance:
    """Performance for a single model across all zones."""
    model_name: str
    metrics: dict[str, PerClassMetrics] = field(default_factory=dict)
    total_predictions: int = 0
    total_correct: int = 0
    last_updated: float = 0.0

    @property
    def accuracy(self) -> float:
        return self.total_correct / self.total_predictions if self.total_predictions > 0 else 0.0

    def record(self, predicted: str, actual: str) -> None:
        self.total_predictions += 1
        if predicted == actual:
            self.total_correct += 1

        for cls in {predicted, actual}:
            if cls not in self.metrics:
                self.metrics[cls] = PerClassMetrics(count=0)

        self.metrics[predicted].count += 1
        if predicted == actual:
            self.metrics[predicted].tp += 1
            self.metrics[predicted].count += 1  # second increment for count
        else:
            self.metrics[predicted].fp += 1
            self.metrics[actual].fn += 1

        self.last_updated = time.time()

    def undo_last_count(self, cls: str) -> None:
        """Correct double-count from record()."""
        if cls in self.metrics:
            self.metrics[cls].count = max(0, self.metrics[cls].count - 1)

    def to_dict(self) -> dict:
        per_class = {}
        for cls_name, m in self.metrics.items():
            per_class[cls_name] = {
                "tp": m.tp, "fp": m.fp, "fn": m.fn,
                "precision": round(m.precision, 3),
                "recall": round(m.recall, 3),
                "f1": round(m.f1, 3),
            }
        return {
            "model_name": self.model_name,
            "accuracy": round(self.accuracy, 3),
            "total_predictions": self.total_predictions,
            "per_class": per_class,
            "last_updated": self.last_updated,
        }


class PerformanceTracker:
    """In-memory performance tracker for dashboard display."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._models: dict[str, ModelPerformance] = {}
        self._confidence_bins: dict[str, list[float]] = defaultdict(list)
        self._max_confidence_samples = 1000

    def record_prediction(
        self,
        model_name: str,
        predicted: str,
        actual: str,
        confidence: Optional[float] = None,
    ) -> None:
        with self._lock:
            if model_name not in self._models:
                self._models[model_name] = ModelPerformance(model_name=model_name)
            perf = self._models[model_name]
            perf.record(predicted, actual)
            # Undo the double-count in record() for simplicity
            if predicted == actual:
                perf.metrics[predicted].count -= 1

            if confidence is not None:
                key = f"{model_name}_{predicted}"
                self._confidence_bins[key].append(confidence)
                if len(self._confidence_bins[key]) > self._max_confidence_samples:
                    self._confidence_bins[key] = self._confidence_bins[key][-self._max_confidence_samples:]

    def get_all_performance(self) -> list[dict]:
        with self._lock:
            return [m.to_dict() for m in self._models.values()]

    def get_performance(self, model_name: str) -> Optional[dict]:
        with self._lock:
            m = self._models.get(model_name)
            return m.to_dict() if m else None

    def get_confidence_distribution(self, model_name: str, n_bins: int = 10) -> dict:
        with self._lock:
            result = {}
            for key, values in self._confidence_bins.items():
                if key.startswith(model_name + "_"):
                    pred_class = key[len(model_name) + 1:]
                    if not values:
                        continue
                    import numpy as np
                    hist, bin_edges = np.histogram(values, bins=n_bins, range=(0, 1))
                    result[pred_class] = {
                        "counts": hist.tolist(),
                        "bin_edges": [round(e, 2) for e in bin_edges.tolist()],
                        "mean": round(np.mean(values), 3),
                        "std": round(np.std(values), 3),
                    }
            return result

    def reset(self) -> None:
        with self._lock:
            self._models.clear()
            self._confidence_bins.clear()


performance_tracker = PerformanceTracker()
