"""Model quantization pipeline for edge deployment.

Post-training static quantization for CSI-FallNet to reduce
model size and inference latency on Raspberry Pi 5 CPU.
"""

import copy
import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.quantization as quant

logger = logging.getLogger(__name__)


class ModelQuantizer:
    """Quantizes PyTorch models for CPU inference."""

    def __init__(self, backend: str = "fbgemm") -> None:
        self._backend = backend

    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Prepare a model for quantization by inserting observers."""
        model.eval()
        model.qconfig = quant.get_default_qconfig(self._backend)
        prepared = quant.prepare(model)
        return prepared

    def calibrate(
        self,
        model: nn.Module,
        calibration_data: torch.utils.data.DataLoader,
        max_batches: int = 100,
    ) -> None:
        """Run calibration data through the model to collect statistics."""
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(calibration_data):
                if isinstance(batch, (list, tuple)):
                    model(batch[0])
                else:
                    model(batch)
                if i >= max_batches:
                    break
        logger.info(f"Calibration complete: {min(i + 1, max_batches)} batches")

    def convert(self, model: nn.Module) -> nn.Module:
        """Convert calibrated model to quantized version."""
        quantized = quant.convert(model)
        return quantized

    def quantize_fall_detector(
        self,
        model: nn.Module,
        calibration_data: Optional[torch.utils.data.DataLoader] = None,
        checkpoint_path: Optional[str] = None,
    ) -> Tuple[nn.Module, dict]:
        """Full quantization pipeline for CSI-FallNet.

        Args:
            model: FallDetector model instance.
            calibration_data: DataLoader with representative CSI windows.
            checkpoint_path: Path to save quantized model.

        Returns:
            (quantized_model, benchmark_results)
        """
        original_size = self._model_size_mb(model)

        if calibration_data is None:
            logger.warning("No calibration data provided, generating synthetic")
            calibration_data = self._synthetic_calibration()

        original_model = copy.deepcopy(model)
        prepared = self.prepare_model(model)
        self.calibrate(prepared, calibration_data)
        quantized = self.convert(prepared)

        quantized_size = self._model_size_mb(quantized)

        benchmark = self._benchmark(original_model, quantized)

        result = {
            "original_size_mb": original_size,
            "quantized_size_mb": quantized_size,
            "size_reduction_pct": round((1 - quantized_size / original_size) * 100, 1),
            "original_latency_ms": benchmark["original_ms"],
            "quantized_latency_ms": benchmark["quantized_ms"],
            "speedup": round(benchmark["original_ms"] / max(benchmark["quantized_ms"], 0.001), 2),
        }

        logger.info(
            f"Quantization complete: {original_size:.2f}MB -> {quantized_size:.2f}MB "
            f"({result['size_reduction_pct']}% reduction), "
            f"latency: {benchmark['original_ms']:.2f}ms -> {benchmark['quantized_ms']:.2f}ms "
            f"({result['speedup']}x speedup)"
        )

        if checkpoint_path:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(quantized.state_dict(), checkpoint_path)
            logger.info(f"Quantized model saved to {checkpoint_path}")

        return quantized, result

    def _synthetic_calibration(self) -> torch.utils.data.DataLoader:
        """Generate synthetic CSI data for calibration."""
        from torch.utils.data import TensorDataset, DataLoader

        data = torch.randn(200, 100, 52, dtype=torch.float32)
        dataset = TensorDataset(data)
        return DataLoader(dataset, batch_size=32)

    @staticmethod
    def _model_size_mb(model: nn.Module) -> float:
        """Calculate model size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024 * 1024)

    @staticmethod
    def _benchmark(
        original: nn.Module, quantized: nn.Module, n_runs: int = 100,
    ) -> dict:
        """Benchmark original vs quantized model latency."""
        dummy_input = torch.randn(1, 100, 52, dtype=torch.float32)

        original.eval()
        with torch.no_grad():
            start = time.perf_counter()
            for _ in range(n_runs):
                original(dummy_input)
            original_ms = (time.perf_counter() - start) / n_runs * 1000

        with torch.no_grad():
            start = time.perf_counter()
            for _ in range(n_runs):
                quantized(dummy_input)
            quantized_ms = (time.perf_counter() - start) / n_runs * 1000

        return {"original_ms": round(original_ms, 3), "quantized_ms": round(quantized_ms, 3)}


def quantize_from_checkpoint(
    checkpoint_path: str = "models/fall_detection/checkpoints/csi_fallnet_best.pth",
    output_path: str = "models/fall_detection/checkpoints/csi_fallnet_quantized.pth",
) -> Optional[dict]:
    """CLI-friendly quantization from a saved checkpoint."""
    from models.fall_detection.model import FallDetector

    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return None

    model = FallDetector(n_subcarriers=52, sequence_length=100)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    quantizer = ModelQuantizer()
    _, results = quantizer.quantize_fall_detector(
        model=model,
        checkpoint_path=output_path,
    )
    return results
