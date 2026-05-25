"""
Post-training calibration for model confidence scores.

Provides temperature scaling (Platt scaling variant) and a rolling
confidence smoother so that predicted probabilities match empirical
accuracy. Used by FallDetector's TwoStageConfirmer to make the 0.85
confidence threshold meaningful.
"""

import logging
from collections import deque
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TemperatureScaling(nn.Module):
    """Post-training calibration via temperature scaling.

    Learns a single temperature parameter T that softens or sharpens
    the softmax output so that confidence matches accuracy.
    """

    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by learned temperature before softmax."""
        return logits / self.temperature

    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> float:
        """Learn optimal temperature on validation set.

        Minimizes NLL (negative log-likelihood) using LBFGS.

        Args:
            logits: raw model outputs, shape (N, C)
            labels: ground-truth class indices, shape (N,)
            lr: learning rate for LBFGS
            max_iter: maximum LBFGS iterations

        Returns:
            Final NLL after calibration.
        """
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        nll = float("inf")
        for _ in range(max_iter):
            def closure() -> torch.Tensor:
                nonlocal nll
                optimizer.zero_grad()
                scaled = self.forward(logits)
                loss = F.cross_entropy(scaled, labels)
                loss.backward()
                nll = loss.item()
                return loss
            optimizer.step(closure)

        logger.info(f"TemperatureScaling: T={self.temperature.item():.4f}, NLL={nll:.4f}")
        return nll

    def compute_ece(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        n_bins: int = 15,
    ) -> float:
        """Compute Expected Calibration Error (ECE).

        Args:
            logits: raw or temperature-scaled logits, shape (N, C)
            labels: ground-truth class indices, shape (N,)
            n_bins: number of confidence bins

        Returns:
            ECE value (lower is better calibrated).
        """
        with torch.no_grad():
            scaled = self.forward(logits)
            probs = F.softmax(scaled, dim=1)
            confidences, predictions = probs.max(dim=1)
            correct = predictions.eq(labels).float()

            bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1)
            ece = 0.0
            for i in range(n_bins):
                lo = bin_boundaries[i]
                hi = bin_boundaries[i + 1]
                mask = (confidences > lo) & (confidences <= hi)
                if mask.sum() > 0:
                    avg_conf = confidences[mask].mean().item()
                    avg_acc = correct[mask].mean().item()
                    ece += mask.sum().item() * abs(avg_acc - avg_conf)
            ece /= len(labels)
        return ece


class ConfidenceSmoother:
    """Rolling average of confidence over consecutive windows.

    Prevents single-frame confidence spikes from triggering the
    two-stage fall confirmation prematurely.
    """

    def __init__(self, window_size: int = 3) -> None:
        self._buffer: deque[float] = deque(maxlen=window_size)

    def update(self, confidence: float) -> float:
        """Add new confidence reading and return smoothed value."""
        self._buffer.append(confidence)
        return self.smoothed_confidence

    @property
    def smoothed_confidence(self) -> float:
        """Current smoothed confidence (rolling mean)."""
        if not self._buffer:
            return 0.0
        return sum(self._buffer) / len(self._buffer)

    def reset(self) -> None:
        """Clear the confidence buffer."""
        self._buffer.clear()
