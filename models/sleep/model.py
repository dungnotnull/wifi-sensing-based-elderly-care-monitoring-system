"""
SleepLSTM: Sleep Quality Monitoring Model

Input:  6 features per 1-minute epoch:
          1. Mean respiration rate
          2. Respiration rate variability (std)
          3. Movement index (mean CSI amplitude variance)
          4. Body movement burst count
          5. Movement rate of change (transition feature)
          6. Wakefulness index (sub-minute motion burst density, 0-1)

Output: 3-class {awake, light, deep} per epoch + Sleep Score (0–100)
"""

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance.

    Down-weights easy examples so the model focuses on hard,
    misclassified ones. Standard formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_weight = (1.0 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            focal_weight = focal_weight * alpha_t

        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class SleepLSTM(nn.Module):
    """LSTM-based sleep stage classifier with 6 input features."""

    def __init__(
        self,
        n_features: int = 6,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_classes: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (batch, N_epochs, 6)

        Returns:
            class_probabilities: shape (batch, N_epochs, 3)
        """
        lstm_out, _ = self.lstm(x)
        x = self.dropout(F.relu(self.fc1(lstm_out)))
        logits = self.fc2(x)
        probs = F.softmax(logits, dim=-1)
        return probs


class SleepScorer:
    """Computes Sleep Score (0–100) from a night's sleep stages."""

    def __init__(self) -> None:
        self._stage_weights = {
            0: 0.0,
            1: 0.6,
            2: 1.0,
        }

    def compute_score(
        self,
        stage_probs: torch.Tensor,
        session_duration_hours: float,
    ) -> Tuple[float, dict]:
        stages = torch.argmax(stage_probs, dim=1)
        n_total = len(stages)
        if n_total == 0:
            return 0.0, {"awake_pct": 0, "light_pct": 0, "deep_pct": 0, "sleep_efficiency": 0}

        awake_pct = (stages == 0).float().mean().item() * 100
        light_pct = (stages == 1).float().mean().item() * 100
        deep_pct = (stages == 2).float().mean().item() * 100
        sleep_efficiency = light_pct + deep_pct

        weighted_sum = sum(self._stage_weights[s.item()] for s in stages) / n_total
        score = (deep_pct / 100 * 40 + sleep_efficiency / 100 * 40 + weighted_sum * 20)

        breakdown = {
            "awake_pct": round(awake_pct, 1),
            "light_pct": round(light_pct, 1),
            "deep_pct": round(deep_pct, 1),
            "sleep_efficiency": round(sleep_efficiency, 1),
        }
        return round(min(score, 100.0), 1), breakdown


class SleepFeatureExtractor:
    """Extracts 6 sleep-related features from CSI data for 1-minute epochs.

    Feature 6 (wakefulness index) is the key discriminative feature
    that separates awake from light sleep: it measures sub-minute
    motion burst density, which is consistently higher when awake.
    """

    def __init__(self, sample_rate: float = 50.0) -> None:
        self.sample_rate = sample_rate
        self.epoch_frames = int(60 * sample_rate)
        self._prev_movement_index: Optional[float] = None

    def extract(
        self, csi_amplitude: np.ndarray, respiration_rate: Optional[float] = None
    ) -> np.ndarray:
        """
        Args:
            csi_amplitude: shape (N_time, n_subcarriers) -- one epoch of data
            respiration_rate: optional external respiration BPM estimate

        Returns:
            6-element feature vector:
              [mean_respiration, respiration_std, movement_index,
               burst_count, movement_rate_of_change, wakefulness_index]
        """
        n_time = csi_amplitude.shape[0]

        per_frame_amplitude = np.mean(np.abs(csi_amplitude), axis=1)
        movement_index = float(np.var(per_frame_amplitude))

        if n_time > 2:
            frame_diffs = np.diff(per_frame_amplitude)
            respiration_std = float(np.std(frame_diffs)) if len(frame_diffs) > 0 else 0.0
        else:
            respiration_std = 0.0

        threshold = 0.3
        crossings = np.diff((np.abs(per_frame_amplitude) > threshold).astype(int))
        burst_count = float(np.sum(crossings == 1))

        mean_respiration = respiration_rate if respiration_rate is not None else 15.0

        if self._prev_movement_index is not None:
            movement_rate_of_change = movement_index - self._prev_movement_index
        else:
            movement_rate_of_change = 0.0
        self._prev_movement_index = movement_index

        # Wakefulness index: sub-minute motion burst density (0-1)
        # Split epoch into 10-second sub-windows and count high-variance windows
        sub_window_frames = max(1, int(10 * self.sample_rate))
        n_sub_windows = max(1, n_time // sub_window_frames)
        active_windows = 0
        for w in range(n_sub_windows):
            start = w * sub_window_frames
            end = min(start + sub_window_frames, n_time)
            sub_amp = per_frame_amplitude[start:end]
            if len(sub_amp) > 1 and np.var(sub_amp) > 0.01:
                active_windows += 1
        wakefulness_index = active_windows / n_sub_windows

        return np.array(
            [
                mean_respiration,
                respiration_std,
                movement_index,
                burst_count,
                movement_rate_of_change,
                wakefulness_index,
            ],
            dtype=np.float32,
        )
