"""
SleepLSTM: Sleep Quality Monitoring Model

Input:  4 features per 1-minute epoch:
          1. Mean respiration rate
          2. Respiration rate variability (std)
          3. Movement index (mean CSI amplitude variance)
          4. Body movement burst count

Output: 3-class {awake, light, deep} per epoch + Sleep Score (0–100)
"""

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SleepLSTM(nn.Module):
    """LSTM-based sleep stage classifier."""

    def __init__(
        self,
        n_features: int = 4,
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
        self.fc2 = nn.Linear(32, n_classes)  # {awake, light, deep}
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (batch, N_epochs, 4_features)

        Returns:
            class_probabilities: shape (batch, N_epochs, 3)
        """
        batch_size, seq_len, _ = x.shape

        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)
        x = self.dropout(F.relu(self.fc1(lstm_out)))  # (batch, seq_len, 32)
        logits = self.fc2(x)  # (batch, seq_len, 3)
        probs = F.softmax(logits, dim=-1)

        return probs


class SleepScorer:
    """Computes Sleep Score (0–100) from a night's sleep stages."""

    def __init__(self) -> None:
        # Stage weights for sleep quality: deep > light > awake
        self._stage_weights = {
            0: 0.0,   # awake
            1: 0.6,   # light
            2: 1.0,   # deep
        }

    def compute_score(
        self,
        stage_probs: torch.Tensor,    # (N_epochs, 3)
        session_duration_hours: float,
    ) -> Tuple[float, dict]:
        """
        Args:
            stage_probs: softmax probabilities per epoch
            session_duration_hours: total sleep session duration

        Returns:
            (sleep_score, breakdown_dict)
        """
        # Get hard class assignments
        stages = torch.argmax(stage_probs, dim=1)  # (N_epochs,)

        # Stage proportions
        n_total = len(stages)
        if n_total == 0:
            return 0.0, {"awake_pct": 0, "light_pct": 0, "deep_pct": 0, "sleep_efficiency": 0}

        awake_pct = (stages == 0).float().mean().item() * 100
        light_pct = (stages == 1).float().mean().item() * 100
        deep_pct = (stages == 2).float().mean().item() * 100

        # Sleep efficiency: percentage of time asleep during session
        sleep_efficiency = light_pct + deep_pct

        # Weighted quality component
        weighted_sum = sum(
            self._stage_weights[s.item()] for s in stages
        ) / n_total

        # Score formula: blend of deep% (40%), efficiency (40%), and weighted quality (20%)
        score = (deep_pct / 100 * 40 + sleep_efficiency / 100 * 40 + weighted_sum * 20)

        breakdown = {
            "awake_pct": round(awake_pct, 1),
            "light_pct": round(light_pct, 1),
            "deep_pct": round(deep_pct, 1),
            "sleep_efficiency": round(sleep_efficiency, 1),
        }

        return round(min(score, 100.0), 1), breakdown


class SleepFeatureExtractor:
    """Extracts sleep-related features from CSI data for 1-minute epochs."""

    def __init__(self, sample_rate: float = 50.0) -> None:
        self.sample_rate = sample_rate
        self.epoch_frames = int(60 * sample_rate)  # frames per 1-minute epoch

    def extract(
        self, csi_amplitude: np.ndarray, respiration_rate: Optional[float] = None
    ) -> np.ndarray:
        """
        Args:
            csi_amplitude: shape (N_time, n_subcarriers) — one epoch of data
            respiration_rate: optional external respiration BPM estimate

        Returns:
            4-element feature vector:
              [mean_respiration, respiration_std, movement_index, burst_count]
        """
        n_time = csi_amplitude.shape[0]

        # Movement index: mean CSI amplitude variance
        per_frame_amplitude = np.mean(np.abs(csi_amplitude), axis=1)  # (N_time,)
        movement_index = np.var(per_frame_amplitude)

        # Respiration-related features (simplified from amplitude variance pattern)
        if n_time > 2:
            frame_diffs = np.diff(per_frame_amplitude)
            respiration_std = np.std(frame_diffs) if len(frame_diffs) > 0 else 0.0
        else:
            respiration_std = 0.0

        # Burst count: number of rapid movement bursts (threshold crossings)
        threshold = 0.3
        crossings = np.diff((np.abs(per_frame_amplitude) > threshold).astype(int))
        burst_count = np.sum(crossings == 1)

        # Mean respiration rate placeholder — real estimate comes from VitalSignsEstimator
        mean_respiration = respiration_rate if respiration_rate is not None else 15.0

        return np.array([mean_respiration, respiration_std, movement_index, burst_count], dtype=np.float32)
