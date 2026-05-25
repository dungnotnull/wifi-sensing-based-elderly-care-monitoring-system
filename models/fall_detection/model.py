"""
CSI-FallNet: Fall Detection Model

Architecture: 1D-CNN → BiLSTM → Attention Pooling → FC Classifier

Input:  CSI amplitude matrix, shape (T=100, C=52) — 2-second sliding window at 50 Hz
Output: Binary {fall, non-fall} with confidence score
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class AttentionPooling(nn.Module):
    """Attention-based pooling over temporal dimension."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, hidden_dim)
        scores = self.attention(x)  # (batch, seq_len, 1)
        weights = F.softmax(scores, dim=1)  # (batch, seq_len, 1)
        weighted = (x * weights).sum(dim=1)  # (batch, hidden_dim)
        return weighted

    def get_raw_weights(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.attention(x)
        return torch.exp(scores) / torch.exp(scores).sum(dim=1, keepdim=True)


class FallDetector(nn.Module):
    """CSI-FallNet: 1D-CNN → BiLSTM → Attention → FC classifier."""

    def __init__(
        self,
        n_subcarriers: int = 52,
        sequence_length: int = 100,
        conv_channels: list[int] = [64, 128, 128],
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.n_subcarriers = n_subcarriers
        self.sequence_length = sequence_length

        # CNN feature extractor
        self.conv1 = nn.Conv1d(n_subcarriers, conv_channels[0], kernel_size=5, padding="same")
        self.bn1 = nn.BatchNorm1d(conv_channels[0])

        self.conv2 = nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm1d(conv_channels[1])

        self.conv3 = nn.Conv1d(conv_channels[1], conv_channels[2], kernel_size=3, padding="same")
        self.bn3 = nn.BatchNorm1d(conv_channels[2])

        self.pool = nn.MaxPool1d(2)

        # BiLSTM
        lstm_input = conv_channels[2]
        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        lstm_output = lstm_hidden * 2  # bidirectional
        self.attention = AttentionPooling(lstm_output)

        # FC classifier
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_output, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)  # {non-fall, fall}

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: CSI amplitude, shape (batch, T=100, C=52)

        Returns:
            logits: shape (batch, 2)
            confidence: softmax probabilities, shape (batch, 2)
        """
        # Transpose to (batch, C, T) for Conv1D
        x = x.permute(0, 2, 1)  # (batch, 52, 100)

        # CNN block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # (batch, 128, 50)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)  # (batch, 128, 25)

        # Transpose to (batch, seq_len, features) for LSTM
        x = x.permute(0, 2, 1)  # (batch, 25, 128)

        # BiLSTM
        lstm_out, _ = self.lstm(x)  # (batch, 25, 512)

        # Attention pooling
        pooled = self.attention(lstm_out)  # (batch, 512)

        # FC classifier
        x = self.dropout(pooled)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)

        confidence = F.softmax(logits, dim=1)

        return logits, confidence

    def predict(self, csi_window: torch.Tensor) -> Tuple[int, float]:
        """Run inference on a single window and return (class, confidence)."""
        self.eval()
        with torch.no_grad():
            if csi_window.dim() == 2:
                csi_window = csi_window.unsqueeze(0)  # add batch dim
            _, confidence = self.forward(csi_window)
            pred_class = torch.argmax(confidence, dim=1).item()
            pred_conf = confidence[0, pred_class].item()
        return pred_class, pred_conf

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class TwoStageConfirmer:
    """Implements two-stage fall confirmation:
    Stage 1: model confidence > threshold
    Stage 2: CSI variance drops below inactivity_threshold for 3 seconds
    """

    def __init__(
        self,
        confidence_threshold: float = 0.85,
        confirmation_window_seconds: float = 3.0,
        inactivity_threshold: float = 0.15,
        sample_rate: float = 50.0,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.confirmation_windows = int(confirmation_window_seconds * sample_rate)
        self.inactivity_threshold = inactivity_threshold
        self._pending_confirmation: Optional[float] = None  # timestamp of initial trigger

    def check(
        self, csi_amplitude: torch.Tensor, fall_confidence: float
    ) -> Optional[bool]:
        """Check two-stage confirmation.

        Returns:
            True if confirmed fall, False if dismissed, None if still pending.
        """
        if self._pending_confirmation is not None:
            # Stage 2: check variance over recent frames
            variance = torch.var(csi_amplitude).item()
            if variance < self.inactivity_threshold:
                self._pending_confirmation = None
                return True
            return None  # still checking

        # Stage 1: initial trigger
        if fall_confidence >= self.confidence_threshold:
            self._pending_confirmation = 0.0
            return None  # pending confirmation

        return False  # no fall detected

    def reset(self) -> None:
        self._pending_confirmation = None
