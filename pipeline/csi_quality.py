"""
CSI Quality Metrics and Validation.

Validates incoming CSI packet quality by computing signal-to-noise ratio,
packet loss rate, null subcarrier detection, and RSSI. Produces a
QualityReport with an overall 0-1 quality score.
"""

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Minimum thresholds for acceptable quality
MIN_SNR_DB = 5.0
MAX_PACKET_LOSS_RATE = 0.1
MAX_NULL_SUBCARRIERS = 5
MIN_RSSI_DBM = -80.0
MIN_QUALITY_SCORE = 0.4

# Number of packets for rolling average
ROLLING_WINDOW = 100

N_SUBCARRIERS = 52


@dataclass
class QualityReport:
    """Quality assessment for a single CSI packet."""

    snr_db: float
    packet_loss_rate: float
    null_subcarriers: list[int]
    rssi_dbm: float
    quality_score: float
    timestamp: float
    zone_id: str

    def is_acceptable(self) -> bool:
        """Return True if quality is above minimum thresholds."""
        return (
            self.quality_score >= MIN_QUALITY_SCORE
            and self.snr_db >= MIN_SNR_DB
            and self.packet_loss_rate <= MAX_PACKET_LOSS_RATE
        )


class CSIQualityChecker:
    """Validates CSI packet quality and tracks rolling statistics per zone.

    For each incoming packet, computes SNR, packet loss (via sequence number
    gaps), null subcarrier count, and RSSI. Produces a QualityReport with
    an overall quality score in [0, 1].
    """

    def __init__(self) -> None:
        # Per-zone sequence tracking for packet loss detection
        self._last_seq: dict[str, int] = {}
        self._seq_gaps: dict[str, deque[int]] = {}
        # Per-zone rolling quality history
        self._quality_history: dict[str, deque[float]] = {}

    def check(self, packet: dict) -> QualityReport:
        """Validate an incoming CSI packet and return a QualityReport.

        Expects the packet dict to contain:
          - csi_amplitude: list[float] (52 subcarriers)
          - rssi: float (optional, defaults to 0.0)
          - sequence_number: int
          - zone_id: str
          - timestamp: float
        """
        zone_id: str = packet.get("zone_id", "unknown")
        amplitude: list[float] = packet.get("csi_amplitude", [])
        rssi: float = float(packet.get("rssi", 0.0))
        seq: int = int(packet.get("sequence_number", 0))
        ts: float = float(packet.get("timestamp", time.time()))

        snr_db = self._compute_snr(amplitude)
        packet_loss = self._compute_packet_loss(zone_id, seq)
        null_subs = self._detect_null_subcarriers(amplitude)

        # Reset last seq so next packet computes a fresh gap
        self._last_seq[zone_id] = seq

        quality_score = self._compute_quality_score(snr_db, packet_loss, null_subs, rssi)

        report = QualityReport(
            snr_db=snr_db,
            packet_loss_rate=packet_loss,
            null_subcarriers=null_subs,
            rssi_dbm=rssi,
            quality_score=quality_score,
            timestamp=ts,
            zone_id=zone_id,
        )

        # Track rolling history
        history = self._quality_history.setdefault(zone_id, deque(maxlen=ROLLING_WINDOW))
        history.append(quality_score)

        return report

    def get_rolling_avg(self, zone_id: str) -> Optional[float]:
        """Return the average quality score over the last ROLLING_WINDOW packets."""
        history = self._quality_history.get(zone_id)
        if not history:
            return None
        return float(np.mean(list(history)))

    # -- Internal computation methods --

    @staticmethod
    def _compute_snr(amplitude: list[float]) -> float:
        """Compute signal-to-noise ratio in dB.

        SNR = 20 * log10(median_amplitude / noise_std)
        where noise_std is the standard deviation of the noise floor
        (the lowest 25% of amplitude values).
        """
        if not amplitude or len(amplitude) < 4:
            return 0.0
        amp = np.array(amplitude, dtype=np.float64)
        median_amp = float(np.median(np.abs(amp)))
        # Noise floor: bottom 25% of absolute amplitudes
        sorted_abs = np.sort(np.abs(amp))
        quarter = max(1, len(sorted_abs) // 4)
        noise_floor = sorted_abs[:quarter]
        noise_std = float(np.std(noise_floor))
        if noise_std < 1e-9:
            # If noise is essentially zero, SNR is very high
            return 60.0 if median_amp > 1e-9 else 0.0
        snr = 20.0 * math.log10(median_amp / noise_std)
        return max(0.0, min(snr, 60.0))

    def _compute_packet_loss(self, zone_id: str, seq: int) -> float:
        """Compute packet loss rate from sequence number gaps.

        Tracks the gap between consecutive sequence numbers and
        returns the average loss over the rolling window.
        """
        gaps = self._seq_gaps.setdefault(zone_id, deque(maxlen=ROLLING_WINDOW))
        last_seq = self._last_seq.get(zone_id)

        if last_seq is None:
            # First packet for this zone, no gap to compute
            return 0.0

        if seq > last_seq:
            gap = seq - last_seq - 1
            gaps.append(gap)
        elif seq < last_seq:
            # Sequence wrapped or out-of-order; assume no loss
            gaps.append(0)

        if not gaps:
            return 0.0
        avg_gap = float(np.mean(list(gaps)))
        # Convert to rate: gap / (gap + 1) = lost / total expected
        return min(avg_gap / (avg_gap + 1.0), 1.0) if avg_gap >= 0 else 0.0

    @staticmethod
    def _detect_null_subcarriers(amplitude: list[float]) -> list[int]:
        """Detect subcarriers stuck at zero or near-zero amplitude.

        A subcarrier is considered null if its absolute amplitude
        is below 1% of the median amplitude across all subcarriers.
        """
        if not amplitude:
            return []
        amp = np.array(amplitude, dtype=np.float64)
        median_amp = float(np.median(np.abs(amp)))
        threshold = max(median_amp * 0.01, 1e-9)
        null_indices = list(np.where(np.abs(amp) < threshold)[0])
        return null_indices

    @staticmethod
    def _compute_quality_score(
        snr_db: float,
        packet_loss: float,
        null_subcarriers: list[int],
        rssi_dbm: float,
    ) -> float:
        """Compute overall 0-1 quality score from individual metrics.

        Weighted combination:
          - SNR: 40% weight (0-30 dB mapped to 0-1)
          - Packet loss: 30% weight (inverted, 0 loss = 1.0)
          - Null subcarriers: 15% weight (inverted, 0 nulls = 1.0)
          - RSSI: 15% weight (-100 to -30 dBm mapped to 0-1)
        """
        # SNR score: 0 dB -> 0.0, 30 dB -> 1.0
        snr_score = min(max(snr_db / 30.0, 0.0), 1.0)

        # Packet loss score: 0% loss -> 1.0, 100% loss -> 0.0
        loss_score = 1.0 - packet_loss

        # Null subcarrier score: 0 nulls -> 1.0, 52 nulls -> 0.0
        null_score = 1.0 - (len(null_subcarriers) / N_SUBCARRIERS)

        # RSSI score: -100 dBm -> 0.0, -30 dBm -> 1.0
        rssi_score = min(max((rssi_dbm + 100.0) / 70.0, 0.0), 1.0)

        score = (
            0.40 * snr_score
            + 0.30 * loss_score
            + 0.15 * null_score
            + 0.15 * rssi_score
        )
        return round(min(max(score, 0.0), 1.0), 3)
