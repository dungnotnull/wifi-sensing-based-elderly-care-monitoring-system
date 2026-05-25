"""
CSI data simulator for development and testing without ESP32 hardware.

Generates realistic synthetic CSI amplitude/phase matrices with
configurable activity patterns (normal movement, falls, breathing, idle).
"""

import math
import random
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CSISimulatorConfig:
    n_subcarriers: int = 52
    sample_rate_hz: float = 50.0
    zone_id: str = "zone_test"

    # Noise parameters
    noise_std_amplitude: float = 0.02
    noise_std_phase: float = 0.05

    # Baseline amplitude (immobile person)
    baseline_amplitude: float = 1.0

    # Movement simulation
    movement_freq_range: tuple[float, float] = (0.5, 3.0)
    movement_amplitude: float = 0.3

    # Breathing simulation
    breathing_rate_bpm: float = 15.0  # breaths per minute
    breathing_amplitude: float = 0.05


class CSISimulator:
    """Generates synthetic CSI data streams for testing."""

    def __init__(self, config: Optional[CSISimulatorConfig] = None) -> None:
        self.config = config or CSISimulatorConfig()
        self._seq: int = 0
        self._start_time: float = time.time()
        self._state: str = "idle"  # idle, moving, falling, breathing
        self._state_start: float = 0.0
        self._rng = np.random.RandomState(42)

    @property
    def elapsed(self) -> float:
        return time.time() - self._start_time

    def set_state(self, state: str) -> None:
        self._state = state
        self._state_start = self.elapsed

    def _generate_amplitude(self) -> np.ndarray:
        cfg = self.config
        amp = np.ones(cfg.n_subcarriers) * cfg.baseline_amplitude

        if self._state == "moving":
            # Time-varying sinusoidal movement across subcarriers
            t = self.elapsed
            for i in range(cfg.n_subcarriers):
                freq = cfg.movement_freq_range[0] + (
                    cfg.movement_freq_range[1] - cfg.movement_freq_range[0]
                ) * (i / cfg.n_subcarriers)
                amp[i] += cfg.movement_amplitude * math.sin(2 * math.pi * freq * t + i * 0.1)

        elif self._state == "falling":
            # Sharp amplitude drop then recovery pattern
            t_since_fall = self.elapsed - self._state_start
            envelope = 1.0 - 0.6 * math.exp(-t_since_fall * 2.0)
            amp *= envelope

        elif self._state == "breathing":
            t = self.elapsed
            breathing_freq = cfg.breathing_rate_bpm / 60.0
            amp += cfg.breathing_amplitude * np.sin(2 * math.pi * breathing_freq * t)

        # Per-subcarrier noise
        amp += self._rng.normal(0, cfg.noise_std_amplitude, cfg.n_subcarriers)
        return amp

    def _generate_phase(self) -> np.ndarray:
        cfg = self.config
        phase = np.zeros(cfg.n_subcarriers)

        if self._state in ("moving", "falling"):
            t = self.elapsed
            for i in range(cfg.n_subcarriers):
                phase[i] = 0.3 * math.sin(2 * math.pi * 1.5 * t + i * 0.05)

        elif self._state == "breathing":
            t = self.elapsed
            breathing_freq = cfg.breathing_rate_bpm / 60.0
            for i in range(cfg.n_subcarriers):
                phase[i] = 0.08 * math.sin(2 * math.pi * breathing_freq * t + i * 0.02)

        phase += self._rng.normal(0, cfg.noise_std_phase, cfg.n_subcarriers)
        return phase

    def generate_packet(self) -> dict:
        """Generate a single CSI packet matching the format from ESP32 firmware."""
        timestamp = time.time()
        self._seq += 1
        packet = {
            "zone_id": self.config.zone_id,
            "timestamp": timestamp,
            "sequence_number": self._seq,
            "csi_amplitude": self._generate_amplitude().tolist(),
            "csi_phase": self._generate_phase().tolist(),
            "rssi": -40.0 + self._rng.normal(0, 2.0),
        }
        return packet

    def stream(self, duration_seconds: float = 0.0):
        """Generator yielding CSI packets at the configured sample rate.

        If duration_seconds=0, streams indefinitely.
        """
        cfg = self.config
        start = time.time()
        while duration_seconds == 0 or (time.time() - start) < duration_seconds:
            yield self.generate_packet()
            time.sleep(1.0 / cfg.sample_rate_hz)


if __name__ == "__main__":
    sim = CSISimulator()
    print("Generating 5 synthetic CSI packets...")
    for i, pkt in enumerate(sim.stream(duration_seconds=0.1)):
        print(
            f"  [{pkt['zone_id']}] seq={pkt['sequence_number']:04d}  "
            f"rssi={pkt['rssi']:.1f}  amp_mean={np.mean(pkt['csi_amplitude']):.3f}"
        )
        if i >= 5:
            break
