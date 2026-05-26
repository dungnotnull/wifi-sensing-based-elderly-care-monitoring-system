"""
Realistic CSI data simulator for development and testing without ESP32 hardware.

Generates synthetic CSI amplitude/phase matrices with configurable
activity patterns (normal movement, falls, breathing, idle) and
real-world WiFi environment artifacts:

  - 50/60 Hz mains hum (power line interference)
  - Bursty interference (microwave ovens, Bluetooth, Zigbee)
  - Multi-path fading (furniture movement producing slow drifts)
  - Thermal / RF frontend noise floor
  - Neighbor AP channel switching (periodic RSSI drops)
  - Subcarrier-selective fading (frequency-domain ripple)
  - Clock drift / sampling jitter
"""

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class CSISimulatorConfig:
    n_subcarriers: int = 52
    sample_rate_hz: float = 50.0
    zone_id: str = "zone_test"

    # Base noise floor
    noise_std_amplitude: float = 0.02
    noise_std_phase: float = 0.05

    # Baseline amplitude (immobile person)
    baseline_amplitude: float = 1.0

    # Movement simulation
    movement_freq_range: tuple[float, float] = (0.5, 3.0)
    movement_amplitude: float = 0.3

    # Breathing simulation
    breathing_rate_bpm: float = 15.0
    breathing_amplitude: float = 0.05

    # ---------- Real-world environment noise ----------
    # Mains hum (50 Hz for Vietnam/Europe, 60 Hz for US)
    mains_hum_freq_hz: float = 50.0
    mains_hum_amplitude: float = 0.015
    mains_hum_enabled: bool = True

    # Bursty interference (microwave, Bluetooth, Zigbee crossing 2.4 GHz)
    interference_burst_probability: float = 0.003
    interference_burst_duration_seconds: float = 0.08
    interference_burst_amplitude: float = 0.40
    interference_enabled: bool = True

    # Multi-path fading (slow drifts from furniture, temperature)
    fading_drift_rate_hz: float = 0.02  # very slow cycle
    fading_depth: float = 0.12
    fading_enabled: bool = True

    # Subcarrier-selective fading (frequency-domain ripple)
    frequency_selective_depth: float = 0.08
    frequency_selective_enabled: bool = True

    # Neighbor AP channel switching
    neighbor_ap_switch_interval_seconds: float = 300.0
    neighbor_ap_rssi_drop_db: float = 8.0
    neighbor_ap_switch_duration_seconds: float = 3.0
    neighbor_ap_enabled: bool = True

    # Clock drift / sampling jitter (0 = no jitter, 1 = max)
    jitter_factor: float = 0.02

    # Thermal noise temperature coefficient
    thermal_noise_drift_rate: float = 0.0001

    # RSSI baseline
    rssi_baseline_dbm: float = -40.0
    rssi_std_db: float = 2.0


class CSISimulator:
    """Generates realistic synthetic CSI data streams for testing.

    Models real WiFi environment noise: mains hum, bursty interference,
    multi-path fading, subcarrier-selective fading, neighbor AP switching,
    clock jitter, and thermal noise drift.
    """

    def __init__(self, config: Optional[CSISimulatorConfig] = None) -> None:
        self.config = config or CSISimulatorConfig()
        self._seq: int = 0
        self._start_time: float = time.time()
        self._state: str = "idle"  # idle, moving, falling, breathing
        self._state_start: float = 0.0
        self._rng = np.random.RandomState(42)

        # Interference burst state
        self._interference_active: bool = False
        self._interference_start: float = 0.0
        self._interference_duration: float = 0.0
        self._interference_burst_amp: float = 0.0

        # Neighbor AP state
        self._neighbor_ap_switch_active: bool = False
        self._neighbor_ap_next_switch: float = self.config.neighbor_ap_switch_interval_seconds

        # Thermal noise accumulator
        self._thermal_noise_state: float = 0.0

        # Pre-compute frequency-selective fading profile (time-invariant per subcarrier)
        self._freq_selective_profile = self._rng.uniform(
            1.0 - self.config.frequency_selective_depth,
            1.0 + self.config.frequency_selective_depth,
            self.config.n_subcarriers,
        )

        # Past amplitudes for calculating temporal correlation (used by caller)
        self.last_amplitude: Optional[np.ndarray] = None

    @property
    def elapsed(self) -> float:
        return time.time() - self._start_time

    def set_state(self, state: str) -> None:
        self._state = state
        self._state_start = self.elapsed

    # ------------------------------------------------------------------
    # Environment noise generators
    # ------------------------------------------------------------------

    def _compute_mains_hum(self, t: float) -> np.ndarray:
        cfg = self.config
        if not cfg.mains_hum_enabled:
            return np.zeros(cfg.n_subcarriers)
        # 50 Hz hum couples strongest into center subcarriers,
        # tapering toward edges (capacitive coupling model)
        center = cfg.n_subcarriers / 2
        coupling = np.exp(-0.5 * ((np.arange(cfg.n_subcarriers) - center) / (cfg.n_subcarriers * 0.25)) ** 2)
        hum = cfg.mains_hum_amplitude * coupling * np.sin(2 * math.pi * cfg.mains_hum_freq_hz * t)
        return hum

    def _compute_interference_burst(self, t: float) -> np.ndarray:
        cfg = self.config
        if not cfg.interference_enabled:
            return np.zeros(cfg.n_subcarriers)

        n_sc = cfg.n_subcarriers

        # Check if a new burst should start
        if not self._interference_active:
            if self._rng.random() < cfg.interference_burst_probability:
                self._interference_active = True
                self._interference_start = t
                self._interference_duration = cfg.interference_burst_duration_seconds * self._rng.uniform(0.5, 2.0)
                self._interference_burst_amp = cfg.interference_burst_amplitude * self._rng.uniform(0.5, 1.5)

        if not self._interference_active:
            return np.zeros(n_sc)

        elapsed_since_burst = t - self._interference_start
        if elapsed_since_burst > self._interference_duration:
            self._interference_active = False
            return np.zeros(n_sc)

        # Burst affects a random contiguous band of subcarriers (20-60% of total)
        burst_width = int(n_sc * self._rng.uniform(0.2, 0.6))
        burst_start = self._rng.randint(0, n_sc - burst_width)
        envelope = np.exp(-3.0 * elapsed_since_burst / self._interference_duration)
        burst = np.zeros(n_sc)
        for i in range(burst_start, burst_start + burst_width):
            burst[i] = self._interference_burst_amp * envelope * self._rng.uniform(0.7, 1.0)
        return burst

    def _compute_fading(self, t: float) -> np.ndarray:
        cfg = self.config
        if not cfg.fading_enabled:
            return np.ones(cfg.n_subcarriers)
        n_sc = cfg.n_subcarriers
        # Each subcarrier fades at a slightly different phase (frequency-dependent fading)
        phases = np.linspace(0, 3 * math.pi, n_sc)
        drift = cfg.fading_depth * np.sin(2 * math.pi * cfg.fading_drift_rate_hz * t + phases)
        return 1.0 + drift

    def _compute_neighbor_ap_effect(self, t: float) -> float:
        cfg = self.config
        if not cfg.neighbor_ap_enabled:
            return 0.0

        if not self._neighbor_ap_switch_active and t >= self._neighbor_ap_next_switch:
            self._neighbor_ap_switch_active = True
            self._neighbor_ap_next_switch = t + cfg.neighbor_ap_switch_interval_seconds * self._rng.uniform(0.5, 2.0)

        if not self._neighbor_ap_switch_active:
            return 0.0

        elapsed = t - (self._neighbor_ap_next_switch - cfg.neighbor_ap_switch_interval_seconds)
        if elapsed > cfg.neighbor_ap_switch_duration_seconds:
            self._neighbor_ap_switch_active = False
            return 0.0

        progress = elapsed / cfg.neighbor_ap_switch_duration_seconds
        envelope = math.sin(math.pi * progress)  # smooth rise and fall
        return cfg.neighbor_ap_rssi_drop_db * envelope

    def _compute_thermal_noise_drift(self) -> float:
        """Slow random walk simulating temperature-based noise floor drift."""
        cfg = self.config
        self._thermal_noise_state += self._rng.normal(0, cfg.thermal_noise_drift_rate)
        self._thermal_noise_state = max(-0.01, min(0.01, self._thermal_noise_state))
        return self._thermal_noise_state

    def _compute_jitter(self) -> float:
        """Sample-level timing jitter as a fraction of the sample period."""
        return self._rng.normal(0, self.config.jitter_factor)

    # ------------------------------------------------------------------
    # Core signal generators
    # ------------------------------------------------------------------

    def _generate_amplitude(self) -> np.ndarray:
        cfg = self.config
        n_sc = cfg.n_subcarriers
        t = self.elapsed

        # Base signal
        amp = np.ones(n_sc) * cfg.baseline_amplitude

        # Activity pattern
        if self._state == "moving":
            for i in range(n_sc):
                freq = cfg.movement_freq_range[0] + (
                    cfg.movement_freq_range[1] - cfg.movement_freq_range[0]
                ) * (i / n_sc)
                amp[i] += cfg.movement_amplitude * math.sin(2 * math.pi * freq * t + i * 0.1)

        elif self._state == "falling":
            t_since_fall = self.elapsed - self._state_start
            envelope = 1.0 - 0.6 * math.exp(-t_since_fall * 2.0)
            amp *= envelope

        elif self._state == "breathing":
            breathing_freq = cfg.breathing_rate_bpm / 60.0
            amp += cfg.breathing_amplitude * np.sin(2 * math.pi * breathing_freq * t)

        # Apply multi-path fading (multiplicative)
        amp *= self._compute_fading(t)

        # Apply frequency-selective fading profile
        if cfg.frequency_selective_enabled:
            amp *= self._freq_selective_profile

        # Apply mains hum (additive)
        amp += self._compute_mains_hum(t)

        # Apply bursty interference (additive)
        amp += self._compute_interference_burst(t)

        # Thermal noise drift (additive to noise floor)
        thermal_drift = self._compute_thermal_noise_drift()
        effective_noise = cfg.noise_std_amplitude + thermal_drift
        amp += self._rng.normal(0, effective_noise, n_sc)

        self.last_amplitude = amp.copy()
        return amp

    def _generate_phase(self) -> np.ndarray:
        cfg = self.config
        n_sc = cfg.n_subcarriers
        t = self.elapsed

        phase = np.zeros(n_sc)

        if self._state in ("moving", "falling"):
            for i in range(n_sc):
                phase[i] = 0.3 * math.sin(2 * math.pi * 1.5 * t + i * 0.05)

        elif self._state == "breathing":
            breathing_freq = cfg.breathing_rate_bpm / 60.0
            for i in range(n_sc):
                phase[i] = 0.08 * math.sin(2 * math.pi * breathing_freq * t + i * 0.02)

        # Mains hum also couples into phase
        if cfg.mains_hum_enabled:
            center = n_sc / 2
            coupling = np.exp(-0.5 * ((np.arange(n_sc) - center) / (n_sc * 0.2)) ** 2)
            phase += cfg.mains_hum_amplitude * 0.5 * coupling * np.sin(2 * math.pi * cfg.mains_hum_freq_hz * t + 0.3)

        thermal_drift = self._compute_thermal_noise_drift()
        effective_noise = cfg.noise_std_phase + thermal_drift * 2
        phase += self._rng.normal(0, effective_noise, n_sc)
        return phase

    def _generate_rssi(self) -> float:
        cfg = self.config
        rssi = cfg.rssi_baseline_dbm

        # Neighbor AP channel switch affects RSSI
        rssi -= self._compute_neighbor_ap_effect(self.elapsed)

        # RSSI drops during falling (body blocks signal)
        if self._state == "falling":
            t_since_fall = self.elapsed - self._state_start
            rssi -= 10.0 * math.exp(-t_since_fall * 1.5)

        # General RSSI variation
        rssi += self._rng.normal(0, cfg.rssi_std_db)

        return round(rssi, 1)

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
            "rssi": self._generate_rssi(),
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
            # Apply clock jitter to sample interval
            sleep_time = (1.0 / cfg.sample_rate_hz) * (1.0 + self._compute_jitter())
            sleep_time = max(0.001, sleep_time)
            time.sleep(sleep_time)


if __name__ == "__main__":
    sim = CSISimulator()
    print("Generating 5 realistic synthetic CSI packets...")
    for i, pkt in enumerate(sim.stream(duration_seconds=0.1)):
        amp = np.array(pkt["csi_amplitude"])
        phase = np.array(pkt["csi_phase"])
        print(
            f"  [{pkt['zone_id']}] seq={pkt['sequence_number']:04d}  "
            f"rssi={pkt['rssi']:.1f} dBm  "
            f"amp_mean={amp.mean():.4f}  amp_std={amp.std():.4f}  "
            f"phase_range=[{phase.min():.3f}, {phase.max():.3f}]"
        )
        if i >= 5:
            break
