"""Vital signs estimation — FFT-based respiration + heart rate.

RuView-compatible implementation. Mirrors the algorithm in RuView's
wifi-densepose-vitals crate (breathing.rs, heartrate.rs):
  - Bandpass: 0.1-0.5 Hz (respiration), 0.8-2.0 Hz (heart rate)
  - Top-K subcarrier selection by signal variance
  - FFT peak frequency detection

RuView source reference:
  v2/crates/wifi-densepose-vitals/src/breathing.rs
  v2/crates/wifi-densepose-vitals/src/heartrate.rs
"""

