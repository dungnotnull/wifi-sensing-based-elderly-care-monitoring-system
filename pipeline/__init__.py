"""ElderCare real-time inference pipeline.

Signal preprocessing mirrors RuView's wifi-densepose-signal crate:
  - Hampel filter (hampel.rs)
  - Phase sanitization (phase_sanitizer.rs)
  - Butterworth bandpass
  - Per-subcarrier z-score normalization

RuView source reference:
  v2/crates/wifi-densepose-signal/src/
"""

