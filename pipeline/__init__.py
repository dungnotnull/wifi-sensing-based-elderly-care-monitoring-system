"""ElderCare real-time inference pipeline.

Signal preprocessing is a faithful Python (scipy) reimplementation
of RuView's Rust signal chain:
  - Hampel filter (outlier removal)
  - Phase sanitization (unwrapping + linear detrend)
  - Butterworth bandpass
  - Per-subcarrier z-score normalization

The inference engine orchestrates per-zone multiprocessing workers
that consume preprocessed CSI and run fall detection, vital signs
estimation (via wifi_densepose Rust extractors), sleep monitoring,
and activity tracking.
"""
