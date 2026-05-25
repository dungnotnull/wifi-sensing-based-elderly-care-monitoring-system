"""Vital signs estimation with automatic backend selection.

The VitalsAdapter uses wifi_densepose Rust-native extractors when
available, and falls back to pure-Python scipy/numpy extractors
(python_fallback module) when wifi_densepose cannot be imported.

Breathing extractor: 0.1-0.5 Hz bandpass + zero-crossing analysis.
Heart rate extractor: 0.8-2.0 Hz bandpass + autocorrelation peak detection.
"""
