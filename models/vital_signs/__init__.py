"""Vital signs estimation via wifi_densepose Rust-native extractors.

The VitalsAdapter wraps wifi_densepose's BreathingExtractor and
HeartRateExtractor — stateful Rust objects that accumulate per-frame
amplitude residuals and return VitalEstimate results when sufficient
history is available.

Breathing extractor: 0.1-0.5 Hz bandpass + zero-crossing analysis.
Heart rate extractor: 0.8-2.0 Hz bandpass + autocorrelation peak detection.
"""
