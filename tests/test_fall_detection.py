"""Tests for FallDetector model."""
import pytest
import torch
from models.fall_detection.model import FallDetector, TwoStageConfirmer


class TestFallDetector:
    def test_model_output_shape(self) -> None:
        model = FallDetector(n_subcarriers=52, sequence_length=100)
        x = torch.randn(4, 100, 52)  # batch=4, T=100, C=52
        logits, confidence = model(x)
        assert logits.shape == (4, 2)
        assert confidence.shape == (4, 2)
        assert torch.allclose(confidence.sum(dim=1), torch.ones(4))

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_model_batch_sizes(self, batch_size: int) -> None:
        model = FallDetector()
        x = torch.randn(batch_size, 100, 52)
        logits, confidence = model(x)
        assert logits.shape[0] == batch_size

    def test_model_parameter_count(self) -> None:
        model = FallDetector()
        n_params = model.get_parameter_count()
        # Should be under 5M for RPi5 deployment
        assert n_params < 5_000_000

    def test_predict_single_window(self) -> None:
        model = FallDetector()
        x = torch.randn(100, 52)
        pred_class, conf = model.predict(x)
        assert pred_class in (0, 1)
        assert 0.0 <= conf <= 1.0


class TestTwoStageConfirmer:
    def test_initial_trigger_below_threshold(self) -> None:
        confirmer = TwoStageConfirmer(confidence_threshold=0.85)
        amp = torch.ones(50, 52)  # high variance = not inactive
        result = confirmer.check(amp, fall_confidence=0.5)
        assert result is False

    def test_initial_trigger_above_threshold(self) -> None:
        confirmer = TwoStageConfirmer(confidence_threshold=0.85)
        amp = torch.ones(50, 52) * 2.0
        result = confirmer.check(amp, fall_confidence=0.90)
        assert result is None  # pending confirmation

    def test_stage_two_confirms_fall(self) -> None:
        confirmer = TwoStageConfirmer(confidence_threshold=0.85, inactivity_threshold=0.15)
        # Stage 1: trigger
        amp = torch.ones(50, 52)
        result = confirmer.check(amp, fall_confidence=0.90)
        assert result is None  # pending

        # Stage 2: low variance = inactivity = fall confirmed
        amp_low = torch.zeros(50, 52)  # variance = 0
        result = confirmer.check(amp_low, fall_confidence=0.50)
        assert result is True


class TestFallConfirmationEvent:
    def test_event_creation(self) -> None:
        from pipeline.inference_engine import FallConfirmationEvent
        event = FallConfirmationEvent(
            zone_id="zone_bedroom",
            timestamp=1000.0,
            confidence=0.95,
        )
        assert event.zone_id == "zone_bedroom"
        assert event.timestamp == 1000.0
        assert event.confidence == 0.95


class TestFallDetectionWorkerIntegration:
    def test_worker_uses_two_stage_confirmer(self) -> None:
        import multiprocessing as mp
        from pipeline.inference_engine import FallDetectionWorker, FallConfirmationEvent

        input_q: mp.Queue = mp.Queue()
        output_q: mp.Queue = mp.Queue()
        fall_event_q: mp.Queue = mp.Queue()
        stop = mp.Event()

        config = {
            "sample_rate": 50.0,
            "window_size": 100,
            "n_subcarriers": 52,
            "confidence_threshold": 0.85,
            "confirmation_window_seconds": 3.0,
            "inactivity_threshold": 0.15,
        }

        worker = FallDetectionWorker(
            name="FallDetection_zone_test",
            zone_id="zone_test",
            input_queue=input_q,
            output_queue=output_q,
            stop_event=stop,
            config=config,
            fall_event_queue=fall_event_q,
        )

        # Verify worker has a FallDetector instance
        assert worker._detector is not None
        assert worker._confirmer is not None
        assert worker._fall_event_queue is fall_event_q
