"""Offline CSI data recording and replay for development and debugging.

Record mode: save all MQTT packets to .csi files with timestamps.
Replay mode: feed recorded packets through the pipeline at original or accelerated speed.
Label mode: annotate fall/non-fall events for training.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

CSI_FILE_EXTENSION = ".csi"
MANIFEST_FILE = "manifest.json"


@dataclass
class CSIFrame:
    """A single recorded CSI frame."""
    timestamp: float
    zone_id: str
    correlation_id: str
    csi_amplitude: list[float]
    csi_phase: list[float]
    label: Optional[str] = None
    rssi: Optional[float] = None
    sequence: Optional[int] = None

    def to_dict(self) -> dict:
        d = {
            "t": self.timestamp,
            "z": self.zone_id,
            "c": self.correlation_id,
            "a": self.csi_amplitude,
            "p": self.csi_phase,
        }
        if self.label is not None:
            d["l"] = self.label
        if self.rssi is not None:
            d["r"] = self.rssi
        if self.sequence is not None:
            d["s"] = self.sequence
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CSIFrame":
        return cls(
            timestamp=d["t"],
            zone_id=d["z"],
            correlation_id=d.get("c", ""),
            csi_amplitude=d["a"],
            csi_phase=d["p"],
            label=d.get("l"),
            rssi=d.get("r"),
            sequence=d.get("s"),
        )


class CSIRecorder:
    """Records CSI packets to files for later replay."""

    def __init__(self, output_dir: str = "data/recordings") -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._current_file: Optional[object] = None
        self._current_path: Optional[Path] = None
        self._frame_count: int = 0
        self._start_time: Optional[float] = None

    def start_recording(self, session_name: Optional[str] = None) -> str:
        """Start a new recording session. Returns the session ID."""
        session_id = session_name or time.strftime("%Y%m%d_%H%M%S")
        self._current_path = self._output_dir / f"{session_id}{CSI_FILE_EXTENSION}"
        self._current_file = open(self._current_path, "w")
        self._frame_count = 0
        self._start_time = time.time()
        logger.info(f"Recording started: {self._current_path}")
        return session_id

    def record_packet(self, packet: dict) -> None:
        """Record a single CSI packet."""
        if self._current_file is None:
            return

        frame = CSIFrame(
            timestamp=packet.get("timestamp", time.time()),
            zone_id=packet.get("zone_id", ""),
            correlation_id=packet.get("correlation_id", ""),
            csi_amplitude=packet.get("csi_amplitude", []).tolist()
                if isinstance(packet.get("csi_amplitude"), np.ndarray)
                else packet.get("csi_amplitude", []),
            csi_phase=packet.get("csi_phase", []).tolist()
                if isinstance(packet.get("csi_phase"), np.ndarray)
                else packet.get("csi_phase", []),
            rssi=packet.get("rssi"),
            sequence=packet.get("sequence"),
        )

        line = json.dumps(frame.to_dict(), separators=(",", ":"))
        self._current_file.write(line + "\n")
        self._frame_count += 1

        if self._frame_count % 1000 == 0:
            self._current_file.flush()
            logger.debug(f"Recorded {self._frame_count} frames")

    def stop_recording(self) -> Optional[str]:
        """Stop recording and save manifest. Returns file path."""
        if self._current_file is None:
            return None

        self._current_file.close()
        duration = time.time() - self._start_time if self._start_time else 0

        manifest = {
            "session_id": self._current_path.stem,
            "file": self._current_path.name,
            "frames": self._frame_count,
            "duration_seconds": round(duration, 2),
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        manifest_path = self._output_dir / MANIFEST_FILE
        existing: list[dict] = []
        if manifest_path.exists():
            try:
                with open(manifest_path, "r") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        existing.append(manifest)
        with open(manifest_path, "w") as f:
            json.dump(existing, f, indent=2)

        path = str(self._current_path)
        logger.info(f"Recording stopped: {self._frame_count} frames, {duration:.1f}s -> {path}")
        self._current_file = None
        self._current_path = None
        return path

    @property
    def is_recording(self) -> bool:
        return self._current_file is not None

    @property
    def frame_count(self) -> int:
        return self._frame_count


class CSIReplayer:
    """Replays recorded CSI data through the pipeline."""

    def __init__(self, callback=None) -> None:
        self._callback = callback
        self._paused = False
        self._stop_requested = False

    def replay_file(
        self,
        file_path: str,
        speed: float = 1.0,
        loop: bool = False,
        zone_override: Optional[str] = None,
    ) -> int:
        """Replay a recorded CSI file.

        Args:
            file_path: Path to .csi recording file.
            speed: Playback speed multiplier (1.0 = real-time, 10.0 = 10x).
            loop: Repeat the recording after completion.
            zone_override: Override zone_id in all packets.

        Returns:
            Number of packets replayed.
        """
        self._stop_requested = False
        total_packets = 0

        while not self._stop_requested:
            packets_replayed = self._replay_once(file_path, speed, zone_override)
            total_packets += packets_replayed
            if not loop:
                break
            logger.info(f"Looping replay: {packets_replayed} packets, total {total_packets}")

        logger.info(f"Replay finished: {total_packets} packets")
        return total_packets

    def _replay_once(self, file_path: str, speed: float, zone_override: Optional[str]) -> int:
        packets_replayed = 0
        prev_timestamp: Optional[float] = None

        with open(file_path, "r") as f:
            for line in f:
                if self._stop_requested:
                    break
                while self._paused and not self._stop_requested:
                    time.sleep(0.1)

                line = line.strip()
                if not line:
                    continue

                try:
                    frame_data = json.loads(line)
                    frame = CSIFrame.from_dict(frame_data)
                except (json.JSONDecodeError, KeyError):
                    continue

                packet = {
                    "timestamp": frame.timestamp,
                    "zone_id": zone_override or frame.zone_id,
                    "correlation_id": frame.correlation_id,
                    "csi_amplitude": frame.csi_amplitude,
                    "csi_phase": frame.csi_phase,
                    "rssi": frame.rssi,
                    "sequence": frame.sequence,
                    "label": frame.label,
                }

                if prev_timestamp is not None and speed > 0:
                    delay = (frame.timestamp - prev_timestamp) / speed
                    if 0 < delay < 60:
                        time.sleep(delay)
                prev_timestamp = frame.timestamp

                if self._callback:
                    self._callback(packet)
                packets_replayed += 1

        return packets_replayed

    def stop(self) -> None:
        self._stop_requested = True

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False


class CSILabeler:
    """Annotate recorded CSI data with event labels for training."""

    LABELS = {
        "f": "fall",
        "s": "sit_down",
        "l": "lie_down",
        "w": "walk",
        "t": "stand_still",
        "b": "breathe",
        "n": "none",
    }

    def __init__(self, recording_path: str) -> None:
        self._path = Path(recording_path)
        if not self._path.exists():
            raise FileNotFoundError(f"Recording not found: {recording_path}")

    def list_unlabeled(self) -> list[int]:
        """Return line numbers of frames without labels."""
        unlabeled: list[int] = []
        with open(self._path, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "l" not in data or data["l"] is None:
                        unlabeled.append(i)
                except json.JSONDecodeError:
                    pass
        return unlabeled

    def label_range(self, start_line: int, end_line: int, label: str) -> int:
        """Apply a label to a range of frames (inclusive)."""
        if label not in self.LABELS.values():
            if label in self.LABELS:
                label = self.LABELS[label]
            else:
                raise ValueError(f"Unknown label: {label}. Use: {list(self.LABELS.values())}")

        lines: list[str] = []
        with open(self._path, "r") as f:
            lines = f.readlines()

        labeled = 0
        for i in range(start_line, min(end_line + 1, len(lines))):
            line = lines[i].strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                data["l"] = label
                lines[i] = json.dumps(data, separators=(",", ":")) + "\n"
                labeled += 1
            except json.JSONDecodeError:
                pass

        with open(self._path, "w") as f:
            f.writelines(lines)

        logger.info(f"Labeled {labeled} frames ({start_line}-{end_line}) as '{label}'")
        return labeled

    def export_labeled(self, output_path: str) -> int:
        """Export all labeled frames as a training-ready numpy archive.

        Returns the number of labeled frames exported.
        """
        amplitudes: list[list[float]] = []
        phases: list[list[float]] = []
        labels: list[str] = []

        with open(self._path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    label = data.get("l")
                    if label is None:
                        continue
                    amplitudes.append(data["a"])
                    phases.append(data["p"])
                    labels.append(label)
                except (json.JSONDecodeError, KeyError):
                    pass

        if not amplitudes:
            logger.warning("No labeled frames found")
            return 0

        label_map = {v: i for i, v in enumerate(self.LABELS.values())}
        label_indices = [label_map.get(l, 0) for l in labels]

        np.savez(
            output_path,
            amplitudes=np.array(amplitudes, dtype=np.float32),
            phases=np.array(phases, dtype=np.float32),
            labels=np.array(label_indices, dtype=np.int64),
            label_names=np.array(list(self.LABELS.values())),
        )
        logger.info(f"Exported {len(amplitudes)} labeled frames to {output_path}")
        return len(amplitudes)
