import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SilentSegment:
    start_ms: float
    end_ms: float
    duration_ms: float

    @property
    def start_sec(self) -> float:
        return self.start_ms / 1000.0

    @property
    def end_sec(self) -> float:
        return self.end_ms / 1000.0

    @property
    def duration_sec(self) -> float:
        return self.duration_ms / 1000.0

    def __repr__(self) -> str:
        return (
            f"SilentSegment(start={self.start_sec:.3f}s, "
            f"end={self.end_sec:.3f}s, "
            f"duration={self.duration_sec:.3f}s)"
        )


@dataclass
class NonSilentSegment:
    start_ms: float
    end_ms: float
    duration_ms: float

    @property
    def start_sec(self) -> float:
        return self.start_ms / 1000.0

    @property
    def end_sec(self) -> float:
        return self.end_ms / 1000.0

    @property
    def duration_sec(self) -> float:
        return self.duration_ms / 1000.0

    def __repr__(self) -> str:
        return (
            f"NonSilentSegment(start={self.start_sec:.3f}s, "
            f"end={self.end_sec:.3f}s, "
            f"duration={self.duration_sec:.3f}s)"
        )


class SilenceDetector:

    def __init__(
        self,
        threshold_db: float = -40.0,
        min_silence_ms: float = 500.0,
        window_ms: float = 10.0,
    ):
        self.threshold_db = threshold_db
        self.min_silence_ms = min_silence_ms
        self.window_ms = window_ms

    def detect(
        self,
        samples: np.ndarray,
        sample_rate: int,
    ) -> List[SilentSegment]:
        window_samples = int(self.window_ms / 1000.0 * sample_rate)
        if window_samples < 1:
            window_samples = 1

        num_frames = len(samples) // window_samples
        if num_frames == 0:
            return []

        rms_values = np.zeros(num_frames)
        for i in range(num_frames):
            start = i * window_samples
            end = start + window_samples
            frame = samples[start:end]
            rms = np.sqrt(np.mean(frame.astype(np.float64) ** 2))
            rms_values[i] = rms

        with np.errstate(divide="ignore"):
            db_values = 20.0 * np.log10(rms_values + 1e-10)

        is_silence = db_values < self.threshold_db

        min_frames = int(self.min_silence_ms / self.window_ms)
        if min_frames < 1:
            min_frames = 1

        segments: List[SilentSegment] = []
        in_silence = False
        silence_start_frame = 0

        for i in range(len(is_silence)):
            if is_silence[i] and not in_silence:
                in_silence = True
                silence_start_frame = i
            elif not is_silence[i] and in_silence:
                in_silence = False
                silence_end_frame = i
                if silence_end_frame - silence_start_frame >= min_frames:
                    start_ms = silence_start_frame * self.window_ms
                    end_ms = silence_end_frame * self.window_ms
                    segments.append(
                        SilentSegment(
                            start_ms=start_ms,
                            end_ms=end_ms,
                            duration_ms=end_ms - start_ms,
                        )
                    )

        if in_silence:
            silence_end_frame = num_frames
            if silence_end_frame - silence_start_frame >= min_frames:
                start_ms = silence_start_frame * self.window_ms
                end_ms = silence_end_frame * self.window_ms
                segments.append(
                    SilentSegment(
                        start_ms=start_ms,
                        end_ms=end_ms,
                        duration_ms=end_ms - start_ms,
                    )
                )

        return segments

    def get_non_silent_segments(
        self,
        silent_segments: List[SilentSegment],
        total_duration_ms: float,
    ) -> List[NonSilentSegment]:
        non_silent: List[NonSilentSegment] = []

        cursor = 0.0
        for seg in silent_segments:
            if seg.start_ms > cursor:
                non_silent.append(
                    NonSilentSegment(
                        start_ms=cursor,
                        end_ms=seg.start_ms,
                        duration_ms=seg.start_ms - cursor,
                    )
                )
            cursor = seg.end_ms

        if cursor < total_duration_ms:
            non_silent.append(
                NonSilentSegment(
                    start_ms=cursor,
                    end_ms=total_duration_ms,
                    duration_ms=total_duration_ms - cursor,
                )
            )

        return non_silent

    def get_rms_array(
        self,
        samples: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        window_samples = int(self.window_ms / 1000.0 * sample_rate)
        if window_samples < 1:
            window_samples = 1

        num_frames = len(samples) // window_samples
        if num_frames == 0:
            return np.array([])

        rms_values = np.zeros(num_frames)
        for i in range(num_frames):
            start = i * window_samples
            end = start + window_samples
            frame = samples[start:end]
            rms_values[i] = np.sqrt(np.mean(frame.astype(np.float64) ** 2))

        return rms_values