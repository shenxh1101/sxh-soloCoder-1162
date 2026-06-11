import os
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

from .silence_detector import NonSilentSegment


class AudioSplitter:

    def __init__(
        self,
        buffer_before_ms: float = 200.0,
        buffer_after_ms: float = 200.0,
        output_dir: Optional[str] = None,
    ):
        self.buffer_before_ms = buffer_before_ms
        self.buffer_after_ms = buffer_after_ms
        self.output_dir = output_dir

    def apply_buffer(
        self,
        segment: NonSilentSegment,
        total_duration_ms: float,
    ) -> Tuple[float, float]:
        start = max(0.0, segment.start_ms - self.buffer_before_ms)
        end = min(total_duration_ms, segment.end_ms + self.buffer_after_ms)
        return start, end

    def get_buffered_segments(
        self,
        non_silent_segments: List[NonSilentSegment],
        total_duration_ms: float,
    ) -> List[NonSilentSegment]:
        buffered: List[NonSilentSegment] = []
        for seg in non_silent_segments:
            start_ms, end_ms = self.apply_buffer(seg, total_duration_ms)
            buffered.append(
                NonSilentSegment(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    duration_ms=end_ms - start_ms,
                )
            )
        return buffered

    def split_audio(
        self,
        audio_segment,
        segments: List[NonSilentSegment],
        base_name: str,
        output_format: str,
    ) -> List[Tuple[str, float, float]]:
        os.makedirs(self.output_dir, exist_ok=True)

        results: List[Tuple[str, float, float]] = []
        for i, seg in enumerate(segments):
            chunk = audio_segment[seg.start_ms:seg.end_ms]
            filename = f"{base_name}_part{i + 1:03d}.{output_format}"
            filepath = os.path.join(self.output_dir, filename)
            chunk.export(filepath, format=output_format)
            results.append((filepath, seg.start_ms, seg.end_ms))

        return results

    def split_audio_batch(
        self,
        audio_segment,
        segments: List[NonSilentSegment],
        base_name: str,
        output_format: str,
        start_index: int = 1,
    ) -> Tuple[List[Tuple[str, float, float]], int]:
        os.makedirs(self.output_dir, exist_ok=True)

        results: List[Tuple[str, float, float]] = []
        idx = start_index
        for seg in segments:
            chunk = audio_segment[seg.start_ms:seg.end_ms]
            filename = f"{base_name}_part{idx:03d}.{output_format}"
            filepath = os.path.join(self.output_dir, filename)
            chunk.export(filepath, format=output_format)
            results.append((filepath, seg.start_ms, seg.end_ms))
            idx += 1

        return results, idx