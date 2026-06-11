import sys
import numpy as np
from typing import List

from .silence_detector import SilentSegment


class PreviewGenerator:

    def __init__(self, width: int = 80, height: int = 20):
        self.width = width
        self.height = height

    def generate(
        self,
        rms_values: np.ndarray,
        silent_segments: List[SilentSegment],
        total_duration_ms: float,
        window_ms: float,
    ) -> str:
        if len(rms_values) == 0:
            return "[No audio data]"

        num_frames = len(rms_values)
        chars_per_frame = self.width / num_frames
        num_rows = self.height

        db_min = -60.0
        db_max = 0.0
        with np.errstate(divide="ignore"):
            db_values = 20.0 * np.log10(rms_values + 1e-10)
        db_clipped = np.clip(db_values, db_min, db_max)
        db_normalized = (db_clipped - db_min) / (db_max - db_min)

        is_silence_frame = np.zeros(num_frames, dtype=bool)
        for seg in silent_segments:
            start_frame = int(seg.start_ms / window_ms)
            end_frame = int(seg.end_ms / window_ms)
            start_frame = max(0, min(start_frame, num_frames - 1))
            end_frame = max(0, min(end_frame, num_frames))
            is_silence_frame[start_frame:end_frame] = True

        lines: List[str] = []
        lines.append("=" * self.width)
        lines.append(f"  Audio Waveform Preview  |  Duration: {total_duration_ms / 1000:.2f}s")
        lines.append("=" * self.width)

        for row in range(num_rows - 1, -1, -1):
            threshold = row / (num_rows - 1) if num_rows > 1 else 0.5
            line_chars: List[str] = []
            for i in range(self.width):
                frame_idx = min(int(i / chars_per_frame), num_frames - 1)
                level = db_normalized[frame_idx]
                is_silent = is_silence_frame[frame_idx]

                if level >= threshold:
                    if is_silent:
                        line_chars.append("░")
                    else:
                        if level > 0.8:
                            line_chars.append("█")
                        elif level > 0.6:
                            line_chars.append("▓")
                        elif level > 0.4:
                            line_chars.append("▒")
                        else:
                            line_chars.append("░")
                else:
                    line_chars.append(" ")
            lines.append("".join(line_chars))

        lines.append("-" * self.width)

        db_scale_str = ""
        for row in range(num_rows - 1, -1, -1):
            db_val = db_min + (db_max - db_min) * row / (num_rows - 1) if num_rows > 1 else db_min
            if row == num_rows - 1:
                db_scale_str = f" {db_val:+.0f} dB"
            elif row == 0:
                db_scale_str += f"{' ' * (self.width - len(db_scale_str) - 6)}{db_val:+.0f} dB"

        if db_scale_str:
            lines.append(db_scale_str)

        lines.append("-" * self.width)
        lines.append(f"  {'█/▓/▒/░'}: signal above threshold  |  {'░'}: silent (below threshold)  |  ' ': noise floor")
        lines.append(f"  Threshold: {db_min:+.0f} dB (bottom) to {db_max:+.0f} dB (top)")
        lines.append("=" * self.width)

        if silent_segments:
            lines.append(f"\n  Detected {len(silent_segments)} silent segment(s):")
            for i, seg in enumerate(silent_segments):
                lines.append(
                    f"    [{i + 1}] {seg.start_sec:.2f}s → {seg.end_sec:.2f}s "
                    f"(duration: {seg.duration_sec:.2f}s)"
                )

        return "\n".join(lines)

    def print_preview(
        self,
        rms_values: np.ndarray,
        silent_segments: List[SilentSegment],
        total_duration_ms: float,
        window_ms: float,
    ) -> None:
        output = self.generate(rms_values, silent_segments, total_duration_ms, window_ms)
        sys.stdout.write(output + "\n")
        sys.stdout.flush()