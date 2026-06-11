from typing import List

from .silence_detector import NonSilentSegment


class SmartMerger:

    def __init__(self, min_segment_ms: float = 1000.0):
        self.min_segment_ms = min_segment_ms

    def merge(
        self, segments: List[NonSilentSegment]
    ) -> List[NonSilentSegment]:
        if len(segments) <= 1:
            return segments

        segments = list(segments)

        while True:
            short_indices = [
                i
                for i, seg in enumerate(segments)
                if seg.duration_ms < self.min_segment_ms
            ]
            if not short_indices:
                break

            merged_any = False
            for idx in sorted(short_indices, reverse=True):
                if idx >= len(segments):
                    continue
                seg = segments[idx]
                if seg.duration_ms >= self.min_segment_ms:
                    continue

                candidates = []
                if idx > 0:
                    prev_seg = segments[idx - 1]
                    candidates.append((idx - 1, prev_seg.duration_ms))
                if idx < len(segments) - 1:
                    next_seg = segments[idx + 1]
                    candidates.append((idx + 1, next_seg.duration_ms))

                if not candidates:
                    break

                candidates.sort(key=lambda x: -x[1])
                target_idx, _ = candidates[0]

                left_idx = min(idx, target_idx)
                right_idx = max(idx, target_idx)

                merged = NonSilentSegment(
                    start_ms=segments[left_idx].start_ms,
                    end_ms=segments[right_idx].end_ms,
                    duration_ms=segments[right_idx].end_ms
                    - segments[left_idx].start_ms,
                )

                segments[left_idx : right_idx + 1] = [merged]
                merged_any = True
                break

            if not merged_any:
                break

        return segments