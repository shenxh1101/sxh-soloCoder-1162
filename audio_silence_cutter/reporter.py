import os
import json
from typing import List, Optional
from dataclasses import dataclass, asdict
from datetime import timedelta


@dataclass
class SegmentReport:
    index: int
    filename: str
    start_sec: float
    end_sec: float
    duration_sec: float
    start_timecode: str
    end_timecode: str


class Reporter:

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir

    @staticmethod
    def _format_timecode(seconds: float) -> str:
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        millis = int((seconds - total_seconds) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    def generate_report(
        self,
        results: List[tuple],
        source_file: str,
        threshold_db: float,
        min_silence_ms: float,
        buffer_before_ms: float,
        buffer_after_ms: float,
        smart_merge_enabled: bool = False,
        min_segment_ms: Optional[float] = None,
    ) -> List[SegmentReport]:
        reports: List[SegmentReport] = []
        for i, (filepath, start_ms, end_ms) in enumerate(results):
            start_sec = start_ms / 1000.0
            end_sec = end_ms / 1000.0
            reports.append(
                SegmentReport(
                    index=i + 1,
                    filename=os.path.basename(filepath),
                    start_sec=start_sec,
                    end_sec=end_sec,
                    duration_sec=end_sec - start_sec,
                    start_timecode=self._format_timecode(start_sec),
                    end_timecode=self._format_timecode(end_sec),
                )
            )
        return reports

    def print_report(
        self,
        reports: List[SegmentReport],
        source_file: str,
        threshold_db: float,
        min_silence_ms: float,
        buffer_before_ms: float,
        buffer_after_ms: float,
        smart_merge_enabled: bool = False,
        min_segment_ms: Optional[float] = None,
    ) -> None:
        print()
        print("=" * 72)
        print("  AUDIO SILENCE CUTTER — SPLIT REPORT")
        print("=" * 72)
        print(f"  Source file:       {source_file}")
        print(f"  Threshold:         {threshold_db:.1f} dBFS")
        print(f"  Min silence:       {min_silence_ms:.0f} ms")
        print(f"  Buffer before:     {buffer_before_ms:.0f} ms")
        print(f"  Buffer after:      {buffer_after_ms:.0f} ms")
        if smart_merge_enabled and min_segment_ms:
            print(f"  Smart merge:       enabled (min segment: {min_segment_ms:.0f} ms)")
        print(f"  Total segments:    {len(reports)}")
        print("-" * 72)
        print(
            f"  {'#':>3}  {'Filename':<30} {'Start':>12} {'End':>12} {'Duration':>10}"
        )
        print("-" * 72)
        for r in reports:
            print(
                f"  {r.index:>3}  {r.filename:<30} "
                f"{r.start_timecode:>12} {r.end_timecode:>12} {r.duration_sec:>9.2f}s"
            )
        print("-" * 72)

        total_duration = sum(r.duration_sec for r in reports)
        print(f"  Total duration:    {total_duration:.2f}s")
        print("=" * 72)
        print()

    def save_report_json(
        self,
        reports: List[SegmentReport],
        source_file: str,
        threshold_db: float,
        min_silence_ms: float,
        buffer_before_ms: float,
        buffer_after_ms: float,
        smart_merge_enabled: bool = False,
        min_segment_ms: Optional[float] = None,
    ) -> str:
        metadata = {
            "source_file": source_file,
            "threshold_db": threshold_db,
            "min_silence_ms": min_silence_ms,
            "buffer_before_ms": buffer_before_ms,
            "buffer_after_ms": buffer_after_ms,
            "smart_merge_enabled": smart_merge_enabled,
            "min_segment_ms": min_segment_ms,
            "total_segments": len(reports),
        }
        data = {
            "metadata": metadata,
            "segments": [asdict(r) for r in reports],
        }

        base_name = os.path.splitext(os.path.basename(source_file))[0]
        json_path = os.path.join(
            self.output_dir or os.path.dirname(source_file) or ".",
            f"{base_name}_report.json",
        )
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"  Report saved to: {json_path}")
        return json_path