import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from .silence_detector import SilenceDetector
from .audio_splitter import AudioSplitter
from .smart_merger import SmartMerger
from .preview import PreviewGenerator
from .envelope import EnvelopeExporter
from .reporter import Reporter


def _load_audio(filepath: str):
    try:
        from pydub import AudioSegment
    except ImportError:
        print(
            "Error: pydub is required. Install it with: pip install pydub",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        audio = AudioSegment.from_file(filepath)
    except Exception as e:
        print(f"Error loading audio file '{filepath}': {e}", file=sys.stderr)
        sys.exit(1)

    return audio


def _get_format(filepath: str) -> str:
    ext = Path(filepath).suffix.lower().lstrip(".")
    format_map = {
        "mp3": "mp3",
        "wav": "wav",
        "flac": "flac",
        "ogg": "ogg",
        "m4a": "mp4",
        "aac": "mp4",
    }
    return format_map.get(ext, ext)


def _audio_to_mono_samples(audio) -> np.ndarray:
    if audio.channels > 1:
        audio = audio.set_channels(1)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float64)
    max_val = float(2 ** (audio.sample_width * 8 - 1))
    return samples / max_val


def process_single_file(
    filepath: str,
    threshold_db: float,
    min_silence_ms: float,
    buffer_before_ms: float,
    buffer_after_ms: float,
    output_dir: str,
    output_format: Optional[str],
    preview: bool,
    export_envelope: bool,
    smart_merge: bool,
    min_segment_ms: float,
    save_report: bool,
    no_report: bool = False,
    preview_only: bool = False,
) -> int:
    audio = _load_audio(filepath)
    samples = _audio_to_mono_samples(audio)
    sample_rate = audio.frame_rate
    total_duration_ms = len(audio)

    fmt = output_format or _get_format(filepath)

    detector = SilenceDetector(
        threshold_db=threshold_db,
        min_silence_ms=min_silence_ms,
    )

    silent_segments = detector.detect(samples, sample_rate)
    non_silent = detector.get_non_silent_segments(silent_segments, total_duration_ms)

    if preview:
        rms_values = detector.get_rms_array(samples, sample_rate)
        preview_gen = PreviewGenerator(width=80, height=20)
        preview_gen.print_preview(
            rms_values, silent_segments, total_duration_ms, detector.window_ms
        )

    if preview_only:
        return 0

    splitter = AudioSplitter(
        buffer_before_ms=buffer_before_ms,
        buffer_after_ms=buffer_after_ms,
        output_dir=output_dir,
    )

    segments = splitter.get_buffered_segments(non_silent, total_duration_ms)

    if smart_merge:
        merger = SmartMerger(min_segment_ms=min_segment_ms)
        segments = merger.merge(segments)

    if not segments:
        print(f"Warning: No non-silent segments found in '{filepath}'.", file=sys.stderr)
        return 0

    base_name = Path(filepath).stem
    results = splitter.split_audio(audio, segments, base_name, fmt)

    if export_envelope:
        try:
            envelope_exporter = EnvelopeExporter()
            for path, start_ms, end_ms in results:
                env_path = Path(path).with_suffix(".png")
                envelope_exporter.export(
                    samples,
                    sample_rate,
                    str(env_path),
                    start_ms=start_ms,
                    end_ms=end_ms,
                    title=f"{Path(path).stem} — Amplitude Envelope",
                )
        except ImportError as e:
            print(f"Warning: {e}", file=sys.stderr)

    reporter = Reporter(output_dir=output_dir)
    reports = reporter.generate_report(
        results, filepath, threshold_db, min_silence_ms,
        buffer_before_ms, buffer_after_ms,
        smart_merge_enabled=smart_merge,
        min_segment_ms=min_segment_ms if smart_merge else None,
    )

    if not no_report:
        reporter.print_report(
            reports, filepath, threshold_db, min_silence_ms,
            buffer_before_ms, buffer_after_ms,
            smart_merge_enabled=smart_merge,
            min_segment_ms=min_segment_ms if smart_merge else None,
        )

    if save_report:
        reporter.save_report_json(
            reports, filepath, threshold_db, min_silence_ms,
            buffer_before_ms, buffer_after_ms,
            smart_merge_enabled=smart_merge,
            min_segment_ms=min_segment_ms if smart_merge else None,
        )

    print(f"  Done: {len(results)} segment(s) written to '{output_dir}'")
    return len(results)


def process_batch(
    directory: str,
    threshold_db: float,
    min_silence_ms: float,
    buffer_before_ms: float,
    buffer_after_ms: float,
    output_dir: str,
    output_format: Optional[str],
    preview: bool,
    export_envelope: bool,
    smart_merge: bool,
    min_segment_ms: float,
    save_report: bool,
    no_report: bool = False,
    preview_only: bool = False,
) -> int:
    audio_extensions = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
    dir_path = Path(directory)

    if not dir_path.is_dir():
        print(f"Error: '{directory}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    audio_files = sorted(
        [
            f
            for f in dir_path.iterdir()
            if f.is_file() and f.suffix.lower() in audio_extensions
        ]
    )

    if not audio_files:
        print(f"No audio files found in '{directory}'.")
        return 0

    print(f"Found {len(audio_files)} audio file(s) in '{directory}'.")
    print()

    total_segments = 0
    for i, audio_file in enumerate(audio_files):
        print(f"[{i + 1}/{len(audio_files)}] Processing: {audio_file.name}")
        try:
            count = process_single_file(
                str(audio_file),
                threshold_db,
                min_silence_ms,
                buffer_before_ms,
                buffer_after_ms,
                output_dir,
                output_format,
                preview,
                export_envelope,
                smart_merge,
                min_segment_ms,
                save_report,
                no_report=no_report,
                preview_only=preview_only,
            )
            total_segments += count
        except Exception as e:
            print(f"  Error processing '{audio_file.name}': {e}", file=sys.stderr)

    print()
    print(f"Batch complete: {total_segments} total segment(s) generated.")
    return total_segments


def main(args: Optional[list] = None):
    parser = argparse.ArgumentParser(
        prog="audio-silence-cutter",
        description="Audio silence detection & automatic splitting tool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview silence detection only
  audio-silence-cutter speech.mp3 --preview-only

  # Split audio with default settings
  audio-silence-cutter speech.mp3

  # Custom threshold and silence duration
  audio-silence-cutter speech.mp3 -t -35 -s 300

  # With smart merge and envelope export
  audio-silence-cutter speech.mp3 --smart-merge --min-segment 2000 --envelope

  # Batch process a folder
  audio-silence-cutter --batch ./recordings/ -o ./output/
        """,
    )

    parser.add_argument(
        "input",
        nargs="?",
        help="Path to input audio file (MP3/WAV/FLAC/OGG).",
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=-40.0,
        help="Silence threshold in dBFS (default: -40.0).",
    )
    parser.add_argument(
        "-s", "--min-silence",
        type=float,
        default=500.0,
        help="Minimum silence duration in ms (default: 500).",
    )
    parser.add_argument(
        "--buffer-before",
        type=float,
        default=200.0,
        help="Buffer time before split point in ms (default: 200).",
    )
    parser.add_argument(
        "--buffer-after",
        type=float,
        default=200.0,
        help="Buffer time after split point in ms (default: 200).",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="./output",
        help="Output directory for split files (default: ./output).",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["mp3", "wav", "flac", "ogg"],
        help="Output format override (default: same as input).",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Only show ASCII waveform preview, do not split.",
    )
    parser.add_argument(
        "--envelope",
        action="store_true",
        help="Export amplitude envelope plots for each segment.",
    )
    parser.add_argument(
        "--smart-merge",
        action="store_true",
        help="Enable smart merge mode (merge short segments).",
    )
    parser.add_argument(
        "--min-segment",
        type=float,
        default=1000.0,
        help="Minimum segment duration for smart merge in ms (default: 1000).",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Do not print the split report.",
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save the report as a JSON file.",
    )
    parser.add_argument(
        "--batch",
        type=str,
        help="Batch process all audio files in the specified directory.",
    )

    parsed = parser.parse_args(args)

    # Determine input source
    input_path: Optional[str] = None
    is_batch = False

    if parsed.batch:
        input_path = parsed.batch
        is_batch = True
    elif parsed.input:
        input_path = parsed.input
    else:
        parser.print_help()
        sys.exit(1)

    preview = True if parsed.preview_only else False
    preview_only = parsed.preview_only
    no_report = parsed.no_report

    if is_batch:
        process_batch(
            directory=input_path,
            threshold_db=parsed.threshold,
            min_silence_ms=parsed.min_silence,
            buffer_before_ms=parsed.buffer_before,
            buffer_after_ms=parsed.buffer_after,
            output_dir=parsed.output_dir,
            output_format=parsed.output_format,
            preview=preview,
            export_envelope=parsed.envelope,
            smart_merge=parsed.smart_merge,
            min_segment_ms=parsed.min_segment,
            save_report=parsed.save_report,
            no_report=no_report,
            preview_only=preview_only,
        )
    else:
        process_single_file(
            filepath=input_path,
            threshold_db=parsed.threshold,
            min_silence_ms=parsed.min_silence,
            buffer_before_ms=parsed.buffer_before,
            buffer_after_ms=parsed.buffer_after,
            output_dir=parsed.output_dir,
            output_format=parsed.output_format,
            preview=preview,
            export_envelope=parsed.envelope,
            smart_merge=parsed.smart_merge,
            min_segment_ms=parsed.min_segment,
            save_report=parsed.save_report,
            no_report=no_report,
            preview_only=preview_only,
        )