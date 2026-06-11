import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .silence_detector import SilenceDetector
from .audio_splitter import AudioSplitter
from .smart_merger import SmartMerger
from .preview import PreviewGenerator
from .envelope import EnvelopeExporter
from .config import merge_config, save_config_template, CONFIG_KEYS
from .reporter import Reporter, FileResult


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


def _extract_cli_values(parsed_args) -> Dict[str, Any]:
    mapping = {
        "threshold_db": "threshold",
        "min_silence_ms": "min_silence",
        "buffer_before_ms": "buffer_before",
        "buffer_after_ms": "buffer_after",
        "output_dir": "output_dir",
        "output_format": "output_format",
        "smart_merge": "smart_merge",
        "min_segment_ms": "min_segment",
        "export_envelope": "envelope",
        "save_report": "save_report",
        "no_report": "no_report",
        "batch_summary": "batch_summary",
    }

    cli_values: Dict[str, Any] = {}
    for config_key, attr in mapping.items():
        raw = getattr(parsed_args, attr, None)
        cli_values[config_key] = raw

    return cli_values


def _has_user_override(parsed_args) -> bool:
    override_fields = [
        "threshold", "min_silence", "buffer_before", "buffer_after",
        "output_dir", "output_format", "smart_merge", "min_segment",
        "envelope", "save_report", "no_report", "batch_summary",
    ]
    for field in override_fields:
        val = getattr(parsed_args, field, None)
        default = parsed_args_original_defaults.get(field)
        if val != default:
            return True
    return False


parsed_args_original_defaults: Dict[str, Any] = {}


def _store_defaults(parser: argparse.ArgumentParser, parsed_args) -> None:
    for action in parser._actions:
        if action.dest and action.dest != "help":
            parsed_args_original_defaults[action.dest] = action.default


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
) -> FileResult:
    audio = _load_audio(filepath)
    samples = _audio_to_mono_samples(audio)
    sample_rate = audio.frame_rate
    total_duration_ms = len(audio)
    source_duration_sec = total_duration_ms / 1000.0

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
            rms_values, silent_segments, total_duration_ms, detector.window_ms,
            buffer_before_ms=buffer_before_ms,
            buffer_after_ms=buffer_after_ms,
        )

    if preview_only:
        return FileResult(
            filename=os.path.basename(filepath),
            segment_count=len(non_silent),
            total_duration_sec=sum(s.duration_ms for s in non_silent) / 1000.0,
            source_duration_sec=source_duration_sec,
            success=True,
        )

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
        return FileResult(
            filename=os.path.basename(filepath),
            segment_count=0,
            total_duration_sec=0.0,
            source_duration_sec=source_duration_sec,
            success=True,
        )

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

    total_out_dur = sum(r.duration_sec for r in reports)

    print(f"  Done: {len(results)} segment(s) written to '{output_dir}'")

    return FileResult(
        filename=os.path.basename(filepath),
        segment_count=len(results),
        total_duration_sec=total_out_dur,
        source_duration_sec=source_duration_sec,
        success=True,
    )


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
    batch_summary: bool,
    no_report: bool = False,
    preview_only: bool = False,
) -> None:
    import time

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
        return

    print(f"Found {len(audio_files)} audio file(s) in '{directory}'.")
    print()

    reporter = Reporter(output_dir=output_dir)
    file_results: List[FileResult] = []

    start_time = time.time()
    for i, audio_file in enumerate(audio_files):
        print(f"[{i + 1}/{len(audio_files)}] Processing: {audio_file.name}")
        file_start = time.time()
        try:
            result = process_single_file(
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
            elapsed = time.time() - file_start
            print(f"  → {result.segment_count} segments, duration: {result.total_duration_sec:.2f}s, took: {elapsed:.1f}s")
            file_results.append(result)
        except Exception as e:
            elapsed = time.time() - file_start
            err_msg = str(e)
            print(f"  → FAILED after {elapsed:.1f}s: {err_msg}", file=sys.stderr)
            file_results.append(FileResult(
                filename=audio_file.name,
                segment_count=0,
                total_duration_sec=0.0,
                source_duration_sec=0.0,
                success=False,
                error_message=err_msg,
            ))

    total_elapsed = time.time() - start_time

    summary = reporter.build_batch_summary(file_results)
    summary.total_files = len(audio_files)

    reporter.print_batch_summary(summary)
    print(f"  Total time: {total_elapsed:.1f}s")

    if batch_summary:
        reporter.save_batch_summary_json(summary, output_dir)


def main(args: Optional[list] = None):
    parser = argparse.ArgumentParser(
        prog="audio-silence-cutter",
        description="Audio silence detection & automatic splitting tool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview silence detection only
  audio-silence-cutter speech.mp3 --preview-only

  # Preview first, then choose to split
  audio-silence-cutter speech.mp3 --preview

  # Split audio with default settings
  audio-silence-cutter speech.mp3

  # Custom threshold and silence duration
  audio-silence-cutter speech.mp3 -t -35 -s 300

  # With smart merge and envelope export
  audio-silence-cutter speech.mp3 --smart-merge --min-segment 2000 --envelope

  # Batch process a folder with summary
  audio-silence-cutter --batch ./recordings/ -o ./output/ --batch-summary

  # Use config file (command-line args override config values)
  audio-silence-cutter speech.mp3 --config my_config.json

  # Save a config template
  audio-silence-cutter --save-config my_config.json
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
        default=None,
        help="Silence threshold in dBFS (default: -40.0).",
    )
    parser.add_argument(
        "-s", "--min-silence",
        type=float,
        default=None,
        help="Minimum silence duration in ms (default: 500).",
    )
    parser.add_argument(
        "--buffer-before",
        type=float,
        default=None,
        help="Buffer time before split point in ms (default: 200).",
    )
    parser.add_argument(
        "--buffer-after",
        type=float,
        default=None,
        help="Buffer time after split point in ms (default: 200).",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Output directory for split files (default: ./output).",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["mp3", "wav", "flac", "ogg"],
        default=None,
        help="Output format override (default: same as input).",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show ASCII waveform preview before splitting.",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Only show ASCII waveform preview, do not split.",
    )
    parser.add_argument(
        "--envelope",
        action="store_true",
        default=None,
        help="Export amplitude envelope plots for each segment.",
    )
    parser.add_argument(
        "--smart-merge",
        action="store_true",
        default=None,
        help="Enable smart merge mode (merge short segments).",
    )
    parser.add_argument(
        "--min-segment",
        type=float,
        default=None,
        help="Minimum segment duration for smart merge in ms (default: 1000).",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        default=None,
        help="Do not print the split report.",
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        default=None,
        help="Save the report as a JSON file.",
    )
    parser.add_argument(
        "--batch",
        type=str,
        default=None,
        help="Batch process all audio files in the specified directory.",
    )
    parser.add_argument(
        "--batch-summary",
        action="store_true",
        default=None,
        help="Save a batch summary report as JSON.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a JSON config file (CLI args override file values).",
    )
    parser.add_argument(
        "--save-config",
        type=str,
        default=None,
        help="Save a config file template to the given path and exit.",
    )

    parsed = parser.parse_args(args)

    _store_defaults(parser, parsed)

    if parsed.save_config:
        save_config_template(parsed.save_config)
        print(f"Config template saved to: {parsed.save_config}")
        return

    input_path: Optional[str] = None
    is_batch = False

    if parsed.batch:
        input_path = parsed.batch
        is_batch = True
    elif parsed.input:
        input_path = parsed.input
    elif not parsed.save_config:
        parser.print_help()
        sys.exit(1)

    cli_values = _extract_cli_values(parsed)

    merged = merge_config(
        cli_args=cli_values,
        config_path=parsed.config,
    )

    preview_only = parsed.preview_only
    preview = parsed.preview or preview_only

    no_report = merged["no_report"]
    batch_summary = merged["batch_summary"]

    if is_batch:
        process_batch(
            directory=input_path,
            threshold_db=merged["threshold_db"],
            min_silence_ms=merged["min_silence_ms"],
            buffer_before_ms=merged["buffer_before_ms"],
            buffer_after_ms=merged["buffer_after_ms"],
            output_dir=merged["output_dir"],
            output_format=merged["output_format"],
            preview=preview,
            export_envelope=merged["export_envelope"],
            smart_merge=merged["smart_merge"],
            min_segment_ms=merged["min_segment_ms"],
            save_report=merged["save_report"],
            batch_summary=batch_summary,
            no_report=no_report,
            preview_only=preview_only,
        )
    else:
        result = process_single_file(
            filepath=input_path,
            threshold_db=merged["threshold_db"],
            min_silence_ms=merged["min_silence_ms"],
            buffer_before_ms=merged["buffer_before_ms"],
            buffer_after_ms=merged["buffer_after_ms"],
            output_dir=merged["output_dir"],
            output_format=merged["output_format"],
            preview=preview,
            export_envelope=merged["export_envelope"],
            smart_merge=merged["smart_merge"],
            min_segment_ms=merged["min_segment_ms"],
            save_report=merged["save_report"],
            no_report=no_report,
            preview_only=preview_only,
        )