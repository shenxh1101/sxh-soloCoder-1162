import os
import numpy as np
from typing import Optional


class EnvelopeExporter:

    def __init__(self, dpi: int = 150):
        self.dpi = dpi

    def export(
        self,
        samples: np.ndarray,
        sample_rate: int,
        output_path: str,
        start_ms: float = 0.0,
        end_ms: Optional[float] = None,
        title: Optional[str] = None,
    ) -> str:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for envelope export. "
                "Install it with: pip install matplotlib"
            )

        start_sample = int(start_ms / 1000.0 * sample_rate)
        if end_ms is not None:
            end_sample = int(end_ms / 1000.0 * sample_rate)
            chunk = samples[start_sample:end_sample]
        else:
            chunk = samples[start_sample:]
            end_ms = start_ms + len(chunk) / sample_rate * 1000.0

        if len(chunk) == 0:
            raise ValueError("Empty audio chunk for envelope export")

        time_axis = np.linspace(
            start_ms / 1000.0, end_ms / 1000.0, len(chunk)
        )

        window_samples = int(10 / 1000.0 * sample_rate)
        if window_samples < 1:
            window_samples = 1

        num_frames = len(chunk) // window_samples
        if num_frames == 0:
            rms_time = np.array([start_ms / 1000.0])
            envelope = np.array([0.0])
        else:
            envelope = np.zeros(num_frames)
            for i in range(num_frames):
                s = i * window_samples
                e = s + window_samples
                frame = chunk[s:e]
                envelope[i] = np.sqrt(np.mean(frame.astype(np.float64) ** 2))
            rms_time = np.linspace(
                start_ms / 1000.0,
                start_ms / 1000.0 + num_frames * window_samples / sample_rate,
                num_frames,
            )

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 6), dpi=self.dpi
        )

        ax1.plot(time_axis, chunk, color="#2c7fb8", linewidth=0.3, alpha=0.8)
        ax1.fill_between(
            time_axis,
            chunk,
            0,
            color="#2c7fb8",
            alpha=0.15,
        )
        ax1.set_ylabel("Amplitude")
        ax1.set_title(title or "Waveform & Amplitude Envelope")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(time_axis[0], time_axis[-1])

        ax2.plot(rms_time, envelope, color="#e74c3c", linewidth=1.2)
        ax2.fill_between(
            rms_time,
            envelope,
            0,
            color="#e74c3c",
            alpha=0.2,
        )
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("RMS Amplitude")
        ax2.set_title("Amplitude Envelope (RMS)")
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(time_axis[0], time_axis[-1])

        plt.tight_layout()
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        return output_path