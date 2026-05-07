#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.plot_paper_figures import ensure_dir, load_mat_data, reorder_to_tsn, split_iq_components


CLASS_NAMES = [
    "class_00: CW",
    "class_01: LFM up-chirp",
    "class_02: LFM down-chirp",
    "class_03: BPSK",
    "class_04: QPSK",
    "class_05: AM",
    "class_06: FM",
    "class_07: Pulse-gated carrier",
    "class_08: Dual-tone",
    "class_09: Hop frequency",
    "class_10: Pulse noise",
    "class_11: Sawtooth phase reset",
]


def save_figure(fig: plt.Figure, out_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_power_map(real_seq: np.ndarray, imag_seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sig = real_seq + 1j * imag_seq
    nperseg = min(128, sig.shape[0])
    noverlap = max(0, nperseg // 2)
    _, t, zxx = stft(sig, nperseg=nperseg, noverlap=noverlap, return_onesided=False)
    power = 20.0 * np.log10(np.abs(np.fft.fftshift(zxx, axes=0)) + 1e-8)
    return t, power


def plot_single_class(real_seq: np.ndarray, imag_seq: np.ndarray, title: str, out_path: Path) -> None:
    t = np.arange(real_seq.shape[0])
    tf_t, power = build_power_map(real_seq, imag_seq)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(t, real_seq, lw=1.0, label="I")
    axes[0].plot(t, imag_seq, lw=1.0, alpha=0.85, label="Q")
    axes[0].set_title("IQ waveform")
    axes[0].set_xlabel("Sample")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    step = max(1, real_seq.shape[0] // 1500)
    axes[1].scatter(real_seq[::step], imag_seq[::step], s=8, alpha=0.55)
    axes[1].set_title("Constellation")
    axes[1].set_xlabel("I")
    axes[1].set_ylabel("Q")
    axes[1].grid(alpha=0.25)
    axes[1].set_aspect("equal", adjustable="box")

    im = axes[2].imshow(
        power,
        aspect="auto",
        origin="lower",
        cmap="turbo",
        extent=[tf_t.min(), tf_t.max(), -0.5, 0.5],
    )
    axes[2].set_title("STFT spectrogram")
    axes[2].set_xlabel("Time bin")
    axes[2].set_ylabel("Norm. freq")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=14)
    save_figure(fig, out_path)


def plot_overview(class_series: list[tuple[str, np.ndarray, np.ndarray]], out_path: Path) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(12, 10))
    for ax, (title, real_seq, imag_seq) in zip(axes.flat, class_series):
        tf_t, power = build_power_map(real_seq, imag_seq)
        ax.imshow(
            power,
            aspect="auto",
            origin="lower",
            cmap="turbo",
            extent=[tf_t.min(), tf_t.max(), -0.5, 0.5],
        )
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Structured12 class spectrogram overview", fontsize=15)
    save_figure(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot preview figures for 12 structured jammer classes.")
    parser.add_argument("--data_dir", type=str, default="synthetic_data_structured12_long")
    parser.add_argument("--out_dir", type=str, default="figures_structured12_classes")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--node_idx", type=int, default=0)
    parser.add_argument("--mat_key", type=str, default="data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    class_series: list[tuple[str, np.ndarray, np.ndarray]] = []
    for class_id, title in enumerate(CLASS_NAMES):
        mat_path = data_dir / f"class_{class_id:02d}.mat"
        arr = load_mat_data(mat_path, key=args.mat_key)
        real, imag = split_iq_components(arr)
        real_tsn = reorder_to_tsn(real, 0, 1, 2)
        imag_tsn = reorder_to_tsn(imag, 0, 1, 2)

        sample_idx = max(0, min(args.sample_idx, real_tsn.shape[1] - 1))
        node_idx = max(0, min(args.node_idx, real_tsn.shape[2] - 1))
        real_seq = real_tsn[:, sample_idx, node_idx]
        imag_seq = imag_tsn[:, sample_idx, node_idx]

        class_series.append((title, real_seq, imag_seq))
        plot_single_class(real_seq, imag_seq, title, out_dir / f"class_{class_id:02d}.png")

    plot_overview(class_series, out_dir / "structured12_overview.png")


if __name__ == "__main__":
    main()
