#!/usr/bin/env python3
"""
Generate paper-style figures for radar interference experiments.

This script supports:
1) Raw data figures from .mat files:
   - IQ waveform
   - Constellation
   - Amplitude/phase
   - STFT spectrogram
   - Multi-node amplitude comparison
   - Class sample distribution
2) Training-result figures from training logs:
   - Train/val loss and accuracy curves
   - OA/AA/Kappa bars
   - Confusion matrix heatmap
   - Per-class accuracy bars
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.signal import stft


TIMESTAMP_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+\s+")


def strip_log_prefix(line: str) -> str:
    return TIMESTAMP_PREFIX.sub("", line).strip()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_mat_data(file_path: Path, key: str = "data") -> np.ndarray:
    try:
        with h5py.File(file_path, "r") as f:
            if key not in f:
                raise KeyError(f"Key '{key}' not found in {file_path}")
            data = np.array(f[key])
            if data.ndim == 4 and data.shape[3] == 1:
                data = data[:, :, :, 0]
            return data
    except Exception:
        mat = sio.loadmat(file_path)
        if key not in mat:
            raise KeyError(f"Key '{key}' not found in {file_path}")
        return mat[key]


def split_iq_components(mat_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if np.iscomplexobj(mat_data):
        return np.real(mat_data).astype(np.float64), np.imag(mat_data).astype(np.float64)

    if mat_data.dtype != object and mat_data.ndim >= 1 and mat_data.shape[-1] == 2:
        return mat_data[..., 0].astype(np.float64), mat_data[..., 1].astype(np.float64)

    if mat_data.dtype == object:
        flat = [np.asarray(x).reshape(-1) for x in mat_data.flat]
        if len(flat) == 0 or flat[0].size < 2:
            raise ValueError("Unsupported object-cell IQ format.")
        real_part = np.array([v[0] for v in flat], dtype=np.float64).reshape(mat_data.shape)
        imag_part = np.array([v[1] for v in flat], dtype=np.float64).reshape(mat_data.shape)
        return real_part, imag_part

    raise ValueError("Unsupported IQ format. Expect complex/last-dim=2/object-cell.")


def reorder_to_tsn(
    arr: np.ndarray,
    time_axis: int,
    sample_axis: int,
    node_axis: int,
) -> np.ndarray:
    axes = [time_axis, sample_axis, node_axis]
    if len(set(axes)) != 3:
        raise ValueError("time_axis/sample_axis/node_axis must be different.")
    if max(axes) >= arr.ndim:
        raise ValueError(f"Axis index exceeds array ndim={arr.ndim}.")
    moved = np.moveaxis(arr, axes, [0, 1, 2])
    return moved


def save_figure(fig: plt.Figure, out_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_iq_waveform(real_seq: np.ndarray, imag_seq: np.ndarray, out_path: Path) -> None:
    t = np.arange(real_seq.shape[0])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, real_seq, lw=1.0, label="I (Real)")
    ax.plot(t, imag_seq, lw=1.0, label="Q (Imag)", alpha=0.85)
    ax.set_title("IQ Waveform")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, out_path)


def plot_constellation(real_seq: np.ndarray, imag_seq: np.ndarray, out_path: Path) -> None:
    step = max(1, real_seq.shape[0] // 2000)
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.scatter(real_seq[::step], imag_seq[::step], s=6, alpha=0.5)
    ax.set_title("IQ Constellation")
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.grid(alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    save_figure(fig, out_path)


def plot_amp_phase(real_seq: np.ndarray, imag_seq: np.ndarray, out_path: Path) -> None:
    amp = np.sqrt(real_seq ** 2 + imag_seq ** 2)
    phase = np.unwrap(np.arctan2(imag_seq, real_seq))
    t = np.arange(real_seq.shape[0])
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, amp, lw=1.0, color="#0c4da2")
    axes[0].set_title("Instantaneous Amplitude")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(alpha=0.25)
    axes[1].plot(t, phase, lw=1.0, color="#a3333d")
    axes[1].set_title("Unwrapped Phase")
    axes[1].set_ylabel("Phase (rad)")
    axes[1].set_xlabel("Sample Index")
    axes[1].grid(alpha=0.25)
    save_figure(fig, out_path)


def plot_spectrogram(
    real_seq: np.ndarray,
    imag_seq: np.ndarray,
    out_path: Path,
    nperseg: int = 256,
    noverlap: int = 192,
) -> None:
    sig = real_seq + 1j * imag_seq
    seg = min(int(nperseg), int(sig.shape[0]))
    ov = min(int(noverlap), max(0, seg - 1))
    _, t, zxx = stft(sig, nperseg=seg, noverlap=ov, return_onesided=False)
    power = 20 * np.log10(np.abs(np.fft.fftshift(zxx, axes=0)) + 1e-8)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    im = ax.imshow(
        power,
        aspect="auto",
        origin="lower",
        cmap="turbo",
        extent=[t.min(), t.max(), -0.5, 0.5],
    )
    ax.set_title("STFT Spectrogram")
    ax.set_xlabel("Time Bin")
    ax.set_ylabel("Normalized Frequency")
    fig.colorbar(im, ax=ax, label="Power (dB)")
    save_figure(fig, out_path)


def plot_multi_node_amplitude(
    real_tsn: np.ndarray,
    imag_tsn: np.ndarray,
    sample_idx: int,
    out_path: Path,
) -> None:
    t = np.arange(real_tsn.shape[0])
    fig, ax = plt.subplots(figsize=(10, 4))
    num_nodes = real_tsn.shape[2]
    for n in range(num_nodes):
        amp = np.sqrt(real_tsn[:, sample_idx, n] ** 2 + imag_tsn[:, sample_idx, n] ** 2)
        ax.plot(t, amp, lw=1.0, label=f"Node {n}")
    ax.set_title("Amplitude Comparison Across Nodes")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.25)
    ax.legend(ncol=min(4, num_nodes))
    save_figure(fig, out_path)


def plot_class_distribution(data_dir: Path, out_path: Path, mat_key: str = "data") -> None:
    class_files = sorted(list(data_dir.glob("*.mat")))
    names: List[str] = []
    counts: List[int] = []
    for fp in class_files:
        try:
            arr = load_mat_data(fp, key=mat_key)
            real, _ = split_iq_components(arr)
            if real.ndim >= 2:
                sample_count = real.shape[1]
            else:
                sample_count = real.shape[0]
            names.append(fp.stem)
            counts.append(int(sample_count))
        except Exception:
            continue
    if not names:
        return

    fig, ax = plt.subplots(figsize=(max(7, 0.8 * len(names)), 4.2))
    bars = ax.bar(names, counts, color="#2f7ebc")
    ax.set_title("Class Sample Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Samples")
    ax.grid(axis="y", alpha=0.25)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), str(c), ha="center", va="bottom", fontsize=8)
    save_figure(fig, out_path)


def parse_log(log_path: Path) -> Dict[str, object]:
    with open(log_path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()
    lines = [strip_log_prefix(x) for x in raw_lines]

    data: Dict[str, object] = {
        "train": {},
        "val": {},
        "metrics": {},
        "matrices": {},
    }

    train_pat = re.compile(
        r"^(?:Node (?P<node>\d+): )?train epoch : (?P<epoch>\d+)\s+train loss: (?P<loss>[-\d.]+)\s+accuracy: (?P<acc>[-\d.]+)%$"
    )
    val_pat = re.compile(
        r"^(?:Node (?P<node>\d+): )?val loss: (?P<loss>[-\d.]+)\s+accuracy: (?P<acc>[-\d.]+)%$"
    )
    metric_pat = re.compile(
        r"^(?:(?P<node_name>Node \d+|组合模型 Scheme6)\s+)?OA:\s*(?P<oa>[-\d.]+)%\s+AA_mean:\s*(?P<aa>[-\d.]+)%\s+Kappa:\s*(?P<kappa>[-\d.]+)$"
    )

    i = 0
    while i < len(lines):
        line = lines[i]

        m = train_pat.match(line)
        if m:
            node = m.group("node")
            key = f"node_{node}" if node is not None else "global"
            data["train"].setdefault(key, {"epoch": [], "loss": [], "acc": []})
            data["train"][key]["epoch"].append(int(m.group("epoch")))
            data["train"][key]["loss"].append(float(m.group("loss")))
            data["train"][key]["acc"].append(float(m.group("acc")))
            i += 1
            continue

        m = val_pat.match(line)
        if m:
            node = m.group("node")
            key = f"node_{node}" if node is not None else "global"
            data["val"].setdefault(key, {"step": [], "loss": [], "acc": []})
            step = len(data["val"][key]["step"])
            data["val"][key]["step"].append(step)
            data["val"][key]["loss"].append(float(m.group("loss")))
            data["val"][key]["acc"].append(float(m.group("acc")))
            i += 1
            continue

        m = metric_pat.match(line)
        if m:
            node_name = m.group("node_name")
            if node_name is None:
                key = "global"
            elif node_name.startswith("Node"):
                key = node_name.lower().replace(" ", "_")
            else:
                key = "combined"
            data["metrics"][key] = {
                "OA": float(m.group("oa")),
                "AA_mean": float(m.group("aa")),
                "Kappa": float(m.group("kappa")),
            }
            i += 1
            continue

        # Confusion matrix parsing
        if "混淆矩阵" in line:
            if line.startswith("Node "):
                m_node = re.search(r"Node (\d+)", line)
                key = f"node_{m_node.group(1)}" if m_node else "global"
            elif "组合模型 Scheme6" in line:
                key = "combined"
            else:
                key = "global"

            rows: List[np.ndarray] = []
            j = i + 1
            while j < len(lines):
                row_line = lines[j]
                if not row_line.startswith("["):
                    break
                clean = row_line.strip().strip("[]").strip()
                if clean:
                    row = np.fromstring(clean, sep=" ")
                    if row.size > 0:
                        rows.append(row)
                j += 1
            if rows:
                mat = np.vstack(rows).astype(int)
                data["matrices"][key] = mat
            i = j
            continue

        i += 1

    return data


def plot_train_val_curves(parsed: Dict[str, object], out_dir: Path) -> None:
    train = parsed.get("train", {})
    val = parsed.get("val", {})
    keys = sorted(set(list(train.keys()) + list(val.keys())))
    if not keys:
        return

    for key in keys:
        tr = train.get(key, {"epoch": [], "loss": [], "acc": []})
        va = val.get(key, {"step": [], "loss": [], "acc": []})

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        if tr["epoch"]:
            axes[0].plot(tr["epoch"], tr["loss"], marker="o", lw=1.2, label="Train")
        if va["step"]:
            axes[0].plot(va["step"], va["loss"], marker="s", lw=1.2, label="Val")
        axes[0].set_title(f"{key}: Loss Curve")
        axes[0].set_xlabel("Epoch/Step")
        axes[0].set_ylabel("Loss")
        axes[0].grid(alpha=0.25)
        axes[0].legend()

        if tr["epoch"]:
            axes[1].plot(tr["epoch"], tr["acc"], marker="o", lw=1.2, label="Train")
        if va["step"]:
            axes[1].plot(va["step"], va["acc"], marker="s", lw=1.2, label="Val")
        axes[1].set_title(f"{key}: Accuracy Curve")
        axes[1].set_xlabel("Epoch/Step")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].grid(alpha=0.25)
        axes[1].legend()

        save_figure(fig, out_dir / f"log_curve_{key}.png")


def plot_metric_bars(parsed: Dict[str, object], out_path: Path) -> None:
    metrics = parsed.get("metrics", {})
    if not metrics:
        return
    keys = sorted(metrics.keys())
    oa = [metrics[k]["OA"] for k in keys]
    aa = [metrics[k]["AA_mean"] for k in keys]
    kp = [metrics[k]["Kappa"] for k in keys]

    x = np.arange(len(keys))
    w = 0.26
    fig, ax = plt.subplots(figsize=(max(7.5, len(keys) * 1.1), 4.2))
    ax.bar(x - w, oa, width=w, label="OA (%)", color="#1f77b4")
    ax.bar(x, aa, width=w, label="AA_mean (%)", color="#ff7f0e")
    ax.bar(x + w, kp, width=w, label="Kappa", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_title("Evaluation Metrics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, out_path)


def plot_confusion_matrix(matrix: np.ndarray, out_path: Path, title: str) -> None:
    if matrix.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    n = matrix.shape[0]
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    thresh = matrix.max() / 2.0 if matrix.max() > 0 else 0.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > thresh else "black",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, out_path)


def plot_per_class_acc(matrix: np.ndarray, out_path: Path, title: str) -> None:
    if matrix.size == 0:
        return
    row_sum = matrix.sum(axis=1)
    acc = np.divide(
        np.diag(matrix),
        row_sum,
        out=np.zeros_like(row_sum, dtype=np.float64),
        where=row_sum > 0,
    )
    fig, ax = plt.subplots(figsize=(max(6.5, len(acc) * 0.65), 4.2))
    x = np.arange(len(acc))
    bars = ax.bar(x, acc * 100.0, color="#3e8fb0")
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.grid(axis="y", alpha=0.25)
    for b, a in zip(bars, acc):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{a * 100.0:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    save_figure(fig, out_path)


def generate_data_figures(
    data_dir: Path,
    out_dir: Path,
    class_file: Optional[str],
    sample_idx: int,
    node_idx: int,
    mat_key: str,
    time_axis: int,
    sample_axis: int,
    node_axis: int,
) -> None:
    mat_files = sorted(data_dir.glob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No .mat file found in {data_dir}")
    chosen = data_dir / class_file if class_file else mat_files[0]
    if not chosen.exists():
        raise FileNotFoundError(f"Class file not found: {chosen}")

    arr = load_mat_data(chosen, key=mat_key)
    real, imag = split_iq_components(arr)
    real_tsn = reorder_to_tsn(real, time_axis, sample_axis, node_axis)
    imag_tsn = reorder_to_tsn(imag, time_axis, sample_axis, node_axis)

    max_sample = real_tsn.shape[1] - 1
    max_node = real_tsn.shape[2] - 1
    sample_idx = max(0, min(sample_idx, max_sample))
    node_idx = max(0, min(node_idx, max_node))

    real_seq = real_tsn[:, sample_idx, node_idx]
    imag_seq = imag_tsn[:, sample_idx, node_idx]

    plot_iq_waveform(real_seq, imag_seq, out_dir / "data_iq_waveform.png")
    plot_constellation(real_seq, imag_seq, out_dir / "data_constellation.png")
    plot_amp_phase(real_seq, imag_seq, out_dir / "data_amplitude_phase.png")
    plot_spectrogram(real_seq, imag_seq, out_dir / "data_stft_spectrogram.png")
    plot_multi_node_amplitude(real_tsn, imag_tsn, sample_idx, out_dir / "data_multi_node_amplitude.png")
    plot_class_distribution(data_dir, out_dir / "data_class_distribution.png", mat_key=mat_key)


def generate_log_figures(log_path: Path, out_dir: Path) -> None:
    parsed = parse_log(log_path)
    plot_train_val_curves(parsed, out_dir)
    plot_metric_bars(parsed, out_dir / "log_metrics_bar.png")

    matrices = parsed.get("matrices", {})
    for key, mat in matrices.items():
        plot_confusion_matrix(mat, out_dir / f"log_confusion_{key}.png", f"Confusion Matrix ({key})")
        plot_per_class_acc(mat, out_dir / f"log_per_class_acc_{key}.png", f"Per-class Accuracy ({key})")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate paper-style radar experiment figures.")
    p.add_argument("--data_dir", type=str, default=None, help="Directory containing class .mat files.")
    p.add_argument("--log_path", type=str, default=None, help="Training log path (.txt).")
    p.add_argument("--out_dir", type=str, default="figures", help="Output figure directory.")
    p.add_argument("--class_file", type=str, default=None, help="Specific class .mat filename, e.g. class_00.mat.")
    p.add_argument("--sample_idx", type=int, default=0, help="Sample index for data figure.")
    p.add_argument("--node_idx", type=int, default=0, help="Node index for data figure.")
    p.add_argument("--mat_key", type=str, default="data", help="MAT variable key.")
    p.add_argument("--time_axis", type=int, default=0, help="Time axis in mat data.")
    p.add_argument("--sample_axis", type=int, default=1, help="Sample axis in mat data.")
    p.add_argument("--node_axis", type=int, default=2, help="Node axis in mat data.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    if args.data_dir is None and args.log_path is None:
        raise ValueError("At least one of --data_dir or --log_path must be provided.")

    if args.data_dir is not None:
        generate_data_figures(
            data_dir=Path(args.data_dir),
            out_dir=out_dir,
            class_file=args.class_file,
            sample_idx=args.sample_idx,
            node_idx=args.node_idx,
            mat_key=args.mat_key,
            time_axis=args.time_axis,
            sample_axis=args.sample_axis,
            node_axis=args.node_axis,
        )

    if args.log_path is not None:
        generate_log_figures(Path(args.log_path), out_dir)

    print(f"Saved figures to: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
