import argparse
import os
import re
import subprocess
import sys
from typing import List, Optional


OA_RE = re.compile(r"OA:\s*([0-9.]+)%\s+AA_mean:\s*([0-9.]+)%\s+Kappa:\s*([0-9.]+)")


def parse_metrics(log_path: str):
    oa = aa = kappa = None
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = OA_RE.search(line)
            if m:
                oa, aa, kappa = m.groups()
    if oa is None:
        raise RuntimeError(f"Failed to parse metrics from {log_path}")
    return float(oa), float(aa), float(kappa)


def latest_log(log_dir: str) -> str:
    candidates = [
        os.path.join(log_dir, name)
        for name in os.listdir(log_dir)
        if name.endswith(".txt") and "log" in name
    ]
    if not candidates:
        raise RuntimeError(f"No log files found in {log_dir}")
    return max(candidates, key=os.path.getmtime)


def run_command(cmd: List[str], cwd: str):
    proc = subprocess.run(cmd, cwd=cwd, check=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def build_base_cmd(args, train_size: int, log_dir: str) -> List[str]:
    return [
        sys.executable,
        "train_main-TR-v2.py",
        "--arch", "scheme5",
        "--datapath", args.datapath,
        "--logDir", log_dir,
        "--epochs", str(args.epochs),
        "--early_stop_patience", str(args.early_stop_patience),
        "--batch_size", str(args.batch_size),
        "--train_size", str(train_size),
        "--num_classes", str(args.num_classes),
        "--seq_len", str(args.seq_len),
        "--patch_size", str(args.patch_size),
        "--stride", str(args.stride),
        "--d_model", str(args.d_model),
        "--mlp_ratio", str(args.mlp_ratio),
        "--nhead", str(args.nhead),
        "--num_blocks", str(args.num_blocks),
        "--cls_head", args.cls_head,
        "--cosine_scale", str(args.cosine_scale),
        "--iterTime", "1",
        "--noCuda",
        "--log-interval", str(args.log_interval),
        "--lr", str(args.lr),
        "--label_smoothing", str(args.label_smoothing),
        "--aug_scale", "0.0",
        "--aug_phase_max", "0.0",
        "--aug_time_shift", "0",
        "--aug_noise_std", "0.0",
        "--aug_snr_prob", "0.0",
        "--node_dropout_prob", "0.0",
    ]


def maybe_append_resume(cmd: List[str], resume: Optional[str], finetune_lr: Optional[float]):
    if resume:
        cmd.extend(["--resume", resume])
    if finetune_lr is not None:
        cmd[cmd.index("--lr") + 1] = str(finetune_lr)


def main():
    parser = argparse.ArgumentParser(description="Run small-sample Scheme5 benchmark.")
    parser.add_argument("--datapath", default="synthetic_data_structured12_long")
    parser.add_argument("--train_sizes", nargs="+", type=int, default=[8, 16, 24])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--early_stop_patience", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=12)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=96)
    parser.add_argument("--mlp_ratio", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--cls_head", default="linear", choices=["linear", "cosine"])
    parser.add_argument("--cosine_scale", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=0.0015)
    parser.add_argument("--finetune_lr", type=float, default=0.0005)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--resume_ckpt", default="")
    parser.add_argument("--out_dir", default="log-TR-v2-small-sample")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.makedirs(args.out_dir, exist_ok=True)

    print("mode,train_size,oa,aa_mean,kappa,log_path")

    for train_size in args.train_sizes:
        direct_log_dir = os.path.join(args.out_dir, f"direct_{train_size}")
        os.makedirs(direct_log_dir, exist_ok=True)
        direct_cmd = build_base_cmd(args, train_size, direct_log_dir)
        run_command(direct_cmd, repo_root)
        direct_log = latest_log(direct_log_dir)
        oa, aa, kappa = parse_metrics(direct_log)
        print(f"direct,{train_size},{oa:.2f},{aa:.2f},{kappa:.4f},{direct_log}")

        if args.resume_ckpt:
            ft_log_dir = os.path.join(args.out_dir, f"finetune_{train_size}")
            os.makedirs(ft_log_dir, exist_ok=True)
            ft_cmd = build_base_cmd(args, train_size, ft_log_dir)
            maybe_append_resume(ft_cmd, args.resume_ckpt, args.finetune_lr)
            run_command(ft_cmd, repo_root)
            ft_log = latest_log(ft_log_dir)
            oa, aa, kappa = parse_metrics(ft_log)
            print(f"finetune,{train_size},{oa:.2f},{aa:.2f},{kappa:.4f},{ft_log}")


if __name__ == "__main__":
    main()
