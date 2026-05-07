import argparse
import os
import numpy as np
import scipy.io as sio


COMPOSITE_CLASS_PAIRS = [
    (0, 6),   # CW + FM
    (1, 9),   # Up-chirp + hop
    (2, 9),   # Down-chirp + hop
    (3, 5),   # BPSK + AM
    (4, 5),   # QPSK + AM
    (6, 7),   # FM + pulse-gated carrier
    (8, 5),   # Dual-tone + AM
    (1, 6),   # Up-chirp + FM
    (2, 6),   # Down-chirp + FM
    (8, 10),  # Dual-tone + pulse-noise
    (7, 10),  # Pulse-gated carrier + pulse-noise
    (9, 11),  # Hop + phase-reset
]


def _unit_complex(theta: np.ndarray) -> np.ndarray:
    return np.exp(1j * theta)


def _normalize_power(x: np.ndarray) -> np.ndarray:
    p = np.mean(np.abs(x) ** 2) + 1e-8
    return x / np.sqrt(p)


def _add_awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    sig_power = np.mean(np.abs(x) ** 2) + 1e-8
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = (
        rng.standard_normal(x.shape).astype(np.float32)
        + 1j * rng.standard_normal(x.shape).astype(np.float32)
    ) * np.sqrt(noise_power / 2.0)
    return x + noise


def _pulse_gate(length: int, period: int, width: int, offset: int) -> np.ndarray:
    gate = np.zeros(length, dtype=np.float32)
    for i in range(offset, length, period):
        gate[i:i + width] = 1.0
    return gate


def _parse_freq_pool(pool_text: str | None) -> np.ndarray | None:
    if pool_text is None:
        return None
    pool_text = pool_text.strip()
    if not pool_text:
        return None
    return np.array([float(x) for x in pool_text.split(',')], dtype=np.float32)


def _sample_hop_freqs(rng: np.random.Generator, seg: int, min_gap: float, pool: np.ndarray | None) -> np.ndarray:
    freqs = []
    max_tries = 128
    for i in range(seg):
        prev = freqs[-1] if i > 0 else None
        for _ in range(max_tries):
            f = float(rng.choice(pool)) if pool is not None else float(rng.uniform(2.0, 16.0))
            if prev is None or abs(f - prev) >= min_gap:
                freqs.append(f)
                break
        else:
            if pool is not None:
                candidates = [float(x) for x in pool if prev is None or abs(float(x) - prev) >= min_gap]
                freqs.append(candidates[0] if candidates else float(pool[0]))
            else:
                freqs.append(float(rng.uniform(2.0, 16.0)))
    return np.array(freqs, dtype=np.float32)


def _gen_class_waveform_core(
    class_id: int,
    length: int,
    rng: np.random.Generator,
    lfm_up_f0_min: float,
    lfm_up_f0_max: float,
    lfm_up_f1_min: float,
    lfm_up_f1_max: float,
    lfm_down_f0_min: float,
    lfm_down_f0_max: float,
    lfm_down_f1_min: float,
    lfm_down_f1_max: float,
    fm_mod_freq_min: float,
    fm_mod_freq_max: float,
    fm_beta_min: float,
    fm_beta_max: float,
    hop_min_gap: float,
    hop_freq_pool: np.ndarray | None,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, length, endpoint=False, dtype=np.float32)
    pi2 = 2.0 * np.pi
    cid = class_id % 12

    # 12 classes with distinct structure to make classification learnable.
    if cid == 0:
        f = rng.uniform(4.0, 8.0)
        x = _unit_complex(pi2 * f * t)
    elif cid == 1:
        f0 = rng.uniform(lfm_up_f0_min, lfm_up_f0_max)
        f1 = rng.uniform(lfm_up_f1_min, lfm_up_f1_max)
        k = f1 - f0
        x = _unit_complex(pi2 * (f0 * t + 0.5 * k * t ** 2))
    elif cid == 2:
        f0 = rng.uniform(lfm_down_f0_min, lfm_down_f0_max)
        f1 = rng.uniform(lfm_down_f1_min, lfm_down_f1_max)
        k = f1 - f0
        x = _unit_complex(pi2 * (f0 * t + 0.5 * k * t ** 2))
    elif cid == 3:
        chips = rng.choice([-1.0, 1.0], size=max(8, length // 12)).astype(np.float32)
        rep = int(np.ceil(length / chips.size))
        phase = np.repeat(chips, rep)[:length] * np.pi / 2.0
        x = _unit_complex(phase)
    elif cid == 4:
        phases = rng.choice([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi], size=max(8, length // 12))
        rep = int(np.ceil(length / phases.size))
        phase = np.repeat(phases, rep)[:length]
        x = _unit_complex(phase)
    elif cid == 5:
        fc = rng.uniform(5.0, 8.0)
        fm = rng.uniform(0.8, 1.6)
        m = rng.uniform(0.3, 0.8)
        env = (1.0 + m * np.sin(pi2 * fm * t)).astype(np.float32)
        x = env * _unit_complex(pi2 * fc * t)
    elif cid == 6:
        fc = rng.uniform(5.0, 9.0)
        fm = rng.uniform(fm_mod_freq_min, fm_mod_freq_max)
        beta = rng.uniform(fm_beta_min, fm_beta_max)
        x = _unit_complex(pi2 * fc * t + beta * np.sin(pi2 * fm * t))
    elif cid == 7:
        fc = rng.uniform(4.0, 10.0)
        gate = _pulse_gate(length, period=max(10, length // 10), width=max(3, length // 35), offset=2)
        x = gate * _unit_complex(pi2 * fc * t)
    elif cid == 8:
        f1, f2 = rng.uniform(3.0, 6.0), rng.uniform(9.0, 14.0)
        x = 0.65 * _unit_complex(pi2 * f1 * t) + 0.35 * _unit_complex(pi2 * f2 * t)
    elif cid == 9:
        seg = 4
        seg_len = length // seg
        parts = []
        hop_freqs = _sample_hop_freqs(rng, seg=seg, min_gap=hop_min_gap, pool=hop_freq_pool)
        for f in hop_freqs:
            tt = np.linspace(0.0, 1.0 / seg, seg_len, endpoint=False, dtype=np.float32)
            parts.append(_unit_complex(pi2 * f * tt))
        x = np.concatenate(parts)
        if x.size < length:
            x = np.pad(x, (0, length - x.size), mode="edge")
    elif cid == 10:
        base = rng.standard_normal(length).astype(np.float32) + 1j * rng.standard_normal(length).astype(np.float32)
        gate = _pulse_gate(length, period=max(12, length // 8), width=max(2, length // 45), offset=0)
        x = (0.4 + 0.9 * gate) * base
    else:
        # Sawtooth-like phase reset pattern.
        period = max(8, length // 14)
        phase = ((np.arange(length) % period) / period).astype(np.float32) * 2.0 * np.pi
        x = _unit_complex(phase)

    return _normalize_power(x.astype(np.complex64))


def _gen_class_waveform(
    class_id: int,
    length: int,
    rng: np.random.Generator,
    base_snr_min: float,
    base_snr_max: float,
    lfm_up_f0_min: float,
    lfm_up_f0_max: float,
    lfm_up_f1_min: float,
    lfm_up_f1_max: float,
    lfm_down_f0_min: float,
    lfm_down_f0_max: float,
    lfm_down_f1_min: float,
    lfm_down_f1_max: float,
    fm_mod_freq_min: float,
    fm_mod_freq_max: float,
    fm_beta_min: float,
    fm_beta_max: float,
    hop_min_gap: float,
    hop_freq_pool: np.ndarray | None,
) -> np.ndarray:
    x = _gen_class_waveform_core(
        class_id=class_id,
        length=length,
        rng=rng,
        lfm_up_f0_min=lfm_up_f0_min,
        lfm_up_f0_max=lfm_up_f0_max,
        lfm_up_f1_min=lfm_up_f1_min,
        lfm_up_f1_max=lfm_up_f1_max,
        lfm_down_f0_min=lfm_down_f0_min,
        lfm_down_f0_max=lfm_down_f0_max,
        lfm_down_f1_min=lfm_down_f1_min,
        lfm_down_f1_max=lfm_down_f1_max,
        fm_mod_freq_min=fm_mod_freq_min,
        fm_mod_freq_max=fm_mod_freq_max,
        fm_beta_min=fm_beta_min,
        fm_beta_max=fm_beta_max,
        hop_min_gap=hop_min_gap,
        hop_freq_pool=hop_freq_pool,
    )
    x = _add_awgn(x, snr_db=rng.uniform(base_snr_min, base_snr_max), rng=rng)
    return x.astype(np.complex64)


def make_structured_iq_array(
    seq_len_total: int,
    samples_per_class: int,
    num_nodes: int,
    class_id: int,
    seed: int,
    base_snr_min: float,
    base_snr_max: float,
    node_snr_min: float,
    node_snr_max: float,
    lfm_up_f0_min: float,
    lfm_up_f0_max: float,
    lfm_up_f1_min: float,
    lfm_up_f1_max: float,
    lfm_down_f0_min: float,
    lfm_down_f0_max: float,
    lfm_down_f1_min: float,
    lfm_down_f1_max: float,
    fm_mod_freq_min: float,
    fm_mod_freq_max: float,
    fm_beta_min: float,
    fm_beta_max: float,
    hop_min_gap: float,
    hop_freq_pool: np.ndarray | None,
    mode: str = "structured",
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    complex_data = np.zeros((seq_len_total, samples_per_class, num_nodes), dtype=np.complex64)

    for s in range(samples_per_class):
        if mode == "composite12":
            c0, c1 = COMPOSITE_CLASS_PAIRS[class_id % len(COMPOSITE_CLASS_PAIRS)]
            mix = rng.uniform(0.35, 0.65)
            phase1 = np.exp(1j * rng.uniform(-np.pi, np.pi))
            x0 = _gen_class_waveform_core(
                class_id=c0,
                length=seq_len_total,
                rng=rng,
                lfm_up_f0_min=lfm_up_f0_min,
                lfm_up_f0_max=lfm_up_f0_max,
                lfm_up_f1_min=lfm_up_f1_min,
                lfm_up_f1_max=lfm_up_f1_max,
                lfm_down_f0_min=lfm_down_f0_min,
                lfm_down_f0_max=lfm_down_f0_max,
                lfm_down_f1_min=lfm_down_f1_min,
                lfm_down_f1_max=lfm_down_f1_max,
                fm_mod_freq_min=fm_mod_freq_min,
                fm_mod_freq_max=fm_mod_freq_max,
                fm_beta_min=fm_beta_min,
                fm_beta_max=fm_beta_max,
                hop_min_gap=hop_min_gap,
                hop_freq_pool=hop_freq_pool,
            )
            x1 = _gen_class_waveform_core(
                class_id=c1,
                length=seq_len_total,
                rng=rng,
                lfm_up_f0_min=lfm_up_f0_min,
                lfm_up_f0_max=lfm_up_f0_max,
                lfm_up_f1_min=lfm_up_f1_min,
                lfm_up_f1_max=lfm_up_f1_max,
                lfm_down_f0_min=lfm_down_f0_min,
                lfm_down_f0_max=lfm_down_f0_max,
                lfm_down_f1_min=lfm_down_f1_min,
                lfm_down_f1_max=lfm_down_f1_max,
                fm_mod_freq_min=fm_mod_freq_min,
                fm_mod_freq_max=fm_mod_freq_max,
                fm_beta_min=fm_beta_min,
                fm_beta_max=fm_beta_max,
                hop_min_gap=hop_min_gap,
                hop_freq_pool=hop_freq_pool,
            )
            base = _normalize_power((mix * x0 + (1.0 - mix) * phase1 * x1).astype(np.complex64))
            base = _add_awgn(base, snr_db=rng.uniform(base_snr_min, base_snr_max), rng=rng)
        else:
            base = _gen_class_waveform(
                class_id=class_id,
                length=seq_len_total,
                rng=rng,
                base_snr_min=base_snr_min,
                base_snr_max=base_snr_max,
                lfm_up_f0_min=lfm_up_f0_min,
                lfm_up_f0_max=lfm_up_f0_max,
                lfm_up_f1_min=lfm_up_f1_min,
                lfm_up_f1_max=lfm_up_f1_max,
                lfm_down_f0_min=lfm_down_f0_min,
                lfm_down_f0_max=lfm_down_f0_max,
                lfm_down_f1_min=lfm_down_f1_min,
                lfm_down_f1_max=lfm_down_f1_max,
                fm_mod_freq_min=fm_mod_freq_min,
                fm_mod_freq_max=fm_mod_freq_max,
                fm_beta_min=fm_beta_min,
                fm_beta_max=fm_beta_max,
                hop_min_gap=hop_min_gap,
                hop_freq_pool=hop_freq_pool,
            )
        for n in range(num_nodes):
            amp = rng.uniform(0.85, 1.15)
            phi = rng.uniform(-0.35, 0.35)
            delay = rng.integers(-3, 4)
            node_sig = np.roll(base, delay) * (amp * np.exp(1j * phi))
            node_sig = _add_awgn(node_sig, snr_db=rng.uniform(node_snr_min, node_snr_max), rng=rng)
            complex_data[:, s, n] = node_sig.astype(np.complex64)
    return complex_data


def make_random_iq_array(seq_len_total: int, samples_per_class: int, num_nodes: int, seed: int) -> np.ndarray:
    """Old behavior: class-independent Gaussian IQ."""
    rng = np.random.default_rng(seed)
    real = rng.standard_normal((seq_len_total, samples_per_class, num_nodes), dtype=np.float32)
    imag = rng.standard_normal((seq_len_total, samples_per_class, num_nodes), dtype=np.float32)
    return (real + 1j * imag).astype(np.complex64)


def to_object_iq(arr_complex: np.ndarray) -> np.ndarray:
    seq_len_total, samples_per_class, num_nodes = arr_complex.shape
    out = np.empty((seq_len_total, samples_per_class, num_nodes), dtype=object)
    for i in range(seq_len_total):
        for j in range(samples_per_class):
            for k in range(num_nodes):
                v = arr_complex[i, j, k]
                out[i, j, k] = np.array([np.float32(np.real(v)), np.float32(np.imag(v))], dtype=np.float32)
    return out


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic .mat radar data')
    parser.add_argument('--out_dir', type=str, default='synthetic_data')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--samples_per_class', type=int, default=16)
    parser.add_argument('--num_nodes', type=int, default=3)
    parser.add_argument('--seq_len', type=int, default=20, help='seq_len after period split')
    parser.add_argument('--num_periods', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--mode',
        type=str,
        default='structured',
        choices=['structured', 'composite12', 'random'],
        help='structured: class-specific jammer patterns; composite12: fixed two-source mixtures; random: old Gaussian IQ.'
    )
    parser.add_argument(
        '--save_format',
        type=str,
        default='complex',
        choices=['complex', 'object'],
        help='complex is much faster and already supported by data loader.'
    )
    parser.add_argument('--base_snr_min', type=float, default=8.0, help='Base waveform AWGN min SNR (dB).')
    parser.add_argument('--base_snr_max', type=float, default=22.0, help='Base waveform AWGN max SNR (dB).')
    parser.add_argument('--node_snr_min', type=float, default=10.0, help='Per-node AWGN min SNR (dB).')
    parser.add_argument('--node_snr_max', type=float, default=24.0, help='Per-node AWGN max SNR (dB).')
    parser.add_argument('--lfm_up_f0_min', type=float, default=2.0, help='Up-chirp start frequency min.')
    parser.add_argument('--lfm_up_f0_max', type=float, default=4.0, help='Up-chirp start frequency max.')
    parser.add_argument('--lfm_up_f1_min', type=float, default=12.0, help='Up-chirp end frequency min.')
    parser.add_argument('--lfm_up_f1_max', type=float, default=16.0, help='Up-chirp end frequency max.')
    parser.add_argument('--lfm_down_f0_min', type=float, default=12.0, help='Down-chirp start frequency min.')
    parser.add_argument('--lfm_down_f0_max', type=float, default=16.0, help='Down-chirp start frequency max.')
    parser.add_argument('--lfm_down_f1_min', type=float, default=2.0, help='Down-chirp end frequency min.')
    parser.add_argument('--lfm_down_f1_max', type=float, default=4.0, help='Down-chirp end frequency max.')
    parser.add_argument('--fm_mod_freq_min', type=float, default=0.5, help='FM class modulation frequency min.')
    parser.add_argument('--fm_mod_freq_max', type=float, default=2.0, help='FM class modulation frequency max.')
    parser.add_argument('--fm_beta_min', type=float, default=1.0, help='FM class modulation index min.')
    parser.add_argument('--fm_beta_max', type=float, default=4.0, help='FM class modulation index max.')
    parser.add_argument('--hop_min_gap', type=float, default=0.0, help='Minimum adjacent hop-frequency gap.')
    parser.add_argument('--hop_freq_pool', type=str, default='', help='Optional comma-separated hop frequency pool.')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seq_len_total = args.seq_len * args.num_periods
    hop_freq_pool = _parse_freq_pool(args.hop_freq_pool)

    for c in range(args.num_classes):
        if args.mode in {'structured', 'composite12'}:
            arr_complex = make_structured_iq_array(
                seq_len_total=seq_len_total,
                samples_per_class=args.samples_per_class,
                num_nodes=args.num_nodes,
                class_id=c,
                seed=args.seed + c,
                base_snr_min=args.base_snr_min,
                base_snr_max=args.base_snr_max,
                node_snr_min=args.node_snr_min,
                node_snr_max=args.node_snr_max,
                lfm_up_f0_min=args.lfm_up_f0_min,
                lfm_up_f0_max=args.lfm_up_f0_max,
                lfm_up_f1_min=args.lfm_up_f1_min,
                lfm_up_f1_max=args.lfm_up_f1_max,
                lfm_down_f0_min=args.lfm_down_f0_min,
                lfm_down_f0_max=args.lfm_down_f0_max,
                lfm_down_f1_min=args.lfm_down_f1_min,
                lfm_down_f1_max=args.lfm_down_f1_max,
                fm_mod_freq_min=args.fm_mod_freq_min,
                fm_mod_freq_max=args.fm_mod_freq_max,
                fm_beta_min=args.fm_beta_min,
                fm_beta_max=args.fm_beta_max,
                hop_min_gap=args.hop_min_gap,
                hop_freq_pool=hop_freq_pool,
                mode=args.mode,
            )
        else:
            arr_complex = make_random_iq_array(
                seq_len_total=seq_len_total,
                samples_per_class=args.samples_per_class,
                num_nodes=args.num_nodes,
                seed=args.seed + c,
            )
        arr = arr_complex if args.save_format == 'complex' else to_object_iq(arr_complex)
        path = os.path.join(args.out_dir, f'class_{c:02d}.mat')
        sio.savemat(path, {'data': arr})
        print(f'Wrote {path} with data shape={arr.shape} mode={args.mode} format={args.save_format}')


if __name__ == '__main__':
    main()
