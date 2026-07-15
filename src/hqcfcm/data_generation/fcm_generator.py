"""
Synthetic Fuzzy Cognitive Map (FCM) trajectory generation.

Ground-truth FCM dynamics follow the classical update rule
``A(t+1) = tanh(W @ A(t))``, optionally perturbed by observation noise. These
synthetic trajectories are the supervised dataset (``A(t) -> A(t+1)`` pairs)
used to train and evaluate QFCM, classical FCM, and MLP models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr


def make_W_random(
    n_concepts: int,
    sparsity: float,
    seed: int,
    min_rho: float = 0.5,
    max_rho: float = 1.0,
) -> np.ndarray:
    """
    Random FCM weight matrix with controlled sparsity.

    Tries to get ``min_rho <= rho < max_rho``, but falls back to the best
    candidate with ``rho < max_rho`` if necessary. ``rho`` is the spectral
    radius of ``W``, kept below 1 so the noiseless dynamics converge.
    """
    rng = np.random.default_rng(seed)

    positions = [(i, j) for i in range(n_concepts) for j in range(n_concepts) if i != j]
    n_nonzero = int(len(positions) * sparsity)

    best_W = None
    best_rho = -np.inf

    for _ in range(100):
        W = np.zeros((n_concepts, n_concepts), dtype=float)

        if n_nonzero > 0:
            chosen = rng.choice(len(positions), size=n_nonzero, replace=False)
            for idx in chosen:
                i, j = positions[idx]
                W[i, j] = rng.uniform(-1.0, 1.0)

        rho = np.max(np.abs(np.linalg.eigvals(W)))

        if min_rho <= rho < max_rho:
            return W

        if rho < max_rho and rho > best_rho:
            best_rho = rho
            best_W = W

    if best_W is not None:
        print(
            f"[WARN] Could not find W with {min_rho} <= rho < {max_rho} "
            f"in 100 attempts. Using best rho={best_rho:.4f} instead."
        )
        return best_W

    raise ValueError(f"Could not find W with rho < {max_rho} in 100 attempts")


def check_W(W: np.ndarray, name: str) -> tuple[float, float]:
    """Diagnostic checks on a weight matrix before dataset generation."""
    rho = np.max(np.abs(np.linalg.eigvals(W)))
    sparsity = np.sum(W == 0) / W.size
    n_nz = np.sum(W != 0)

    nonzero_vals = W[W != 0]
    w_min = nonzero_vals.min() if len(nonzero_vals) > 0 else 0.0
    w_max = nonzero_vals.max() if len(nonzero_vals) > 0 else 0.0

    print(f"\n{'=' * 40}")
    print(f"  {name}")
    print(f"{'=' * 40}")
    print(
        f"  rho(W)    = {rho:.4f}  "
        f"{'[OK] rho < 1' if rho < 1 else '[WARNING] rho >= 1'}"
    )
    print(f"  Sparsity  = {sparsity:.1%}  ({n_nz} non-zero edges)")
    print(f"  W range   = [{w_min:.2f}, {w_max:.2f}]")
    return rho, sparsity


def simulate_fcm(
    W: np.ndarray,
    A0: np.ndarray,
    rng: np.random.Generator,
    *,
    T_max: int = 300,
    conv_thr: float = 1e-4,
    noise_std: float = 0.03,
) -> tuple[np.ndarray, int]:
    """
    Simulate one FCM trajectory from initial condition ``A0``.

    Update rule: ``A(t+1) = tanh(W @ A(t))``.

    Gaussian observation noise is added at each step to simulate measurement
    uncertainty. Convergence is checked on the clean (noiseless) state.
    """
    A = np.asarray(A0, dtype=float).copy()
    trajectory = [A.copy()]
    t_conv = T_max

    for t in range(T_max - 1):
        A_next = np.tanh(W @ A)
        noise = rng.normal(0.0, noise_std, size=A.shape)

        if np.linalg.norm(A_next - A) < conv_thr:
            t_conv = t + 1
            trajectory.append(np.clip(A_next + noise, -1.0, 1.0))
            break

        trajectory.append(np.clip(A_next + noise, -1.0, 1.0))
        A = A_next

    return np.array(trajectory), t_conv


def generate_dataset(
    W: np.ndarray,
    *,
    M: int = 50,
    seed: int = 0,
    T_max: int = 300,
    conv_thr: float = 1e-4,
    noise_std: float = 0.03,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Generate M trajectories from random initial conditions A0 ~ U[-1,1]^N."""
    rng = np.random.default_rng(seed)
    n_concepts = W.shape[0]
    A0_list = rng.uniform(-1.0, 1.0, size=(M, n_concepts))

    trajectories = []
    conv_times = []
    fixed_points = []

    for m in range(M):
        traj, t_conv = simulate_fcm(
            W,
            A0_list[m],
            rng=rng,
            T_max=T_max,
            conv_thr=conv_thr,
            noise_std=noise_std,
        )
        trajectories.append(traj)
        conv_times.append(t_conv)
        fixed_points.append(traj[-1])

    return trajectories, np.array(conv_times), np.array(fixed_points)


def validate_dataset(
    trajectories: list[np.ndarray],
    conv_times: np.ndarray,
    fixed_points: np.ndarray,
    W_true: np.ndarray,
    *,
    T_max: int = 300,
    noise_std: float = 0.03,
    corr_threshold: float = 0.3,
    name: str = "",
) -> tuple[float, float, int]:
    """Three-level validation of the generated dataset (convergence, diversity, causal plausibility)."""
    n_concepts = W_true.shape[0]

    print(f"\n{'=' * 40}")
    print(f"  Dataset validation: {name}")
    print(f"{'=' * 40}")

    pct_converged = np.mean(conv_times < T_max) * 100
    mean_conv = conv_times[conv_times < T_max].mean() if pct_converged > 0 else T_max

    print("\n  [a] Convergence")
    print(f"       % converged:       {pct_converged:.0f}%  (target: > 95%)")
    print(f"       Mean steps (conv): {mean_conv:.1f}")
    print(f"       Status: {'[OK]' if pct_converged >= 95 else '[WARN]'}")

    all_steps = np.vstack(trajectories)
    A0_arr = np.array([t[0] for t in trajectories])
    fp_arr = np.array(fixed_points)

    per_concept_std = all_steps.std(axis=0)
    if noise_std > 0:
        per_concept_snr = per_concept_std / noise_std
        snr_mean = per_concept_snr.mean()
        snr_min = per_concept_snr.min()
    else:
        snr_mean = np.inf
        snr_min = np.inf

    print("\n  [b] Trajectory diversity")
    print(f"       Spread of initial conditions: {A0_arr.std(axis=0).mean():.4f}")
    print(f"       Spread of fixed points:       {fp_arr.std(axis=0).mean():.4f}")
    print(f"       Mean SNR:                     {snr_mean:.2f}  (target: > 2)")
    print(f"       Min SNR across concepts:      {snr_min:.2f}")
    print(f"       Status: {'[OK]' if snr_mean > 2 else '[WARN]'}")

    print("\n  [c] Causal plausibility (Pearson r along full trajectories)")
    correlations = []
    sign_violations, checks = 0, 0
    skipped_constant = 0

    for i in range(n_concepts):
        for j in range(n_concepts):
            if i != j and abs(W_true[i, j]) > corr_threshold:
                x = all_steps[:, i]
                y = all_steps[:, j]

                if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                    print(
                        f"       W[{i},{j}]={W_true[i, j]:+.1f}  r=nan  p=nan  [CONST]"
                    )
                    skipped_constant += 1
                    continue

                r, pval = pearsonr(x, y)
                correlations.append((r, pval))

                sign_ok = np.sign(r) == np.sign(W_true[i, j])
                checks += 1
                if not sign_ok:
                    sign_violations += 1

                sig = "*" if pval < 0.05 else " "
                match = "[OK]" if sign_ok else "[!=]"
                print(
                    f"       W[{i},{j}]={W_true[i, j]:+.1f}  "
                    f"r={r:+.3f}{sig}  p={pval:.3f}  {match}"
                )

    n_sig = sum(1 for _, p in correlations if p < 0.05)

    print("       (* = statistically significant, p < 0.05)")
    if checks > 0:
        print(f"       Significant correlations: {n_sig}/{checks}")
        print(
            f"       Sign mismatches: {sign_violations}/{checks}  (informational only)"
        )
    else:
        print("       Significant correlations: 0/0")
        print("       Sign mismatches: 0/0  (informational only)")
    print(f"       Constant-input pairs skipped: {skipped_constant}")
    print("       Status: [OK]")

    return pct_converged, snr_mean, n_sig


def build_training_set(
    trajectories: list[np.ndarray],
    *,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract (A(t), A(t+1)) pairs from all trajectories and split into
    train/test sets by scenario (a whole trajectory goes to either split,
    to avoid leaking information between adjacent timesteps).
    """
    if not (0.0 < test_fraction < 1.0):
        raise ValueError("test_fraction must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    M = len(trajectories)
    idx = np.arange(M)
    rng.shuffle(idx)

    n_test = max(1, int(M * test_fraction))
    test_idx = set(idx[:n_test])
    train_idx = set(idx[n_test:])

    def extract_pairs(indices):
        X, Y = [], []
        for m in indices:
            traj = trajectories[m]
            for t in range(len(traj) - 1):
                X.append(traj[t])
                Y.append(traj[t + 1])
        return np.array(X), np.array(Y)

    X_train, Y_train = extract_pairs(train_idx)
    X_test, Y_test = extract_pairs(test_idx)

    print(f"\n  Training pairs: {X_train.shape[0]}")
    print(f"  Test pairs:     {X_test.shape[0]}")
    return X_train, Y_train, X_test, Y_test


def save_dataset_summary_txt(
    name: str,
    data_dict: dict[str, Any],
    output_dir: str | Path = ".",
) -> None:
    """Save a human-readable TXT summary of one dataset."""
    out_path = Path(output_dir) / f"dataset_{name}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"Dataset: {name}\n")
        f.write("=" * 60 + "\n\n")

        f.write("W_true:\n")
        np.savetxt(f, data_dict["W_true"], fmt="%.6f")
        f.write("\n")

        f.write(f"Number of trajectories: {len(data_dict['trajectories'])}\n")
        f.write(f"Convergence times shape: {data_dict['conv_times'].shape}\n")
        f.write(f"Fixed points shape:      {data_dict['fixed_points'].shape}\n")
        f.write(f"X_train shape:           {data_dict['X_train'].shape}\n")
        f.write(f"Y_train shape:           {data_dict['Y_train'].shape}\n")
        f.write(f"X_test shape:            {data_dict['X_test'].shape}\n")
        f.write(f"Y_test shape:            {data_dict['Y_test'].shape}\n\n")

        f.write("Convergence times:\n")
        np.savetxt(f, data_dict["conv_times"].reshape(-1, 1), fmt="%d")
        f.write("\n")

        f.write("Fixed points:\n")
        np.savetxt(f, data_dict["fixed_points"], fmt="%.6f")
        f.write("\n")

        f.write("First 5 trajectories:\n")
        for k, traj in enumerate(data_dict["trajectories"][:5]):
            f.write(f"\nTrajectory {k} - shape {traj.shape}\n")
            np.savetxt(f, traj, fmt="%.6f")
