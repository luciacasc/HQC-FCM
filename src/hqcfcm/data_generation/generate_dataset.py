"""
CLI entry point for synthetic FCM dataset generation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from hqcfcm.data_generation.fcm_generator import (
    build_training_set,
    check_W,
    generate_dataset,
    make_W_random,
    save_dataset_summary_txt,
    validate_dataset,
)


def load_yaml_config(config_path: str | Path) -> dict:
    """Load a YAML config file safely."""
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_W_from_config(dataset_cfg: dict) -> np.ndarray:
    """
    Build the ground-truth FCM matrix W from the dataset config.

    Supported modes:
      - matrix.type = "custom"
      - matrix.type = "random"
    """
    n_concepts = int(dataset_cfg["n_concepts"])
    matrix_cfg = dataset_cfg["matrix"]
    matrix_type = matrix_cfg["type"]

    if matrix_type == "custom":
        W = np.array(matrix_cfg["values"], dtype=float)
        if W.shape != (n_concepts, n_concepts):
            raise ValueError(
                f"Custom W must have shape ({n_concepts}, {n_concepts}), got {W.shape}"
            )
        return W

    if matrix_type == "random":
        return make_W_random(
            n_concepts=n_concepts,
            sparsity=float(matrix_cfg["sparsity"]),
            seed=int(dataset_cfg["master_seed"] + matrix_cfg.get("seed_offset", 0)),
            min_rho=float(matrix_cfg.get("min_rho", 0.5)),
            max_rho=float(matrix_cfg.get("max_rho", 1.0)),
        )

    raise ValueError(f"Unsupported matrix.type: {matrix_type}")


def generate_from_config(config_path: str | Path) -> None:
    """Generate one synthetic dataset from a YAML config file."""
    config = load_yaml_config(config_path)
    dataset_cfg = config["dataset"]

    name = dataset_cfg["name"]
    output_dir = Path(dataset_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    n_concepts = int(dataset_cfg["n_concepts"])
    m_scenarios = int(dataset_cfg["m_scenarios"])
    t_max = int(dataset_cfg["t_max"])
    conv_thr = float(dataset_cfg["conv_thr"])
    noise_std = float(dataset_cfg["noise_std"])
    master_seed = int(dataset_cfg["master_seed"])
    test_fraction = float(dataset_cfg.get("test_fraction", 0.2))
    corr_threshold = float(dataset_cfg.get("corr_threshold", 0.3))

    W = build_W_from_config(dataset_cfg)
    check_W(W, name)

    trajectories, conv_times, fixed_points = generate_dataset(
        W,
        M=m_scenarios,
        seed=master_seed,
        T_max=t_max,
        conv_thr=conv_thr,
        noise_std=noise_std,
    )

    validate_dataset(
        trajectories,
        conv_times,
        fixed_points,
        W,
        T_max=t_max,
        noise_std=noise_std,
        corr_threshold=corr_threshold,
        name=name,
    )

    X_train, Y_train, X_test, Y_test = build_training_set(
        trajectories,
        test_fraction=test_fraction,
        seed=master_seed,
    )

    cols = [f"A{i}_t" for i in range(n_concepts)] + [
        f"A{i}_t1" for i in range(n_concepts)
    ]

    train_df = pd.DataFrame(np.hstack([X_train, Y_train]), columns=cols)
    test_df = pd.DataFrame(np.hstack([X_test, Y_test]), columns=cols)

    train_path = output_dir / f"train_{name}.csv"
    test_path = output_dir / f"test_{name}.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    dataset_dict = {
        "W_true": W,
        "trajectories": trajectories,
        "conv_times": conv_times,
        "fixed_points": fixed_points,
        "X_train": X_train,
        "Y_train": Y_train,
        "X_test": X_test,
        "Y_test": Y_test,
    }

    save_dataset_summary_txt(name, dataset_dict, output_dir=output_dir)

    print("\nSaved:")
    print(f"  - {train_path}")
    print(f"  - {test_path}")
    print(f"  - {output_dir / f'dataset_{name}.txt'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate one synthetic FCM dataset from a YAML config."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a dataset YAML config, e.g. configs/dataset/motivated.yaml",
    )
    args = parser.parse_args()

    generate_from_config(args.config)


if __name__ == "__main__":
    main()
