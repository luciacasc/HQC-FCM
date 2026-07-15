"""
Autoregressive rollout demo.

Trains a small QFCM model on the bundled "motivated" dataset config, then
uses hqcfcm.postprocessing.rollout to compare a ground-truth trajectory
against the model's own autoregressive rollout from the same initial
condition, and plots the result.

This replaces an ad-hoc, hard-coded analysis script from the original
repository (case_of_study_dyn.py) with a self-contained, portable example
that works with any trained checkpoint and any n_concepts.

Usage
-----
    python examples/rollout_demo.py --config configs/models/smoketest.yaml \\
        --dataset-config configs/dataset/motivated.yaml --scenario 0
"""

from __future__ import annotations

import argparse

import numpy as np

from hqcfcm.data_generation.fcm_generator import generate_dataset
from hqcfcm.main import flatten_experiment_config, load_yaml_config, run_experiment
from hqcfcm.postprocessing.rollout import rollout_mse_per_concept, rollout_trajectory
from hqcfcm.utils.plotting import plot_trajectory_rollout


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True,
        help="Model/training YAML config, e.g. configs/models/smoketest.yaml",
    )
    parser.add_argument(
        "--dataset-config", required=True,
        help="Dataset YAML config used to regenerate the ground-truth "
        "trajectories, e.g. configs/dataset/motivated.yaml",
    )
    parser.add_argument(
        "--scenario", type=int, default=0,
        help="Index of the test scenario to roll out (default: 0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Train (or re-train) the model from the experiment config.
    raw_config = load_yaml_config(args.config)
    config = flatten_experiment_config(raw_config)
    result = run_experiment(config)
    model = result["model"]

    # 2. Regenerate the same ground-truth trajectories used to build the
    #    training/test CSVs, so we have a full multi-step trajectory to
    #    compare against (the CSVs only contain one-step (t, t+1) pairs).
    dataset_cfg = load_yaml_config(args.dataset_config)["dataset"]
    W_true = np.array(dataset_cfg["matrix"]["values"], dtype=float)

    trajectories, _, _ = generate_dataset(
        W_true,
        M=dataset_cfg["m_scenarios"],
        seed=dataset_cfg["master_seed"],
        T_max=dataset_cfg["t_max"],
        conv_thr=dataset_cfg["conv_thr"],
        noise_std=dataset_cfg["noise_std"],
    )
    ground_truth = trajectories[args.scenario]

    # 3. Roll the model out autoregressively from the same initial condition.
    predicted = rollout_trajectory(model, ground_truth[0], n_steps=len(ground_truth) - 1)

    # 4. Report per-concept MSE and save a comparison plot.
    mse = rollout_mse_per_concept(ground_truth, predicted)
    for i, m in enumerate(mse):
        print(f"  A{i}: rollout MSE = {m:.5f}")

    plot_trajectory_rollout(
        ground_truth=ground_truth,
        predicted=predicted,
        name=f"{config['name']}_scenario{args.scenario}",
        save_dir=config.get("save_dir", "results"),
    )


if __name__ == "__main__":
    main()
