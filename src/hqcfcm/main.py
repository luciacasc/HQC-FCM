"""
Main entry point for HQC-FCM benchmark experiments.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from hqcfcm.models.cfcm import ClassicalFCM
from hqcfcm.models.mlp import MLPFCM
from hqcfcm.models.qfcm import QFCM
from hqcfcm.postprocessing.w_recovery import (
    extract_cfcm_weight_matrix,
    recover_qfcm_weight_matrix,
)
from hqcfcm.training.artifacts import save_model_artifacts
from hqcfcm.training.train import train_model
from hqcfcm.utils.io import load_and_sample_data
from hqcfcm.utils.plotting import plot_model_results


def load_yaml_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\nCurrent working directory: {Path.cwd()}"
        )
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Run an HQC-FCM benchmark experiment.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file."
    )
    return parser.parse_args()


def flatten_experiment_config(raw_config: dict) -> dict:
    """
    Convert a nested YAML config into a flat config dictionary expected by
    the rest of the codebase.

    Expected YAML structure:
      experiment:
      data:
      model:
      training:
      verbose:
      W_true:
    """
    exp_cfg = raw_config.get("experiment", {})
    data_cfg = raw_config.get("data", {})
    model_cfg = raw_config.get("model", {})
    train_cfg = raw_config.get("training", {})
    n_train = data_cfg.get("n_train")
    n_test = data_cfg.get("n_test")

    config = {
        "name": exp_cfg["name"],
        "save_dir": exp_cfg.get("save_dir", "results"),
        "seed": int(exp_cfg.get("seed", 42)),
        "train_csv": data_cfg["train_csv"],
        "test_csv": data_cfg["test_csv"],
        "n_train": int(n_train) if n_train is not None else None,
        "n_test": int(n_test) if n_test is not None else None,
        "model_type": model_cfg["model_type"],
        "n_concepts": int(model_cfg["n_concepts"]),
        "verbose": raw_config.get("verbose", True),
    }

    if config["model_type"] == "qfcm":
        config["n_layers"] = int(model_cfg["n_layers"])
        config["lambda_l1"] = float(train_cfg.get("lambda_l1", 1e-4))
        config["grid_points"] = int(train_cfg.get("grid_points", 20))
        config["diff_method"] = str(model_cfg.get("diff_method", "best"))
        config["entanglement"] = str(model_cfg.get("entanglement", "none"))
        config["device_name"] = str(model_cfg.get("device_name", "lightning.qubit"))

    elif config["model_type"] == "cfcm":
        config["lambda_l1"] = float(train_cfg.get("lambda_l1", 1e-4))
        config["use_bias"] = bool(model_cfg.get("use_bias", True))

    elif config["model_type"] == "mlp":
        config["hidden_dims"] = model_cfg.get("hidden_dims", [16])
        config["lambda_l1"] = float(train_cfg.get("lambda_l1", 1e-5))
        config["use_bias"] = bool(model_cfg.get("use_bias", False))
        config["use_bias"] = bool(model_cfg.get("use_bias", False))

    else:
        raise ValueError(f"Unknown model_type: {config['model_type']}")

    config["lr"] = float(train_cfg.get("lr", 5e-3))
    config["n_epochs"] = int(train_cfg.get("n_epochs", 300))
    config["batch_size"] = int(train_cfg.get("batch_size", 8))
    config["patience"] = int(train_cfg.get("patience", 75))

    if raw_config.get("W_true") is not None:
        config["W_true"] = raw_config["W_true"]

    return config


def build_model(config: dict):
    model_type = config["model_type"]
    n_concepts = config["n_concepts"]
    seed = config.get("seed", 42)

    if model_type == "qfcm":
        return QFCM(
            n_concepts=n_concepts,
            n_layers=config["n_layers"],
            lambda_l1=config.get("lambda_l1", 1e-4),
            seed=seed,
            device_name=config.get("device_name", "lightning.qubit"),
            diff_method=config.get("diff_method", "best"),
            entanglement=config.get("entanglement", "none"),
        )

    if model_type == "cfcm":
        return ClassicalFCM(
            n_concepts=n_concepts,
            lambda_l1=config.get("lambda_l1", 1e-4),
            use_bias=config.get("use_bias", True),
            seed=seed,
        )

    if model_type == "mlp":
        hidden_dims = tuple(config.get("hidden_dims", [16]))
        return MLPFCM(
            n_concepts=n_concepts,
            hidden_dims=hidden_dims,
            lambda_l1=config.get("lambda_l1", 1e-5),
            use_bias=config.get("use_bias", False),
            seed=seed,
        )

    raise ValueError(f"Unknown model_type: {model_type}")


def maybe_get_W_true(config: dict):
    W_true = config.get("W_true")
    if W_true is None:
        return None
    return np.asarray(W_true, dtype=float)


def print_run_summary(config: dict, data: dict) -> None:
    print("=" * 70)
    print("HQC-FCM benchmark run")
    print("=" * 70)
    print(f"Model type   : {config['model_type']}")
    print(f"Run name     : {config['name']}")
    print(f"N concepts   : {config['n_concepts']}")
    print(f"Train / test : {len(data['X_train'])} / {len(data['X_test'])}")
    print(f"Train CSV    : {config['train_csv']}")
    print(f"Test CSV     : {config['test_csv']}")

    if config["model_type"] == "qfcm":
        print(f"N layers     : {config['n_layers']}")
        print(f"Diff method   : {config.get('diff_method', 'best')}")
        print(f"Entanglement  : {config.get('entanglement', 'none')}")
        print(f"Device        : {config.get('device_name', 'lightning.qubit')}")

    if config["model_type"] == "mlp":
        print(f"Hidden dims  : {config.get('hidden_dims', [16])}")

    print(f"Save dir     : {config.get('save_dir', 'results')}")
    print("=" * 70)


def run_experiment(config: dict):
    """Run a full experiment from config."""
    torch.set_default_dtype(torch.float64)

    data = load_and_sample_data(config)

    X_train, Y_train = data["X_train"], data["Y_train"]
    X_test, Y_test = data["X_test"], data["Y_test"]
    idx_tr, idx_te = data["idx_tr"], data["idx_te"]

    model = build_model(config)
    W_true = maybe_get_W_true(config)

    print_run_summary(config, data)

    train_result = train_model(
        model=model,
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        lr=config.get("lr", 5e-3),
        n_epochs=config.get("n_epochs", 300),
        batch_size=config.get("batch_size", 8),
        seed=config.get("seed", 42),
        verbose=config.get("verbose", True),
        save_dir=config.get("save_dir", "results"),
        name=config["name"],
        patience=config.get("patience", 75),
        extra_config=config,
    )

    # --- Postprocessing: recover an effective, interpretable W matrix. ---
    # This step is deliberately separate from the model class itself (see
    # hqcfcm.postprocessing.w_recovery), so it can evolve independently.
    post_result = None
    W_learned = None

    if config["model_type"] == "cfcm":
        W_learned = extract_cfcm_weight_matrix(train_result["model"])

    elif config["model_type"] == "qfcm":
        post_result = recover_qfcm_weight_matrix(
            train_result["model"],
            grid_points=config.get("grid_points", 20),
        )
        W_learned = post_result["W_lstsq"]

    # MLP has no interpretable causal matrix; W_learned stays None.

    save_model_artifacts(
        model_type=config["model_type"],
        name=config["name"],
        model=train_result["model"],
        history=train_result["history"],
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        train_indices=idx_tr,
        test_indices=idx_te,
        save_dir=config.get("save_dir", "results"),
        extra_config=config,
        W_true=W_true,
        post_result=post_result,
        run_metrics=train_result.get("timing", {}),
    )

    plot_model_results(
        model_type=config["model_type"],
        train_result=train_result,
        name=config["name"],
        save_dir=config.get("save_dir", "results"),
        W_true=W_true,
        W_learned=W_learned,
        post_result=post_result,
        n_concepts=config["n_concepts"],
    )

    print("\nRun completed successfully.")

    return {
        "model": train_result["model"],
        "train_result": train_result,
        "post_result": post_result,
        "W_true": W_true,
        "W_learned": W_learned,
        "data": data,
    }


def main():
    args = parse_args()
    config = load_yaml_config(args.config)
    run_experiment(flatten_experiment_config(config))


if __name__ == "__main__":
    main()
