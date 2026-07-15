"""
Artifact saving utilities for HQC-FCM benchmark experiments.

This module is the single place responsible for writing experiment outputs
to disk (checkpoints aside, which are written by
:mod:`hqcfcm.training.train`). In particular, QFCM postprocessing results
(computed by :mod:`hqcfcm.postprocessing.w_recovery`, which performs no I/O
of its own) are saved here exactly once.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def save_common_artifacts(
    *,
    name: str,
    model,
    history: dict,
    X_train,
    Y_train,
    X_test,
    Y_test,
    train_indices=None,
    test_indices=None,
    save_dir: str = "results",
    extra_config: dict[str, Any] | None = None,
    run_metrics: dict[str, Any] | None = None,
) -> None:
    """
    Save experiment artifacts shared by all models.

    Saved items include:
    - train/test inputs, targets, and predictions
    - training history as CSV
    - metadata JSON
    - tidy CSV of test predictions
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)
    X_test = np.asarray(X_test)
    Y_test = np.asarray(Y_test)

    model.eval()
    with torch.no_grad():
        Xtr_t = torch.tensor(X_train, dtype=torch.get_default_dtype())
        Xte_t = torch.tensor(X_test, dtype=torch.get_default_dtype())

        Ytr_pred = model(Xtr_t).detach().cpu().numpy()
        Yte_pred = model(Xte_t).detach().cpu().numpy()

    artifacts = {
        "name": name,
        "timestamp": datetime.now().isoformat(),
        "history": history,
        "config": extra_config if extra_config is not None else {},
        "run_metrics": run_metrics if run_metrics is not None else {},
        "train_indices": None
        if train_indices is None
        else np.asarray(train_indices).tolist(),
        "test_indices": None
        if test_indices is None
        else np.asarray(test_indices).tolist(),
    }

    if hasattr(model, "n_parameters"):
        artifacts["n_parameters"] = int(model.n_parameters)

    with (save_dir / f"{name}_artifacts.json").open("w", encoding="utf-8") as f:
        json.dump(artifacts, f, indent=2)

    np.save(save_dir / f"{name}_X_train.npy", X_train)
    np.save(save_dir / f"{name}_Y_train.npy", Y_train)
    np.save(save_dir / f"{name}_Y_train_pred.npy", Ytr_pred)

    np.save(save_dir / f"{name}_X_test.npy", X_test)
    np.save(save_dir / f"{name}_Y_test.npy", Y_test)
    np.save(save_dir / f"{name}_Y_test_pred.npy", Yte_pred)

    pd.DataFrame(history).to_csv(save_dir / f"{name}_history.csv", index=False)

    n_concepts = Y_test.shape[1]
    test_df = pd.DataFrame(X_test, columns=[f"A{i}_t" for i in range(n_concepts)])
    for i in range(n_concepts):
        test_df[f"A{i}_t1_true"] = Y_test[:, i]
        test_df[f"A{i}_t1_pred"] = Yte_pred[:, i]

    test_df.to_csv(save_dir / f"{name}_test_predictions.csv", index=False)

    print(f"  Common artifacts saved under: {save_dir}/ (prefix: {name})")


def save_qfcm_artifacts(
    *,
    name: str,
    model,
    save_dir: str = "results",
    W_true=None,
    post_result: dict | None = None,
) -> None:
    """
    Save QFCM-specific artifacts.

    ``post_result`` is the dict returned by
    :func:`hqcfcm.postprocessing.w_recovery.recover_qfcm_weight_matrix`; this
    function is the only place where it is written to disk.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    np.save(save_dir / f"{name}_theta.npy", model.theta.detach().cpu().numpy())
    np.save(save_dir / f"{name}_alpha_full.npy", model.alpha.detach().cpu().numpy())
    np.save(
        save_dir / f"{name}_alpha_masked.npy",
        model.alpha_masked.detach().cpu().numpy(),
    )
    np.save(save_dir / f"{name}_mask.npy", model.mask.detach().cpu().numpy())

    if W_true is not None:
        np.save(save_dir / f"{name}_W_true.npy", np.asarray(W_true))

    if post_result is not None:
        for key in ("W_lstsq", "b_lstsq", "C_grid", "Y_grid_pred"):
            if key in post_result:
                np.save(save_dir / f"{name}_{key}.npy", np.asarray(post_result[key]))

    print(f"  QFCM artifacts saved under: {save_dir}/ (prefix: {name})")


def save_cfcm_artifacts(
    *,
    name: str,
    model,
    save_dir: str = "results",
    W_true=None,
) -> None:
    """Save Classical FCM-specific artifacts."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    np.save(save_dir / f"{name}_W.npy", model.W.detach().cpu().numpy())
    np.save(save_dir / f"{name}_W_masked.npy", model.W_masked.detach().cpu().numpy())
    np.save(save_dir / f"{name}_b.npy", model.b.detach().cpu().numpy())
    np.save(save_dir / f"{name}_mask.npy", model.mask.detach().cpu().numpy())

    if W_true is not None:
        np.save(save_dir / f"{name}_W_true.npy", np.asarray(W_true))

    print(f"  CFCM artifacts saved under: {save_dir}/ (prefix: {name})")


def save_mlp_artifacts(
    *,
    name: str,
    model,
    save_dir: str = "results",
) -> None:
    """
    Save MLP-specific artifacts.

    For the MLP we save the full state_dict for reproducibility, and the
    first-layer weight matrix for diagnostics only (it is not an FCM causal
    matrix).
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), save_dir / f"{name}_model_state.pt")

    if hasattr(model, "first_layer_weight"):
        first_w = model.first_layer_weight().detach().cpu().numpy()
        np.save(save_dir / f"{name}_first_layer_weight.npy", first_w)

    print(f"  MLP artifacts saved under: {save_dir}/ (prefix: {name})")


def save_model_artifacts(
    *,
    model_type: str,
    name: str,
    model,
    history: dict,
    X_train,
    Y_train,
    X_test,
    Y_test,
    train_indices=None,
    test_indices=None,
    save_dir: str = "results",
    extra_config: dict[str, Any] | None = None,
    run_metrics: dict[str, Any] | None = None,
    W_true=None,
    post_result: dict | None = None,
) -> None:
    """
    Unified artifact-saving entry point.

    Parameters
    ----------
    model_type : str
        One of {"qfcm", "cfcm", "mlp"}.
    name : str
        Run name prefix.
    model : torch.nn.Module
        Trained model.
    history : dict
        Training history.
    X_train, Y_train, X_test, Y_test : array-like
        Train/test inputs and targets.
    train_indices, test_indices : array-like or None
        Sampled row indices from the original CSV files.
    save_dir : str
        Base output directory.
    extra_config : dict | None
        Extra configuration to store in metadata.
    run_metrics : dict | None
        Runtime metrics such as train time.
    W_true : array-like or None
        Optional ground-truth FCM matrix.
    post_result : dict | None
        Optional QFCM postprocessing output, see
        :func:`hqcfcm.postprocessing.w_recovery.recover_qfcm_weight_matrix`.
    """
    save_common_artifacts(
        name=name,
        model=model,
        history=history,
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        train_indices=train_indices,
        test_indices=test_indices,
        save_dir=save_dir,
        extra_config=extra_config,
        run_metrics=run_metrics,
    )

    if model_type == "qfcm":
        save_qfcm_artifacts(
            name=name, model=model, save_dir=save_dir, W_true=W_true,
            post_result=post_result,
        )
    elif model_type == "cfcm":
        save_cfcm_artifacts(name=name, model=model, save_dir=save_dir, W_true=W_true)
    elif model_type == "mlp":
        save_mlp_artifacts(name=name, model=model, save_dir=save_dir)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
