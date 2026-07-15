"""
Generic training utilities for HQC-FCM benchmark models.

This module contains a model-agnostic training loop, assuming the model
implements:
- ``forward(...)``
- ``compute_loss(...)``
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from time import perf_counter
from typing import Any

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def train_model(
    model,
    X_train,
    Y_train,
    X_test,
    Y_test,
    *,
    lr: float = 5e-3,
    n_epochs: int = 300,
    batch_size: int = 8,
    seed: int = 42,
    verbose: bool = True,
    save_dir: str = "results",
    name: str = "run",
    patience: int = 75,
    extra_config: dict[str, Any] | None = None,
):
    """
    Train a model on the supervised task A(t) -> A(t+1).

    Parameters
    ----------
    model : torch.nn.Module
        Model implementing ``compute_loss(A_t, A_t1)``.
    X_train, Y_train, X_test, Y_test : array-like
        Train/test inputs and targets.
    lr : float
        Learning rate.
    n_epochs : int
        Maximum number of epochs.
    batch_size : int
        Mini-batch size.
    seed : int
        Random seed.
    verbose : bool
        If True, print periodic logs.
    save_dir : str
        Output directory.
    name : str
        Run name prefix.
    patience : int
        Early stopping patience based on test MSE.
    extra_config : dict | None
        Additional config fields to save in the checkpoint/run metadata.

    Returns
    -------
    dict
        Dictionary containing the trained model, history, timing, best score,
        and stopped epoch.
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)

    torch.manual_seed(seed)

    X_tr = torch.tensor(X_train, dtype=torch.get_default_dtype())
    Y_tr = torch.tensor(Y_train, dtype=torch.get_default_dtype())
    X_te = torch.tensor(X_test, dtype=torch.get_default_dtype())
    Y_te = torch.tensor(Y_test, dtype=torch.get_default_dtype())

    loader = DataLoader(
        TensorDataset(X_tr, Y_tr),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=50,
        min_lr=1e-5,
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "test_mse": [],
        "param_norm": [],
    }

    best_test_mse = float("inf")
    epochs_no_improve = 0
    stopped_epoch = 0

    start_time = perf_counter()
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0

        for X_batch, Y_batch in loader:
            optimizer.zero_grad()
            total_loss, mse_loss, reg_loss = model.compute_loss(X_batch, Y_batch)
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

        epoch_loss /= len(loader)

        model.eval()
        with torch.no_grad():
            _, test_mse, _ = model.compute_loss(X_te, Y_te)

            param_norm = 0.0
            for p in model.parameters():
                param_norm += p.abs().sum().item()

        scheduler.step(test_mse.item())

        history["train_loss"].append(epoch_loss)
        history["test_mse"].append(test_mse.item())
        history["param_norm"].append(param_norm)

        if test_mse.item() < best_test_mse:
            best_test_mse = test_mse.item()
            epochs_no_improve = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_test_mse": best_test_mse,
                "config": {
                    "lr": lr,
                    "batch_size": batch_size,
                    "n_epochs": n_epochs,
                    "patience": patience,
                    "seed": seed,
                    **(extra_config or {}),
                },
            }

            torch.save(checkpoint, f"{save_dir}/checkpoints/{name}_best.pt")
        else:
            epochs_no_improve += 1

        if verbose and (epoch % 50 == 0 or epoch == n_epochs - 1):
            print(
                f"  Epoch {epoch:4d} | "
                f"loss={epoch_loss:.5f} | "
                f"test_mse={test_mse.item():.5f} | "
                f"param_norm={param_norm:.4f} | "
                f"no_improve={epochs_no_improve}/{patience}"
            )

        stopped_epoch = epoch

        if epochs_no_improve >= patience:
            print(
                f"\n  Early stopping at epoch {epoch} "
                f"(no improvement for {patience} epochs)."
            )
            break

    end_time = perf_counter()
    train_time_sec = end_time - start_time

    ckpt_path = f"{save_dir}/checkpoints/{name}_best.pt"
    best_ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    run_data = {
        "name": name,
        "timestamp": datetime.now().isoformat(),
        "best_test_mse": best_test_mse,
        "stopped_epoch": stopped_epoch,
        "history": history,
        "config": {
            "lr": lr,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "patience": patience,
            "seed": seed,
            **(extra_config or {}),
        },
    }

    with open(f"{save_dir}/{name}_run.json", "w") as f:
        json.dump(run_data, f, indent=2)

    return {
        "model": model,
        "history": history,
        "timing": {"train_time_sec": train_time_sec},
        "best_test_mse": best_test_mse,
        "stopped_epoch": stopped_epoch,
    }
