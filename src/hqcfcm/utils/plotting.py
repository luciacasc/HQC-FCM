"""
Plotting utilities for HQC-FCM benchmark experiments.
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_plot_dir(save_dir: str) -> str:
    plot_dir = os.path.join(save_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir


def plot_training_curves(
    *,
    history: dict,
    name: str,
    save_dir: str = "results",
    norm_key: str = "param_norm",
    norm_title: str = "Parameter norm",
):
    """
    Plot training loss, test MSE, and a norm-like diagnostic curve.

    Parameters
    ----------
    history : dict
        Training history dictionary.
    name : str
        Run name prefix.
    save_dir : str
        Directory where ``plots/`` will be created.
    norm_key : str
        Key used in history for the third diagnostic curve.
    norm_title : str
        Plot title for the third curve.
    """
    plot_dir = _ensure_plot_dir(save_dir)
    epochs = range(len(history["train_loss"]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(
        epochs, history["train_loss"], label="train loss",
        color="steelblue", linewidth=1.5,
    )
    axes[0].plot(
        epochs, history["test_mse"], label="test MSE",
        color="seagreen", linewidth=1.5, linestyle="--",
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training curves")
    axes[0].set_yscale("log")
    axes[0].legend()

    if norm_key in history:
        axes[1].plot(epochs, history[norm_key], color="mediumpurple", linewidth=1.5)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("L1 norm")
        axes[1].set_title(norm_title)
    else:
        axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{name}_training_curves.png"), dpi=150)
    plt.close()


def plot_weight_heatmaps(
    *,
    W_true: np.ndarray,
    W_learned: np.ndarray,
    name: str,
    n_concepts: int,
    save_dir: str = "results",
    learned_label: str = "W learned",
):
    """Plot heatmaps of W_true, W_learned, and absolute error."""
    plot_dir = _ensure_plot_dir(save_dir)

    labels = [f"A{i}" for i in range(n_concepts)]
    vmax = max(np.abs(W_true).max(), np.abs(W_learned).max())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, (W, title) in zip(
        axes[:2], [(W_true, "W true"), (W_learned, learned_label)]
    ):
        im = ax.imshow(W, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(n_concepts))
        ax.set_xticklabels(labels)
        ax.set_yticks(range(n_concepts))
        ax.set_yticklabels(labels)
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

        for i in range(n_concepts):
            for j in range(n_concepts):
                ax.text(j, i, f"{W[i, j]:.3f}", ha="center", va="center", fontsize=9)

    W_err = np.abs(W_true - W_learned)
    im = axes[2].imshow(W_err, cmap="Reds", vmin=0.0)
    axes[2].set_xticks(range(n_concepts))
    axes[2].set_xticklabels(labels)
    axes[2].set_yticks(range(n_concepts))
    axes[2].set_yticklabels(labels)
    axes[2].set_title("Absolute error")
    plt.colorbar(im, ax=axes[2])

    for i in range(n_concepts):
        for j in range(n_concepts):
            axes[2].text(
                j, i, f"{W_err[i, j]:.3f}", ha="center", va="center", fontsize=9
            )

    plt.suptitle(f"{name} — W recovery", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{name}_W_heatmap.png"), dpi=150)
    plt.close()


def plot_qfcm_functional_response(
    *,
    C_grid: np.ndarray,
    Y_grid_pred: np.ndarray,
    W_lstsq: np.ndarray,
    name: str,
    save_dir: str = "results",
):
    """Plot QFCM functional response contours for the 2-concept case only."""
    plot_dir = _ensure_plot_dir(save_dir)

    if C_grid.shape[1] != 2:
        raise ValueError("Functional response plot is supported only for n_concepts=2.")

    g_size = int(np.sqrt(len(C_grid)))
    g = np.linspace(-1.0, 1.0, g_size)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for k, ax in enumerate(axes):
        Z = Y_grid_pred[:, k].reshape(g_size, g_size)

        im = ax.contourf(g, g, Z, levels=20, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
        ax.set_xlabel("A0(t)")
        ax.set_ylabel("A1(t)")
        ax.set_title(f"Prediction A{k}(t+1)")
        plt.colorbar(im, ax=ax)

        Z_th = np.tanh(W_lstsq[k, 0] * g[:, None] + W_lstsq[k, 1] * g[None, :])
        ax.contour(
            g, g, Z_th.T, levels=10, colors="black",
            linewidths=0.8, linestyles="dashed", alpha=0.5,
        )

    plt.suptitle(
        "Functional response (black dashed lines = tanh(W_lstsq · x), bias ignored)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{name}_functional_response.png"), dpi=150)
    plt.close()


def plot_trajectory_rollout(
    *,
    ground_truth: np.ndarray,
    predicted: np.ndarray,
    name: str,
    save_dir: str = "results",
    concept_names: Optional[list[str]] = None,
):
    """
    Plot ground-truth vs. autoregressively predicted trajectories, one panel
    per concept. See :mod:`hqcfcm.postprocessing.rollout`.

    Parameters
    ----------
    ground_truth, predicted : np.ndarray
        Arrays of shape ``(T, n_concepts)``.
    name : str
        Run name prefix.
    save_dir : str
        Directory where ``plots/`` will be created.
    concept_names : list[str] | None
        Optional display names for each concept; defaults to ``A0, A1, ...``.
    """
    plot_dir = _ensure_plot_dir(save_dir)

    n_concepts = ground_truth.shape[1]
    if concept_names is None:
        concept_names = [f"A{i}" for i in range(n_concepts)]

    t_axis = np.arange(ground_truth.shape[0])

    fig, axes = plt.subplots(n_concepts, 1, figsize=(8, 2.2 * n_concepts), sharex=True)
    if n_concepts == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(t_axis, ground_truth[:, i], color="firebrick", lw=1.8, label="Ground truth")
        ax.plot(
            t_axis, predicted[:, i], color="royalblue", lw=1.4,
            linestyle="--", label="Model rollout",
        )
        ax.set_ylabel(concept_names[i])
        ax.set_ylim(-1.1, 1.1)
        if i == 0:
            ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Timestep $t$")
    plt.suptitle(f"{name} — autoregressive rollout vs. ground truth")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{name}_rollout.png"), dpi=150)
    plt.close()


def plot_model_results(
    *,
    model_type: str,
    train_result: dict,
    name: str,
    save_dir: str = "results",
    W_true: Optional[np.ndarray] = None,
    W_learned: Optional[np.ndarray] = None,
    post_result: Optional[dict] = None,
    n_concepts: Optional[int] = None,
):
    """
    Unified plotting entry point.

    Parameters
    ----------
    model_type : str
        One of {"qfcm", "cfcm", "mlp"}.
    """
    history = train_result["history"]

    norm_key = "param_norm"
    norm_title = "Parameter norm"

    if model_type == "qfcm":
        norm_key = "alpha_norm" if "alpha_norm" in history else "param_norm"
        norm_title = "Alpha norm"
    elif model_type == "cfcm":
        norm_key = "W_l1_norm" if "W_l1_norm" in history else "param_norm"
        norm_title = "W norm"

    plot_training_curves(
        history=history, name=name, save_dir=save_dir,
        norm_key=norm_key, norm_title=norm_title,
    )

    if W_true is not None and W_learned is not None and n_concepts is not None:
        learned_label = "W learned (lstsq)" if model_type == "qfcm" else "W learned"
        plot_weight_heatmaps(
            W_true=W_true, W_learned=W_learned, name=name,
            n_concepts=n_concepts, save_dir=save_dir, learned_label=learned_label,
        )

    if (
        model_type == "qfcm"
        and post_result is not None
        and n_concepts == 2
        and "C_grid" in post_result
        and "Y_grid_pred" in post_result
        and "W_lstsq" in post_result
    ):
        plot_qfcm_functional_response(
            C_grid=post_result["C_grid"],
            Y_grid_pred=post_result["Y_grid_pred"],
            W_lstsq=post_result["W_lstsq"],
            name=name,
            save_dir=save_dir,
        )

    print(f"  Plots saved under: {os.path.join(save_dir, 'plots')}/")
