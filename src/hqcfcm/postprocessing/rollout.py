"""
Autoregressive trajectory rollout for a trained model.

This generalizes an ad-hoc analysis that used to live in a throwaway,
hard-coded script (``case_of_study_dyn.py`` in the original repository): given
a trained model and an initial concept vector, repeatedly feed the model's
own one-step-ahead prediction back as its next input, producing a full
trajectory that can be compared qualitatively against a ground-truth
simulation from :mod:`hqcfcm.data_generation.fcm_generator`.

This is a qualitative diagnostic, complementary to the quantitative one-step
MSE reported during training: it shows whether small one-step errors
compound into a diverging trajectory over many steps, or whether the model
tracks the ground truth closely over the whole horizon.
"""

from __future__ import annotations

import numpy as np
import torch


def rollout_trajectory(
    model,
    A0: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    """
    Autoregressively roll out a trained model from an initial condition.

    Parameters
    ----------
    model : torch.nn.Module
        A trained model implementing ``forward(A_t) -> A_t1`` for a single
        sample of shape ``(n_concepts,)``.
    A0 : np.ndarray
        Initial concept vector, shape ``(n_concepts,)``.
    n_steps : int
        Number of steps to roll out (the returned trajectory has
        ``n_steps + 1`` rows, including ``A0``).

    Returns
    -------
    np.ndarray
        Predicted trajectory, shape ``(n_steps + 1, n_concepts)``.
    """
    A0 = np.asarray(A0, dtype=float)
    n_concepts = A0.shape[0]

    trajectory = np.zeros((n_steps + 1, n_concepts))
    trajectory[0] = A0

    model.eval()
    with torch.no_grad():
        for t in range(n_steps):
            x = torch.tensor(trajectory[t], dtype=torch.get_default_dtype())
            trajectory[t + 1] = model(x).cpu().numpy()

    return trajectory


def rollout_mse_per_concept(
    ground_truth: np.ndarray,
    predicted: np.ndarray,
) -> np.ndarray:
    """
    Per-concept MSE between a ground-truth and a predicted rollout.

    Both arrays must have shape ``(T, n_concepts)``; only steps ``1:`` are
    compared, since step 0 is the shared initial condition by construction.

    Returns
    -------
    np.ndarray
        Shape ``(n_concepts,)`` array of per-concept MSE values.
    """
    ground_truth = np.asarray(ground_truth, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if ground_truth.shape != predicted.shape:
        raise ValueError(
            f"Shape mismatch: ground_truth {ground_truth.shape} "
            f"vs predicted {predicted.shape}"
        )

    return np.mean((ground_truth[1:] - predicted[1:]) ** 2, axis=0)
