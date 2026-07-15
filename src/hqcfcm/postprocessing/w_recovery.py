"""
Recovery of an effective, classical-FCM-style weight matrix from a trained model.

Rationale
---------
The QFCM model does not learn an explicit causal matrix ``W``: it learns
quantum-circuit parameters (``theta``, ``alpha``) instead. To compare it with
the classical FCM baseline, and to interpret what it learned, we recover an
*effective* linear matrix ``W_lstsq`` by:

1. sampling the model's response on a grid (or random cloud, for
   ``n_concepts > 2``) of input concept vectors,
2. linearizing the response with ``arctanh`` (the inverse of the ``tanh``
   nonlinearity used by the classical FCM and MLP baselines),
3. fitting a linear map ``A(t) -> arctanh(A(t+1))`` by least squares.

This recovery procedure is intentionally kept **outside** the ``QFCM`` model
class: it is a diagnostic/analysis step, not part of the model definition,
and keeping it separate means the recovery algorithm (e.g. the grid
strategy, or the regression method) can be changed, tested, and versioned
independently of the model architecture.

For the classical FCM baseline, no recovery is needed: the learned matrix is
already ``model.W_masked``, and :func:`extract_cfcm_weight_matrix` simply
exposes it through the same interface used for QFCM, so that calling code
(e.g. ``main.py``) can treat both model types uniformly.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import torch
from numpy.linalg import lstsq as np_lstsq

from hqcfcm.models.qfcm import QFCM, code_coords


class WRecoveryResult(TypedDict):
    """Result of :func:`recover_qfcm_weight_matrix`."""

    W_lstsq: np.ndarray
    b_lstsq: np.ndarray
    C_grid: np.ndarray
    Y_grid_pred: np.ndarray


def _build_sampling_grid(
    n_concepts: int,
    grid_points: int,
    seed: int,
) -> np.ndarray:
    """
    Build the grid (or random cloud) of concept vectors used to probe the model.

    For ``n_concepts == 2`` a regular ``grid_points x grid_points`` mesh over
    ``[-1, 1]^2`` is used, which also enables the 2D functional-response plot.
    For higher dimensions, a uniform random cloud of ``grid_points ** 2``
    points is used instead, to keep the number of model evaluations bounded.
    """
    g = np.linspace(-1.0, 1.0, grid_points)

    if n_concepts == 2:
        g0, g1 = np.meshgrid(g, g)
        return np.stack([g0.ravel(), g1.ravel()], axis=1)

    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(grid_points**2, n_concepts))


def recover_qfcm_weight_matrix(
    model: QFCM,
    *,
    grid_points: int = 20,
    seed: int = 42,
) -> WRecoveryResult:
    """
    Recover an effective FCM matrix from a trained QFCM model.

    Parameters
    ----------
    model : QFCM
        A trained (or untrained) QFCM model. The model is switched to
        ``eval()`` mode and is not modified otherwise.
    grid_points : int
        Grid resolution for ``n_concepts == 2``, or ``sqrt`` of the number of
        random probe points otherwise.
    seed : int
        Seed for the random probe cloud used when ``n_concepts != 2``.

    Returns
    -------
    WRecoveryResult
        A dict with the recovered matrix ``W_lstsq``, intercept ``b_lstsq``,
        and the probe grid/response (``C_grid``, ``Y_grid_pred``) used to fit
        them, useful for diagnostic plots.

    Notes
    -----
    This function performs no file I/O. Persisting the result is the
    responsibility of the caller (see :mod:`hqcfcm.training.artifacts`).
    """
    model.eval()

    theta = model.theta.detach().clone()
    alpha_masked = model.alpha_masked.detach().clone()
    n_concepts = model.n_concepts

    C_grid = _build_sampling_grid(n_concepts, grid_points, seed)
    Y_grid_pred = np.zeros((len(C_grid), n_concepts))

    with torch.no_grad():
        for idx, x in enumerate(C_grid):
            x_t = torch.tensor(x, dtype=torch.get_default_dtype())
            theta_aux = code_coords(theta, alpha_masked, x_t)
            out = model.circuit(theta_aux)
            Y_grid_pred[idx] = torch.stack(out).cpu().numpy()

    W_lstsq = np.zeros((n_concepts, n_concepts))
    b_lstsq = np.zeros(n_concepts)

    design_matrix = np.column_stack([C_grid, np.ones(len(C_grid))])

    for k in range(n_concepts):
        y_pred_k = Y_grid_pred[:, k]
        y_linearized = np.arctanh(np.clip(y_pred_k, -0.9999, 0.9999))
        solution, _, _, _ = np_lstsq(design_matrix, y_linearized, rcond=None)
        W_lstsq[k] = solution[:n_concepts]
        b_lstsq[k] = solution[n_concepts]

    return {
        "W_lstsq": W_lstsq,
        "b_lstsq": b_lstsq,
        "C_grid": C_grid,
        "Y_grid_pred": Y_grid_pred,
    }


def extract_cfcm_weight_matrix(model) -> np.ndarray:
    """
    Extract the effective weight matrix from a trained classical FCM model.

    Unlike QFCM, the classical FCM learns ``W`` directly, so no numerical
    recovery is needed. This function exists to give callers a single,
    uniform "get the effective W" entry point across model types.
    """
    return model.W_masked.detach().cpu().numpy()
