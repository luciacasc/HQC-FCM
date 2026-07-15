import numpy as np
import torch

from hqcfcm.models.cfcm import ClassicalFCM
from hqcfcm.models.qfcm import QFCM
from hqcfcm.postprocessing.w_recovery import (
    extract_cfcm_weight_matrix,
    recover_qfcm_weight_matrix,
)


def test_extract_cfcm_weight_matrix_matches_masked_W():
    model = ClassicalFCM(n_concepts=3, seed=0)

    W = extract_cfcm_weight_matrix(model)

    assert W.shape == (3, 3)
    np.testing.assert_allclose(W, model.W_masked.detach().cpu().numpy())
    # Self-causality must be masked out.
    assert np.allclose(np.diag(W), 0.0)


def test_recover_qfcm_weight_matrix_shapes_and_finiteness():
    torch.set_default_dtype(torch.float64)
    model = QFCM(n_concepts=3, n_layers=2, seed=0, entanglement="none")

    result = recover_qfcm_weight_matrix(model, grid_points=5, seed=0)

    assert result["W_lstsq"].shape == (3, 3)
    assert result["b_lstsq"].shape == (3,)
    assert result["C_grid"].shape[1] == 3
    assert result["Y_grid_pred"].shape == (result["C_grid"].shape[0], 3)
    assert np.all(np.isfinite(result["W_lstsq"]))
    assert np.all(np.isfinite(result["b_lstsq"]))


def test_recover_qfcm_weight_matrix_2concepts_uses_regular_grid():
    torch.set_default_dtype(torch.float64)
    model = QFCM(n_concepts=2, n_layers=1, seed=0, entanglement="none")

    result = recover_qfcm_weight_matrix(model, grid_points=4, seed=0)

    # For n_concepts == 2, a regular grid_points x grid_points mesh is used.
    assert result["C_grid"].shape == (16, 2)
