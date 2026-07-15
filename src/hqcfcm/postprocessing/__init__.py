"""
Postprocessing and analysis utilities, kept separate from model definitions.

- :mod:`hqcfcm.postprocessing.w_recovery`: recovers an effective FCM matrix
  from a trained model (numerical recovery for QFCM, direct extraction for
  classical FCM).
- :mod:`hqcfcm.postprocessing.rollout`: autoregressive trajectory rollout for
  qualitative model-vs-ground-truth comparisons.
"""

from hqcfcm.postprocessing.rollout import rollout_mse_per_concept, rollout_trajectory
from hqcfcm.postprocessing.w_recovery import (
    extract_cfcm_weight_matrix,
    recover_qfcm_weight_matrix,
)

__all__ = [
    "recover_qfcm_weight_matrix",
    "extract_cfcm_weight_matrix",
    "rollout_trajectory",
    "rollout_mse_per_concept",
]
