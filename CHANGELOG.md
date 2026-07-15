# Changelog

All notable changes to this project are documented in this file.

## [0.1.0] — 2026-07-14

First public release under the new name **HQC-FCM** (previously developed as
`qfcm-benchmark` / `QFCM`).

### Changed
- Renamed the project and Python package: `qfcm_benchmark` → `hqcfcm`.
- **W-matrix postprocessing separated from the model.** The QFCM model no
  longer has a `postprocess_W` method; recovering an effective FCM matrix
  from a trained model is now handled by
  `hqcfcm.postprocessing.w_recovery`, independent of the model's
  architecture. `main.py` and `training/artifacts.py` were updated
  accordingly.
- Renamed `utils/plot_model_results.py` → `utils/plotting.py`.

### Fixed
- Removed a duplicate-save bug: the old `QFCM.postprocess_W` wrote
  `W_lstsq`/`b_lstsq`/`C_grid`/`Y_grid_pred` to disk itself, and
  `training/artifacts.py` wrote the same arrays again. Saving now happens
  exactly once, in `training/artifacts.py`.
- Migrated `configs/models/cfcm.yaml` and `configs/models/mlp.yaml` from a
  stale flat config schema to the nested `experiment/data/model/training`
  schema expected by `main.py` — these configs did not run at all before.
  Also fixed their referenced CSV filenames to match the naming convention
  actually produced by the dataset generator (`train_{name}.csv`).
- Fixed a stale path in `generate_dataset.py`'s `--config` help text
  (`configs/datasets/...` → `configs/dataset/...`).
- Test suite is now self-contained: it generates its own dataset fixtures
  in a temporary directory instead of depending on a manually pre-populated
  (and git-ignored) `data/generated/` folder.

### Added
- `hqcfcm.postprocessing.rollout`: a generalized autoregressive trajectory
  rollout utility and comparison plot, extracted from an ad-hoc, hard-coded
  analysis script (`case_of_study_dyn.py`) in the original repository.
- `examples/rollout_demo.py`: a portable, parametrized example showing how
  to reproduce that kind of qualitative model-vs-ground-truth analysis.
- Unit tests for `data_generation.fcm_generator` and for the new
  `postprocessing` module (previously untested).
- `CITATION.cff` and `.zenodo.json` for citation and DOI metadata.
- GitHub Actions workflow running the test suite on push/PR.

### Removed
- `case_of_study_dyn.py`, `case_of_study_plot_paper.py`, and the associated
  PDF figures: ad-hoc, hard-coded scripts (absolute local paths, commented-out
  blocks) tied to one specific analysis, not reusable as library code. Their
  one genuinely useful idea (autoregressive rollout comparison) was
  generalized and kept — see "Added" above.
- `configs/dataset/case_of_study_5C.yaml` and `configs/models/case_of_study.yaml`,
  which existed only to support the removed scripts.
