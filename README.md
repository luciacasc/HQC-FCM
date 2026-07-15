# HQC-FCM — Hybrid Quantum-Classical Fuzzy Cognitive Maps

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Tests](https://github.com/luciacasc/HQC-FCM/actions/workflows/tests.yml/badge.svg)](https://github.com/luciacasc/HQC-FCM/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![DOI](https://zenodo.org/badge/1301660759.svg)](https://doi.org/10.5281/zenodo.21381058)

A benchmark framework for **Hybrid Quantum-Classical Fuzzy Cognitive Maps
(HQC-FCM)** against classical Fuzzy Cognitive Maps and an MLP baseline, on
synthetic FCM-style time-series data.



## What this is

A Fuzzy Cognitive Map (FCM) models a system as a small set of interacting
concepts, whose future state depends on the current state through a causal
weight matrix `W`:

```
A(t+1) = tanh(W · A(t))
```

This project asks: can a small, near-term quantum circuit learn this kind of
dynamics competitively with classical approaches? It benchmarks three model
families on the same supervised task, `A(t) -> A(t+1)`:

> A companion paper describing this method has been submitted for peer
> review; a citation will be added here once it is published.


- **HQC-FCM (QFCM)** — a quantum model built on data re-uploading
  ([Pérez-Salinas et al., 2020](https://quantum-journal.org/papers/q-2020-02-06-226/)):
  each concept is encoded on one qubit, and at every circuit layer the base
  rotation angles are shifted by a learnable, linear function of the full
  concept vector.
- **Classical FCM** — the direct classical counterpart: a learned matrix `W`
  (masked to forbid self-causality) inside a `tanh` nonlinearity.
- **MLP** — a standard multilayer perceptron baseline with no
  FCM-specific structure, for context.

The project provides synthetic dataset generation, YAML-driven experiment
configuration, a shared training loop, postprocessing to recover an
interpretable weight matrix from the quantum model, and result plotting.

## Project layout

```text
HQC-FCM/
├── configs/
│   ├── dataset/            # YAML configs for synthetic dataset generation
│   └── models/             # YAML configs for training experiments
├── examples/
│   └── rollout_demo.py     # autoregressive rollout vs. ground truth
├── src/
│   └── hqcfcm/
│       ├── data_generation/   # synthetic FCM trajectory generation
│       ├── models/            # QFCM, ClassicalFCM, MLPFCM
│       ├── postprocessing/    # W recovery + trajectory rollout (see below)
│       ├── training/          # model-agnostic training loop + artifacts
│       ├── utils/             # config/IO helpers, plotting
│       └── main.py            # experiment orchestration / CLI entrypoint
├── tests/
├── CITATION.cff
├── .zenodo.json
├── pyproject.toml
└── README.md
```

### Why postprocessing is a separate module

The QFCM model does not learn an explicit matrix `W` — it learns quantum
circuit parameters. To interpret it and compare it with the classical
baseline, an effective `W` is *recovered* after training, by sampling the
model's response on a grid of inputs and fitting a linear map by least
squares.

This recovery step lives in `hqcfcm.postprocessing.w_recovery`, not inside
the `QFCM` class itself. Keeping it separate means:

- the model class only defines the circuit and its training loss — nothing
  else;
- the recovery algorithm (grid strategy, regression method, sampling seed,
  …) can change, get its own unit tests, and be versioned independently of
  the model architecture;
- classical FCM gets the same "get effective W" interface for free
  (`extract_cfcm_weight_matrix`), so calling code treats both model types
  uniformly.

`hqcfcm.postprocessing.rollout` similarly generalizes an autoregressive
"roll the model out over many steps and compare to ground truth" analysis
that used to live in a one-off script — see [`examples/rollout_demo.py`](examples/rollout_demo.py).

## Installation

The project uses a standard `src/` layout and can be installed in editable
mode from the repository root:

```bash
pip install -e .
```

For running the test suite, install the `dev` extra instead:

```bash
pip install -e ".[dev]"
```

This installs the dependencies declared in `pyproject.toml`: `numpy`,
`pandas`, `scipy`, `pyyaml`, `torch`, `matplotlib`, `pennylane` (plus
`pytest` for `dev`). Python 3.10 or newer is required.

## Quick start

Generate a synthetic dataset, then train a model on it:

```bash
hqcfcm-generate-dataset --config configs/dataset/motivated.yaml
hqcfcm-train --config configs/models/smoketest.yaml
```

You can also run either command as a Python module:

```bash
python -m hqcfcm.data_generation.generate_dataset --config configs/dataset/motivated.yaml
python -m hqcfcm.main --config configs/models/smoketest.yaml
```

When a training run completes, it will:

1. load the configured train/test CSVs and sample the requested rows,
2. build the selected model (`qfcm`, `cfcm`, or `mlp`),
3. train it with early stopping on test MSE,
4. recover an effective `W` matrix where applicable (`qfcm`, `cfcm`),
5. save artifacts and plots under the configured output directory.

## Configuration

### Dataset generation (`configs/dataset/*.yaml`)

```yaml
dataset:
  name: motivated
  output_dir: data/generated
  n_concepts: 3
  m_scenarios: 50
  t_max: 300
  conv_thr: 1.0e-4
  noise_std: 0.03
  master_seed: 42
  test_fraction: 0.2
  matrix:
    type: custom          # or "random"
    values:
      - [0.0, 0.0, 0.0]
      - [0.6, 0.0, 0.0]
      - [-0.4, 0.5, 0.0]
```

`matrix.type: random` generates a random sparse `W` instead of a fixed one —
see `configs/dataset/sparse.yaml` for an example, including `sparsity`,
`min_rho`/`max_rho` (spectral radius bounds for convergence), and
`seed_offset`.

### Training experiments (`configs/models/*.yaml`)

```yaml
experiment:
  name: smoketest
  save_dir: results/qfcm_motivated
  seed: 42

data:
  train_csv: data/generated/train_motivated.csv
  test_csv: data/generated/test_motivated.csv
  n_train: 20
  n_test: 5

model:
  model_type: qfcm        # qfcm | cfcm | mlp
  n_concepts: 3
  n_layers: 3             # qfcm only

training:
  lr: 0.005
  n_epochs: 3
  batch_size: 4
  patience: 2
  lambda_l1: 1.0e-4
  grid_points: 20         # qfcm postprocessing grid resolution

verbose: true

W_true:                   # optional: ground-truth W, for comparison plots
  - [0.0, 0.0, 0.0]
  - [0.6, 0.0, 0.0]
  - [-0.4, 0.5, 0.0]
```

| Section | Key | Meaning |
|---|---|---|
| `experiment` | `name`, `save_dir`, `seed` | run identity, output location, RNG seed |
| `data` | `train_csv`, `test_csv`, `n_train`, `n_test` | dataset paths and sample sizes |
| `model` | `model_type`, `n_concepts`, + model-specific keys | which model and its architecture |
| `training` | `lr`, `n_epochs`, `batch_size`, `patience`, `lambda_l1` | optimization hyperparameters |
| top-level | `verbose`, `W_true` | logging, optional ground truth for plots |

Model-specific keys:

- `qfcm`: `n_layers`, `diff_method` (default `"best"`), `entanglement`
  (`"none"` \| `"chain_cz"` \| `"ring_cz"`), `device_name` (default
  `"lightning.qubit"`). QFCM has no bias term: the circuit has no additive
  output offset, only `theta`/`alpha`.
- `cfcm`: `use_bias` (default `true`)
- `mlp`: `hidden_dims` (list of hidden layer sizes), `use_bias` (default
  `false`, applied to every linear layer)

### Dataset format

CSV files are expected to contain concept activations at time `t` and
`t+1`. For `n_concepts = 3`:

```text
A0_t, A1_t, A2_t, A0_t1, A1_t1, A2_t1
```

In general: `A{i}_t` for inputs, `A{i}_t1` for next-step targets.

## Outputs

Written under `experiment.save_dir`:

- `{name}_artifacts.json` — run metadata, history, indices, parameter count
- `{name}_X_*.npy`, `{name}_Y_*.npy` — train/test inputs, targets, predictions
- `{name}_history.csv` — per-epoch training history
- `{name}_test_predictions.csv` — tidy per-sample test predictions
- `checkpoints/{name}_best.pt` — best model checkpoint
- model-specific arrays (e.g. `{name}_theta.npy`, `{name}_alpha_masked.npy`
  for `qfcm`; `{name}_W.npy`, `{name}_W_masked.npy` for `cfcm`)
- `{name}_W_lstsq.npy`, `{name}_b_lstsq.npy` — recovered effective matrix
  (`qfcm` only, from `postprocessing.w_recovery`)
- `plots/` — training curves, `W` heatmaps, and (for 2-concept `qfcm` runs)
  the functional-response contour plot

## Reproducibility

Reproducibility is controlled by `experiment.seed`, which seeds dataset
sampling, model initialization, and the training data loader. Given the same
config and input data, runs should reproduce the same sampling and,
libraries permitting, comparable training behavior.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The test suite generates its own dataset fixtures into a temporary
directory, so it does not require pre-generating `data/generated/` by hand.

## Citing this work

If you use this software, please cite it — see [`CITATION.cff`](CITATION.cff)
(GitHub renders a "Cite this repository" button from this file automatically).

Once you publish a GitHub release, Zenodo (via the GitHub integration) will
mint a DOI for it automatically, using the metadata in
[`.zenodo.json`](.zenodo.json). Update the DOI badge at the top of this file
once you have it.

## Bibliography

Pérez-Salinas, Adrián, et al. "Data re-uploading for a universal quantum
classifier." *Quantum* 4 (2020): 226.

## License

Distributed under the GNU General Public License v3.0 or later — see
[`LICENSE`](LICENSE).
