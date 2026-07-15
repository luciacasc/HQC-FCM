"""
I/O utilities for configuration loading and dataset preparation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def _cfg_get(config: dict, key: str, default=None):
    """Look up ``key`` at the top level, then inside each of the standard
    nested sections (``data``, ``experiment``, ``model``, ``training``)."""
    if key in config:
        return config[key]
    for section in ("data", "experiment", "model", "training"):
        section_cfg = config.get(section)
        if isinstance(section_cfg, dict) and key in section_cfg:
            return section_cfg[key]
    return default


def ensure_dir(path: str | os.PathLike) -> None:
    """Create a directory if it does not already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def load_yaml_config(config_path: str | os.PathLike) -> dict[str, Any]:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    config_path : str or PathLike
        Path to the YAML config file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config if config is not None else {}


def _resolve_csv_path(
    csv_path: str | os.PathLike, data_dir: str | os.PathLike = "."
) -> Path:
    """
    Resolve a CSV path robustly.

    Rules:
    - absolute path -> use as is
    - relative existing path -> use as is
    - otherwise -> resolve relative to data_dir
    """
    csv_path = Path(csv_path)

    if csv_path.is_absolute():
        return csv_path.resolve()

    if csv_path.exists():
        return csv_path.resolve()

    return (Path(data_dir) / csv_path).resolve()


def load_fcm_data(
    *,
    data_dir: str | os.PathLike,
    train_csv: str,
    test_csv: str,
    n_concepts: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Load train/test CSV files and generate the expected column names."""
    train_path = _resolve_csv_path(train_csv, data_dir)
    test_path = _resolve_csv_path(test_csv, data_dir)

    if not train_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_path}")

    cols_t = [f"A{i}_t" for i in range(n_concepts)]
    cols_t1 = [f"A{i}_t1" for i in range(n_concepts)]

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    missing_train = [c for c in cols_t + cols_t1 if c not in train_df.columns]
    missing_test = [c for c in cols_t + cols_t1 if c not in test_df.columns]

    if missing_train:
        raise ValueError(f"Training CSV is missing columns: {missing_train}")
    if missing_test:
        raise ValueError(f"Test CSV is missing columns: {missing_test}")

    return train_df, test_df, cols_t, cols_t1


def sample_train_test(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols_t: list[str],
    cols_t1: list[str],
    n_train: int,
    n_test: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample train/test subsets from full dataframes."""
    if n_train > len(train_df):
        raise ValueError(
            f"Requested n_train={n_train}, but train_df has only {len(train_df)} rows."
        )
    if n_test > len(test_df):
        raise ValueError(
            f"Requested n_test={n_test}, but test_df has only {len(test_df)} rows."
        )

    rng = np.random.default_rng(seed)

    idx_tr = rng.choice(len(train_df), size=n_train, replace=False)
    idx_te = rng.choice(len(test_df), size=n_test, replace=False)

    X_train = train_df.iloc[idx_tr][cols_t].to_numpy()
    Y_train = train_df.iloc[idx_tr][cols_t1].to_numpy()
    X_test = test_df.iloc[idx_te][cols_t].to_numpy()
    Y_test = test_df.iloc[idx_te][cols_t1].to_numpy()

    return X_train, Y_train, X_test, Y_test, idx_tr, idx_te


def load_and_sample_data(config: dict) -> dict:
    """
    Load train/test CSVs and sample subsets.

    Supports both:
    1. flat config keys: ``train_csv``, ``test_csv``, ``n_train``, ``n_test``,
       ``n_concepts``, ``seed``
    2. nested YAML keys: ``data.train_csv``, ``data.test_csv``,
       ``data.n_train``, ``data.n_test``, ``model.n_concepts``,
       ``experiment.seed``
    """
    data_dir = _cfg_get(config, "data_dir", ".")
    train_csv = _cfg_get(config, "train_csv")
    test_csv = _cfg_get(config, "test_csv")
    n_train = _cfg_get(config, "n_train")
    n_test = _cfg_get(config, "n_test")
    n_concepts = int(_cfg_get(config, "n_concepts"))
    seed = int(_cfg_get(config, "seed", 42))

    if train_csv is None or test_csv is None:
        raise ValueError("Config must provide train_csv and test_csv")

    tr, te, cols_t, cols_t1 = load_fcm_data(
        data_dir=data_dir,
        train_csv=train_csv,
        test_csv=test_csv,
        n_concepts=n_concepts,
    )

    n_train = len(tr) if n_train is None else int(n_train)
    n_test = len(te) if n_test is None else int(n_test)

    X_train, Y_train, X_test, Y_test, idx_tr, idx_te = sample_train_test(
        train_df=tr,
        test_df=te,
        cols_t=cols_t,
        cols_t1=cols_t1,
        n_train=n_train,
        n_test=n_test,
        seed=seed,
    )

    train_path = _resolve_csv_path(train_csv, data_dir)
    test_path = _resolve_csv_path(test_csv, data_dir)

    print(f"Subset: train={n_train}, test={n_test}")
    print(f"Columns: {cols_t} -> {cols_t1}")
    print(f"Train CSV: {train_path}")
    print(f"Test CSV: {test_path}")

    return {
        "X_train": X_train,
        "Y_train": Y_train,
        "X_test": X_test,
        "Y_test": Y_test,
        "idx_tr": idx_tr,
        "idx_te": idx_te,
        "train_df": tr,
        "test_df": te,
        "train_path": str(train_path),
        "test_path": str(test_path),
    }
