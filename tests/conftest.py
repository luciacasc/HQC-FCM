"""Shared pytest fixtures for the HQC-FCM test suite."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hqcfcm.data_generation.generate_dataset import generate_from_config

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _chdir_repo_root(monkeypatch):
    """
    Run every test from the repository root.

    This mirrors how the CLI is actually invoked (e.g.
    ``hqcfcm-train --config configs/models/smoketest.yaml`` from the repo
    root), so config-relative paths behave the same way in tests as they do
    for a real user.
    """
    monkeypatch.chdir(REPO_ROOT)


@pytest.fixture(scope="session")
def motivated_dataset(tmp_path_factory):
    """
    Generate the "motivated" synthetic dataset once per test session, into a
    temporary directory.

    This makes the test suite self-contained: it does not depend on
    ``data/generated/`` having been populated beforehand by manually running
    ``hqcfcm-generate-dataset`` (that directory is git-ignored and empty on a
    fresh clone).
    """
    output_dir = tmp_path_factory.mktemp("data") / "generated"

    with (REPO_ROOT / "configs" / "dataset" / "motivated.yaml").open() as f:
        cfg = yaml.safe_load(f)
    cfg["dataset"]["output_dir"] = str(output_dir)

    tmp_cfg_path = tmp_path_factory.mktemp("cfg") / "motivated.yaml"
    with tmp_cfg_path.open("w") as f:
        yaml.safe_dump(cfg, f)

    generate_from_config(tmp_cfg_path)

    name = cfg["dataset"]["name"]
    return {
        "train_csv": str(output_dir / f"train_{name}.csv"),
        "test_csv": str(output_dir / f"test_{name}.csv"),
    }
