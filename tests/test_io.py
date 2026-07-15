from pathlib import Path

from hqcfcm.utils.io import load_and_sample_data


def test_load_and_sample_data(motivated_dataset):
    config = {
        "train_csv": motivated_dataset["train_csv"],
        "test_csv": motivated_dataset["test_csv"],
        "n_concepts": 3,
        "n_train": 20,
        "n_test": 5,
        "seed": 42,
    }
    data = load_and_sample_data(config)

    assert data["X_train"].shape == (20, 3)
    assert data["Y_train"].shape == (20, 3)
    assert data["X_test"].shape == (5, 3)
    assert data["Y_test"].shape == (5, 3)

    assert len(data["idx_tr"]) == 20
    assert len(data["idx_te"]) == 5

    assert Path(data["train_path"]).exists()
    assert Path(data["test_path"]).exists()
