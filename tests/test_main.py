from hqcfcm.main import flatten_experiment_config, run_experiment
from hqcfcm.utils.io import load_yaml_config


def test_run_experiment_smoketest(motivated_dataset, tmp_path):
    raw_config = load_yaml_config("configs/models/smoketest.yaml")
    config = flatten_experiment_config(raw_config)

    # Point the run at the session-generated dataset and a scratch output
    # directory, instead of relying on a pre-populated data/generated/.
    config["train_csv"] = motivated_dataset["train_csv"]
    config["test_csv"] = motivated_dataset["test_csv"]
    config["save_dir"] = str(tmp_path / "results")

    result = run_experiment(config)

    assert "model" in result
    assert "train_result" in result
    assert "data" in result
    assert result["data"]["X_train"].shape == (20, 3)
    assert result["data"]["X_test"].shape == (5, 3)
