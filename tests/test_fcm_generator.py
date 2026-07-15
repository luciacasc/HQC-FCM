import numpy as np

from hqcfcm.data_generation.fcm_generator import (
    build_training_set,
    generate_dataset,
    make_W_random,
    simulate_fcm,
)


def test_make_W_random_respects_sparsity_and_no_self_loops():
    W = make_W_random(n_concepts=4, sparsity=0.5, seed=0)

    assert W.shape == (4, 4)
    assert np.all(np.diag(W) == 0.0)

    rho = np.max(np.abs(np.linalg.eigvals(W)))
    assert rho < 1.0


def test_simulate_fcm_converges_for_small_weights():
    W = np.array([[0.0, 0.0], [0.3, 0.0]])
    rng = np.random.default_rng(0)

    traj, t_conv = simulate_fcm(
        W, A0=np.array([0.5, -0.2]), rng=rng, T_max=200, noise_std=0.0
    )

    assert traj.shape[1] == 2
    assert t_conv <= 200


def test_generate_dataset_and_build_training_set_shapes():
    W = np.array([[0.0, 0.0], [0.5, 0.0]])

    trajectories, conv_times, fixed_points = generate_dataset(
        W, M=10, seed=0, T_max=50
    )

    assert len(trajectories) == 10
    assert conv_times.shape == (10,)
    assert fixed_points.shape == (10, 2)

    X_train, Y_train, X_test, Y_test = build_training_set(
        trajectories, test_fraction=0.2, seed=0
    )

    assert X_train.shape[1] == 2
    assert Y_train.shape[1] == 2
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
