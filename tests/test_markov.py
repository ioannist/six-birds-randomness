import numpy as np

from randomness_ledger.markov import (
    is_stochastic_matrix,
    kernel_power,
    make_ergodic,
    normalize_rows,
    simulate_chain,
    stationary_dist,
)


def test_stochasticity_from_normalization() -> None:
    rng = np.random.default_rng(1234)
    raw = rng.random((5, 5))
    P = normalize_rows(raw)
    assert is_stochastic_matrix(P)


def test_normalize_rows_zero_row_becomes_uniform() -> None:
    raw = np.array(
        [
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 1.0],
        ]
    )
    P = normalize_rows(raw)
    assert is_stochastic_matrix(P)
    assert np.allclose(P[1], np.full(3, 1.0 / 3.0))


def test_make_ergodic_is_stochastic_and_positive() -> None:
    sparse = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    P = make_ergodic(sparse, eps=1e-3)
    assert is_stochastic_matrix(P)
    assert np.all(P > 0.0)


def test_stationary_distribution_known_two_state_chain() -> None:
    P = np.array([[0.9, 0.1], [0.2, 0.8]])
    pi = stationary_dist(P)
    assert abs(float(pi.sum()) - 1.0) < 1e-12
    assert np.all(pi >= -1e-15)
    assert np.allclose(pi @ P, pi, atol=1e-10)
    assert np.allclose(pi, np.array([2.0 / 3.0, 1.0 / 3.0]), atol=1e-6)


def test_kernel_power_cases() -> None:
    P = np.array([[0.8, 0.2], [0.1, 0.9]])
    assert np.array_equal(kernel_power(P, 1), P)
    assert np.array_equal(kernel_power(P, 0), np.eye(2))
    assert np.allclose(kernel_power(P, 2), P @ P)


def test_simulation_identity_kernel_stays_constant() -> None:
    P = np.eye(4)
    rng = np.random.default_rng(7)
    states = simulate_chain(P, T=25, x0=2, rng=rng)
    assert len(states) == 25
    assert np.all(states == 2)
    assert np.all((states >= 0) & (states < 4))
