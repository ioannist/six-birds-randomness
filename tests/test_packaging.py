import numpy as np

from randomness_ledger.markov import normalize_rows, stationary_dist
from randomness_ledger.packaging import (
    macro_kernel,
    pushforward_dist,
    stationary_conditional_lift,
    uniform_lift,
)


def test_pushforward_dist_known_example() -> None:
    mu_micro = np.array([0.5, 0.0, 0.25, 0.25, 0.0])
    pi_map = np.array([0, 1, 1, 0, 1])
    got = pushforward_dist(mu_micro, pi_map, k=2)
    expected = np.array([0.75, 0.25])
    assert np.array_equal(got, expected)


def test_uniform_lift_is_valid_and_pushes_back() -> None:
    pi_map = np.array([0, 0, 1, 1, 1])
    mu_macro = np.array([0.2, 0.8])

    mu_micro = uniform_lift(mu_macro, pi_map)
    assert np.all(mu_micro >= 0.0)
    assert np.isclose(mu_micro.sum(), 1.0, atol=1e-12)

    pushed = pushforward_dist(mu_micro, pi_map, k=2)
    assert np.allclose(pushed, mu_macro, atol=1e-12)


def test_stationary_conditional_lift_recovers_stationary_distribution() -> None:
    pi_map = np.array([0, 0, 1, 1])
    pi_stationary = np.array([0.1, 0.2, 0.3, 0.4])
    pi_macro = pushforward_dist(pi_stationary, pi_map, k=2)

    lifted = stationary_conditional_lift(pi_macro, pi_map, pi_stationary)
    assert np.allclose(lifted, pi_stationary, atol=1e-12)


def test_macro_kernel_row_stochastic() -> None:
    rng = np.random.default_rng(42)
    P = normalize_rows(rng.random((4, 4)))
    pi_map = np.array([0, 0, 1, 1])

    macro_uniform = macro_kernel(P, pi_map, tau=3, lift="uniform")
    assert np.allclose(macro_uniform.sum(axis=1), 1.0, atol=1e-10)
    assert np.all(macro_uniform >= -1e-15)

    pi_stat = stationary_dist(P)
    macro_stationary = macro_kernel(
        P, pi_map, tau=3, lift="stationary", pi_stationary=pi_stat
    )
    assert np.allclose(macro_stationary.sum(axis=1), 1.0, atol=1e-10)
    assert np.all(macro_stationary >= -1e-15)


def test_macro_kernel_identity_partition_matches_micro_kernel() -> None:
    rng = np.random.default_rng(9)
    P = normalize_rows(rng.random((5, 5)))
    pi_map = np.arange(P.shape[0])

    macro_uniform = macro_kernel(P, pi_map, tau=1, lift="uniform")
    assert macro_uniform.shape == P.shape
    assert np.allclose(macro_uniform, P, atol=1e-12)

    macro_stationary = macro_kernel(P, pi_map, tau=1, lift="stationary")
    assert macro_stationary.shape == P.shape
    assert np.allclose(macro_stationary, P, atol=1e-12)

    macro_tau0 = macro_kernel(P, pi_map, tau=0, lift="uniform")
    assert np.allclose(macro_tau0, np.eye(P.shape[0]), atol=1e-12)
