import numpy as np

from randomness_ledger.markov import normalize_rows
from randomness_ledger.metrics import (
    closure_deficit,
    decomposition_check,
    intrinsic_term,
    macro_cond_entropy,
    route_mismatch,
    step_entropy,
)


def test_closure_deficit_near_zero_for_lumpable_chain() -> None:
    # Fibers: A={0,1}, B={2,3}. Aggregated A/B probabilities match within fibers.
    P = np.array(
        [
            [0.35, 0.35, 0.15, 0.15],  # A -> [0.7, 0.3]
            [0.40, 0.30, 0.10, 0.20],  # A -> [0.7, 0.3]
            [0.08, 0.12, 0.40, 0.40],  # B -> [0.2, 0.8]
            [0.05, 0.15, 0.45, 0.35],  # B -> [0.2, 0.8]
        ],
        dtype=float,
    )
    pi_map = np.array([0, 0, 1, 1])
    cd = closure_deficit(P, pi_map, tau=1)
    assert cd < 1e-8


def test_decomposition_residual_small_on_random_chain() -> None:
    rng = np.random.default_rng(20260305)
    n, k, tau = 6, 3, 2
    P = normalize_rows(rng.random((n, n)) + 0.1)
    pi_map = np.array([0, 0, 1, 1, 2, 2])

    hyy = macro_cond_entropy(P, pi_map, tau=tau)
    hyx = intrinsic_term(P, pi_map, tau=tau)
    cd = closure_deficit(P, pi_map, tau=tau)
    res = decomposition_check(P, pi_map, tau=tau)

    assert abs(res) < 1e-6
    assert cd >= -1e-12
    assert np.isclose(hyy, hyx + cd, atol=1e-6)


def test_step_entropy_matches_manual_weighted_row_entropy() -> None:
    macroP = np.array([[0.75, 0.25], [0.10, 0.90]], dtype=float)
    pi_macro = np.array([0.5, 0.5], dtype=float)
    expected = -0.5 * (
        0.75 * np.log(0.75)
        + 0.25 * np.log(0.25)
        + 0.10 * np.log(0.10)
        + 0.90 * np.log(0.90)
    )
    got = step_entropy(macroP, macro_stationary=pi_macro)
    assert np.isclose(got, expected, atol=1e-12)


def test_route_mismatch_tv_equals_half_l1() -> None:
    rng = np.random.default_rng(123)
    P = normalize_rows(rng.random((5, 5)) + 0.1)
    pi_map = np.array([0, 0, 1, 1, 1])

    rm_l1 = route_mismatch(P, pi_map, tau=1, lift="uniform", norm="l1")
    rm_tv = route_mismatch(P, pi_map, tau=1, lift="uniform", norm="tv")
    assert np.isclose(rm_tv, 0.5 * rm_l1, atol=1e-12)
