"""Counterexamples and independent mathematical contracts from the review."""

import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from randomness_ledger.estimators import nll_order1, nll_order2
from randomness_ledger.generators import gen_exactly_lumpable, gen_hidden_types
from randomness_ledger.hashing import random_oracle_preimage_success
from randomness_ledger.markov import kernel_power, normalize_rows, stationary_dist
from randomness_ledger.metrics import (
    closure_deficit,
    decomposition_check,
    intrinsic_term,
    macro_cond_entropy,
    packaged_history_entropy,
    route_mismatch,
)
from randomness_ledger.packaging import macro_kernel, stationary_conditional_lift


def test_null_microstates_do_not_contribute_infinite_kl() -> None:
    P = np.array([[1., 0., 0.], [0., 1., 0.], [1., 0., 0.]])
    labels, pi = np.array([0, 1, 1]), np.array([0., 1., 0.])
    # A null state in an occupied fiber has disjoint future support.
    assert closure_deficit(P, labels, 1, pi) == 0.0
    assert decomposition_check(P, labels, 1, pi) == 0.0


def test_positive_mixture_support_survives_product_underflow() -> None:
    P = np.array([[1., 0., 0.], [1., 0., 1e-100], [1., 0., 0.]])
    pi, labels = np.array([1., 1e-320, 0.]), np.array([0, 0, 1])
    # The rare atom and its future probability are representable; their product
    # is not. The analytic expected KL is finite and below float resolution.
    assert np.isfinite(closure_deficit(P, labels, 1, pi))


def test_normalizing_tiny_positive_weights_preserves_ratios() -> None:
    assert np.allclose(normalize_rows(np.array([[1e-20, 4e-20]])), [[.2, .8]])
    assert np.allclose(normalize_rows(np.array([[1e308, 1e308]])), [[.5, .5]])


def test_tiny_positive_fiber_retains_its_conditional_law() -> None:
    pi = np.array([1. - 5e-16, 4e-16, 1e-16])
    labels = np.array([0, 1, 1])
    lifted = stationary_conditional_lift(np.array([0., 1.]), labels, pi)
    assert np.allclose(lifted, [0., .8, .2], atol=1e-15, rtol=0)
    assert np.isclose(closure_deficit(np.eye(3), labels, 1, pi), 0., atol=1e-30, rtol=0)
    # This kernel preserves the unequal .8/.2 conditional law in its tiny fiber.
    P = np.array([[1., 0., 0.], [0., .75, .25], [0., 1., 0.]])
    fine = np.arange(3)
    assert np.isclose(macro_cond_entropy(P, fine, 1, pi),
                      4e-16 * (-.75 * np.log(.75) - .25 * np.log(.25)), atol=1e-30, rtol=1e-14)


def test_stationary_fallback_is_invariant_instead_of_uniform() -> None:
    rng = np.random.default_rng(5)
    P = rng.random((4, 4))
    P /= P.sum(axis=1, keepdims=True)
    pi = stationary_dist(P, max_iter=1)
    assert np.linalg.norm(pi @ P - pi, 1) < 1e-12
    assert np.linalg.norm(np.full(4, .25) @ P - .25, 1) > .4


def test_stationary_solver_handles_reducible_and_periodic_kernels() -> None:
    P = np.array([[0., 1., 0.], [.5, 0., .5], [0., 1., 0.]])
    pi = stationary_dist(P, max_iter=1)
    assert np.allclose(pi, [.25, .5, .25], atol=1e-12, rtol=0)
    R = np.array([[1., 0., 0.], [0., 1., 0.], [.3, .7, 0.]])
    pi_r = stationary_dist(R, max_iter=1)
    assert np.linalg.norm(pi_r @ R - pi_r, 1) < 1e-12


@pytest.mark.parametrize("P", [np.zeros((2, 2)), np.array([[1., -.01], [0., 1.]]),
                                  np.full((2, 2), .51), np.empty((0, 0))])
def test_metrics_reject_invalid_dynamics_instead_of_repairing_them(P: np.ndarray) -> None:
    with pytest.raises(ValueError):
        closure_deficit(P, np.array([0, 1]), 1)


@pytest.mark.parametrize("pi", [np.array([.5, .5]), np.array([1.000001, 0.]),
                                   np.array([-1e-16, 1.])])
def test_stationary_metric_rejects_invalid_law(pi: np.ndarray) -> None:
    P = np.array([[.9, .1], [.2, .8]])
    with pytest.raises(ValueError):
        closure_deficit(P, np.array([0, 1]), 1, pi)


def test_pinsker_bound_requires_stationary_lift_even_with_full_support() -> None:
    eps = .001
    P = np.array([[1-eps, eps, 0.], [0., 1-eps, eps], [1., 0., 0.]])
    labels = np.array([0, 1, 1])
    assert np.all(stationary_dist(P) > 0)
    cd = closure_deficit(P, labels, 1)
    rm_s = route_mismatch(P, labels, 1, lift="stationary")
    rm_u = route_mismatch(P, labels, 1, lift="uniform")
    assert cd >= .5 * rm_s**2 - 1e-14
    assert cd < .5 * rm_u**2


def test_deterministic_refinement_and_lag_are_not_monotone() -> None:
    P = np.roll(np.eye(4), 1, axis=1)
    coarse, fine = np.zeros(4, dtype=int), np.array([0, 0, 1, 1])
    assert abs(closure_deficit(P, coarse, 1)) < 1e-14
    assert np.isclose(closure_deficit(P, fine, 1), np.log(2))
    assert np.isclose(closure_deficit(P, fine, 2), 0)
    assert np.isclose(closure_deficit(P, fine, 3), np.log(2))
    assert intrinsic_term(P, fine, 3) == 0


def _entropy(p: np.ndarray) -> float:
    positive = p[p > 0]
    return float(-np.sum(positive * np.log(positive)))


def _path_history_entropy(P: np.ndarray, labels: np.ndarray, pi: np.ndarray, L: int) -> float:
    """Independent full micro-path enumeration, grouping packaged pasts."""
    n, k = len(pi), int(labels.max()) + 1
    table: dict[tuple[int, ...], np.ndarray] = {}
    for path in itertools.product(range(n), repeat=L + 1):
        mass = pi[path[0]]
        for a, b in zip(path[:-1], path[1:]):
            mass *= P[a, b]
        context = tuple(int(labels[x]) for x in path[:-1])
        row = table.setdefault(context, np.zeros(k))
        row[labels[path[-1]]] += mass
    return sum(row.sum() * _entropy(row / row.sum()) for row in table.values() if row.sum() > 0)


def test_history_entropy_matches_independent_micro_path_enumeration() -> None:
    P = np.array([[.6, .3, .1], [.2, .4, .4], [.1, .2, .7]])
    labels, pi = np.array([0, 0, 1]), stationary_dist(P)
    entropies = []
    for L in range(4):
        expected = _path_history_entropy(kernel_power(P, 2), labels, pi, L)
        actual = packaged_history_entropy(P, labels, 2, L, pi)
        assert np.isclose(actual, expected, atol=1e-12, rtol=0)
        entropies.append(actual)
    assert np.all(np.diff(entropies) <= 1e-12)
    assert entropies[-1] >= intrinsic_term(P, labels, 2, pi) - 1e-12
    assert np.isclose(entropies[1], macro_cond_entropy(P, labels, 2, pi))
    assert 0 <= entropies[1] - entropies[-1] <= closure_deficit(P, labels, 2, pi) + 1e-12


def test_lumpable_generator_exports_actual_macro_kernel_and_closes_all_histories() -> None:
    P, labels, meta = gen_exactly_lumpable(3, [1, 2, 3], 7, .4)
    assert np.allclose(macro_kernel(P, labels, 1), meta["K"], atol=1e-12, rtol=0)
    h1 = packaged_history_entropy(P, labels, 1, 1)
    assert np.isclose(packaged_history_entropy(P, labels, 1, 3), h1, atol=1e-12, rtol=0)


def test_hidden_type_memory_gain_is_positive_at_population_level() -> None:
    P, labels, _ = gen_hidden_types(3, [3, 2, 3], .5, 20260305, .9)
    hs = [packaged_history_entropy(P, labels, 1, L) for L in range(1, 6)]
    assert .005 < hs[0] - hs[1] < .0052
    assert hs[1] - hs[2] > .002
    assert np.all(np.diff(hs) < 0)


@pytest.mark.parametrize("seed", range(5))
def test_information_identities_and_bound_on_positive_chains(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n, k = 5 + seed, 2 + seed % 3
    P = normalize_rows(rng.random((n, n)) + .01)
    labels = rng.permutation(np.arange(n) % k)
    pi = stationary_dist(P)
    for tau in (0, 1, 3):
        cd = closure_deficit(P, labels, tau, pi)
        hyy = macro_cond_entropy(P, labels, tau, pi)
        hyx = intrinsic_term(P, labels, tau, pi)
        rm = route_mismatch(P, labels, tau, "stationary", pi)
        assert abs(hyy - hyx - cd) < 2e-14
        assert cd >= .5 * rm**2 - 2e-14
        assert -2e-14 <= hyx <= hyy + 2e-14 <= np.log(k) + 4e-14


def test_identical_predictors_have_zero_gap_on_common_targets() -> None:
    P1 = np.array([[.99, .01], [.25, .75]])
    P2 = np.broadcast_to(P1, (2, 2, 2)).copy()
    y = np.array([0, 1, 1, 0, 0, 1])
    assert nll_order1(y, P1, start=2) == nll_order2(y, P2)
    assert abs(nll_order1(y, P1) - nll_order2(y, P2)) > .1


def test_log_loss_is_infinite_for_an_impossible_observation() -> None:
    assert np.isposinf(nll_order1(np.array([0, 1]), np.eye(2)))
    with pytest.raises(ValueError):
        nll_order1(np.array([0, 1]), np.full((2, 2), 1.5))


def test_random_oracle_baseline_matches_exhaustive_tiny_functions() -> None:
    # Enumerate every map from a two-input domain to two output values, every
    # target input, and every distinct query set of size q.
    for q in range(3):
        successes = total = 0
        for f in itertools.product(range(2), repeat=2):
            for target in range(2):
                for queries in itertools.combinations(range(2), q):
                    successes += any(f[x] == f[target] for x in queries)
                    total += 1
        assert random_oracle_preimage_success(q, 1, 1) == successes / total
    assert random_oracle_preimage_success(1, 256, 62) > 0
    assert random_oracle_preimage_success(2**20, 20, 20) == 1


def test_budget_selection_uses_validation_and_scores_common_test_targets(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib", reason="budget plotting uses the optional viz dependencies")
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run([
        sys.executable, "experiments/budget_curves/run_budget_curves.py",
        "--T", "500", "--burn_in", "20", "--max_order", "3", "--outdir", str(tmp_path),
    ], cwd=root, capture_output=True, text=True, check=True)
    run = Path(completed.stdout.strip())
    with (run / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert {int(r["n_test_targets"]) for r in rows} == {122}
    assert {int(r["evaluation_start"]) for r in rows} == {3}
    for L, row in enumerate(rows):
        selected = int(np.argmin([float(r["validation_nll"]) for r in rows[:L+1]]))
        assert int(row["selected_order"]) == selected
        assert float(row["nll_selected"]) == float(rows[selected]["nll_exact"])
        assert float(row["nll"]) == min(float(r["nll_exact"]) for r in rows[:L+1])
    config = json.loads((run / "config.json").read_text())
    assert "oracle" in config["nll_semantics"]
