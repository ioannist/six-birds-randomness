import numpy as np

from randomness_ledger.estimators import (
    fit_markov_order1,
    fit_markov_order2,
    nll_order1,
    nll_order2,
)
from randomness_ledger.markov import normalize_rows, simulate_chain


def simulate_order2(P2true: np.ndarray, T: int, seed: int) -> np.ndarray:
    """Simulate a second-order discrete process with uniform initial pair."""
    if T < 3:
        raise ValueError("T must be >= 3")
    rng = np.random.default_rng(seed)
    k = P2true.shape[0]
    y = np.empty(T, dtype=np.int64)
    y[0] = int(rng.integers(0, k))
    y[1] = int(rng.integers(0, k))
    for t in range(1, T - 1):
        y[t + 1] = int(rng.choice(k, p=P2true[y[t - 1], y[t]]))
    return y


def test_prediction_gap_near_zero_for_order1_source() -> None:
    rng = np.random.default_rng(20260305)
    k = 3
    T = 4000
    Ptrue = normalize_rows(rng.random((k, k)) + 0.1)
    y = simulate_chain(Ptrue, T=T, x0=0, rng=rng)

    train = y[:2000]
    test = y[2000:]

    P1 = fit_markov_order1(train, k)
    P2 = fit_markov_order2(train, k)
    gap = nll_order1(test, P1) - nll_order2(test, P2)

    assert np.allclose(P1.sum(axis=1), 1.0, atol=1e-12)
    assert np.allclose(P2.sum(axis=2), 1.0, atol=1e-12)
    assert abs(gap) < 0.02


def test_prediction_gap_positive_for_hidden_phase_order2_source() -> None:
    k = 2
    T = 4000
    P2true = np.zeros((k, k, k), dtype=float)
    P2true[0, 0] = np.array([0.99, 0.01])
    P2true[0, 1] = np.array([0.01, 0.99])
    P2true[1, 0] = np.array([0.01, 0.99])
    P2true[1, 1] = np.array([0.99, 0.01])

    y = simulate_order2(P2true, T=T, seed=20260306)
    train = y[:2000]
    test = y[2000:]

    P1 = fit_markov_order1(train, k=k)
    P2 = fit_markov_order2(train, k=k)
    nll1 = nll_order1(test, P1)
    nll2 = nll_order2(test, P2)
    gap = nll1 - nll2

    assert np.isfinite(nll1)
    assert np.isfinite(nll2)
    assert gap > 0.1
