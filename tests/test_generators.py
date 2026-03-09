import numpy as np

from randomness_ledger.generators import (
    gen_exactly_lumpable,
    gen_hidden_types,
    gen_metastable,
    gen_perturbed_lumpable,
)
from randomness_ledger.metrics import closure_deficit
from randomness_ledger.packaging import macro_kernel


def _assert_valid_output(P: np.ndarray, pi_map: np.ndarray) -> None:
    n = P.shape[0]
    assert P.ndim == 2
    assert P.shape == (n, n)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-10)
    assert float(P.min()) > 0.0

    assert pi_map.ndim == 1
    assert pi_map.shape[0] == n
    assert np.issubdtype(pi_map.dtype, np.integer)
    k = int(pi_map.max()) + 1
    counts = np.bincount(pi_map, minlength=k)
    assert np.all(counts > 0)

    macroP = macro_kernel(P, pi_map, tau=1, lift="uniform")
    assert macroP.shape == (k, k)


def test_generators_basic_validity() -> None:
    P1, pi1, meta1 = gen_exactly_lumpable(
        n_macro=3, fiber_sizes=[2, 2, 2], seed=11, aperiodic_eps=1e-3
    )
    P2, pi2, meta2 = gen_perturbed_lumpable(
        n_macro=3,
        fiber_sizes=[2, 3, 1],
        seed=12,
        aperiodic_eps=1e-3,
        heterogeneity_alpha=0.35,
    )
    P3, pi3, meta3 = gen_metastable(block_sizes=[2, 2, 2], p_in=1.0, p_out=0.1, seed=13)
    P4, pi4, meta4 = gen_hidden_types(
        n_macro=3, fiber_sizes=[2, 2, 2], type_split=0.5, seed=14, strength=0.8
    )

    _assert_valid_output(P1, pi1)
    _assert_valid_output(P2, pi2)
    _assert_valid_output(P3, pi3)
    _assert_valid_output(P4, pi4)

    assert meta1["kind"] == "exactly_lumpable"
    assert meta2["kind"] == "perturbed_lumpable"
    assert meta3["kind"] == "metastable"
    assert meta4["kind"] == "hidden_types"


def test_exactly_lumpable_has_tiny_closure_deficit() -> None:
    P, pi_map, _ = gen_exactly_lumpable(
        n_macro=3, fiber_sizes=[2, 2, 2], seed=123, aperiodic_eps=1e-3
    )
    cd = closure_deficit(P, pi_map, tau=1)
    assert cd < 1e-8
