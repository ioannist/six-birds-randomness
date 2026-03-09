import numpy as np

from randomness_ledger.markov import normalize_rows
from randomness_ledger.obs_models import (
    gen_hidden_markov_observations,
    make_gaussian_emission_model,
    make_mixed_emission_model,
)


def test_gaussian_obs_generation_shapes_and_determinism() -> None:
    rng = np.random.default_rng(11)
    P = normalize_rows(rng.random((5, 5)) + 0.1)
    group_map = np.array([0, 0, 1, 1, 2], dtype=np.int64)

    model = make_gaussian_emission_model(
        n_states=P.shape[0], d=4, seed=77, group_map=group_map
    )
    x1, o1 = gen_hidden_markov_observations(P, model, T=200, seed=123)
    x2, o2 = gen_hidden_markov_observations(P, model, T=200, seed=123)

    assert x1.shape == (200,)
    assert o1.shape == (200, 4)
    assert np.issubdtype(x1.dtype, np.integer)
    assert o1.dtype == np.float32
    assert np.isfinite(o1).all()
    assert np.array_equal(x1, x2)
    assert np.array_equal(o1, o2)


def test_mixed_obs_generation_shapes_and_determinism() -> None:
    rng = np.random.default_rng(12)
    P = normalize_rows(rng.random((6, 6)) + 0.1)
    group_map = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)

    model = make_mixed_emission_model(n_states=P.shape[0], d=4, seed=88, group_map=group_map)
    x1, o1 = gen_hidden_markov_observations(P, model, T=200, seed=123)
    x2, o2 = gen_hidden_markov_observations(P, model, T=200, seed=123)

    assert x1.shape == (200,)
    assert o1.shape == (200, 4)
    assert np.issubdtype(x1.dtype, np.integer)
    assert o1.dtype == np.float32
    assert np.isfinite(o1).all()
    assert np.array_equal(x1, x2)
    assert np.array_equal(o1[:5], o2[:5])
