"""Observation-model utilities for hidden Markov state sequences."""

from __future__ import annotations

from typing import Callable

import numpy as np

from randomness_ledger.markov import simulate_chain


def _validate_group_map(group_map: np.ndarray, n_states: int) -> np.ndarray:
    """Validate optional micro->group map."""
    arr = np.asarray(group_map)
    if arr.ndim != 1 or arr.shape[0] != n_states:
        raise ValueError("group_map must have shape (n_states,)")
    if not np.all(np.isfinite(arr)):
        raise ValueError("group_map must contain finite values")
    if not np.all(arr == np.floor(arr)):
        raise ValueError("group_map must contain integer labels")
    out = arr.astype(np.int64)
    if np.any(out < 0):
        raise ValueError("group_map labels must be >= 0")
    return out


def make_gaussian_emission_model(
    n_states: int,
    d: int,
    seed: int,
    group_map: np.ndarray | None = None,
    mean_scale: float = 1.0,
    group_jitter: float = 0.2,
    noise_scale: float = 0.3,
) -> dict:
    """Construct Gaussian emission parameters in R^d."""
    if n_states < 1 or d < 1:
        raise ValueError("n_states and d must be >= 1")
    rng = np.random.default_rng(int(seed))

    if group_map is not None:
        gmap = _validate_group_map(group_map, n_states)
        groups, inverse = np.unique(gmap, return_inverse=True)
        macro_means = rng.normal(0.0, float(mean_scale), size=(len(groups), d))
        means = macro_means[inverse] + float(group_jitter) * rng.normal(size=(n_states, d))
    else:
        means = rng.normal(0.0, float(mean_scale), size=(n_states, d))

    means = np.asarray(means, dtype=float)
    if not np.all(np.isfinite(means)):
        raise ValueError("non-finite gaussian means generated")

    return {
        "type": "gaussian",
        "d": int(d),
        "means": means,
        "noise_scale": float(noise_scale),
    }


def make_mixed_emission_model(
    n_states: int,
    d: int,
    seed: int,
    group_map: np.ndarray | None = None,
    hidden_dim: int | None = None,
    mean_scale: float = 1.0,
    group_jitter: float = 0.2,
    noise_scale: float = 0.15,
) -> dict:
    """Construct mixed nonlinear emission parameters in R^d."""
    if n_states < 1 or d < 1:
        raise ValueError("n_states and d must be >= 1")
    hdim = max(int(d), 8) if hidden_dim is None else int(hidden_dim)
    if hdim < 1:
        raise ValueError("hidden_dim must be >= 1")

    rng = np.random.default_rng(int(seed))
    if group_map is not None:
        gmap = _validate_group_map(group_map, n_states)
        groups, inverse = np.unique(gmap, return_inverse=True)
        macro_codes = rng.normal(0.0, float(mean_scale), size=(len(groups), hdim))
        codes = macro_codes[inverse] + float(group_jitter) * rng.normal(size=(n_states, hdim))
    else:
        codes = rng.normal(0.0, float(mean_scale), size=(n_states, hdim))

    scale = 1.0 / np.sqrt(float(hdim))
    W1 = rng.normal(0.0, scale, size=(d, hdim))
    W2 = rng.normal(0.0, scale, size=(d, hdim))
    b = rng.normal(0.0, 0.2, size=(d,))
    means = 0.5 * np.tanh(codes @ W1.T + b) + 0.5 * np.sin(codes @ W2.T)

    if not np.all(np.isfinite(means)):
        raise ValueError("non-finite mixed means generated")

    return {
        "type": "mixed",
        "d": int(d),
        "hidden_dim": int(hdim),
        "codes": np.asarray(codes, dtype=float),
        "W1": np.asarray(W1, dtype=float),
        "W2": np.asarray(W2, dtype=float),
        "b": np.asarray(b, dtype=float),
        "means": np.asarray(means, dtype=float),
        "noise_scale": float(noise_scale),
    }


def _sample_from_model_dict(obs_model: dict, x_t: int, rng: np.random.Generator) -> np.ndarray:
    """Sample one observation from a supported model dict."""
    model_type = obs_model.get("type")
    if model_type not in {"gaussian", "mixed"}:
        raise ValueError(f"unsupported obs_model type: {model_type}")
    if "means" not in obs_model:
        raise ValueError("obs_model must include 'means'")

    means = np.asarray(obs_model["means"], dtype=float)
    if means.ndim != 2:
        raise ValueError("obs_model['means'] must have shape (n_states, d)")
    n_states, d = means.shape
    state = int(x_t)
    if not 0 <= state < n_states:
        raise ValueError("state index out of range for obs model")

    noise_scale = float(obs_model.get("noise_scale", 0.0))
    obs = means[state] + noise_scale * rng.normal(size=d)
    if not np.all(np.isfinite(obs)):
        raise ValueError("non-finite observation sampled")
    return np.asarray(obs, dtype=float)


def gen_hidden_markov_observations(
    P: np.ndarray,
    obs_model: dict | Callable[[int, np.random.Generator], np.ndarray],
    T: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate hidden Markov states and continuous observations."""
    if isinstance(T, bool) or not isinstance(T, (int, np.integer)) or int(T) < 1:
        raise ValueError("T must be an integer >= 1")

    kernel = np.asarray(P, dtype=float)
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("P must be square with shape (n, n)")
    if not np.all(np.isfinite(kernel)):
        raise ValueError("P must contain finite values")
    n_states = int(kernel.shape[0])

    rng = np.random.default_rng(int(seed))
    x0 = int(rng.integers(n_states))
    x = simulate_chain(kernel, int(T), x0=x0, rng=rng).astype(np.int64, copy=False)

    if callable(obs_model):
        first = np.asarray(obs_model(int(x[0]), rng), dtype=float)
        if first.ndim != 1:
            raise ValueError("callable obs_model must return shape (d,)")
        d = int(first.shape[0])
        if d < 1:
            raise ValueError("callable obs_model returned empty vector")
        o = np.empty((int(T), d), dtype=np.float32)
        o[0] = first.astype(np.float32)
        for t in range(1, int(T)):
            sample = np.asarray(obs_model(int(x[t]), rng), dtype=float)
            if sample.shape != (d,):
                raise ValueError("callable obs_model returned inconsistent shape")
            o[t] = sample.astype(np.float32)
    else:
        o = np.empty((int(T), int(np.asarray(obs_model["means"]).shape[1])), dtype=np.float32)
        for t in range(int(T)):
            o[t] = _sample_from_model_dict(obs_model, int(x[t]), rng).astype(np.float32)

    if not np.isfinite(o).all():
        raise ValueError("non-finite values found in observations")
    return x, o
