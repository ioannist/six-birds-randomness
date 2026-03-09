"""Sequence-based estimators for Markov predictive models."""

from __future__ import annotations

import numpy as np


def _validate_k(k: int) -> int:
    """Validate state-space size."""
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer")
    k_int = int(k)
    if k_int < 1:
        raise ValueError("k must be >= 1")
    return k_int


def _validate_smoothing(smoothing: float) -> float:
    """Validate additive smoothing value."""
    value = float(smoothing)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("smoothing must be finite and >= 0")
    return value


def _validate_sequence(y_seq: np.ndarray, k: int, min_len: int) -> np.ndarray:
    """Validate 1D discrete sequence with labels in [0, k-1]."""
    seq = np.asarray(y_seq)
    if seq.ndim != 1:
        raise ValueError("y_seq must be a 1D array")
    if seq.shape[0] < min_len:
        raise ValueError(f"y_seq must have length >= {min_len}")
    if not np.all(np.isfinite(seq)):
        raise ValueError("y_seq must contain only finite values")
    if not np.all(seq == np.floor(seq)):
        raise ValueError("y_seq must contain integer labels")

    seq_int = seq.astype(np.int64)
    if np.any(seq_int < 0) or np.any(seq_int >= k):
        raise ValueError("y_seq labels must be in [0, k-1]")
    return seq_int


def fit_markov_order1(y_seq: np.ndarray, k: int, smoothing: float = 1e-6) -> np.ndarray:
    """Fit first-order Markov predictor P(y_{t+1} | y_t) from a sequence.

    Additive smoothing is intentional: it keeps all transition probabilities
    positive so held-out NLL stays finite for unseen transitions.
    """
    k_int = _validate_k(k)
    smooth = _validate_smoothing(smoothing)
    seq = _validate_sequence(y_seq, k_int, min_len=2)

    counts = np.full((k_int, k_int), smooth, dtype=float)
    np.add.at(counts, (seq[:-1], seq[1:]), 1.0)

    row_sums = counts.sum(axis=1, keepdims=True)
    P1 = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 1e-15)
    zero_rows = row_sums[:, 0] <= 1e-15
    if np.any(zero_rows):
        P1[zero_rows] = 1.0 / k_int
    return P1


def fit_markov_order2(y_seq: np.ndarray, k: int, smoothing: float = 1e-6) -> np.ndarray:
    """Fit second-order predictor P(y_{t+1} | y_{t-1}, y_t) from a sequence.

    Additive smoothing prevents zero-probability unseen contexts/targets,
    avoiding `-inf` NLL when evaluating sparse higher-order data.
    """
    k_int = _validate_k(k)
    smooth = _validate_smoothing(smoothing)
    seq = _validate_sequence(y_seq, k_int, min_len=3)

    counts = np.full((k_int, k_int, k_int), smooth, dtype=float)
    a = seq[:-2]
    b = seq[1:-1]
    c = seq[2:]
    np.add.at(counts, (a, b, c), 1.0)

    sums = counts.sum(axis=2, keepdims=True)
    P2 = np.divide(counts, sums, out=np.zeros_like(counts), where=sums > 1e-15)
    zero_contexts = sums[:, :, 0] <= 1e-15
    if np.any(zero_contexts):
        P2[zero_contexts] = 1.0 / k_int
    return P2


def nll_order1(y_seq: np.ndarray, P1: np.ndarray) -> float:
    """Average one-step negative log-likelihood under order-1 model."""
    matrix = np.asarray(P1, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("P1 must have shape (k, k)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("P1 must contain only finite values")

    k_int = matrix.shape[0]
    seq = _validate_sequence(y_seq, k_int, min_len=2)

    probs = matrix[seq[:-1], seq[1:]]
    return float(-np.mean(np.log(np.clip(probs, 1e-300, 1.0))))


def nll_order2(y_seq: np.ndarray, P2: np.ndarray) -> float:
    """Average one-step negative log-likelihood under order-2 model."""
    tensor = np.asarray(P2, dtype=float)
    if tensor.ndim != 3 or tensor.shape[0] != tensor.shape[1] or tensor.shape[1] != tensor.shape[2]:
        raise ValueError("P2 must have shape (k, k, k)")
    if not np.all(np.isfinite(tensor)):
        raise ValueError("P2 must contain only finite values")

    k_int = tensor.shape[0]
    seq = _validate_sequence(y_seq, k_int, min_len=3)

    probs = tensor[seq[:-2], seq[1:-1], seq[2:]]
    return float(-np.mean(np.log(np.clip(probs, 1e-300, 1.0))))


def prediction_gap(y_seq: np.ndarray, k: int, smoothing: float = 1e-6) -> float:
    """Return NLL1 - NLL2 when both models are fit on the same sequence.

    For an unbiased estimate, fit on train and evaluate on held-out data.
    """
    P1 = fit_markov_order1(y_seq, k, smoothing=smoothing)
    P2 = fit_markov_order2(y_seq, k, smoothing=smoothing)
    return nll_order1(y_seq, P1) - nll_order2(y_seq, P2)
