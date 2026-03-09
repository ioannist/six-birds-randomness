from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from randomness_ledger.estimators import (
    fit_markov_order1,
    fit_markov_order2,
    nll_order1,
    nll_order2,
)
from randomness_ledger.markov import make_ergodic, normalize_rows
from randomness_ledger.metrics import (
    closure_deficit,
    decomposition_check,
    intrinsic_term,
    macro_cond_entropy,
    route_mismatch,
)
from randomness_ledger.packaging import macro_kernel


def test_estimators_smoothing_unseen_transitions_and_contexts() -> None:
    k = 3
    train = np.array([0, 0, 0, 1, 0, 0], dtype=int)
    test = np.array([0, 2, 1, 2, 0, 1], dtype=int)

    P1 = fit_markov_order1(train, k, smoothing=1e-6)
    P2 = fit_markov_order2(train, k, smoothing=1e-6)
    nll1 = nll_order1(test, P1)
    nll2 = nll_order2(test, P2)

    assert P1.shape == (k, k)
    assert P2.shape == (k, k, k)
    assert np.allclose(P1.sum(axis=1), 1.0, atol=1e-12)
    assert np.allclose(P2.sum(axis=2), 1.0, atol=1e-12)
    assert np.isfinite(nll1)
    assert np.isfinite(nll2)


def test_metrics_finite_with_explicit_zero_probabilities() -> None:
    P = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    pi_map = np.array([0, 0, 1, 1], dtype=int)
    pi_stationary = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)

    hyx = intrinsic_term(P, pi_map, tau=1, pi_stationary=pi_stationary)
    hyy = macro_cond_entropy(P, pi_map, tau=1, pi_stationary=pi_stationary)
    cd = closure_deficit(P, pi_map, tau=1, pi_stationary=pi_stationary)
    resid = decomposition_check(P, pi_map, tau=1, pi_stationary=pi_stationary)
    rm = route_mismatch(P, pi_map, tau=1, lift="uniform", pi_stationary=pi_stationary)

    vals = [hyx, hyy, cd, resid, rm]
    assert all(np.isfinite(v) for v in vals)
    assert abs(resid) < 1e-8
    assert rm >= -1e-12


def test_large_tau_stability() -> None:
    rng = np.random.default_rng(123)
    P = normalize_rows(rng.random((6, 6)) + 0.1)
    P = make_ergodic(P, eps=1e-3)
    pi_map = np.array([0, 0, 1, 1, 2, 2], dtype=int)
    tau = 50

    macro_u = macro_kernel(P, pi_map, tau=tau, lift="uniform")
    macro_s = macro_kernel(P, pi_map, tau=tau, lift="stationary")

    hyx = intrinsic_term(P, pi_map, tau=tau)
    hyy = macro_cond_entropy(P, pi_map, tau=tau)
    cd = closure_deficit(P, pi_map, tau=tau)
    resid = decomposition_check(P, pi_map, tau=tau)
    rm_u = route_mismatch(P, pi_map, tau=tau, lift="uniform")
    rm_s = route_mismatch(P, pi_map, tau=tau, lift="stationary")

    assert np.allclose(macro_u.sum(axis=1), 1.0, atol=1e-10)
    assert np.allclose(macro_s.sum(axis=1), 1.0, atol=1e-10)

    vals = [hyx, hyy, cd, resid, rm_u, rm_s]
    assert all(np.isfinite(v) for v in vals)
    assert abs(resid) < 1e-6


def test_empty_macro_labels_rejected() -> None:
    P = np.eye(4, dtype=float)
    pi_map = np.array([0, 2, 2, 0], dtype=int)

    with pytest.raises(ValueError):
        _ = macro_kernel(P, pi_map, tau=1, lift="uniform")


def _latest_child_dir(base: Path) -> Path:
    runs = sorted([p for p in base.iterdir() if p.is_dir()])
    if not runs:
        raise RuntimeError(f"no run directories found under {base}")
    return runs[-1]


def _canonical_metrics_hash(metrics_csv: Path) -> str:
    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            row_copy = dict(row)
            row_copy.pop("run_id", None)
            rows.append(row_copy)

    rows.sort(
        key=lambda r: (
            r.get("family", ""),
            str(r.get("seed", "")),
            str(r.get("replicate", "")),
            str(r.get("tau", "")),
            json.dumps(r, sort_keys=True, separators=(",", ":")),
        )
    )
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_markov_bench_canonical_reproducibility_hash(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = root / "configs" / "markov_bench_smoke.json"

    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"
    out_a.mkdir(parents=True, exist_ok=True)
    out_b.mkdir(parents=True, exist_ok=True)

    cmd_a = [
        sys.executable,
        "experiments/markov_bench/run_sweep.py",
        "--config",
        str(config),
        "--outdir",
        str(out_a),
    ]
    cmd_b = [
        sys.executable,
        "experiments/markov_bench/run_sweep.py",
        "--config",
        str(config),
        "--outdir",
        str(out_b),
    ]

    subprocess.run(cmd_a, cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(cmd_b, cwd=root, check=True, capture_output=True, text=True)

    run_a = _latest_child_dir(out_a)
    run_b = _latest_child_dir(out_b)

    hash_a = _canonical_metrics_hash(run_a / "metrics.csv")
    hash_b = _canonical_metrics_hash(run_b / "metrics.csv")

    print(f"canonical_hash_a={hash_a}")
    print(f"canonical_hash_b={hash_b}")

    assert hash_a == hash_b
