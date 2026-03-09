"""Run clustering-based packaging sweeps on observation datasets."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shlex
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from randomness_ledger.estimators import (  # noqa: E402
    fit_markov_order1,
    fit_markov_order2,
    nll_order1,
    nll_order2,
)


def _run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def _parse_k_list(k_list: str) -> list[int]:
    items = [s.strip() for s in k_list.split(",") if s.strip() != ""]
    if not items:
        raise ValueError("k_list must contain at least one integer")
    out: list[int] = []
    for item in items:
        val = int(item)
        if val < 2:
            raise ValueError("all k values must be >= 2")
        out.append(val)
    return out


def _latest_dataset_path() -> Path:
    base = Path("results/rep_packaging")
    if not base.is_dir():
        raise FileNotFoundError("results/rep_packaging does not exist")
    runs = sorted([p for p in base.iterdir() if p.is_dir()])
    for run_dir in reversed(runs):
        candidate = run_dir / "dataset.npz"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("no dataset.npz found under results/rep_packaging")


def _sq_dists(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances from points to centers."""
    diff = X[:, None, :] - centers[None, :, :]
    return np.sum(diff * diff, axis=2)


def kmeans_fit(
    X: np.ndarray, k: int, rng: np.random.Generator, max_iter: int = 100, tol: float = 1e-6
) -> tuple[np.ndarray, float]:
    """Fit k-means with k-means++ initialization and return centers/inertia."""
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim != 2:
        raise ValueError("X must be 2D")
    n, d = X_arr.shape
    if k < 1 or k > n:
        raise ValueError(f"k must be in [1, n], got {k} with n={n}")

    centers = np.empty((k, d), dtype=float)
    first_idx = int(rng.integers(n))
    centers[0] = X_arr[first_idx]

    closest_sq = np.sum((X_arr - centers[0]) ** 2, axis=1)
    for j in range(1, k):
        total = float(np.sum(closest_sq))
        if total <= 0.0:
            idx = int(rng.integers(n))
        else:
            probs = closest_sq / total
            idx = int(rng.choice(n, p=probs))
        centers[j] = X_arr[idx]
        dist_j = np.sum((X_arr - centers[j]) ** 2, axis=1)
        closest_sq = np.minimum(closest_sq, dist_j)

    for _ in range(int(max_iter)):
        d2 = _sq_dists(X_arr, centers)
        labels = np.argmin(d2, axis=1)
        min_d2 = d2[np.arange(n), labels]

        new_centers = np.zeros_like(centers)
        counts = np.bincount(labels, minlength=k)

        for j in range(k):
            if counts[j] > 0:
                new_centers[j] = X_arr[labels == j].mean(axis=0)
            else:
                # Re-seed empty cluster to the farthest currently assigned point.
                far_idx = int(np.argmax(min_d2))
                new_centers[j] = X_arr[far_idx]
                min_d2[far_idx] = -1.0

        shift = float(np.max(np.linalg.norm(new_centers - centers, axis=1)))
        centers = new_centers
        if shift < tol:
            break

    final_d2 = _sq_dists(X_arr, centers)
    final_labels = np.argmin(final_d2, axis=1)
    inertia = float(np.sum(final_d2[np.arange(n), final_labels]))
    return centers, inertia


def kmeans_predict(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Assign nearest-center labels."""
    X_arr = np.asarray(X, dtype=float)
    centers_arr = np.asarray(centers, dtype=float)
    if X_arr.ndim != 2 or centers_arr.ndim != 2 or X_arr.shape[1] != centers_arr.shape[1]:
        raise ValueError("X and centers must be 2D with matching feature dimension")
    d2 = _sq_dists(X_arr, centers_arr)
    return np.argmin(d2, axis=1).astype(np.int64)


def empirical_cmi_x_y1_given_y0(
    x: np.ndarray, y: np.ndarray, nX: int, kY: int
) -> tuple[float, float, float]:
    """Estimate I(X_t;Y_{t+1}|Y_t) via entropy difference from transition counts."""
    x_arr = np.asarray(x, dtype=np.int64)
    y_arr = np.asarray(y, dtype=np.int64)
    if x_arr.shape != y_arr.shape or x_arr.ndim != 1:
        raise ValueError("x and y must be 1D arrays with matching shape")
    if x_arr.size < 2:
        raise ValueError("x and y must have length >= 2")

    x0 = x_arr[:-1]
    y0 = y_arr[:-1]
    y1 = y_arr[1:]
    N = y1.size

    c_y0y1 = np.zeros((kY, kY), dtype=float)
    np.add.at(c_y0y1, (y0, y1), 1.0)

    c_xy0y1 = np.zeros((nX * kY, kY), dtype=float)
    ctx_xy = x0 * kY + y0
    np.add.at(c_xy0y1, (ctx_xy, y1), 1.0)

    def conditional_entropy(counts: np.ndarray) -> float:
        context_totals = counts.sum(axis=1)
        valid = context_totals > 0
        if not np.any(valid):
            return 0.0
        probs = np.zeros_like(counts)
        probs[valid] = counts[valid] / context_totals[valid, None]
        mask = probs > 0
        row_h = np.zeros(counts.shape[0], dtype=float)
        row_h[valid] = -np.sum(probs[valid] * np.log(np.where(mask[valid], probs[valid], 1.0)), axis=1)
        weights = context_totals / float(np.sum(context_totals))
        return float(np.sum(weights * row_h))

    h_y1_given_y0 = conditional_entropy(c_y0y1)
    h_y1_given_xy0 = conditional_entropy(c_xy0y1)
    cd = float(h_y1_given_y0 - h_y1_given_xy0)
    return cd, h_y1_given_y0, h_y1_given_xy0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clustering-based packaging sweep.")
    parser.add_argument("--dataset", default=None, help="Path to dataset.npz (optional).")
    parser.add_argument("--method", default="kmeans", choices=["kmeans"])
    parser.add_argument("--k_list", default="2,4,8,16")
    parser.add_argument("--train_frac", type=float, default=0.5)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--n_init", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260306)
    parser.add_argument("--outdir", default="results/rep_packaging_clustering")
    args = parser.parse_args()

    if not (0.0 < args.train_frac < 1.0):
        raise ValueError("train_frac must be in (0, 1)")
    if args.max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    if args.n_init < 1:
        raise ValueError("n_init must be >= 1")

    k_list = _parse_k_list(args.k_list)
    t0 = time.perf_counter()

    dataset_path = Path(args.dataset) if args.dataset is not None else _latest_dataset_path()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")

    with np.load(dataset_path) as data:
        if "x" not in data.files or "o" not in data.files:
            raise ValueError("dataset must contain arrays 'x' and 'o'")
        x = np.asarray(data["x"], dtype=np.int64)
        o = np.asarray(data["o"], dtype=float)
        y_true = np.asarray(data["y_true"], dtype=np.int64) if "y_true" in data.files else None

    if x.ndim != 1:
        raise ValueError("x must be 1D")
    if o.ndim != 2 or o.shape[0] != x.shape[0]:
        raise ValueError("o must be 2D with same first dimension as x")
    if not np.all(np.isfinite(o)):
        raise ValueError("o contains non-finite values")

    T = int(x.shape[0])
    d = int(o.shape[1])
    nX = int(np.max(x)) + 1
    T_train = int(args.train_frac * T)
    if T_train < 20 or (T - T_train) < 20:
        raise ValueError("train/test split too short; increase T or adjust train_frac")

    o_train = o[:T_train]

    run_id = _run_id()
    run_dir = Path(args.outdir) / run_id
    figs_dir = run_dir / "figs"
    run_dir.mkdir(parents=True, exist_ok=False)
    figs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for k in k_list:
        best_inertia = float("inf")
        best_centers = None
        for init_idx in range(args.n_init):
            rng_i = np.random.default_rng(int(args.seed + 1000 * k + init_idx))
            centers, inertia = kmeans_fit(
                o_train, k=k, rng=rng_i, max_iter=args.max_iter, tol=1e-6
            )
            if inertia < best_inertia:
                best_inertia = float(inertia)
                best_centers = centers
        assert best_centers is not None

        y = kmeans_predict(o, best_centers)
        train_y = y[:T_train]
        test_y = y[T_train:]
        if test_y.size < 3:
            raise ValueError("test split too short for order-2 evaluation")

        P1 = fit_markov_order1(train_y, k)
        P2 = fit_markov_order2(train_y, k)
        nll1 = float(nll_order1(test_y, P1))
        nll2 = float(nll_order2(test_y, P2))
        gap = float(nll1 - nll2)

        cd_raw, h_y1_given_y0, h_y1_given_xy0 = empirical_cmi_x_y1_given_y0(x, y, nX=nX, kY=k)
        denom = max(float(h_y1_given_y0), 1e-12)
        cd_norm = float(cd_raw / denom)

        counts = np.bincount(y, minlength=k).astype(float)
        frac = counts / float(np.sum(counts))
        row = {
            "run_id": run_id,
            "dataset_path": dataset_path.as_posix(),
            "method": args.method,
            "seed": int(args.seed),
            "T": T,
            "d": d,
            "nX": nX,
            "k": int(k),
            "n_init": int(args.n_init),
            "max_iter": int(args.max_iter),
            "inertia_best": best_inertia,
            "used_clusters": int(np.unique(y).size),
            "min_cluster_frac": float(np.min(frac)),
            "nll1": nll1,
            "nll2": nll2,
            "gap": gap,
            "cd_emp": float(cd_raw),
            "cd_emp_raw": float(cd_raw),
            "cd_emp_norm": float(cd_norm),
            "h_y1_given_y0": float(h_y1_given_y0),
            "h_y1_given_xy0": float(h_y1_given_xy0),
        }
        rows.append(row)

    rows = sorted(rows, key=lambda r: int(r["k"]))
    metrics_path = run_dir / "metrics.csv"
    fieldnames = [
        "run_id",
        "dataset_path",
        "method",
        "seed",
        "T",
        "d",
        "nX",
        "k",
        "n_init",
        "max_iter",
        "inertia_best",
        "used_clusters",
        "min_cluster_frac",
        "nll1",
        "nll2",
        "gap",
        "cd_emp",
        "cd_emp_raw",
        "cd_emp_norm",
        "h_y1_given_y0",
        "h_y1_given_xy0",
    ]
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    k_vals = np.array([int(r["k"]) for r in rows], dtype=int)
    cd_vals = np.array([float(r["cd_emp"]) for r in rows], dtype=float)
    nll1_vals = np.array([float(r["nll1"]) for r in rows], dtype=float)
    nll2_vals = np.array([float(r["nll2"]) for r in rows], dtype=float)

    plt.figure(figsize=(6, 4))
    plt.plot(k_vals, cd_vals, marker="o")
    plt.xlabel("k")
    plt.ylabel("cd_emp")
    plt.title("Empirical CD vs k")
    plt.tight_layout()
    plt.savefig(figs_dir / "cd_vs_k.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(k_vals, nll1_vals, marker="o", label="nll1")
    plt.plot(k_vals, nll2_vals, marker="o", label="nll2")
    plt.xlabel("k")
    plt.ylabel("nll")
    plt.title("Held-out NLL vs k")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figs_dir / "nll_vs_k.png", dpi=150)
    plt.close()

    cd_diffs = cd_vals[1:] - cd_vals[:-1] if cd_vals.size >= 2 else np.array([], dtype=float)
    violations = int(np.sum(cd_diffs > 1e-3))
    summary = {
        "dataset_path": dataset_path.as_posix(),
        "T": T,
        "d": d,
        "k_list": [int(k) for k in k_vals],
        "cd_first": float(cd_vals[0]) if cd_vals.size else float("nan"),
        "cd_last": float(cd_vals[-1]) if cd_vals.size else float("nan"),
        "cd_drop": float(cd_vals[0] - cd_vals[-1]) if cd_vals.size else float("nan"),
        "cd_nonincreasing_violations": violations,
        "nll1_first": float(nll1_vals[0]) if nll1_vals.size else float("nan"),
        "nll1_last": float(nll1_vals[-1]) if nll1_vals.size else float("nan"),
        "rows": [
            {
                "k": int(r["k"]),
                "cd_emp": float(r["cd_emp"]),
                "nll1": float(r["nll1"]),
                "nll2": float(r["nll2"]),
            }
            for r in rows
        ],
    }
    _write_json(run_dir / "summary.json", summary)

    config = {
        "run_id": run_id,
        "dataset_path": dataset_path.as_posix(),
        "cli_args": vars(args),
    }
    _write_json(run_dir / "config.json", config)

    runtime_seconds = float(time.perf_counter() - t0)
    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(shlex.quote(a) for a in sys.argv),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
        "runtime_seconds": runtime_seconds,
    }
    _write_json(run_dir / "manifest.json", manifest)

    print(str(run_dir))


if __name__ == "__main__":
    main()
