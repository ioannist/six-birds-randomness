"""Analyze Markov benchmark sweep outputs and write diagnostic plots."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _safe_float(value: Any) -> float:
    """Parse numeric values from CSV; return NaN for empty/missing values."""
    if value is None:
        return float("nan")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    s = str(value).strip()
    if s == "":
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _resolve_run_dir(run_dir_arg: str | None) -> Path:
    """Resolve analysis run directory from argument or latest available run."""
    if run_dir_arg is not None:
        run_dir = Path(run_dir_arg)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"run_dir does not exist: {run_dir}")
        if not (run_dir / "metrics.csv").is_file():
            raise FileNotFoundError(f"metrics.csv not found in run_dir: {run_dir}")
        return run_dir

    base = Path("results/markov_bench")
    if not base.is_dir():
        raise FileNotFoundError("results/markov_bench directory does not exist")

    candidates = sorted([p for p in base.iterdir() if p.is_dir()])
    for candidate in reversed(candidates):
        if (candidate / "metrics.csv").is_file():
            return candidate
    raise FileNotFoundError("no run directory containing metrics.csv found")


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation with NaN for degenerate/short vectors."""
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.allclose(np.std(x), 0.0) or np.allclose(np.std(y), 0.0):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _load_rows(metrics_path: Path) -> list[dict[str, Any]]:
    """Load metrics CSV rows with selected numeric fields parsed as floats."""
    numeric_fields = {
        "rm_uniform",
        "cd",
        "intrinsic",
        "hyy",
        "resid",
        "tau",
        "heterogeneity_alpha",
    }
    rows: list[dict[str, Any]] = []
    with metrics_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if key in numeric_fields:
                    row[key] = _safe_float(value)
                else:
                    row[key] = value
            rows.append(row)
    return rows


def _plot_rm_vs_cd(rows: list[dict[str, Any]], figs_dir: Path) -> float:
    """Scatter RM vs CD with optional regression line; returns Pearson r."""
    x = np.array([_safe_float(r.get("rm_uniform")) for r in rows], dtype=float)
    y = np.array([_safe_float(r.get("cd")) for r in rows], dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    r = _pearson_r(x, y)

    plt.figure(figsize=(6, 4))
    if x.size > 0:
        plt.scatter(x, y, s=35, alpha=0.8)
    else:
        plt.text(0.5, 0.5, "No finite RM/CD rows", ha="center", va="center")

    if x.size >= 2:
        m, b = np.polyfit(x, y, 1)
        xx = np.linspace(np.min(x), np.max(x), 100)
        plt.plot(xx, m * xx + b, linewidth=1.5)

    plt.xlabel("rm_uniform")
    plt.ylabel("cd")
    plt.title("RM vs CD")
    plt.tight_layout()
    out = figs_dir / "rm_vs_cd_scatter.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return r


def _plot_cd_vs_heterogeneity_by_tau(rows: list[dict[str, Any]], figs_dir: Path) -> None:
    """Plot CD against heterogeneity_alpha for perturbed_lumpable rows by tau."""
    subset = []
    for r in rows:
        if r.get("family") != "perturbed_lumpable":
            continue
        alpha = _safe_float(r.get("heterogeneity_alpha"))
        cd = _safe_float(r.get("cd"))
        tau = _safe_float(r.get("tau"))
        if np.isfinite(alpha) and np.isfinite(cd) and np.isfinite(tau):
            subset.append((tau, alpha, cd))

    plt.figure(figsize=(6, 4))
    if subset:
        grouped: dict[float, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
        for tau, alpha, cd in subset:
            grouped[tau][alpha].append(cd)

        for tau in sorted(grouped.keys()):
            alpha_to_vals = grouped[tau]
            xs = sorted(alpha_to_vals.keys())
            ys = [float(np.mean(alpha_to_vals[a])) for a in xs]
            label = f"tau={int(tau) if float(tau).is_integer() else tau:g}"
            plt.plot(xs, ys, marker="o", linewidth=1.5, label=label)
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No perturbed_lumpable rows", ha="center", va="center")

    plt.xlabel("heterogeneity_alpha")
    plt.ylabel("cd")
    plt.title("CD vs heterogeneity_alpha by tau")
    plt.tight_layout()
    plt.savefig(figs_dir / "cd_vs_heterogeneity_by_tau.png", dpi=150)
    plt.close()


def _plot_entropy_decomposition(rows: list[dict[str, Any]], figs_dir: Path) -> None:
    """Plot mean intrinsic + CD (+residual) and overlay mean HYY by family/tau."""
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        intrinsic = _safe_float(r.get("intrinsic"))
        cd = _safe_float(r.get("cd"))
        hyy = _safe_float(r.get("hyy"))
        resid = _safe_float(r.get("resid"))
        tau = _safe_float(r.get("tau"))
        fam = str(r.get("family"))
        if np.isfinite(intrinsic) and np.isfinite(cd) and np.isfinite(hyy) and np.isfinite(resid) and np.isfinite(tau):
            grouped[(fam, tau)].append(
                {"intrinsic": intrinsic, "cd": cd, "hyy": hyy, "resid": resid}
            )

    plt.figure(figsize=(9, 4))
    if grouped:
        keys = sorted(grouped.keys(), key=lambda t: (t[0], t[1]))
        labels: list[str] = []
        intrinsic_vals = []
        cd_vals = []
        resid_vals = []
        hyy_vals = []
        for fam, tau in keys:
            vals = grouped[(fam, tau)]
            labels.append(f"{fam}\nτ={int(tau) if float(tau).is_integer() else tau:g}")
            intrinsic_vals.append(float(np.mean([v["intrinsic"] for v in vals])))
            cd_vals.append(float(np.mean([v["cd"] for v in vals])))
            resid_vals.append(float(np.mean([v["resid"] for v in vals])))
            hyy_vals.append(float(np.mean([v["hyy"] for v in vals])))

        x = np.arange(len(labels))
        intrinsic_arr = np.array(intrinsic_vals, dtype=float)
        cd_arr = np.array(cd_vals, dtype=float)
        resid_arr = np.array(resid_vals, dtype=float)
        hyy_arr = np.array(hyy_vals, dtype=float)

        plt.bar(x, intrinsic_arr, label="intrinsic_mean")
        plt.bar(x, cd_arr, bottom=intrinsic_arr, label="cd_mean")
        plt.bar(x, resid_arr, bottom=intrinsic_arr + cd_arr, label="resid_mean")
        plt.plot(x, hyy_arr, "ko-", linewidth=1.2, markersize=4, label="hyy_mean")
        plt.xticks(x, labels, rotation=0)
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No finite decomposition rows", ha="center", va="center")
        plt.xticks([])

    plt.ylabel("nats")
    plt.title("Entropy decomposition by family and tau")
    plt.tight_layout()
    plt.savefig(figs_dir / "entropy_decomposition_stacked.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Markov benchmark sweep results.")
    parser.add_argument(
        "--run_dir",
        default=None,
        help="Optional results/markov_bench/<run_id> directory. If omitted, use latest.",
    )
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    metrics_path = run_dir / "metrics.csv"
    figs_dir = run_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(metrics_path)
    print(f"resolved_run_dir: {run_dir}")
    print(f"rows_loaded: {len(rows)}")

    r_all = _plot_rm_vs_cd(rows, figs_dir)
    _plot_cd_vs_heterogeneity_by_tau(rows, figs_dir)
    _plot_entropy_decomposition(rows, figs_dir)

    x_all = np.array([_safe_float(r.get("rm_uniform")) for r in rows], dtype=float)
    y_all = np.array([_safe_float(r.get("cd")) for r in rows], dtype=float)
    mask_all = np.isfinite(x_all) & np.isfinite(y_all)
    x_all = x_all[mask_all]
    y_all = y_all[mask_all]

    pert_rows = [r for r in rows if r.get("family") == "perturbed_lumpable"]
    x_pert = np.array([_safe_float(r.get("rm_uniform")) for r in pert_rows], dtype=float)
    y_pert = np.array([_safe_float(r.get("cd")) for r in pert_rows], dtype=float)
    mask_pert = np.isfinite(x_pert) & np.isfinite(y_pert)
    x_pert = x_pert[mask_pert]
    y_pert = y_pert[mask_pert]
    r_pert = _pearson_r(x_pert, y_pert)

    print(f"pearson_r_rm_uniform_cd_all: {r_all}")
    print(f"pearson_r_rm_uniform_cd_perturbed_lumpable: {r_pert}")

    plot_files = sorted([p.name for p in figs_dir.glob("*.png")])
    print("plots_written:")
    for name in plot_files:
        print(f"- {name}")


if __name__ == "__main__":
    main()
