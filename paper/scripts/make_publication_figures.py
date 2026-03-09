#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def to_float(value: str | None) -> float:
    if value is None:
        return float("nan")
    text = value.strip()
    if text == "":
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def conceptual_schematic(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 3.9))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_axis_off()

    arrow_color = "#262626"
    edge_color = "#333333"
    box_specs = [
        (0.08, 0.58, 0.20, 0.18, r"$X_t$", "#eef4ff"),
        (0.39, 0.58, 0.23, 0.18, r"$Y_t = \Pi(X_t)$", "#f6f6f6"),
        (0.74, 0.58, 0.18, 0.18, r"$Y_{t+\tau}$", "#eef8f0"),
    ]
    for x, y, w, h, label, facecolor in box_specs:
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=1.5,
            edgecolor=edge_color,
            facecolor=facecolor,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2.0,
            y + h / 2.0,
            label,
            ha="center",
            va="center",
            fontsize=12.5,
            transform=ax.transAxes,
        )

    arrow = dict(arrowstyle="-|>", lw=1.6, color=arrow_color, mutation_scale=11)
    ax.annotate("", xy=(0.39, 0.67), xytext=(0.28, 0.67), xycoords="axes fraction", arrowprops=arrow)
    ax.text(
        0.39,
        0.81,
        r"packaging $\Pi$",
        ha="center",
        va="bottom",
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none"),
        transform=ax.transAxes,
    )

    ax.annotate("", xy=(0.74, 0.67), xytext=(0.62, 0.67), xycoords="axes fraction", arrowprops=arrow)
    ax.text(
        0.74,
        0.81,
        "packaged prediction",
        ha="center",
        va="bottom",
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none"),
        transform=ax.transAxes,
    )

    ax.annotate("", xy=(0.74, 0.37), xytext=(0.18, 0.37), xycoords="axes fraction", arrowprops=arrow)
    ax.text(
        0.46,
        0.30,
        r"full micro prediction from $X_t$",
        ha="center",
        va="center",
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none"),
        transform=ax.transAxes,
    )

    eq_box = patches.FancyBboxPatch(
        (0.17, 0.08),
        0.66,
        0.13,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=0.9,
        edgecolor="#d0d0d0",
        facecolor="#fafafa",
        transform=ax.transAxes,
    )
    ax.add_patch(eq_box)
    ax.text(
        0.5,
        0.145,
        r"$H(Y_{t+\tau}\mid Y_t)=H(Y_{t+\tau}\mid X_t)+\mathsf{CD}_\tau(\Pi)$",
        ha="center",
        va="center",
        fontsize=13.2,
        transform=ax.transAxes,
    )

    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def markov_rm_vs_cd(path: Path, out_path: Path) -> None:
    rows = read_csv_rows(path)
    points = []
    for r in rows:
        x = to_float(r.get("rm_uniform"))
        y = to_float(r.get("cd"))
        fam = (r.get("family") or "").strip() or "unknown"
        if np.isfinite(x) and np.isfinite(y):
            points.append((x, y, fam))

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    families = sorted({p[2] for p in points})
    for i, fam in enumerate(families):
        xs = [p[0] for p in points if p[2] == fam]
        ys = [p[1] for p in points if p[2] == fam]
        ax.scatter(xs, ys, marker=markers[i % len(markers)], label=fam)

    if len(points) >= 2:
        x_all = np.array([p[0] for p in points], dtype=float)
        y_all = np.array([p[1] for p in points], dtype=float)
        m, b = np.polyfit(x_all, y_all, 1)
        x_line = np.linspace(np.min(x_all), np.max(x_all), 100)
        ax.plot(x_line, m * x_line + b, linestyle="--", linewidth=1.5)

    ax.set_xlabel(r"Route mismatch $\mathrm{RM}_{\tau}^{\mathrm{uniform}}$")
    ax.set_ylabel(r"Closure deficit $\mathsf{CD}_{\tau}(\Pi)$")
    ax.legend(frameon=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def markov_cd_vs_heterogeneity(path: Path, out_path: Path) -> None:
    rows = read_csv_rows(path)
    curves: dict[int, list[tuple[float, float]]] = {}
    for r in rows:
        if (r.get("family") or "").strip() != "perturbed_lumpable":
            continue
        alpha = to_float(r.get("heterogeneity_alpha"))
        cd = to_float(r.get("cd"))
        tau = to_float(r.get("tau"))
        if np.isfinite(alpha) and np.isfinite(cd) and np.isfinite(tau):
            curves.setdefault(int(round(tau)), []).append((alpha, cd))

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for tau in sorted(curves):
        pts = sorted(curves[tau], key=lambda t: t[0])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=rf"$\tau={tau}$")

    ax.set_xlabel(r"Heterogeneity $\alpha$")
    ax.set_ylabel(r"Closure deficit $\mathsf{CD}_{\tau}(\Pi)$")
    ax.legend(frameon=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def budget_curve(path: Path, out_path: Path) -> None:
    rows = read_csv_rows(path)
    by_order: dict[int, list[float]] = {}
    hyy_vals = []
    for r in rows:
        order = to_float(r.get("order"))
        nll = to_float(r.get("nll"))
        hyy = to_float(r.get("hyy_theory"))
        if np.isfinite(order) and np.isfinite(nll):
            by_order.setdefault(int(round(order)), []).append(nll)
        if np.isfinite(hyy):
            hyy_vals.append(hyy)

    orders = sorted(by_order)
    nlls = [float(np.mean(by_order[o])) for o in orders]
    hyy = float(np.mean(hyy_vals)) if hyy_vals else float("nan")

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(orders, nlls, marker="o", label="held-out NLL")
    if np.isfinite(hyy):
        ax.axhline(hyy, linestyle="--", linewidth=1.5, label=r"$H(Y_{t+\tau}\mid Y_t)$ theory")
    ax.set_xlabel("Memory order")
    ax.set_ylabel("Held-out predictive log loss (nats)")
    ax.legend(frameon=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def hashing_inversion(path: Path, out_path: Path) -> None:
    rows = read_csv_rows(path)
    grouped: dict[int, list[tuple[float, float]]] = {}
    for r in rows:
        if (r.get("distribution") or "").strip() != "uniform":
            continue
        x = to_float(r.get("baseline_q_over_2n"))
        y = to_float(r.get("empirical_success"))
        n_bits = to_float(r.get("n_bits"))
        if np.isfinite(x) and np.isfinite(y) and np.isfinite(n_bits) and x <= 0.25:
            grouped.setdefault(int(round(n_bits)), []).append((x, y))

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for n_bits in sorted(grouped):
        pts = sorted(grouped[n_bits], key=lambda t: t[0])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", linestyle="-", label=rf"$n={n_bits}$")

    max_x = max((p[0] for pts in grouped.values() for p in pts), default=0.25)
    ax.plot([0.0, max_x], [0.0, max_x], linestyle="--", linewidth=1.3, label="y=x")
    ax.set_xlabel(r"$q/2^n$ (baseline)")
    ax.set_ylabel("Empirical inversion success")
    ax.legend(frameon=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def hashing_collision_ratio(path: Path, out_path: Path) -> None:
    rows = read_csv_rows(path)
    order = ["uniform", "medium_mixture", "low_entropy"]
    vals = {k: float("nan") for k in order}
    for r in rows:
        d = (r.get("distribution") or "").strip()
        if d in vals:
            vals[d] = to_float(r.get("trunc_collision_ratio"))

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    x = np.arange(len(order))
    y = [vals[d] for d in order]
    ax.bar(x, y)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("Truncated collision ratio")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def appendix_rep_cd(path: Path, out_path: Path) -> None:
    rows = read_csv_rows(path)
    by_k: dict[int, list[float]] = {}
    for r in rows:
        k = to_float(r.get("k"))
        cd = to_float(r.get("cd_emp"))
        if np.isfinite(k) and np.isfinite(cd):
            by_k.setdefault(int(round(k)), []).append(cd)

    ks = sorted(by_k)
    cds = [float(np.mean(by_k[k])) for k in ks]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(ks, cds, marker="o")
    ax.set_xlabel(r"Clusters $k$")
    ax.set_ylabel(r"Empirical CMI $\widehat{I}(X_t;Y_{t+1}\mid Y_t)$")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def appendix_rep_nll(path: Path, out_path: Path) -> None:
    rows = read_csv_rows(path)
    by_k1: dict[int, list[float]] = {}
    by_k2: dict[int, list[float]] = {}
    for r in rows:
        k = to_float(r.get("k"))
        nll1 = to_float(r.get("nll1"))
        nll2 = to_float(r.get("nll2"))
        if np.isfinite(k):
            kk = int(round(k))
            if np.isfinite(nll1):
                by_k1.setdefault(kk, []).append(nll1)
            if np.isfinite(nll2):
                by_k2.setdefault(kk, []).append(nll2)

    ks = sorted(set(by_k1) | set(by_k2))
    n1 = [float(np.mean(by_k1[k])) for k in ks]
    n2 = [float(np.mean(by_k2[k])) for k in ks]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(ks, n1, marker="o", label=r"order-1 NLL")
    ax.plot(ks, n2, marker="s", label=r"order-2 NLL")
    ax.set_xlabel(r"Clusters $k$")
    ax.set_ylabel("Held-out NLL (nats)")
    ax.legend(frameon=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = root / "data"
    out = root / "figures" / "generated"
    ensure_dir(out)

    generated = [
        out / "concept_closure_deficit_schematic.pdf",
        out / "markov_rm_vs_cd.pdf",
        out / "markov_cd_vs_heterogeneity.pdf",
        out / "budget_curve_vs_theory.pdf",
        out / "hashing_inversion_vs_q_over_2n.pdf",
        out / "hashing_collision_ratio_by_distribution.pdf",
        out / "appendix_rep_cd_vs_k.pdf",
        out / "appendix_rep_nll_vs_k.pdf",
    ]

    conceptual_schematic(generated[0])
    markov_rm_vs_cd(data / "markov_metrics.csv", generated[1])
    markov_cd_vs_heterogeneity(data / "markov_metrics.csv", generated[2])
    budget_curve(data / "budget_metrics.csv", generated[3])
    hashing_inversion(data / "hashing_metrics.csv", generated[4])
    hashing_collision_ratio(data / "hashing_randomness_tests.csv", generated[5])
    appendix_rep_cd(data / "rep_clustering_metrics.csv", generated[6])
    appendix_rep_nll(data / "rep_clustering_metrics.csv", generated[7])

    for p in generated:
        print(p)


if __name__ == "__main__":
    main()
