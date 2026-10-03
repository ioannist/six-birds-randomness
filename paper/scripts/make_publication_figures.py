#!/usr/bin/env python3
"""Publication figures for the manuscript, built from the tracked data in paper/data."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

# Reference palette: categorical slots 1-3 validate on all pairs (scatter-safe).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
FAMILY = {
    "perturbed_lumpable": (BLUE, "o", "perturbed lumpable"),
    "metastable": (ORANGE, "s", "metastable"),
    "hidden_types": (AQUA, "^", "hidden types"),
}

WIDTH = 6.3  # inches, matches the text block


def setup_style() -> None:
    usetex = shutil.which("latex") is not None
    plt.rcParams.update({
        "text.usetex": usetex,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"] if usetex else ["DejaVu Serif"],
        "mathtext.fontset": "cm",
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "lines.markersize": 5.5,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    })


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def panel_label(ax, text: str) -> None:
    ax.text(-0.02, 1.04, text, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="right", color=INK)


# ---------------------------------------------------------------- Figure 1


def concept_figure(out: Path) -> None:
    """Closed versus non-closed package: per-microstate futures and their mixture."""
    closed = [[0.6, 0.3, 0.1]] * 3
    open_ = [[0.80, 0.15, 0.05], [0.25, 0.60, 0.15], [0.15, 0.25, 0.60]]
    weights = [0.5, 0.3, 0.2]
    titles = [r"(a) closed package: $\mathsf{CD}=0$",
              r"(b) non-closed package: $\mathsf{CD}>0$"]

    fig = plt.figure(figsize=(WIDTH, 2.6))
    for col, rows in enumerate([closed, open_]):
        x0 = 0.5 * col
        fig.patches.append(patches.FancyBboxPatch(
            (x0 + 0.075, 0.03), 0.215, 0.80, boxstyle="round,pad=0.0,rounding_size=0.02",
            linewidth=0.8, edgecolor=AXIS, facecolor="#f4f4f1", transform=fig.transFigure,
            zorder=-5))
        fig.text(x0 + 0.25, 0.97, titles[col], ha="center", va="top", fontsize=9.5)
        fig.text(x0 + 0.1825, 0.855, r"package $y$", ha="center", va="bottom",
                 fontsize=8, color=INK2)
        for i, (row, w) in enumerate(zip(rows, weights)):
            ax = fig.add_axes([x0 + 0.10, 0.60 - 0.265 * i, 0.165, 0.19])
            ax.set_facecolor("none")
            ax.bar(range(3), row, width=0.6, color=BLUE, zorder=2)
            ax.set_ylim(0, 1)
            ax.set_xticks(range(3), ["A", "B", "C"] if i == 2 else ["", "", ""])
            ax.set_yticks([])
            ax.grid(False)
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="x", length=0, pad=1.5, labelsize=7)
            fig.text(x0 + 0.06, 0.695 - 0.265 * i, rf"$x_{i + 1}$", ha="right",
                     va="center", fontsize=9.5)
            fig.text(x0 + 0.06, 0.635 - 0.265 * i, rf"$\pi={w}$", ha="right",
                     va="center", fontsize=7, color=INK2)
        ax = fig.add_axes([x0 + 0.345, 0.335, 0.11, 0.19])
        ax.bar(range(3), np.average(np.array(rows), axis=0, weights=weights),
               width=0.6, color=ORANGE, zorder=2)
        ax.set_ylim(0, 1)
        ax.set_xticks(range(3), ["A", "B", "C"])
        ax.set_yticks([])
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="x", length=0, pad=1.5, labelsize=7)
        fig.text(x0 + 0.40, 0.56, "package forecast\n" + r"$\bar p_y$ (weighted mix)",
                 ha="center", va="bottom", fontsize=7.6, color=INK2)
        fig.patches.append(patches.FancyArrowPatch(
            (x0 + 0.295, 0.43), (x0 + 0.335, 0.43), arrowstyle="-|>", mutation_scale=9,
            linewidth=1.0, color=INK2, transform=fig.transFigure))
        note = (r"every $p_{x_i}$ equals $\bar p_y$" if col == 0
                else r"the $p_{x_i}$ differ from $\bar p_y$")
        fig.text(x0 + 0.40, 0.20, note, ha="center", va="top", fontsize=7.6, color=INK2)
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------- Figure 2


def markov_rm_cd(path: Path, out: Path) -> None:
    rows = read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.7), sharey=True)
    for ax, key, title in [
        (axes[0], "rm_stationary", r"stationary-lift mismatch $\mathrm{RM}^{\mathrm{stat}}_\tau$"),
        (axes[1], "rm_uniform", r"uniform-lift mismatch $\mathrm{RM}^{\mathrm{unif}}_\tau$"),
    ]:
        for fam, (color, marker, label) in FAMILY.items():
            for tau, filled in [(1, True), (2, False)]:
                pts = [(float(r[key]), float(r["cd"])) for r in rows
                       if r["family"] == fam and int(r["tau"]) == tau and float(r["cd"]) > 1e-12]
                if not pts:
                    continue
                xs, ys = zip(*pts)
                ax.scatter(xs, ys, s=30, marker=marker, zorder=3,
                           facecolor=color if filled else "white", edgecolor=color, linewidth=1.2,
                           label=f"{label}, $\\tau={tau}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(title)
        ax.set_xlim(1.5e-3, 0.6)
        ax.set_ylim(1.5e-6, 0.12)
    grid = np.logspace(np.log10(1.5e-3), np.log10(0.6), 100)
    axes[0].plot(grid, 0.5 * grid**2, color=INK2, linewidth=1.1, zorder=1)
    axes[0].text(0.06, 0.5 * 0.06**2 / 3.0, r"$\tfrac12(\mathrm{RM}^{\mathrm{stat}})^2$",
                 rotation=33, fontsize=8, color=INK2, ha="center", va="top")
    axes[0].set_ylabel(r"closure deficit $\mathsf{CD}_\tau(\Pi)$ (nats)")
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.17),
               columnspacing=1.2, handletextpad=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------- Figure 3


def markov_knobs(path: Path, out: Path) -> None:
    rows = read_csv(path)
    specs = [
        ("perturbed_lumpable", "heterogeneity_alpha", r"heterogeneity $\alpha$"),
        ("metastable", "p_out", r"escape weight $p_{\mathrm{out}}$"),
        ("hidden_types", "strength", r"type strength $s$"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.15))
    for ax, (fam, knob, xlabel) in zip(axes, specs):
        color, marker, label = FAMILY[fam]
        for tau, filled in [(1, True), (2, False)]:
            pts = sorted((float(r[knob]), float(r["cd"])) for r in rows
                         if r["family"] == fam and int(r["tau"]) == tau)
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, np.maximum(ys, 0.0), color=color, linewidth=1.3,
                    linestyle="-" if filled else (0, (3, 2)), zorder=2)
            ax.scatter(xs, np.maximum(ys, 0.0), marker=marker, s=26, zorder=3,
                       facecolor=color if filled else "white", edgecolor=color, linewidth=1.2,
                       label=rf"$\tau={tau}$")
        ax.set_title(label + (r", $\tau=1$ only" if fam == "hidden_types" else ""), y=1.1)
        ax.set_xlabel(xlabel)
        ax.set_ylim(bottom=0)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        if fam != "hidden_types":
            ax.legend(loc="upper left", handletextpad=0.2)
    axes[0].set_ylabel(r"$\mathsf{CD}_\tau(\Pi)$ (nats)")
    fig.tight_layout(w_pad=1.2)
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------- Figure 4


def budget_figure(metrics_path: Path, pop_path: Path, out: Path) -> None:
    rows = read_csv(metrics_path)
    pop = json.loads(pop_path.read_text())["budget_chain"]
    orders = [int(r["order"]) for r in rows]
    population = [float(r["history_entropy_theory"]) for r in rows]
    per_order = [float(r["nll_exact"]) for r in rows]
    selected = [float(r["nll_selected"]) for r in rows]
    floor = pop["intrinsic"]
    hzy = pop["h_next_given_current"]

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.7), gridspec_kw={"width_ratios": [1, 1.25]})

    ax = axes[0]
    ax.plot(orders, population, color=BLUE, marker="o", zorder=3,
            label=r"population $H(Y_{t+1}\mid S_L)$")
    ax.axhline(floor, color=INK2, linewidth=1.0)
    ax.text(5, floor + 0.004, r"microstate floor $H(Y_{t+1}\mid X_t)$", ha="right",
            va="bottom", fontsize=7.8, color=INK2)
    ax.annotate("", xy=(0.25, floor), xytext=(0.25, hzy),
                arrowprops=dict(arrowstyle="<->", lw=0.9, color=ORANGE))
    ax.text(0.4, (floor + hzy) / 2, rf"$\mathsf{{CD}}_1={pop['cd']:.3f}$", color=INK,
            fontsize=8, va="center")
    ax.set_xlabel(r"packaged memory order $L$")
    ax.set_ylabel("log loss (nats)")
    ax.set_ylim(0.92, 1.11)
    ax.set_xticks(orders)
    ax.legend(loc="upper right")
    panel_label(ax, "a")

    ax = axes[1]
    ax.plot(orders, population, color=BLUE, marker="o", zorder=3, label="population entropy")
    ax.plot(orders, selected, color=ORANGE, marker="s", zorder=4,
            label="validation-selected model, test loss")
    ax.scatter(orders, per_order, marker="o", s=26, facecolor="white", edgecolor=ORANGE,
               linewidth=1.2, zorder=5, label="each fitted order, test loss")
    ax.set_ylim(1.025, 1.10)
    ax.annotate(rf"order 5: {per_order[5]:.3f} (off scale)", xy=(5, 1.0995), xytext=(3.4, 1.0985),
                fontsize=7.6, color=INK2, ha="center",
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=INK2))
    ax.set_xlabel(r"packaged memory order $L$")
    ax.set_xticks(orders)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.86))
    panel_label(ax, "b")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------- Figure 5


def hashing_figure(path: Path, out: Path) -> None:
    rows = read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.75))

    ax = axes[0]
    colors = {8: BLUE, 12: ORANGE, 16: AQUA, 20: INK2}
    markers = {8: "o", 12: "s", 16: "^", 20: "D"}
    grid = np.logspace(-6.2, 0, 200)
    ax.plot(grid, grid, color=AXIS, linewidth=1.0, zorder=1)
    for n in (8, 12, 16, 20):
        pts = [(float(r["baseline_exact"]), float(r["empirical_success"]), float(r["success_se"]))
               for r in rows if r["distribution"] == "uniform" and int(r["n_bits"]) == n]
        ref, emp, se = map(np.array, zip(*pts))
        hit = emp > 0
        ax.errorbar(ref[hit], emp[hit], yerr=se[hit], fmt=markers[n], color=colors[n],
                    markersize=4.5, elinewidth=0.8, capsize=0, zorder=3, label=rf"$n={n}$")
        ax.scatter(ref[~hit], np.full((~hit).sum(), 1.5e-3), marker=markers[n], s=18,
                   facecolor="white", edgecolor=colors[n], linewidth=1.0, zorder=3)
    ax.axhline(2.4e-3, color=GRID, linewidth=0.8)
    ax.text(1.2e-2, 1.5e-3, "no success\nin 300 trials", fontsize=7, color=MUTED, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-6, 1.3)
    ax.set_ylim(1.1e-3, 1.3)
    ax.set_xlabel("ideal random-function reference")
    ax.set_ylabel("empirical inversion success")
    ax.legend(loc="upper left", handletextpad=0.2)
    panel_label(ax, "a")

    ax = axes[1]
    for dist, color, marker, label in [
        ("uniform", BLUE, "o", "uniform ($2^{32}$ inputs)"),
        ("medium_mixture", ORANGE, "s", "50/50 mixture"),
        ("low_entropy", AQUA, "^", "dictionary (256 inputs)"),
    ]:
        pts = sorted((int(r["q"]), float(r["empirical_success"])) for r in rows
                     if r["distribution"] == dist and int(r["n_bits"]) == 16)
        q, s = zip(*pts)
        ax.plot(q, s, color=color, marker=marker, label=label, zorder=3)
    ref = sorted((int(r["q"]), float(r["baseline_exact"])) for r in rows
                 if r["distribution"] == "uniform" and int(r["n_bits"]) == 16)
    ax.plot(*zip(*ref), color=INK2, linewidth=1.0, zorder=2, label="random-function reference")
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"query budget $q$ (16-bit digest)")
    ax.set_ylabel("inversion success")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.62))
    panel_label(ax, "b")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------- Figure 6


def clustering_figure(path: Path, out: Path) -> None:
    rows = sorted(read_csv(path), key=lambda r: int(r["k"]))
    k = [int(r["k"]) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.3))
    ax = axes[0]
    ax.plot(k, [float(r["cd_emp_raw"]) for r in rows], color=BLUE, marker="o")
    ax.set_xlabel(r"number of clusters $k$")
    ax.set_ylabel(r"plug-in $\widehat I(X_t;Y_{t+1}\mid Y_t)$ (nats)")
    ax.set_xticks(k)
    ax.set_ylim(bottom=0)
    panel_label(ax, "a")
    ax = axes[1]
    ax.plot(k, [float(r["nll1"]) for r in rows], color=BLUE, marker="o", label="order 1")
    ax.plot(k, [float(r["nll2"]) for r in rows], color=ORANGE, marker="s", label="order 2")
    ax.plot(k, np.log(k), color=INK2, linewidth=1.0, label=r"$\log k$ (uniform guess)")
    ax.set_xlabel(r"number of clusters $k$")
    ax.set_ylabel("held-out log loss (nats)")
    ax.set_xticks(k)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    panel_label(ax, "b")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    setup_style()
    root = Path(__file__).resolve().parents[1]
    data = root / "data"
    out = root / "figures" / "generated"
    out.mkdir(parents=True, exist_ok=True)
    targets = {
        "fig_concept_closure.pdf": lambda p: concept_figure(p),
        "fig_markov_rm_cd.pdf": lambda p: markov_rm_cd(data / "markov_metrics.csv", p),
        "fig_markov_knobs.pdf": lambda p: markov_knobs(data / "markov_metrics.csv", p),
        "fig_budget.pdf": lambda p: budget_figure(
            data / "budget_metrics.csv", data / "population_quantities.json", p),
        "fig_hashing.pdf": lambda p: hashing_figure(data / "hashing_metrics.csv", p),
        "fig_clustering.pdf": lambda p: clustering_figure(data / "rep_clustering_metrics.csv", p),
    }
    for name, build in targets.items():
        build(out / name)
        print(out / name)


if __name__ == "__main__":
    main()
