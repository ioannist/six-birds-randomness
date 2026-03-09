"""Train a minimal discrete bottleneck encoder and compare to clustering baseline."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shlex
import subprocess
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
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    items = [s.strip() for s in k_list.split(",") if s.strip()]
    if not items:
        raise ValueError("k_list must contain at least one integer")
    out = [int(x) for x in items]
    if any(k < 2 for k in out):
        raise ValueError("all k values must be >= 2")
    return out


def _latest_dataset_path() -> Path:
    base = Path("results/rep_packaging")
    if not base.is_dir():
        raise FileNotFoundError("results/rep_packaging does not exist")
    runs = sorted([p for p in base.iterdir() if p.is_dir()])
    for run in reversed(runs):
        p = run / "dataset.npz"
        if p.is_file():
            return p
    raise FileNotFoundError("no dataset.npz found under results/rep_packaging")


def _safe_resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except Exception:
        return Path(path)


def _load_clustering_metrics(metrics_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with metrics_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _find_existing_baseline(dataset_path: Path, k_list: list[int]) -> Path | None:
    base = Path("results/rep_packaging_clustering")
    if not base.is_dir():
        return None
    target = _safe_resolve(dataset_path)
    runs = sorted([p for p in base.iterdir() if p.is_dir()])
    wanted = set(k_list)

    for run in reversed(runs):
        metrics = run / "metrics.csv"
        if not metrics.is_file():
            continue
        rows = _load_clustering_metrics(metrics)
        if not rows:
            continue
        dataset_vals = {r.get("dataset_path", "") for r in rows}
        matches_dataset = False
        for ds in dataset_vals:
            p = Path(ds)
            if _safe_resolve(p) == target:
                matches_dataset = True
                break
        if not matches_dataset:
            continue

        ks = {int(r["k"]) for r in rows if "k" in r}
        if wanted.issubset(ks):
            return run
    return None


def _run_baseline(dataset_path: Path, k_list_str: str, seed: int) -> Path:
    cmd = [
        sys.executable,
        "experiments/rep_packaging/run_clustering_packaging.py",
        "--dataset",
        dataset_path.as_posix(),
        "--k_list",
        k_list_str,
        "--seed",
        str(seed),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "auto baseline run failed:\n"
            f"command: {' '.join(shlex.quote(c) for c in cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    for ln in reversed(lines):
        p = Path(ln)
        if p.is_dir():
            return p
    raise RuntimeError(f"could not parse baseline run directory from stdout:\n{proc.stdout}")


class DiscreteEncoderModel(nn.Module):
    def __init__(self, d: int, hidden_dim: int, k: int) -> None:
        super().__init__()
        self.k = int(k)
        self.encoder = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, k),
        )
        self.transition_head = nn.Linear(k, k, bias=False)
        self.obs_head = nn.Sequential(
            nn.Linear(k, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d),
        )

    def encode_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def _compute_loss(
    model: DiscreteEncoderModel,
    o_t: torch.Tensor,
    o_tp1: torch.Tensor,
    tau: float,
    lambda_obs: float,
    lambda_usage: float,
    lambda_conf: float,
    training: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits_t = model.encode_logits(o_t)
    logits_tp1 = model.encode_logits(o_tp1)

    q_t = F.softmax(logits_t, dim=-1)
    q_tp1 = F.softmax(logits_tp1, dim=-1).detach()

    if training:
        z_t = F.gumbel_softmax(logits_t, tau=tau, hard=True, dim=-1)
    else:
        idx = torch.argmax(logits_t, dim=-1)
        z_t = F.one_hot(idx, num_classes=model.k).float()

    pred_next_code_logits = model.transition_head(z_t)
    pred_next_obs = model.obs_head(z_t)

    code_loss = -(q_tp1 * F.log_softmax(pred_next_code_logits, dim=-1)).sum(dim=-1).mean()
    obs_loss = F.mse_loss(pred_next_obs, o_tp1)

    p_bar = q_t.mean(dim=0)
    p_bar = torch.clamp(p_bar, min=1e-12)
    uniform = torch.full_like(p_bar, 1.0 / model.k)
    usage_loss = (p_bar * (torch.log(p_bar) - torch.log(uniform))).sum()

    conf_loss = -(q_t * torch.log(torch.clamp(q_t, min=1e-12))).sum(dim=-1).mean()

    total = code_loss + lambda_obs * obs_loss + lambda_usage * usage_loss + lambda_conf * conf_loss
    parts = {
        "code": float(code_loss.detach().cpu()),
        "obs": float(obs_loss.detach().cpu()),
        "usage": float(usage_loss.detach().cpu()),
        "conf": float(conf_loss.detach().cpu()),
        "total": float(total.detach().cpu()),
    }
    return total, parts


def _temperature_for_epoch(epoch: int, epochs: int, tau_start: float, tau_end: float) -> float:
    if epochs <= 1:
        return float(tau_end)
    alpha = float(epoch) / float(epochs - 1)
    return float((1.0 - alpha) * tau_start + alpha * tau_end)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train discrete bottleneck encoder.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--k_list", default="4,8")
    parser.add_argument("--train_frac", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tau_start", type=float, default=1.0)
    parser.add_argument("--tau_end", type=float, default=0.3)
    parser.add_argument("--lambda_obs", type=float, default=1.0)
    parser.add_argument("--lambda_usage", type=float, default=0.1)
    parser.add_argument("--lambda_conf", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260307)
    parser.add_argument("--baseline_seed", type=int, default=20260306)
    parser.add_argument("--outdir", default="results/rep_packaging_nn")
    args = parser.parse_args()

    if not (0.0 < args.train_frac < 1.0):
        raise ValueError("train_frac must be in (0, 1)")
    if args.epochs < 1:
        raise ValueError("epochs must be >= 1")
    if args.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if args.hidden_dim < 1:
        raise ValueError("hidden_dim must be >= 1")
    if args.lr <= 0:
        raise ValueError("lr must be > 0")
    if args.tau_start <= 0 or args.tau_end <= 0:
        raise ValueError("taus must be > 0")

    k_list = _parse_k_list(args.k_list)
    t0 = time.perf_counter()

    dataset_path = Path(args.dataset) if args.dataset else _latest_dataset_path()
    dataset_path = _safe_resolve(dataset_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")

    with np.load(dataset_path) as data:
        if "o" not in data.files:
            raise ValueError("dataset must contain 'o'")
        o = np.asarray(data["o"], dtype=np.float32)
    if o.ndim != 2:
        raise ValueError("o must be 2D")
    if not np.all(np.isfinite(o)):
        raise ValueError("o contains non-finite values")

    T, d = int(o.shape[0]), int(o.shape[1])
    T_train = int(args.train_frac * T)
    if T_train < 20 or (T - T_train) < 20:
        raise ValueError("train/test split too short; increase T or adjust train_frac")

    baseline_run = _find_existing_baseline(dataset_path, k_list)
    if baseline_run is None:
        baseline_run = _run_baseline(dataset_path, args.k_list, args.baseline_seed)
    baseline_metrics = _load_clustering_metrics(baseline_run / "metrics.csv")
    baseline_by_k: dict[int, dict[str, Any]] = {}
    for row in baseline_metrics:
        k = int(row["k"])
        baseline_by_k[k] = row
    missing = [k for k in k_list if k not in baseline_by_k]
    if missing:
        raise RuntimeError(f"baseline missing requested k values: {missing}")

    run_id = _run_id()
    run_dir = Path(args.outdir) / run_id
    figs_dir = run_dir / "figs"
    run_dir.mkdir(parents=True, exist_ok=False)
    figs_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    o_t_all = o[: T_train - 1]
    o_tp1_all = o[1:T_train]
    n_pairs = int(o_t_all.shape[0])
    n_val = max(1, int(0.2 * n_pairs))
    n_tr = n_pairs - n_val
    if n_tr < 10:
        raise ValueError("not enough training pairs after validation split")

    tr_t = torch.tensor(o_t_all[:n_tr], dtype=torch.float32, device=device)
    tr_tp1 = torch.tensor(o_tp1_all[:n_tr], dtype=torch.float32, device=device)
    va_t = torch.tensor(o_t_all[n_tr:], dtype=torch.float32, device=device)
    va_tp1 = torch.tensor(o_tp1_all[n_tr:], dtype=torch.float32, device=device)
    all_o_tensor = torch.tensor(o, dtype=torch.float32, device=device)

    rows: list[dict[str, Any]] = []
    for k in k_list:
        torch.manual_seed(int(args.seed + 31 * k))
        model = DiscreteEncoderModel(d=d, hidden_dim=args.hidden_dim, k=k).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=float(args.lr))

        best_state = None
        best_val = float("inf")
        best_epoch = -1
        best_train_loss = float("nan")
        train_losses: list[float] = []
        val_losses: list[float] = []

        for epoch in range(int(args.epochs)):
            tau = _temperature_for_epoch(epoch, int(args.epochs), args.tau_start, args.tau_end)
            model.train()
            rng = np.random.default_rng(int(args.seed + 100_000 * k + epoch))
            perm = rng.permutation(n_tr)

            epoch_loss = 0.0
            seen = 0
            for start in range(0, n_tr, int(args.batch_size)):
                idx = perm[start : start + int(args.batch_size)]
                idx_t = torch.tensor(idx.tolist(), dtype=torch.long, device=device)
                batch_t = tr_t.index_select(0, idx_t)
                batch_tp1 = tr_tp1.index_select(0, idx_t)
                optim.zero_grad(set_to_none=True)
                loss, _ = _compute_loss(
                    model,
                    batch_t,
                    batch_tp1,
                    tau=tau,
                    lambda_obs=float(args.lambda_obs),
                    lambda_usage=float(args.lambda_usage),
                    lambda_conf=float(args.lambda_conf),
                    training=True,
                )
                loss.backward()
                optim.step()
                bs = int(batch_t.shape[0])
                epoch_loss += float(loss.detach().cpu()) * bs
                seen += bs
            train_loss = epoch_loss / float(max(seen, 1))
            train_losses.append(train_loss)

            model.eval()
            with torch.no_grad():
                val_loss, _ = _compute_loss(
                    model,
                    va_t,
                    va_tp1,
                    tau=tau,
                    lambda_obs=float(args.lambda_obs),
                    lambda_usage=float(args.lambda_usage),
                    lambda_conf=float(args.lambda_conf),
                    training=False,
                )
                val_loss_f = float(val_loss.detach().cpu())
            val_losses.append(val_loss_f)

            if val_loss_f < best_val:
                best_val = val_loss_f
                best_epoch = epoch
                best_train_loss = train_loss
                best_state = {n: p.detach().cpu().clone() for n, p in model.state_dict().items()}

        if best_state is None:
            raise RuntimeError("training failed to produce a checkpoint")
        model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            logits = model.encode_logits(all_o_tensor)
            y_nn = np.asarray(torch.argmax(logits, dim=1).cpu().tolist(), dtype=np.int64)

        train_y = y_nn[:T_train]
        test_y = y_nn[T_train:]
        if len(test_y) < 3:
            raise ValueError("test split too short for NLL evaluation")

        P1 = fit_markov_order1(train_y, k)
        P2 = fit_markov_order2(train_y, k)
        nll1_nn = float(nll_order1(test_y, P1))
        nll2_nn = float(nll_order2(test_y, P2))
        gap_nn = float(nll1_nn - nll2_nn)

        counts = np.bincount(y_nn, minlength=k).astype(float)
        probs = counts / float(np.sum(counts))
        used_codes = int(np.sum(counts > 0))
        min_code_frac = float(np.min(probs))
        marginal_entropy = float(-np.sum(probs[probs > 0.0] * np.log(probs[probs > 0.0])))
        used_code_ratio = float(used_codes / float(k))
        collapsed = bool(used_codes < int(k))

        base = baseline_by_k[k]
        nll1_cluster = float(base["nll1"])
        nll2_cluster = float(base["nll2"])
        delta_nll1 = float(nll1_nn - nll1_cluster)
        delta_nll2 = float(nll2_nn - nll2_cluster)

        rows.append(
            {
                "run_id": run_id,
                "dataset_path": dataset_path.as_posix(),
                "baseline_run_dir": Path(baseline_run).as_posix(),
                "seed": int(args.seed),
                "k": int(k),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "hidden_dim": int(args.hidden_dim),
                "lr": float(args.lr),
                "best_epoch": int(best_epoch),
                "used_codes": used_codes,
                "used_code_ratio": used_code_ratio,
                "collapsed": collapsed,
                "marginal_entropy": marginal_entropy,
                "min_code_frac": min_code_frac,
                "nll1_nn": nll1_nn,
                "nll2_nn": nll2_nn,
                "gap_nn": gap_nn,
                "nll1_cluster": nll1_cluster,
                "nll2_cluster": nll2_cluster,
                "delta_nll1": delta_nll1,
                "delta_nll2": delta_nll2,
                "best_val_loss": float(best_val),
                "best_train_loss": float(best_train_loss),
            }
        )

        plt.figure(figsize=(6, 4))
        plt.plot(np.arange(len(train_losses)), train_losses, label="train")
        plt.plot(np.arange(len(val_losses)), val_losses, label="val")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title(f"Loss curve (k={k})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figs_dir / f"train_loss_k{k}.png", dpi=150)
        plt.close()

    rows = sorted(rows, key=lambda r: int(r["k"]))

    fieldnames = [
        "run_id",
        "dataset_path",
        "baseline_run_dir",
        "seed",
        "k",
        "epochs",
        "batch_size",
        "hidden_dim",
        "lr",
        "best_epoch",
        "used_codes",
        "used_code_ratio",
        "collapsed",
        "marginal_entropy",
        "min_code_frac",
        "nll1_nn",
        "nll2_nn",
        "gap_nn",
        "nll1_cluster",
        "nll2_cluster",
        "delta_nll1",
        "delta_nll2",
        "best_val_loss",
        "best_train_loss",
    ]
    with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    k_vals = np.array([int(r["k"]) for r in rows], dtype=int)
    nll1_cluster_vals = np.array([float(r["nll1_cluster"]) for r in rows], dtype=float)
    nll1_nn_vals = np.array([float(r["nll1_nn"]) for r in rows], dtype=float)
    nll2_cluster_vals = np.array([float(r["nll2_cluster"]) for r in rows], dtype=float)
    nll2_nn_vals = np.array([float(r["nll2_nn"]) for r in rows], dtype=float)

    plt.figure(figsize=(6, 4))
    plt.plot(k_vals, nll1_cluster_vals, marker="o", label="clustering")
    plt.plot(k_vals, nll1_nn_vals, marker="o", label="nn")
    plt.xlabel("k")
    plt.ylabel("held-out order-1 NLL")
    plt.title("Order-1 NLL comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figs_dir / "nll1_compare_vs_k.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(k_vals, nll2_cluster_vals, marker="o", label="clustering")
    plt.plot(k_vals, nll2_nn_vals, marker="o", label="nn")
    plt.xlabel("k")
    plt.ylabel("held-out order-2 NLL")
    plt.title("Order-2 NLL comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figs_dir / "nll2_compare_vs_k.png", dpi=150)
    plt.close()

    deltas = np.array([float(r["delta_nll1"]) for r in rows], dtype=float)
    best_idx = int(np.argmin(deltas))
    summary = {
        "dataset_path": dataset_path.as_posix(),
        "baseline_run_dir": Path(baseline_run).as_posix(),
        "k_list": [int(k) for k in k_vals],
        "improved_any_nll1": bool(np.any(deltas < 0.0)),
        "best_k_by_delta_nll1": int(rows[best_idx]["k"]),
        "best_delta_nll1": float(deltas[best_idx]),
        "rows": [
            {
                "k": int(r["k"]),
                "nll1_cluster": float(r["nll1_cluster"]),
                "nll1_nn": float(r["nll1_nn"]),
                "delta_nll1": float(r["delta_nll1"]),
                "nll2_cluster": float(r["nll2_cluster"]),
                "nll2_nn": float(r["nll2_nn"]),
                "delta_nll2": float(r["delta_nll2"]),
                "used_codes": int(r["used_codes"]),
                "used_code_ratio": float(r["used_code_ratio"]),
                "collapsed": bool(r["collapsed"]),
            }
            for r in rows
        ],
    }
    _write_json(run_dir / "summary.json", summary)

    config = {
        "run_id": run_id,
        "dataset_path": dataset_path.as_posix(),
        "baseline_run_dir": Path(baseline_run).as_posix(),
        "cli_args": vars(args),
        "loss_weights": {
            "lambda_obs": float(args.lambda_obs),
            "lambda_usage": float(args.lambda_usage),
            "lambda_conf": float(args.lambda_conf),
        },
    }
    _write_json(run_dir / "config.json", config)

    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(shlex.quote(a) for a in sys.argv),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
        "torch_version": torch.__version__,
        "runtime_seconds": float(time.perf_counter() - t0),
    }
    _write_json(run_dir / "manifest.json", manifest)

    print(str(run_dir))


if __name__ == "__main__":
    main()
