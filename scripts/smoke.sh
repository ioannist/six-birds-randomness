#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

capture_run_snapshot() {
  local base="$1"
  local snapshot="$2"
  if [[ -d "$base" ]]; then
    find "$base" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | LC_ALL=C sort > "$snapshot"
  else
    : > "$snapshot"
  fi
}

detect_new_run_dir() {
  local base="$1"
  local before_snapshot="$2"
  local after_snapshot
  local new_dir
  after_snapshot="$(mktemp)"
  capture_run_snapshot "$base" "$after_snapshot"
  new_dir="$(comm -13 "$before_snapshot" "$after_snapshot" | tail -n 1 || true)"
  rm -f "$after_snapshot"
  if [[ -z "$new_dir" ]]; then
    echo "ERROR: no new run directory detected under $base" >&2
    return 1
  fi
  printf '%s/%s\n' "$base" "$new_dir"
}

start_ts="$(date +%s)"

# 1) Tiny Markov bench + analysis
markov_base="results/markov_bench"
markov_before="$(mktemp)"
capture_run_snapshot "$markov_base" "$markov_before"
python experiments/markov_bench/run_sweep.py --config configs/markov_bench_smoke.json
markov_run="$(detect_new_run_dir "$markov_base" "$markov_before")"
rm -f "$markov_before"
python experiments/markov_bench/analyze.py --run_dir "$markov_run"
echo "markov_bench run_dir=$markov_run"

# 2) Tiny budget curves
budget_base="results/budget_curves"
budget_before="$(mktemp)"
capture_run_snapshot "$budget_base" "$budget_before"
python experiments/budget_curves/run_budget_curves.py \
  --preset hidden_types_strong \
  --max_order 3 \
  --T 4000 \
  --tau 1 \
  --seed 20260305
budget_run="$(detect_new_run_dir "$budget_base" "$budget_before")"
rm -f "$budget_before"
echo "budget_curves run_dir=$budget_run"

# 3) Tiny hashing toy
hash_base="results/hashing_toy"
hash_before="$(mktemp)"
capture_run_snapshot "$hash_base" "$hash_before"
python experiments/hashing_toy/run_hashing_toy.py \
  --n_bits 8,12 \
  --q_list 1,16,64,256 \
  --trials 100 \
  --randtest_samples 1000 \
  --randtest_rep_n_bits 12 \
  --seed 20260306
hash_run="$(detect_new_run_dir "$hash_base" "$hash_before")"
rm -f "$hash_before"
echo "hashing_toy run_dir=$hash_run"

# 4) Tiny representation packaging + clustering
rep_base="results/rep_packaging"
rep_before="$(mktemp)"
capture_run_snapshot "$rep_base" "$rep_before"
python experiments/rep_packaging/make_dataset.py \
  --preset hidden_types_strong \
  --obs gaussian \
  --d 4 \
  --T 1500 \
  --tau 1 \
  --burn_in 100 \
  --seed 20260305
rep_run="$(detect_new_run_dir "$rep_base" "$rep_before")"
rm -f "$rep_before"
dataset_path="$rep_run/dataset.npz"
echo "rep_packaging_dataset run_dir=$rep_run"

cluster_base="results/rep_packaging_clustering"
cluster_before="$(mktemp)"
capture_run_snapshot "$cluster_base" "$cluster_before"
python experiments/rep_packaging/run_clustering_packaging.py \
  --dataset "$dataset_path" \
  --k_list 2,4 \
  --n_init 2 \
  --max_iter 30 \
  --seed 20260306
cluster_run="$(detect_new_run_dir "$cluster_base" "$cluster_before")"
rm -f "$cluster_before"
echo "rep_packaging_clustering run_dir=$cluster_run"

end_ts="$(date +%s)"
runtime_seconds="$((end_ts - start_ts))"

echo "SMOKE_OK runtime_seconds=$runtime_seconds"
