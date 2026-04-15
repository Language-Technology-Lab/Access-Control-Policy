#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_v1.sh — batch relation-extraction on all 6 dataset subsets
#
# Runs access_control_run.py for each subset, appends one row per run to
# performance_results.csv, and prints the final table.
#
# Usage:
#   bash run_v1.sh
#   nohup bash run_v1.sh > run_v1.log 2>&1 &
#
# Prerequisites:
#   - OPENAI_API_KEY set in .env or the environment
#   - datasets/ populated with SubgraphsWithTriplesImages & JSON dirs
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_IN="$BASE_DIR/datasets/SubgraphsWithTriplesImages"
BASE_OUT="$BASE_DIR/experiments/relation_extraction"
BASE_GT="$BASE_DIR/datasets/SubgraphsWithTriplesJSON"
OUTPUT_CSV="$BASE_DIR/performance_results.csv"

# ── CSV header (pipe-separated) ─────────────────────────────────────────────
PERF_HEADER="model name | dataset name | legend or not | entity_TP | entity_FP | entity_FN | entity_P_micro | entity_R_micro | entity_F1_micro | entity_P_macro | entity_R_macro | entity_F1_macro | rel_TP | rel_TN | rel_FP | rel_FN | rel_P_micro | rel_R_micro | rel_F1_micro | rel_P_macro | rel_R_macro | rel_F1_macro | overall_accuracy"
echo "$PERF_HEADER" > "$OUTPUT_CSV"

# ── Ground-truth mapping ─────────────────────────────────────────────────────
gt_dir_for() {
  local name="$1"
  case "$name" in
    subgraphs_01|subgraphs_01_wo_legend)   echo "${BASE_GT}/subgraphs_01"  ;;
    subgraphs_001|subgraphs_001_wo_legend) echo "${BASE_GT}/subgraphs_001" ;;
    subgraphs_06|subgraphs_06_wo_legend)   echo "${BASE_GT}/subgraphs_06"  ;;
    *) echo "${BASE_GT}/${name}" ;;
  esac
}

# ── Single-run helper ─────────────────────────────────────────────────────────
run_one() {
  local name="$1"
  local no_legend="${2:-}"
  local gt_dir
  gt_dir="$(gt_dir_for "$name")"
  local extra=()
  [[ -n "$no_legend" ]] && extra+=(--no_legend)

  echo "▶  Running: $name ${no_legend:+(no legend)}"
  python "$BASE_DIR/access_control_run.py" \
    --input  "${BASE_IN}/${name}" \
    --output "${BASE_OUT}/${name}" \
    --gt_input "$gt_dir" \
    --method relation_extraction \
    --model gpt-5-mini --image_detail low --few_shot zero \
    "${extra[@]}"
}

# ── Execute all subsets ──────────────────────────────────────────────────────
run_one "subgraphs_01"
run_one "subgraphs_001"
run_one "subgraphs_01_wo_legend"  "1"
run_one "subgraphs_001_wo_legend" "1"
run_one "subgraphs_06"
run_one "subgraphs_06_wo_legend"  "1"

# ── Print results ────────────────────────────────────────────────────────────
echo ""
echo "────────────────────────────────────────────────────────────────"
echo "All runs completed.  Performance table:"
echo "────────────────────────────────────────────────────────────────"
column -s '|' -t < "$OUTPUT_CSV"
echo "────────────────────────────────────────────────────────────────"
echo "Full CSV: $OUTPUT_CSV"
