#!/usr/bin/env bash
# Run relation_extraction on all 6 datasets (gpt-5-mini, low image_detail, zero few_shot)
# Each run appends one row to performance_results.csv (via access_control_new.py).
# At the end we print the full CSV table.
# Usage: bash run_v1.sh   or   ./run_v1.sh
# Nohup (background, survives logout): nohup bash run_v1.sh > run_v1.log 2>&1 &

set -e

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"

BASE_IN="$BASE_DIR/datasets/SubgraphsWithTriplesImages"
BASE_OUT="$BASE_DIR/experiments/relation_extraction"
BASE_GT="$BASE_DIR/datasets/SubgraphsWithTriplesJSON"
OUTPUT_CSV="$BASE_DIR/performance_results.csv"

# Same header as run_evaluation.sh / performance_results.csv (pipe-separated)
PERF_HEADER="model name | dataset name | legend or not | entity_TP | entity_FP | entity_FN | entity_P_micro | entity_R_micro | entity_F1_micro | entity_P_macro | entity_R_macro | entity_F1_macro | rel_TP | rel_TN | rel_FP | rel_FN | rel_P_micro | rel_R_micro | rel_F1_micro | rel_P_macro | rel_R_macro | rel_F1_macro | overall_accuracy"
echo "$PERF_HEADER" > "$OUTPUT_CSV"

# Map dataset name to ground-truth JSON dir (same GT for with/without legend)
gt_dir_for() {
  local name="$1"
  case "$name" in
    subgraphs_01|subgraphs_01_wo_legend) echo "${BASE_GT}/subgraphs_01" ;;
    subgraphs_001|subgraphs_001_wo_legend) echo "${BASE_GT}/subgraphs_001" ;;
    subgraphs_06|subgraphs_06_wo_legend) echo "${BASE_GT}/subgraphs_06" ;;
    *) echo "${BASE_GT}/${name}" ;;  # fallback
  esac
}

run_one() {
  local name="$1"
  local no_legend="${2:-}"
  local gt_dir
  gt_dir="$(gt_dir_for "$name")"
  local extra=()
  [[ -n "$no_legend" ]] && extra+=(--no_legend)
  python "$BASE_DIR/access_control_new.py" \
    --input "${BASE_IN}/${name}" \
    --output "${BASE_OUT}/${name}" \
    --gt_input "$gt_dir" \
    --method relation_extraction \
    --model gpt-5-mini --image_detail low --few_shot zero \
    "${extra[@]}"
}

run_one "subgraphs_01"
run_one "subgraphs_001"
run_one "subgraphs_01_wo_legend"  "1"
run_one "subgraphs_001_wo_legend" "1"
run_one "subgraphs_06"
run_one "subgraphs_06_wo_legend" "1"

echo ""
echo "----------------------------------------------------------------"
echo "🏁 All runs completed. Performance table (performance_results.csv):"
echo "----------------------------------------------------------------"
column -s '|' -t < "$OUTPUT_CSV"
echo "----------------------------------------------------------------"
echo "📄 Full CSV: $OUTPUT_CSV"
