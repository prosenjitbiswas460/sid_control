#!/usr/bin/env bash
# Full pipeline for one config.
#   ./scripts/run_all.sh configs/ml1m.yaml
#   PART_A_ONLY=1 ./scripts/run_all.sh configs/ml1m.yaml   # CPU-only, minutes
set -euo pipefail

CONFIG="${1:?usage: run_all.sh <config.yaml>}"
PY="${PYTHON:-python}"
PART_A_ONLY="${PART_A_ONLY:-0}"

# Tokenizers to train models for. Part A analyses all of them regardless;
# Part B only needs the system under test plus the two bounds.
TRAIN_TOKENIZERS="${TRAIN_TOKENIZERS:-}"

echo "=== [1/5] prepare data + tokenizers ==="
$PY scripts/prepare_data.py --config "$CONFIG"

echo
echo "=== [2/5] Part A: prefix purity + realisability (no GPU) ==="
$PY scripts/analyze_prefix_control.py --config "$CONFIG"
$PY scripts/eval_control_policies.py --config "$CONFIG" --levels 1

if [[ "$PART_A_ONLY" == "1" ]]; then
  $PY scripts/make_figures.py --config "$CONFIG"
  echo "Part A only; stopping."
  exit 0
fi

if [[ -z "$TRAIN_TOKENIZERS" ]]; then
  TRAIN_TOKENIZERS=$($PY - "$CONFIG" <<'EOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print(" ".join(f"{t['kind']}_{t['text_source']}" for t in cfg["tokenizers"]))
EOF
)
fi

echo
echo "=== [3/5] train: $TRAIN_TOKENIZERS ==="
for tok in $TRAIN_TOKENIZERS; do
  echo "--- training $tok ---"
  $PY scripts/train.py --config "$CONFIG" --tokenizer "$tok"
done

echo
echo "=== [4/5] Part B: control evaluation ==="
for tok in $TRAIN_TOKENIZERS; do
  echo "--- evaluating $tok ---"
  $PY scripts/run_control_eval.py --config "$CONFIG" --tokenizer "$tok"
done

echo
echo "=== [5/5] figures + summary ==="
$PY scripts/make_figures.py --config "$CONFIG"
$PY scripts/summarize.py --config "$CONFIG"
