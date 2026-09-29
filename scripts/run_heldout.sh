#!/usr/bin/env bash
# Frozen-SID held-out constraint (CPU). Does not rebuild corpus or tokenizers.
#   ./scripts/run_heldout.sh
#   ./scripts/run_heldout.sh amazon_beauty ml1m
set -eu
PY="${PYTHON:-python}"
if [[ $# -gt 0 ]]; then
  datasets=("$@")
else
  datasets=(amazon_beauty amazon_sports ml1m)
fi

echo "=== tests $(date) ==="
$PY -m pytest tests/test_smoke.py -q

for c in "${datasets[@]}"; do
  echo "=== heldout attrs $c $(date) ==="
  $PY scripts/build_heldout_attributes.py --config "configs/${c}.yaml"
  echo "=== heldout eval $c $(date) ==="
  $PY scripts/eval_heldout_constraint.py --config "configs/${c}.yaml"
done
echo "ALL DONE $(date)"
