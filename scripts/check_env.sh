#!/usr/bin/env bash
# Quick sanity check before a long server run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

echo "repo root: $ROOT"
echo "python:    $(command -v python3 || command -v python)"
python3 - <<'PY'
import sys
print("PYTHONPATH ok:", sys.path[0] if sys.path else "(empty)")
import sidctl
print("sidctl:", sidctl.__file__)
from sidctl.data import load_corpus
from sidctl.sid import build_tokenizer
from sidctl.models import TigerGR
print("imports: OK")
try:
    import torch
    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch:", e)
PY
echo "check_env: passed"
