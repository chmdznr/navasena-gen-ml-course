#!/bin/zsh
# Eksekusi notebook penuh; output ke scratchpad, tidak ke repo.
# Env: NB_PYTHON (python with nbconvert + kernel m01venv), NB_OUT_DIR (output dir)
set -e
PY=${NB_PYTHON:-python3}
OUT=${NB_OUT_DIR:-/tmp/m01-executed}
mkdir -p "$OUT"
cd "$(dirname "$0")/.."
$PY -m jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.kernel_name=m01venv --ExecutePreprocessor.timeout=1800 \
  --output-dir "$OUT" "$1"
echo "OK: $1"
