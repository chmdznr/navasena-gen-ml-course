#!/bin/zsh
# Eksekusi notebook penuh; output ke scratchpad, tidak ke repo.
set -e
SCRATCH=/private/tmp/claude-501/-Users-chmdznr-work-navasena-navasena-gen-ml-course/c45d5983-d2e4-4e18-9f76-56fd52059487/scratchpad
mkdir -p $SCRATCH/executed
cd "$(dirname "$0")/.."
$SCRATCH/venv/bin/python -m jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.kernel_name=m01venv --ExecutePreprocessor.timeout=1800 \
  --output-dir $SCRATCH/executed "$1"
echo "OK: $1"
