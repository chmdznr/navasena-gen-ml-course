#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "=== Step 1: Generate Python figures ==="
PYTHON=${PYTHON:-python3}
for script in figures/gen_*.py; do
    if [ -f "$script" ]; then
        echo "  Running $script..."
        $PYTHON "$script"
    fi
done

echo "=== Step 2: Generate Mermaid diagrams ==="
MMDC=${MMDC:-/opt/homebrew/bin/mmdc}
for mmd in figures/*.mmd; do
    if [ -f "$mmd" ]; then
        out="${mmd%.mmd}.png"
        echo "  Converting $mmd → $out..."
        $MMDC -i "$mmd" -o "$out" -s 3 -b transparent 2>/dev/null
    fi
done

echo "=== Step 3: Compile LaTeX ==="
xelatex -interaction=nonstopmode -halt-on-error module01_slides.tex
xelatex -interaction=nonstopmode -halt-on-error module01_slides.tex

echo "=== Done! Output: module01_slides.pdf ==="
