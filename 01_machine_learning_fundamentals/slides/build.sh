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

echo "=== Step 2: Compile LaTeX ==="
xelatex -interaction=nonstopmode -halt-on-error module01_slides.tex
xelatex -interaction=nonstopmode -halt-on-error module01_slides.tex

if [ -f speaker_notes_src.tex ]; then
    echo "=== Step 3: Compile speaker notes ==="
    xelatex -interaction=nonstopmode -halt-on-error speaker_notes_src.tex
    xelatex -interaction=nonstopmode -halt-on-error speaker_notes_src.tex
    mv speaker_notes_src.pdf speaker_notes.pdf
fi

echo "=== Done! Output: module01_slides.pdf ==="
