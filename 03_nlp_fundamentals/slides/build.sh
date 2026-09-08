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
xelatex -interaction=nonstopmode -halt-on-error module03_slides.tex
xelatex -interaction=nonstopmode -halt-on-error module03_slides.tex

if [ -f speaker_notes_src.tex ]; then
    echo "=== Step 4: Compile speaker notes ==="
    xelatex -interaction=nonstopmode -halt-on-error speaker_notes_src.tex
    xelatex -interaction=nonstopmode -halt-on-error speaker_notes_src.tex
    mv -f speaker_notes_src.pdf speaker_notes.pdf
fi

echo "=== Done! Output: module03_slides.pdf + speaker_notes.pdf ==="
