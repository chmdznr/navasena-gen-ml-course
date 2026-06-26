#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "=== Compiling LaTeX ==="
xelatex -interaction=nonstopmode -halt-on-error nca_genl_slides.tex
xelatex -interaction=nonstopmode -halt-on-error nca_genl_slides.tex

echo "=== Done! Output: nca_genl_slides.pdf ==="
