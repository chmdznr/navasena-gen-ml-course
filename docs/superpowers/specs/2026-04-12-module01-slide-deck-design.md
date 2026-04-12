# Module 01 Slide Deck — Design Specification

## Overview

LaTeX Beamer presentation covering all 10 notebooks of Module 01 (Machine Learning Fundamentals) for the NCA-GENL Navasena Gen-ML course.

## Requirements

- **Format:** LaTeX Beamer, compiled with `xelatex` or `pdflatex`
- **Audience:** Dual-use — instructor presents in class, then shared to students as reference
- **Duration:** ~60 minutes presentation
- **Slides:** ~45 slides total, 4-5 per topic
- **Language:** Bahasa Indonesia with English technical terms
- **Visual style:** NVIDIA branded dark theme (#1A1A2E background, #76B900 accent)
- **Target:** SMA/diploma students (simple analogies, minimal jargon)

## Narrative Structure (Story-Driven)

### Act 1: "Apa itu Machine Learning?" (5 slides)
- Title slide: course name, NVIDIA partner branding, Navasena
- What is ML: analogy "learning from experience", Spotify/spam examples
- Supervised vs Unsupervised vs Reinforcement: diagram with icons
- Roadmap Module 01: visual timeline of 10 sessions
- Tools: Python + scikit-learn + Google Colab + NVIDIA GPU

### Act 2: "Menebak Angka" (6 slides)
- What is regression: ice cream + temperature analogy
- Linear Regression: scatter plot + best-fit line (Python-generated)
- Evaluation metrics: R², MAE, RMSE explained simply
- Time Series: predicting the future (line chart + decomposition)
- SARIMA & Seasonality: forecast visualization with confidence interval
- Act 2 summary

### Act 3: "Menebak Kategori" (10 slides)
- Classification vs Regression: comparison diagram
- Logistic Regression: probability of Yes/No, sigmoid curve
- Confusion Matrix: COVID test analogy (TP, TN, FP, FN)
- Decision Tree: flowchart-style tree (TikZ)
- KNN: "who are your neighbors?" 2D visualization
- SVM: widest margin visualization with support vectors
- Comparison: 4 model decision boundaries side-by-side (Python-generated)
- Overfitting vs Underfitting: 3-panel visual (too simple, just right, too complex)
- When to use which model: summary table
- Act 3 summary

### Act 4: "Kekuatan Tim" (8 slides)
- Analogy: 1 student vs 30 students voting
- Voting: hard vs soft voting diagram
- Bagging & Random Forest: many trees diagram
- Boosting: learning from mistakes iteratively
- XGBoost: Kaggle champion, NVIDIA GPU contribution
- Feature Importance: bar chart
- No Free Lunch theorem: no single best model
- Act 4 summary

### Act 5: "Menemukan Pola Tersembunyi" (5 slides)
- Unsupervised learning: no labels!
- K-Means: magnet & nails analogy, cluster visualization (Python-generated)
- Elbow & Silhouette: how many clusters? (dual chart)
- Hierarchical & DBSCAN: dendrogram + outlier detection
- Act 5 summary

### Act 6: "Turbo Mode: NVIDIA GPU" (8 slides)
- CPU vs GPU: restaurant analogy diagram (1 master chef vs 1000 simple cooks)
- NVIDIA RAPIDS cuML: change import, get 10-50x speedup
- Code comparison: sklearn vs cuML side-by-side (identical API)
- XGBoost GPU: one-line change (`device="cuda"`)
- Benchmark results: speedup bar chart (Python-generated)
- ONNX & deployment pipeline: training → export → serve diagram
- NVIDIA ecosystem table: cuML, cuDF, TensorRT, NeMo, Triton
- Act 6 summary

### Closing (3 slides)
- Full module summary: table of 10 topics with type (supervised/unsupervised)
- What's next: Module 2 Deep Learning roadmap timeline
- Thank you + contact

## Visual Style

### Colors
| Element | Color | Hex |
|---------|-------|-----|
| Background | Dark navy | #1A1A2E |
| Primary text | White | #FFFFFF |
| Accent/highlight | NVIDIA Green | #76B900 |
| Secondary accent | Light green | #A3D944 |
| Card/box background | Dark gray | #2D2D44 |
| Warning | Orange | #FF6F00 |
| Error/negative | Red | #EF5350 |

### Typography
- Headings: Sans-serif bold (Fira Sans if available, otherwise Beamer default)
- Body: Sans-serif regular
- Code: Monospace (Fira Code / Source Code Pro / Beamer default mono)

### Layout Conventions
- Each slide: title at top + NVIDIA green accent bar
- Footer: slide number + "NCA-GENL | Machine Learning Fundamentals"
- Each "Act" starts with a transition slide (full-screen title + key question)
- Python-generated visualizations included as PDF/PNG

## Visualizations to Generate

Priority order: Python > TikZ > Mermaid

### Python-generated (saved as PDF/PNG):
1. `viz_regression_line.py` — Scatter + regression line for Act 2
2. `viz_decision_boundaries.py` — 4-model comparison for Act 3
3. `viz_overfitting_panels.py` — Underfitting/optimal/overfitting 3-panel for Act 3
4. `viz_ensemble_comparison.py` — RF vs Boosting boundary for Act 4
5. `viz_clusters.py` — K-Means cluster visualization for Act 5
6. `viz_gpu_speedup.py` — CPU vs GPU benchmark bar chart for Act 6

### TikZ (inline in LaTeX):
1. Decision tree flowchart (Act 3)
2. CPU vs GPU analogy diagram (Act 6)
3. Deployment pipeline diagram (Act 6)
4. Course roadmap timeline (Act 1 + Closing)

### Mermaid → PNG (via mmdc):
1. ML types diagram: Supervised/Unsupervised/Reinforcement (Act 1)
2. Boosting iterative process (Act 4)

## File Structure

```
01_machine_learning_fundamentals/
  slides/
    module01_slides.tex          # Main Beamer file
    figures/                     # Generated visualizations
      viz_regression_line.pdf
      viz_decision_boundaries.pdf
      viz_overfitting_panels.pdf
      viz_ensemble_comparison.pdf
      viz_clusters.pdf
      viz_gpu_speedup.pdf
      mermaid_ml_types.png
      mermaid_boosting.png
    build.sh                     # Script to generate figures + compile
    module01_slides.pdf          # Output (gitignored)
```

## Build Process

```bash
cd 01_machine_learning_fundamentals/slides
./build.sh  # Generates Python figures, Mermaid PNGs, then compiles LaTeX
```

## Code Snippets in Slides

Minimal — only at key moments:
1. Act 6: sklearn vs cuML import comparison (3 lines each)
2. Act 6: XGBoost `device="cpu"` vs `device="cuda"` (1-line diff)

Use `listings` or `minted` package with dark theme syntax highlighting matching the slide background.

## Constraints

- Must compile on TeX Live 2026 (Homebrew)
- All Python figure scripts must run without GPU (generate sample/synthetic data)
- Mermaid diagrams via `/opt/homebrew/bin/mmdc` with scale 3 for high res
- Final PDF should be <20MB (optimize images)
