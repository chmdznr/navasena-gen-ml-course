# Handoff: Navasena Bootcamp Revisi — Session Jun 1, 2026 → Next Session

> **Tujuan**: Context untuk session berikutnya supaya bisa langsung lanjut tanpa harus recall dari nol.
> **Status saat handoff**: Working tree bersih, 4 commit baru ahead of origin/master.

## State Saat Ini

Branch: `master` (lokal, ahead 4 dari `origin/master`).
Working tree: clean — tidak ada uncommitted changes.

Recent commits (Jun 1, 2026):
```
2822ad3  feat(slides): add Module 03 NLP Fundamentals slide deck (20 slides)
5b58528  feat(module03): add theory markdown cells to EN+ID NLP notebooks
64b1026  fix(slides): polish Module 01 outstanding items
2cf48ca  chore: ignore macOS .DS_Store and ensure LaTeX build artifacts are ignored
```

Module terakhir yang disentuh: **Module 04/05/06 (theory cells)** — lihat update di bawah. Status modul:
- Module 01 (ML) — slide deck 48 halaman, notebook OK
- Module 02 (Deep Learning) — 5 notebook OK, slide deck belum ada
- Module 03 (NLP) — notebook + theory cells + slide deck 20 hal OK
- Module 04 (LLM) — 1 notebook **+ theory cells (8 md)** ✅, slide deck belum ada
- Module 05 (RAG) — 1 notebook **+ intro & theory cells (8 md)** ✅, slide deck belum ada
- Module 06 (NVIDIA Ecosystem) — 3 notebook **+ theory cells (4/14/7 md)** ✅, slide deck belum ada

### Update Jun 1 (sesi lanjutan) — theory cells Module 04/05/06 SELESAI
- Pushed 5 commit sebelumnya ke origin.
- Design spec: `docs/superpowers/specs/2026-06-01-module04-05-06-theory-cells-design.md` (commit `4fc502e`).
- Theory cells dibuat via workflow (draft → review akurasi ∥ gaya → sintesis), lalu di-apply dengan guard hash code-cell (semua code cell BYTE-IDENTICAL, markdown-only). Commit: `9c6a0ab` (m04), `44d0ebe` (m05), `2ae19e2` (m06).
- Catatan branding: penutup Module 06 RAG di-netralkan dari "course NCA-GENL" → "rangkaian course ini" (selaras de-branding slide). Kalau ternyata NCA-GENL memang dipertahankan di notebook, ini bisa di-revert.
- **Belum di-push** — commit theory cells masih lokal.

## Pola Kerja yang Sudah Terbentuk

### 1. Build artifacts & gitignore
- `.gitignore` sudah cover: `*.aux *.log *.nav *.out *.snm *.toc *.vrb` (LaTeX), `.DS_Store`, `__pycache__/`
- **Selalu** jalankan `find . -maxdepth 1 \( -name "*.aux" -o -name "*.log" \) -delete` sebelum `xelatex` (zsh nullglob error kalau pakai `*.aux` mentah)
- **Selalu** 2-pass `xelatex` untuk resolve cross-references

### 2. Python env untuk figure generators
- Default `/opt/homebrew/bin/python3` TIDAK punya matplotlib/sklearn
- Gunakan: `PYTHON=/Users/chmdznr/work/kemendag/sip/docling/bin/python`
- Docling env sudah ada matplotlib 3.10.8, baru di-install `scikit-learn` 1.8.0 via `uv pip install`
- `uv` ada di `/Users/chmdznr/.local/bin/uv`
- `mmdc` (Mermaid CLI) ada di `/opt/homebrew/bin/mmdc`
  - Pattern render: `mmdc -i input.mmd -o output.png -s 3 -b transparent` (scale 3 untuk high-res)

### 3. Pattern slide deck (Module 01 & 03 sudah jadi template)
- Lokasi: `XX_module_name/slides/moduleXX_slides.tex`
- Theme: `aspectratio=169` (16:9) Beamer dengan custom Navasena dark
- Color palette: `nvbg #1A1A2E`, `nvcard #2D2D44`, `nvgreen #76B900`, `nvlightgreen #A3D944`, `nvgray #AAAACC`, `nvred #EF5350`, `nvorange #FF6F00`, `nvblue #42A5F5`
- Custom command `\acttitle{N}{Title}{Subtitle}` untuk section divider
- Footer: "Module Name | page X/Y" (NO NCA-GENL, NO NVIDIA)
- Build script `build.sh` chains: gen_*.py → mmdc → 2x xelatex
- Figures di `slides/figures/`: `gen_*.py` + `*.pdf` (matplotlib) + `*.mmd` + `*.png` (Mermaid)

### 4. Pattern notebook improvements
- Selalu backup ke `/tmp/` sebelum edit (`cp notebook.ipynb /tmp/backup.ipynb`)
- Edit via Python script: load JSON, modify `cells` list, write back
- Insert markdown cells: `insert_at(idx, cell)` — insert REVERSE order (highest idx first) supaya indices sebelumnya tetap valid
- Markdown cell helper:
  ```python
  def md(source):
      return {
          "cell_type": "markdown",
          "metadata": {},
          "source": source if isinstance(source, list) else source.split('\n')
      }
  ```
- **WARNING**: JSON reformat oleh `json.dump` akan add banyak insertions/deletions noise di git diff. Normal, jangan kaget. Functional content tetap sama.

### 5. LaTeX layout fixes
- Overfull `\hbox` di kolom 0.48-0.5\textwidth: pakai `\parbox{\dimexpr\linewidth-8pt}` atau `\resizebox{\linewidth}{!}{...}` untuk wrap TikZ
- Overfull `\vbox` (slide content > text height): kurangi `vspace`, smaller font, smaller TikZ scale, atau split ke 2 slide
- Table besar (10+ rows): `\arraystretch` 1.05, `\tiny` font, `p{0.30\linewidth}` columns
- `aspectratio=169` artinya `\linewidth` di kolom 0.48 lebih sempit dari 4:3 — penting untuk diingat
- `latex-overfull-guard.sh` hook aktif: block `>= 2pt` overfull, block semua `vbox` overflow

## Outstanding / Bisa Dilanjutkan

### Prioritas tinggi (recommended next)
1. **Push 4 commit ke origin** — `git push` (branch ahead 4, no conflicts expected)
2. **Module 04 (LLM) slide deck** — pakai pattern Module 03 sebagai template
   - 1 notebook: `04_llm/01_llm_basics.ipynb` (16 cells, 1 markdown)
   - Topik: tokenization, transformer basics, HuggingFace pipeline
3. **Module 05 (RAG) slide deck** — sama, dari `05_rag/01_rag_fundamentals.ipynb` (9 cells, 1 markdown)
4. **Module 06 (NVIDIA Ecosystem) slide deck** — 3 notebook, mungkin split ke 3 acts:
   - Basic: `01_nvidia_ecosystem_basic.ipynb` (3 cells, 0 markdown)
   - NeMo: `02_nvidia_nemo_demo.ipynb` (28 cells, 13 markdown)
   - RAG: `03_nvidia_rag.ipynb` (10 cells, 3 markdown)

### Prioritas sedang
5. **Module 02 (Deep Learning) slide deck** — 5 notebook, sangat besar
6. ~~**Notebook theory cells untuk Module 04/05/06**~~ ✅ SELESAI (Jun 1, lihat update di atas)
7. **Rebrand lama**: cek apakah `04_llm/`, `05_rag/`, `06_nvidia_ecosystem/` masih punya NCA-GENL / NVIDIA branding di notebook atau file PDF pendukung

### Bisa di-defer
- Cheat sheet PDF untuk Modul 03-06 (Modul 01 & 02 sudah punya)
- Mermaid diagram untuk pipeline baru (RAG pipeline, LLM inference flow)
- Tutorial video script
- Quiz/assignment untuk tiap modul

## Files yang Mungkin Ingin Dibaca di Next Session

- `CLAUDE.md` (root) — project overview
- `01_machine_learning_fundamentals/slides/module01_slides.tex` — template LaTeX (paling mature)
- `03_nlp_fundamentals/slides/module03_slides.tex` — template lebih baru
- `03_nlp_fundamentals/01_nlp_fundamentals_en.ipynb` & `_id.ipynb` — contoh notebook dengan theory cells

## Command Cheat Sheet

```bash
# Lihat status
git status && git log --oneline -5

# Recompile slide deck
cd XX_module/slides
find . -maxdepth 1 -name "*.aux" -delete -o -name "*.log" -delete
xelatex -interaction=nonstopmode -halt-on-error moduleXX_slides.tex
xelatex -interaction=nonstopmode -halt-on-error moduleXX_slides.tex

# Regenerate figures
PYTHON=/Users/chmdznr/work/kemendag/sip/docling/bin/python
$PYTHON figures/gen_*.py
/opt/homebrew/bin/mmdc -i figures/foo.mmd -o figures/foo.png -s 3 -b transparent

# Build end-to-end
cd XX_module/slides && bash build.sh

# Push saat sudah siap
git push
```

## Caveats Penting

- **Slide 3b di Module 01** ("Time Series = Regression + Waktu") adalah slide tambahan baru — feedback client sebelumnya (Apr 12) mempertanyakan Time Series placement di Act 2. Slide ini menjelaskan koherensi, tapi kalau client masih tidak setuju, alternatif: pindahkan Time Series ke Act 5 (clustering) atau buat Act baru "Act 2.5: Time Series"
- **Module 05 RAG** notebook hanya 9 cells, 1 markdown — mungkin perlu development lebih substantial sebelum slide deck (bukan sekadar reformat)
- **Module 06 NVIDIA** punya NeMo demo yang berat GPU — perlu diingat slide deck-nya hanya konseptual, bukan hands-on
- **Time zones**: commit message menggunakan timestamp lokal (WIB, GMT+7). Memory `claude-mem` juga pakai timestamp WIB.
