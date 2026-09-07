#!/usr/bin/env python3
"""Gate deck Modul 02: invarian sumber .tex + hasil build. Exit 0 = lulus.
Set M02_DECK_PARTIAL=1 saat deck belum lengkap (cek jumlah frame/intuisi/notes hanya WARN)."""
import os, re, subprocess, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bahasa_rules import check

ROOT = pathlib.Path(__file__).resolve().parent.parent / "slides"
TEX, PDF, LOG = ROOT / "module02_slides.tex", ROOT / "module02_slides.pdf", ROOT / "module02_slides.log"
NOTES_TEX, NOTES_PDF = ROOT / "speaker_notes_src.tex", ROOT / "speaker_notes.pdf"
PARTIAL = os.environ.get("M02_DECK_PARTIAL") == "1"
errs, warns = [], []

src = TEX.read_text(encoding="utf-8")
if not PDF.exists() or PDF.stat().st_mtime < TEX.stat().st_mtime:
    errs.append("PDF tidak ada atau lebih tua dari .tex")
if LOG.exists():
    log = LOG.read_text(encoding="utf-8", errors="ignore")
    if re.search(r"^!", log, re.M):
        errs.append("log memuat error '!'")
    n_over = len(re.findall(r"^Overfull", log, re.M))
    if n_over:
        errs.append(f"{n_over} Overfull di log")
else:
    errs.append("log tidak ada")

# Hitung hanya di badan dokumen: preambul memuat \begin{frame} di dalam makro \acttitle
# dan satu contoh \acttitle{...} di komentar, yang bukan halaman deck.
body = src.split(r"\begin{document}", 1)[-1]
n_frames = len(re.findall(r"\\begin\{frame\}", body)) + len(re.findall(r"\\acttitle\{", body))
if not 60 <= n_frames <= 80:
    (warns if PARTIAL else errs).append(f"{n_frames} frame (mau 60–80)")
n_int = len(re.findall(r"\\intuisi\{", src))
if n_int < 8:
    (warns if PARTIAL else errs).append(f"\\intuisi hanya {n_int} (mau ≥ 8)")
if len(re.findall(r"\\begin\{lstlisting\}", src)) > 2:
    errs.append("lstlisting > 2")
errs += check(src, limit=3)

def pdf_pages(p):
    out = subprocess.run(["pdfinfo", str(p)], capture_output=True, text=True).stdout
    m = re.search(r"Pages:\s+(\d+)", out)
    return int(m.group(1)) if m else -1

if NOTES_TEX.exists() and NOTES_PDF.exists():
    n_slides = len(re.findall(r"\\framebox\{Slide", NOTES_TEX.read_text(encoding="utf-8")))
    pages = pdf_pages(PDF)
    if n_slides != pages:
        errs.append(f"speaker notes {n_slides} entri vs deck {pages} halaman")
    errs += [f"notes {e}" for e in check(NOTES_TEX.read_text(encoding="utf-8"), limit=5)]
else:
    (warns if PARTIAL else errs).append("speaker_notes_src.tex / speaker_notes.pdf belum ada")

for w in warns:
    print(f"WARN {w}")
if errs:
    print("FAIL slides:" + "".join(f"\n   - {e}" for e in errs)); sys.exit(1)
print(f"PASS slides ({n_frames} frame, {n_int} intuisi)")
