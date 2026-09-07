"""Aturan bahasa Modul 02 (spec §2). Dipakai validate_notebooks.py, validate_slides.py,
validate_quiz.py, dan validate_cheatsheet.py."""
import re

FORBIDDEN = [
    r"(?<!SD-)\bTurbo\b", r"\bbersinar\b", r"titik manis", r"\bharmonis\b", r"\bsekedar\b", r"\bmenghapal\b",
    r"(?i)data latihan", r"(?i)data ujian", r"kolom ikut dilatih",
    r"\bkalian\b", r"\bAnda\b", r"10-50x", r"10–50x",
    r"Module 0", r"10-100x", r"10–100x", r"6x lebih cepat",
    r"5-10x", r"5–10x", r"2-5x lebih cepat", r"2–5x lebih cepat",
    r"kotak-kotak", r"(?i)scaling wajib", r"(?i)senjata rahasia", r"Perbandingan Besar",
]
LIMITED = [r"Mari kita", r"Perhatikan bahwa", r"Sekarang, mari"]


def check(text, limit):
    """Kembalikan daftar pesan error (kosong = lulus)."""
    errs = [f"forbidden: {p}" for p in FORBIDDEN if re.search(p, text)]
    n = sum(len(re.findall(p, text)) for p in LIMITED)
    if n > limit:
        errs.append(f"limited openers {n} > {limit}")
    return errs
