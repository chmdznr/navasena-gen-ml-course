"""Aturan bahasa Modul 01 (spec §2). Dipakai validate_notebooks.py dan validate_slides.py."""
import re

FORBIDDEN = [
    r"\bTurbo\b", r"\bbersinar\b", r"titik manis", r"\bharmonis\b", r"\bsekedar\b", r"\bmenghapal\b",
    r"(?i)data latihan", r"(?i)data ujian", r"kolom ikut dilatih",
    r"\bkalian\b", r"\bAnda\b", r"10-50x", r"10–50x",
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
