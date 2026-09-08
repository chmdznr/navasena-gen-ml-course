#!/usr/bin/env python3
"""Gate struktur + konten 2 notebook Modul 03. Exit 0 = lulus, 1 = ada yang gagal."""
import json, re, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bahasa_rules import check

ROOT = pathlib.Path(__file__).resolve().parent.parent
COMMON_MARKERS = ["🏋️ Latihan", "dari 2", "Tujuan Pembelajaran", "Estimasi sesi"]
# Nama file lama (EN/ID terpisah) tidak boleh dirujuk lagi; klaim speedup tanpa
# konteks dan sisa bahasa Inggris di judul section juga dilarang.
COMMON_FORBIDDEN = [
    "_id.ipynb", "_en.ipynb", "notebook bahasa Indonesia", "notebook English",
    "Spam vs Ham", "Module 0", "10-100x", "10–100x", "files.upload",
]
REGISTRY = {
    "01_nlp_fundamentals.ipynb": [
        "nlp_id", "STOPWORD_ID", "dok_train", "fitur_kata",   # fitur dihitung dari train saja
        "baseline", "TfidfVectorizer", "SentenceTransformer",
        "AutoTokenizer", "w11wo/indonesian-roberta-base-sentiment-classifier",
        "en_core_web_sm", "kebocoran data",
    ],
    "02_nlp_on_steroids.ipynb": [
        "cudf", "cuml", "nvidia-smi", "minhash", "ngrams_tokenize",
        "cudf.pandas", "cuml.accel", "warm-up", "NeMo Curator",
    ],
}
fail = False
for name, markers in REGISTRY.items():
    errs, path = [], ROOT / name
    if not path.exists():
        print(f"FAIL {name}: file tidak ada"); fail = True; continue
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"FAIL {name}: bukan JSON valid ({e})"); fail = True; continue
    if "widgets" in nb.get("metadata", {}):
        errs.append("metadata.widgets ada")
    src_all = "\n".join("".join(c["source"]) for c in nb["cells"])
    for c in nb["cells"]:
        if c["cell_type"] == "code" and c.get("outputs"):
            errs.append("ada outputs tersimpan"); break
        if not isinstance(c["source"], list):
            errs.append("source bukan list"); break
    for m in markers + COMMON_MARKERS:
        if m not in src_all:
            errs.append(f"marker hilang: {m}")
    for f in COMMON_FORBIDDEN:
        if f in src_all:
            errs.append(f"forbidden: {f}")
    # Setiap pip install harus punya batas versi — pin adalah invarian batch 6.
    for baris in re.findall(r"^\s*!pip install .*$", src_all, re.M):
        if "{" not in baris and not re.search(r"[<>=]=?\d", baris):
            errs.append(f"pip tanpa pin: {baris.strip()[:70]}")
    errs += check(src_all, limit=1)
    # Latihan harus tepat sebelum Ringkasan (atau di akhir notebook).
    idx_l = [i for i, c in enumerate(nb["cells"]) if "🏋️ Latihan" in "".join(c["source"])]
    idx_r = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "markdown" and any(
        l.lstrip().startswith("#") and "Ringkasan" in l for l in "".join(c["source"]).splitlines())]
    if idx_l and idx_r and not (idx_l[0] < idx_r[-1] <= idx_l[0] + 2):
        errs.append(f"Latihan ({idx_l[0]}) tidak tepat sebelum Ringkasan ({idx_r[-1]})")
    status = "FAIL" if errs else "PASS"
    fail |= bool(errs)
    print(f"{status} {name} ({len(nb['cells'])} sel)" + "".join(f"\n   - {e}" for e in errs))
sys.exit(1 if fail else 0)
