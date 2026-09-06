#!/usr/bin/env python3
"""Helper kecil untuk mengedit .ipynb secara programatik. Dipakai skrip fix per notebook."""
import json, sys

def load(path):
    return json.load(open(path, encoding="utf-8"))

def _lines(text):
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

def _cell(kind, text):
    c = {"cell_type": kind, "metadata": {}, "source": _lines(text)}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c

def src(nb, i):
    return "".join(nb["cells"][i]["source"])

def find(nb, substr, start=0):
    for i in range(start, len(nb["cells"])):
        if substr in src(nb, i):
            return i
    raise KeyError(f"tidak ada sel berisi: {substr!r}")

def set_src(nb, i, text):
    nb["cells"][i]["source"] = _lines(text)

def replace(nb, i, old, new, count=1):
    s = src(nb, i)
    assert old in s, f"sel {i}: {old!r} tidak ditemukan"
    set_src(nb, i, s.replace(old, new, count))

def insert_md(nb, i, text):
    nb["cells"].insert(i, _cell("markdown", text))

def insert_code(nb, i, text):
    nb["cells"].insert(i, _cell("code", text))

def delete(nb, i):
    del nb["cells"][i]

def clean(nb):
    nb["metadata"].pop("widgets", None)
    for c in nb["cells"]:
        c["metadata"] = {k: v for k, v in c.get("metadata", {}).items() if k in ("id",)}
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None
        if isinstance(c["source"], str):
            c["source"] = _lines(c["source"])

def save(nb, path):
    clean(nb)
    json.dump(nb, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(path, "a").write("\n")

if __name__ == "__main__":
    nb = load(sys.argv[1])
    for i, c in enumerate(nb["cells"]):
        print(f"[{i}] {c['cell_type']:8} {src(nb, i)[:90]!r}")
