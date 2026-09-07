#!/usr/bin/env python3
"""Gate struktur + konten 10 notebook Modul 01. Exit 0 = lulus, 1 = ada yang gagal.
Usage: python validate_notebooks.py"""
import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bahasa_rules import check

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOADER = "raw.githubusercontent.com"
COMMON_MARKERS = ["🏋️ Latihan", "dari 10"]
COMMON_FORBIDDEN = ["files.upload", "algorithm=", "SMOTE", "onnx", "cuml-cu12"]
REGISTRY = {
    "01_linear_regression.ipynb": [LOADER, "y_train.mean()"],
    "02_logistic_regression.ipynb": [LOADER, "class_weight", "duration"],
    "03_decision_tree_algorithm.ipynb": [LOADER, "cross_val_score"],
    "04_knn_classifier.ipynb": [LOADER, "cross_val_score"],
    "05_support_vector_machines.ipynb": [LOADER, "decision_function", "class_weight"],
    "06_ensemble_learning_techniques_voting_bagging_boosting_stacking.ipynb": [LOADER, "make_moons"],
    "07_xgboost.ipynb": [LOADER, "reindex", "evals_result"],
    "08_kmeans_hierarchial_clustering.ipynb": [LOADER, "log1p", "crosstab"],
    "09_time_series_analysis.ipynb": [LOADER, "naive"],
    "10_nvidia_gpu_acceleration.ipynb": ["def lapor(", "CUML_AVAILABLE"],
}

fail = False
for name, markers in REGISTRY.items():
    errs = []
    path = ROOT / name
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
    errs += check(src_all, limit=1)
    # Latihan harus tepat sebelum Ringkasan
    idx_l = [i for i, c in enumerate(nb["cells"]) if "🏋️ Latihan" in "".join(c["source"])]
    idx_r = [i for i, c in enumerate(nb["cells"]) if "📝 Ringkasan" in "".join(c["source"])]
    if idx_l and idx_r and not (idx_l[0] < idx_r[0] <= idx_l[0] + 2):
        errs.append(f"Latihan ({idx_l[0]}) tidak tepat sebelum Ringkasan ({idx_r[0]})")
    status = "FAIL" if errs else "PASS"
    fail |= bool(errs)
    print(f"{status} {name}" + ("".join(f"\n   - {e}" for e in errs)))
sys.exit(1 if fail else 0)
