#!/usr/bin/env python3
"""Gate struktur + konten 6 notebook Modul 02. Exit 0 = lulus, 1 = ada yang gagal."""
import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bahasa_rules import check

ROOT = pathlib.Path(__file__).resolve().parent.parent
COMMON_MARKERS = ["🏋️ Latihan", "dari 6", "Input(shape", "set_seed", "QUICK"]
COMMON_FORBIDDEN = ["input_shape=", "input_length=", "ImageDataGenerator", "pip install -q tensorflow",
                    "Modul 06", "Module 0", "10-100x", "10–100x", "6x lebih cepat", "files.upload"]
REGISTRY = {
    "01_building_first_neural_networks_tensorflow.ipynb": ["validation", "confusion_matrix", "Baseline"],
    "02_activation_functions_in_tensorflow.ipynb": ["val_accuracy", "softmax"],
    "03_cnn_transfer_learning_cifar10.ipynb": ["RandomFlip", "ResNet50", "CIFAR"],
    "04_rnn_lstm.ipynb": ["Embedding(", "LSTM", "GRU"],
    "05_nvidia_gpu_deep_learning.ipynb": ["def lapor(", "tensorrt>=10,<11", "mixed_precision"],
    "06_generative_ai_intro.ipynb": ["set_memory_growth", "clear_session", "diffusers=="],
}
fail = False
for name, markers in REGISTRY.items():
    errs = []
    try:
        nb = json.loads((ROOT / name).read_text(encoding="utf-8"))
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
    idx_l = [i for i, c in enumerate(nb["cells"]) if "🏋️ Latihan" in "".join(c["source"])]
    idx_r = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "markdown" and any(
        l.lstrip().startswith("#") and ("Ringkasan" in l or "Kesimpulan" in l) for l in "".join(c["source"]).splitlines())]
    if idx_l and idx_r and not (idx_l[0] < idx_r[-1] <= idx_l[0] + 2):
        errs.append(f"Latihan ({idx_l[0]}) tidak tepat sebelum Ringkasan ({idx_r[-1]})")
    status = "FAIL" if errs else "PASS"
    fail |= bool(errs)
    print(f"{status} {name}" + "".join(f"\n   - {e}" for e in errs))
sys.exit(1 if fail else 0)
