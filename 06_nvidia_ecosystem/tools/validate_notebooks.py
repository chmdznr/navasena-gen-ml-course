#!/usr/bin/env python3
# 06_nvidia_ecosystem/tools/validate_notebooks.py
"""Structural + content gate for Module 06 notebooks (the 5-notebook v2 arc, nb01-nb05).

The module is NOW 5 notebooks (01_gpu_precision_optimization .. 05_capstone_guarded_deploy).
Older 8/9-notebook companions (nb00/nb06-nb08) are gone and must NOT be expected here.

Each notebook must:
  1. exist (exactly 5 present: 0[1-5]_*.ipynb) and be valid notebook JSON,
  2. clear a DEPTH FLOOR (>= 15 cells AND >= 5000 chars of markdown) -- the NEW gate v1 lacked,
  3. carry its v2 content markers (substrings proving a topic/technique is present),
  4. be free of stale-stub strings from the deleted v1/old companion.

Exit code 0 = all pass; 1 = any failure.
Usage: python validate_notebooks.py [nb01 nb02 ...]   (default: sweep all 5)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # 06_nvidia_ecosystem/

# --- Depth floor: the NEW gate v1 lacked. A notebook below either bound is a thin stub. ---
MIN_CELLS = 15
MIN_MD_CHARS = 5000

# --- Stale tokens from the deleted v1 stubs / old companion -- forbidden in EVERY notebook. ---
# Matched case-insensitively. 'gpt2' is the bare-name stub model (do NOT confuse with legit prose);
# 'tensorrt' alone is a legit M06 topic, only the pip-package spelling 'nvidia-tensorrt' is stale.
GLOBAL_FORBIDDEN = [
    "phi-2",            # old stub generator
    "gpt2",             # old stub model
    "nvidia-tensorrt",  # bogus/placeholder pip package from v1 (TensorRT itself is fine)
    "ner_en_bert",      # old NeMo NER stub example
    "megatrongpt",      # old NeMo stub
    "tinyllama",        # Module 05 generator, not M06
    "cudf",             # RAPIDS-as-generator stub framing (lives elsewhere)
    "module 05",        # cross-module leakage
    "delapan notebook", # old 8-notebook framing
    "sembilan notebook",# old 9-notebook framing
    "8 notebook",       # old companion label
    "9 notebook",       # old companion label
    "skalabel",         # ai-slop calque
]

# marker = substring (any of the alternatives in a tuple) that MUST appear (case-insensitive).
#   A bare string means "must appear"; a tuple means "at least one alternative must appear".
# forbidden = per-notebook extra forbidden strings, on top of GLOBAL_FORBIDDEN.
REGISTRY = {
    # ① GPU / precision / quantization / TensorRT vs TensorRT-LLM  [cert: Software Development]
    "01_gpu_precision_optimization.ipynb": {
        "markers": [
            "tensor core",                  # Tensor Cores accelerate matmul
            ("fp32", "fp16"),               # precision ladder
            "fp8",                          # FP8 precision tier
            "quantiz",                      # quantization concept
            ("load_in_4bit", "4-bit"),      # 4-bit quantization (3.09 -> 1.16 GB)
            "tensorrt-llm",                 # TensorRT vs TensorRT-LLM distinction
            ("nemotron", "nim"),            # the M06 model/stack thread runs through nb01
        ],
        "forbidden": [],
    },
    # ② Triton -> Dynamo -> NIM serving + RAG-on-NIM recap  [cert: Software Development]
    "02_serving_and_deploy.ipynb": {
        "markers": [
            "triton",                              # exam baseline server
            "dynamo",                              # current flagship (GTC2025)
            "nvidia/nemotron-3-nano-30b-a3b",      # the served reasoning model id
            "integrate.api.nvidia.com",            # OpenAI-compatible NIM endpoint
            ("extra_body", "enable_thinking"),     # reasoning-off shown inline
            ("streaming", "stream"),               # streaming demo
            "throughput",                          # throughput discussion
        ],
        "forbidden": [],
    },
    # ③ Fairness metrics + governance (Model Card, ethics)  [cert: Trustworthy AI]
    "03_fairness_and_governance.ipynb": {
        "markers": [
            "fairness",
            "demographic parity",            # fairness metric
            "equalized odds",                # fairness metric
            ("model card", "model_card"),    # governance artifact
            ("judge", "bias"),               # bias via LLM-judge
        ],
        "forbidden": [],
    },
    # ④ REAL NemoGuard NIMs + jailbreak self-check + PII + UU PDP + grounding  [MODULE IDENTITY]
    "04_safety_guardrails_privacy.ipynb": {
        "markers": [
            ("nemoguard", "content-safety"),                       # NemoGuard content-safety NIM
            "nvidia/llama-3.1-nemoguard-8b-content-safety",        # exact Aegis content-safety model id
            "nvidia/llama-3.1-nemoguard-8b-topic-control",         # exact topic-control model id
            ("topic control", "topic-control"),                    # topic-control rail
            "jailbreak",                                           # jailbreak detection (self-check classifier)
            ("/v1/classify", "classify"),                          # jailbreak-detect is a classifier endpoint
            "pii",                                                 # PII masking
            "[nik]",                                               # PII placeholder literals
            "[phone]",
            "[account]",
            "uu pdp",                                              # Indonesian data-protection law
            "grounding",                                           # grounding rail
        ],
        "forbidden": [],
    },
    # ⑤ FastAPI /ask wiring all rails + TestClient smoke + runbook  [cert: ties both]
    "05_capstone_guarded_deploy.ipynb": {
        "markers": [
            "fastapi",                              # guarded serving capstone
            "basemodel",                            # Pydantic request/response schemas
            "/ask",                                 # the serving endpoint
            "/health",                              # health-check endpoint
            "testclient",                           # in-process smoke test (Colab-safe)
            "uvicorn",                              # runbook: serve for real
            ("nemoguard", "content-safety"),        # content-safety rail wired in
            "jailbreak",                            # input jailbreak rail
            "grounding",                            # output grounding rail
            "pii",                                  # PII masking in the pipeline
            "nvidia/nemotron-3-nano-30b-a3b",       # NIM generator behind the guard
        ],
        "forbidden": [],
    },
}

EXPECTED = list(REGISTRY.keys())  # exactly the 5 v2 Colab notebooks
# The off-Colab Jetson lab (nb06) is a real deliverable, but it is a hardware runbook
# (bash cells run on a Jetson, not Colab) -- allowed alongside the 5, not a stray stub.
OFFCOLAB_LAB = "06_jetson_edge_deploy.ipynb"


def load_source(nb_path: Path) -> tuple[str, int, int]:
    """Return (full_source_lower, total_cells, markdown_char_count)."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    parts: list[str] = []
    md_chars = 0
    for cell in cells:
        src = cell.get("source", [])
        src = src if isinstance(src, str) else "".join(src)
        parts.append(src)
        if cell.get("cell_type") == "markdown":
            md_chars += len(src)
    return "\n".join(parts).lower(), len(cells), md_chars


def _present(src: str, marker) -> bool:
    """marker is a str (must appear) or a tuple (>=1 alternative must appear)."""
    if isinstance(marker, tuple):
        return any(alt.lower() in src for alt in marker)
    return marker.lower() in src


def check(nb_name: str) -> list[str]:
    nb_basename = Path(nb_name).name  # normalize "06_nvidia_ecosystem/foo.ipynb" -> "foo.ipynb"
    path = HERE / nb_basename
    errors: list[str] = []
    if not path.exists():
        return [f"{nb_name}: file not found"]
    try:
        src, n_cells, md_chars = load_source(path)
    except json.JSONDecodeError as e:
        return [f"{nb_name}: invalid notebook JSON -- {e}"]

    # Depth floor.
    if n_cells < MIN_CELLS:
        errors.append(f"{nb_name}: only {n_cells} cells (depth floor >= {MIN_CELLS})")
    if md_chars < MIN_MD_CHARS:
        errors.append(f"{nb_name}: only {md_chars} markdown chars (depth floor >= {MIN_MD_CHARS})")

    spec = REGISTRY.get(nb_basename, {"markers": [], "forbidden": []})
    for m in spec["markers"]:
        if not _present(src, m):
            shown = " | ".join(m) if isinstance(m, tuple) else m
            errors.append(f"{nb_name}: MISSING required marker [{shown}]")
    for f in spec["forbidden"] + GLOBAL_FORBIDDEN:
        if f.lower() in src:
            errors.append(f"{nb_name}: FORBIDDEN content present {f!r}")
    return errors


def main(argv: list[str]) -> int:
    targets = argv or EXPECTED
    all_errors: list[str] = []

    if not argv:
        # Exactly 5 notebooks present -- no more, no fewer.
        # Sweep ALL numeric-prefixed notebooks (0[0-9]_*) so a resurrected v1 stub
        # (00_*, 06_*-08_*) is flagged as 'unexpected', not silently ignored by a
        # narrow 0[1-5] glob. The 06_ Jetson lab is off-Colab and allowed (excluded here).
        present = sorted(p.name for p in HERE.glob("[0-9][0-9]_*.ipynb") if p.name != OFFCOLAB_LAB)
        if present != sorted(EXPECTED):
            all_errors.append(
                f"notebook set mismatch: found {present}, expected exactly the 5 v2 notebooks {sorted(EXPECTED)}"
            )
            print("[FAIL] notebook-set")
            for k in sorted(set(EXPECTED) - set(present)):
                print(f"    - missing: {k}")
            for k in sorted(set(present) - set(EXPECTED)):
                print(f"    - unexpected: {k}")

    for nb in targets:
        errs = check(nb)
        all_errors += errs
        print(f"[{'FAIL' if errs else 'PASS'}] {nb}")
        for e in errs:
            print(f"    - {e}")

    if all_errors:
        print(f"\nNOTEBOOKS VALIDATION FAILED: {len(all_errors)} issue(s)")
        return 1
    print(
        f"\nNOTEBOOKS OK: {len(targets)} notebooks (5-notebook v2 arc), depth floor cleared "
        f"(>= {MIN_CELLS} cells & >= {MIN_MD_CHARS} md chars), all v2 markers present, no stale stub content."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
