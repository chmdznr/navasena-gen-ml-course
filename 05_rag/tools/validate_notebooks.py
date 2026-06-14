# 05_rag/tools/validate_notebooks.py
"""Structural + content gate for Module 05 notebooks.

Checks each notebook is valid JSON, contains required content markers (substrings
that prove a topic/technique is present), and is free of known footguns.
Exit code 0 = all pass; 1 = any failure. Usage: python validate_notebooks.py [nb01 nb02 ...]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # 05_rag/

# marker = a substring that MUST appear somewhere in the notebook source.
# forbidden = a substring that must NOT appear (footguns / dead code).
REGISTRY = {
    "01_rag_fundamentals.ipynb": {
        "markers": [
            "paraphrase-multilingual-MiniLM-L12-v2",  # multilingual embeddings
            "apply_chat_template",                    # robust prompting
            "max_new_tokens",                         # not max_length
            "load_in_4bit",                           # T4-safe Qwen
            "do_sample=False",                        # deterministic factual QA — must be written without spaces around = (exact spelling)
            "Cocos",                                  # out-of-corpus probe is actually run
        ],
        "forbidden": [
            "import datasets",      # dead dependency removed
            "from datasets import", # dead dependency removed
            "max_length=512",       # footgun replaced by max_new_tokens
            "presiden",             # stale time-sensitive fact removed
        ],
    },
    "02_ingest_and_chunk.ipynb": {
        "markers": [
            "from tools.rag_utils import",  # DRY: imports the tested helpers — must be a single-line import (not parenthesized/multi-line)
            "DocumentConverter",            # Docling primary path
            "pdfplumber",                   # lightweight fallback path
            "chunk_quality_score",          # quality scoring shown
            ".sentence(", ".fixed(", ".semantic(",  # all three strategies compared
            "sample_id_document.pdf",       # baked-in fallback wired
            "PdfPipelineOptions",           # Docling pipeline made explicit (OCR/TableFormer config)
            "iterate_items",                # DoclingDocument structure is surfaced (labels/tables)
        ],
        "forbidden": [],
    },
    "03_retrieve_better_rerank.ipynb": {
        "markers": [
            "paraphrase-multilingual-MiniLM-L12-v2",  # bi-encoder (stage 1)
            "bge-reranker-v2-m3",                     # cross-encoder reranker (stage 2)
            "rank_change_table",                      # tested rank-change helper
            "apply_chat_template",                    # grounded generation
        ],
        "forbidden": [],
    },
    "04_measure_ragas.ipynb": {
        "markers": [
            "EvaluationDataset",                     # modern RAGAS dataset (not legacy HF-Dataset)
            "LangchainLLMWrapper",                   # cloud judge wrapped for RAGAS
            "LLMContextPrecisionWithReference",      # rerank-sensitive headline metric
            "Faithfulness",                          # grounding metric
            "NVIDIA_API_KEY",                        # key read from Colab Secrets, not hardcoded
            "paraphrase-multilingual-MiniLM-L12-v2", # local RAGAS embeddings + bi-encoder
            "ragas==0.4.3",                          # pin to avoid the vertexai ImportError, ragas issue #2753
            "langchain-community>=0.3.0,<0.4.0",    # pin to avoid the vertexai ImportError, ragas issue #2753
        ],
        "forbidden": [
            "ragas.metrics import context_precision",  # guard against legacy 0.1 lowercase API
        ],
    },
    "05_scale_the_index.ipynb": {
        "markers": [
            "IndexFlatIP",             # cosine via inner product
            "normalize_L2",            # L2-normalize for cosine
            "write_index",             # FAISS persistence
            "IndexHNSWFlat",           # approximate-index taxonomy
            "recall_at_k",             # tested benchmark helper
            "PersistentClient",        # ChromaDB persistence
            "embedding_function=None", # bring-your-own-vectors (avoids ONNX auto-download)
        ],
        "forbidden": [
            "client.persist()",        # removed in chromadb 0.4+ (AttributeError); auto-persist now
        ],
    },
    # later plans extend this registry for nb06..nb08
}


def load_source(nb_path: Path) -> str:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    parts = []
    for cell in nb.get("cells", []):
        src = cell.get("source", [])
        parts.append(src if isinstance(src, str) else "".join(src))
    return "\n".join(parts)


def check(nb_name: str) -> list[str]:
    nb_basename = Path(nb_name).name  # normalize so "05_rag/foo.ipynb" → "foo.ipynb"
    path = HERE / nb_basename
    errors: list[str] = []
    if not path.exists():
        return [f"{nb_name}: file not found"]
    try:
        src = load_source(path)
    except json.JSONDecodeError as e:
        return [f"{nb_name}: invalid notebook JSON — {e}"]
    spec = REGISTRY.get(nb_basename, {"markers": [], "forbidden": []})
    for m in spec["markers"]:
        if m not in src:
            errors.append(f"{nb_name}: MISSING required marker {m!r}")
    for f in spec["forbidden"]:
        if f in src:
            errors.append(f"{nb_name}: FORBIDDEN content present {f!r}")
    return errors


def main(argv: list[str]) -> int:
    targets = argv or list(REGISTRY.keys())
    all_errors: list[str] = []
    for nb in targets:
        errs = check(nb)
        all_errors += errs
        print(f"[{'FAIL' if errs else 'PASS'}] {nb}")
        for e in errs:
            print(f"    - {e}")
    print(f"\n{'FAILED' if all_errors else 'OK'}: {len(all_errors)} issue(s)")
    return 1 if all_errors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
