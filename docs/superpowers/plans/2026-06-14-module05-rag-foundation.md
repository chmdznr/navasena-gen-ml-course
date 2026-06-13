# Module 05 RAG — Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish Module 05's tested foundation — GPU-free RAG helper library, a notebook structural validator, a reworked `01_rag_fundamentals` notebook (footguns fixed), and a new `02_ingest_and_chunk` notebook that replaces the 6 hardcoded sentences with a real Indonesian document.

**Architecture:** Pure-logic helpers (chunking, token counting, chunk-quality scoring) live in `05_rag/tools/rag_utils.py` as the single source of truth, unit-tested locally with `pytest` (CPU only, no model downloads via dependency-injected tokenizer). Colab notebooks `git clone` the course repo and import these helpers (DRY), keeping only GPU/model code inline. A `validate_notebooks.py` script is the structural gate for every notebook (valid JSON, required content markers, footgun checks). This is Plan 1 of 5; it produces a working, validated 2-notebook foundation on its own.

**Tech Stack:** Python 3, `pytest`, `transformers<5`, `sentence-transformers` (paraphrase-multilingual-MiniLM-L12-v2), `faiss-cpu`, Qwen2.5-3B-Instruct (4-bit via bitsandbytes), Docling + pdfplumber, `reportlab` (sample-PDF generation). Markdown in Bahasa Indonesia; code + comments in English.

**Reference spec:** `docs/superpowers/specs/2026-06-14-module05-rag-rework-design.md`

---

## File Structure

| Path | Responsibility |
|---|---|
| `05_rag/tools/rag_utils.py` | CREATE — pure-logic helpers: `simple_token_count`, `TokenCounter`, `split_sentences`, `Chunk`, `TextChunker` (3 strategies), `chunk_quality_score`. Source of truth; GPU-free. |
| `05_rag/tools/test_rag_utils.py` | CREATE — local `pytest` suite for `rag_utils` (CPU only). |
| `05_rag/tools/validate_notebooks.py` | CREATE — structural/content gate for all 8 notebooks via a per-notebook registry. |
| `05_rag/data/make_sample.py` | CREATE — builds `sample_id_document.pdf` from baked-in Indonesian text via reportlab. |
| `05_rag/data/sample_id_document.pdf` | CREATE (generated, committed) — baked-in fallback corpus so notebook cells always run. |
| `05_rag/01_rag_fundamentals.ipynb` | MODIFY — rework: multilingual embed, Qwen 4-bit, `apply_chat_template`, `max_new_tokens`, deterministic decoding, run all test questions incl. out-of-corpus, drop dead `datasets` import. |
| `05_rag/02_ingest_and_chunk.ipynb` | CREATE — Docling+pdfplumber ingestion, 3-strategy chunking via `rag_utils`, quality-score comparison, embed chunks. |

**Canonical Colab bootstrap cell** (reused verbatim by every notebook from nb02 onward; defined once here):

```python
# --- Navasena course bootstrap (Colab) ---
import os, sys
if not os.path.exists("navasena-gen-ml-course"):
    !git clone --depth 1 https://github.com/<org>/navasena-gen-ml-course.git
sys.path.append(os.path.abspath("navasena-gen-ml-course/05_rag"))
from tools.rag_utils import TokenCounter, TextChunker, chunk_quality_score, split_sentences
```
> Replace `<org>` with the real GitHub org/owner during Task 4 (confirm the remote with `git remote get-url origin`). If the repo is private, the notebook falls back to pasting `rag_utils.py` contents — Task 3 step covers emitting that fallback cell.

---

## Task 1: Pure-logic helper library (`rag_utils.py`) — TDD

**Files:**
- Create: `05_rag/tools/rag_utils.py`
- Test: `05_rag/tools/test_rag_utils.py`

- [ ] **Step 1: Write the failing tests**

```python
# 05_rag/tools/test_rag_utils.py
import pytest
from rag_utils import (
    simple_token_count, TokenCounter, split_sentences, Chunk,
    TextChunker, chunk_quality_score,
)

# Deterministic fake tokenizer: 1 token per whitespace word (hermetic, no downloads)
def word_tok(text: str) -> int:
    return len(text.split())

def test_simple_token_count_counts_words_and_punct():
    assert simple_token_count("Halo, dunia!") == 4  # Halo , dunia !

def test_split_sentences_indonesian():
    text = "Borobudur adalah candi Buddha. Letaknya di Magelang. Dibangun abad ke-8!"
    assert split_sentences(text) == [
        "Borobudur adalah candi Buddha.",
        "Letaknya di Magelang.",
        "Dibangun abad ke-8!",
    ]

def test_split_sentences_empty():
    assert split_sentences("   ") == []

def test_sentence_chunks_respect_token_budget():
    counter = TokenCounter(tokenize_fn=word_tok)
    chunker = TextChunker(counter, max_tokens=5, overlap_tokens=0)
    text = "satu dua tiga. empat lima enam. tujuh delapan."
    chunks = chunker.sentence(text)
    assert all(c.n_tokens <= 5 for c in chunks)
    assert "".join(c.text for c in chunks).replace(" ", "") != ""  # non-empty coverage

def test_fixed_chunks_have_overlap():
    counter = TokenCounter(tokenize_fn=word_tok)
    chunker = TextChunker(counter, max_tokens=4, overlap_tokens=2)
    text = " ".join(f"w{i}" for i in range(10))  # 10 words
    chunks = chunker.fixed(text)
    assert len(chunks) >= 3
    # consecutive chunks share overlap_tokens words
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    assert first_words[-2:] == second_words[:2]

def test_semantic_chunks_split_on_blank_lines():
    counter = TokenCounter(tokenize_fn=word_tok)
    chunker = TextChunker(counter, max_tokens=100, overlap_tokens=0)
    text = "para satu kalimat.\n\npara dua kalimat lain."
    chunks = chunker.semantic(text)
    assert len(chunks) == 2

def test_chunk_quality_score_perfect_is_high():
    # uniform, within-budget chunks => high score
    chunks = [Chunk(text="a b c", start=0, end=5, n_tokens=3),
              Chunk(text="d e f", start=5, end=10, n_tokens=3)]
    score = chunk_quality_score(chunks, max_tokens=4, original_len=10)
    assert 80 <= score <= 100

def test_chunk_quality_score_oversized_penalized():
    chunks = [Chunk(text="x", start=0, end=1, n_tokens=1),
              Chunk(text="y"*50, start=1, end=51, n_tokens=50)]
    score = chunk_quality_score(chunks, max_tokens=4, original_len=51)
    assert score < 60

def test_chunk_quality_score_empty_is_zero():
    assert chunk_quality_score([], max_tokens=4, original_len=10) == 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd 05_rag/tools && python -m pytest test_rag_utils.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag_utils'` (or import errors).

- [ ] **Step 3: Implement `rag_utils.py`**

```python
# 05_rag/tools/rag_utils.py
"""Pure-Python RAG helpers for Navasena Module 05.

GPU/model-free so they can be unit-tested locally (CPU) and imported by the
Colab notebooks. This file is the SOURCE OF TRUTH — notebooks import these
helpers and must not re-implement them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Callable, List

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORDish = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def simple_token_count(text: str) -> int:
    """Whitespace+punctuation token estimate. Notebooks override with a real tokenizer."""
    return len(_WORDish.findall(text))


@dataclass
class TokenCounter:
    """Counts tokens via an injectable function (default: `simple_token_count`).

    In notebooks: TokenCounter(tokenize_fn=lambda t: len(qwen_tok.encode(t))).
    """
    tokenize_fn: Callable[[str], int] = simple_token_count

    def count(self, text: str) -> int:
        return self.tokenize_fn(text)


def split_sentences(text: str) -> List[str]:
    """Split on sentence terminators; trims and drops empties."""
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


@dataclass
class Chunk:
    text: str
    start: int  # char offset into the source text
    end: int
    n_tokens: int


class TextChunker:
    """Three chunking strategies, all token-budgeted via a TokenCounter."""

    def __init__(self, counter: TokenCounter, max_tokens: int = 128, overlap_tokens: int = 24):
        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be < max_tokens")
        self.counter = counter
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def _emit(self, words: List[str], cursor: int) -> Chunk:
        text = " ".join(words)
        return Chunk(text=text, start=cursor, end=cursor + len(text), n_tokens=self.counter.count(text))

    def fixed(self, text: str) -> List[Chunk]:
        """Sliding word window packed to the token budget, with word overlap."""
        words = text.split()
        if not words:
            return []
        chunks: List[Chunk] = []
        i, cursor = 0, 0
        while i < len(words):
            window: List[str] = []
            j = i
            while j < len(words) and self.counter.count(" ".join(window + [words[j]])) <= self.max_tokens:
                window.append(words[j])
                j += 1
            if not window:  # single word exceeds budget — emit it alone to make progress
                window = [words[i]]
                j = i + 1
            chunks.append(self._emit(window, cursor))
            if j >= len(words):
                break
            # step back `overlap_tokens` words for the next window
            step = max(1, len(window) - self.overlap_tokens)
            consumed = " ".join(words[i:i + step])
            cursor += len(consumed) + 1
            i += step
        return chunks

    def sentence(self, text: str) -> List[Chunk]:
        """Greedily pack whole sentences until the token budget is reached."""
        sents = split_sentences(text)
        chunks: List[Chunk] = []
        buf: List[str] = []
        cursor = 0
        for s in sents:
            trial = " ".join(buf + [s])
            if buf and self.counter.count(trial) > self.max_tokens:
                chunks.append(self._emit(buf, cursor))
                cursor += len(" ".join(buf)) + 1
                buf = [s]
            else:
                buf.append(s)
        if buf:
            chunks.append(self._emit(buf, cursor))
        return chunks

    def semantic(self, text: str) -> List[Chunk]:
        """Paragraph-based: split on blank lines, then pack paragraphs to budget."""
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks: List[Chunk] = []
        buf: List[str] = []
        cursor = 0
        for p in paras:
            trial = " ".join(buf + [p])
            if buf and self.counter.count(trial) > self.max_tokens:
                chunks.append(self._emit(buf, cursor))
                cursor += len(" ".join(buf)) + 1
                buf = [p]
            else:
                buf.append(p)
        if buf:
            chunks.append(self._emit(buf, cursor))
        return chunks


def chunk_quality_score(chunks: List[Chunk], max_tokens: int, original_len: int) -> float:
    """0-100 quality score combining three sub-metrics:

    - budget_adherence: fraction of chunks whose n_tokens <= max_tokens
    - size_consistency:  1 - coefficient_of_variation(n_tokens), clamped to [0,1]
    - coverage:          min(1, sum(chunk_chars) / original_len)  (overlap can push >1)

    Returns 0.0 for an empty chunk list.
    """
    if not chunks:
        return 0.0
    sizes = [c.n_tokens for c in chunks]
    budget_adherence = sum(1 for n in sizes if n <= max_tokens) / len(sizes)
    m = mean(sizes)
    cov = (pstdev(sizes) / m) if m > 0 else 1.0
    size_consistency = max(0.0, min(1.0, 1.0 - cov))
    total_chars = sum(len(c.text) for c in chunks)
    coverage = min(1.0, total_chars / original_len) if original_len > 0 else 0.0
    score = (0.4 * budget_adherence + 0.3 * size_consistency + 0.3 * coverage) * 100.0
    return round(score, 1)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd 05_rag/tools && python -m pytest test_rag_utils.py -v`
Expected: PASS (9 passed). If `test_fixed_chunks_have_overlap` fails, check the `step`/overlap math in `fixed()`.

- [ ] **Step 5: Commit**

```bash
git add 05_rag/tools/rag_utils.py 05_rag/tools/test_rag_utils.py
git commit -m "feat(module05): add tested GPU-free RAG helper library (chunking + quality scorer)"
```

---

## Task 2: Baked-in sample Indonesian document

**Files:**
- Create: `05_rag/data/make_sample.py`
- Create: `05_rag/data/sample_id_document.pdf` (generated, committed)

- [ ] **Step 1: Write `make_sample.py`**

```python
# 05_rag/data/make_sample.py
"""Generate the baked-in sample corpus (sample_id_document.pdf).

Original Indonesian factual prose (paraphrased, not copied from any source) so the
notebooks always have a real multi-paragraph document to ingest even without an upload.
Run: python make_sample.py  ->  writes sample_id_document.pdf next to this file.
"""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

PARAS = [
    "Candi Borobudur adalah candi Buddha terbesar di dunia yang terletak di "
    "Kabupaten Magelang, Jawa Tengah. Candi ini dibangun pada masa Wangsa Syailendra "
    "sekitar abad ke-8 hingga ke-9 Masehi.",
    "Bangunan Borobudur tersusun atas sembilan teras berundak: enam teras berbentuk "
    "persegi dan tiga teras berbentuk lingkaran. Di puncaknya terdapat sebuah stupa "
    "induk besar yang dikelilingi oleh stupa-stupa berlubang.",
    "Dinding candi dihiasi sekitar 2.672 panel relief dan terdapat 504 arca Buddha. "
    "Relief tersebut menceritakan ajaran Buddha serta kehidupan masyarakat pada masanya.",
    "Borobudur sempat terbengkalai dan tertutup material letusan gunung berapi. "
    "Candi ini ditemukan kembali pada tahun 1814 atas perhatian Thomas Stamford Raffles, "
    "lalu dipugar secara besar-besaran pada tahun 1975 hingga 1982 dengan dukungan UNESCO.",
    "Pada tahun 1991 Borobudur ditetapkan sebagai Situs Warisan Dunia UNESCO. "
    "Hingga kini candi ini menjadi salah satu tujuan wisata dan ziarah keagamaan "
    "yang paling banyak dikunjungi di Indonesia.",
]

def build(out: Path) -> None:
    doc = SimpleDocTemplate(str(out), pagesize=A4, title="Candi Borobudur")
    styles = getSampleStyleSheet()
    story = [Paragraph("Candi Borobudur", styles["Title"]), Spacer(1, 12)]
    for p in PARAS:
        story.append(Paragraph(p, styles["BodyText"]))
        story.append(Spacer(1, 8))
    doc.build(story)
    print(f"wrote {out} ({out.stat().st_size} bytes)")

if __name__ == "__main__":
    build(Path(__file__).with_name("sample_id_document.pdf"))
```

- [ ] **Step 2: Install reportlab and generate the PDF**

Run: `uv pip install reportlab && cd 05_rag/data && python make_sample.py`
Expected: `wrote .../sample_id_document.pdf (NNNN bytes)` with N > 2000.

- [ ] **Step 3: Verify the PDF extracts clean text (no GPU)**

Run: `cd 05_rag/data && python -c "import pdfplumber; print(''.join((p.extract_text() or '') for p in pdfplumber.open('sample_id_document.pdf').pages)[:200])"`
Expected: starts with `Candi Borobudur` and contains `Magelang`.

- [ ] **Step 4: Commit**

```bash
git add 05_rag/data/make_sample.py 05_rag/data/sample_id_document.pdf
git commit -m "feat(module05): add baked-in Indonesian sample document (Borobudur) for ingestion"
```

---

## Task 3: Notebook structural validator (`validate_notebooks.py`)

**Files:**
- Create: `05_rag/tools/validate_notebooks.py`

This is the executable gate for Tasks 4–5 (and every later notebook). It mirrors `04_llm/tools/` in spirit: per-notebook required-marker registry + footgun checks.

- [ ] **Step 1: Implement the validator**

```python
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
            "do_sample=False",                        # deterministic factual QA
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
            "from tools.rag_utils import",  # DRY: imports the tested helpers
            "DocumentConverter",            # Docling primary path
            "pdfplumber",                   # lightweight fallback path
            "chunk_quality_score",          # quality scoring shown
            ".sentence(", ".fixed(", ".semantic(",  # all three strategies compared
            "sample_id_document.pdf",       # baked-in fallback wired
        ],
        "forbidden": [],
    },
    # later plans extend this registry for nb03..nb08
}


def load_source(nb_path: Path) -> str:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    parts = []
    for cell in nb.get("cells", []):
        parts.append("".join(cell.get("source", [])))
    return "\n".join(parts)


def check(nb_name: str) -> list[str]:
    path = HERE / nb_name
    errors: list[str] = []
    if not path.exists():
        return [f"{nb_name}: file not found"]
    try:
        src = load_source(path)
    except json.JSONDecodeError as e:
        return [f"{nb_name}: invalid notebook JSON — {e}"]
    spec = REGISTRY.get(nb_name, {"markers": [], "forbidden": []})
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
```

- [ ] **Step 2: Run it against the current (un-reworked) nb01 to confirm it reports the gaps**

Run: `cd 05_rag/tools && python validate_notebooks.py 01_rag_fundamentals.ipynb`
Expected: FAIL — missing `paraphrase-multilingual-MiniLM-L12-v2`, `apply_chat_template`, `max_new_tokens`, `load_in_4bit`, `do_sample=False`, `Cocos`; forbidden `import datasets` / `max_length=512` / `presiden` present. (This proves the gate detects the footguns Task 4 fixes.)

- [ ] **Step 3: Commit**

```bash
git add 05_rag/tools/validate_notebooks.py
git commit -m "feat(module05): add notebook structural validator with per-notebook marker registry"
```

---

## Task 4: Rework `01_rag_fundamentals.ipynb`

**Files:**
- Modify: `05_rag/01_rag_fundamentals.ipynb`

Goal: same gentle "hello-world of RAG" narrative, but technically correct. Keep markdown in Bahasa Indonesia. The notebook stays self-contained (no repo import needed — nb01 is the entry point a learner may open first).

**Target cell sequence** (markdown `M`, code `C`):

1. `M` — Judul + tujuan: apa itu RAG, kenapa perlu (hallucination + knowledge cutoff). *(keep existing prose, lightly edited)*
2. `C` — Install (pinned):
```python
!pip install -q "transformers<5" "sentence-transformers>=3.0" faiss-cpu accelerate bitsandbytes
```
3. `C` — Imports (NO `datasets`):
```python
import torch, faiss, numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
```
4. `M` — Embedding: teks → vektor; makna mirip → vektor berdekatan; pakai model multilingual karena korpus & pertanyaan berbahasa Indonesia.
5. `C` — Embedder:
```python
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
dim = embedder.get_sentence_embedding_dimension()  # 384
```
6. `M` — Knowledge base: kumpulan fakta. *Fakta dibuat "timeless" (tanpa klaim politik bertanggal).*
7. `C` — KB (timeless facts, mixed ID/EN to mirror real multilingual use):
```python
knowledge_base = [
    "Menara Eiffel terletak di kota Paris, ibu kota Prancis.",
    "Bahasa pemrograman Python diciptakan oleh Guido van Rossum.",
    "Candi Borobudur berada di Magelang, Jawa Tengah, dan merupakan candi Buddha terbesar di dunia.",
    "The Great Wall of China is over 13,000 miles long.",
    "Fotosintesis adalah proses tumbuhan mengubah cahaya matahari menjadi energi.",
    "Komodo adalah kadal terbesar di dunia dan hidup di Indonesia.",
]
```
8. `C` — Embed + FAISS index (L2 here; cosine introduced in nb05):
```python
kb_vectors = embedder.encode(knowledge_base, convert_to_numpy=True).astype("float32")
index = faiss.IndexFlatL2(dim)
index.add(kb_vectors)
```
9. `M` — Generator: Qwen2.5-3B-Instruct, dimuat 4-bit agar muat di T4.
10. `C` — Load generator (4-bit, fp16 compute — NOT bf16):
```python
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                         bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
model_name = "Qwen/Qwen2.5-3B-Instruct"
tok = AutoTokenizer.from_pretrained(model_name)
gen = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb, device_map="auto")
```
11. `M` — Pipeline RAG: retrieve → augment → generate. Jelaskan kenapa decoding deterministik (greedy) lebih aman untuk QA faktual.
12. `C` — Retrieval + grounded generation using `apply_chat_template`, `max_new_tokens`, deterministic:
```python
def retrieve(query, k=3):
    qv = embedder.encode([query], convert_to_numpy=True).astype("float32")
    dist, idx = index.search(qv, k)
    return [knowledge_base[i] for i in idx[0]], dist[0]

def ask(query, k=3):
    docs, dist = retrieve(query, k)
    context = "\n".join(f"- {d}" for d in docs)
    messages = [
        {"role": "system", "content": "Jawab HANYA berdasarkan konteks. Jika tidak ada di konteks, katakan tidak tahu."},
        {"role": "user", "content": f"Konteks:\n{context}\n\nPertanyaan: {query}"},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(gen.device)
    out = gen.generate(**inputs, max_new_tokens=128, do_sample=False)
    answer = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return answer.strip(), docs
```
13. `M` — Uji: pertanyaan yang ADA di KB vs pertanyaan di LUAR KB (kenapa fallback penting).
14. `C` — Run MULTIPLE questions incl. the out-of-corpus probe (all executed, none commented out):
```python
for q in ["Di mana letak Menara Eiffel?",
          "Siapa pencipta Python?",
          "Di mana Candi Borobudur?",
          "Where is Cocos Island?"]:   # <-- deliberately NOT in the KB
    ans, docs = ask(q)
    print(f"Q: {q}\nDoc teratas: {docs[0]}\nA: {ans}\n{'-'*50}")
```
15. `M` — Ringkasan komponen (tabel) + jembatan ke nb02 (korpus asli & chunking) dan ke nb05 (cosine vs L2).

**Steps:**

- [ ] **Step 1:** Open `05_rag/01_rag_fundamentals.ipynb` and edit cells to match the target sequence above (use `NotebookEdit` per cell). Remove the `datasets` import, the regex `clean_response`, the manual `<|im_start|>` string-splitting, `max_length=512`, `do_sample=True`/temperature, and the stale "presiden 2024" KB entry.
- [ ] **Step 2:** Validate JSON well-formedness.
  Run: `python -c "import json; json.load(open('05_rag/01_rag_fundamentals.ipynb')); print('valid json')"`
  Expected: `valid json`
- [ ] **Step 3:** Run the validator gate.
  Run: `cd 05_rag/tools && python validate_notebooks.py 01_rag_fundamentals.ipynb`
  Expected: `[PASS] 01_rag_fundamentals.ipynb` / `OK: 0 issue(s)`
- [ ] **Step 4 (manual GPU gate):** On Colab T4, Runtime → Run all. Expected: in-KB questions answer correctly from the top doc; the **Cocos Island** question yields an "tidak tahu / not in context" style answer (grounding works). Record runtime in a markdown note.
- [ ] **Step 5: Commit**
```bash
git add 05_rag/01_rag_fundamentals.ipynb
git commit -m "fix(module05): rework nb01 — multilingual embed, Qwen2.5-3B 4-bit, apply_chat_template, deterministic QA, out-of-corpus probe"
```

---

## Task 5: New notebook `02_ingest_and_chunk.ipynb`

**Files:**
- Create: `05_rag/02_ingest_and_chunk.ipynb`

Goal: replace hardcoded sentences with a real document; teach 3 chunking strategies and how to *measure* chunk quality. Imports the tested helpers from `rag_utils` (DRY).

**Target cell sequence:**

1. `M` — Tujuan: dari dokumen nyata ke potongan (chunk) yang siap di-retrieve. Kenapa chunking penting (konteks LLM terbatas, retrieval per-potongan).
2. `C` — **Canonical bootstrap cell** (from the File Structure section above) — clone repo + `from tools.rag_utils import ...`.
3. `C` — Install ingestion deps:
```python
!pip install -q docling pdfplumber "transformers<5"
```
4. `M` — Memuat dokumen: Docling (utama, paham layout) dengan fallback pdfplumber agar sel selalu jalan di Colab gratis.
5. `C` — Upload-or-fallback + extract with Docling, fallback pdfplumber:
```python
import os
from pathlib import Path
try:
    from google.colab import files
    up = files.upload()
    pdf_path = next(iter(up)) if up else None
except Exception:
    pdf_path = None
if not pdf_path:
    pdf_path = "navasena-gen-ml-course/05_rag/data/sample_id_document.pdf"  # baked-in fallback

def extract_text(path):
    try:
        from docling.document_converter import DocumentConverter
        return DocumentConverter().convert(path).document.export_to_markdown()
    except Exception as e:
        print(f"Docling tidak tersedia ({e}); pakai pdfplumber.")
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            return "\n\n".join((pg.extract_text() or "") for pg in pdf.pages)

raw_text = extract_text(pdf_path)
print(raw_text[:400])
```
6. `M` — Menghitung token dengan tokenizer Qwen (agar ukuran chunk sesuai model yang dipakai di nb01).
7. `C` — Real tokenizer wired into the injectable TokenCounter:
```python
from transformers import AutoTokenizer
qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
counter = TokenCounter(tokenize_fn=lambda t: len(qwen_tok.encode(t)))
```
8. `M` — Tiga strategi chunking: per-kalimat, ukuran-tetap (overlap), semantik (per-paragraf). Trade-off masing-masing.
9. `C` — Build chunks with all three strategies + score each:
```python
strategies = {
    "sentence":  TextChunker(counter, max_tokens=128, overlap_tokens=0).sentence,
    "fixed":     TextChunker(counter, max_tokens=128, overlap_tokens=24).fixed,
    "semantic":  TextChunker(counter, max_tokens=128, overlap_tokens=0).semantic,
}
results = {}
for name, fn in strategies.items():
    chunks = fn(raw_text)
    score = chunk_quality_score(chunks, max_tokens=128, original_len=len(raw_text))
    results[name] = (chunks, score)
    print(f"{name:9s}: {len(chunks):3d} chunks  quality={score}")
```
10. `M` — Membaca skor: budget adherence + konsistensi ukuran + coverage. Strategi mana menang untuk dokumen ini & kenapa.
11. `C` — Comparison table + token-size histogram (matplotlib) per strategy. *(plotting only; CPU)*
12. `M` — Memilih chunk terbaik & meng-embed untuk dipakai notebook berikutnya.
13. `C` — Embed the chosen strategy's chunks (reuse nb01 embedder) and show the count/dim; note these feed nb03.
14. `M` — Ringkasan + jembatan ke nb03 (retrieve lebih baik: over-fetch + rerank).

**Steps:**

- [ ] **Step 1:** Author the notebook cell-by-cell (`NotebookEdit` with `edit_mode: insert`) per the sequence above.
- [ ] **Step 2:** Validate JSON.
  Run: `python -c "import json; json.load(open('05_rag/02_ingest_and_chunk.ipynb')); print('valid json')"`
  Expected: `valid json`
- [ ] **Step 3:** Run the validator gate.
  Run: `cd 05_rag/tools && python validate_notebooks.py 02_ingest_and_chunk.ipynb`
  Expected: `[PASS]` / `OK: 0 issue(s)`
- [ ] **Step 4 (manual GPU/Colab gate):** On Colab, Run all. Confirm: fallback PDF loads when no upload; all three strategies produce chunks; quality scores print; histograms render. Note runtime.
- [ ] **Step 5: Commit**
```bash
git add 05_rag/02_ingest_and_chunk.ipynb
git commit -m "feat(module05): add nb02 ingest & chunk — Docling+pdfplumber, 3 strategies, quality scoring"
```

---

## Self-Review

**Spec coverage (Foundation slice of §5/§7/§8):**
- nb01 rework (footguns, multilingual, 4-bit, apply_chat_template, out-of-corpus) → Task 4 ✅
- nb02 ingest & chunk (Docling+fallback, 3 strategies, quality scorer) → Task 5 ✅
- Tested chunking + quality logic → Tasks 1 ✅
- Baked-in sample document → Task 2 ✅
- `tools/` validator (notebook gate) → Task 3 ✅
- Deferred to later plans (correctly out of scope here): nb03–08, slides/quiz/cheatsheet validators, RAGAS, Chroma. Tracked in spec §11.

**Placeholder scan:** No TBD/TODO. The only intentional fill-in is `<org>` in the clone URL (Task 4 step instructs confirming the real remote) — flagged, not silent.

**Type/name consistency:** `TokenCounter(tokenize_fn=...)`, `TextChunker(counter, max_tokens, overlap_tokens)`, `Chunk(text,start,end,n_tokens)`, `chunk_quality_score(chunks, max_tokens, original_len)`, `split_sentences` — names match across `rag_utils.py`, `test_rag_utils.py`, the validator markers (`.sentence(`/`.fixed(`/`.semantic(`, `chunk_quality_score`), and both notebooks' import + usage. Embedder id `paraphrase-multilingual-MiniLM-L12-v2` and model id `Qwen/Qwen2.5-3B-Instruct` are identical in nb01, nb02, and the validator registry.

**Note on the Colab import pattern:** Task 5's bootstrap clones the repo to import `rag_utils`. If the course repo is private at build time, emit the paste-fallback cell (inline the `rag_utils.py` contents) and adjust the nb02 validator marker accordingly — decided during Task 5 Step 1.
