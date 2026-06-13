# Module 05 nb03 — Retrieve Better (Reranking) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add notebook `03_retrieve_better_rerank.ipynb` that turns single-stage dense retrieval into two-stage retrieve→rerank (bi-encoder over-fetch → cross-encoder rerank → top-3 → grounded answer), with a quantified before/after rank-change visual and a latency decomposition.

**Architecture:** Self-contained Colab notebook (clones the repo only to import the tested `rag_utils` helpers). A purpose-built ~24-passage Indonesian corpus with lexically-similar near-duplicates makes reranking visibly reorder results. Pure-logic rank-change math lives in `rag_utils.rank_change_table()` (unit-tested locally, GPU-free); the notebook calls it. Reranker is `BAAI/bge-reranker-v2-m3` via `sentence_transformers.CrossEncoder`. Generation reuses Qwen2.5-3B-Instruct (4-bit) with `apply_chat_template`.

**Tech Stack:** transformers<5, sentence-transformers (multilingual MiniLM bi-encoder + bge-reranker-v2-m3 CrossEncoder), faiss-cpu, Qwen2.5-3B-Instruct 4-bit, matplotlib. Markdown Bahasa Indonesia; code/comments English.

**Reference spec:** `docs/superpowers/specs/2026-06-14-module05-rag-rework-design.md` §5 (nb03).

---

## File Structure

| Path | Responsibility |
|---|---|
| `05_rag/tools/rag_utils.py` | MODIFY — add `rank_change_table(candidate_ids, bi_scores, rerank_scores, top_k)` (pure, GPU-free). |
| `05_rag/tools/test_rag_utils.py` | MODIFY — add tests for `rank_change_table`. |
| `05_rag/tools/validate_notebooks.py` | MODIFY — add `03_retrieve_better_rerank.ipynb` registry entry. |
| `05_rag/03_retrieve_better_rerank.ipynb` | CREATE — the two-stage retrieval notebook. |

---

## Task 1: `rank_change_table` helper (TDD)

**Files:** Modify `05_rag/tools/rag_utils.py`, `05_rag/tools/test_rag_utils.py`

- [ ] **Step 1: Append failing tests to `test_rag_utils.py`**

```python
from rag_utils import rank_change_table

def test_rank_change_promotion_and_kept():
    # bi-encoder order [10,20,30]; reranker prefers 30 (worst in bi) then 10 then 20
    rows = rank_change_table([10, 20, 30], [0.9, 0.8, 0.7], [0.10, 0.05, 0.99], top_k=2)
    assert [r["doc_id"] for r in rows] == [30, 10, 20]          # sorted by rerank rank
    assert rows[0] == {"doc_id": 30, "bi_rank": 3, "rerank_rank": 1, "delta": 2,
                       "bi_score": 0.7, "rerank_score": 0.99, "kept": True}
    assert rows[1]["doc_id"] == 10 and rows[1]["rerank_rank"] == 2 and rows[1]["kept"] is True
    assert rows[2]["doc_id"] == 20 and rows[2]["rerank_rank"] == 3 and rows[2]["kept"] is False
    assert rows[2]["delta"] == -1                               # bi_rank 2 -> rerank 3

def test_rank_change_no_reorder_all_zero_delta():
    rows = rank_change_table([1, 2, 3], [0.9, 0.8, 0.7], [0.9, 0.8, 0.7], top_k=3)
    assert all(r["delta"] == 0 for r in rows)
    assert all(r["kept"] for r in rows)

def test_rank_change_length_mismatch_raises():
    with pytest.raises(ValueError):
        rank_change_table([1, 2], [0.1], [0.2, 0.3], top_k=1)
```

- [ ] **Step 2: Run to verify FAIL**

Run: `cd 05_rag/tools && python3 -m pytest test_rag_utils.py -k rank_change -v`  → FAIL (ImportError: cannot import name 'rank_change_table').

- [ ] **Step 3: Implement in `rag_utils.py`** (append at end)

```python
def rank_change_table(candidate_ids, bi_scores, rerank_scores, top_k):
    """Build a bi-encoder -> cross-encoder rank-change table.

    candidate_ids: doc ids in BI-ENCODER order (best first), length N, unique.
    bi_scores:     bi-encoder scores parallel to candidate_ids.
    rerank_scores: cross-encoder scores parallel to candidate_ids.
    top_k:         number kept after reranking.

    Returns a list of dicts, one per candidate, SORTED by rerank rank (best first):
      {doc_id, bi_rank, rerank_rank, delta, bi_score, rerank_score, kept}
    where ranks are 1-based, delta = bi_rank - rerank_rank (>0 = promoted),
    kept = rerank_rank <= top_k.
    """
    n = len(candidate_ids)
    if not (len(bi_scores) == len(rerank_scores) == n):
        raise ValueError("candidate_ids, bi_scores, rerank_scores must be the same length")
    bi_rank = {cid: i + 1 for i, cid in enumerate(candidate_ids)}
    order = sorted(range(n), key=lambda i: rerank_scores[i], reverse=True)
    rows = []
    for rerank_pos, i in enumerate(order, start=1):
        cid = candidate_ids[i]
        rows.append({
            "doc_id": cid,
            "bi_rank": bi_rank[cid],
            "rerank_rank": rerank_pos,
            "delta": bi_rank[cid] - rerank_pos,
            "bi_score": bi_scores[i],
            "rerank_score": rerank_scores[i],
            "kept": rerank_pos <= top_k,
        })
    return rows
```

- [ ] **Step 4: Run to verify PASS**

Run: `cd 05_rag/tools && python3 -m pytest test_rag_utils.py -v` → all pass (15 total: existing 12 + 3 new).

- [ ] **Step 5: Commit**

```bash
git add 05_rag/tools/rag_utils.py 05_rag/tools/test_rag_utils.py
git commit -m "feat(module05): add tested rank_change_table helper for reranking (nb03)"
```

---

## Task 2: validator entry for nb03

**Files:** Modify `05_rag/tools/validate_notebooks.py`

- [ ] **Step 1:** Add a registry entry for `"03_retrieve_better_rerank.ipynb"` with `markers` = `["paraphrase-multilingual-MiniLM-L12-v2", "bge-reranker-v2-m3", "rank_change_table", "apply_chat_template"]` and `forbidden` = `[]`. Inline comments: bi-encoder model / cross-encoder reranker / tested helper / grounded generation. Keep nb01 & nb02 entries unchanged.

- [ ] **Step 2:** Run against the not-yet-built nb03 to confirm the gate fires:
  Run: `cd 05_rag/tools && python3 validate_notebooks.py 03_retrieve_better_rerank.ipynb` → `[FAIL]` (file not found is acceptable as the "not yet built" signal; once the notebook exists it must report MISSING markers until Task 3 lands). nb01 & nb02 must still PASS.

- [ ] **Step 3: Commit**

```bash
git add 05_rag/tools/validate_notebooks.py
git commit -m "feat(module05): add nb03 validator markers (bi-encoder, reranker, rank_change_table, generation)"
```

---

## Task 3: build `03_retrieve_better_rerank.ipynb`

**Files:** Create `05_rag/03_retrieve_better_rerank.ipynb`

Self-contained; markdown Bahasa Indonesia, code/comments English. **Cell sequence (M/C):**

1. `M` — Tujuan: retrieval satu tahap (dense/bi-encoder) cepat tapi kasar — bagus untuk *recall*, lemah untuk *precision* di posisi teratas. Solusi: dua tahap — over-fetch banyak kandidat lalu *rerank* dengan cross-encoder yang membaca (query, passage) bersama.
2. `C` — Install: `!pip install -q "transformers<5" sentence-transformers faiss-cpu accelerate bitsandbytes`
3. `C` — Bootstrap (clone repo + `from tools.rag_utils import rank_change_table`):
```python
import os, sys
REPO = "navasena-gen-ml-course"
if not os.path.exists(REPO):
    !git clone --depth 1 https://github.com/chmdznr/navasena-gen-ml-course.git
sys.path.append(os.path.abspath(f"{REPO}/05_rag"))
from tools.rag_utils import rank_change_table
```
4. `M` — Bi-encoder vs cross-encoder: bi-encoder meng-embed query & dokumen TERPISAH (cepat, bisa pra-hitung, tapi tak melihat interaksi); cross-encoder memproses (query, passage) BERSAMA (akurat, tapi mahal — tak bisa pra-hitung). Maka: bi-encoder untuk menyaring banyak→sedikit, cross-encoder untuk mengurutkan ulang yang sedikit.
5. `C` — Corpus (~24 passage Bahasa Indonesia, sertakan pasangan mirip-leksikal yang menjebak dense, mis. "Komodo (hewan/kadal terbesar)" vs "Pulau Komodo (taman nasional)"; "Bromo gunung berapi" vs "wisata Bromo sunrise"; dll). Each is a short factual sentence. Store as `corpus = [...]`.
6. `C` — Embed corpus with `SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")` → FAISS `IndexFlatL2` (note: cosine/IndexFlatIP diperkenalkan di nb05; di sini L2 cukup untuk tahap penyaringan).
7. `M` — Tahap 1: over-fetch. Ambil lebih banyak kandidat dari yang dibutuhkan (mis. k=12) supaya jawaban benar kemungkinan besar ADA di kandidat walau belum di posisi #1.
8. `C` — Designed query (mis. `query = "Hewan kadal terbesar di dunia ada di mana?"`) → bi-encoder retrieve top `K=12`: return candidate ids + bi distances. Convert L2 distance to a similarity-ish `bi_score` (e.g. `-distance` or `1/(1+distance)`) for display. Print the 12 candidates with bi rank.
9. `M` — Tahap 2: rerank dengan cross-encoder.
10. `C` — Load reranker + score, build table + keep top-3:
```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
pairs = [(query, corpus[cid]) for cid in candidate_ids]
rerank_scores = reranker.predict(pairs).tolist()
rows = rank_change_table(candidate_ids, bi_scores, rerank_scores, top_k=3)
top3 = [r for r in rows if r["kept"]]
```
11. `M` — Membaca rank-change: kolom mana naik/turun, dan mengapa cross-encoder bisa mempromosikan passage yang benar yang tadinya tertimbun.
12. `C` — Signature visual: (a) print rank-change table (doc_id, bi_rank→rerank_rank, delta, kept, snippet); (b) matplotlib scatter `bi_score` vs `rerank_score` (korelasi lemah = reranker menambah sinyal baru); (c) bar chart `delta` per doc_id (naik=positif). All from `rows`.
13. `M` — Biaya: cross-encoder mahal — di-skalakan dengan jumlah kandidat. Itulah kenapa kita TIDAK rerank seluruh korpus, hanya hasil over-fetch.
14. `C` — Latency decomposition: untuk `n in [5,10,20,40]` ukur waktu Tahap-1 (embed query + FAISS search) vs Tahap-2 (rerank n pairs) dengan `time.perf_counter()`; plot dua garis. Tegaskan: Tahap-2 tumbuh ~linear dengan n, Tahap-1 ~datar.
15. `M` — Tutup loop: beri konteks ter-rerank ke generator.
16. `C` — Grounded generation: load Qwen2.5-3B-Instruct (4-bit, `BitsAndBytesConfig` fp16 compute), build context dari `top3` passages dengan id, `apply_chat_template`, `max_new_tokens=160`, `do_sample=False`; print jawaban + sitasi doc_id.
17. `M` — Ringkasan + jembatan ke nb04 (Measure/RAGAS): "reranking menaikkan context precision — di nb04 kita UKUR efeknya dengan RAGAS, pakai pertanyaan yang sama."

**Validator markers required in the notebook:** `paraphrase-multilingual-MiniLM-L12-v2`, `bge-reranker-v2-m3`, `rank_change_table`, `apply_chat_template`.

**Steps:**
- [ ] **Step 1:** Author the notebook cell-by-cell per the sequence (NotebookEdit insert).
- [ ] **Step 2:** Valid JSON: `python3 -c "import json; json.load(open('05_rag/03_retrieve_better_rerank.ipynb')); print('valid json')"`.
- [ ] **Step 3:** Validator: `cd 05_rag/tools && python3 validate_notebooks.py 03_retrieve_better_rerank.ipynb` → `[PASS]`. Also confirm nb01+nb02 still PASS.
- [ ] **Step 4 (manual GPU gate):** Colab T4 Run all — bi-encoder retrieves 12, reranker reorders (rank-change table shows non-zero deltas), 3 plots render, latency stage-2 grows, Qwen answers with citations. (Cannot run locally.)
- [ ] **Step 5: Commit**
```bash
git add 05_rag/03_retrieve_better_rerank.ipynb
git commit -m "feat(module05): add nb03 retrieve-better — bi-encoder over-fetch + bge cross-encoder rerank + rank-change viz + grounded answer"
```

---

## Self-Review
- Spec §5 nb03 coverage: over-fetch + cross-encoder rerank (Tasks 3 cells 7-10) ✅; rank-change table + scatter + latency (cells 12,14, Task 1 helper) ✅; grounded answer with sources (cell 16) ✅.
- Placeholder scan: the designed query/corpus content is authored in Task 3 (no TBD). Exact runtime rank deltas depend on the live models — flagged as a manual-gate observation, not a placeholder.
- Type consistency: `rank_change_table(candidate_ids, bi_scores, rerank_scores, top_k)` returns dicts with keys `doc_id/bi_rank/rerank_rank/delta/bi_score/rerank_score/kept`; the notebook (cells 10,12) consumes exactly those keys. Models `paraphrase-multilingual-MiniLM-L12-v2`, `BAAI/bge-reranker-v2-m3`, `Qwen/Qwen2.5-3B-Instruct` consistent with nb01/nb02 + validator markers.
- Risk: `CrossEncoder("BAAI/bge-reranker-v2-m3")` load — if a Colab sentence-transformers version mishandles this reranker arch, fall back to `FlagEmbedding.FlagReranker`; note in the cell as a comment but try CrossEncoder first (non-gated, ~568M, T4-fine).
