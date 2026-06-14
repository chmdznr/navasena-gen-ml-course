# Module 05 nb05 — Scale the Index Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** Add `05_scale_the_index.ipynb` that pays off the deferred cosine fix (IndexFlatIP + normalize_L2), benchmarks approximate FAISS indexes (Flat/IVF/HNSW) on a synthetic large corpus with real numbers, and adds persistence + metadata filtering via ChromaDB — capped by a FAISS-vs-Chroma decision matrix.

**Architecture:** Self-contained Colab notebook, CPU-only (faiss-cpu + chromadb, no GPU, runs <1 min). Reuses the 12-passage Indonesian employee-handbook corpus (cosine-fix + Chroma sections) and a synthetic `make_blobs(centers=50)` corpus (N=20000, d=384) for the index benchmark. Embeddings = paraphrase-multilingual-MiniLM-L12-v2 (consistent with nb01-04). Pure-logic `recall_at_k` lives in `rag_utils.py` (unit-tested). Grounded in CURRENT verified APIs: chromadb 1.5.x (PersistentClient auto-persist, `configuration={'hnsw':{'space':'cosine'}}`, `embedding_function=None`), FAISS normalize_L2+IndexFlatIP.

**Tech Stack:** faiss-cpu, `chromadb>=1.0,<2`, sentence-transformers (MiniLM multilingual), scikit-learn (make_blobs), numpy, pandas, matplotlib. Markdown Bahasa Indonesia; code/comments English.

**Reference spec:** `docs/superpowers/specs/2026-06-14-module05-rag-rework-design.md` §5 (nb05).

---

## File Structure

| Path | Responsibility |
|---|---|
| `05_rag/tools/rag_utils.py` | MODIFY — add `recall_at_k(approx_ids, gt_ids)` (pure, GPU-free). |
| `05_rag/tools/test_rag_utils.py` | MODIFY — add tests for `recall_at_k`. |
| `05_rag/tools/validate_notebooks.py` | MODIFY — add `05_scale_the_index.ipynb` registry entry. |
| `05_rag/05_scale_the_index.ipynb` | CREATE — the cosine/benchmark/persistence notebook. |

---

## Task 1: `recall_at_k` helper (TDD)

**Files:** Modify `05_rag/tools/rag_utils.py`, `05_rag/tools/test_rag_utils.py`

- [ ] **Step 1: Append tests to `test_rag_utils.py`**
```python
from rag_utils import recall_at_k

def test_recall_at_k_perfect():
    assert recall_at_k([[1, 2, 3]], [[1, 2, 3]]) == 1.0

def test_recall_at_k_partial():
    assert recall_at_k([[1, 2, 9]], [[1, 2, 3]]) == 2 / 3   # 2 of 3 ground-truth found

def test_recall_at_k_mean_over_queries():
    # q1 perfect (1.0), q2 finds 1 of 2 (0.5) -> mean 0.75
    assert recall_at_k([[1, 2], [5, 6]], [[1, 2], [5, 9]]) == 0.75

def test_recall_at_k_empty():
    assert recall_at_k([], []) == 0.0
```

- [ ] **Step 2: Run to verify FAIL** — `cd 05_rag/tools && python3 -m pytest test_rag_utils.py -k recall -v` → ImportError.

- [ ] **Step 3: Implement (append to `rag_utils.py`)**
```python
def recall_at_k(approx_ids, gt_ids):
    """Mean recall@k of approximate retrieval vs exact ground truth.

    approx_ids, gt_ids: parallel sequences (n_queries x k) of retrieved row ids
    (lists or numpy rows). For each query, recall = |approx ∩ gt| / |gt|.
    Returns the mean over queries, or 0.0 if there are no queries.
    """
    recalls = []
    for approx, gt in zip(approx_ids, gt_ids):
        gset = set(int(x) for x in gt)
        if not gset:
            continue
        hits = sum(1 for x in approx if int(x) in gset)
        recalls.append(hits / len(gset))
    return sum(recalls) / len(recalls) if recalls else 0.0
```

- [ ] **Step 4: Run to verify PASS** — `cd 05_rag/tools && python3 -m pytest test_rag_utils.py -v` → all pass (existing + 4 new). Paste summary.

- [ ] **Step 5: Commit**
```bash
git add 05_rag/tools/rag_utils.py 05_rag/tools/test_rag_utils.py
git commit -m "feat(module05): add tested recall_at_k helper for nb05 index benchmark"
```

---

## Task 2: validator entry for nb05

**Files:** Modify `05_rag/tools/validate_notebooks.py`

- [ ] **Step 1:** Add `"05_scale_the_index.ipynb"` to REGISTRY (after nb04; do not change others):
```python
    "05_scale_the_index.ipynb": {
        "markers": [
            "IndexFlatIP",            # cosine via inner product
            "normalize_L2",           # L2-normalize for cosine
            "write_index",            # FAISS persistence
            "IndexHNSWFlat",          # approximate-index taxonomy
            "recall_at_k",            # tested benchmark helper
            "PersistentClient",       # ChromaDB persistence
            "embedding_function=None",# bring-your-own-vectors (avoids ONNX auto-download)
        ],
        "forbidden": [
            "client.persist()",       # removed in chromadb 0.4+ (AttributeError); auto-persist now
        ],
    },
```
Update the trailing comment to "nb06..nb08".

- [ ] **Step 2:** `cd 05_rag/tools && python3 validate_notebooks.py 05_scale_the_index.ipynb` → `[FAIL]` (file not found). nb01-04 still PASS. Paste stdout.

- [ ] **Step 3: Commit**
```bash
git add 05_rag/tools/validate_notebooks.py
git commit -m "feat(module05): add nb05 validator markers (cosine, FAISS taxonomy, Chroma persistence)"
```

---

## Task 3: build `05_scale_the_index.ipynb`

**Files:** Create `05_rag/05_scale_the_index.ipynb`. Self-contained, CPU-only; markdown Bahasa Indonesia, code/comments English. **Cell sequence (verified APIs):**

1. `M` — Judul + 3 janji dari jembatan nb04: (1) ganti L2→cosine, (2) pahami trade-off index aproksimasi di skala besar (benchmark), (3) tambah persistence + metadata filtering (ChromaDB). Tegaskan CPU-only, cepat (<1 menit), tanpa GPU.
2. `C` — Install + bootstrap:
```python
!pip install -q "transformers<5" "sentence-transformers>=3.0" faiss-cpu "chromadb>=1.0,<2" scikit-learn
```
then clone repo + `from tools.rag_utils import recall_at_k`.
3. `M` — Masalah warisan: L2 vs cosine (cosine = arah vektor, cocok untuk teks) + index masih di RAM. Intuisi panah searah = mirip.
4. `C` — Cosine fix on handbook corpus. Reuse the 12-passage handbook `corpus` inline (from nb03). Then:
```python
import faiss, numpy as np
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vecs = embedder.encode(corpus, convert_to_numpy=True).astype("float32")
dim = vecs.shape[1]
# L2 (lama)
l2 = faiss.IndexFlatL2(dim); l2.add(vecs)
# Cosine: normalize (IN-PLACE, returns None -> copy first!) + inner product
vecs_n = vecs.copy(); faiss.normalize_L2(vecs_n)
ip = faiss.IndexFlatIP(dim); ip.add(vecs_n)
query = "Berapa hari cuti tahunan pegawai tetap?"
qv = embedder.encode([query], convert_to_numpy=True).astype("float32")
dl2, il2 = l2.search(qv, 3)
qn = qv.copy(); faiss.normalize_L2(qn)          # normalize the QUERY too
dip, iip = ip.search(qn, 3)
print("L2 (jarak, kecil=mirip):", [(int(i), float(d)) for i, d in zip(il2[0], dl2[0])])
print("Cosine (sim, besar=mirip):", [(int(i), float(d)) for i, d in zip(iip[0], dip[0])])
```
Print side-by-side ranking; explain cosine is the standard for text.
5. `M` — Menyimpan index ke disk: FAISS write/read + keterbatasan jujur (FAISS hanya simpan vektor+row-id int64, BUKAN teks/metadata → butuh sidecar JSON). Foreshadow ChromaDB. Catatan keamanan: hanya `read_index` file yang Anda buat/percaya.
6. `C` — FAISS save + JSON sidecar + reload:
```python
import json
faiss.write_index(ip, "/content/handbook.index")
sidecar = {"id_map": [f"doc_{i}" for i in range(len(corpus))],
           "texts": {f"doc_{i}": corpus[i] for i in range(len(corpus))},
           "index_type": "FlatIP", "dim": dim, "metric": "cosine", "normalized": True}
json.dump(sidecar, open("/content/handbook.sidecar.json", "w"))
idx2 = faiss.read_index("/content/handbook.index")
meta = json.load(open("/content/handbook.sidecar.json"))
assert idx2.ntotal == len(corpus)
_, I = idx2.search(qn, 3)
print([(meta["id_map"][i], meta["texts"][meta["id_map"][i]][:40]) for i in I[0]])
```
7. `M` — Skala besar: kenapa perlu index aproksimasi. Flat=exact/lambat (ground truth), IVF=bagi ruang jadi sel (nlist) cari sebagian (nprobe), HNSW=graf navigasi (M, efSearch), IVFPQ=kompresi (sebut singkat). Definisikan recall@k vs Flat. Jalankan di korpus SINTETIS (butuh banyak vektor).
8. `C` — Synthetic clustered corpus + ground truth:
```python
import time, numpy as np, faiss
from sklearn.datasets import make_blobs
N, d = 20000, 384                      # bump to 50000 jika mau; tetap <1 menit CPU
xb, _ = make_blobs(n_samples=N, n_features=d, centers=50, random_state=42)
xb = xb.astype("float32"); faiss.normalize_L2(xb)        # cosine regime (normalized-L2 == cosine)
rng = np.random.default_rng(0)
xq = xb[rng.choice(N, 1000, replace=False)].copy()        # 1000 query dari korpus
gt = faiss.IndexFlatL2(d); gt.add(xb)
_, gtI = gt.search(xq, 10)                                 # ground truth (exact)
```
9. `C` — Benchmark Flat / IVFFlat / HNSW (build time, ms/query, recall@10):
```python
import pandas as pd
def bench(name, build):
    t0 = time.perf_counter(); index = build(); build_s = time.perf_counter() - t0
    index.search(xq[:10], 10)                              # warm-up
    t0 = time.perf_counter(); _, I = index.search(xq, 10); qms = (time.perf_counter()-t0)/len(xq)*1000
    return {"index": name, "build_s": round(build_s,3), "ms/query": round(qms,3),
            "recall@10": round(recall_at_k(I, gtI), 3)}

def make_flat():
    ix = faiss.IndexFlatL2(d); ix.add(xb); return ix
def make_ivf():
    nlist = int(N**0.5)                                    # ~141
    ix = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, nlist)
    ix.train(xb); ix.add(xb); ix.nprobe = 8; return ix
def make_hnsw():
    ix = faiss.IndexHNSWFlat(d, 32); ix.hnsw.efConstruction = 200
    ix.add(xb); ix.hnsw.efSearch = 64; return ix

df = pd.DataFrame([bench("Flat", make_flat), bench("IVFFlat", make_ivf), bench("HNSW", make_hnsw)])
print(df.to_string(index=False))
```
10. `C` — nprobe sweep → recall-vs-latency Pareto (matplotlib):
```python
import matplotlib.pyplot as plt
ivf = make_ivf()
xs, ys, labels = [], [], []
for nprobe in (1, 4, 8, 16, 32, 64):
    ivf.nprobe = nprobe
    ivf.search(xq[:10], 10)
    t0 = time.perf_counter(); _, I = ivf.search(xq, 10); ms = (time.perf_counter()-t0)/len(xq)*1000
    xs.append(ms); ys.append(recall_at_k(I, gtI)); labels.append(nprobe)
plt.plot(xs, ys, marker="o")
for x, y, n in zip(xs, ys, labels): plt.annotate(f"nprobe={n}", (x, y), fontsize=8)
plt.xlabel("ms/query"); plt.ylabel("recall@10"); plt.title("IVF: trade-off recall vs latensi (knob nprobe)")
plt.tight_layout(); plt.show()
```
Teach: nprobe↑ → recall↑ tapi latensi↑; HNSW knob-nya efSearch.
11. `M` — ChromaDB: persistence + metadata filtering yang FAISS tak punya. Kita beri vektor SENDIRI (Chroma tak embed sendiri). Flag API version-sensitive: `configuration={'hnsw':{'space':'cosine'}}`, `embedding_function=None`, TANPA `client.persist()`.
12. `C` — (Opsional) sqlite fallback note + import:
```python
# Colab modern (sqlite>=3.35) biasanya langsung jalan. Jika muncul RuntimeError 'unsupported sqlite3':
#   !pip install -q pysqlite3-binary
#   __import__("pysqlite3"); import sys; sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
#   (taruh SEBELUM import chromadb, lalu RESTART runtime)
import chromadb
from chromadb.config import Settings
```
13. `C` — PersistentClient + cosine collection + add handbook (precomputed embeddings + metadata):
```python
client = chromadb.PersistentClient(path="/content/chroma_db", settings=Settings(anonymized_telemetry=False))
col = client.get_or_create_collection(name="handbook",
        configuration={"hnsw": {"space": "cosine"}},   # cara 1.x (bukan metadata= lama)
        embedding_function=None)                        # None = pakai vektor kita, tak unduh ONNX
kategori = ["cuti","cuti","cuti","jam_kerja","cuti","jam_kerja","wfh","gaji","asuransi","lembur","etik","dinas"]
metas = [{"kategori": kategori[i], "doc_id": i} for i in range(len(corpus))]
embs = embedder.encode(corpus, normalize_embeddings=True).tolist()   # list[list[float]]
col.add(ids=[f"doc_{i}" for i in range(len(corpus))], documents=corpus, embeddings=embs, metadatas=metas)
print("jumlah dokumen:", col.count())                   # 12
```
(Adjust `kategori` to match the 12 handbook passages sensibly.)
14. `C` — Query with metadata WHERE filter (the FAISS-missing feature):
```python
q = "kebijakan cuti tahunan"
qv = embedder.encode([q], normalize_embeddings=True).tolist()
plain = col.query(query_embeddings=qv, n_results=3, include=["documents","metadatas","distances"])  # 'ids' jangan di include
print("Tanpa filter:", plain["ids"][0])
filtered = col.query(query_embeddings=qv, n_results=3, where={"kategori": "cuti"},
                     include=["documents","distances"])
print("Filter kategori=cuti:", filtered["ids"][0], filtered["distances"][0])  # cosine: kecil=mirip
```
Explain: metadata filtering ini yang FAISS mentah tak punya tanpa bookkeeping sendiri.
15. `C` — Prove persistence (reopen on same path):
```python
del client
client2 = chromadb.PersistentClient(path="/content/chroma_db")
print([c.name for c in client2.list_collections()])      # ['handbook']
col2 = client2.get_collection("handbook", embedding_function=None)
print("count setelah buka ulang:", col2.count())         # 12 (TANPA persist() di mana pun)
print("space:", col2.configuration["hnsw"]["space"])      # 'cosine'
```
16. `M` — Matriks keputusan FAISS vs ChromaDB (tabel): persistence (write_index+sidecar manual vs auto sqlite), metadata filter (tidak ada vs where bawaan), simpan teks (tidak vs ya), kontrol index/tuning (FAISS penuh IVF/HNSW/PQ vs Chroma HNSW di-manage), skala+memori (FAISS IVFPQ menang), prototyping cepat (Chroma menang). Verdict: FAISS untuk kontrol/skala/benchmark; Chroma untuk persistence+metadata+iterasi cepat.
17. `M` — Ringkasan (3 gap ditutup) + jembatan ke nb06 (Conversational RAG menambah memori multi-turn di atas retriever yang kini cosine, persistent, bisa difilter). PR opsional: N=50k, tambah IVFPQ (sumbu memori-vs-recall), coba d=768.

**Validator markers required:** `IndexFlatIP`, `normalize_L2`, `write_index`, `IndexHNSWFlat`, `recall_at_k`, `PersistentClient`, `embedding_function=None`. **Forbidden:** `client.persist()`.

**Steps:**
- [ ] **Step 1:** Author the notebook cell-by-cell (NotebookEdit insert).
- [ ] **Step 2:** Valid JSON: `python3 -c "import json; json.load(open('05_rag/05_scale_the_index.ipynb')); print('valid json')"`.
- [ ] **Step 3:** Validator: `cd 05_rag/tools && python3 validate_notebooks.py 05_scale_the_index.ipynb` → `[PASS]`. nb01-04 still PASS.
- [ ] **Step 4 (manual gate):** Colab CPU Run all — cosine vs L2 ranking prints; FAISS reload works; benchmark table shows Flat/IVF/HNSW with realistic recall (0.85-0.98); Pareto curve renders; Chroma add/query/where + reopen all work. (Local run also feasible since CPU-only — optional smoke.)
- [ ] **Step 5: Commit**
```bash
git add 05_rag/05_scale_the_index.ipynb
git commit -m "feat(module05): add nb05 scale-the-index — cosine fix, FAISS Flat/IVF/HNSW benchmark, ChromaDB persistence + metadata filtering"
```

---

## Self-Review
- Spec §5 nb05 coverage: cosine `IndexFlatIP`+normalize (cell 4) ✅; FAISS taxonomy + benchmark (cells 7-10) ✅; save/load + metadata sidecar (cell 6) ✅; Chroma PersistentClient + metadata filtering (cells 13-15) ✅; FAISS-vs-Chroma decision matrix (cell 16) ✅; latency benchmark (cells 9-10) ✅.
- Placeholders: synthetic sizes (N=20000, d=384), nlist=sqrt(N), HNSW M=32 all concrete. `kategori` list authored in cell 13.
- Type/name consistency: `recall_at_k(approx_ids, gt_ids)` returns float; consumed in cells 9-10. APIs match verified research: `faiss.normalize_L2` (in-place, copy first), `IndexFlatIP/IVFFlat/HNSWFlat`, `chromadb.PersistentClient`, `configuration={'hnsw':{'space':'cosine'}}`, `embedding_function=None`, no `client.persist()`. Embedder `paraphrase-multilingual-MiniLM-L12-v2` consistent with nb01-04.
- Risks (from research): chromadb pin `>=1.0,<2`; sqlite fallback documented (commented, pre-import, restart); benchmark MUST use `make_blobs(centers=50)` else recall pathologically low; `normalize_L2` in-place footgun (`.copy()`); legacy `client.persist()`/`metadata=` cosine flagged + forbidden marker; corpus duplicated from nb03 (comment points to source).
