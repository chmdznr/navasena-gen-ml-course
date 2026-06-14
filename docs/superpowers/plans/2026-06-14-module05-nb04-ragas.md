# Module 05 nb04 — Measure (RAGAS) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** Add `04_measure_ragas.ipynb` that QUANTIFIES the reranking effect from nb03 using the RAGAS library with a **cloud judge** (NVIDIA NIM default, free) + **local multilingual embeddings**, comparing two RAG variants (bi-encoder-only top-3 vs bi-encoder+cross-encoder-rerank top-3) on the same Indonesian employee-handbook corpus.

**Architecture:** Self-contained Colab notebook. The RAG pipeline's GENERATOR stays LOCAL (Qwen2.5-3B 4-bit); only the RAGAS JUDGE is cloud (Llama-3.1-8B via NIM → cross-family judge avoids self-preference bias, uses 0 VRAM). RAGAS embeddings are LOCAL `paraphrase-multilingual-MiniLM-L12-v2`. API key read from Colab Secrets (never hardcoded). Grounded in CURRENT RAGAS 0.4.x API (fields `user_input/response/retrieved_contexts/reference`; `EvaluationDataset.from_list`; metrics `Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference, LLMContextRecall`; `evaluate(dataset, metrics, llm=, embeddings=, run_config=)`).

**Tech Stack:** `ragas>=0.3,<0.5`, langchain-nvidia-ai-endpoints, langchain-huggingface, nest-asyncio, sentence-transformers, faiss-cpu, Qwen2.5-3B-Instruct 4-bit, pandas, matplotlib. Markdown Bahasa Indonesia; code/comments English.

**Reference spec:** `docs/superpowers/specs/2026-06-14-module05-rag-rework-design.md` §5 (nb04). NOTE: spec said "local Qwen judge"; this plan uses a **cloud cross-family judge** (user decision) — stronger (anti self-preference) + 0 VRAM. Spec + companion assets to resync during the final companion pass.

---

## File Structure

| Path | Responsibility |
|---|---|
| `05_rag/tools/validate_notebooks.py` | MODIFY — add `04_measure_ragas.ipynb` registry entry. |
| `05_rag/04_measure_ragas.ipynb` | CREATE — the RAGAS measurement notebook. |

No new `rag_utils` helper (metrics come from RAGAS; retrieve/generate are notebook-local, reranking reuses `rank_change_table` from nb03).

---

## Task 1: validator entry for nb04

**Files:** Modify `05_rag/tools/validate_notebooks.py`

- [ ] **Step 1:** Add a registry entry for `"04_measure_ragas.ipynb"`:
```python
    "04_measure_ragas.ipynb": {
        "markers": [
            "EvaluationDataset",                    # modern RAGAS dataset
            "LangchainLLMWrapper",                  # cloud judge wrapped for RAGAS
            "LLMContextPrecisionWithReference",     # the rerank-sensitive headline metric
            "Faithfulness",                         # grounding metric
            "NVIDIA_API_KEY",                       # key read from Colab Secrets (not hardcoded)
            "paraphrase-multilingual-MiniLM-L12-v2",# local RAGAS embeddings + bi-encoder
        ],
        "forbidden": [
            "ragas.metrics import context_precision",  # guard against legacy 0.1 lowercase API
        ],
    },
```
Keep nb01/02/03 entries unchanged. Leave the `# later plans extend...` comment updated to nb05..nb08.

- [ ] **Step 2:** Run against the not-yet-built nb04: `cd 05_rag/tools && python3 validate_notebooks.py 04_measure_ragas.ipynb` → `[FAIL]` (file not found). nb01/02/03 still PASS. Paste stdout.

- [ ] **Step 3: Commit**
```bash
git add 05_rag/tools/validate_notebooks.py
git commit -m "feat(module05): add nb04 validator markers (RAGAS EvaluationDataset, cloud judge, context-precision, key-from-secrets)"
```

---

## Task 2: build `04_measure_ragas.ipynb`

**Files:** Create `05_rag/04_measure_ragas.ipynb`. Self-contained; markdown Bahasa Indonesia, code/comments English. **Cell sequence:**

1. `M` — Tujuan: nb03 MENUNJUKKAN rerank menggeser passage off-topic keluar top-3 secara visual; nb04 MEMBUKTIKAN-nya dengan ANGKA. Tanpa ukur = menebak. Eksperimen: korpus+pertanyaan+generator SAMA seperti nb03; satu-satunya yang berubah = `retrieved_contexts` (tanpa-rerank vs dengan-rerank). Inti: judge = model CLOUD (gratis), generator = Qwen LOKAL.

2. `M` — 4 metrik RAGAS (ramah pemula, tanpa rumus): (1) **Faithfulness** — jawaban bersandar konteks (anti-halusinasi); (2) **Response Relevancy** — jawaban nyambung ke pertanyaan (pakai embedding); (3) **Context Precision** — proporsi potongan retrieved yang relevan & di posisi atas (← yang diperbaiki reranker); (4) **Context Recall** — potongan pembawa jawaban ikut ter-retrieve. Tabel: metrik | butuh judge? | butuh embedding? | butuh ground_truth?. Tegaskan 2 dari 4 butuh `reference`. Skala 0..1, makin tinggi makin baik.

3. `C` — Install:
```python
!pip install -q "transformers<5" "sentence-transformers>=3.0" faiss-cpu accelerate bitsandbytes \
  "ragas>=0.3,<0.5" langchain-core langchain-nvidia-ai-endpoints langchain-huggingface nest-asyncio
```
Then bootstrap (clone repo if missing, `sys.path.append(.../05_rag)`, `from tools.rag_utils import rank_change_table`).

4. `M` — Judge CLOUD, generator LOKAL — kenapa bagus: judge lewat API = 0 VRAM T4; yang di T4 cuma Qwen 4-bit + MiniLM + reranker. Dua untung: (a) hemat VRAM; (b) **anti self-preference bias** — judge Llama (NIM) menilai jawaban Qwen, beda keluarga model. Jujur: judge cloud butuh API key GRATIS (cara dapat di sel berikut). Cara aman: Colab Secrets (ikon 🔑), Name `NVIDIA_API_KEY`, toggle Notebook access ON. JANGAN hardcode key.

5. `C` — Judge + embeddings + key (verified API):
```python
import os
try:
    from google.colab import userdata
    os.environ["NVIDIA_API_KEY"] = userdata.get("NVIDIA_API_KEY")   # from Colab Secrets
except Exception:
    import getpass
    os.environ["NVIDIA_API_KEY"] = getpass.getpass("NVAPI key (nvapi-...): ")
assert os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"), "Key harus diawali 'nvapi-' (dapat gratis di build.nvidia.com)"

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from ragas.llms import LangchainLLMWrapper
judge = ChatNVIDIA(model="meta/llama-3.1-8b-instruct", temperature=0)   # ChatNVIDIA.get_available_models() to list valid IDs
evaluator_llm = LangchainLLMWrapper(judge)

# --- SWAP: Google Gemini (free, aistudio.google.com/apikey) ---
# !pip install -q langchain-google-genai ; os.environ["GOOGLE_API_KEY"]=userdata.get("GOOGLE_API_KEY")
# from langchain_google_genai import ChatGoogleGenerativeAI
# judge = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0); evaluator_llm = LangchainLLMWrapper(judge)
# --- SWAP: OpenAI (paid) ---
# from langchain_openai import ChatOpenAI; judge = ChatOpenAI(model="gpt-4o-mini", temperature=0); evaluator_llm = LangchainLLMWrapper(judge)

from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
hf_emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True})
evaluator_embeddings = LangchainEmbeddingsWrapper(hf_emb)

import nest_asyncio; nest_asyncio.apply()   # Colab event-loop belt-and-suspenders
```

6. `C` — Rebuild nb03 pipeline (recap, one cell): the same 12-passage Indonesian handbook `corpus`, `SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")` + `faiss.IndexFlatL2`, `CrossEncoder("BAAI/bge-reranker-v2-m3")`, and Qwen2.5-3B-Instruct 4-bit (`BitsAndBytesConfig` NF4, `bnb_4bit_compute_dtype=torch.float16` NOT bf16, `device_map="auto"`). Comment: identical pipeline to nb03 — the only experimental variable downstream is rerank on/off.

7. `C` — Test set (~7 questions + reference). Parallel lists `questions` and `references`, answerable from corpus (cuti tahunan→"dua belas hari", cuti sakit→"sepuluh hari", cuti melahirkan→"tiga bulan", jam kerja→"09.00", gaji→"tanggal 25", WFH→"dua hari/minggu", lembur hari libur→"dua kali lipat"). Markdown note: sengaja KECIL (~7) untuk hormati limit free-tier judge (NIM ~40 RPM) + runtime — tiap pertanyaan memicu banyak panggilan judge per metrik.

8. `C` — Helpers (the only difference between variants is rerank):
```python
def retrieve(query, use_rerank):
    qv = embedder.encode([query], convert_to_numpy=True).astype("float32")
    if not use_rerank:
        _, idx = index.search(qv, 3)
        return [corpus[i] for i in idx[0]]
    _, idx = index.search(qv, 12)                      # over-fetch
    cand = idx[0].tolist()
    scores = reranker.predict([(query, corpus[c]) for c in cand]).tolist()
    order = sorted(range(len(cand)), key=lambda i: scores[i], reverse=True)[:3]
    return [corpus[cand[i]] for i in order]

def generate(query, contexts):
    ctx = "\n".join(f"[doc] {c}" for c in contexts)
    messages = [{"role": "system", "content": "Jawab HANYA berdasarkan konteks. Jika tak ada, katакан tidak tahu."},
                {"role": "user", "content": f"Konteks:\n{ctx}\n\nPertanyaan: {query}"}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = tok(prompt, return_tensors="pt").to(gen.device)
    out = gen.generate(**inp, max_new_tokens=160, do_sample=False)
    return tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
```
Markdown note: kita REGENERATE jawaban Qwen per varian (konteks lebih baik bisa mengubah jawaban) → Faithfulness/Relevancy menangkap efek end-to-end.

9. `C` — Phase 1: build two EvaluationDatasets:
```python
from ragas import EvaluationDataset
rows_A, rows_B = [], []
for q, ref in zip(questions, references):
    ctxA = retrieve(q, use_rerank=False); ctxB = retrieve(q, use_rerank=True)
    rows_A.append({"user_input": q, "reference": ref, "retrieved_contexts": ctxA, "response": generate(q, ctxA)})
    rows_B.append({"user_input": q, "reference": ref, "retrieved_contexts": ctxB, "response": generate(q, ctxB)})
ds_A = EvaluationDataset.from_list(rows_A)
ds_B = EvaluationDataset.from_list(rows_B)
print("Contoh — konteks A vs B untuk Q0:"); print("A:", rows_A[0]["retrieved_contexts"]); print("B:", rows_B[0]["retrieved_contexts"])
```

10. `C` — Phase 2: RAGAS evaluate both (identical judge/metrics/config):
```python
from ragas import evaluate
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference, LLMContextRecall
from ragas.run_config import RunConfig
metrics = [LLMContextPrecisionWithReference(), LLMContextRecall(), Faithfulness(), ResponseRelevancy()]
rc = RunConfig(max_workers=4, timeout=120)   # fewer workers = fewer 429 on free tier
res_A = evaluate(dataset=ds_A, metrics=metrics, llm=evaluator_llm, embeddings=evaluator_embeddings, run_config=rc)
res_B = evaluate(dataset=ds_B, metrics=metrics, llm=evaluator_llm, embeddings=evaluator_embeddings, run_config=rc)
```
Markdown note: notebook TERBERAT — puluhan panggilan judge per pertanyaan × 2 varian; wajar beberapa menit. Kalau muncul error per-baris, ulangi dengan `evaluate(..., raise_exceptions=True)` untuk debug.

11. `M` — Menafsirkan skor: ≥0.7 baik / 0.5–0.7 perlu perhatian / <0.5 buruk. Per metrik: low Faithfulness=halusinasi; low Context Precision=banyak noise di top-k (← rerank perbaiki); low Context Recall=potongan jawaban tak ter-retrieve (chunking/embedding); low Relevancy=jawaban melenceng. Jangan kejar 1.0 buta.

12. `C` — Comparison table + bar chart (robust to column drift + NaN):
```python
import pandas as pd, matplotlib.pyplot as plt
df_A, df_B = res_A.to_pandas(), res_B.to_pandas()
print("Kolom hasil:", list(df_A.columns))
meta = {"user_input", "reference", "retrieved_contexts", "response"}
metric_cols = [c for c in df_A.columns if c not in meta]
print("NaN per metrik (A):", df_A[metric_cols].isna().sum().to_dict(), "| (B):", df_B[metric_cols].isna().sum().to_dict())
summary = pd.DataFrame({"bi-encoder top-3": df_A[metric_cols].mean(),
                        "bi+rerank top-3": df_B[metric_cols].mean()})
summary["delta"] = summary["bi+rerank top-3"] - summary["bi-encoder top-3"]
print(summary.round(3))
summary[["bi-encoder top-3", "bi+rerank top-3"]].plot.bar(figsize=(9,4))
plt.ylabel("skor (0..1)"); plt.title("RAGAS: tanpa-rerank vs dengan-rerank"); plt.xticks(rotation=20); plt.tight_layout(); plt.show()
```
Markdown note: `mean()` melewati NaN — makanya kita CETAK jumlah NaN dulu; di set ~7 baris, 1 NaN bisa menggeser rata-rata banyak.

13. `C` — Worst-question triage: merge df_A & df_B on `user_input`, compute per-question context-precision delta, show the question where rerank helped most (print its A-contexts vs B-contexts so the off-topic chunk pushed out is visible); also show the lowest-Faithfulness row (its answer + context = "kalimat yang tak didukung konteks").

14. `M` — Caveat bias judge: LLM-judge bisa menyukai jawaban segaya/sekeluarga dirinya. Karena judge (Llama/NIM) BEDA keluarga dari generator (Qwen), kita hindari self-preference terburuk. Caveat lain: set kecil (~7) = rata-rata berisik; judge juga LLM, bisa salah-parse (→ NaN); limit free-tier; hasil bergantung korpus/pertanyaan ini. RAGAS itu kompas, bukan vonis.

15. `M` — Ringkasan + jembatan ke nb05: kita UKUR efek rerank (context precision naik). nb03 membuat retrieval lebih baik, nb04 membuktikannya. Tapi index kita masih L2 mentah & hanya di RAM — nb05 perbaiki cosine (`IndexFlatIP`+normalize) & tambah persistence (save/load FAISS, ChromaDB). PR opsional: ganti judge ke Gemini, lihat apakah skor bergeser (demo varians judge).

**Validator markers required in notebook:** `EvaluationDataset`, `LangchainLLMWrapper`, `LLMContextPrecisionWithReference`, `Faithfulness`, `NVIDIA_API_KEY`, `paraphrase-multilingual-MiniLM-L12-v2`. **Forbidden:** must NOT contain a hardcoded `nvapi-` key (the controller will grep the committed notebook for the real key before merge).

**Steps:**
- [ ] **Step 1:** Author the notebook cell-by-cell (NotebookEdit insert).
- [ ] **Step 2:** Valid JSON: `python3 -c "import json; json.load(open('05_rag/04_measure_ragas.ipynb')); print('valid json')"`.
- [ ] **Step 3:** Validator: `cd 05_rag/tools && python3 validate_notebooks.py 04_measure_ragas.ipynb` → `[PASS]`. nb01/02/03 still PASS.
- [ ] **Step 4 (key safety):** `python3 -c "import json; s=''.join(''.join(c['source']) for c in json.load(open('05_rag/04_measure_ragas.ipynb'))['cells']); import re; print('hardcoded nvapi keys:', len(re.findall(r'nvapi-[A-Za-z0-9_\-]{20,}', s)))"` → MUST be 0 (only the `startswith('nvapi-')` literal allowed).
- [ ] **Step 5 (manual GPU+API gate):** Colab T4 + a free NVIDIA key in Secrets, Run all. (Cannot run locally — no key, no GPU.)
- [ ] **Step 6: Commit**
```bash
git add 05_rag/04_measure_ragas.ipynb
git commit -m "feat(module05): add nb04 measure — RAGAS (cloud NIM judge + local embeddings) comparing rerank on/off on the handbook corpus"
```

---

## Self-Review
- Spec §5 nb04 coverage: 4 RAGAS metrics ✅; thresholds ✅ (cell 11); worst-question triage ✅ (cell 13); self-preference-bias caveat ✅ (cell 14, reframed for cross-family cloud judge). Divergence from spec's "local judge" documented above; companion-asset resync deferred.
- Placeholders: test-set questions/references authored in Task 2 cell 7 (no TBD). Exact RAGAS `to_pandas()` column names derived at runtime (`metric_cols`), not hardcoded — handles version drift.
- Type/name consistency: dataset fields `user_input/response/retrieved_contexts/reference` used in cell 9 and consumed by `metric_cols` exclusion in cell 12; `retrieve(query, use_rerank)`/`generate(query, contexts)` defined cell 8, used cell 9; `evaluator_llm`/`evaluator_embeddings` defined cell 5, used cell 10. Models `meta/llama-3.1-8b-instruct` (judge), `paraphrase-multilingual-MiniLM-L12-v2` (embeddings+bi-encoder), `BAAI/bge-reranker-v2-m3`, `Qwen/Qwen2.5-3B-Instruct` consistent with nb03 + validator markers.
- KEY SAFETY: notebook reads `NVIDIA_API_KEY` from Colab Secrets/getpass; never hardcoded; validator step 4 + controller grep enforce this before merge.
- Risk (carried from research): NaN scores (print counts + raise_exceptions toggle), free-tier 429 (max_workers=4, small set), model-ID drift (get_available_models comment), small-corpus small-delta (test set includes synonym-answer questions; framed honestly).
