# Module 05 (RAG) Rework — Design Spec

- **Date:** 2026-06-14
- **Status:** Approved (design); pending written-spec review → implementation plan
- **Module:** `05_rag/` (Navasena NCA-GENL course)
- **Sibling reference:** Module 04 LLM (8 notebooks, 49 frames/10 acts, 31 quiz Qs, validators in `04_llm/tools/`)

## 1. Context & Motivation

Module 05 today is a **stub, not a peer**. It is a single 16-cell notebook
(`05_rag/01_rag_fundamentals.ipynb`) that runs one happy-path RAG loop
(`all-MiniLM-L6-v2` → FAISS `IndexFlatL2` k=1 → TinyLlama-1.1B) over **6 hardcoded
sentences**, plus a 17-frame/5-act code-free slide deck, a 12-question quiz, and a
1-page cheatsheet that all mirror that one notebook.

A grounded audit (workflow `module05-rag-gap-map`, 7 agents over the live Navasena
files + Adinesia weeks 1/2/3/5) confirmed the gaps:

| Severity | Gap |
|---|---|
| 🔴 Critical | No evaluation anywhere — module ends at one printed answer (no recall@k, no faithfulness/groundedness). |
| 🔴 Critical | No real document ingestion/chunking — corpus is 6 Python strings; the document-prep half of RAG is absent. |
| 🟠 High | Single-stage k=1 brute-force retrieval — no reranking, no top-k tuning, no multi-doc context. |
| 🟠 High | L2-vs-cosine mismatch left unexplained (MiniLM is cosine-native; notebook uses raw L2). |
| 🟠 High | No relevance threshold / no-retrieval fallback; the instructive out-of-corpus case is commented out. |
| 🟠 High | Depth/format parity gap vs Module 04 (1 nb vs 8; 17 frames vs ~40; 12 Qs vs 31; no validators). |
| 🟡 Medium | No conversational/multi-turn RAG; no vector-DB/persistence; fragile generation plumbing (regex cleanup, manual ChatML split, `do_sample` on factual QA, `max_length` not `max_new_tokens`). |
| 🟡 Low | Dead `datasets` import; stale hardcoded "2024 president" fact; English-only embeddings for an Indonesian audience. |

**The opportunity:** Adinesia weeks 1/2/3/5 provide near-drop-in, non-gated, T4-safe
assets for every gap (chunking + quality scorer, FAISS-vs-Chroma, cross-encoder
reranker with before/after analysis, RAGAS with a local-judge adaptation recipe,
conversational memory manager).

## 2. Goals / Non-Goals

**Goals**
- Bring Module 05 to full parity with Module 04: an **8-notebook arc** + resynced
  slides (~40 frames/~10 acts), ~30-question quiz, 8-card cheatsheet, and `tools/`
  validators.
- Close every critical and high gap: evaluation (RAGAS), real ingestion/chunking,
  two-stage reranking, cosine correctness, conversational RAG, and deployment.
- Stay within Navasena's constraints (below) and keep an SMA-level audience in mind.

**Non-Goals**
- No paid-API dependency (Gemini/OpenAI/Groq) on the default path.
- No gated models (no HF auth).
- No multi-GPU / beyond-free-Colab requirements.

## 3. Locked Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Scope | **Full 8-notebook peer** to Module 04 | User chose maximum depth/parity. |
| Generator + RAGAS judge | **Qwen2.5-3B-Instruct** (4-bit), reused as judge | M04 stack continuity, better grounded answers, single-model dep. |
| Embeddings | **`paraphrase-multilingual-MiniLM-L12-v2`** (~384-dim) | Indonesian audience/corpus; near drop-in for current MiniLM. `intfloat/multilingual-e5-small` noted as a stronger-retrieval upgrade. |
| Reranker | **`BAAI/bge-reranker-v2-m3`** (multilingual, non-gated) | Replaces English ms-marco-MiniLM; cosine-native so it fixes the L2 mismatch naturally. |
| Ingestion | **Docling-first** + `pdfplumber`/`PyPDF2` lightweight fallback | Matches local tooling; fallback guarantees the cell runs on free Colab. |
| Framework | **Hybrid** — hand-rolled core; LangChain only for RAGAS wrapper + conversational memory | Beginners see mechanics; framework appears as labeled convenience. |
| Deploy | **Add a deploy artifact** (FastAPI `/ask` + runbook) | Mirrors M04's Deploy arc → clean 8-notebook peer. NIM as optional hosted path. |
| Optional hosted path | **NVIDIA NIM** (free tier) | Allowed exception to the no-API rule, presented as optional. |

## 4. Architecture

**The 8-notebook arc** (mirrors M04's Build→Use→Adapt→Measure→…→Deploy shape):

| # | Notebook | Teaches | Closes |
|---|---|---|---|
| 01 | Fundamentals (reworked) | canonical retrieve→augment→generate, footguns fixed | hygiene |
| 02 | Ingest & Chunk | Docling PDF→text, 3 chunking strategies + 0–100 quality scorer | 🔴 ingestion |
| 03 | Retrieve Better (Rerank) | bi-encoder over-fetch → cross-encoder rerank, before/after analysis | 🟠 k=1 |
| 04 | Measure (RAGAS) | faithfulness / answer-relevancy / context precision+recall, local judge | 🔴 eval |
| 05 | Scale the Index | cosine via `IndexFlatIP`+normalize, FAISS taxonomy, save/load + Chroma | 🟠 L2 / persistence |
| 06 | Conversational RAG | window+summary memory, history-aware query rewriting, multi-turn loop | 🟡 multi-turn |
| 07 | Capstone: Ask-My-Document | upload PDF → chunk → rerank → grounded answer w/ citations | ties together |
| 08 | Deploy | FastAPI `/ask` + edge/runbook, NIM optional | parity |

**One running document threads through all 8.** Default: a single Indonesian
Wikipedia article (adjustable to a public handbook/UU excerpt). nb01 uses a tiny
in-memory KB to build the mental model; nb02 introduces the real document; nb03–07
grow the *same* pipeline; nb08 ships it. Learners watch one system grow rather than
seeing 8 disconnected demos.

**Stack corrections that ripple through every notebook:**
- Embeddings → multilingual (Indonesian audience/corpus).
- Reranker → multilingual (`bge-reranker-v2-m3`), which makes cosine the correct default.

## 5. Per-Notebook Specs

Template: goal · flow · stack · signature artifact · T4 note. Markdown in Bahasa
Indonesia; code/comments in English; common engineering loan-words kept untranslated.

### 01 · Fundamentals *(rework existing nb)*
- **Goal:** the mental model — retrieve→augment→generate, done correctly.
- **Flow:** small in-memory KB (timeless facts) → multilingual embed → FAISS → run
  ALL test questions including a deliberate out-of-corpus probe (show retrieval fail).
- **Stack:** multilingual MiniLM, FAISS `IndexFlatL2` (cosine deferred to nb05),
  Qwen2.5-3B via `tokenizer.apply_chat_template`, `max_new_tokens`, deterministic
  (greedy) decoding with a note on why low-temp suits factual QA.
- **Signature artifact:** answer with vs without retrieval + the out-of-corpus failure.
- **T4 note:** drop dead `datasets` import; Qwen-3B in 4-bit (~2 GB).

### 02 · Ingest & Chunk
- **Goal:** build the real corpus.
- **Flow:** upload PDF (baked-in sample fallback) → Docling `DocumentConverter` →
  text → 3 chunking strategies (sentence / fixed+overlap / semantic) → `TokenCounter`
  + 0–100 chunk-quality scorer (token distribution, consistency CoV, coverage).
- **Stack:** Docling (primary) + pdfplumber fallback; Qwen tokenizer for counting; ID
  sentence segmentation.
- **Signature artifact:** comparison table — same doc, 3 strategies, quality scores +
  chunk-size histograms → "why semantic+overlap wins here."
- **T4 note:** Docling first-run downloads layout models (slow once); cache + show the
  lightweight fallback path explicitly.

### 03 · Retrieve Better (Rerank)
- **Goal:** turn mediocre retrieval into good retrieval.
- **Flow:** bi-encoder over-fetch k=10–20 → `bge-reranker-v2-m3` cross-encoder rerank →
  top-3 → Qwen generator → grounded answer.
- **Stack:** multilingual embedder + bge-reranker-v2-m3 + Qwen2.5-3B.
- **Signature artifact:** rank-change table + score scatter (bi-encoder vs reranked) +
  latency decomposition (stage-2 grows with #candidates, stage-1 ~flat).
- **T4 note:** load reranker, score, `gc.collect()` before generator if tight.

### 04 · Measure (RAGAS, local judge)
- **Goal:** know if RAG is good, not guess.
- **Flow:** build small RAG → generate answers+contexts for ~8–10 ID test questions
  w/ ground_truth → compute ALL FOUR metrics per-sample (Faithfulness,
  ResponseRelevancy, LLMContextPrecision, LLMContextRecall — actually compute all 4,
  do not hardcode 3 like Adinesia nb04) → interpret with thresholds (≥0.7 good /
  0.5–0.7 ok / <0.5 poor) → 4-panel dashboard + worst-question triage.
- **Stack:** `ragas` + LangChain `LangchainLLMWrapper` wrapping local Qwen2.5-3B as
  judge (non-gated adaptation of Adinesia's Gemini recipe); multilingual embedder for
  embedding-based metrics.
- **Signature artifact:** dashboard + triage call-out (e.g., "Q#7 faithfulness 0.3 →
  here is the hallucinated sentence").
- **T4 note:** carry M04's self-preference-bias caveat (model judging its own family);
  judge in 4-bit; heaviest notebook — document expected runtime. Lightweight manual
  faithfulness-checklist fallback if RAGAS chokes.

### 05 · Scale the Index (Vector Stores)
- **Goal:** make it correct, persistent, scalable.
- **Flow:** fix L2→cosine via `IndexFlatIP` + `normalize_L2` (now the correct default
  given e5/bge are cosine-native) → FAISS taxonomy table (Flat/IVF/HNSW/IVFPQ:
  speed·memory·accuracy) → IVF `train` + nlist/nprobe demo → `write_index`/`read_index`
  + JSON metadata sidecar → Chroma `PersistentClient` w/ metadata filtering →
  FAISS-vs-Chroma decision matrix + latency benchmark.
- **Stack:** FAISS, ChromaDB, multilingual embedder.
- **Signature artifact:** decision matrix backed by a real latency benchmark.

### 06 · Conversational RAG
- **Goal:** the missing multi-turn capability.
- **Flow:** `ConversationalMemoryManager` (window + summary) → history-aware query
  rewriting (follow-ups/pronouns retrieve correctly) → `contextual_search` blends
  current query + history → interactive chat loop (quit/stats/clear) → per-turn
  analyzer with exportable JSON.
- **Stack:** LangChain memory primitives + Qwen2.5-3B + nb03 reranked retriever.
- **Signature artifact:** transcript where "Apa ibukotanya?" → "Berapa penduduknya?"
  correctly resolves "-nya" via a rewritten query.

### 07 · Capstone: Ask-My-Document
- **Goal:** stitch everything into one product.
- **Flow:** upload PDF → Docling chunk → embed → FAISS → two-stage search (nb03) →
  top-k → Qwen answers WITH `chunk_id` citations → interactive `ipywidgets` Q&A loop;
  baked-in sample doc so it always runs.
- **Signature artifact:** answer with inline citations + expandable "lihat sumber".

### 08 · Deploy
- **Goal:** parity with M04's Deploy arc — ship the pipeline.
- **Flow:** wrap capstone pipeline in FastAPI `/ask` (Pydantic request/response,
  answer+sources), minimal auth/rate-limit note, portable runbook (env, model cache,
  run locally / on edge), NVIDIA NIM optional hosted-generator path.
- **Signature artifact:** `curl` → JSON-with-citations round-trip + deployment checklist.
- **T4 note:** Colab can't host a long-running server → "write code + in-process smoke
  test," runbook is the real deliverable (same posture as M04's deploy notebook).

**Pedagogical link:** nb03 *makes* retrieval better (rerank); nb04 *proves* it
(context-precision rises when reranked). Reuse the same test questions across 03/04 so
a number moves because of an engineering choice made one notebook earlier.

## 6. Cross-Cutting Deliverables

| Asset | Now | Target |
|---|---|---|
| Slides (`module05_slides.tex`) | 17 frames / 5 acts | ~40 frames / ~10 acts, code-free (one act per notebook). New mermaid figures (two-stage retrieval, chunking strategies, RAGAS quadrant) via `mmdc` @ scale 2–3. |
| Quiz | 12 Q | ~30 pure-concept Q: chunking trade-offs, bi- vs cross-encoder, cosine-vs-L2, the 4 RAGAS metrics, no-retrieval fallback, conversational memory, deploy. |
| Cheatsheet (HTML+PDF) | 1 card | 8 cards (one per notebook) + glossary; PDF via Chrome headless. |
| Validators (`05_rag/tools/`) | none | `validate_notebook.py` / `validate_quiz.py` / `validate_slides.py` (zero-code-block gate) / `validate_cheatsheet.py` (8-card presence), mirroring `04_llm/tools/`. |

## 7. File Layout

```
05_rag/
  01_rag_fundamentals.ipynb        (reworked)
  02_ingest_and_chunk.ipynb        02–08 new
  03_retrieve_better_rerank.ipynb
  04_measure_ragas.ipynb
  05_scale_the_index.ipynb
  06_conversational_rag.ipynb
  07_capstone_ask_my_document.ipynb
  08_rag_deployment.ipynb
  data/sample_id_document.pdf      (baked-in fallback)
  rag-quiz.html
  rag-cheatsheet.{html,pdf}
  slides/module05_slides.tex
  slides/figures/*.mmd | *.png
  tools/validate_*.py
```
Numbering `01–08` (not M04's `00–07`) preserves the existing `01_rag_fundamentals` name;
RAG has no "from-scratch" precursor like the transformer notebook.

## 8. Verified T4 Recipe & Constraints

- Pin **`transformers<5`**; pin `ragas` + `langchain` to known-good versions
  (`LangchainLLMWrapper` API drifts between minors). Validators check pins.
- **fp16 compute dtype, not bf16** — bf16 crashes 4-bit loads on T4.
- **4-bit Qwen2.5-3B** (generator + judge) + `gc.collect()` / `torch.cuda.empty_cache()`
  between heavy loads. nb03 (generator+reranker) and nb04 (judge+embedder) are the
  memory-tight notebooks.
- **All models non-gated:** Qwen2.5-3B-Instruct, paraphrase-multilingual-MiniLM-L12-v2,
  BAAI/bge-reranker-v2-m3, ChromaDB, Docling. NIM is the only (optional) hosted path.

## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| nb04 RAGAS with a local open judge is slow + parsing can be flaky | Small test set (~8–10 Q), 4-bit judge, robust output parsing, document runtime, manual faithfulness-checklist fallback in-notebook. |
| Docling first-run download heavy on free Colab | pdfplumber/PyPDF2 fallback path shown explicitly; cache after first run. |
| Memory pressure loading multiple models on 16 GB T4 | 4-bit, load/unload pattern, `gc.collect()` between stages. |
| Large scope (8 nb + 4 companion assets) | Build/validate incrementally, notebook by notebook, each passing its validator before moving on (M04 cadence). |
| Version drift (ragas/langchain) | Pin exact versions; validators enforce. |

## 10. Open Items (confirm during spec review)

- **Running document:** default = an Indonesian Wikipedia article. Alternative: a short
  public handbook / UU excerpt. Decide the concrete source.
- **Embeddings:** `paraphrase-multilingual-MiniLM-L12-v2` (default) vs
  `intfloat/multilingual-e5-small` (stronger retrieval, prefix convention).
- **nb05 Chroma half:** full notebook (as specced) vs fold persistence into nb02/03 if
  it feels thin.

## 11. Build Cadence

Notebook by notebook: 01 (rework) → 02 → 03 → 04 → 05 → 06 → 07 → 08, each validated
before the next. Companion assets (slides/quiz/cheatsheet/validators) resynced last,
once the notebook arc is final — same sequence that worked for Module 04. Feature
branch → validators green → merge to master → Drive sync for bootcamp participants.
