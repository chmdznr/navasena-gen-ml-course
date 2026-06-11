# Navasena ← Adinesia: Gap Map & Enrichment Roadmap

- **Tanggal:** 2026-06-11
- **Status:** Roadmap (analisis, belum eksekusi)
- **Sumber:** Workflow `adinesia-navasena-gap-map` (13 agen membaca 6 modul Navasena + 6 minggu Adinesia)
- **Materi sumber:** `/Users/chmdznr/work/adinesia/bootcamp_batch_1` (bootcamp 6 minggu, milik penulis yang sama — bebas diadaptasi)
- **Standar target:** Colab T4 16GB, model NON-GATED (Qwen/Phi/Mistral), markdown ID + code EN

> Catatan: Adinesia ≈ paruh **LLM/RAG/MLOps** dari kurikulum. Ia memperkaya **Modul 04 & 05** terutama, + memotivasi **Modul 07 baru** (deployment & ethics). Ia **tidak** menyentuh Modul 01 (ML klasik), 02 (DL), maupun 06 (TensorRT/Triton).

---

# Navasena ← Adinesia: Gap Map & Enrichment Plan

This analysis maps the instructor's prior Adinesia bootcamp (6 weeks) onto the current Navasena Gen-ML course (6 modules) to identify what to harvest, adapt, and ship. The guiding constraint throughout: **free Colab T4 16GB, NON-GATED models (Qwen/Phi/Mistral), Indonesian markdown + English code.**

---

## 1. Overlap / Gap Matrix

### 1a. Adinesia weeks → Navasena modules (overlap & relative strength)

| Adinesia week | Closest Navasena module | Relationship | Stronger side | Note |
|---|---|---|---|---|
| **W1** Data prep (PDF→text→spaCy→chunk→JSONL) | 03_nlp / 05_rag (none owns ingestion) | **GAP filled** — Navasena has no data-prep/chunking layer | Adinesia (only side) | Canonical pre-step before RAG. Highest cheap win. |
| **W2** Embeddings & vector stores (FAISS/Chroma, two-stage retrieval, reranking) | 05_rag | **Partial overlap + big gap** — Navasena does IndexFlatL2 only; no Chroma, no IVF/HNSW, no reranker | Adinesia (much deeper retrieval) | W2 is retrieval-only (no generation); Navasena closes the loop with its Qwen. |
| **W3** Prompt eng + LangChain + conversational RAG + A/B testing | 04_llm + 05_rag | **Partial overlap + gap** — Navasena has zero LangChain, zero conversational memory, zero prompt A/B | Adinesia (net-new capability) | Heavy cloud-API dependency (Gemini/Groq/OpenAI) — must re-host on local Qwen. |
| **W4** LoRA/PEFT fine-tuning + W&B + HF Hub deploy | 04_llm | **Pure GAP** — Navasena has NO fine-tuning at all | Adinesia (only side) | Uses non-gated flan-t5-base; swap to Qwen causal-LM for narrative consistency. |
| **W5** Evaluation & A/B: RAG vs Fine-Tune (RAGAS/ROUGE/BLEU/BERTScore, stats, Unsloth) | 04_llm + 05_rag | **Pure GAP** — Navasena has NO evaluation methodology anywhere | Adinesia (only side) | Free metric notebooks (ROUGE/BLEU/stats) are drop-in; RAGAS judge & Unsloth base need swaps. |
| **W6** Deployment & ethics (quant, llama.cpp/Ollama/vLLM/TGI, FastAPI/Docker, bias/PII) | 04_llm (02_production) + 06_nvidia | **Partial overlap (quant) + huge gap (serving + ethics)** | Adinesia (serving/ethics), Navasena (cleaner 4-bit) | Ethics/PII is **entirely net-new**. Docker/TGI can't run in Colab — demote to concept. |

### 1b. The gaps, stated as themes Navasena lacks

| Capability area | Navasena status today | Adinesia source(s) | Verdict |
|---|---|---|---|
| **Document ingestion & chunking** | Absent (RAG uses 6 hardcoded strings) | W1 (TokenCounter/TextChunker/DocumentProcessor) | **Fill — flagship cheap win** |
| **Retrieval depth** (Chroma, IVF/HNSW, cosine vs L2, two-stage rerank, hybrid BM25) | IndexFlatL2 + k=1 only | W2 (FAISS index types, CrossEncoder rerank), W3 (local reranker) | **Fill — biggest RAG quality lever** |
| **Fine-tuning (LoRA/QLoRA/PEFT)** | Absent | W4 (PEFT pipeline), W5 (Unsloth QLoRA) | **Fill — top strategic gap** |
| **Evaluation & statistical rigor** | Absent (eyeballing) | W5 (ROUGE/BLEU/BERTScore/RAGAS + paired t-test/Cohen's d) | **Fill — high value, mostly free** |
| **Prompt engineering + LangChain + memory** | Absent (raw `.generate()`) | W3 (LCEL chains, 3 memory types, prompt A/B) | **Fill — partly** |
| **Serving frameworks** (vLLM/Ollama/llama.cpp, FastAPI) | Absent (02_production stops at single-process) | W6 (NB02/03/04) | **Fill — selectively (Colab caveats)** |
| **Responsible AI / bias / PII** | Absent entirely | W6 NB05 (Detoxify, Presidio) | **Fill — net-new, high-impact, free** |
| **Classical ML MLOps** (CV, Pipeline, MLflow, drift) | Absent in Module 01 | *No Adinesia coverage* | Out of scope here — Adinesia is all LLM/RAG. Note for future. |

**Key observation:** Adinesia is overwhelmingly the **LLM/RAG/MLOps** half of the course. It does **not** touch classical ML (Module 01), deep-learning fundamentals (Module 02), or the NVIDIA performance-engineering theme (Module 06's TensorRT/Triton story). So Adinesia enriches Modules **04 and 05 primarily**, with a smaller ethics/serving spillover that motivates a **new Module 07**.

---

## 2. Enrichment Proposal

Ordered by landing location. Each item: what it teaches, source asset, required changes, effort (S = ≤0.5 day port, M = 1–2 days, L = 3+ days incl. theory/slides).

### Module 05 (RAG) — the biggest beneficiary

Navasena's `05_rag` is currently one thin notebook. The plan promotes it to a proper 4–5 notebook module.

#### **`05_rag/00_data_ingestion_and_chunking.ipynb`** — *NEW*
- **Teaches:** PDF/image → clean text (PyPDF2/pdfplumber, Docling optional) → token-aware chunking (sentence/sliding-overlap/fixed/semantic) → chunk-quality metrics → JSONL export.
- **Adapt from:** W1 NB04 (`TokenCounter`+`TextChunker`+`analyze_chunk_quality`+`DocumentProcessor`) and W1 NB02 (PDF extractor comparison). Theory from `Week_1_Foundational_Theory.md`.
- **Must change:** (a) ID markdown; (b) drop `!nvcc`, `spacy[cuda12x]`→plain `spacy`; (c) replace hardcoded `/content/drive/...` paths with `files.upload()` / repo-relative; (d) add the **Qwen2.5 tokenizer** to the token-count cell so chunk sizing matches Module 04's model; (e) gate Docling behind `try/except`; (f) swap demo PDF for an Indonesian doc. No gated/paid deps.
- **Effort:** **M**. Keep `week1_chunker.py` as the graded mini-lab (PDF/URL → chunks.jsonl).

#### **`05_rag/01_rag_fundamentals.ipynb`** — *UPGRADE existing*
- **Teaches (fixes):** normalize embeddings + `IndexFlatIP`/cosine (not raw L2); Qwen chat template via `apply_chat_template`; switch generator **TinyLlama-1.1B float32** → **Qwen2.5-3B-Instruct** (or Mistral-7B-4bit); multilingual embedder note (`multilingual-e5` vs English-centric all-MiniLM).
- **Adapt from:** existing notebook + W2 NB01 (cosine/PCA viz). **Effort:** **S–M**.

#### **`05_rag/02_vector_stores_and_retrieval.ipynb`** — *NEW*
- **Teaches:** FAISS index-type tradeoffs (Flat/IVF/HNSW/PQ), persistence, ChromaDB metadata filtering, **two-stage retrieval** (bi-encoder → CrossEncoder reranker). Single highest-value missing RAG concept.
- **Adapt from:** W2 NB02 + NB03 (`RetrievalPipeline`) + W3 local CrossEncoder (`BAAI/bge-small-en-v1.5`, no API key).
- **Must change:** ID markdown; unpin stale `sentence-transformers==2.2.2`/`chromadb==0.4.18`; dark-theme plots; drop Streamlit. Non-gated. **Effort:** **M**.

#### **`05_rag/03_conversational_rag.ipynb`** — *NEW (optional)*
- **Teaches:** conversational memory on FAISS RAG, LCEL chains wrapping local Qwen, three memory types (Buffer/Window/Summary).
- **Adapt from:** W3 NB02 + NB03.
- **Must change — CRITICAL:** W3 generates via **Gemini/Groq/OpenAI cloud APIs** → re-host on `HuggingFacePipeline` wrapping local **Qwen2.5-3B**. "tiktoken/$cost" → Qwen tokenizer count. Drop `promptfoo`. **Effort:** **L**.

### Module 04 (LLM) — fine-tuning + evaluation, the strategic gaps

#### **`04_llm/03_llm_finetuning_lora.ipynb`** — *NEW (top-priority gap)*
- **Teaches:** LoRA/QLoRA theory → instruction dataset prep → 4-bit QLoRA config → train (TRL `SFTTrainer`) → base-vs-finetuned eval → save adapter → optional HF push.
- **Adapt from:** W4 (`Week_4_LoRA_Finetuning_Colab.ipynb`, eval 4-panel figure, LoRA theory PDF+diagrams) and/or W5 Unsloth recipe.
- **Must change:** **Model:** W4 uses **flan-t5-base (seq2seq)** → swap to decoder-only **Qwen2.5-1.5B-Instruct** (FP16 LoRA) or **QLoRA on Qwen2.5-3B** (4-bit). `SEQ_2_SEQ_LM`→`CAUSAL_LM`, targets `q/k/v/o_proj`, `AutoModelForCausalLM`, real `apply_chat_template`. **⚠ GATED:** W5 Unsloth defaults to `Llama-3.2-1B` (upstream gated) → swap to Qwen2.5/Phi-3.5. W&B → default TensorBoard. Fix W4 bugs: `AutoModelForSeq2Seq`→`...LM`, f-string inline-`if`, `as_target_tokenizer()`/`evaluation_strategy`. Collapse W4's 7 scripts into ONE notebook; ID instruction set. **Effort:** **L (marquee).**

#### **`04_llm/04_llm_evaluation.ipynb`** — *NEW (high value, mostly free)*
- **Teaches:** NLG metrics (ROUGE/BLEU/BERTScore/perplexity), **statistical significance** (paired t-test, Cohen's d, 95% CI), RAGAS (faithfulness/relevancy/context).
- **Adapt from:** W5 `02_nlg_metrics` + `03_statistical_testing` (**free, near drop-in**) + `01_ragas` + theory PDF.
- **Must change — CRITICAL:** RAGAS judge = **Gemini-2.5-flash (paid)** → local **Qwen2.5-3B / Mistral-7B-4bit** via `LangchainLLMWrapper`. ROUGE/BLEU/stats = free spine; RAGAS = optional tail. **Effort:** **M**.

#### **`04_llm/03b_prompt_engineering.ipynb`** — *NEW (optional, can fold into 01_llm_basics)*
- **Teaches:** zero/few-shot, CoT, self-consistency, prompt A/B with `scipy` + plots.
- **Adapt from:** W3 NB01 (`PromptTester/Evaluator/Optimizer`). **Must change:** Gemini/Groq → local Qwen. **Effort:** **S–M (low priority).**

### New Module 07 (Deployment & Responsible AI) — net-new altitude

W6 is the one Adinesia week with no Navasena counterpart. Recommend **new `07_deployment_and_ethics/`** rather than overloading 04.

#### **`07_deployment_and_ethics/01_quantization_and_local_serving.ipynb`** — *NEW*
- **Teaches:** GGUF vs GPTQ vs AWQ vs BnB decision framework; local serving via **llama.cpp-python + Ollama** (T4/CPU-runnable).
- **Adapt from:** W6 NB01 (decision table) + NB02 (llama.cpp/Ollama).
- **Must change:** gated Llama-3.2 → non-gated **Qwen2.5-GGUF** / Ollama `qwen2.5:1.5b`. Avoid redundancy with 02_production's 4-bit (keep only framework table + GGUF/Ollama). **Effort:** **M**.

#### **`07_deployment_and_ethics/02_serving_api_vllm_fastapi.ipynb`** — *NEW*
- **Teaches:** vLLM (PagedAttention, OpenAI-compatible server) on Qwen2.5-1.5B; FastAPI `/ask` + rate-limit + Prometheus, in-Colab via background `uvicorn`+ngrok.
- **Adapt from:** W6 NB03 (vLLM) + NB04 (FastAPI skeleton).
- **Must change — T4 CAVEATS:** **Docker can't run in Colab** → "run OUTSIDE Colab" artifacts. **TGI can't run in Colab** → concept + command box. Pin Colab-T4-known-good vLLM. Swap gated Llama-3.2 → Qwen2.5-1.5B. **Effort:** **L (most fragile).**

#### **`07_deployment_and_ethics/03_responsible_ai_bias_pii.ipynb`** — *NEW (highest-value net-new, free)*
- **Teaches:** toxicity (Detoxify), PII detect+redact (Presidio+spaCy), fairness metrics (demographic parity, equalized odds), combined `SafetyPipeline`.
- **Adapt from:** W6 NB05 + AI Ethics Checklist.
- **Must change — CRITICAL:** DeepEval `BiasMetric` = **OpenAI (paid, unflagged)** → free Detoxify/sklearn fairness. Drop broken Guardrails 0.5 block. Add **Indonesian PII** (NIK, NPWP, +62) to Presidio. **Effort:** **M**.

### Module 03 (NLP) — secondary spillover (optional)
W1 NB03 spaCy doc-quality scorer could deepen the ID/EN NLP notebooks, but **lower priority** — chunking lands better in 05.

---

## 3. Sequencing & Priority

Everything LLM-facing reuses the **just-completed Module 04** (Qwen2.5-3B in `01_llm_basics`, Mistral-7B-4bit + bitsandbytes in `02_llm_production`).

**Wave 1 — Quick wins, low risk, high value:**
1. `05_rag/00_data_ingestion_and_chunking` (M) — no gated/paid deps, fills the most glaring RAG gap. *No Module-04 dep.*
2. `04_llm/04_llm_evaluation` (M) — ROUGE/BLEU/stats core free & near-drop-in; teaches rigor the course lacks.
3. `05_rag/01_rag_fundamentals` upgrade (S–M) — TinyLlama→Qwen, L2→cosine, regex→chat-template.

**Wave 2 — Core strategic gaps:**
4. `04_llm/03_llm_finetuning_lora` (L) — marquee gap; most adaptation. Builds on 02_production's 4-bit.
5. `05_rag/02_vector_stores_and_retrieval` (M) — reranking = biggest RAG quality lever.
6. `07_deployment_and_ethics/03_responsible_ai_bias_pii` (M) — net-new, free, no hard Colab blockers.

**Wave 3 — Higher-effort / Colab-fragile:**
7. `05_rag/03_conversational_rag` + `04_llm/03b_prompt_engineering` (L / S–M) — LangChain-on-local-HF.
8. `07_deployment_and_ethics/01_quantization_and_local_serving` (M) — Ollama/llama.cpp.
9. `07_deployment_and_ethics/02_serving_api_vllm_fastapi` (L) — most T4-fragile (vLLM pinning, Docker/TGI demotion). Last.

---

## 4. Risks / Notes / Low-Confidence Flags

**T4-feasibility:** Docker & TGI **cannot run in Colab kernel** → ship as "run outside Colab" / concept boxes. vLLM has CUDA/version friction → pin verified version, smoke-test (highest fragility). Docling heavy/flaky first run → PyPDF2/pdfplumber default, Docling behind `try/except`. ngrok needs free authtoken.

**Licensing / models — POLICY UPDATE (2026-06-11):** Cloud LLM APIs are now **ALLOWED in Navasena** for accessing larger models. **Preferred: NVIDIA NIM (`build.nvidia.com`)** — OpenAI-compatible endpoint, free tier, on-theme with Module 06; participants register for `NVIDIA_API_KEY`. Gemini API is an acceptable alternative. **Default pattern:** local non-gated model (offline, no key) **+ optional cloud big-model path** (graceful skip if no key). This relaxes the earlier "must swap paid→local" rule:
- W5 **RAGAS judge / W6 BiasMetric / W3 generation** — MAY use a cloud big model (NIM/Gemini) as the strong/independent option, but keep a **local fallback** so the notebook runs without a key.
- Still avoid **gated model-weight downloads** (e.g., `meta-llama/Llama-3.2` from HF) on the runnable path — that's participant-friction at download time, different from cloud API access. Use Qwen/Phi/Mistral for local weights; reach big models like Llama-3.1-70B via **NIM API** instead of downloading.
- `promptfoo` (Node + OpenAI key) — still drop or optional appendix.

**Redundancy:** W6 NB01 quant overlaps 02_production (harvest only decision table); W2 NB01 partly duplicates existing 05_rag (cherry-pick PCA-viz + GPU benchmark); W5 has duplicate Unsloth iterations (keep one).

**Code bugs to fix on adoption:** W4 `AutoModelForSeq2Seq`→`...LM`; W4 model-card f-string inline-`if`; deprecated `as_target_tokenizer()`/`evaluation_strategy`.

**House-style normalization (every port):** markdown → ID; strip heavy emoji; `seaborn-v0_8` → repo dark-theme; unpin stale versions; hardcoded `/content/drive/...` → `files.upload()`/repo-relative; re-render mermaid via `mmdc` scale 2–3 with ID labels.

**Low-confidence (verify before effort):** LangChain-on-local-HF latency on T4 (pilot one chain first); Unsloth/vLLM Colab version sensitivity (budget smoke-test); RAGAS with local 3B judge = noisier (frame "educational"); LoRA source = blend **W4 structure + W5 decoder-only/QLoRA on a Qwen base**, not a straight port.

**Out of scope (no Adinesia source):** Module 01 ML MLOps, Module 02 PyTorch bridge, Module 06 real TensorRT/Triton — genuine course gaps this enrichment does NOT address.
