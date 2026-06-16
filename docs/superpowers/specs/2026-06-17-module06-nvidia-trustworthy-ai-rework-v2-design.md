# Module 06 Rework v2 — NVIDIA Ecosystem & Trustworthy AI (cert-aligned, depth-first)

**Date:** 2026-06-17
**Status:** Design approved (brainstorm), pending spec review
**Supersedes:** `2026-06-15-module06-nvidia-trustworthy-ai-rework-design.md` (the 9-notebook "Arc C 4+4" arc) and the notebooks it produced (`00`–`08` on branch `module06-rework`).

## 1. Why a v2 rework

The 9-notebook v1 was built, Colab-verified, and shipped with green validators — yet the content is **under-built and embarrassing as teaching material**. This is an honest, evidence-backed diagnosis, not a vibe:

- **Thin.** Markdown per notebook is ~1/3 of the M05 peer standard. nb02/nb03 are 8 cells with ~1.3–1.6 KB of markdown; M05 notebooks run 5.8–10.6 KB markdown over 14–24 cells.
- **Black-box helpers ("lazy coding").** Every notebook does `wget nvidia_utils.py` then calls `nim_chat(...)`. The single most NVIDIA-specific mechanic in the module — disabling Nemotron reasoning via `extra_body={"chat_template_kwargs":{"enable_thinking":False}}` — is **hidden in the helper and never shown or taught**.
- **Guardrails narrated, not executed.** The module's headline ("NeMo Guardrails") is taught as plain `if` + `nim_chat` classifier calls; the YAML/Colang is **pasted as text**. The notebooks never actually run NeMo Guardrails. Root cause: v1 tried to run NeMo Guardrails/Colang with a *reasoning* model (Nemotron) → `<think>` leakage → they retreated to narration.
- **Repetitive skeleton × 9** → reads as padded; the "too many notebooks" instinct is correct for the thin front half.

**Process lesson:** v1's validators + "Colab-verified" only checked *code runs* and *structural markers*, not *teaching depth* or *authentic execution*. Green ≠ good. v2 adds a depth floor and an execution gate.

## 2. Decisions (locked in brainstorm 2026-06-17)

| Decision | Choice |
|---|---|
| **Center of gravity** | Cert-aligned NVIDIA ecosystem + Trustworthy AI **breadth**, done deep, with real code shown and demos executed. |
| **Scope** | **Self-contained cert-review** package — may recap RAG/deploy even though it overlaps M04/M05. |
| **Structure** | **2 tracks, ~5 dense Colab notebooks** + 1 optional off-Colab hardware lab. |
| **Depth** | Each notebook ≥ M05 density (markdown, real scenarios, richer demos, meaningful exercises). |
| **Helpers** | `nvidia_utils.py` stays for DRY, but every NVIDIA-specific mechanic is **shown inline the first time it appears**; no black-box wget-then-call. |
| **Guardrails** | **Executed for real** via NemoGuard NIMs (see §5), not narrated. |
| **Tooling currency** | Triton = exam baseline + note Dynamo as current flagship; NemoGuard NIMs for safety; Nemotron-3-nano generator/judge. |
| **Off-Colab lab** | **NVIDIA-native edge LLM on Jetson Orin** — TensorRT Edge-LLM (docker: INT4 AWQ quant → ONNX → TensorRT engine → on-device inference). Mac dropped (no NVIDIA GPU). |

### Cert grounding (NCA-GENL, official 2026 blueprint)
Domains & weights: Core ML **30%** · Software Development **24%** · Experimentation **22%** · Data Analysis & Viz **14%** · **Trustworthy AI 10%**. The NVIDIA hardware/CUDA deep-internals are a *thin* exam slice; the ecosystem footprint sits in Software Development as **deploy/serving** (Triton is the named tool) + transformer/LLM internals. Trustworthy AI is its own domain → it earns the larger share of M06 (module identity). RAPIDS/cuDF is already taught in Module 03 → **not repeated**.

## 3. Deliverables (6)

### Track 1 — NVIDIA Stack (maps to Software Development domain)

**nb01 · GPU, Precision & Optimasi Inferensi**
- Tensor Cores & parallelism (why GPU for matmul); precision FP32/FP16/FP8 with a real matmul + memory benchmark (retain v1's verified numbers: 6.18/3.10 GB 2.0×, 34.81/6.23 ms 5.6×).
- Inference quantization (4-bit) with real memory delta (3.09→1.16 GB, 2.7×). Contrast with M04's *training* QLoRA.
- **TensorRT** (general optimizer) vs **TensorRT-LLM** (paged KV-cache, in-flight batching, FP8/INT4) — concept + a real quantization run. TensorRT-LLM full engine build is x86-only → taught as concept with the `trtllm-serve` one-liner shown as reference.
- Cert callout: Software Development (optimize/deploy).

**nb02 · Serving & Deploy: Triton → Dynamo → NIM** *(RAG recap folded in)*
- The serving problem (batching, concurrency, multi-model). **Triton Inference Server** as the exam baseline (architecture, dynamic batching); note **Dynamo** ("Dynamo-Triton", GTC 2025 flagship: disaggregated prefill/decode, KV-aware routing) as the current direction.
- **NIM** = containerized, OpenAI-compatible microservice. **Show the real code inline**: `OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=...)`, `client.chat.completions.create(...)`, and the `extra_body={"chat_template_kwargs":{"enable_thinking":False}}` reasoning-off trick **explained, not hidden**. Nemotron-3-nano.
- **One client, many backends** theme: hosted NIM (Colab) ≡ self-host NIM (RTX) ≡ TensorRT-optimized edge engine on Jetson — all speak the same OpenAI `/v1` contract; only the backend changes. (Sets up nb06.)
- **RAG-on-NIM recap** (self-contained, lean): retrieval + NIM generate + `[n]` citations; links to M05 for depth.
- Cert callout: Software Development (serving/deploy), LLM integration.

### Track 2 — Trustworthy AI (the 10% domain — module identity, larger share)

**nb03 · Fairness, Bias & Tata Kelola**
- NVIDIA Trustworthy AI pillars. **Fairness metrics computed for real** (demographic parity, equalized odds) on a small dataset; thresholds (< 0.1). Bias detection via **LLM-judge** on free text (contrast: deterministic metrics vs flexible judge).
- Governance: **Model Card** + ethics checklist (transparency/accountability). UU PDP framing where relevant.
- Cert callout: Trustworthy AI (bias/fairness, alignment/transparency).

**nb04 · Safety, Guardrails & Privasi — NemoGuard NYATA**
- **Run real NemoGuard NIMs** (free-tier, OpenAI-compatible, verified 2026-06-17):
  - `nvidia/llama-3.1-nemoguard-8b-content-safety` (or newer `nvidia/llama-3.1-nemotron-safety-guard-8b-v3`) — Aegis 23-category taxonomy; system prompt = policy; returns JSON `{"User Safety":"unsafe","Safety Categories":"..."}`. Classifies user turn AND bot turn.
  - `nvidia/llama-3.1-nemoguard-8b-topic-control` — system ends with `TOPIC_SAFETY_OUTPUT_RESTRICTION`; returns `on-topic`/`off-topic`.
  - Show the real call + structured output; contrast with the "what it does under the hood" direct-classifier view for teaching.
- **Jailbreak**: `nvidia/nemoguard-jailbreak-detect` is NOT a chat model (Random Forest over 768-dim embedding, `/v1/classify`) → not free-tier-callable in the simple pattern. Use **NeMo Guardrails OSS `self check input`** with a small non-reasoning judge routed to nemotron-nano. This is the genuinely-executed self-check half.
- **Privacy**: PII masking (NIK/HP/rekening → `[NIK]`/`[PHONE]`/`[ACCOUNT]`), UU PDP (consent, breach ≤72 h / denda ≤2%). **Grounding** (anti-hallucination) as a transparency rail.
- Cert callout: Trustworthy AI (safety/guardrails, privacy/consent).

**nb05 · Capstone: Deploy `/ask` Berpagar**
- FastAPI `/ask` wiring the rails end-to-end: input (self-check jailbreak + content-safety + topic-control) → RAG retrieve+generate (NIM) → output (grounding + content-safety) + PII masking. `%%writefile` a self-contained service; real `TestClient` smoke tests (200 grounded+cited, 422 bad req, blocked cases); runbook.
- Cert callout: ties Software Development (deploy) + Trustworthy AI together.

### Off-Colab lab (Jetson only)

**nb06 · Edge Deploy di Jetson Orin — TensorRT Edge-LLM** — *not run in Colab; hands-on on the user's Orin*
- The genuine NVIDIA-native edge path (docker-based — confirmed via Jetson AI Lab tutorials 2026-06-17). Docker on Jetson is the norm (the whole Jetson AI Lab is container-based). NIM self-host is name-dropped as reference only (NIM needs x86 datacenter/RTX + driver R560+ + AI Enterprise license; not Jetson) — the real edge stack is below.
- **Model = the M04 specialist** `qwen3-0.6b-spesialis` — **skip re-training**, reuse the fine-tuned weights; Edge-LLM does its own **INT4 AWQ** quantization (the quant step IS the cert-relevant optimization, so re-quantizing from full precision is correct here).
  - ⚠️ **Prerequisite (verify):** Edge-LLM needs the **safetensors** checkpoint. HF check (2026-06-17) shows only `chmdznr/qwen3-0.6b-spesialis-gguf` (GGUF) is published — **no safetensors repo**. GGUF is NOT usable as Edge-LLM input. **Exact path (located in `04_llm/07_slm_deployment.ipynb`):** cell 6 ALREADY merges the QLoRA adapter to **fp16 safetensors** at `./qwen3-merged-fp16` (`save_pretrained(..., safe_serialization=True)` + tokenizer/chat_template) — the GGUF (cell 9) is converted from it. It is simply never uploaded; cell 13 pushes only the GGUF. **Fix = add ~3 lines after cell 6** (`create_repo` + `HfApi().upload_folder("./qwen3-merged-fp16")` → `chmdznr/qwen3-0.6b-spesialis`), then re-run 07 once. This makes 07 export BOTH safetensors (TensorRT Edge-LLM target) + GGUF (llama.cpp target) from one run. GGUF→safetensors NOT recommended (Q4 → degraded → double-quant loss). Not a blocker for nb01–05.
- **TensorRT Edge-LLM** = NVIDIA's C++ inference runtime for LLMs/VLMs on Jetson (Orin Nano → Thor); compiles models to optimized TensorRT engines, no Python in the inference path. Hands-on flow (0.6B specialist, INT4 AWQ, Orin Nano 8 GB — trivially fits):
  1. Export → ONNX from the safetensors checkpoint in the NVIDIA PyTorch container (`docker run --runtime nvidia nvcr.io/nvidia/pytorch:25.12-py3`); clone `github.com/NVIDIA/TensorRT-Edge-LLM`; INT4 AWQ quantize.
  2. Build the TensorRT engine on the Orin (`llm_build --maxBatchSize 1 --maxInputLen 512 --maxKVCacheCapacity 1024`).
  3. On-device inference via the native `llm_inference` binary.
  - Prereqs: JetPack 6.2.x, CUDA 12.6, TensorRT 10.x, 20–50 GB free.
- **Webservice tie-in:** Edge-LLM is a C++ CLI/runtime, NOT an OpenAI HTTP server. For the `/ask` webservice contract: wrap the Edge-LLM engine in a thin FastAPI `/v1` shim (ties to nb05), or use **vLLM-on-Jetson** (native `/v1`). (NanoLLM — dustynv, MLC-backed, rich agent/web-UI/RAG/speech — is instructive but **archived/deprecated** on Jetson AI Lab; optional "see also", not the spine.)
- Lesson: the serving **contract is identical** to the hosted NIM in nb02 — same client code, different backend (hosted NIM → TensorRT-optimized edge engine).
- Cert callout: TensorRT/quantization + edge deployment (Software Development). Marked **"jalankan di Jetson, bukan Colab"**; verified on the user's Orin.

## 4. Cross-cutting design

- **Helper policy.** `nvidia_utils.py`: first appearance of `nim_client`, `nim_chat`/`extra_body`, NemoGuard calls, and PII masking are shown inline in the notebook with explanation; the helper is introduced *after* as DRY reuse, and the notebook states what it wraps. No teaching-critical code hidden behind wget.
- **Depth floor (new gate).** `validate_notebooks.py` adds: min cells per notebook, min markdown chars (≥ ~5 KB, calibrated to M05's ~5.2 KB floor), and **execution markers** proving NemoGuard NIMs were actually called (e.g. presence of `nemoguard` model IDs + structured output in cells). "Thin" and "narrated-only guardrails" can no longer pass.
- **Bahasa luwes.** Per `bahasa-indonesia-engineering-luwes`: keep common English engineering loanwords untranslated; no awkward calques; no ambiguous phrasing; audience lulusan SMA. ai-slop-checker pass before ship.
- **Self-contained key handling.** `NVIDIA_API_KEY` via Colab Secrets; one key serves generator + judge + NemoGuard NIMs.

## 5. Companion re-alignment (downstream)

The reworked companion (slides 40-frame, quiz 34-Q, cheatsheet 8-card, 4 validators — completed 2026-06-17) is keyed to the **9-notebook v1 arc**. Concepts are ~90% reusable, but **structure and counts must re-map to the new 6-deliverable arc** (5 Colab nb + 1 Jetson lab): act/section structure, the cheatsheet card→notebook mapping, the "sembilan notebook" framing, quiz `notebook` tags, and all four validators' notebook lists. This is a **separate phase after the notebooks settle** — do not redo it until nb01–06 are final.

## 6. Migration & verification

- **From v1:** the v1 notebooks `00`–`08` are consolidated into `01`–`05` + `06` (lab). Salvage verified assets (benchmarks/numbers, PII helper, fairness metrics, capstone FastAPI skeleton). Old files removed once new ones are verified.
- **Verification per notebook:** authored to depth → locally NIM-verified (real calls, incl. NemoGuard) → Colab confirm-run to populate outputs → depth-floor + execution validators green. nb06 verified on the Jetson (not Colab).
- **Git:** continue on a feature branch; commit per notebook; push/merge only when asked.

## 7. Open items to verify at build

1. Exact current Content-Safety model ID — prefer `nvidia/llama-3.1-nemotron-safety-guard-8b-v3` if confirmed live on free tier; fallback `nvidia/llama-3.1-nemoguard-8b-content-safety`.
2. NemoGuard NIM free-tier rate limits under a per-student key (multiple guard calls per `/ask` turn).
3. TensorRT-LLM depth on T4 — keep to concept + quantization run; no full engine build in Colab.
4. nb06: source the `qwen3-0.6b-spesialis` **safetensors** (the published artifact is GGUF-only — won't work for Edge-LLM); confirm Orin module/VRAM + JetPack 6.2.x; pick the OpenAI-webservice tie-in (FastAPI shim over Edge-LLM vs vLLM-on-Jetson).

## 8. Non-goals

- Not re-teaching RAPIDS/cuDF (Module 03), generic LLM fine-tuning (Module 04), or full RAG depth (Module 05) — only self-contained recaps.
- Not building NIM self-host on Jetson/Mac (infeasible) — Ollama/vLLM is the Jetson hands-on.
- Not deep CUDA-kernel internals (thin exam slice).
