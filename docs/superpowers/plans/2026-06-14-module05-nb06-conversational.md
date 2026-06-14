# Module 05 nb06 — Conversational RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** Add `06_conversational_rag.ipynb` that adds multi-turn memory on top of the reranked retriever: a hand-rolled `ConversationalMemoryManager` (window + summary) + history-aware query rewriting (so follow-up/pronoun questions retrieve correctly), a scripted conversation demo + optional interactive loop, and an honest "this is how a well-established library does it" LangChain reference.

**Architecture:** Self-contained Colab notebook (GPU/T4 — Qwen2.5-3B 4-bit). Hand-rolled memory is the core (transparent + robust); the memory window/summary logic lives in `rag_utils.ConversationalMemoryManager` (GPU-free, unit-tested, summarizer injected). The notebook injects a Qwen-based summarizer + uses Qwen for history-aware rewriting and generation, over the handbook corpus + bi-encoder + bge-reranker (from nb03). The LangChain reference is shown as honest read-only markdown (the library's memory API is in flux: old classes deprecated → `RunnableWithMessageHistory` deprecated → LangGraph current; `create_history_aware_retriever` moved to `langchain_classic`).

**Tech Stack:** transformers<5, sentence-transformers (multilingual MiniLM), faiss-cpu, bge-reranker-v2-m3, Qwen2.5-3B-Instruct 4-bit. Markdown Bahasa Indonesia; code/comments English.

**Reference spec:** `docs/superpowers/specs/2026-06-14-module05-rag-rework-design.md` §5 (nb06). Divergence from spec's "LangChain memory primitives": hand-rolled core (user decision) — justified by LangChain memory deprecation churn; LangChain shown as honest reference.

---

## File Structure

| Path | Responsibility |
|---|---|
| `05_rag/tools/rag_utils.py` | MODIFY — add `ConversationalMemoryManager` (window+summary, injected summarizer; GPU-free). |
| `05_rag/tools/test_rag_utils.py` | MODIFY — add tests for the memory manager. |
| `05_rag/tools/validate_notebooks.py` | MODIFY — add `06_conversational_rag.ipynb` entry. |
| `05_rag/06_conversational_rag.ipynb` | CREATE — the conversational RAG notebook. |

---

## Task 1: `ConversationalMemoryManager` (TDD)

**Files:** Modify `05_rag/tools/rag_utils.py`, `05_rag/tools/test_rag_utils.py`

- [ ] **Step 1: Append tests to `test_rag_utils.py`**
```python
from rag_utils import ConversationalMemoryManager

def _fake_sum(old, turn):
    u, a = turn
    return (old + " | " if old else "") + f"{u}=>{a}"

def test_memory_window_keeps_last_n():
    m = ConversationalMemoryManager(_fake_sum, window=2)
    for i in range(4):
        m.add_turn(f"q{i}", f"a{i}")
    assert len(m.turns) == 2 and m.turns[0] == ("q2", "a2")

def test_memory_summarizes_evicted_turns():
    m = ConversationalMemoryManager(_fake_sum, window=2)
    for i in range(3):
        m.add_turn(f"q{i}", f"a{i}")          # q0 evicted -> summary
    assert "q0=>a0" in m.summary and len(m.turns) == 2

def test_memory_context_has_summary_and_window():
    m = ConversationalMemoryManager(_fake_sum, window=1)
    m.add_turn("a", "1"); m.add_turn("b", "2")  # 'a' evicted to summary, 'b' in window
    ctx = m.context()
    assert "Ringkasan" in ctx and "User: b" in ctx and "Asisten: 2" in ctx

def test_memory_clear_resets():
    m = ConversationalMemoryManager(_fake_sum, window=2)
    m.add_turn("x", "y"); m.clear()
    assert m.turns == [] and m.summary == "" and m.stats()["turns_kept"] == 0
```

- [ ] **Step 2: Run to verify FAIL** — `cd 05_rag/tools && python3 -m pytest test_rag_utils.py -k memory -v` → ImportError.

- [ ] **Step 3: Implement (append to `rag_utils.py`)**
```python
class ConversationalMemoryManager:
    """Window + summary conversational memory. GPU-free: the summarizer is INJECTED.

    Keeps the last `window` turns verbatim; older turns are folded into a running
    `summary` via summarize_fn(old_summary, (user, assistant)) -> new_summary.
    In notebooks, summarize_fn wraps a local LLM (Qwen). Pure-logic + testable here.
    """
    def __init__(self, summarize_fn, window=4):
        if window < 1:
            raise ValueError("window must be >= 1")
        self.summarize_fn = summarize_fn
        self.window = window
        self.turns = []      # list of (user, assistant)
        self.summary = ""

    def add_turn(self, user, assistant):
        self.turns.append((user, assistant))
        while len(self.turns) > self.window:
            old = self.turns.pop(0)
            self.summary = self.summarize_fn(self.summary, old)

    def context(self):
        parts = []
        if self.summary:
            parts.append(f"Ringkasan percakapan sebelumnya: {self.summary}")
        for u, a in self.turns:
            parts.append(f"User: {u}")
            parts.append(f"Asisten: {a}")
        return "\n".join(parts)

    def stats(self):
        return {"turns_kept": len(self.turns), "has_summary": bool(self.summary)}

    def clear(self):
        self.turns = []
        self.summary = ""
```

- [ ] **Step 4: Run to verify PASS** — `cd 05_rag/tools && python3 -m pytest test_rag_utils.py -v` → all pass. Paste summary.

- [ ] **Step 5: Commit**
```bash
git add 05_rag/tools/rag_utils.py 05_rag/tools/test_rag_utils.py
git commit -m "feat(module05): add tested ConversationalMemoryManager (window+summary, injected summarizer) for nb06"
```

---

## Task 2: validator entry for nb06

**Files:** Modify `05_rag/tools/validate_notebooks.py`

- [ ] **Step 1:** Add `"06_conversational_rag.ipynb"` to REGISTRY (after nb05; don't change others):
```python
    "06_conversational_rag.ipynb": {
        "markers": [
            "ConversationalMemoryManager",     # tested hand-rolled memory
            "apply_chat_template",             # Qwen generation
            "bge-reranker-v2-m3",              # reranked retriever (from nb03)
            "rewrite_followup",                # history-aware query rewriting
            "create_history_aware_retriever",  # honest LangChain reference
        ],
        "forbidden": [
            "ConversationBufferMemory",        # deprecated LangChain memory; we hand-roll
        ],
    },
```
Update the trailing comment to "nb07..nb08".

- [ ] **Step 2:** `cd 05_rag/tools && python3 validate_notebooks.py 06_conversational_rag.ipynb` → `[FAIL]` (file not found). nb01-05 still PASS. Paste stdout.

- [ ] **Step 3: Commit**
```bash
git add 05_rag/tools/validate_notebooks.py
git commit -m "feat(module05): add nb06 validator markers (memory manager, history-aware rewrite, LangChain reference)"
```

---

## Task 3: build `06_conversational_rag.ipynb`

**Files:** Create `05_rag/06_conversational_rag.ipynb`. Self-contained, GPU; markdown Bahasa Indonesia, code/comments English. **Cell sequence:**

1. `M` — Tujuan: RAG satu-shot tak ingat percakapan. Masalah inti: pertanyaan lanjutan dengan kata ganti ("-nya", "itu") atau elipsis ("kalau yang sakit?") tak bisa di-retrieve apa adanya — embedding-nya tak nyambung ke dokumen mana pun tanpa konteks. Solusi: memori + history-aware rewriting.
2. `C` — install (`transformers<5`, `sentence-transformers>=3.0`, faiss-cpu, accelerate, bitsandbytes) + bootstrap (clone repo, `from tools.rag_utils import ConversationalMemoryManager`).
3. `M` — Dua jenis memori: **window** (N giliran terakhir, verbatim) + **summary** (Qwen meringkas giliran lama agar konteks tetap ringkas) + **history-aware rewriting** (tulis ulang pertanyaan lanjutan jadi pertanyaan MANDIRI sebelum retrieve). Kita hand-roll ketiganya (transparan + stabil); di akhir kita tunjukkan padanan library.
4. `C` — Pipeline recap (handbook corpus + embedder + `CrossEncoder("BAAI/bge-reranker-v2-m3")` + Qwen2.5-3B 4-bit, `BitsAndBytesConfig` NF4 fp16). Build FAISS over the 12-passage corpus.
5. `C` — Qwen helpers:
```python
def qwen(prompt, max_new_tokens=160):
    messages = [{"role": "system", "content": "Jawab ringkas dalam Bahasa Indonesia."},
                {"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to(gen.device)
    out = gen.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()

def summarize_turn(old_summary, turn):                  # injected into the memory manager
    user, assistant = turn
    return qwen(f"Ringkasan saat ini: {old_summary or '(kosong)'}\n"
                f"Tambahkan giliran ini ke ringkasan secara singkat:\nUser: {user}\nAsisten: {assistant}\n\nRingkasan baru:", 80)

def rewrite_followup(history_context, follow_up):       # history-aware query rewriting
    if not history_context.strip():
        return follow_up
    return qwen(f"Riwayat percakapan:\n{history_context}\n\nPertanyaan lanjutan: {follow_up}\n\n"
                f"Tulis ulang pertanyaan lanjutan menjadi satu pertanyaan MANDIRI yang lengkap "
                f"(ganti kata ganti seperti 'itu','-nya' dengan entitas dari riwayat). "
                f"Keluarkan HANYA pertanyaannya:", 60)
```
6. `C` — Retriever (rerank, from nb03) + memory:
```python
def retrieve_rerank(query, k_over=12, k_top=3):
    qv = embedder.encode([query], convert_to_numpy=True).astype("float32")
    _, idx = index.search(qv, k_over); cand = idx[0].tolist()
    scores = reranker.predict([(query, corpus[c]) for c in cand]).tolist()
    order = sorted(range(len(cand)), key=lambda i: scores[i], reverse=True)[:k_top]
    return [corpus[cand[i]] for i in order]

memory = ConversationalMemoryManager(summarize_fn=summarize_turn, window=4)
```
7. `M` — Alur satu giliran: rewrite (pakai riwayat) → retrieve+rerank (pakai query mandiri) → generate (jawaban + konteks) → simpan ke memori.
8. `C` — One conversational turn:
```python
def chat_turn(follow_up):
    standalone = rewrite_followup(memory.context(), follow_up)
    chunks = retrieve_rerank(standalone)
    ctx = "\n".join(f"- {c}" for c in chunks)
    answer = qwen(f"Konteks:\n{ctx}\n\nPertanyaan: {standalone}\n\nJawab HANYA dari konteks.")
    memory.add_turn(follow_up, answer)
    return standalone, answer
```
9. `M` — Demo percakapan (skrip): 4 giliran yang menguji memori — giliran 2-4 tak lengkap tanpa riwayat.
10. `C` — Scripted demo:
```python
memory.clear()
turns = [
    "Berapa hari cuti tahunan untuk pegawai tetap?",
    "Kalau cuti sakit?",                     # elipsis -> "Berapa hari cuti sakit?"
    "Apakah perlu surat dokter?",            # -> "Apakah cuti sakit perlu surat dokter?"
    "Bagaimana cara mengajukannya?",         # '-nya' -> "...mengajukan cuti?"
]
for t in turns:
    standalone, answer = chat_turn(t)
    print(f"User    : {t}")
    print(f"  (rewrite -> {standalone})")
    print(f"Asisten : {answer}\n{'-'*60}")
```
11. `M` — Baca hasil: lihat bagaimana "Kalau cuti sakit?" ditulis ulang jadi pertanyaan mandiri sehingga retrieval menemukan dokumen yang benar — tanpa rewriting, retrieval akan gagal.
12. `C` — `print(memory.stats())` + export percakapan ke JSON (`json.dump`).
13. `C` — (Opsional) loop interaktif:
```python
# Jalankan sel ini untuk ngobrol sendiri; ketik 'quit' untuk berhenti, 'stats'/'clear' untuk perintah.
while True:
    msg = input("Anda: ").strip()
    if msg.lower() == "quit": break
    if msg.lower() == "stats": print(memory.stats()); continue
    if msg.lower() == "clear": memory.clear(); print("memori dikosongkan"); continue
    s, a = chat_turn(msg); print(f"  (rewrite -> {s})\nAsisten: {a}")
```
14. `M` — **Referensi: begini cara library standar (LangChain) — tapi APInya sedang berpindah.** Honest reference (fenced code in markdown, NOT executed). Show the directly-analogous piece + the memory-API churn:
```text
# History-aware retrieval (padanan rewrite_followup kita) — pindah ke langchain_classic:
from langchain_classic.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "Berdasarkan riwayat, tulis ulang pertanyaan lanjutan jadi mandiri."),
    MessagesPlaceholder("chat_history"), ("human", "{input}")])
har = create_history_aware_retriever(llm, retriever, prompt)

# Penyimpanan memori di LangChain SEDANG BERPINDAH:
#   ConversationBufferMemory / ...WindowMemory / ...SummaryMemory  -> DEPRECATED (0.3.x)
#   RunnableWithMessageHistory                                     -> DEPRECATED (core 1.3.3)
#   Sekarang: LangGraph (InMemorySaver + MessagesState + trim_messages)
```
Note jujur: justru karena API library berganti 3x dalam ~2 tahun, kita hand-roll inti yang sederhana & stabil — dan kita paham mekanismenya.
15. `M` — Ringkasan + jembatan ke nb07 (Capstone: Ask-My-Document — satukan ingest+chunk, retrieve+rerank, conversational, jawaban bersitasi atas PDF yang diupload).

**Validator markers required:** `ConversationalMemoryManager`, `apply_chat_template`, `bge-reranker-v2-m3`, `rewrite_followup`, `create_history_aware_retriever`. **Forbidden:** `ConversationBufferMemory`.

**Steps:**
- [ ] **Step 1:** Author the notebook cell-by-cell (NotebookEdit insert).
- [ ] **Step 2:** Valid JSON.
- [ ] **Step 3:** Validator: `cd 05_rag/tools && python3 validate_notebooks.py 06_conversational_rag.ipynb` → `[PASS]`. nb01-05 still PASS.
- [ ] **Step 4 (manual GPU gate):** Colab T4 Run all — scripted demo shows the follow-ups rewritten into standalone questions + correct answers; memory.stats works; interactive loop optional.
- [ ] **Step 5: Commit**
```bash
git add 05_rag/06_conversational_rag.ipynb
git commit -m "feat(module05): add nb06 conversational RAG — hand-rolled window+summary memory + history-aware rewriting + LangChain reference"
```

---

## Self-Review
- Spec §5 nb06 coverage: ConversationalMemoryManager window+summary (Task 1 + cell 6) ✅; history-aware query rewriting (cell 5 `rewrite_followup`) ✅; contextual generation blending history (cell 8) ✅; interactive chat loop quit/stats/clear (cell 13) ✅; per-turn analyzer + exportable JSON (cell 12) ✅. Divergence (hand-rolled vs LangChain) documented; LangChain shown as honest reference (cell 14).
- Placeholders: scripted turns authored (cell 10). 
- Type/name consistency: `ConversationalMemoryManager(summarize_fn, window)` with `.add_turn/.context/.stats/.clear`; `summarize_turn`/`rewrite_followup`/`retrieve_rerank`/`chat_turn` defined before use; models `paraphrase-multilingual-MiniLM-L12-v2`, `BAAI/bge-reranker-v2-m3`, `Qwen/Qwen2.5-3B-Instruct` consistent with nb03/nb05 + validator markers.
- Risks: GPU memory (Qwen + reranker + embedder co-resident — same as nb03, fits T4 4-bit); rewriting quality depends on Qwen-3B (small model may rewrite imperfectly — scripted demo chosen so output is inspectable; frame honestly); LangChain reference is read-only markdown (not executed → no langchain/langgraph dep, no version-trap); summary memory only triggers after `window` turns (the 4-turn demo with window=4 may not evict — set window=2 in the demo OR note summary kicks in on longer chats). NOTE: set the demo to use a SMALLER window (e.g., rebuild `memory = ConversationalMemoryManager(summarize_fn=summarize_turn, window=2)` before the scripted demo) so the summary path is actually exercised and visible.
