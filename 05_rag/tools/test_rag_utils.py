# 05_rag/tools/test_rag_utils.py
import pytest
from rag_utils import (
    simple_token_count, TokenCounter, split_sentences, Chunk,
    TextChunker, chunk_quality_score, rank_change_table, recall_at_k,
    ConversationalMemoryManager, format_pages, source_label, cited_labels,
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
    chunker = TextChunker(counter, max_tokens=5, overlap_words=0)
    text = "satu dua tiga. empat lima enam. tujuh delapan."
    chunks = chunker.sentence(text)
    assert all(c.n_tokens <= 5 for c in chunks)
    assert "".join(c.text for c in chunks).replace(" ", "") != ""  # non-empty coverage

def test_fixed_chunks_have_overlap():
    counter = TokenCounter(tokenize_fn=word_tok)
    chunker = TextChunker(counter, max_tokens=4, overlap_words=2)
    text = " ".join(f"w{i}" for i in range(10))  # 10 words
    chunks = chunker.fixed(text)
    assert len(chunks) >= 3
    # consecutive chunks share overlap_words words
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    assert first_words[-2:] == second_words[:2]

def test_semantic_chunks_split_on_blank_lines():
    counter = TokenCounter(tokenize_fn=word_tok)
    chunker = TextChunker(counter, max_tokens=100, overlap_words=0)
    text = "para satu kalimat.\n\npara dua kalimat lain."
    chunks = chunker.semantic(text)
    assert len(chunks) == 2

def test_chunk_quality_score_perfect_is_high():
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

def test_chunker_raises_on_bad_overlap():
    counter = TokenCounter(tokenize_fn=word_tok)
    with pytest.raises(ValueError):
        TextChunker(counter, max_tokens=4, overlap_words=4)   # overlap == max
    with pytest.raises(ValueError):
        TextChunker(counter, max_tokens=4, overlap_words=-1)  # negative

def test_sentence_oversized_single_sentence_passthrough():
    # a single sentence longer than the budget is emitted intact (documented behavior)
    counter = TokenCounter(tokenize_fn=word_tok)
    chunker = TextChunker(counter, max_tokens=3, overlap_words=0)
    chunks = chunker.sentence("satu dua tiga empat lima enam.")  # 6 words, one sentence
    assert len(chunks) == 1
    assert chunks[0].n_tokens > 3

def test_semantic_subchunks_oversized_paragraph():
    # a paragraph exceeding the budget falls back to sentence packing -> multiple chunks
    counter = TokenCounter(tokenize_fn=word_tok)
    chunker = TextChunker(counter, max_tokens=5, overlap_words=0)
    text = "satu dua tiga empat. lima enam tujuh delapan."  # one paragraph, 2 sentences, 8 words
    chunks = chunker.semantic(text)
    assert len(chunks) == 2
    assert all(c.n_tokens <= 5 for c in chunks)


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


def test_recall_at_k_perfect():
    assert recall_at_k([[1, 2, 3]], [[1, 2, 3]]) == 1.0

def test_recall_at_k_partial():
    assert recall_at_k([[1, 2, 9]], [[1, 2, 3]]) == 2 / 3   # 2 of 3 ground-truth found

def test_recall_at_k_mean_over_queries():
    # q1 perfect (1.0), q2 finds 1 of 2 (0.5) -> mean 0.75
    assert recall_at_k([[1, 2], [5, 6]], [[1, 2], [5, 9]]) == 0.75

def test_recall_at_k_empty():
    assert recall_at_k([], []) == 0.0


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


def test_format_pages_single():
    assert format_pages([6]) == "6"

def test_format_pages_consecutive_run_collapses():
    assert format_pages([6, 7, 8]) == "6–8"

def test_format_pages_gap_then_run():
    assert format_pages([6, 7, 9]) == "6–7, 9"

def test_format_pages_unsorted_and_deduped():
    assert format_pages([9, 6, 7, 6]) == "6–7, 9"

def test_format_pages_non_consecutive():
    assert format_pages([3, 5]) == "3, 5"

def test_format_pages_empty_is_placeholder():
    assert format_pages([]) == "?"

def test_source_label_with_pages_and_heading():
    assert source_label(1, [6, 7], "Multi-Head Attention") == "[S1] hlm 6–7 — Multi-Head Attention"

def test_source_label_without_heading():
    assert source_label(2, [3]) == "[S2] hlm 3"

def test_source_label_empty_pages():
    assert source_label(3, [], None) == "[S3] hlm ?"


def test_cited_labels_valid_in_range():
    assert cited_labels("Self-attention lebih cepat [S1] dan paralel [S3].", 4) == ([1, 3], [])

def test_cited_labels_flags_out_of_range():
    assert cited_labels("Klaim mengada-ada [S5].", 4) == ([], [5])

def test_cited_labels_dedups_and_sorts():
    assert cited_labels("[S2][S2][S1] lalu [S2]", 3) == ([1, 2], [])

def test_cited_labels_none_present():
    assert cited_labels("Tidak ada sitasi sama sekali.", 4) == ([], [])

def test_cited_labels_mixed_valid_invalid():
    assert cited_labels("[S1] benar, [S9] salah.", 4) == ([1], [9])
