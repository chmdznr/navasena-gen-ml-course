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
