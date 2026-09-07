import pytest

from tldw_chatbook.Chunking.engine.chunker import Chunker


def _assert_offsets_fidelity(text: str, chunks: list[dict]):
    for item in chunks:
        md = item.get("metadata") or {}
        s = md.get("start_offset")
        e = md.get("end_offset")
        assert isinstance(s, int) and isinstance(e, int), "Offsets must be integers"
        assert 0 <= s <= len(text), "start_offset out of bounds"
        assert 0 <= e <= len(text), "end_offset out of bounds"
        assert e >= s, "end_offset must be >= start_offset"
        assert text[s:e] == item.get("text", ""), "Chunk text must equal source slice"


def test_hierarchical_offsets_with_code_fences():

    text = "Intro line\n\n" "```\n" "print('hi')\n" "print('hi')\n" "```\n\n" "Outro line.\n"
    ck = Chunker()
    # Use sentences to mirror typical scraping config; keep small sizes to force chunk boundaries
    chunks = ck.chunk_text_hierarchical_flat(text, method="sentences", max_size=2, overlap=0)
    assert chunks, "Expected non-empty chunks"
    _assert_offsets_fidelity(text, chunks)
    # Ensure at least one chunk is tagged as a code fence block
    assert any((c.get("metadata") or {}).get("paragraph_kind") == "code_fence" for c in chunks)


def test_hierarchical_offsets_with_repeated_content_monotonic():

    # Repeated paragraphs and tokens can confuse naive substring mapping
    para = "Alpha Beta Alpha Beta Alpha Beta"
    text = f"{para}\n\n{para}\n\n{para}."
    ck = Chunker()
    chunks = ck.chunk_text_hierarchical_flat(text, method="words", max_size=4, overlap=0)
    assert chunks, "Expected non-empty chunks"
    _assert_offsets_fidelity(text, chunks)
    # Global monotonicity by document order
    starts = [(c.get("metadata") or {}).get("start_offset", -1) for c in chunks]
    assert all(isinstance(s, int) for s in starts)
    assert starts == sorted(starts), "Chunks should be emitted in non-decreasing document order"
