from tldw_chatbook.Chunking.engine import Chunker


def test_hierarchical_rewrite_method_offsets_disabled():
    text = "First sentence, with punctuation. Second sentence; with clauses."
    chunker = Chunker()

    chunks = chunker.chunk_text_hierarchical_flat(
        text,
        method="propositions",
        max_size=2,
        overlap=0,
    )

    assert chunks, "Expected hierarchical chunking to return chunks"
    for ch in chunks:
        md = ch.get("metadata", {})
        assert md.get("offsets_valid") is False
        assert md.get("start_offset") is None
        assert md.get("end_offset") is None
