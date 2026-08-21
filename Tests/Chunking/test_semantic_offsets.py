import pytest

from tldw_chatbook.Chunking.engine.strategies.semantic import SemanticChunkingStrategy


def test_semantic_offsets_match_source_slices():
    strategy = SemanticChunkingStrategy()
    if not getattr(strategy, "_sklearn_available", False):
        pytest.skip("scikit-learn not available for semantic chunking")

    text = "Alpha one.\n\nBeta two?  Gamma three!"
    chunks = strategy.chunk_with_metadata(text, max_size=1, overlap=0, unit="characters")

    assert chunks, "Expected semantic chunking to return chunks"
    for chunk in chunks:
        start = chunk.metadata.start_char
        end = chunk.metadata.end_char
        assert text[start:end] == chunk.text
