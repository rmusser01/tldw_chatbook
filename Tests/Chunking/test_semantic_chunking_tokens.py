import pytest

from tldw_chatbook.Chunking.engine.strategies.semantic import SemanticChunkingStrategy


def test_semantic_chunking_tokens_unit_runs_without_tokenizer():

    strategy = SemanticChunkingStrategy()
    if not strategy._sklearn_available:
        pytest.skip("scikit-learn not available for semantic chunking")

    text = "Sentence one. Sentence two. Sentence three."
    chunks = strategy.chunk(text, max_size=6, overlap=0, unit="tokens")
    assert chunks, "Expected semantic chunking to return chunks with token unit"
