"""Behavioral coverage for the Chunk_Lib shim's backward-compat functions.

Task 4 review M3: the ported test_chunker_v2.py tests for
``improved_chunking_process`` and ``chunk_for_embedding`` are skipped because
those functions live in the server's package __init__ (not vendored — spec
§5.1); chatbook's compat equivalents live in the Chunk_Lib shim. This file
ports the upstream assertions (dev @ 385afa95,
tldw_Server_API/tests/Chunking/test_chunker_v2.py::TestBackwardCompatibility)
against the SHIM implementations, so the skip is a re-pointing, not a hole.

Signature note: upstream's chunk_for_embedding(text, file_name, **kwargs)
accepts chunk options as kwargs; the shim keeps chatbook's legacy explicit
signature (custom_chunk_options dict). The upstream assertion set is
signature-agnostic (list-of-dicts, non-empty), so it ports directly.
"""
from tldw_chatbook.Chunking.Chunk_Lib import (
    chunk_for_embedding,
    improved_chunking_process,
)


def test_improved_chunking_process_backcompat():
    """Upstream: TestBackwardCompatibility::test_improved_chunking_process."""
    text = "Test text. Another sentence. Third sentence."
    options = {"method": "sentences", "max_size": 2, "overlap": 1}

    chunks = improved_chunking_process(text, options)

    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(isinstance(chunk, dict) for chunk in chunks)
    assert all("text" in chunk for chunk in chunks)
    assert all("metadata" in chunk for chunk in chunks)


def test_chunk_for_embedding_backcompat():
    """Upstream: TestBackwardCompatibility::test_chunk_for_embedding.

    The upstream call passes max_size as a kwarg; the shim's legacy signature
    takes the options dict (chatbook's pre-existing contract).
    """
    text = "Test text for embedding. " * 10
    chunks = chunk_for_embedding(
        text, "test_file.txt", custom_chunk_options={"max_size": 50}
    )

    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(isinstance(chunk, dict) for chunk in chunks)


def test_chunk_for_embedding_headers_wrap_source_text():
    """Shim-specific contract: every chunk wraps the original text with the
    document/chunk header envelope the embedding path depends on."""
    text = "Alpha sentence one. Beta sentence two. Gamma sentence three."
    chunks = chunk_for_embedding(
        text, "doc.txt", custom_chunk_options={"method": "sentences", "max_size": 1}
    )

    assert len(chunks) >= 2
    for i, chunk in enumerate(chunks, start=1):
        assert chunk["source_document_name"] == "doc.txt"
        assert "[DOCUMENT: doc.txt]" in chunk["text_for_embedding"]
        assert f"[CHUNK: {i} OF {len(chunks)}]" in chunk["text_for_embedding"]
        assert "---BEGIN CHUNK CONTENT---" in chunk["text_for_embedding"]
        assert "---END CHUNK CONTENT---" in chunk["text_for_embedding"]
        # the original, un-wrapped text is preserved verbatim
        assert chunk["original_chunk_text"] in text
        assert chunk["original_chunk_text"] in chunk["text_for_embedding"]
        assert chunk["chunk_metadata"]["chunk_index"] == i
