"""Contract tests for ChunkingService across every supported method."""

from __future__ import annotations

import pytest


class TestEveryChunkingMethodReturnsUsableText:
    """Every supported method must yield dicts whose ``text`` is a string.

    ``Chunker.chunk_text`` is not uniform: the text methods (words, sentences,
    paragraphs, tokens, semantic) return plain strings, while the
    structure-aware ones (json, xml, ebook_chapters) return dicts carrying their
    text alongside metadata. ``ChunkingService`` assumed strings and called
    ``.split()`` on them, so **every json and xml chunk request died** with
    "'dict' object has no attribute 'split'" (task-841).

    Found by exercising each method rather than by reading the dispatch, which
    looks uniform. This is the third instance in one session of a caller
    assuming a shape its callee does not produce -- see
    ``backlog/docs/lessons-testing-evidence.md``.
    """

    PROSE = (
        "The first sentence is here. The second follows it. A third arrives.\n\n"
        "A second paragraph begins. It has two sentences.\n\nThird paragraph."
    ) * 2

    CASES = [
        ("words", PROSE),
        ("sentences", PROSE),
        ("paragraphs", PROSE),
        ("tokens", PROSE),
        ("semantic", PROSE),
        ("json", '{"data": {"k1": "some text here", "k2": "more", "k3": "third"}}'),
        ("xml", "<root><item>alpha beta gamma</item><item>delta epsilon</item></root>"),
    ]

    @pytest.mark.parametrize(("method", "content"), CASES)
    def test_method_yields_string_text(self, method, content):
        from tldw_chatbook.RAG_Search.chunking_service import ChunkingService

        try:
            chunks = ChunkingService().chunk_text(
                content, chunk_size=20, chunk_overlap=5, method=method
            )
        except Exception as exc:  # noqa: BLE001 - classified below
            # ``semantic`` needs NLTK's punkt tokeniser data, which is a runtime
            # download rather than a declared dependency. Skip when it is absent
            # rather than fail: the contract under test is the chunk SHAPE, and
            # pretending the data gap is a shape bug would bury both.
            if "punkt" in str(exc) or "NLTK" in str(exc):
                pytest.skip(f"{method} needs NLTK data that is not installed here")
            raise

        assert chunks, f"{method} produced no chunks at all"
        for chunk in chunks:
            assert isinstance(chunk, dict), f"{method} yielded {type(chunk).__name__}"
            assert isinstance(chunk.get("text"), str), (
                f"{method} chunk text is {type(chunk.get('text')).__name__}, "
                "not a string -- downstream .split() and storage both break"
            )
            assert chunk["text"].strip(), f"{method} yielded an empty chunk"

    def test_a_structured_chunk_without_a_text_field_is_not_dropped(self):
        """Serialise rather than discard: content still reaches the index."""
        from tldw_chatbook.RAG_Search.chunking_service import _chunk_to_text

        assert _chunk_to_text({"heading": "H", "body": "B"})
        assert _chunk_to_text({"text": "plain"}) == "plain"
        assert _chunk_to_text("already a string") == "already a string"
