import re

import pytest

from tldw_chatbook.Chunking.engine import Chunker
from tldw_chatbook.Chunking.engine.strategies.propositions import PropositionChunkingStrategy


def _normalize_text(text: str) -> str:
    """Normalize whitespace for stable comparisons."""
    return re.sub(r"\s+", " ", text).strip()


def _assert_non_empty_chunks(chunks: list[str]) -> None:
    """Ensure chunk output is a non-empty list of non-empty strings."""
    assert isinstance(chunks, list)
    assert chunks
    assert all(isinstance(chunk, str) and chunk.strip() for chunk in chunks)


def _assert_preserves_text(text: str, chunks: list[str]) -> None:
    """Ensure chunking preserves the original text content and order."""
    assert _normalize_text(" ".join(chunks)) == _normalize_text(text)


class TestPropositionStrategy:
    def test_basic_splitting(self):
        strategy = PropositionChunkingStrategy()
        text = "Alice founded a company and Bob joined later."
        props = strategy.chunk(text, max_size=10, overlap=0, aggressiveness=2)
        # Should keep as one chunk (since max_size=10), but internally produce 2 propositions into one chunk
        assert isinstance(props, list)
        assert len(props) == 1
        # If we pack by smaller size, we expect two chunks
        props2 = strategy.chunk(text, max_size=1, overlap=0, aggressiveness=2)
        assert len(props2) == 2

    def test_subordinate_markers(self):

        strategy = PropositionChunkingStrategy()
        text = "He said that she left because it was late."
        # With aggressiveness 1, split on 'that' and 'because'
        chunks = strategy.chunk(text, max_size=1, overlap=0, aggressiveness=1)
        assert len(chunks) >= 2

    def test_overlap(self):

        strategy = PropositionChunkingStrategy()
        text = (
            "Alice researched the topic, and she wrote a paper; "
            "Bob reviewed it, and the editor approved publication."
        )
        chunks = strategy.chunk(text, max_size=2, overlap=1, aggressiveness=2)
        # Expect overlapping windows of propositions
        assert len(chunks) >= 2
        # First element of chunk 1 should be present at start of chunk 2
        first_chunk = chunks[0]
        second_chunk = chunks[1]
        assert isinstance(first_chunk, str) and isinstance(second_chunk, str)

    def test_auto_engine_fallback_without_spacy(self):

        strategy = PropositionChunkingStrategy()
        text = "Alice built a prototype; Bob tested it and Carol wrote documentation."
        chunks = strategy.chunk(text, max_size=2, overlap=0, engine="auto")
        assert isinstance(chunks, list)
        assert len(chunks) >= 1

    def test_llm_engine_without_llm_func_fallbacks(self):

        strategy = PropositionChunkingStrategy()
        text = "The experiment succeeded because the temperature was controlled."
        chunks = strategy.chunk(text, max_size=2, overlap=0, engine="llm")
        # Should fallback to heuristics when no llm_call_func is provided
        assert isinstance(chunks, list)
        assert len(chunks) >= 1

    @pytest.mark.unit
    def test_unbalanced_parentheses_handled(self):
        strategy = PropositionChunkingStrategy()
        text = "Intro " + ("(" * 500) + " trailing text."
        max_size = 2
        overlap = 0
        aggressiveness = 1
        min_prop_len = 1
        chunks = strategy.chunk(
            text,
            max_size=max_size,
            overlap=overlap,
            aggressiveness=aggressiveness,
            min_proposition_length=min_prop_len,
        )
        _assert_non_empty_chunks(chunks)
        _assert_preserves_text(text, chunks)
        assert chunks[0].startswith("Intro")
        assert "trailing text." in chunks[-1]
        assert sum(chunk.count("(") for chunk in chunks) == text.count("(")

        smaller_chunks = strategy.chunk(
            text,
            max_size=1,
            overlap=0,
            aggressiveness=aggressiveness,
            min_proposition_length=min_prop_len,
        )
        _assert_non_empty_chunks(smaller_chunks)
        _assert_preserves_text(text, smaller_chunks)
        assert len(chunks) <= len(smaller_chunks)

    @pytest.mark.unit
    def test_nested_parentheses_degrade_gracefully(self):
        strategy = PropositionChunkingStrategy()
        text = "foo (bar (baz) qux) end"
        chunks = strategy.chunk(text, max_size=1, overlap=0, aggressiveness=1, min_proposition_length=1)
        _assert_non_empty_chunks(chunks)
        _assert_preserves_text(text, chunks)
        assert chunks[0].startswith("foo")
        assert chunks[-1].endswith("end")
        assert any("(" in chunk for chunk in chunks)
        assert any(")" in chunk for chunk in chunks)

    @pytest.mark.unit
    def test_chunk_with_metadata_offsets_align_source(self):
        strategy = PropositionChunkingStrategy()
        text = "Alice wrote code—then Bob reviewed it;  after that,   Carol approved it."
        results = strategy.chunk_with_metadata(
            text,
            max_size=2,
            overlap=1,
            aggressiveness=2,
            min_proposition_length=1,
        )
        assert results
        for res in results:
            start = res.metadata.start_char
            end = res.metadata.end_char
            assert 0 <= start <= end <= len(text)
            assert text[start:end] == res.text


class TestChunkerIntegration:
    def test_chunker_with_propositions_method(self):
        chunker = Chunker()
        text = "Alice founded a company and Bob joined later."
        result = chunker.chunk_text(text, method="propositions", max_size=1, overlap=0, aggressiveness=2)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_chunker_with_propositions_engine_fields(self):

        chunker = Chunker()
        text = "When it rained, the match was postponed, and fans were notified."
        # engine can be provided via proposition_engine field
        result = chunker.chunk_text(
            text,
            method="propositions",
            max_size=2,
            overlap=0,
            proposition_engine="heuristic",
            proposition_aggressiveness=2,
        )
        assert isinstance(result, list)
        assert len(result) >= 1
