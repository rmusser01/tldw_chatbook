"""Pins for TASK-18155's census helpers.

The census reached a NULL that retired a P2c premise, so its classification
logic is load-bearing. It also shipped a defect in its first run: every
`negative` query registered MISS *by construction* (an empty
``relevant_slugs`` can never produce a hit), which briefly inflated the
qualifying population. These pins exist so that class of error reds instead
of being read.

Pure helpers only -- no index, no `RAG_EVAL` gate.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_CENSUS = (
    Path(__file__).resolve().parents[2]
    / "Docs/superpowers/qa/2026-08-18-granularity-census/granularity_census.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("granularity_census", _CENSUS)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def census():
    if not _CENSUS.exists():                      # pragma: no cover
        pytest.skip(f"census script absent: {_CENSUS}")
    return _load()


def test_importable_without_the_rag_eval_gate(census):
    """The gate guards `main`, not import -- otherwise these pins cannot run."""
    assert callable(census.classify)
    assert callable(census.parity_text)


class TestClassify:
    def test_a_hit_is_never_qualifying(self, census):
        """A query already retrieving its target cannot be RESCUED."""
        verdict, _ = census.classify(("doc-a",), "keyword", "semantic", hit=True)
        assert verdict == "hit"

    def test_a_reachable_miss_qualifies(self, census):
        verdict, reason = census.classify(("doc-a",), "keyword", "semantic", hit=False)
        assert verdict == "qualifying"
        assert reason == ""

    def test_negative_query_is_excluded_not_qualifying(self, census):
        """THE BUG THIS FILE EXISTS FOR.

        A `negative` has no relevant slug, so `hit` is False no matter what
        retrieval did. Counting it as a rescuable miss inflates the
        population with queries whose miss is the CORRECT outcome.
        """
        verdict, reason = census.classify((), "negative", "semantic", hit=False)
        assert verdict == "excluded"
        assert reason == census.EXCLUDED_NEGATIVE

    def test_prompt_in_semantic_is_excluded_no_vector_index(self, census):
        """Prompts have an FTS sub-leg and deliberately no vector index, so
        no freed slot can admit a document that is not in the index."""
        verdict, reason = census.classify(
            ("prompt-x",), "prompt", "semantic", hit=False
        )
        assert verdict == "excluded"
        assert reason == census.EXCLUDED_UNINDEXED

    def test_prompt_in_hybrid_is_NOT_excluded(self, census):
        """Hybrid reaches prompts through its keyword leg, so the
        vector-index exclusion must not leak into that mode."""
        verdict, _ = census.classify(("prompt-x",), "prompt", "hybrid", hit=False)
        assert verdict == "qualifying"


class TestParityText:
    def test_conversation_gets_the_sender_prefix(self, census):
        """`conversation_document` indexes ``f"{sender}: {content}"``; chunking
        the raw fixture text would measure a shorter document than exists."""
        assert census.parity_text("conversation", "hello") == "user: hello"

    @pytest.mark.parametrize("source_type", ["note", "media", "prompt"])
    def test_other_source_types_are_unchanged(self, census, source_type):
        assert census.parity_text(source_type, "hello") == "hello"

    def test_prefix_lengthens_the_text_it_chunks(self, census):
        """The point of the parity fix: the indexed text is strictly longer,
        which is what can move a document across a chunk boundary."""
        raw = "word " * 10
        assert len(census.parity_text("conversation", raw).split()) == len(raw.split()) + 1


class TestRegisteredConstants:
    def test_bar_is_the_inherited_five(self, census):
        """Inherited verbatim from PRF clause 1 and the clarification gate --
        not chosen to fit this arc's result."""
        assert census.BAR == 5

    def test_plain_is_not_measured(self, census):
        """`plain` returns whole items, so it is already document-granular
        and structurally cannot move."""
        assert "plain" not in census.MODES
        assert census.MODES == ("semantic", "hybrid")
