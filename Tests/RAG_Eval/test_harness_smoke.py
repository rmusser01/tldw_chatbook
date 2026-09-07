# Tests/RAG_Eval/test_harness_smoke.py
"""Env-gated smoke test for the retrieval eval harness runtime.

This is the one test that proves the harness is a *measuring instrument* and
not a mock: it writes the fixture corpus through the app's real writer APIs,
indexes it through the app's real indexing helper into an isolated vector
store, and then retrieves through the **production seam**
(`LibraryLocalRagSearchService`) rather than by calling the engine directly.
Every layer Task 6 will measure is therefore exercised here first, once.

It is skipped unless `RAG_EVAL=1` *and* the embeddings extras are installed
*and* the embedding model is already in the local model cache — see
`Tests/RAG_Eval/harness/environment.py` for the exact gate. The always-on
Task 2/3/4 modules in this directory carry no such gate.
"""
from __future__ import annotations

from Tests.RAG_Eval.harness.environment import harness_gate

pytestmark = harness_gate()

#: The planted keyword-exact fixture: a rare literal token ("Zephyr-9") that
#: exists in exactly one corpus document, so a hit is unambiguous.
PLANTED_SLUG = "note-zephyr-flywheel"
PLANTED_QUERY = "Zephyr-9 flywheel assembly balance tolerance"


def test_model_downloads_are_blocked_for_the_duration_of_a_harness_run():
    """Offline mode is *in effect*, not merely requested.

    Asserts the resolved state rather than the environment variable on
    purpose. `HF_HUB_OFFLINE` is frozen into
    `huggingface_hub.constants.HF_HUB_OFFLINE` at import time, so setting the
    env var from a fixture leaves the env reading "1" while offline mode is
    still off — exactly the inert configuration this test exists to catch.
    A regression here means a cache miss downloads ~87 MB into the user's
    real cache instead of failing.
    """
    from huggingface_hub import constants
    from transformers.utils.hub import is_offline_mode as transformers_is_offline

    assert constants.is_offline_mode() is True
    # transformers reaches the same global through its own import; if that
    # ever stops being true, hf_hub being offline would not stop it.
    assert transformers_is_offline() is True


def test_a_corpus_with_nothing_semantically_indexable_is_refused(tmp_path):
    """An empty VECTOR index must NOT be built quietly.

    DISCLOSED UPDATE (2026-08-11, TASK-15020/B2): this used to be "a corpus
    of only UNWRITABLE documents", because prompts had no writer. They have
    one now, so a prompts-only corpus is written and keyword-retrievable —
    and still has an empty vector index, which is the condition that
    actually matters. Semantic would score 0.000 on every query and read as
    total retrieval failure; worse than before, hybrid would report real
    numbers beside it, so the confusion is now MORE plausible rather than
    less.
    """
    import pytest

    from Tests.RAG_Eval.harness.goldenset import CorpusDoc
    from Tests.RAG_Eval.harness.ingest import EvalRuntimeError, build_eval_runtime

    only_prompts = [
        CorpusDoc("p1", "prompt", "Prompt one", "Do this. Then that. Then stop."),
    ]
    with pytest.raises(EvalRuntimeError) as excinfo:
        build_eval_runtime(only_prompts, tmp_path)
    assert "no semantically indexable document" in str(excinfo.value)


def test_corpus_ingests_and_semantic_search_finds_a_planted_doc(tmp_path):
    from tldw_chatbook.Library.library_local_rag_search_service import (
        LibraryLocalRagSearchService,
    )

    from Tests.RAG_Eval.harness.goldenset import CORPUS_PATH, load_corpus
    from Tests.RAG_Eval.harness.ingest import build_eval_runtime

    from Tests.RAG_Eval.harness.ingest import UNINDEXED_SOURCE_TYPES

    corpus = load_corpus(CORPUS_PATH)
    runtime = build_eval_runtime(corpus, tmp_path)
    try:
        # TWO-SIDED ACCOUNTING (updated for TASK-15020/B2). Before B2 the
        # two sides were "written" and "skipped"; prompts are written now,
        # so the split that remains is "in the vector index" vs "in a source
        # DB and reachable only through the keyword leg". Every fixture is
        # mapped either way — being in `slug_to_source` is what makes a
        # document scoreable — and the unindexed set is stated rather than
        # inferred, because a prompt scoring 0.000 in SEMANTIC is a
        # structural fact and a note scoring 0.000 in semantic is a finding.
        # A bare `>=` here would have hidden a silently dropped note.
        assert len(runtime.slug_to_source) == len(corpus)
        assert set(runtime.unindexed) == {
            doc.slug for doc in corpus if doc.source_type in UNINDEXED_SOURCE_TYPES
        }
        assert runtime.unindexed, (
            "nothing is unindexed, so this accounting proves nothing — the "
            "corpus should still carry the prompt fixtures"
        )

        # Every INDEXABLE fixture document reached the vector store. Chunks,
        # not documents, so `>=`: the long fixtures split into several.
        indexable = [
            doc for doc in corpus if doc.source_type not in UNINDEXED_SOURCE_TYPES
        ]
        stats = runtime.service.vector_store.get_collection_stats()
        assert not stats.get("error"), stats
        assert stats["count"] >= len(indexable)

        # ... and comes back through the PRODUCTION seam, not the engine.
        seam = LibraryLocalRagSearchService(runtime.app)
        result = runtime.run(
            seam.search(
                PLANTED_QUERY,
                ("media", "notes", "conversations"),
                "rag",
                top_k=5,
            )
        )
        rows = result["results"]
        expected_source_type, expected_source_id = runtime.slug_to_source[PLANTED_SLUG]
        hits = {
            (row["provenance"].get("source_type"), row["source_id"]) for row in rows
        }
        assert (expected_source_type, expected_source_id) in hits, (
            f"planted doc {PLANTED_SLUG!r} "
            f"({expected_source_type} {expected_source_id}) missing from top 5: {hits}"
        )
    finally:
        runtime.close()
