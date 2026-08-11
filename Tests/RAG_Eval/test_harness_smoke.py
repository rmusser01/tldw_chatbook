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


def test_a_corpus_of_only_unwritable_documents_is_refused(tmp_path):
    """The one skip that must NOT be quiet.

    Skipping the prompt fixtures is the measurement; skipping *everything*
    is an empty index, and an empty index scores 0.000 on every query in
    every mode — which reads as total retrieval failure rather than as "the
    harness wrote nothing". Named here so that failure has a cause attached
    to it.
    """
    import pytest

    from Tests.RAG_Eval.harness.goldenset import CorpusDoc
    from Tests.RAG_Eval.harness.ingest import EvalRuntimeError, build_eval_runtime

    only_prompts = [
        CorpusDoc("p1", "prompt", "Prompt one", "Do this. Then that. Then stop."),
    ]
    with pytest.raises(EvalRuntimeError) as excinfo:
        build_eval_runtime(only_prompts, tmp_path)
    assert "unwritable" in str(excinfo.value)


def test_corpus_ingests_and_semantic_search_finds_a_planted_doc(tmp_path):
    from tldw_chatbook.Library.library_local_rag_search_service import (
        LibraryLocalRagSearchService,
    )

    from Tests.RAG_Eval.harness.goldenset import CORPUS_PATH, load_corpus
    from Tests.RAG_Eval.harness.ingest import build_eval_runtime

    from Tests.RAG_Eval.harness.ingest import UNWRITABLE_SOURCE_TYPES

    corpus = load_corpus(CORPUS_PATH)
    runtime = build_eval_runtime(corpus, tmp_path)
    try:
        # Every WRITABLE fixture document reached a real source DB and was
        # mapped. The prompt fixtures are deliberately not written (there is
        # no prompts writer and no seam that would serve them), so the
        # accounting is stated in both directions rather than loosened: the
        # skipped set is exactly the unwritable source types, and everything
        # else is present. A bare `>=` here would have hidden a silently
        # dropped note.
        writable = [
            doc for doc in corpus if doc.source_type not in UNWRITABLE_SOURCE_TYPES
        ]
        assert len(runtime.slug_to_source) == len(writable)
        assert set(runtime.unwritable) == {
            doc.slug for doc in corpus if doc.source_type in UNWRITABLE_SOURCE_TYPES
        }
        assert runtime.unwritable, (
            "no document was skipped, so this accounting proves nothing — the "
            "corpus should still carry the prompt fixtures"
        )

        # Every written fixture document reached the vector store. Chunks,
        # not documents, so `>=`: the long fixtures split into several.
        stats = runtime.service.vector_store.get_collection_stats()
        assert not stats.get("error"), stats
        assert stats["count"] >= len(writable)

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
