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


def test_corpus_ingests_and_semantic_search_finds_a_planted_doc(tmp_path):
    from tldw_chatbook.Library.library_local_rag_search_service import (
        LibraryLocalRagSearchService,
    )

    from Tests.RAG_Eval.harness.goldenset import CORPUS_PATH, load_corpus
    from Tests.RAG_Eval.harness.ingest import build_eval_runtime

    corpus = load_corpus(CORPUS_PATH)
    runtime = build_eval_runtime(corpus, tmp_path)
    try:
        # Every fixture document reached a real source DB and was mapped.
        assert len(runtime.slug_to_source) == len(corpus)

        # Every fixture document reached the vector store. Chunks, not
        # documents, so `>=`: the long fixtures split into several.
        stats = runtime.service.vector_store.get_collection_stats()
        assert not stats.get("error"), stats
        assert stats["count"] >= len(corpus)

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
