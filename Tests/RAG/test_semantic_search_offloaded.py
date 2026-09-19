"""The Library semantic query must run off the Textual event loop (TASK-32804.9).

ChromaDB's synchronous query() was called directly inside the async
`_semantic_search`, freezing the whole UI ~212 ms on the first (cold) search
while every sibling on the path (service construction, get_stats, the embedding
half) was already offloaded via asyncio.to_thread. This pins the store call to a
worker thread deterministically (by thread identity, not timing).
"""

import asyncio
import threading
from types import SimpleNamespace

from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService


def _run(include_citations):
    main_ident = threading.get_ident()
    seen = {}

    def _record(*_a, **_k):
        seen["thread"] = threading.get_ident()
        return []  # empty results -> the rest of _semantic_search short-circuits

    class _Embeddings:
        async def create_embeddings_async(self, _texts):
            return [[0.1, 0.2, 0.3]]

    vector_store = SimpleNamespace(
        search_with_citations=_record,
        search=_record,
    )
    fake_self = SimpleNamespace(embeddings=_Embeddings(), vector_store=vector_store)

    asyncio.run(
        RAGService._semantic_search(
            fake_self, "query", top_k=5, include_citations=include_citations
        )
    )
    return main_ident, seen["thread"]


def test_semantic_query_with_citations_runs_on_a_worker_thread():
    main_ident, store_ident = _run(include_citations=True)
    assert store_ident != main_ident, "the store query ran on the event-loop thread"


def test_semantic_query_without_citations_runs_on_a_worker_thread():
    main_ident, store_ident = _run(include_citations=False)
    assert store_ident != main_ident, "the store query ran on the event-loop thread"
