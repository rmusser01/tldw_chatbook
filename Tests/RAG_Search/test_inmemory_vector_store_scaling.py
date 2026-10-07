"""B3: InMemoryVectorStore results identical to naive cosine; LRU/dedupe
semantics preserved; scaling measured against the per-vector baseline.

Review-B finding: ``InMemoryVectorStore.search`` computed each candidate's
cosine with a fresh ``np.linalg.norm`` of the QUERY per row, ranked with a
full Python sort, deduped adds with ``id in self.ids`` (an O(n) list scan)
and maintained the LRU as a plain list (O(n) remove per touch). The rewrite
keeps the public surface and the observable semantics -- identical ranking,
score normalization, LRU eviction order, dedupe-replace, cap behavior -- and
changes only the internals (dict id index, OrderedDict LRU, one cached
matrix + row norms, one matmul per search).

The naive-cosine oracle below is the golden baseline: it was captured
against the pre-rewrite implementation and must keep passing afterwards.
"""

import time
from typing import List

import numpy as np
import pytest

from tldw_chatbook.RAG_Search.simplified.vector_store import InMemoryVectorStore


def _make_store(n: int, dim: int, seed: int = 7, **kwargs) -> InMemoryVectorStore:
    """A store with ``n`` random unit-scale vectors of ``dim`` dimensions."""
    rng = np.random.default_rng(seed)
    embeddings = rng.normal(size=(n, dim)).astype(np.float32)
    store = InMemoryVectorStore(**kwargs)
    store.add(
        ids=[f"doc-{i}" for i in range(n)],
        embeddings=embeddings,
        documents=[f"document {i}" for i in range(n)],
        metadata=[{"chunk_index": i} for i in range(n)],
    )
    return store


def _naive_search(store: InMemoryVectorStore, query, k: int) -> List[int]:
    """The pre-rewrite ranking, recomputed independently: cosine with
    per-vector normalization (zero-vector epsilon guard), stable descending
    sort by score, lowest index winning ties."""
    q = np.asarray(query, dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-12)
    scored = []
    for i, emb in enumerate(store.embeddings):
        e = np.asarray(emb, dtype=np.float32)
        e = e / (np.linalg.norm(e) + 1e-12)
        scored.append((float(np.dot(q, e)), i))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [i for _, i in scored[:k]]


def test_search_matches_naive_cosine():
    store = _make_store(50, 8)
    query = [0.1] * 8
    got = [r.id for r in store.search(query, top_k=5)]
    want = [store.ids[i] for i in _naive_search(store, query, 5)]
    assert got == want


def test_search_scores_match_naive_within_tolerance():
    """Scores (already (score+1)/2 normalized) track the oracle's ranking
    scores to float32 dot-product precision."""
    store = _make_store(50, 8)
    query = np.asarray([0.3, -0.2, 0.1, 0.05, 0.4, 0.0, -0.1, 0.25], dtype=np.float32)
    results = store.search(query, top_k=10)

    naive = _naive_search(store, query, 10)
    assert [r.id for r in results] == [store.ids[i] for i in naive]
    for result, idx in zip(results, naive):
        row = np.asarray(store.embeddings[idx], dtype=np.float32)
        row = row / (np.linalg.norm(row) + 1e-12)
        qn = query / (np.linalg.norm(query) + 1e-12)
        expected_normalized = (float(np.dot(qn, row)) + 1) / 2
        assert result.score == pytest.approx(expected_normalized, abs=1e-6)


def test_lru_eviction_and_dedupe():
    store = InMemoryVectorStore(max_documents=3)
    store.add(
        ids=["a", "b", "c"],
        embeddings=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        documents=["a", "b", "c"],
        metadata=[{}, {}, {}],
    )
    store.search([1.0, 0.0], top_k=1)  # returns only "a" -> most recently used
    store.add(
        ids=["d"],
        embeddings=[[0.5, 0.5]],
        documents=["d"],
        metadata=[{}],
    )  # evicts LRU, which is "b" (not "a")
    assert set(store.ids) == {"a", "c", "d"}
    store.add(
        ids=["a"],
        embeddings=[[0.9, 0.1]],
        documents=["a2"],
        metadata=[{"replaced": True}],
    )  # dedupe: replaces, does not duplicate
    assert store.ids.count("a") == 1 and len(store.embeddings) == 3
    # The replacement took hold (documents/metadata updated in place).
    hit = [r for r in store.search([0.9, 0.1], top_k=1)]
    assert hit and hit[0].metadata.get("replaced") is True


def test_eviction_at_cap_keeps_search_correct():
    """After several evictions the matrix cache must reflect the survivors,
    and ranking must still match the oracle over the remaining rows."""
    store = InMemoryVectorStore(max_documents=10)
    rng = np.random.default_rng(11)
    dim = 6
    for round_no in range(5):
        batch = rng.normal(size=(4, dim)).astype(np.float32)
        store.add(
            ids=[f"r{round_no}-{i}" for i in range(4)],
            embeddings=batch,
            documents=[f"r{round_no}-{i}" for i in range(4)],
            metadata=[{} for _ in range(4)],
        )
    assert len(store.ids) == 10
    query = rng.normal(size=dim)
    got = [r.id for r in store.search(query, top_k=4)]
    want = [store.ids[i] for i in _naive_search(store, query, 4)]
    assert got == want


def test_allowlist_scoped_search_matches_naive():
    """Scoped (metadata_allowlist) search still ranks the in-scope candidates
    exactly as the oracle would over the same subset."""
    store = _make_store(30, 5, seed=3)
    for i in range(0, 30, 2):
        store.metadata[i]["in_scope"] = True
    query = [0.2] * 5
    results = store.search(query, top_k=5, metadata_allowlist={"in_scope": {"True"}})
    got = [r.id for r in results]
    allowed = [
        i
        for i, meta in enumerate(store.metadata)
        if meta.get("in_scope") is True
    ]
    scored = sorted(
        ((i, float(np.dot(np.asarray(query, dtype=np.float32)
                          / (np.linalg.norm(query) + 1e-12),
                          np.asarray(store.embeddings[i], dtype=np.float32)
                          / (np.linalg.norm(store.embeddings[i]) + 1e-12))))
         for i in allowed),
        key=lambda pair: pair[1],
        reverse=True,
    )
    want = [store.ids[i] for i, _ in scored[:5]]
    assert got == want


def test_scaling_benchmark_100_queries_over_5000_vectors():
    """Records the 5000x16 index + 100-query wall time (evidence for B3).

    The number itself is printed for the task notes; the assertion only
    guards against pathological regressions so the benchmark stays runnable.
    """
    store = _make_store(5000, 16, seed=5)
    rng = np.random.default_rng(6)
    queries = rng.normal(size=(100, 16))
    start = time.perf_counter()
    for q in queries:
        store.search(q, top_k=10)
    elapsed = time.perf_counter() - start
    print(f"\nB3 scaling: 100 queries over 5000x16 vectors in {elapsed:.3f}s")
    assert elapsed < 30.0
