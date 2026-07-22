import pytest

from tldw_chatbook.RAG_Search.simplified.config import (
    RAGConfig, EmbeddingConfig, ChunkingConfig, VectorStoreConfig,
)
from tldw_chatbook.RAG_Search.simplified.collection_fingerprint import (
    fingerprinted_collection_name,
)
from tldw_chatbook.RAG_Search.simplified.collection_indexes import (
    adopt_legacy_collection,
    maybe_adopt_legacy_collection,
)


def _cfg(persist_dir, distance_metric="cosine"):
    return RAGConfig(
        embedding=EmbeddingConfig(model="mock", device="cpu"),
        chunking=ChunkingConfig(chunk_size=400, chunk_overlap=100),
        vector_store=VectorStoreConfig(
            type="chroma", persist_directory=persist_dir,
            collection_name="default", distance_metric=distance_metric),
    )


def _seed_legacy(persist_dir, name="default", n=3, hnsw_space="cosine"):
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path=str(persist_dir),
                                       settings=Settings(anonymized_telemetry=False, allow_reset=True))
    col = client.get_or_create_collection(name=name, metadata={"hnsw:space": hnsw_space})
    col.add(ids=[f"id{i}" for i in range(n)],
            embeddings=[[float(i)] * 8 for i in range(n)],
            documents=[f"doc {i}" for i in range(n)])
    return client


@pytest.mark.requires_chromadb
def test_adopt_moves_docs_and_removes_legacy(chroma_persist_dir):
    _seed_legacy(chroma_persist_dir)
    cfg = _cfg(chroma_persist_dir)
    target = fingerprinted_collection_name(cfg)
    maybe_adopt_legacy_collection(cfg)

    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path=str(chroma_persist_dir),
                                       settings=Settings(anonymized_telemetry=False, allow_reset=True))
    names = [c.name for c in client.list_collections()]
    assert target in names and "default" not in names
    assert client.get_collection(target).count() == 3
    assert client.get_collection(target).metadata.get("source") == "legacy-adopted"
    assert client.get_collection(target).metadata.get("verified") is False


@pytest.mark.requires_chromadb
def test_adopt_is_idempotent(chroma_persist_dir):
    _seed_legacy(chroma_persist_dir)
    cfg = _cfg(chroma_persist_dir)
    maybe_adopt_legacy_collection(cfg)
    maybe_adopt_legacy_collection(cfg)  # second run must not raise or duplicate
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path=str(chroma_persist_dir),
                                       settings=Settings(anonymized_telemetry=False, allow_reset=True))
    assert client.get_collection(fingerprinted_collection_name(cfg)).count() == 3


@pytest.mark.requires_chromadb
def test_no_legacy_is_noop(chroma_persist_dir):
    cfg = _cfg(chroma_persist_dir)
    maybe_adopt_legacy_collection(cfg)  # nothing to adopt
    assert adopt_legacy_collection(
        chroma_persist_dir, "default", fingerprinted_collection_name(cfg), {}
    ) is False


def test_memory_type_is_noop():
    cfg = RAGConfig(vector_store=VectorStoreConfig(type="memory", collection_name="default"))
    maybe_adopt_legacy_collection(cfg)  # must not touch disk / raise


@pytest.mark.requires_chromadb
def test_target_exists_is_noop(chroma_persist_dir):
    """If the fingerprinted target already exists, adoption must not touch it
    or the (still-present) legacy collection."""
    _seed_legacy(chroma_persist_dir)
    cfg = _cfg(chroma_persist_dir)
    target = fingerprinted_collection_name(cfg)

    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path=str(chroma_persist_dir),
                                       settings=Settings(anonymized_telemetry=False, allow_reset=True))
    client.get_or_create_collection(name=target, metadata={"source": "built"})

    result = adopt_legacy_collection(
        chroma_persist_dir, "default", target, {"source": "legacy-adopted", "verified": False}
    )
    assert result is False
    names = [c.name for c in client.list_collections()]
    assert "default" in names  # legacy left untouched
    assert client.get_collection(target).count() == 0  # pre-existing target untouched


@pytest.mark.requires_chromadb
def test_create_rag_service_adopts_legacy_collection_e2e(chroma_persist_dir):
    """Regression lock for the client-Settings-collision landmine.

    ``maybe_adopt_legacy_collection``'s own Chroma client
    (``collection_indexes._client``) and the real service's lazily-built
    Chroma client (``ChromaVectorStore.client``) run sequentially in one
    process against the *same* ``persist_directory``. chromadb (1.5.8)
    caches one client instance per persist path (``SharedSystemClient``) for
    the whole process and raises ``ValueError`` if a later
    ``PersistentClient(...)`` call at that path uses ``Settings`` that
    aren't ``==`` the first-registered ones. The two tests above only
    exercise the migration function in isolation; this drives the real
    ``create_rag_service`` entry point end-to-end so the second, real-service
    client construction actually happens in-process and the collision would
    surface if the Settings ever drifted apart.
    """
    N = 3
    _seed_legacy(chroma_persist_dir, n=N)
    cfg = _cfg(chroma_persist_dir)
    target = fingerprinted_collection_name(cfg)

    from tldw_chatbook.RAG_Search.simplified.rag_factory import create_rag_service

    svc = create_rag_service("hybrid_basic", config=cfg)

    # Force the real service's own lazy Chroma client + collection at the
    # same persist_directory -- this is the Settings-collision path. Must
    # not raise.
    collection = svc.vector_store.collection
    assert collection is not None

    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path=str(chroma_persist_dir),
                                       settings=Settings(anonymized_telemetry=False, allow_reset=True))
    names = [c.name for c in client.list_collections()]
    assert target in names and "default" not in names
    assert client.get_collection(target).count() == N


@pytest.mark.requires_chromadb
def test_adopt_preserves_legacy_distance_metric(chroma_persist_dir):
    """The legacy collection's actual hnsw:space (index distance metric) must
    survive adoption unchanged, even when the config we're adopting under
    records a *different* distance_metric in its provenance. chromadb 1.5.8's
    Collection.modify() raises ValueError if "hnsw:space" appears in the new
    metadata at all (even to restate the same value) -- so the implementation
    must never pass that key to modify(), relying on chromadb leaving the
    collection's actual index configuration untouched instead.
    """
    _seed_legacy(chroma_persist_dir, hnsw_space="l2")
    # Config claims cosine, deliberately mismatched vs. the legacy l2 index,
    # to prove we don't relabel the real metric based on provenance.
    cfg = _cfg(chroma_persist_dir, distance_metric="cosine")
    target = fingerprinted_collection_name(cfg)
    maybe_adopt_legacy_collection(cfg)

    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path=str(chroma_persist_dir),
                                       settings=Settings(anonymized_telemetry=False, allow_reset=True))
    adopted = client.get_collection(target)
    assert adopted.count() == 3
    # The authoritative record of the collection's real distance metric in
    # chromadb 1.x is its configuration, not the free-form metadata dict.
    assert adopted.configuration_json["hnsw"]["space"] == "l2"
    # Provenance is still layered on top for everything else.
    assert adopted.metadata.get("source") == "legacy-adopted"
    assert adopted.metadata.get("distance_metric") == "cosine"
