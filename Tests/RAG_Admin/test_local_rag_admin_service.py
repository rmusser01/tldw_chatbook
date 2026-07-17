import asyncio
from types import SimpleNamespace

import pytest

from tldw_chatbook.RAG_Admin.local_rag_admin_service import LocalRAGAdminService
from tldw_chatbook.RAG_Search import ingestion_indexing
from tldw_chatbook.Utils.optional_deps import embeddings_rag_deps_installed


class FakeChunkingService:
    def __init__(self):
        self.records = {}
        self.next_id = 1

    def get_all_templates(self, include_system=True):
        return list(self.records.values())

    def get_template_by_name(self, name):
        return self.records.get(name)

    def get_template_by_id(self, template_id):
        for record in self.records.values():
            if record["id"] == template_id:
                return record
        return None

    def create_template(self, *, name, description, template_json, is_system=False):
        template_id = self.next_id
        self.next_id += 1
        self.records[name] = {
            "id": template_id,
            "name": name,
            "description": description,
            "template_json": template_json,
            "is_system": is_system,
        }
        return template_id

    def update_template(self, template_id, *, description=None, template_json=None):
        record = self.get_template_by_id(template_id)
        if record is None:
            raise ValueError("missing template")
        if description is not None:
            record["description"] = description
        if template_json is not None:
            record["template_json"] = template_json


class FakeCollection:
    name = "demo"
    metadata = {"provider": "local", "embedding_dimension": 2}

    def count(self):
        return 2

    def get(self, **kwargs):
        return {
            "ids": ["a", "b"],
            "documents": ["alpha", "beta"],
            "metadatas": [{"source": "one"}, {"source": "two"}],
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
        }


class FakeChromaClient:
    def get_collection(self, name):
        assert name == "demo"
        return FakeCollection()

    def list_collections(self):
        return [FakeCollection()]


class FakeVectorStore:
    """Duck-typed stand-in for RAG_Search.simplified.vector_store.ChromaVectorStore."""

    def __init__(self):
        self.client = FakeChromaClient()
        self.deleted = []

    def delete_collection(self, name):
        self.deleted.append(name)


class FakeMediaService:
    def __init__(self):
        self.calls = []

    def reprocess_media(self, media_id, **options):
        self.calls.append(("reprocess_media", media_id, options))
        return {"status": "queued", "media_id": media_id, "job_id": "local-job-1", "options": options}


def test_local_rag_admin_service_delegates_reprocess_to_local_media_service():
    media_service = FakeMediaService()
    service = LocalRAGAdminService(None, media_service=media_service)

    result = service.reprocess_media(
        9,
        perform_chunking=True,
        generate_embeddings=True,
    )

    assert result["status"] == "queued"
    assert result["job_id"] == "local-job-1"
    assert media_service.calls == [
        ("reprocess_media", 9, {"perform_chunking": True, "generate_embeddings": True})
    ]


def test_local_rag_admin_service_persists_and_filters_template_tags():
    service = LocalRAGAdminService(None, chunking_service=FakeChunkingService())

    created = service.create_template(
        name="tagged",
        description="Tagged template",
        template={
            "chunking": {"method": "words", "config": {"max_size": 2, "overlap": 0}},
        },
        tags=["rag", "local"],
    )
    filtered = service.list_templates(tags=["rag"])
    service.update_template("tagged", tags=["updated"])
    updated = service.get_template("tagged")

    assert created["tags"] == ["rag", "local"]
    assert [record["name"] for record in filtered] == ["tagged"]
    assert updated["tags"] == ["updated"]


def test_local_rag_admin_service_applies_server_style_template_to_text():
    service = LocalRAGAdminService(None, chunking_service=FakeChunkingService())
    service.create_template(
        name="words-two",
        description="Two word chunks",
        template={
            "chunking": {"method": "words", "config": {"max_size": 2, "overlap": 0, "language": "en"}},
        },
    )

    result = service.apply_template(
        "words-two",
        text="alpha beta gamma delta",
        include_metadata=True,
    )

    assert result["template_name"] == "words-two"
    assert result["chunks"] == ["alpha beta", "gamma delta"]
    assert result["metadata"]["chunk_count"] == 2
    assert result["metadata"]["method"] == "words"


def test_local_rag_admin_service_exports_embedding_collection_records():
    service = LocalRAGAdminService(None, vector_store=FakeVectorStore())

    exported = service.export_collection("demo")

    assert exported["name"] == "demo"
    assert exported["count"] == 2
    assert exported["metadata"]["provider"] == "local"
    assert exported["items"] == [
        {
            "id": "a",
            "document": "alpha",
            "metadata": {"source": "one"},
            "embedding": [0.1, 0.2],
        },
        {
            "id": "b",
            "document": "beta",
            "metadata": {"source": "two"},
            "embedding": [0.3, 0.4],
        },
    ]


def test_local_rag_admin_service_lists_and_deletes_via_injected_store():
    store = FakeVectorStore()
    service = LocalRAGAdminService(None, vector_store=store)

    listed = service.list_collections()
    service.delete_collection("demo")

    assert listed == [{"name": "demo", "metadata": {"provider": "local", "embedding_dimension": 2}}]
    assert store.deleted == ["demo"]


def test_local_rag_admin_service_coerces_name_only_collection_listings():
    """Some chromadb versions return bare names from list_collections."""
    store = FakeVectorStore()
    store.client.list_collections = lambda: ["demo"]
    service = LocalRAGAdminService(None, vector_store=store)

    assert service.list_collections() == [{"name": "demo", "metadata": {}}]


def test_local_rag_admin_service_uses_shared_rag_service_store():
    """task-248: with no injected store, the admin surface must operate on the
    exact vector store the shared RAG service (and therefore RAG semantic
    search) reads."""
    store = FakeVectorStore()
    ingestion_indexing.set_shared_rag_service(SimpleNamespace(vector_store=store))
    try:
        service = LocalRAGAdminService(None)

        listed = service.list_collections()
        service.delete_collection("demo")

        assert [record["name"] for record in listed] == ["demo"]
        assert store.deleted == ["demo"]
    finally:
        ingestion_indexing.reset_shared_rag_service()


def test_local_rag_admin_service_collection_ops_unavailable_without_store():
    ingestion_indexing.set_shared_rag_service(SimpleNamespace(vector_store=None))
    try:
        service = LocalRAGAdminService(None)
        with pytest.raises(ValueError, match="unavailable"):
            service.list_collections()
    finally:
        ingestion_indexing.reset_shared_rag_service()


def test_local_rag_admin_service_detail_requires_chroma_backed_store():
    """The in-memory fallback store exposes no raw client; detail/export must
    fail loudly instead of pretending."""
    ingestion_indexing.set_shared_rag_service(
        SimpleNamespace(vector_store=SimpleNamespace(client=None))
    )
    try:
        service = LocalRAGAdminService(None)
        with pytest.raises(ValueError, match="ChromaDB"):
            service.get_collection_detail("demo")
    finally:
        ingestion_indexing.reset_shared_rag_service()


@pytest.mark.integration
@pytest.mark.skipif(not embeddings_rag_deps_installed(), reason="embeddings_rag deps not installed")
def test_write_via_rag_service_is_visible_to_search_and_admin(tmp_path):
    """task-248 unification proof: a document indexed through the RAG service
    is (a) returned by semantic search and (b) visible to the retrieval-admin
    collection surface, because both read the same persistent Chroma store."""
    pytest.importorskip("chromadb")
    from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
    from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService

    cfg = RAGConfig()
    cfg.embedding.model = "mock"  # deterministic bag-of-words backend, offline
    cfg.embedding.device = "cpu"
    cfg.vector_store.type = "chroma"
    cfg.vector_store.persist_directory = tmp_path / "chromadb"
    cfg.chunking.chunk_size = 60
    cfg.chunking.chunk_overlap = 10
    cfg.search.enable_cache = False
    rag_service = RAGService(cfg)

    content = (
        "The zanzibar quokka manifesto describes how quokkas organise "
        "snorkeling expeditions near flamingo lagoons."
    )
    result = asyncio.run(
        rag_service.index_document("media_42", content, title="Quokka Manifesto")
    )
    assert result.success, result.error

    # Read path 1: RAG semantic search finds it (AC #1 / #4).
    results = asyncio.run(
        rag_service.search(
            "zanzibar quokka snorkeling",
            top_k=3,
            search_type="semantic",
            include_citations=False,
        )
    )
    assert results, "semantic search returned nothing for the indexed document"

    # Read path 2: the admin collection surface sees the same collection.
    ingestion_indexing.set_shared_rag_service(rag_service)
    try:
        admin = LocalRAGAdminService(None)
        names = [record["name"] for record in admin.list_collections()]
        assert cfg.vector_store.collection_name in names

        detail = admin.get_collection_detail(cfg.vector_store.collection_name)
        assert detail["count"] > 0

        exported = admin.export_collection(
            cfg.vector_store.collection_name, include_embeddings=False
        )
        assert any("zanzibar" in (item["document"] or "") for item in exported["items"])
    finally:
        ingestion_indexing.reset_shared_rag_service()
        rag_service.close()
