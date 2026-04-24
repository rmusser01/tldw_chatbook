import pytest

from tldw_chatbook.RAG_Admin.local_rag_admin_service import LocalRAGAdminService


class FakeMediaDB:
    def __init__(self):
        self.rows = {
            42: {
                "id": 42,
                "uuid": "media-42",
                "title": "Local Document",
                "content": "Alpha beta gamma.",
            }
        }
        self.processed_chunks = []

    def get_media_by_ids_for_embedding(self, media_ids):
        return [self.rows[media_id] for media_id in media_ids if media_id in self.rows]

    def process_chunks(self, media_id, chunks):
        self.processed_chunks.append((media_id, chunks))


class FakeCollection:
    def __init__(self, ids=None):
        self.name = "local_media_embeddings"
        self.ids = list(ids or [])
        self.metadata = {"embedding_dimension": 3}

    def get(self, **kwargs):
        return {
            "ids": list(self.ids),
            "metadatas": [{"media_id": "42"} for _ in self.ids],
        }

    def count(self):
        return len(self.ids)


class FakeChromaClient:
    def __init__(self, collection):
        self.collection = collection

    def get_collection(self, name):
        return self.collection


class FakeChromaManager:
    def __init__(self):
        self.collection = FakeCollection()
        self.client = FakeChromaClient(self.collection)
        self.calls = []

    def get_user_default_collection_name(self):
        return "local_media_embeddings"

    def process_and_store_content(self, **kwargs):
        self.calls.append(("process_and_store_content", kwargs))
        self.collection.ids = ["42_chunk_0", "42_chunk_1"]

    def vector_search(self, **kwargs):
        self.calls.append(("vector_search", kwargs))
        return [
            {
                "id": "42_chunk_0",
                "content": "Alpha beta gamma.",
                "metadata": {"media_id": "42", "chunk_index": 0},
                "distance": 0.12,
            }
        ]

    def delete_from_collection(self, ids, collection_name=None):
        self.calls.append(("delete_from_collection", list(ids), collection_name))
        self.collection.ids = [existing for existing in self.collection.ids if existing not in set(ids)]


def test_local_rag_admin_generates_status_and_deletes_media_embeddings():
    chroma = FakeChromaManager()
    service = LocalRAGAdminService(media_db=FakeMediaDB(), chroma_manager=chroma)

    before = service.get_media_embeddings_status(42)
    generated = service.generate_media_embeddings(
        42,
        embedding_model="local-embed",
        chunk_size=250,
        chunk_overlap=50,
        force_regenerate=True,
    )
    after = service.get_media_embeddings_status(42)
    deleted = service.delete_media_embeddings(42)

    assert before["has_embeddings"] is False
    assert generated["status"] == "completed"
    assert generated["embedding_count"] == 2
    assert after["has_embeddings"] is True
    assert after["embedding_count"] == 2
    assert deleted["deleted_count"] == 2
    assert chroma.calls[0] == ("delete_from_collection", [], "local_media_embeddings")
    assert chroma.calls[1][0] == "process_and_store_content"
    assert chroma.calls[1][1]["content"] == "Alpha beta gamma."
    assert chroma.calls[1][1]["embedding_model_id_override"] == "local-embed"
    assert chroma.calls[1][1]["chunk_options"] == {"chunk_size": 250, "chunk_overlap": 50}


def test_local_rag_admin_searches_media_embeddings_with_chroma():
    chroma = FakeChromaManager()
    service = LocalRAGAdminService(media_db=FakeMediaDB(), chroma_manager=chroma)

    result = service.search_media_embeddings(
        query="alpha",
        top_k=2,
        collection="local_media_embeddings",
        embedding_model="local-embed",
        filters={"media_id": "42"},
    )

    assert result["backend"] == "local"
    assert result["collection"] == "local_media_embeddings"
    assert result["count"] == 1
    assert result["results"][0] == {
        "id": "42_chunk_0",
        "document": "Alpha beta gamma.",
        "metadata": {"media_id": "42", "chunk_index": 0},
        "distance": 0.12,
    }
    assert chroma.calls == [
        (
            "vector_search",
            {
                "query": "alpha",
                "collection_name": "local_media_embeddings",
                "k": 2,
                "embedding_model_id_override": "local-embed",
                "where_filter": {"media_id": "42"},
                "include_fields": ["documents", "metadatas", "distances"],
            },
        )
    ]


def test_local_rag_admin_reprocesses_media_chunks_and_embeddings():
    media_db = FakeMediaDB()
    chroma = FakeChromaManager()
    service = LocalRAGAdminService(media_db=media_db, chroma_manager=chroma)

    result = service.reprocess_media(
        42,
        perform_chunking=True,
        generate_embeddings=True,
        chunk_method="words",
        chunk_size=2,
        chunk_overlap=0,
        embedding_model="local-embed",
        force_regenerate_embeddings=True,
    )

    assert result["backend"] == "local"
    assert result["media_id"] == 42
    assert result["status"] == "completed"
    assert result["chunks_created"] == 2
    assert result["embeddings_started"] is True
    assert result["embedding_count"] == 2
    assert media_db.processed_chunks == [
        (
            42,
            [
                {"text": "Alpha beta", "start_index": 0, "end_index": 10},
                {"text": "gamma.", "start_index": 11, "end_index": 17},
            ],
        )
    ]
    assert chroma.calls[0] == ("delete_from_collection", [], "local_media_embeddings")
    assert chroma.calls[1][0] == "process_and_store_content"


def test_local_rag_admin_reprocess_rejects_template_only_options():
    service = LocalRAGAdminService(media_db=FakeMediaDB(), chroma_manager=FakeChromaManager())

    with pytest.raises(ValueError, match="template-driven chunking"):
        service.reprocess_media(
            42,
            perform_chunking=True,
            chunking_template_name="article-template",
        )
