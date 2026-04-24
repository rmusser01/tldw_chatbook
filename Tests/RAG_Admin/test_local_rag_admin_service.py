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

    def get_media_by_ids_for_embedding(self, media_ids):
        return [self.rows[media_id] for media_id in media_ids if media_id in self.rows]


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
