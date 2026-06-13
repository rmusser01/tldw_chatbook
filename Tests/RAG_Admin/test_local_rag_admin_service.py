from tldw_chatbook.RAG_Admin.local_rag_admin_service import LocalRAGAdminService


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


class FakeChromaManager:
    client = FakeChromaClient()


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
    service = LocalRAGAdminService(None, chroma_manager=FakeChromaManager())

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
