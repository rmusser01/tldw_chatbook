import pytest

from tldw_chatbook.RAG_Admin.rag_admin_scope_service import RAGAdminScopeService


class FakeLocalService:
    def __init__(self, templates=None, collection_detail=None):
        self.templates = templates or []
        self.collection_detail = collection_detail or {}
        self.calls = []

    def list_templates(self, **kwargs):
        self.calls.append(("list_templates", kwargs))
        return list(self.templates)

    def get_collection_detail(self, collection_name):
        self.calls.append(("get_collection_detail", collection_name))
        return dict(self.collection_detail[collection_name])

    def get_media_embeddings_status(self, media_id):
        self.calls.append(("get_media_embeddings_status", media_id))
        return {"media_id": media_id, "has_embeddings": True, "embedding_count": 2}

    def generate_media_embeddings(self, media_id, **kwargs):
        self.calls.append(("generate_media_embeddings", media_id, kwargs))
        return {"media_id": media_id, "status": "completed", "embedding_count": 2}

    def delete_media_embeddings(self, media_id):
        self.calls.append(("delete_media_embeddings", media_id))
        return {"media_id": media_id, "status": "success", "deleted_count": 2}


class FakeServerService:
    def __init__(self, templates=None, collection_detail=None):
        self.templates = templates or []
        self.collection_detail = collection_detail or {}
        self.calls = []

    async def list_templates(self, **kwargs):
        self.calls.append(("list_templates", kwargs))
        return list(self.templates)

    async def get_collection_detail(self, collection_name):
        self.calls.append(("get_collection_detail", collection_name))
        return dict(self.collection_detail[collection_name])

    async def generate_media_embeddings(self, media_id, **kwargs):
        self.calls.append(("generate_media_embeddings", media_id, kwargs))
        return {"media_id": media_id, "status": "accepted", "job_id": "job-1"}

    async def list_media_embedding_jobs(self, **kwargs):
        self.calls.append(("list_media_embedding_jobs", kwargs))
        return {"data": [{"uuid": "job-1", "status": "completed"}], "pagination": {"count": 1}}

    async def reprocess_media(self, media_id, **kwargs):
        self.calls.append(("reprocess_media", media_id, kwargs))
        return {"media_id": media_id, "status": "completed", "embeddings_started": True}

    async def validate_template_config(self, template_config):
        self.calls.append(("validate_template_config", template_config))
        return {"valid": True}

    async def match_templates(self, **kwargs):
        self.calls.append(("match_templates", kwargs))
        return {"matches": [{"name": "article-template", "score": 0.85}]}

    async def learn_template(self, **kwargs):
        self.calls.append(("learn_template", kwargs))
        return {"template": {"name": kwargs["name"]}, "saved": kwargs.get("save", False)}


@pytest.mark.asyncio
async def test_scope_service_routes_template_list_by_backend():
    local = FakeLocalService(
        templates=[
            {
                "id": 7,
                "name": "local-demo",
                "description": "Local demo",
                "template_json": '{"chunking": {"method": "words", "config": {"max_size": 100}}}',
                "is_system": 0,
            }
        ]
    )
    server = FakeServerService(
        templates=[
            {
                "id": 8,
                "uuid": "cb547cde-3720-4f1e-b3a7-90425ee6b38d",
                "name": "server-demo",
                "description": "Server demo",
                "template_json": '{"chunking": {"method": "sentences", "config": {"max_size": 8}}}',
                "is_builtin": False,
                "tags": ["rag"],
                "version": 1,
                "created_at": "2026-04-20T00:00:00Z",
                "updated_at": "2026-04-20T00:00:00Z",
            }
        ]
    )
    scope = RAGAdminScopeService(local_service=local, server_service=server)

    local_records = await scope.list_templates(mode="local")
    server_records = await scope.list_templates(mode="server")

    assert local_records[0]["backend"] == "local"
    assert local_records[0]["record_id"] == "local:chunking_template:local-demo"
    assert server_records[0]["backend"] == "server"
    assert server_records[0]["tags"] == ["rag"]


@pytest.mark.asyncio
async def test_scope_service_uses_stats_endpoint_for_server_collection_detail():
    server = FakeServerService(
        collection_detail={
            "demo": {
                "name": "demo",
                "count": 3,
                "embedding_dimension": 1536,
                "metadata": {"provider": "openai"},
            }
        }
    )
    scope = RAGAdminScopeService(local_service=FakeLocalService(), server_service=server)

    detail = await scope.get_collection_detail(mode="server", collection_name="demo")

    assert detail["backend"] == "server"
    assert detail["name"] == "demo"
    assert detail["count"] == 3
    assert server.calls == [("get_collection_detail", "demo")]


@pytest.mark.asyncio
async def test_scope_service_routes_server_media_embedding_operations_to_server_backend():
    server = FakeServerService()
    scope = RAGAdminScopeService(local_service=FakeLocalService(), server_service=server)

    generated = await scope.generate_media_embeddings(
        mode="server",
        media_id=42,
        embedding_model="nomic",
        embedding_provider="ollama",
    )
    jobs = await scope.list_media_embedding_jobs(mode="server", status="completed")

    assert generated == {"media_id": 42, "status": "accepted", "job_id": "job-1"}
    assert jobs["pagination"]["count"] == 1
    assert server.calls == [
        (
            "generate_media_embeddings",
            42,
            {"embedding_model": "nomic", "embedding_provider": "ollama"},
        ),
        ("list_media_embedding_jobs", {"status": "completed", "limit": 50, "offset": 0}),
    ]


@pytest.mark.asyncio
async def test_scope_service_routes_local_media_embedding_lifecycle_to_local_backend():
    local = FakeLocalService()
    scope = RAGAdminScopeService(local_service=local, server_service=FakeServerService())

    status = await scope.get_media_embeddings_status(mode="local", media_id=42)
    generated = await scope.generate_media_embeddings(
        mode="local",
        media_id=42,
        embedding_model="local-embed",
        chunk_size=250,
    )
    deleted = await scope.delete_media_embeddings(mode="local", media_id=42)

    assert status["has_embeddings"] is True
    assert generated["status"] == "completed"
    assert deleted["deleted_count"] == 2
    assert local.calls == [
        ("get_media_embeddings_status", 42),
        ("generate_media_embeddings", 42, {"embedding_model": "local-embed", "chunk_size": 250}),
        ("delete_media_embeddings", 42),
    ]


@pytest.mark.asyncio
async def test_scope_service_routes_media_reprocess_to_server_only():
    server = FakeServerService()
    scope = RAGAdminScopeService(local_service=FakeLocalService(), server_service=server)

    response = await scope.reprocess_media(
        mode="server",
        media_id=42,
        perform_chunking=True,
        generate_embeddings=True,
        chunking_template_name="article-template",
    )

    assert response == {"media_id": 42, "status": "completed", "embeddings_started": True}
    assert server.calls == [
        (
            "reprocess_media",
            42,
            {
                "perform_chunking": True,
                "generate_embeddings": True,
                "chunking_template_name": "article-template",
            },
        )
    ]

    with pytest.raises(ValueError, match="Server retrieval-admin backend is required"):
        await scope.reprocess_media(mode="local", media_id=42)


@pytest.mark.asyncio
async def test_scope_service_routes_template_helpers_to_server_only():
    server = FakeServerService()
    scope = RAGAdminScopeService(local_service=FakeLocalService(), server_service=server)
    template_config = {"chunking": {"method": "sentences"}}

    validation = await scope.validate_template_config(mode="server", template_config=template_config)
    matches = await scope.match_templates(mode="server", media_type="article", title="Example Article")
    learned = await scope.learn_template(
        mode="server",
        name="learned",
        example_text="# Heading\nBody",
        save=True,
    )

    assert validation["valid"] is True
    assert matches["matches"][0]["name"] == "article-template"
    assert learned == {"template": {"name": "learned"}, "saved": True}
    assert server.calls == [
        ("validate_template_config", template_config),
        ("match_templates", {"media_type": "article", "title": "Example Article"}),
        (
            "learn_template",
            {
                "name": "learned",
                "example_text": "# Heading\nBody",
                "save": True,
            },
        ),
    ]

    with pytest.raises(ValueError, match="Server retrieval-admin backend is required"):
        await scope.validate_template_config(mode="local", template_config=template_config)
