import pytest

from tldw_chatbook.RAG_Admin.server_rag_admin_service import ServerRAGAdminService
from tldw_chatbook.tldw_api import (
    BatchMediaEmbeddingsRequest,
    BatchMediaEmbeddingsResponse,
    ChunkingTemplateCreateRequest,
    ChunkingTemplateDiagnosticsResponse,
    ChunkingTemplateLearnRequest,
    ChunkingTemplateLearnResponse,
    ChunkingTemplateListResponse,
    ChunkingTemplateMatchResponse,
    ChunkingTemplateResponse,
    ChunkingTemplateUpdateRequest,
    ChunkingTemplateValidationResponse,
    EmbeddingCollectionResponse,
    EmbeddingCollectionStatsResponse,
    GenerateMediaEmbeddingsRequest,
    GenerateMediaEmbeddingsResponse,
    MediaEmbeddingJobListResponse,
    MediaEmbeddingJobResponse,
    MediaEmbeddingsSearchRequest,
    MediaEmbeddingsSearchResponse,
    MediaEmbeddingsStatusResponse,
)


class FakeClient:
    def __init__(self):
        self.calls = []

    async def list_chunking_templates(self, **kwargs):
        self.calls.append(("list_chunking_templates", kwargs))
        return ChunkingTemplateListResponse.model_validate(
            {
                "templates": [
                    {
                        "id": 1,
                        "uuid": "cb547cde-3720-4f1e-b3a7-90425ee6b38d",
                        "name": "server-demo",
                        "description": "Server demo",
                        "template_json": '{"chunking": {"method": "words", "config": {"max_size": 256}}}',
                        "is_builtin": False,
                        "tags": ["rag"],
                        "created_at": "2026-04-20T00:00:00Z",
                        "updated_at": "2026-04-20T00:00:00Z",
                        "version": 1,
                        "user_id": "u1",
                    }
                ],
                "total": 1,
            }
        )

    async def create_chunking_template(self, request_data):
        self.calls.append(("create_chunking_template", request_data))
        return ChunkingTemplateResponse.model_validate(
            {
                "id": 2,
                "uuid": "1ce34b1a-5d36-4d4f-b4f4-6cd77b31377e",
                "name": "created",
                "description": "Created template",
                "template_json": '{"chunking": {"method": "words", "config": {"max_size": 512}}}',
                "is_builtin": False,
                "tags": ["notes"],
                "created_at": "2026-04-20T00:00:00Z",
                "updated_at": "2026-04-20T00:00:00Z",
                "version": 1,
                "user_id": "u1",
            }
        )

    async def update_chunking_template(self, template_name, request_data):
        self.calls.append(("update_chunking_template", template_name, request_data))
        return ChunkingTemplateResponse.model_validate(
            {
                "id": 2,
                "uuid": "1ce34b1a-5d36-4d4f-b4f4-6cd77b31377e",
                "name": template_name,
                "description": "Updated template",
                "template_json": '{"chunking": {"method": "sentences", "config": {"max_size": 8}}}',
                "is_builtin": False,
                "tags": ["notes", "updated"],
                "created_at": "2026-04-20T00:00:00Z",
                "updated_at": "2026-04-20T01:00:00Z",
                "version": 2,
                "user_id": "u1",
            }
        )

    async def get_chunking_template_diagnostics(self):
        self.calls.append(("get_chunking_template_diagnostics",))
        return ChunkingTemplateDiagnosticsResponse(
            db_class="sqlite.MediaDatabase",
            capability="native",
            fallback_enabled=True,
            hint="hint",
        )

    async def validate_chunking_template(self, template_config):
        self.calls.append(("validate_chunking_template", template_config))
        return ChunkingTemplateValidationResponse(valid=True)

    async def match_chunking_templates(self, **kwargs):
        self.calls.append(("match_chunking_templates", kwargs))
        return ChunkingTemplateMatchResponse(
            matches=[{"name": "article-template", "score": 0.85, "priority": 3}]
        )

    async def learn_chunking_template(self, request_data):
        self.calls.append(("learn_chunking_template", request_data))
        return ChunkingTemplateLearnResponse(
            template={"name": "learned", "chunking": {"method": "sentences"}},
            saved=True,
        )

    async def list_embedding_collections(self):
        self.calls.append(("list_embedding_collections",))
        return [EmbeddingCollectionResponse(name="demo_collection", metadata={"provider": "openai"})]

    async def get_embedding_collection_stats(self, collection_name):
        self.calls.append(("get_embedding_collection_stats", collection_name))
        return EmbeddingCollectionStatsResponse(
            name=collection_name,
            count=3,
            embedding_dimension=1536,
            metadata={"provider": "openai"},
        )

    async def delete_embedding_collection(self, collection_name):
        self.calls.append(("delete_embedding_collection", collection_name))
        return None

    async def get_media_embeddings_status(self, media_id):
        self.calls.append(("get_media_embeddings_status", media_id))
        return MediaEmbeddingsStatusResponse(media_id=media_id, has_embeddings=True, embedding_count=2)

    async def generate_media_embeddings(self, media_id, request_data):
        self.calls.append(("generate_media_embeddings", media_id, request_data))
        return GenerateMediaEmbeddingsResponse(
            media_id=media_id,
            status="accepted",
            message="Embedding generation started",
            embedding_model="nomic",
            job_id="job-1",
        )

    async def generate_media_embeddings_batch(self, request_data):
        self.calls.append(("generate_media_embeddings_batch", request_data))
        return BatchMediaEmbeddingsResponse(status="accepted", job_ids=["job-1"], submitted=1)

    async def search_media_embeddings(self, request_data):
        self.calls.append(("search_media_embeddings", request_data))
        return MediaEmbeddingsSearchResponse(
            results=[{"id": "chunk-1", "document": "alpha", "metadata": {"media_id": "42"}}],
            count=1,
        )

    async def delete_media_embeddings(self, media_id):
        self.calls.append(("delete_media_embeddings", media_id))
        return {"status": "success", "message": f"Embeddings deleted for media item {media_id}"}

    async def get_media_embedding_job(self, job_id):
        self.calls.append(("get_media_embedding_job", job_id))
        return MediaEmbeddingJobResponse.model_validate(
            {"uuid": job_id, "status": "completed", "media_id": 42}
        )

    async def list_media_embedding_jobs(self, **kwargs):
        self.calls.append(("list_media_embedding_jobs", kwargs))
        return MediaEmbeddingJobListResponse(
            data=[{"uuid": "job-1", "status": "completed"}],
            pagination={"limit": 10, "offset": 0, "count": 1},
        )


@pytest.mark.asyncio
async def test_server_rag_admin_service_builds_requests_and_unwraps_models():
    client = FakeClient()
    service = ServerRAGAdminService(client=client)

    listed = await service.list_templates(include_builtin=False, include_custom=True, tags=["rag"], user_id="u1")
    created = await service.create_template(
        name="created",
        description="Created template",
        tags=["notes"],
        template={"chunking": {"method": "words", "config": {"max_size": 512}}},
        user_id="u1",
    )
    updated = await service.update_template(
        "created",
        description="Updated template",
        tags=["notes", "updated"],
        template={"chunking": {"method": "sentences", "config": {"max_size": 8}}},
    )
    diagnostics = await service.get_template_diagnostics()
    collections = await service.list_collections()
    stats = await service.get_collection_detail("demo_collection")
    await service.delete_collection("demo_collection")

    assert client.calls[0] == (
        "list_chunking_templates",
        {"include_builtin": False, "include_custom": True, "tags": ["rag"], "user_id": "u1"},
    )
    assert isinstance(client.calls[1][1], ChunkingTemplateCreateRequest)
    assert client.calls[1][1].name == "created"
    assert isinstance(client.calls[2][2], ChunkingTemplateUpdateRequest)
    assert client.calls[2][1] == "created"
    assert client.calls[3] == ("get_chunking_template_diagnostics",)
    assert client.calls[4] == ("list_embedding_collections",)
    assert client.calls[5] == ("get_embedding_collection_stats", "demo_collection")
    assert client.calls[6] == ("delete_embedding_collection", "demo_collection")

    assert listed[0]["name"] == "server-demo"
    assert created["name"] == "created"
    assert updated["description"] == "Updated template"
    assert diagnostics["capability"] == "native"
    assert collections[0]["name"] == "demo_collection"
    assert stats["count"] == 3


@pytest.mark.asyncio
async def test_server_rag_admin_service_exposes_template_helper_operations():
    client = FakeClient()
    service = ServerRAGAdminService(client=client)
    template_config = {"chunking": {"method": "sentences", "config": {"max_size": 8}}}

    validation = await service.validate_template_config(template_config)
    matches = await service.match_templates(
        media_type="article",
        title="Example Article",
        url="https://example.test/article",
        filename="article.md",
    )
    learned = await service.learn_template(
        name="learned",
        example_text="# Heading\nBody",
        description="Learned template",
        save=True,
        classifier={"media_type": "article"},
    )

    assert client.calls[0] == ("validate_chunking_template", template_config)
    assert client.calls[1] == (
        "match_chunking_templates",
        {
            "media_type": "article",
            "title": "Example Article",
            "url": "https://example.test/article",
            "filename": "article.md",
        },
    )
    assert client.calls[2][0] == "learn_chunking_template"
    assert isinstance(client.calls[2][1], ChunkingTemplateLearnRequest)
    assert validation["valid"] is True
    assert matches["matches"][0]["name"] == "article-template"
    assert learned["saved"] is True


@pytest.mark.asyncio
async def test_server_rag_admin_service_exposes_media_embedding_operations():
    client = FakeClient()
    service = ServerRAGAdminService(client=client)

    status = await service.get_media_embeddings_status(42)
    generated = await service.generate_media_embeddings(
        42,
        embedding_model="nomic",
        embedding_provider="ollama",
        force_regenerate=True,
        priority=75,
    )
    batch = await service.generate_media_embeddings_batch(
        media_ids=[42],
        embedding_model="nomic",
        embedding_provider="ollama",
        priority=80,
    )
    search = await service.search_media_embeddings(
        query="alpha",
        top_k=3,
        collection="user_1_media_embeddings",
        embedding_model="nomic",
        embedding_provider="ollama",
        filters={"media_id": "42"},
    )
    deleted = await service.delete_media_embeddings(42)
    job = await service.get_media_embedding_job("job-1")
    jobs = await service.list_media_embedding_jobs(status="completed", limit=10, offset=0)

    assert client.calls[0] == ("get_media_embeddings_status", 42)
    assert client.calls[1][0:2] == ("generate_media_embeddings", 42)
    assert isinstance(client.calls[1][2], GenerateMediaEmbeddingsRequest)
    assert client.calls[1][2].force_regenerate is True
    assert client.calls[2][0] == "generate_media_embeddings_batch"
    assert isinstance(client.calls[2][1], BatchMediaEmbeddingsRequest)
    assert client.calls[3][0] == "search_media_embeddings"
    assert isinstance(client.calls[3][1], MediaEmbeddingsSearchRequest)
    assert client.calls[4] == ("delete_media_embeddings", 42)
    assert client.calls[5] == ("get_media_embedding_job", "job-1")
    assert client.calls[6] == ("list_media_embedding_jobs", {"status": "completed", "limit": 10, "offset": 0})

    assert status["media_id"] == 42
    assert generated["job_id"] == "job-1"
    assert batch["submitted"] == 1
    assert search["count"] == 1
    assert deleted["status"] == "success"
    assert job["uuid"] == "job-1"
    assert jobs["pagination"]["count"] == 1
