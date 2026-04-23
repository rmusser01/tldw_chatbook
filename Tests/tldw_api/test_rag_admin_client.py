from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    BatchMediaEmbeddingsRequest,
    BatchMediaEmbeddingsResponse,
    ChunkingTemplateApplyRequest,
    ChunkingTemplateApplyResponse,
    ChunkingTemplateCreateRequest,
    ChunkingTemplateDiagnosticsResponse,
    ChunkingTemplateListResponse,
    ChunkingTemplateResponse,
    ChunkingTemplateUpdateRequest,
    EmbeddingCollectionResponse,
    EmbeddingCollectionStatsResponse,
    GenerateMediaEmbeddingsRequest,
    GenerateMediaEmbeddingsResponse,
    MediaEmbeddingJobListResponse,
    MediaEmbeddingJobResponse,
    MediaEmbeddingsSearchRequest,
    MediaEmbeddingsSearchResponse,
    MediaEmbeddingsStatusResponse,
    TLDWAPIClient,
)


@pytest.mark.asyncio
async def test_list_chunking_templates_serializes_filters_and_returns_typed_response(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        return_value={
            "templates": [
                {
                    "id": 1,
                    "uuid": "cb547cde-3720-4f1e-b3a7-90425ee6b38d",
                    "name": "rag_words",
                    "description": "RAG words template",
                    "template_json": '{"chunking": {"method": "words", "config": {"max_size": 400}}}',
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
    monkeypatch.setattr(client, "_request", mocked)

    response = await client.list_chunking_templates(
        include_builtin=True,
        include_custom=False,
        tags=["rag"],
        user_id="u1",
    )

    args, kwargs = mocked.await_args
    assert args[:2] == ("GET", "/api/v1/chunking/templates")
    assert kwargs["params"] == {
        "include_builtin": True,
        "include_custom": False,
        "tags": ["rag"],
        "user_id": "u1",
    }
    assert isinstance(response, ChunkingTemplateListResponse)
    assert response.total == 1
    assert response.templates[0].name == "rag_words"


@pytest.mark.asyncio
async def test_chunking_template_crud_apply_and_diagnostics_routes_wire(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {
                "id": 2,
                "uuid": "1ce34b1a-5d36-4d4f-b4f4-6cd77b31377e",
                "name": "demo",
                "description": "Demo template",
                "template_json": '{"chunking": {"method": "words", "config": {"max_size": 256}}}',
                "is_builtin": False,
                "tags": ["notes"],
                "created_at": "2026-04-20T00:00:00Z",
                "updated_at": "2026-04-20T00:00:00Z",
                "version": 1,
                "user_id": "u1",
            },
            {
                "id": 2,
                "uuid": "1ce34b1a-5d36-4d4f-b4f4-6cd77b31377e",
                "name": "demo",
                "description": "Demo template",
                "template_json": '{"chunking": {"method": "words", "config": {"max_size": 256}}}',
                "is_builtin": False,
                "tags": ["notes"],
                "created_at": "2026-04-20T00:00:00Z",
                "updated_at": "2026-04-20T00:00:00Z",
                "version": 1,
                "user_id": "u1",
            },
            {
                "id": 2,
                "uuid": "1ce34b1a-5d36-4d4f-b4f4-6cd77b31377e",
                "name": "demo",
                "description": "Updated template",
                "template_json": '{"chunking": {"method": "sentences", "config": {"max_size": 8}}}',
                "is_builtin": False,
                "tags": ["notes", "updated"],
                "created_at": "2026-04-20T00:00:00Z",
                "updated_at": "2026-04-20T01:00:00Z",
                "version": 2,
                "user_id": "u1",
            },
            {},
            {
                "template_name": "demo",
                "chunks": [{"text": "alpha beta", "metadata": {"index": 0}}],
                "metadata": {"chunk_count": 1, "template_version": 2},
            },
            {
                "db_class": "sqlite.MediaDatabase",
                "capability": "native",
                "missing_methods": [],
                "fallback_enabled": True,
                "hint": "Use a native media DB session that implements chunking-template methods.",
            },
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    created = await client.create_chunking_template(
        ChunkingTemplateCreateRequest(
            name="demo",
            description="Demo template",
            tags=["notes"],
            template={
                "preprocessing": [],
                "chunking": {"method": "words", "config": {"max_size": 256}},
                "postprocessing": [],
            },
            user_id="u1",
        )
    )
    fetched = await client.get_chunking_template("demo")
    updated = await client.update_chunking_template(
        "demo",
        ChunkingTemplateUpdateRequest(
            description="Updated template",
            tags=["notes", "updated"],
            template={
                "chunking": {"method": "sentences", "config": {"max_size": 8}},
            },
        ),
    )
    await client.delete_chunking_template("demo", hard_delete=True)
    applied = await client.apply_chunking_template(
        ChunkingTemplateApplyRequest(
            template_name="demo",
            text="alpha beta",
            override_options={"max_size": 32},
        ),
        include_metadata=True,
    )
    diagnostics = await client.get_chunking_template_diagnostics()

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/chunking/templates")
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/chunking/templates/demo")
    assert mocked.await_args_list[2].args[:2] == ("PUT", "/api/v1/chunking/templates/demo")
    assert mocked.await_args_list[3].args[:2] == ("DELETE", "/api/v1/chunking/templates/demo")
    assert mocked.await_args_list[3].kwargs["params"] == {"hard_delete": True}
    assert mocked.await_args_list[4].args[:2] == ("POST", "/api/v1/chunking/templates/apply")
    assert mocked.await_args_list[4].kwargs["params"] == {"include_metadata": True}
    assert mocked.await_args_list[5].args[:2] == ("GET", "/api/v1/chunking/templates/diagnostics")

    assert isinstance(created, ChunkingTemplateResponse)
    assert isinstance(fetched, ChunkingTemplateResponse)
    assert isinstance(updated, ChunkingTemplateResponse)
    assert isinstance(applied, ChunkingTemplateApplyResponse)
    assert isinstance(diagnostics, ChunkingTemplateDiagnosticsResponse)
    assert applied.metadata["chunk_count"] == 1
    assert diagnostics.capability == "native"


@pytest.mark.asyncio
async def test_embedding_collection_routes_wire_and_stats_are_typed(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            [
                {
                    "name": "demo_collection",
                    "metadata": {"provider": "openai", "embedding_dimension": 1536},
                }
            ],
            {
                "name": "demo_collection",
                "count": 3,
                "embedding_dimension": 1536,
                "metadata": {"provider": "openai"},
            },
            {},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    collections = await client.list_embedding_collections()
    stats = await client.get_embedding_collection_stats("demo_collection")
    await client.delete_embedding_collection("demo_collection")

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/embeddings/collections")
    assert mocked.await_args_list[1].args[:2] == (
        "GET",
        "/api/v1/embeddings/collections/demo_collection/stats",
    )
    assert mocked.await_args_list[2].args[:2] == (
        "DELETE",
        "/api/v1/embeddings/collections/demo_collection",
    )

    assert isinstance(collections, list)
    assert isinstance(collections[0], EmbeddingCollectionResponse)
    assert isinstance(stats, EmbeddingCollectionStatsResponse)
    assert stats.embedding_dimension == 1536


@pytest.mark.asyncio
async def test_media_embedding_admin_routes_wire_and_return_typed_payloads(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {
                "media_id": 42,
                "has_embeddings": True,
                "embedding_count": 12,
                "embedding_model": "nomic",
                "last_generated": "2026-04-23T10:00:00Z",
            },
            {
                "media_id": 42,
                "status": "accepted",
                "message": "Embedding generation started",
                "embedding_count": None,
                "embedding_model": "nomic",
                "chunks_processed": None,
                "job_id": "job-1",
            },
            {
                "status": "partial",
                "job_ids": ["job-1"],
                "submitted": 1,
                "failed_media_ids": [43],
                "failure_reasons": ["media_id=43: not found"],
            },
            {
                "results": [
                    {
                        "id": "chunk-1",
                        "document": "alpha",
                        "metadata": {"media_id": "42"},
                        "distance": 0.12,
                    }
                ],
                "count": 1,
            },
            {"status": "success", "message": "Embeddings deleted for media item 42"},
            {"uuid": "job-1", "status": "completed", "media_id": 42},
            {
                "data": [{"uuid": "job-1", "status": "completed"}],
                "pagination": {"limit": 10, "offset": 5, "count": 1},
            },
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    status = await client.get_media_embeddings_status(42)
    generated = await client.generate_media_embeddings(
        42,
        GenerateMediaEmbeddingsRequest(
            embedding_model="nomic",
            embedding_provider="ollama",
            force_regenerate=True,
            priority=75,
        ),
    )
    batch = await client.generate_media_embeddings_batch(
        BatchMediaEmbeddingsRequest(
            media_ids=[42, 43],
            embedding_model="nomic",
            embedding_provider="ollama",
            priority=80,
        )
    )
    search = await client.search_media_embeddings(
        MediaEmbeddingsSearchRequest(
            query="alpha",
            top_k=3,
            collection="user_1_media_embeddings",
            embedding_model="nomic",
            embedding_provider="ollama",
            filters={"media_id": "42"},
        )
    )
    deleted = await client.delete_media_embeddings(42)
    job = await client.get_media_embedding_job("job-1")
    jobs = await client.list_media_embedding_jobs(status="completed", limit=10, offset=5)

    assert isinstance(status, MediaEmbeddingsStatusResponse)
    assert isinstance(generated, GenerateMediaEmbeddingsResponse)
    assert isinstance(batch, BatchMediaEmbeddingsResponse)
    assert isinstance(search, MediaEmbeddingsSearchResponse)
    assert isinstance(job, MediaEmbeddingJobResponse)
    assert isinstance(jobs, MediaEmbeddingJobListResponse)
    assert deleted["status"] == "success"
    assert search.results[0].metadata["media_id"] == "42"

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/media/42/embeddings/status")
    assert mocked.await_args_list[1].args[:2] == ("POST", "/api/v1/media/42/embeddings")
    assert mocked.await_args_list[1].kwargs["json_data"] == {
        "embedding_model": "nomic",
        "embedding_provider": "ollama",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "force_regenerate": True,
        "priority": 75,
    }
    assert mocked.await_args_list[2].args[:2] == ("POST", "/api/v1/media/embeddings/batch")
    assert mocked.await_args_list[2].kwargs["json_data"] == {
        "media_ids": [42, 43],
        "model": "nomic",
        "provider": "ollama",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "force_regenerate": False,
        "priority": 80,
    }
    assert mocked.await_args_list[3].args[:2] == ("POST", "/api/v1/media/embeddings/search")
    assert mocked.await_args_list[3].kwargs["json_data"] == {
        "query": "alpha",
        "top_k": 3,
        "collection": "user_1_media_embeddings",
        "model": "nomic",
        "provider": "ollama",
        "filters": {"media_id": "42"},
    }
    assert mocked.await_args_list[4].args[:2] == ("DELETE", "/api/v1/media/42/embeddings")
    assert mocked.await_args_list[5].args[:2] == ("GET", "/api/v1/media/embeddings/jobs/job-1")
    assert mocked.await_args_list[6].args[:2] == ("GET", "/api/v1/media/embeddings/jobs")
    assert mocked.await_args_list[6].kwargs["params"] == {
        "limit": 10,
        "offset": 5,
        "status": "completed",
    }
