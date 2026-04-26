from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    ChunkingTemplateApplyRequest,
    ChunkingTemplateApplyResponse,
    ChunkingTemplateCreateRequest,
    ChunkingTemplateDiagnosticsResponse,
    ChunkingTemplateListResponse,
    ChunkingTemplateResponse,
    ChunkingTemplateUpdateRequest,
    EmbeddingCollectionCreateRequest,
    EmbeddingCollectionResponse,
    EmbeddingCollectionStatsResponse,
    MediaEmbeddingJobListResponse,
    MediaEmbeddingJobResponse,
    MediaEmbeddingsBatchRequest,
    MediaEmbeddingsBatchResponse,
    MediaEmbeddingsGenerateRequest,
    MediaEmbeddingsGenerateResponse,
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
            {
                "name": "new_collection",
                "metadata": {
                    "provider": "openai",
                    "embedding_model": "text-embedding-3-small",
                },
            },
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

    created = await client.create_embedding_collection(
        EmbeddingCollectionCreateRequest(
            name="new_collection",
            metadata={"purpose": "tests"},
            embedding_model="text-embedding-3-small",
            provider="openai",
        )
    )
    collections = await client.list_embedding_collections()
    stats = await client.get_embedding_collection_stats("demo_collection")
    await client.delete_embedding_collection("demo_collection")

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/embeddings/collections")
    assert mocked.await_args_list[0].kwargs["json_data"] == {
        "name": "new_collection",
        "metadata": {"purpose": "tests"},
        "embedding_model": "text-embedding-3-small",
        "provider": "openai",
    }
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/embeddings/collections")
    assert mocked.await_args_list[2].args[:2] == (
        "GET",
        "/api/v1/embeddings/collections/demo_collection/stats",
    )
    assert mocked.await_args_list[3].args[:2] == (
        "DELETE",
        "/api/v1/embeddings/collections/demo_collection",
    )

    assert isinstance(created, EmbeddingCollectionResponse)
    assert created.name == "new_collection"
    assert isinstance(collections, list)
    assert isinstance(collections[0], EmbeddingCollectionResponse)
    assert isinstance(stats, EmbeddingCollectionStatsResponse)
    assert stats.embedding_dimension == 1536


@pytest.mark.asyncio
async def test_media_embedding_routes_wire_and_return_typed_responses(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {
                "media_id": 7,
                "has_embeddings": True,
                "embedding_count": 3,
                "embedding_model": "text-embedding-3-small",
                "last_generated": None,
            },
            {
                "media_id": 7,
                "status": "accepted",
                "message": "Embedding generation started",
                "embedding_count": None,
                "embedding_model": "text-embedding-3-small",
                "chunks_processed": None,
                "job_id": "job-7",
            },
            {
                "status": "partial",
                "job_ids": ["job-7"],
                "submitted": 1,
                "failed_media_ids": [8],
                "failure_reasons": ["media_id=8: not found"],
            },
            {
                "results": [
                    {
                        "id": "chunk-1",
                        "document": "alpha",
                        "metadata": {"media_id": "7"},
                        "distance": 0.12,
                    }
                ],
                "count": 1,
            },
            {"status": "success", "message": "Embeddings deleted for media item 7"},
            {"uuid": "job-7", "status": "completed", "media_id": 7},
            {
                "data": [{"uuid": "job-7", "status": "completed", "media_id": 7}],
                "pagination": {"limit": 10, "offset": 5, "count": 1},
            },
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    status = await client.get_media_embeddings_status(7)
    generated = await client.generate_media_embeddings(
        7,
        MediaEmbeddingsGenerateRequest(
            embedding_model="text-embedding-3-small",
            embedding_provider="openai",
            chunk_size=512,
            chunk_overlap=64,
            force_regenerate=True,
            priority=80,
        ),
    )
    batch = await client.generate_media_embeddings_batch(
        MediaEmbeddingsBatchRequest(
            media_ids=[7, 8],
            embedding_model="text-embedding-3-small",
            embedding_provider="openai",
            chunk_size=512,
            chunk_overlap=64,
            force_regenerate=True,
            priority=80,
        )
    )
    search = await client.search_media_embeddings(
        MediaEmbeddingsSearchRequest(
            query="alpha",
            top_k=3,
            collection="user_1_media_embeddings",
            embedding_model="text-embedding-3-small",
            embedding_provider="openai",
            filters={"media_id": "7"},
        )
    )
    deleted = await client.delete_media_embeddings(7)
    job = await client.get_media_embedding_job("job-7")
    jobs = await client.list_media_embedding_jobs(status="completed", limit=10, offset=5)

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/media/7/embeddings/status")
    assert mocked.await_args_list[1].args[:2] == ("POST", "/api/v1/media/7/embeddings")
    assert mocked.await_args_list[1].kwargs["json_data"] == {
        "embedding_model": "text-embedding-3-small",
        "embedding_provider": "openai",
        "chunk_size": 512,
        "chunk_overlap": 64,
        "force_regenerate": True,
        "priority": 80,
    }
    assert mocked.await_args_list[2].args[:2] == ("POST", "/api/v1/media/embeddings/batch")
    assert mocked.await_args_list[3].args[:2] == ("POST", "/api/v1/media/embeddings/search")
    assert mocked.await_args_list[4].args[:2] == ("DELETE", "/api/v1/media/7/embeddings")
    assert mocked.await_args_list[5].args[:2] == ("GET", "/api/v1/media/embeddings/jobs/job-7")
    assert mocked.await_args_list[6].args[:2] == ("GET", "/api/v1/media/embeddings/jobs")
    assert mocked.await_args_list[6].kwargs["params"] == {
        "status": "completed",
        "limit": 10,
        "offset": 5,
    }

    assert isinstance(status, MediaEmbeddingsStatusResponse)
    assert isinstance(generated, MediaEmbeddingsGenerateResponse)
    assert isinstance(batch, MediaEmbeddingsBatchResponse)
    assert isinstance(search, MediaEmbeddingsSearchResponse)
    assert isinstance(job, MediaEmbeddingJobResponse)
    assert isinstance(jobs, MediaEmbeddingJobListResponse)
    assert status.embedding_count == 3
    assert generated.job_id == "job-7"
    assert batch.failed_media_ids == [8]
    assert search.results[0].metadata["media_id"] == "7"
    assert deleted["status"] == "success"
    assert job.status == "completed"
    assert jobs.pagination["count"] == 1
