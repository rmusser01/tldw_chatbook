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
    EmbeddingCollectionResponse,
    EmbeddingCollectionStatsResponse,
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
