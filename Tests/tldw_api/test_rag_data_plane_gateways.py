"""Tests for server RAG and vector data-plane namespace gateways."""

from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api.client import TLDWAPIClient


@pytest.mark.asyncio
async def test_embeddings_rag_and_vector_store_gateways_route_server_data_plane_surfaces(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(client, "_request", mocked)

    await client.call_server_embeddings_endpoint(
        "POST",
        "models/download",
        payload={"provider": "sentence-transformers", "model": "all-MiniLM-L6-v2"},
    )
    await client.call_server_rag_endpoint(
        "POST",
        "/api/v1/rag/search",
        payload={"query": "offline sync", "top_k": 5},
    )
    await client.call_server_vector_stores_endpoint(
        "GET",
        "store-1/vectors/batches/batch-1",
        params={"include_errors": "true"},
    )

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/embeddings/models/download")
    assert mocked.await_args_list[0].kwargs["json_data"] == {
        "provider": "sentence-transformers",
        "model": "all-MiniLM-L6-v2",
    }
    assert mocked.await_args_list[1].args[:2] == ("POST", "/api/v1/rag/search")
    assert mocked.await_args_list[1].kwargs["json_data"] == {"query": "offline sync", "top_k": 5}
    assert mocked.await_args_list[2].args[:2] == (
        "GET",
        "/api/v1/vector_stores/store-1/vectors/batches/batch-1",
    )
    assert mocked.await_args_list[2].kwargs["params"] == {"include_errors": "true"}


@pytest.mark.asyncio
async def test_rag_data_plane_gateways_reject_cross_namespace_and_unsafe_routes(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(client, "_request", mocked)

    unsafe_calls = [
        client.call_server_embeddings_endpoint("GET", "/api/v1/rag/search"),
        client.call_server_rag_endpoint("GET", "/api/v1/embeddings/models"),
        client.call_server_vector_stores_endpoint("GET", "/api/v1/admin/users"),
        client.call_server_vector_stores_endpoint("GET", "../admin/users"),
        client.call_server_rag_endpoint("OPTIONS", "capabilities"),
    ]
    for call in unsafe_calls:
        with pytest.raises(ValueError):
            await call

    mocked.assert_not_awaited()
