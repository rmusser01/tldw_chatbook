"""Tests for remaining small server parity namespace gateways."""

from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api.client import TLDWAPIClient


@pytest.mark.asyncio
async def test_misc_server_parity_gateways_route_namespace_scoped_surfaces(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(client, "_request", mocked)

    calls = [
        client.call_server_audio_endpoint("POST", "segment/transcript", payload={"media_id": "m1"}),
        client.call_server_prompt_studio_endpoint("GET", "projects", params={"limit": 10}),
        client.call_server_quizzes_endpoint("POST", "import/json", payload={"items": []}),
        client.call_server_email_endpoint("GET", "search", params={"q": "invoice"}),
        client.call_server_chunking_endpoint("POST", "chunk_text", payload={"text": "hello"}),
        client.call_server_llm_endpoint("GET", "providers/mlx/status"),
        client.call_server_outputs_endpoint("GET", "download/by-name", params={"name": "trace.json"}),
        client.call_server_sharing_endpoint("GET", "admin/audit"),
        client.call_server_voice_endpoint("GET", "workflows/templates"),
        client.call_server_audit_endpoint("GET", "export", params={"format": "jsonl"}),
        client.call_server_invites_endpoint("POST", "redeem", payload={"token": "abc"}),
        client.call_server_research_endpoint("GET", "semantic-scholar-search", params={"q": "rag"}),
        client.call_server_web_scraping_endpoint("POST", "service/initialize"),
        client.call_server_websub_endpoint("GET", "callback/user-1/token-1"),
        client.call_server_chatbooks_endpoint("GET", "health"),
        client.call_server_config_endpoint("GET", "quickstart"),
        client.call_server_diag_endpoint("GET", "coverage"),
        client.call_server_llamafile_endpoint("GET", "metrics"),
    ]
    for call in calls:
        await call

    expected_paths = [
        ("POST", "/api/v1/audio/segment/transcript"),
        ("GET", "/api/v1/prompt-studio/projects"),
        ("POST", "/api/v1/quizzes/import/json"),
        ("GET", "/api/v1/email/search"),
        ("POST", "/api/v1/chunking/chunk_text"),
        ("GET", "/api/v1/llm/providers/mlx/status"),
        ("GET", "/api/v1/outputs/download/by-name"),
        ("GET", "/api/v1/sharing/admin/audit"),
        ("GET", "/api/v1/voice/workflows/templates"),
        ("GET", "/api/v1/audit/export"),
        ("POST", "/api/v1/invites/redeem"),
        ("GET", "/api/v1/research/semantic-scholar-search"),
        ("POST", "/api/v1/web-scraping/service/initialize"),
        ("GET", "/api/v1/websub/callback/user-1/token-1"),
        ("GET", "/api/v1/chatbooks/health"),
        ("GET", "/api/v1/config/quickstart"),
        ("GET", "/api/v1/diag/coverage"),
        ("GET", "/api/v1/llamafile/metrics"),
    ]
    assert [call.args[:2] for call in mocked.await_args_list] == expected_paths
    assert mocked.await_args_list[0].kwargs["json_data"] == {"media_id": "m1"}
    assert mocked.await_args_list[1].kwargs["params"] == {"limit": 10}
    assert mocked.await_args_list[7].kwargs["params"] is None


@pytest.mark.asyncio
async def test_misc_server_parity_gateways_reject_cross_namespace_and_unsafe_routes(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(client, "_request", mocked)

    unsafe_calls = [
        client.call_server_audio_endpoint("GET", "/api/v1/admin/audio"),
        client.call_server_prompt_studio_endpoint("GET", "/api/v1/prompts"),
        client.call_server_quizzes_endpoint("GET", "../admin/quizzes"),
        client.call_server_email_endpoint("TRACE", "search"),
        client.call_server_chunking_endpoint("GET", "/api/v1/rag/chunk_text"),
        client.call_server_llm_endpoint("GET", "/api/v1/llamacpp/models"),
        client.call_server_outputs_endpoint("GET", "../files/secrets"),
        client.call_server_sharing_endpoint("OPTIONS", "admin/audit"),
        client.call_server_voice_endpoint("GET", "/api/v1/workflows"),
        client.call_server_audit_endpoint("GET", "/api/v1/admin/audit"),
        client.call_server_invites_endpoint("GET", "../tokens"),
        client.call_server_research_endpoint("GET", "/api/v1/paper-search/arxiv"),
        client.call_server_web_scraping_endpoint("GET", "/api/v1/media/ingest"),
        client.call_server_websub_endpoint("GET", "../callbacks"),
        client.call_server_chatbooks_endpoint("GET", "/api/v1/chatbooks/../admin"),
        client.call_server_config_endpoint("GET", "/api/v1/admin/config"),
        client.call_server_diag_endpoint("GET", "api/v1/admin/diag"),
        client.call_server_llamafile_endpoint("OPTIONS", "metrics"),
    ]
    for call in unsafe_calls:
        with pytest.raises(ValueError):
            await call

    mocked.assert_not_awaited()
