from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    FlashcardsImportLimitsResponse,
    ProviderValidateRequest,
    ProviderValidateResponse,
    ProvidersStatusResponse,
    ServerDocsInfoResponse,
    ServerHealthResponse,
    ServerMetricsResponse,
    ServerReadinessResponse,
    TLDWAPIClient,
    TokenizerConfigResponse,
    TokenizerUpdateRequest,
)


@pytest.mark.asyncio
async def test_server_runtime_routes_wire_health_config_and_provider_discovery(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {
                "status": "ok",
                "checks": {"database": {"status": "healthy"}},
                "timestamp": "2026-04-25T12:00:00Z",
                "auth_mode": "multi_user",
            },
            {"status": "alive"},
            {
                "status": "ready",
                "ready": True,
                "engine": {"queue_depth": 0},
                "db": {"ok": True},
                "time": "2026-04-25T12:00:00Z",
            },
            {
                "cpu": {"percent": 12.5},
                "memory": {"total": 100, "available": 80, "percent": 20.0, "used": 20, "free": 80},
                "disk": {"total": 1000, "used": 200, "free": 800, "percent": 20.0},
            },
            {
                "timestamp": "2026-04-25T12:00:00Z",
                "risk_level": "low",
                "status": "secure",
                "summary": {"high_risk_events": 0},
            },
            {
                "configured": True,
                "auth_mode": "multi_user",
                "api_key": "YOUR_API_KEY",
                "api_key_configured": True,
                "base_url": "http://localhost:8000",
                "configured_providers": ["OpenAI"],
                "ffmpeg_available": True,
                "capabilities": {"hasAudio": True},
                "supported_features": {"hasSlides": True},
                "examples": {"curl": "curl ..."},
            },
            {
                "max_lines": 10000,
                "max_line_length": 32768,
                "max_field_length": 8192,
                "overrides": {"query_params": ["max_lines"]},
            },
            {"mode": "whitespace", "divisor": 4, "available_modes": ["whitespace", "char_approx"]},
            {"mode": "char_approx", "divisor": 5, "available_modes": ["whitespace", "char_approx"]},
            {
                "backend": "sqlite",
                "configured": True,
                "standard_queues": ["default", "high", "low"],
                "flags": {"JOBS_LEASE_SECONDS": 60},
                "notes": "DSN is not exposed.",
            },
            {
                "providers": [
                    {
                        "name": "openai",
                        "configured": True,
                        "requires_api_key": True,
                        "key_hint": "sk-...abcd",
                        "key_source": "env",
                    }
                ],
                "any_configured": True,
            },
            {"provider": "openai", "valid": True, "error": None},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    health = await client.get_server_health()
    live = await client.get_server_liveness()
    ready = await client.get_server_readiness()
    metrics = await client.get_server_metrics()
    security = await client.get_server_security_health()
    docs_info = await client.get_server_docs_info()
    limits = await client.get_flashcards_import_limits()
    tokenizer = await client.get_tokenizer_config()
    tokenizer_update = await client.update_tokenizer_config(
        TokenizerUpdateRequest(mode="char_approx", divisor=5)
    )
    jobs = await client.get_jobs_config()
    providers = await client.list_config_providers()
    validation = await client.validate_provider_key(
        ProviderValidateRequest(provider="openai", api_key="sk-test")
    )

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/health")
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/health/live")
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/health/ready")
    assert mocked.await_args_list[3].args[:2] == ("GET", "/api/v1/health/metrics")
    assert mocked.await_args_list[4].args[:2] == ("GET", "/api/v1/health/security")
    assert mocked.await_args_list[5].args[:2] == ("GET", "/api/v1/config/docs-info")
    assert mocked.await_args_list[6].args[:2] == ("GET", "/api/v1/config/flashcards-import-limits")
    assert mocked.await_args_list[7].args[:2] == ("GET", "/api/v1/config/tokenizer")
    assert mocked.await_args_list[8].args[:2] == ("PUT", "/api/v1/config/tokenizer")
    assert mocked.await_args_list[8].kwargs["json_data"] == {"mode": "char_approx", "divisor": 5}
    assert mocked.await_args_list[9].args[:2] == ("GET", "/api/v1/config/jobs")
    assert mocked.await_args_list[10].args[:2] == ("GET", "/api/v1/config/providers")
    assert mocked.await_args_list[11].args[:2] == ("POST", "/api/v1/config/validate-provider")
    assert mocked.await_args_list[11].kwargs["json_data"] == {"provider": "openai", "api_key": "sk-test"}
    assert isinstance(health, ServerHealthResponse)
    assert health.auth_mode == "multi_user"
    assert live.status == "alive"
    assert isinstance(ready, ServerReadinessResponse)
    assert ready.ready is True
    assert isinstance(metrics, ServerMetricsResponse)
    assert security.risk_level == "low"
    assert isinstance(docs_info, ServerDocsInfoResponse)
    assert docs_info.capabilities["hasAudio"] is True
    assert isinstance(limits, FlashcardsImportLimitsResponse)
    assert isinstance(tokenizer, TokenizerConfigResponse)
    assert tokenizer_update.mode == "char_approx"
    assert jobs.backend == "sqlite"
    assert isinstance(providers, ProvidersStatusResponse)
    assert providers.providers[0].key_source == "env"
    assert isinstance(validation, ProviderValidateResponse)
    assert validation.valid is True
