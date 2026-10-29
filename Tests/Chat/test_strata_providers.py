"""Named external Strata servers reuse the local Chat Completions boundary.

Wire fixtures follow Strata 46c152a and PolyStrata 504ff71 (serve/server.py).
They qualify client compatibility, not live inference or model quality.
"""

import asyncio
from io import BytesIO
import json
from types import SimpleNamespace
import tomllib

import httpx
import pytest
import requests

from tldw_chatbook.Agents.native_tools import provider_supports_native_tools
from tldw_chatbook.Chat.Chat_Functions import chat_api_call
from tldw_chatbook.Chat.console_context_window import (
    ContextWindowCache,
    ContextWindowTarget,
)
from tldw_chatbook.Chat.console_provider_gateway import (
    AuxiliaryCompletionRequest,
    ConsoleProviderGateway,
    ConsoleProviderResolution,
    ProviderThinkingDelta,
    ProviderToolCalls,
    _thinking_stream_capability,
)
from tldw_chatbook.Chat.console_provider_support import (
    build_local_thinking_payload_fields,
    console_generation_control_support,
    reasoning_effort_values_sent,
    resolve_console_provider_identity,
)
from tldw_chatbook.Chat.console_session_settings import settings_provider_catalog
from tldw_chatbook.Chat.local_reasoning_metadata import read_local_reasoning_metadata
from tldw_chatbook.Chat.provider_catalog import provider_display_name
from tldw_chatbook.Chat.provider_endpoint_contract import resolve_provider_endpoint
from tldw_chatbook.Chat.provider_readiness import (
    KEYLESS_PROVIDER_KEYS,
    get_provider_readiness,
    is_self_hosted_provider,
)
from tldw_chatbook.Chat.provider_setup_persistence import (
    canonical_provider_key,
    provider_endpoint_key,
)
from tldw_chatbook.config import CONFIG_TOML_CONTENT, LOCAL_PROVIDERS
from tldw_chatbook.LLM_Calls import LLM_API_Calls_Local as local
from tldw_chatbook.LLM_Calls.pricing_catalog import PricingCatalog
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_provider_identity import (
    resolve_provider_list_key,
)
from tldw_chatbook.LLM_Provider_Catalog.openai_compatible_model_discovery import (
    discover_openai_compatible_models,
)
from tldw_chatbook.provider_registry import RECORDS_BY_KEY


pytestmark = [
    pytest.mark.bootstrap_profile,
    pytest.mark.parametrize("provider", ["strata", "polystrata"]),
]


def test_named_identity_settings_and_defaults(provider):
    identity = resolve_console_provider_identity(provider)
    assert identity.is_supported
    assert identity.execution_key == identity.readiness_key == provider
    assert not identity.uses_direct_llama_path
    assert provider in {entry.execution_key for entry in settings_provider_catalog()}
    assert canonical_provider_key(provider) == provider
    assert provider_endpoint_key(provider) == "api_url"
    assert provider in KEYLESS_PROVIDER_KEYS
    assert is_self_hosted_provider(provider)
    record = RECORDS_BY_KEY[provider]
    assert record.classification == "local"
    assert provider_display_name(provider) == (
        "Strata" if provider == "strata" else "PolyStrata"
    )
    defaults = tomllib.loads(CONFIG_TOML_CONTENT)
    assert defaults["providers"][record.config_key] == []
    assert LOCAL_PROVIDERS[record.config_key] == []
    assert defaults["api_settings"][provider]["api_url"] == "http://127.0.0.1:8080/v1"
    assert defaults["api_settings"][provider]["api_key_env_var"] == "STRATA_API_KEY"
    assert (
        resolve_provider_list_key(provider, defaults["providers"]).provider_list_key
        == record.config_key
    )


@pytest.mark.parametrize("suffix", ["", "/v1", "/v1/chat/completions"])
def test_endpoint_forms_remain_openai_compatible(provider, suffix):
    endpoint = resolve_provider_endpoint(provider, "http://127.0.0.1:8080" + suffix)
    assert not endpoint.errors
    assert endpoint.chat_url == "http://127.0.0.1:8080/v1/chat/completions"
    assert endpoint.models_url == "http://127.0.0.1:8080/v1/models"


def test_reasoning_controls_use_strata_wire_fields(provider):
    assert build_local_thinking_payload_fields(provider, "none", 2048) == {
        "reasoning_effort": "none",
        "reasoning_budget_tokens": 2048,
    }
    assert reasoning_effort_values_sent(
        provider, ("none", "minimal", "low", "medium", "high", "xhigh")
    ) == (
        "none",
        "low",
        "medium",
        "high",
    )
    for control in ("reasoning_effort", "thinking_budget_tokens"):
        assert (
            console_generation_control_support(provider, "selected", control)
            == "supported"
        )


def test_server_default_reasoning_does_not_require_a_model_name_heuristic(provider):
    default = _thinking_stream_capability(provider, model="private-server-alias")
    assert default["thinking_stream_disposition"] == "displayable"
    assert (
        _thinking_stream_capability(provider, reasoning_effort="none")[
            "thinking_stream_disposition"
        ]
        == "ignored"
    )


def test_optional_credentials_use_the_upstream_server_variable(provider):
    config = tomllib.loads(CONFIG_TOML_CONTENT)
    assert get_provider_readiness(provider, config, environ={}).ready
    assert (
        get_provider_readiness(
            provider, config, environ={"STRATA_API_KEY": "fixture-key"}
        ).api_key
        == "fixture-key"
    )


def test_named_local_provider_has_zero_estimated_cost(provider):
    pricing = PricingCatalog(config={}).get_pricing(provider, "server-model")
    assert pricing is not None
    assert pricing.input_per_mtok == pricing.output_per_mtok == 0.0


@pytest.mark.asyncio
async def test_auxiliary_keyless_resolution_cannot_revive_stored_credential(
    provider, monkeypatch
):
    response = requests.Response()
    response.status_code = 200
    response._content = b'{"choices":[{"message":{"content":"summary"}}]}'
    posted = []

    def post(url, **kwargs):
        posted.append((url, kwargs))
        return response

    monkeypatch.setattr(
        local,
        "create_pooled_default_session",
        lambda **kwargs: SimpleNamespace(post=post),
    )
    monkeypatch.setattr(
        local,
        "get_runtime_config_snapshot",
        lambda: SimpleNamespace(
            values={
                "api_settings": {
                    provider: {"api_key": "must-not-revive"},
                }
            }
        ),
    )
    resolution = ConsoleProviderResolution(
        provider=provider,
        execution_key=provider,
        readiness_key=provider,
        model="selected",
        ready=True,
        api_key=None,
        base_url="http://127.0.0.1:8123/v1",
    )
    gateway = ConsoleProviderGateway()
    try:
        result = await gateway.complete_auxiliary(
            AuxiliaryCompletionRequest(
                resolution=resolution,
                messages=({"role": "user", "content": "summarize"},),
                response_format=None,
                max_output_tokens=128,
            )
        )
    finally:
        await gateway.aclose()
    assert result.text == "summary"
    assert len(posted) == 1
    assert posted[0][0] == "http://127.0.0.1:8123/v1/chat/completions"
    assert "Authorization" not in posted[0][1]["headers"]
    assert posted[0][1]["json"]["model"] == "selected"


@pytest.mark.parametrize(
    "credential_source, expected_key",
    [("stored", "configured-key"), ("environment", "env-key"), ("none", None)],
)
def test_dispatch_uses_own_configured_endpoint_model_and_credential(
    provider, monkeypatch, credential_source, expected_key
):
    response = requests.Response()
    response.status_code = 200
    response._content = b'{"choices":[{"message":{"content":"ok"}}]}'
    posted = []

    def post(url, **kwargs):
        posted.append((url, kwargs))
        return response

    monkeypatch.setattr(
        local,
        "create_pooled_default_session",
        lambda **kwargs: SimpleNamespace(post=post),
    )
    monkeypatch.setenv("STRATA_TEST_API_KEY", "env-key")
    monkeypatch.setattr(
        local,
        "get_runtime_config_snapshot",
        lambda: SimpleNamespace(
            values={
                "api_settings": {
                    provider: {
                        "api_url": "http://127.0.0.1:8124/v1",
                        "model": "configured-model",
                        "api_key": "configured-key",
                        "api_key_env_var": "STRATA_TEST_API_KEY",
                        "credential_source": credential_source,
                    },
                    "llama_cpp": {
                        "api_url": "http://wrong-provider.invalid",
                        "model": "wrong-model",
                        "api_key": "wrong-key",
                    },
                }
            }
        ),
    )
    result = chat_api_call(
        provider, [{"role": "user", "content": "hi"}], streaming=False
    )
    assert result["choices"][0]["message"]["content"] == "ok"
    assert posted[0][0] == "http://127.0.0.1:8124/v1/chat/completions"
    assert posted[0][1]["json"]["model"] == "configured-model"
    assert posted[0][1]["headers"].get("Authorization") == (
        f"Bearer {expected_key}" if expected_key else None
    )


@pytest.mark.asyncio
async def test_discovery_and_capacity_use_selected_model_metadata(provider):
    requests_seen = []
    payload = {
        "object": "list",
        "data": [
            {"id": "other", "meta": {"n_ctx": 4096}},
            {
                "id": "selected",
                "status": {"value": "unloaded"},
                "meta": {"n_ctx": 65536},
                "architecture": {
                    "input_modalities": ["text"],
                    "output_modalities": ["text"],
                },
            },
        ],
    }

    def reply(request):
        requests_seen.append(request)
        assert request.url.path == "/v1/models"
        assert request.headers["authorization"] == "Bearer fixture-key"
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
        result = await discover_openai_compatible_models(
            provider=provider,
            provider_list_key=provider,
            endpoint="http://127.0.0.1:8080",
            api_key="fixture-key",
            client=client,
        )
        assert result.status == "success"
        assert [model.model_id for model in result.models] == ["other", "selected"]
        capacity = await ContextWindowCache().resolve(
            ContextWindowTarget(
                provider,
                provider,
                "http://127.0.0.1:8080/v1",
                "selected",
                "fixture-key",
            ),
            client,
        )
        assert (capacity.tokens, capacity.verified) == (65536, True)
    assert len(requests_seen) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("caps", [None, False, True, "invalid_json", "oversized"])
async def test_native_tools_are_gated_by_selected_model_props(provider, caps):
    requests_seen = []

    def reply(request):
        requests_seen.append(request)
        assert request.url.path == "/props"
        assert request.url.params["model"] == "selected"
        assert request.headers["authorization"] == "Bearer fixture-key"
        if caps is None:
            return httpx.Response(404)
        if caps == "invalid_json":
            return httpx.Response(200, text="{")
        if caps == "oversized":
            return httpx.Response(200, content=b" " * 262145)
        return httpx.Response(
            200,
            json={
                "chat_template": "unreviewed template",
                "chat_template_caps": {
                    "supports_tools": caps,
                    "supports_tool_calls": caps,
                },
            },
        )

    assert not provider_supports_native_tools(provider)
    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
        gateway = ConsoleProviderGateway(http_client=client)
        resolution = ConsoleProviderResolution(
            provider=provider,
            execution_key=provider,
            model="selected",
            ready=True,
            base_url="http://127.0.0.1:8080/v1",
            api_key="fixture-key",
        )
        initial = await gateway._resolve_reasoning_history(resolution, {})
        assert not provider_supports_native_tools(
            provider, reasoning_replay=initial.reasoning_replay
        )
        await asyncio.gather(*gateway._reasoning_metadata_refreshes.values())
        resolved = await gateway._resolve_reasoning_history(resolution, {})
        assert resolved.ready
        assert resolved.reasoning_replay is not None
        assert not resolved.reasoning_replay.verified
        assert resolved.reasoning_replay.mode == "server_default"
        assert provider_supports_native_tools(
            provider, reasoning_replay=resolved.reasoning_replay
        ) is (caps is True)
        await gateway.aclose()
    assert len(requests_seen) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("extra_bytes", [0, 1])
async def test_template_metadata_read_keeps_exact_byte_bound(provider, extra_bytes):
    body = b'{"chat_template":"untrusted"}'.ljust(262144 + extra_bytes, b" ")

    def reply(request):
        assert dict(request.url.params) == {"model": provider, "autoload": "false"}
        assert request.headers["authorization"] == "Bearer fixture-key"
        assert request.extensions["timeout"]["read"] == 1.0
        return httpx.Response(200, content=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
        call = read_local_reasoning_metadata(
            client,
            "http://127.0.0.1:8080/props",
            headers={"Authorization": "Bearer fixture-key"},
            model=provider,
        )
        if extra_bytes:
            with pytest.raises(ValueError, match="exceeds size limit"):
                await call
        else:
            assert await call == ("untrusted", False)


@pytest.mark.asyncio
async def test_template_metadata_read_has_a_total_deadline(provider):
    async def never_reply(request):
        await asyncio.Event().wait()

    async with httpx.AsyncClient(transport=httpx.MockTransport(never_reply)) as client:
        outer_deadline = asyncio.timeout(2.0)
        with pytest.raises(TimeoutError):
            async with outer_deadline:
                await read_local_reasoning_metadata(
                    client,
                    "http://127.0.0.1:8080/props",
                    headers=None,
                    model=provider,
                )
        assert not outer_deadline.expired()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_named_adapter_preserves_tools_reasoning_and_content(
    provider, streaming, monkeypatch
):
    tool = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "calculator", "arguments": '{"expression":"1+1"}'},
    }
    message = {
        "content": "A heartbeat is normal.",
        "reasoning_content": "Compute it.",
        "tool_calls": [tool],
    }
    response = requests.Response()
    response.status_code = 200
    response.encoding = "utf-8"
    if streaming:
        body = ": keep-alive\n\nevent: heartbeat\nid: ignored\nretry: 1000\n\n"
        body += (
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {**message, "tool_calls": [{"index": 0, **tool}]},
                            "finish_reason": "tool_calls",
                        }
                    ]
                }
            )
            + "\n\ndata: [DONE]\n\n"
        )
        response.raw = BytesIO(body.encode())
    else:
        response._content = json.dumps(
            {"choices": [{"message": message, "finish_reason": "tool_calls"}]}
        ).encode()
    posted = []

    def post(url, **kwargs):
        posted.append((url, kwargs))
        return response

    monkeypatch.setattr(
        local,
        "create_pooled_default_session",
        lambda **kwargs: SimpleNamespace(post=post),
    )
    monkeypatch.setattr(
        local,
        "get_runtime_config_snapshot",
        lambda: SimpleNamespace(
            values={
                "api_settings": {
                    provider: {
                        "api_url": "http://127.0.0.1:8080/v1",
                        "api_key": "must-not-resurrect",
                    },
                    "llama_cpp": {
                        "api_url": "http://wrong-provider.invalid",
                        "api_key": "wrong-key",
                    },
                }
            }
        ),
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    resolution = ConsoleProviderResolution(
        provider=provider,
        execution_key=provider,
        readiness_key=provider,
        model="selected",
        ready=True,
        base_url="http://127.0.0.1:8123/v1",
        api_key=None,
        streaming=streaming,
        reasoning_effort="low",
        thinking_budget_tokens=2048,
        thinking_stream_disposition="displayable",
        thinking_round_trip_version=1,
    )
    gateway = ConsoleProviderGateway()
    try:
        chunks = [
            item
            async for item in gateway.stream_chat(
                resolution, [{"role": "user", "content": "hi"}], tools=tools
            )
        ]
    finally:
        await gateway.aclose()
    assert (
        "".join(item for item in chunks if isinstance(item, str)) == message["content"]
    )
    assert [
        item.text for item in chunks if isinstance(item, ProviderThinkingDelta)
    ] == [message["reasoning_content"]]
    assert [
        item.tool_calls for item in chunks if isinstance(item, ProviderToolCalls)
    ] == [(tool,)]
    assert len(posted) == 1
    url, request = posted[0]
    assert url == "http://127.0.0.1:8123/v1/chat/completions"
    assert "Authorization" not in request["headers"]
    assert request["json"]["model"] == "selected"
    assert request["json"]["tools"] == tools
    assert request["json"]["stream"] is streaming
    assert request["json"]["reasoning_effort"] == "low"
    assert request["json"]["reasoning_budget_tokens"] == 2048
    assert "chat_template_kwargs" not in request["json"]
