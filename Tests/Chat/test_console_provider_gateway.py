import asyncio
import builtins
import dataclasses
from copy import deepcopy
import http.server
import json
import threading
import time

import httpx
import pytest

from tldw_chatbook.Chat.Chat_Deps import (
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatConfigurationError,
    ChatProviderError,
    ChatRateLimitError,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
from tldw_chatbook.Chat.console_dispatch_checkpoint import ConsoleEgressClass
from tldw_chatbook.Chat.console_provider_gateway import (
    MAX_AUXILIARY_OUTPUT_TOKENS,
    AuxiliaryCompletionRequest,
    AuxiliaryCompletionResult,
    GENERATION_READ_TIMEOUT_SECONDS,
    NO_PROVIDER_CONTENT_COPY,
    PROBE_TIMEOUT_SECONDS,
    UNSUPPORTED_PROVIDER_RESPONSE_COPY,
    ConsoleProviderGateway,
    ConsoleProviderResolution,
    ConsoleProviderStreamSignals,
    LlamaCppProviderConfig,
    ProviderProprietaryThinkingEvidence,
    ProviderThinkingDelta,
    ProviderThinkingCaptureError,
    ProviderToolCalls,
    build_llamacpp_chat_payload,
    normalize_llamacpp_base_url,
    safe_provider_error_copy,
)
from tldw_chatbook.Utils.sensitive_llm_logging import is_sensitive_llm_request
from tldw_chatbook.Chat.console_provider_support import (
    resolve_console_provider_identity,
)
from tldw_chatbook.Chat import console_provider_gateway as gateway_module
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationConflictError,
    ContinuationRestoreTarget,
    parse_provider_continuation_json,
)
from tldw_chatbook.Chat.console_history_budget import ProviderContinuationSidecar
from tldw_chatbook.Chat.console_exchange_capture import (
    CaptureBudget,
    CaptureDetail,
    build_request_capture,
)
from tldw_chatbook.LLM_Calls.hosted_chat import HostedChatTurn


def test_provider_thinking_events_are_bounded_and_content_free_in_repr() -> None:
    canary = "DISPLAYABLE-THINKING-CANARY"
    event = ProviderThinkingDelta(
        text=canary,
        provider="llama_cpp",
        model="qwen",
        protocol="chat_completions",
        source_format="start_anchored_think",
    )

    assert event.text == canary
    assert canary not in repr(event)
    assert dataclasses.fields(event)[0].name == "text"
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.model = "other"  # type: ignore[misc]


def test_proprietary_thinking_event_cannot_carry_content_surrogates() -> None:
    event = ProviderProprietaryThinkingEvidence(
        provider="moonshot",
        model="kimi-k3",
        protocol="chat_completions",
        source_format="reasoning_content",
    )

    assert {field.name for field in dataclasses.fields(event)} == {
        "provider",
        "model",
        "protocol",
        "source_format",
    }
    assert not hasattr(event, "__dict__")
    assert "PRIVATE-REASONING-CANARY" not in repr(event)
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.provider = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"text": ""},
        {"text": "x" * (256 * 1024 + 1)},
        {"provider": ""},
        {"model": "x" * 201},
        {"protocol": ""},
        {"source_format": ""},
    ],
)
def test_provider_thinking_delta_rejects_invalid_or_oversized_values(
    kwargs: dict[str, str],
) -> None:
    values = {
        "text": "safe",
        "provider": "llama_cpp",
        "model": "qwen",
        "protocol": "chat_completions",
        "source_format": "start_anchored_think",
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match="Invalid provider thinking event") as error:
        ProviderThinkingDelta(**values)

    invalid_text = values.get("text", "")
    if invalid_text:
        assert invalid_text not in str(error.value)


def test_provider_resolution_defaults_to_ignored_thinking_capability() -> None:
    resolution = ConsoleProviderResolution(
        provider="unknown",
        base_url="https://example.test/v1",
        model="reasoner",
        ready=True,
        execution_key="unknown",
    )

    assert resolution.thinking_stream_disposition == "ignored"
    assert resolution.thinking_round_trip_version is None
    assert resolution.may_emit_thinking is False


def test_gateway_consumes_provider_owned_reasoning_disposition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gateway_module.MoonshotFinishPolicy,
        "reasoning_disposition",
        "ignored",
    )

    assert gateway_module._thinking_stream_capability("moonshot") == {
        "thinking_stream_disposition": "ignored",
        "thinking_round_trip_version": None,
    }


@pytest.mark.parametrize(
    ("disposition", "version"),
    [
        ("unknown", None),
        ("ignored", 1),
        ("displayable", None),
        ("displayable", True),
        ("displayable", 2),
        ("proprietary", None),
    ],
)
def test_provider_resolution_rejects_incoherent_thinking_capability(
    disposition: str,
    version: int | None,
) -> None:
    with pytest.raises(ValueError, match="Invalid provider thinking capability"):
        ConsoleProviderResolution(
            provider="test",
            base_url="https://example.test/v1",
            model="reasoner",
            ready=True,
            execution_key="test",
            thinking_stream_disposition=disposition,  # type: ignore[arg-type]
            thinking_round_trip_version=version,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("selected_provider", "provider_key", "model", "base_url", "api_mode", "protocol"),
    [
        (
            "Moonshot",
            "moonshot",
            "kimi-latest",
            "https://api.moonshot.ai/v1",
            None,
            "chat_completions",
        ),
        (
            "ZAI",
            "zai",
            "glm-4.5",
            "https://api.z.ai/api/paas/v4",
            None,
            "chat_completions",
        ),
        (
            "deepseek",
            "deepseek",
            "deepseek-v4-flash",
            "https://api.deepseek.com",
            None,
            "chat_completions",
        ),
        (
            "deepseek",
            "deepseek",
            "deepseek-v4-flash",
            "https://api.deepseek.com",
            "  ChAt_CoMpLeTiOnS  ",
            "chat_completions",
        ),
        (
            "deepseek",
            "deepseek",
            "deepseek-v4-flash",
            "https://api.deepseek.com",
            "  ReSpOnSeS  ",
            "responses",
        ),
    ],
)
async def test_real_resolution_pins_provider_continuation_protocol_before_prepare(
    selected_provider: str,
    provider_key: str,
    model: str,
    base_url: str,
    api_mode: str | None,
    protocol: str,
) -> None:
    settings = {
        "api_key": f"{provider_key}-test-key",
        "model": model,
        "api_base_url": base_url,
    }
    if api_mode is not None:
        settings["api_mode"] = api_mode
    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {provider_key: settings}},
        environ={},
    )

    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider=selected_provider)
    )
    checkpoint = parse_provider_continuation_json(
        {
            "schema_version": 1,
            "checkpoint_revision": 1,
            "provider": provider_key,
            "protocol": protocol,
            "model": model,
            "api_base_url": base_url,
            "state": "complete",
            "rounds": [
                {
                    "assistant_content": "answer",
                    "reasoning_blocks": ["private"],
                    "calls": [
                        {
                            "call_id": "call_1",
                            "name": "lookup",
                            "arguments": "{}",
                            "state": "completed",
                            "result": "done",
                        }
                    ],
                }
            ],
        }
    )
    target = ContinuationRestoreTarget(provider_key, model, protocol, base_url)

    assert resolution.ready is True
    assert resolution.continuation_protocol == protocol
    assert resolution.api_mode == (protocol if provider_key == "deepseek" else None)
    expected_disposition = (
        "proprietary" if provider_key in {"moonshot", "zai"} else "ignored"
    )
    assert resolution.thinking_stream_disposition == expected_disposition
    assert resolution.thinking_round_trip_version == (
        1 if expected_disposition != "ignored" else None
    )
    assert resolution.may_emit_thinking is (expected_disposition != "ignored")
    prepared = gateway.prepare_chat_request(
        resolution,
        [{"_owner": "a1", "role": "assistant", "content": "answer"}],
        continuation_target=target,
        continuation_sidecar=(ProviderContinuationSidecar("a1", checkpoint),),
        continuation_owner_key="_owner",
    )
    assert prepared.continuation_groups[0].checkpoint == checkpoint


@pytest.mark.asyncio
async def test_deepseek_resolution_rejects_invalid_present_api_mode() -> None:
    gateway = ConsoleProviderGateway(
        config_provider=lambda: {
            "api_settings": {
                "deepseek": {
                    "api_key": "DEEPSEEK-SECRET-CANARY",
                    "model": "deepseek-v4-flash",
                    "api_mode": "response",
                }
            }
        },
        environ={},
    )

    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="deepseek")
    )

    assert resolution.ready is False
    assert "DeepSeek" in resolution.visible_copy
    assert "API mode" in resolution.visible_copy
    assert "DEEPSEEK-SECRET-CANARY" not in resolution.visible_copy


def test_gateway_prepare_budgets_private_owner_group_on_real_production_path() -> None:
    checkpoint = parse_provider_continuation_json(
        {
            "schema_version": 1,
            "checkpoint_revision": 1,
            "provider": "deepseek",
            "protocol": "responses",
            "model": "deepseek-v4-flash",
            "api_base_url": "https://api.deepseek.com/v1",
            "state": "complete",
            "rounds": [
                {
                    "assistant_content": "old answer",
                    "reasoning_blocks": ["GATEWAY-PRIVATE-CANARY " * 30],
                    "calls": [
                        {
                            "call_id": "call_1",
                            "name": "lookup",
                            "arguments": "{}",
                            "state": "completed",
                            "result": "done",
                        }
                    ],
                }
            ],
        }
    )
    messages = [
        {"_owner": "u1", "role": "user", "content": "old"},
        {
            "_owner": "a1",
            "role": "assistant",
            "content": "old answer",
        },
        {"_owner": "u2", "role": "user", "content": "current"},
    ]
    gateway = ConsoleProviderGateway(environ={})
    resolution = ConsoleProviderResolution(
        provider="deepseek",
        base_url="https://api.deepseek.com/v1",
        model="deepseek-v4-flash",
        ready=True,
        execution_key="deepseek",
        max_tokens=10,
        continuation_protocol="responses",
    )
    target = ContinuationRestoreTarget(
        provider="deepseek",
        protocol="responses",
        model="deepseek-v4-flash",
        api_base_url="https://api.deepseek.com/v1",
    )

    prepared = gateway.prepare_chat_request(
        resolution,
        messages,
        context_window_override_tokens=600,
        continuation_target=target,
        continuation_sidecar=(ProviderContinuationSidecar("a1", checkpoint),),
        continuation_owner_key="_owner",
    )
    ordinary = gateway.prepare_chat_request(
        resolution,
        [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "current"},
        ],
        context_window_override_tokens=600,
    )

    assert prepared.dropped_units == 1
    assert ordinary.dropped_units == 0
    assert [row["content"] for row in prepared.messages_payload] == ["current"]
    assert all("provider_continuation" not in row for row in prepared.messages_payload)
    assert "GATEWAY-PRIVATE-CANARY" not in repr(prepared)
    assert messages[1] == {
        "_owner": "a1",
        "role": "assistant",
        "content": "old answer",
    }

    with pytest.raises(ContinuationConflictError, match="restore target mismatch"):
        gateway.prepare_chat_request(
            resolution,
            messages,
            continuation_target=dataclasses.replace(target, model="wrong-model"),
            continuation_sidecar=(ProviderContinuationSidecar("a1", checkpoint),),
            continuation_owner_key="_owner",
        )
    with pytest.raises(ContinuationConflictError, match="restore target mismatch"):
        gateway.prepare_chat_request(
            dataclasses.replace(resolution, provider="moonshot"),
            messages,
            continuation_target=target,
            continuation_sidecar=(ProviderContinuationSidecar("a1", checkpoint),),
            continuation_owner_key="_owner",
        )


def test_normalize_llamacpp_base_url_strips_known_suffixes_to_root() -> None:
    root = "http://localhost:8080"
    assert normalize_llamacpp_base_url("http://localhost:8080/completion") == root
    assert normalize_llamacpp_base_url("http://localhost:8080/v1") == root
    assert (
        normalize_llamacpp_base_url("http://localhost:8080/v1/chat/completions") == root
    )
    assert normalize_llamacpp_base_url("http://localhost:8080") == root
    assert (
        normalize_llamacpp_base_url("localhost:8080/completion") == root
    )  # scheme-less
    # a reverse-proxy prefix is NOT an exact suffix -> left unchanged
    assert (
        normalize_llamacpp_base_url("http://host/proxy/v1/chat/completions")
        == "http://host/proxy/v1/chat/completions"
    )


def test_llamacpp_payload_includes_supported_sampling_params() -> None:
    payload = build_llamacpp_chat_payload(
        model="m",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
        temperature=0.4,
        top_p=0.7,
        min_p=0.03,
        top_k=20,
        max_tokens=300,
    )

    assert payload == {
        "model": "m",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
        "temperature": 0.4,
        "top_p": 0.7,
        "min_p": 0.03,
        "top_k": 20,
        "max_tokens": 300,
    }


def test_direct_llamacpp_payload_consumes_ephemeral_origin_marker_copy_only() -> None:
    from tldw_chatbook.Chat.console_project_instructions import EPHEMERAL_ORIGIN_KEY

    messages = [
        {
            "role": "user",
            "content": "context",
            EPHEMERAL_ORIGIN_KEY: "project_instructions",
        }
    ]
    payload = build_llamacpp_chat_payload(model="m", messages=messages, stream=False)
    assert EPHEMERAL_ORIGIN_KEY not in payload["messages"][0]
    assert messages[0][EPHEMERAL_ORIGIN_KEY] == "project_instructions"


def test_llamacpp_payload_omits_blank_provider_defaults() -> None:
    payload = build_llamacpp_chat_payload(
        model="m",
        messages=[],
        stream=False,
        temperature=None,
        top_p=None,
        min_p=None,
        top_k=None,
        max_tokens=None,
    )

    assert payload == {"model": "m", "messages": [], "stream": False}


def test_llamacpp_payload_includes_explicit_top_k_zero() -> None:
    payload = build_llamacpp_chat_payload(
        model="m",
        messages=[],
        stream=False,
        top_k=0,
    )

    assert payload == {"model": "m", "messages": [], "stream": False, "top_k": 0}


def test_llamacpp_payload_disables_thinking_for_trailing_assistant_message() -> None:
    """A trailing assistant message is a response prefill; llama.cpp rejects
    prefills when the chat template's thinking mode is enabled, so the
    payload must ask the server to disable it."""
    payload = build_llamacpp_chat_payload(
        model="m",
        messages=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Sure, here is"},
        ],
        stream=True,
    )

    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_llamacpp_payload_omits_thinking_kwarg_for_trailing_user_message() -> None:
    payload = build_llamacpp_chat_payload(
        model="m",
        messages=[
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "hello"},
        ],
        stream=True,
    )

    assert "chat_template_kwargs" not in payload


def test_llamacpp_payload_omits_thinking_kwarg_for_empty_messages() -> None:
    payload = build_llamacpp_chat_payload(
        model="m",
        messages=[],
        stream=False,
    )

    assert "chat_template_kwargs" not in payload


class TestLlamacppThinkingPayload:
    def test_effort_composes_chat_template_kwargs(self):
        payload = build_llamacpp_chat_payload(
            model="qwen", messages=[{"role": "user", "content": "hi"}],
            stream=True, reasoning_effort="low",
        )
        assert payload["chat_template_kwargs"] == {"reasoning_effort": "low"}

    def test_budget_composes_reasoning_budget_tokens(self):
        payload = build_llamacpp_chat_payload(
            model="qwen", messages=[{"role": "user", "content": "hi"}],
            stream=False, thinking_budget_tokens=2048,
        )
        assert payload["reasoning_budget_tokens"] == 2048

    def test_none_effort_disables_thinking(self):
        payload = build_llamacpp_chat_payload(
            model="qwen", messages=[{"role": "user", "content": "hi"}],
            stream=True, reasoning_effort="none",
        )
        assert payload["chat_template_kwargs"]["enable_thinking"] is False

    def test_prefill_overrides_effort(self):
        payload = build_llamacpp_chat_payload(
            model="qwen",
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "Sure"},
            ],
            stream=True, reasoning_effort="xhigh",
        )
        # prefill > none > effort (llama.cpp rejects prefill + thinking)
        assert payload["chat_template_kwargs"] == {
            "reasoning_effort": "xhigh",
            "enable_thinking": False,
        }

    def test_no_thinking_fields_by_default(self):
        payload = build_llamacpp_chat_payload(
            model="qwen", messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        assert "chat_template_kwargs" not in payload
        assert "reasoning_budget_tokens" not in payload


@pytest.mark.asyncio
async def test_llamacpp_prefers_explicit_model_but_still_probes_reachability():
    seen_paths = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        assert request.url.path == "/health"
        return httpx.Response(200, json={"status": "ok"})

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://127.0.0.1:9099",
        )
    )

    resolved = await gateway.resolve_llamacpp(
        LlamaCppProviderConfig(
            base_url="http://127.0.0.1:9099", explicit_model="explicit-model"
        )
    )

    assert resolved.ready is True
    assert resolved.model == "explicit-model"
    assert seen_paths == ["/health"]


@pytest.mark.asyncio
async def test_llamacpp_prefers_configured_model_but_still_probes_reachability():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(404, text="no health route, but server is reachable")

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://127.0.0.1:9099",
        )
    )

    resolved = await gateway.resolve_llamacpp(
        LlamaCppProviderConfig(
            base_url="http://127.0.0.1:9099", configured_model="configured-model"
        )
    )

    assert resolved.ready is True
    assert resolved.model == "configured-model"


@pytest.mark.asyncio
async def test_llamacpp_explicit_model_blocks_when_reachability_probe_cannot_connect():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        raise httpx.ConnectError("connection refused", request=request)

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://127.0.0.1:9099",
        )
    )

    resolved = await gateway.resolve_llamacpp(
        LlamaCppProviderConfig(
            base_url="http://127.0.0.1:9099", explicit_model="explicit-model"
        )
    )

    assert resolved.ready is False
    assert resolved.model == "explicit-model"
    assert "not reachable" in resolved.visible_copy


@pytest.mark.asyncio
async def test_llamacpp_uses_first_models_endpoint_result_when_no_configured_model():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": "server-model"}]})

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://127.0.0.1:9099",
        )
    )

    resolved = await gateway.resolve_llamacpp(
        LlamaCppProviderConfig(base_url="http://127.0.0.1:9099")
    )

    assert resolved.ready is True
    assert resolved.model == "server-model"


@pytest.mark.asyncio
async def test_llamacpp_unreachable_server_returns_blocked_recovery_copy():
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://127.0.0.1:9099",
        )
    )

    resolved = await gateway.resolve_llamacpp(
        LlamaCppProviderConfig(base_url="http://127.0.0.1:9099")
    )

    assert resolved.ready is False
    assert resolved.model is None
    assert "Provider blocked" in resolved.visible_copy
    assert "127.0.0.1:9099" in resolved.visible_copy


@pytest.mark.asyncio
async def test_llamacpp_empty_models_without_configured_model_returns_blocked_recovery_copy():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []})

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://127.0.0.1:9099",
        )
    )

    resolved = await gateway.resolve_llamacpp(
        LlamaCppProviderConfig(base_url="http://127.0.0.1:9099")
    )

    assert resolved.ready is False
    assert resolved.model is None
    assert (
        resolved.visible_copy
        == "Provider blocked: select or configure a llama.cpp model."
    )


@pytest.mark.asyncio
async def test_llamacpp_non_object_models_payload_returns_blocked_recovery_copy():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://127.0.0.1:9099",
        )
    )

    resolved = await gateway.resolve_llamacpp(
        LlamaCppProviderConfig(base_url="http://127.0.0.1:9099")
    )

    assert resolved.ready is False
    assert resolved.model is None
    assert (
        resolved.visible_copy
        == "Provider blocked: select or configure a llama.cpp model."
    )


@pytest.mark.asyncio
async def test_resolve_for_send_dispatches_llamacpp_selection():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "server-model"}]})

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://127.0.0.1:9099",
        )
    )

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="llama_cpp", base_url="http://127.0.0.1:9099")
    )

    assert resolved.ready is True
    assert resolved.provider == "llama_cpp"
    assert resolved.model == "server-model"
    assert resolved.thinking_stream_disposition == "displayable"
    assert resolved.thinking_round_trip_version == 1
    assert resolved.may_emit_thinking is True


@pytest.mark.asyncio
async def test_resolve_for_send_copies_sampling_fields_to_llamacpp_resolution():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "server-model"}]})

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://127.0.0.1:9099",
        )
    )

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider="llama_cpp",
            base_url="http://127.0.0.1:9099",
            temperature=0.4,
            top_p=0.8,
            min_p=0.05,
            top_k=30,
            max_tokens=500,
            seed=11,
            presence_penalty=0.2,
            frequency_penalty=0.3,
            reasoning_effort="high",
            reasoning_summary="auto",
            verbosity="medium",
            thinking_effort="low",
            thinking_budget_tokens=2048,
            streaming=False,
        )
    )

    assert resolved.ready is True
    assert resolved.temperature == 0.4
    assert resolved.top_p == 0.8
    assert resolved.min_p == 0.05
    assert resolved.top_k == 30
    assert resolved.max_tokens == 500
    assert resolved.seed == 11
    assert resolved.presence_penalty == 0.2
    assert resolved.frequency_penalty == 0.3
    assert resolved.reasoning_effort == "high"
    assert resolved.reasoning_summary == "auto"
    assert resolved.verbosity == "medium"
    assert resolved.thinking_effort == "low"
    assert resolved.thinking_budget_tokens == 2048
    assert resolved.streaming is False


@pytest.mark.asyncio
async def test_resolve_for_send_normalizes_scheme_less_llamacpp_base_url_before_http():
    seen_urls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(200, json={"data": [{"id": "server-model"}]})

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="llama_cpp", base_url="127.0.0.1:9099/v1")
    )

    assert resolved.ready is True
    assert resolved.base_url == "http://127.0.0.1:9099"
    assert seen_urls == ["http://127.0.0.1:9099/v1/models"]


@pytest.mark.asyncio
async def test_resolve_for_send_blocks_invalid_llamacpp_base_url_before_http():
    requests = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": [{"id": "server-model"}]})

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="llama_cpp", base_url="file:///etc/passwd")
    )

    assert resolved.ready is False
    assert resolved.base_url == "file:///etc/passwd"
    assert "invalid llama.cpp base URL" in resolved.visible_copy
    assert requests == []


@pytest.mark.asyncio
async def test_gateway_resolves_direct_llamacpp_without_importing_chat_functions(
    monkeypatch,
):
    real_import = builtins.__import__

    def fail_chat_functions_import(name, *args, **kwargs):
        if name == "tldw_chatbook.Chat.Chat_Functions":
            raise AssertionError(
                "direct llama resolution should not import Chat_Functions"
            )
        return real_import(name, *args, **kwargs)

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(200, json={"status": "ok"})

    monkeypatch.setattr(builtins, "__import__", fail_chat_functions_import)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    gateway = ConsoleProviderGateway(http_client=client)

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider="llama_cpp", base_url="http://127.0.0.1:9099", explicit_model="m"
        )
    )

    assert resolved.ready is True
    await client.aclose()


@pytest.mark.asyncio
async def test_resolve_for_send_blocks_unsupported_provider_with_recovery_copy():
    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(500))
        )
    )

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider="future_provider",
            temperature=0.3,
            top_p=0.9,
            min_p=0.02,
            top_k=40,
            max_tokens=600,
            streaming=False,
        )
    )

    assert resolved.ready is False
    assert resolved.provider == "future_provider"
    assert resolved.temperature == 0.3
    assert resolved.top_p == 0.9
    assert resolved.min_p == 0.02
    assert resolved.top_k == 40
    assert resolved.max_tokens == 600
    assert resolved.streaming is False
    assert resolved.visible_copy == (
        "Provider blocked: 'future_provider' is not available in Console yet. Choose a supported provider."
    )


@pytest.mark.asyncio
async def test_resolve_for_send_openai_uses_env_key_and_execution_key() -> None:
    gateway = ConsoleProviderGateway(
        config_provider=lambda: {
            "api_settings": {"openai": {"api_key_env_var": "OPENAI_API_KEY"}}
        },
        environ={"OPENAI_API_KEY": "sk-test-secret"},
    )

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider="openai", explicit_model="gpt-4.1", streaming=False
        )
    )

    assert resolved.ready is True
    assert resolved.provider == "openai"
    assert resolved.readiness_key == "openai"
    assert resolved.execution_key == "openai"
    assert resolved.api_key == "sk-test-secret"
    assert resolved.api_key_source == "env:OPENAI_API_KEY"
    assert "sk-test-secret" not in resolved.visible_copy
    assert "sk-test-secret" not in repr(resolved)


@pytest.mark.asyncio
async def test_resolve_for_send_all_chat_api_handlers_are_console_supported() -> None:
    from tldw_chatbook.Chat.Chat_Functions import API_CALL_HANDLERS
    from tldw_chatbook.Chat.provider_readiness import PROVIDERS_REQUIRING_API_KEY_KEYS

    handler_keys = frozenset(API_CALL_HANDLERS)
    api_settings: dict[str, dict[str, str]] = {}
    for provider in handler_keys:
        identity = resolve_console_provider_identity(
            provider,
            handler_keys=handler_keys,
        )
        settings = api_settings.setdefault(
            identity.readiness_key,
            {"model": f"{identity.readiness_key}-model"},
        )
        if identity.readiness_key in PROVIDERS_REQUIRING_API_KEY_KEYS:
            settings["api_key"] = f"test-key-for-{identity.readiness_key}"

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"status": "ok"})
        )
    ) as client:
        gateway = ConsoleProviderGateway(
            http_client=client,
            config_provider=lambda: {"api_settings": api_settings},
            environ={},
        )

        for provider in sorted(handler_keys):
            identity = resolve_console_provider_identity(
                provider,
                handler_keys=handler_keys,
            )
            resolved = await gateway.resolve_for_send(
                ConsoleProviderSelection(
                    provider=provider, explicit_model="console-sweep-model"
                )
            )

            assert resolved.ready is True, provider
            assert resolved.readiness_key == identity.readiness_key, provider
            assert resolved.execution_key == identity.execution_key, provider
            assert "not available in Console yet" not in resolved.visible_copy
            assert "WIP" not in resolved.visible_copy


@pytest.mark.asyncio
async def test_resolve_for_send_supported_provider_missing_key_blocks_without_wip() -> (
    None
):
    gateway = ConsoleProviderGateway(
        config_provider=lambda: {
            "api_settings": {"anthropic": {"api_key_env_var": "ANTHROPIC_API_KEY"}}
        },
        environ={},
    )

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="anthropic", explicit_model="claude-sonnet")
    )

    assert resolved.ready is False
    assert "Missing API key" in resolved.visible_copy
    assert "not wired" not in resolved.visible_copy
    assert "WIP" not in resolved.visible_copy


@pytest.mark.asyncio
async def test_resolve_for_send_custom_alias_uses_custom_openai_execution_key() -> None:
    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"custom": {"model": "m"}}},
        environ={},
    )

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="Custom", configured_model="m")
    )

    assert resolved.ready is True
    assert resolved.readiness_key == "custom"
    assert resolved.execution_key == "custom-openai-api"


@pytest.mark.asyncio
async def test_resolve_for_send_blocks_generic_base_url_override_that_differs_from_config() -> (
    None
):
    gateway = ConsoleProviderGateway(
        config_provider=lambda: {
            "api_settings": {"ollama": {"api_url": "http://127.0.0.1:11434"}}
        },
        environ={},
    )

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider="ollama",
            explicit_model="llama3",
            base_url="http://user:secret@127.0.0.1:9999/v1",
        )
    )

    assert resolved.ready is False
    assert "save the endpoint in Settings" in resolved.visible_copy
    assert "Selected endpoint: http://127.0.0.1:9999/v1" in resolved.visible_copy
    assert "Saved endpoint: http://127.0.0.1:11434" in resolved.visible_copy
    assert "user" not in resolved.visible_copy
    assert "secret" not in resolved.visible_copy


@pytest.mark.asyncio
async def test_resolve_for_send_preserves_explicit_cloud_url_without_configured_endpoint() -> (
    None
):
    gateway = ConsoleProviderGateway(
        config_provider=lambda: {
            "api_settings": {
                "openai": {"api_key": "unit-test-key", "model": "gpt-4.1"}
            },
        },
        environ={},
    )

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider="openai",
            explicit_model="gpt-4.1",
            base_url="http://127.0.0.1:9999/v1",
        )
    )

    assert resolved.ready is True
    assert resolved.readiness_key == "openai"
    assert resolved.execution_key == "openai"
    assert resolved.base_url == "http://127.0.0.1:9999/v1"
    assert "save the endpoint in Settings" not in resolved.visible_copy


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "model", "expected_base_url"),
    [
        ("openai", "gpt-test", "https://api.openai.com/v1"),
        ("anthropic", "claude-test", "https://api.anthropic.com/v1"),
    ],
)
async def test_resolve_for_send_materializes_builtin_cloud_endpoint(
    provider: str,
    model: str,
    expected_base_url: str,
) -> None:
    gateway = ConsoleProviderGateway(
        config_provider=lambda: {
            "api_settings": {provider: {"api_key": "unit-test-key", "model": model}}
        },
        environ={},
    )

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider=provider, explicit_model=model)
    )

    assert resolved.ready is True
    assert resolved.base_url == expected_base_url
    assert resolved.resolved_destination is not None
    assert resolved.resolved_destination.endpoint_identity == (
        expected_base_url.split("/v1", maxsplit=1)[0]
    )
    assert (
        resolved.resolved_destination.egress_class
        is ConsoleEgressClass.PUBLIC_NETWORK
    )


@pytest.mark.asyncio
async def test_gateway_attaches_on_device_destination_after_llamacpp_normalization():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "server-model"}]})

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider="llama_cpp",
            base_url="127.42.7.9:9099/v1/chat/completions",
        )
    )

    assert resolved.ready is True
    assert resolved.base_url == "http://127.42.7.9:9099"
    assert resolved.resolved_destination is not None
    assert resolved.resolved_destination.endpoint_identity == "http://127.42.7.9:9099"
    assert resolved.resolved_destination.egress_class is ConsoleEgressClass.ON_DEVICE


@pytest.mark.asyncio
async def test_gateway_unknown_custom_destination_identity_is_credential_free():
    endpoint = (
        "https://user:URL-SECRET@models.example.test:8443/private/v1"
        "?api_key=URL-SECRET#fragment"
    )
    gateway = ConsoleProviderGateway(
        config_provider=lambda: {
            "api_settings": {
                "openai": {
                    "api_key": "CONFIG-SECRET",
                    "model": "gpt-test",
                    "api_base_url": endpoint,
                }
            }
        },
        environ={},
    )

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="openai", explicit_model="gpt-test")
    )

    assert resolved.ready is True
    assert resolved.resolved_destination is not None
    assert resolved.resolved_destination.endpoint_identity == (
        "https://models.example.test:8443"
    )
    assert resolved.resolved_destination.egress_class is ConsoleEgressClass.UNKNOWN
    rendered = repr(resolved.resolved_destination)
    for secret in ("user", "URL-SECRET", "CONFIG-SECRET", "private", "api_key"):
        assert secret not in rendered


@pytest.mark.asyncio
async def test_resolve_for_send_materializes_configured_huggingface_router() -> None:
    router_base_url = "https://router.example.test/hf-inference"
    api_base_url = "https://api-base.example.test/v1"
    gateway = ConsoleProviderGateway(
        config_provider=lambda: {
            "api_settings": {
                "huggingface": {
                    "api_key": "unit-test-key",
                    "model": "org/model",
                    "use_router_url_format": True,
                    "router_base_url": router_base_url,
                    "api_base_url": api_base_url,
                }
            }
        },
        environ={},
    )

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="huggingface", explicit_model="org/model")
    )

    assert resolved.ready is True
    assert resolved.base_url == router_base_url


@pytest.mark.asyncio
async def test_resolve_for_send_nonrouter_huggingface_preserves_api_base_precedence() -> (
    None
):
    api_base_url = "https://api-base.example.test/v1"
    gateway = ConsoleProviderGateway(
        config_provider=lambda: {
            "api_settings": {
                "huggingface": {
                    "api_key": "unit-test-key",
                    "model": "org/model",
                    "use_router_url_format": False,
                    "router_base_url": "https://router.example.test/hf-inference",
                    "api_base_url": api_base_url,
                }
            }
        },
        environ={},
    )

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="huggingface", explicit_model="org/model")
    )

    assert resolved.ready is True
    assert resolved.base_url == api_base_url


@pytest.mark.asyncio
async def test_resolve_for_send_router_huggingface_preserves_builtin_default() -> None:
    gateway = ConsoleProviderGateway(
        config_provider=lambda: {
            "api_settings": {
                "huggingface": {
                    "api_key": "unit-test-key",
                    "model": "org/model",
                    "use_router_url_format": True,
                }
            }
        },
        environ={},
    )

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="huggingface", explicit_model="org/model")
    )

    assert resolved.ready is True
    assert resolved.base_url == "https://router.huggingface.co/hf-inference"


@pytest.mark.asyncio
async def test_resolve_for_send_accepts_generic_base_url_matching_config_with_trailing_slash() -> (
    None
):
    gateway = ConsoleProviderGateway(
        config_provider=lambda: {
            "api_settings": {"ollama": {"api_url": "http://127.0.0.1:11434"}}
        },
        environ={},
    )

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider="ollama",
            explicit_model="llama3",
            base_url="http://127.0.0.1:11434/",
        )
    )

    assert resolved.ready is True
    assert resolved.base_url == "http://127.0.0.1:11434/"
    assert resolved.model == "llama3"


@pytest.mark.asyncio
async def test_resolve_for_send_accepts_generic_base_url_matching_default_port() -> (
    None
):
    gateway = ConsoleProviderGateway(
        config_provider=lambda: {
            "api_settings": {"ollama": {"api_url": "http://example.test"}}
        },
        environ={},
    )

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider="ollama",
            explicit_model="llama3",
            base_url="http://example.test:80/",
        )
    )

    assert resolved.ready is True


@pytest.mark.asyncio
async def test_resolve_for_send_blocks_malformed_generic_base_url_without_crashing() -> (
    None
):
    gateway = ConsoleProviderGateway(
        config_provider=lambda: {
            "api_settings": {"ollama": {"api_url": "http://127.0.0.1:11434"}}
        },
        environ={},
    )

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider="ollama", explicit_model="llama3", base_url="http://[::1"
        )
    )

    assert resolved.ready is False
    assert "save the endpoint in Settings" in resolved.visible_copy


@pytest.mark.asyncio
async def test_resolve_for_send_reads_config_provider_at_resolution_time() -> None:
    configs = [
        {
            "api_settings": {
                "openai": {"api_key_env_var": "OPENAI_API_KEY", "model": "old-model"}
            }
        },
        {
            "api_settings": {
                "openai": {"api_key_env_var": "OPENAI_API_KEY", "model": "new-model"}
            }
        },
    ]

    def config_provider() -> dict[str, object]:
        return configs.pop(0)

    gateway = ConsoleProviderGateway(
        config_provider=config_provider,
        environ={"OPENAI_API_KEY": "sk-test-secret"},
    )

    first = await gateway.resolve_for_send(ConsoleProviderSelection(provider="openai"))
    second = await gateway.resolve_for_send(ConsoleProviderSelection(provider="openai"))

    assert first.ready is True
    assert first.model == "old-model"
    assert second.ready is True
    assert second.model == "new-model"
    assert configs == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "selected_base_url",
    [
        "https://workspace.example.test/compatible-mode/v1",
        "https://workspace.example.test/compatible-mode/v1/responses",
        "https://workspace.example.test/compatible-mode/v1/chat/completions/",
    ],
)
async def test_qwencloud_resolution_pins_normalized_mode_and_base(
    selected_base_url: str,
) -> None:
    gateway = ConsoleProviderGateway(
        config_provider=lambda: {
            "api_settings": {
                "qwencloud": {
                    "api_key": "qwen-test-key",
                    "api_mode": "  ReSpOnSeS  ",
                    "api_base_url": (
                        "https://workspace.example.test/compatible-mode/v1"
                    ),
                    "model": "qwen3.8-max",
                }
            }
        },
        environ={},
    )

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider="QwenCloud",
            base_url=selected_base_url,
        )
    )

    assert resolved.ready is True
    assert resolved.execution_key == "qwencloud"
    assert resolved.api_mode == "responses"
    assert resolved.continuation_protocol is None
    assert resolved.base_url == "https://workspace.example.test/compatible-mode/v1"


@pytest.mark.asyncio
@pytest.mark.parametrize("canonical_first", [False, True])
async def test_qwencloud_resolution_prefers_canonical_fields_and_alias_fallbacks(
    canonical_first: bool,
) -> None:
    alias = {
        "api_key": "alias-key",
        "api_mode": "responses",
        "api_base_url": "https://alias.example.test/compatible-mode/v1",
        "model": "qwen-alias-model",
    }
    canonical = {"api_mode": "chat_completions"}
    entries = (
        [("qwencloud", canonical), ("QwenCloud", alias)]
        if canonical_first
        else [("QwenCloud", alias), ("qwencloud", canonical)]
    )
    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": dict(entries)},
        environ={},
    )

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="QwenCloud")
    )

    assert resolved.ready is True
    assert resolved.api_key == "alias-key"
    assert resolved.api_mode == "chat_completions"
    assert resolved.model == "qwen-alias-model"
    assert resolved.base_url == "https://alias.example.test/compatible-mode/v1"


@pytest.mark.asyncio
@pytest.mark.parametrize("canonical_first", [False, True])
async def test_qwencloud_resolution_blocks_malformed_canonical_table_without_alias_leakage(
    canonical_first: bool,
) -> None:
    alias = {
        "api_key": "ALIAS-SECRET-CANARY",
        "model": "ALIAS-MODEL-CANARY",
    }
    entries = (
        [("qwencloud", []), ("QwenCloud", alias)]
        if canonical_first
        else [("QwenCloud", alias), ("qwencloud", [])]
    )
    source = {"api_settings": dict(entries)}
    original = deepcopy(source)
    gateway = ConsoleProviderGateway(config_provider=lambda: source, environ={})

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="QwenCloud")
    )

    assert source == original
    assert resolved.ready is False
    assert "provider settings" in resolved.visible_copy.lower()
    assert "api_settings.qwencloud" in resolved.visible_copy
    assert "ALIAS-SECRET-CANARY" not in resolved.visible_copy
    assert "ALIAS-MODEL-CANARY" not in resolved.visible_copy


@pytest.mark.asyncio
@pytest.mark.parametrize("canonical_first", [False, True])
async def test_qwencloud_resolution_ignores_malformed_alias_when_canonical_is_valid(
    canonical_first: bool,
) -> None:
    canonical = {
        "api_key": "canonical-key",
        "api_mode": "responses",
        "api_base_url": "https://canonical.example.test/compatible-mode/v1",
        "model": "canonical-model",
    }
    entries = (
        [("qwencloud", canonical), ("QwenCloud", [])]
        if canonical_first
        else [("QwenCloud", []), ("qwencloud", canonical)]
    )
    source = {"api_settings": dict(entries)}
    original = deepcopy(source)
    gateway = ConsoleProviderGateway(config_provider=lambda: source, environ={})

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="QwenCloud")
    )

    assert source == original
    assert resolved.ready is True
    assert resolved.api_key == "canonical-key"
    assert resolved.model == "canonical-model"
    assert resolved.base_url == "https://canonical.example.test/compatible-mode/v1"


@pytest.mark.asyncio
async def test_qwencloud_resolution_blocks_alias_only_malformed_table():
    source = {"api_settings": {"QwenCloud": ["SECRET-CANARY"]}}
    original = deepcopy(source)
    gateway = ConsoleProviderGateway(config_provider=lambda: source, environ={})

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="QwenCloud")
    )

    assert source == original
    assert resolved.ready is False
    assert "provider settings" in resolved.visible_copy.lower()
    assert "api_settings.qwencloud" in resolved.visible_copy
    assert "SECRET-CANARY" not in resolved.visible_copy


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("settings", "setting_copy"),
    [
        (
            {
                "api_mode": "response",
                "api_base_url": ("https://workspace.example.test/compatible-mode/v1"),
            },
            "API mode",
        ),
        (
            {
                "api_mode": "responses",
                "api_base_url": (
                    "https://workspace.example.test/compatible-mode/v1?token="
                    "ENDPOINT-PAYLOAD-CANARY"
                ),
            },
            "API base URL",
        ),
    ],
)
async def test_qwencloud_resolution_rejects_invalid_mode_before_dispatch(
    settings: dict[str, str],
    setting_copy: str,
) -> None:
    gateway = ConsoleProviderGateway(
        config_provider=lambda: {
            "api_settings": {
                "qwencloud": {
                    **settings,
                    "api_key": "QWENCLOUD-KEY-CANARY",
                    "model": "PAYLOAD-MODEL-CANARY",
                }
            }
        },
        environ={},
        chat_api_call_fn=lambda **_kwargs: pytest.fail(
            "invalid QwenCloud settings must fail before dispatch"
        ),
    )

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="QwenCloud")
    )

    assert resolved.ready is False
    assert "QwenCloud" in resolved.visible_copy
    assert setting_copy in resolved.visible_copy
    assert "QWENCLOUD-KEY-CANARY" not in resolved.visible_copy
    assert "PAYLOAD-MODEL-CANARY" not in resolved.visible_copy
    assert "ENDPOINT-PAYLOAD-CANARY" not in resolved.visible_copy


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "configured_base_url",
    [42, False, [], {}, "", "   ", None],
    ids=("integer", "boolean", "list", "mapping", "empty", "whitespace", "none"),
)
async def test_qwencloud_resolution_rejects_present_malformed_saved_base_before_network_or_dispatch(
    configured_base_url: object,
) -> None:
    requests: list[httpx.Request] = []
    dispatches: list[dict[str, object]] = []

    async def network_trap(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": [{"id": "unexpected-model"}]})

    def dispatch_trap(**kwargs: object) -> object:
        dispatches.append(dict(kwargs))
        return {"choices": [{"message": {"content": "unexpected"}}]}

    async with httpx.AsyncClient(transport=httpx.MockTransport(network_trap)) as client:
        gateway = ConsoleProviderGateway(
            http_client=client,
            config_provider=lambda: {
                "api_settings": {
                    "qwencloud": {
                        "api_key": "QWENCLOUD-KEY-CANARY",
                        "api_mode": "responses",
                        "api_base_url": configured_base_url,
                        "model": "PAYLOAD-MODEL-CANARY",
                    }
                }
            },
            environ={},
            chat_api_call_fn=dispatch_trap,
        )

        resolved = await gateway.resolve_for_send(
            ConsoleProviderSelection(provider="QwenCloud")
        )

    assert resolved.ready is False
    assert "QwenCloud" in resolved.visible_copy
    assert "API base URL" in resolved.visible_copy
    assert "QWENCLOUD-KEY-CANARY" not in resolved.visible_copy
    assert "PAYLOAD-MODEL-CANARY" not in resolved.visible_copy
    assert requests == []
    assert dispatches == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "model", "settings", "expected_api_mode"),
    [
        ("openai", "gpt-4.1", {"api_key": "openai-key"}, None),
        (
            "deepseek",
            "deepseek-chat",
            {"api_key": "deepseek-key"},
            "chat_completions",
        ),
        (
            "anthropic",
            "claude-sonnet-4-6",
            {"api_key": "anthropic-key"},
            None,
        ),
    ],
)
async def test_non_qwen_resolution_api_mode_isolated_to_deepseek(
    provider: str,
    model: str,
    settings: dict[str, str],
    expected_api_mode: str | None,
) -> None:
    gateway = ConsoleProviderGateway(
        config_provider=lambda: {
            "api_settings": {provider: {**settings, "model": model}}
        },
        environ={},
    )

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider=provider)
    )

    assert resolved.ready is True
    assert resolved.api_mode == expected_api_mode
    assert resolved.continuation_protocol == expected_api_mode


@pytest.mark.asyncio
async def test_all_qwencloud_kwargs_paths_forward_pinned_mode_and_base() -> None:
    pinned_base = "https://workspace-a.example.test/compatible-mode/v1"
    gateway = ConsoleProviderGateway(
        config_provider=lambda: {
            "api_settings": {
                "qwencloud": {
                    "api_key": "qwen-test-key",
                    "api_mode": "responses",
                    "api_base_url": f"{pinned_base}/responses",
                    "model": "qwen3.8-max",
                }
            }
        },
        environ={},
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="QwenCloud")
    )
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hello."},
    ]
    prepared = gateway.prepare_chat_request(resolution, messages)
    auxiliary = AuxiliaryCompletionRequest(
        resolution=resolution,
        messages=tuple(messages),
        response_format=None,
        max_output_tokens=64,
    )

    kwargs_paths = (
        gateway._chat_api_kwargs_from_prepared(resolution, prepared),
        gateway._chat_api_kwargs(resolution, messages),
        gateway._auxiliary_chat_api_kwargs(auxiliary, resolution),
    )

    for kwargs in kwargs_paths:
        assert kwargs["api_mode"] == "responses"
        assert kwargs["api_base_url"] == pinned_base

    for provider, base_url in (
        ("openai", "https://api.openai.com/v1"),
        ("deepseek", "https://api.deepseek.com"),
        ("anthropic", "https://api.anthropic.com/v1"),
    ):
        other = dataclasses.replace(
            resolution,
            provider=provider,
            readiness_key=provider,
            execution_key=provider,
            base_url=base_url,
            api_mode=None,
        )
        other_prepared = gateway.prepare_chat_request(other, messages)
        other_auxiliary = AuxiliaryCompletionRequest(
            resolution=other,
            messages=tuple(messages),
            response_format=None,
            max_output_tokens=64,
        )
        primary_kwargs = gateway._chat_api_kwargs(other, messages)
        prepared_kwargs = gateway._chat_api_kwargs_from_prepared(
            other,
            other_prepared,
        )
        auxiliary_kwargs = gateway._auxiliary_chat_api_kwargs(
            other_auxiliary,
            other,
        )

        assert "api_mode" not in primary_kwargs
        assert "api_mode" not in prepared_kwargs
        assert "api_mode" not in auxiliary_kwargs
        if provider == "anthropic":
            assert primary_kwargs["api_base_url"] == base_url
            assert prepared_kwargs["api_base_url"] == base_url
        else:
            assert "api_base_url" not in primary_kwargs
            assert "api_base_url" not in prepared_kwargs
        assert auxiliary_kwargs["api_base_url"] == base_url


@pytest.mark.asyncio
async def test_qwencloud_run_ignores_midrun_config_mutation() -> None:
    base_a = "https://workspace-a.example.test/compatible-mode/v1"
    config: dict[str, object] = {
        "api_settings": {
            "qwencloud": {
                "api_key": "qwen-test-key",
                "api_mode": "responses",
                "api_base_url": base_a,
                "model": "qwen3.8-max",
            }
        }
    }
    calls: list[dict[str, object]] = []

    def fake_chat_api_call(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return {"choices": [{"message": {"content": "ok"}}]}

    gateway = ConsoleProviderGateway(
        config_provider=lambda: config,
        environ={},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="QwenCloud", streaming=False)
    )
    config["api_settings"] = {
        "qwencloud": {
            "api_key": "mutated-key",
            "api_mode": "chat_completions",
            "api_base_url": (
                "https://workspace-b.example.test/compatible-mode/v1/chat/completions"
            ),
            "model": "mutated-model",
        }
    }

    for content in ("turn one", "turn two"):
        assert [
            chunk
            async for chunk in gateway.stream_chat(
                resolution,
                [{"role": "user", "content": content}],
            )
        ] == ["ok"]

    auxiliary = AuxiliaryCompletionRequest(
        resolution=resolution,
        messages=({"role": "user", "content": "auxiliary"},),
        response_format=None,
        max_output_tokens=64,
    )
    assert (await gateway.complete_auxiliary(auxiliary)).text == "ok"

    assert len(calls) == 3
    for call in calls:
        assert call["api_mode"] == "responses"
        assert call["api_base_url"] == base_a
        assert call["api_key"] == "qwen-test-key"
        assert call["model"] == "qwen3.8-max"


@pytest.mark.asyncio
async def test_llamacpp_stream_chat_yields_content_chunks():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.method == "POST"
        body = (
            b'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
            b"data: [DONE]\n\n"
        )
        return httpx.Response(200, content=body)

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://127.0.0.1:9099",
        )
    )

    chunks = [
        chunk
        async for chunk in gateway.stream_llamacpp_chat(
            base_url="http://127.0.0.1:9099",
            model="test-model",
            messages=[{"role": "user", "content": "say hello"}],
        )
    ]

    assert chunks == ["hel", "lo"]


@pytest.mark.asyncio
async def test_llamacpp_stream_chat_falls_back_to_non_streaming_when_stream_rejected():
    request_payloads = []

    async def handler(request: httpx.Request) -> httpx.Response:
        request_payloads.append(request.read())
        if len(request_payloads) == 1:
            return httpx.Response(400, json={"error": "streaming disabled"})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "fallback completion"}}]},
        )

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://127.0.0.1:9099",
        )
    )

    chunks = [
        chunk
        async for chunk in gateway.stream_llamacpp_chat(
            base_url="http://127.0.0.1:9099",
            model="test-model",
            messages=[{"role": "user", "content": "say hello"}],
        )
    ]

    assert chunks == ["fallback completion"]
    assert b'"stream":true' in request_payloads[0]
    assert b'"stream":false' in request_payloads[1]


@pytest.mark.asyncio
async def test_llamacpp_stream_chat_falls_back_when_sse_has_no_content_chunks():
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(200, content=b"data: {not-json}\n\ndata: [DONE]\n\n")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "fallback after bad sse"}}]},
        )

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://127.0.0.1:9099",
        )
    )

    chunks = [
        chunk
        async for chunk in gateway.stream_llamacpp_chat(
            base_url="http://127.0.0.1:9099",
            model="test-model",
            messages=[{"role": "user", "content": "say hello"}],
        )
    ]

    assert chunks == ["fallback after bad sse"]
    assert calls == 2


@pytest.mark.asyncio
async def test_llamacpp_stream_chat_ignores_non_object_json_sse_lines():
    async def handler(request: httpx.Request) -> httpx.Response:
        body = (
            b"data: []\n\n"
            b"data: null\n\n"
            b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
            b"data: [DONE]\n\n"
        )
        return httpx.Response(200, content=body)

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://127.0.0.1:9099",
        )
    )

    chunks = [
        chunk
        async for chunk in gateway.stream_llamacpp_chat(
            base_url="http://127.0.0.1:9099",
            model="test-model",
            messages=[{"role": "user", "content": "say hello"}],
        )
    ]

    assert chunks == ["ok"]


def make_gateway_with_sse(lines: list[str]) -> ConsoleProviderGateway:
    """Gateway whose llama.cpp endpoint streams the given SSE lines."""
    body = "".join(f"{line}\n\n" for line in lines).encode()

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(200, content=body)

    return ConsoleProviderGateway(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://127.0.0.1:8080",
        )
    )


def make_gateway_with_completion(payload: dict) -> ConsoleProviderGateway:
    """Gateway whose llama.cpp endpoint answers one non-streaming completion."""

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(200, json=payload)

    return ConsoleProviderGateway(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://127.0.0.1:8080",
        )
    )


class TestDirectPathThinkingEvents:
    @pytest.mark.asyncio
    async def test_stream_emits_typed_start_anchored_thinking_with_frozen_identity(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        canary = "DISPLAYABLE-THINKING-CANARY"
        lines = [
            f'data: {{"choices":[{{"delta":{{"content":"<think>{canary}</think>Answer"}}}}]}}',
            "data: [DONE]",
        ]
        gateway = make_gateway_with_sse(lines)

        items = [
            item
            async for item in gateway.stream_llamacpp_chat(
                base_url="http://127.0.0.1:8080",
                model="qwen",
                messages=[{"role": "user", "content": "hi"}],
                provider="local_llamacpp",
                protocol="chat_completions",
            )
        ]

        assert items[1:] == ["Answer"]
        event = items[0]
        assert isinstance(event, ProviderThinkingDelta)
        assert event.text == canary
        assert (
            event.provider,
            event.model,
            event.protocol,
            event.source_format,
        ) == (
            "local_llamacpp",
            "qwen",
            "chat_completions",
            "start_anchored_think",
        )
        assert canary not in repr(event)
        assert canary not in caplog.text

    @pytest.mark.asyncio
    async def test_stream_with_no_think_tag_emits_no_thinking_event(self) -> None:
        gateway = make_gateway_with_sse(
            [
                'data: {"choices":[{"delta":{"content":"Answer"}}]}',
                "data: [DONE]",
            ]
        )

        items = [
            item
            async for item in gateway.stream_llamacpp_chat(
                base_url="http://127.0.0.1:8080",
                model="qwen",
                messages=[{"role": "user", "content": "hi"}],
            )
        ]

        assert items == ["Answer"]

    @pytest.mark.asyncio
    async def test_unclosed_thinking_emits_partial_then_content_free_failure(
        self,
    ) -> None:
        canary = "UNCLOSED-THINKING-CANARY"
        gateway = make_gateway_with_sse(
            [
                f'data: {{"choices":[{{"delta":{{"content":"<think>{canary}"}}}}]}}',
                "data: [DONE]",
            ]
        )
        stream = gateway.stream_llamacpp_chat(
            base_url="http://127.0.0.1:8080",
            model="qwen",
            messages=[{"role": "user", "content": "hi"}],
        )

        event = await anext(stream)
        assert isinstance(event, ProviderThinkingDelta)
        with pytest.raises(ProviderThinkingCaptureError) as error:
            await anext(stream)
        assert canary not in str(error.value)

    @pytest.mark.asyncio
    async def test_nonstream_console_send_emits_thinking_before_visible_answer(
        self,
    ) -> None:
        canary = "NONSTREAM-THINKING-CANARY"
        gateway = make_gateway_with_completion(
            {"choices": [{"message": {"content": f"<think>{canary}</think>Answer"}}]}
        )
        resolution = ConsoleProviderResolution(
            provider="llama_cpp",
            base_url="http://127.0.0.1:8080",
            model="qwen",
            ready=True,
            execution_key="llama_cpp",
            streaming=False,
            thinking_stream_disposition="displayable",
            thinking_round_trip_version=1,
        )

        items = [
            item
            async for item in gateway.stream_chat(
                resolution, [{"role": "user", "content": "hi"}]
            )
        ]

        assert isinstance(items[0], ProviderThinkingDelta)
        assert items[0].text == canary
        assert items[1:] == ["Answer"]

    @pytest.mark.asyncio
    async def test_nonstream_unclosed_thinking_emits_delta_before_safe_failure(
        self,
    ) -> None:
        canary = "NONSTREAM-UNCLOSED-THINKING-CANARY"
        gateway = make_gateway_with_completion(
            {"choices": [{"message": {"content": f"<think>{canary}"}}]}
        )
        resolution = ConsoleProviderResolution(
            provider="llama_cpp",
            base_url="http://127.0.0.1:8080",
            model="qwen",
            ready=True,
            execution_key="llama_cpp",
            streaming=False,
            thinking_stream_disposition="displayable",
            thinking_round_trip_version=1,
        )
        stream = gateway.stream_chat(
            resolution, [{"role": "user", "content": "hi"}]
        )

        event = await anext(stream)
        assert isinstance(event, ProviderThinkingDelta)
        assert event.text == canary
        with pytest.raises(ProviderThinkingCaptureError) as error:
            await anext(stream)
        assert canary not in str(error.value)


class _TerminalHostedResponse:
    def __init__(self, items: list[dict], turn: HostedChatTurn) -> None:
        self._items = iter(items)
        self.terminal_turn = turn

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._items)

    def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_proprietary_hosted_reasoning_emits_one_content_free_terminal_event(
    caplog: pytest.LogCaptureFixture,
) -> None:
    canary = "PRIVATE-REASONING-CANARY"
    turn = HostedChatTurn(
        text="Answer",
        tool_calls=(),
        assistant_message={"role": "assistant", "content": "Answer"},
        finish_reason="stop",
        reasoning_content=canary,
    )
    gateway = ConsoleProviderGateway(
        chat_api_call_fn=lambda **_kwargs: _TerminalHostedResponse(
            [{"choices": [{"delta": {"content": "Answer"}}]}], turn
        ),
        environ={},
    )
    resolution = ConsoleProviderResolution(
        provider="moonshot",
        base_url="https://api.moonshot.ai/v1",
        model="kimi-k3",
        ready=True,
        execution_key="moonshot",
        continuation_protocol="chat_completions",
        thinking_stream_disposition="proprietary",
        thinking_round_trip_version=1,
    )

    items = [
        item
        async for item in gateway.stream_chat(
            resolution, [{"role": "user", "content": "hi"}]
        )
    ]

    assert items[0] == "Answer"
    assert len(items) == 2
    evidence = items[1]
    assert isinstance(evidence, ProviderProprietaryThinkingEvidence)
    assert (
        evidence.provider,
        evidence.model,
        evidence.protocol,
        evidence.source_format,
    ) == (
        "moonshot",
        "kimi-k3",
        "chat_completions",
        "reasoning_content",
    )
    assert canary not in repr(evidence)
    assert canary not in caplog.text


@pytest.mark.asyncio
async def test_proprietary_capability_without_current_reasoning_emits_no_event() -> None:
    turn = HostedChatTurn(
        text="Answer",
        tool_calls=(),
        assistant_message={"role": "assistant", "content": "Answer"},
        finish_reason="stop",
        reasoning_content=None,
    )
    gateway = ConsoleProviderGateway(
        chat_api_call_fn=lambda **_kwargs: _TerminalHostedResponse(
            [{"choices": [{"delta": {"content": "Answer"}}]}], turn
        ),
        environ={},
    )
    resolution = ConsoleProviderResolution(
        provider="zai",
        base_url="https://api.z.ai/api/paas/v4",
        model="glm-5.2",
        ready=True,
        execution_key="zai",
        continuation_protocol="chat_completions",
        thinking_stream_disposition="proprietary",
        thinking_round_trip_version=1,
    )

    items = [
        item
        async for item in gateway.stream_chat(
            resolution, [{"role": "user", "content": "hi"}]
        )
    ]

    assert items == ["Answer"]


@pytest.mark.asyncio
async def test_vllm_displayable_disposition_splits_start_anchored_thinking() -> None:
    canary = "VLLM-THINKING-CANARY"
    gateway = ConsoleProviderGateway(
        chat_api_call_fn=lambda **_kwargs: iter(
            [{"choices": [{"delta": {"content": f"<think>{canary}</think>Answer"}}]}]
        ),
        environ={},
    )
    resolution = ConsoleProviderResolution(
        provider="vllm",
        base_url="http://127.0.0.1:8000/v1",
        model="local-model",
        ready=True,
        execution_key="vllm",
        thinking_stream_disposition="displayable",
        thinking_round_trip_version=1,
    )

    items = [
        item
        async for item in gateway.stream_chat(
            resolution, [{"role": "user", "content": "hi"}]
        )
    ]

    assert isinstance(items[0], ProviderThinkingDelta)
    assert items[0].text == canary
    assert items[0].provider == "vllm"
    assert items[1:] == ["Answer"]


@pytest.mark.asyncio
async def test_ignored_generic_reasoning_fields_and_tags_remain_visible_only() -> None:
    canary = "IGNORED-REASONING-CANARY"
    gateway = ConsoleProviderGateway(
        chat_api_call_fn=lambda **_kwargs: iter(
            [
                {
                    "choices": [
                        {
                            "delta": {
                                "reasoning_content": canary,
                                "content": "<think>literal</think>Answer",
                            }
                        }
                    ]
                }
            ]
        ),
        environ={},
    )
    resolution = ConsoleProviderResolution(
        provider="unknown",
        base_url="https://example.test/v1",
        model="model",
        ready=True,
        execution_key="unknown",
    )

    items = [
        item
        async for item in gateway.stream_chat(
            resolution, [{"role": "user", "content": "hi"}]
        )
    ]

    assert items == ["<think>literal</think>Answer"]
    assert canary not in repr(items)


class TestDirectPathThinkFiltering:
    @pytest.mark.asyncio
    async def test_stream_strips_start_anchored_think_block(self):
        # SSE lines whose content deltas spell:
        #   "<think>ponder</think>Hello"
        lines = [
            'data: {"choices":[{"delta":{"content":"<think>pon"}}]}',
            'data: {"choices":[{"delta":{"content":"der</think>Hello"}}]}',
            "data: [DONE]",
        ]
        gateway = make_gateway_with_sse(lines)
        items = [
            item
            async for item in gateway.stream_llamacpp_chat(
                base_url="http://127.0.0.1:8080",
                model="qwen",
                messages=[{"role": "user", "content": "hi"}],
            )
        ]
        assert "".join(
            item.text for item in items if isinstance(item, ProviderThinkingDelta)
        ) == "ponder"
        assert "".join(item for item in items if isinstance(item, str)) == "Hello"

    @pytest.mark.asyncio
    async def test_stream_passes_mid_reply_literal_tag(self):
        lines = [
            'data: {"choices":[{"delta":{"content":"XML: <think>x</think>"}}]}',
            "data: [DONE]",
        ]
        gateway = make_gateway_with_sse(lines)
        chunks = [
            chunk
            async for chunk in gateway.stream_llamacpp_chat(
                base_url="http://127.0.0.1:8080",
                model="qwen",
                messages=[{"role": "user", "content": "hi"}],
            )
        ]
        assert "".join(chunks) == "XML: <think>x</think>"

    @pytest.mark.asyncio
    async def test_stream_ignores_reasoning_content_deltas(self):
        lines = [
            'data: {"choices":[{"delta":{"reasoning_content":"secret"}}]}',
            'data: {"choices":[{"delta":{"content":"Answer"}}]}',
            "data: [DONE]",
        ]
        gateway = make_gateway_with_sse(lines)
        chunks = [
            chunk
            async for chunk in gateway.stream_llamacpp_chat(
                base_url="http://127.0.0.1:8080",
                model="qwen",
                messages=[{"role": "user", "content": "hi"}],
            )
        ]
        assert "".join(chunks) == "Answer"

    @pytest.mark.asyncio
    async def test_complete_strips_start_anchored_think_block(self):
        gateway = make_gateway_with_completion(
            {"choices": [{"message": {"content": "<think>x</think>Done"}}]}
        )
        text = await gateway.complete_llamacpp_chat(
            base_url="http://127.0.0.1:8080",
            model="qwen",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert text == "Done"

    @pytest.mark.asyncio
    async def test_think_only_stream_skips_nonstreaming_fallback(self):
        # A reply that is entirely start-anchored think text must not cost a
        # second (non-streaming fallback) round-trip: the retry would return
        # the same filtered-to-empty text.
        lines = [
            'data: {"choices":[{"delta":{"content":"<think>only"}}]}',
            'data: {"choices":[{"delta":{"content":" pondering</think>"}}]}',
            "data: [DONE]",
        ]
        body = "".join(f"{line}\n\n" for line in lines).encode()
        requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, content=body)

        gateway = ConsoleProviderGateway(
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(handler),
                base_url="http://127.0.0.1:8080",
            )
        )
        items = [
            item
            async for item in gateway.stream_llamacpp_chat(
                base_url="http://127.0.0.1:8080",
                model="qwen",
                messages=[{"role": "user", "content": "hi"}],
            )
        ]
        assert "".join(
            item.text for item in items if isinstance(item, ProviderThinkingDelta)
        ) == "only pondering"
        assert not any(isinstance(item, str) for item in items)
        assert len(requests) == 1


@pytest.mark.asyncio
async def test_stream_chat_dispatches_llamacpp_resolution():
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(
            200,
            content=b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n',
        )

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://127.0.0.1:9099",
        )
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider="llama_cpp",
            base_url="http://127.0.0.1:9099",
            explicit_model="test-model",
        )
    )

    chunks = [
        chunk
        async for chunk in gateway.stream_chat(
            resolution, [{"role": "user", "content": "hello"}]
        )
    ]

    assert chunks == ["ok"]


@pytest.mark.asyncio
async def test_stream_chat_non_streaming_resolution_yields_completion_once() -> None:
    seen_payloads = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "done"}}]})

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    resolution = ConsoleProviderResolution(
        provider="llama_cpp",
        base_url="http://127.0.0.1:9099",
        model="m",
        ready=True,
        streaming=False,
        temperature=0.2,
    )

    chunks = [
        chunk
        async for chunk in gateway.stream_chat(
            resolution, [{"role": "user", "content": "hi"}]
        )
    ]

    assert chunks == ["done"]
    assert seen_payloads == [
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "temperature": 0.2,
        }
    ]


@pytest.mark.asyncio
async def test_stream_chat_generic_non_streaming_yields_completion_once() -> None:
    calls = []

    def fake_chat_api_call(**kwargs):
        calls.append(kwargs)
        return "generic done"

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"openai": {"api_key": "sk-test"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider="openai",
            explicit_model="gpt-4.1",
            streaming=False,
            temperature=0.2,
            top_p=0.9,
            min_p=0.05,
            top_k=40,
            max_tokens=256,
            seed=123,
            presence_penalty=0.4,
            frequency_penalty=0.5,
            reasoning_effort="high",
            reasoning_summary="auto",
            verbosity="medium",
        )
    )

    chunks = [
        chunk
        async for chunk in gateway.stream_chat(
            resolution, [{"role": "user", "content": "hi"}]
        )
    ]

    assert chunks == ["generic done"]
    assert calls == [
        {
            "api_endpoint": "openai",
            "messages_payload": [{"role": "user", "content": "hi"}],
            "api_key": "sk-test",
            "model": "gpt-4.1",
            "streaming": False,
            "temp": 0.2,
            "topp": 0.9,
            "maxp": 0.9,
            "minp": 0.05,
            "topk": 40,
            "max_tokens": 256,
            "seed": 123,
            "presence_penalty": 0.4,
            "frequency_penalty": 0.5,
            "reasoning_effort": "high",
            "reasoning_summary": "auto",
            "verbosity": "medium",
        }
    ]


@pytest.mark.asyncio
async def test_stream_chat_generic_sync_generator_yields_ordered_chunks() -> None:
    def fake_chat_api_call(**_kwargs):
        yield "hel"
        yield {"choices": [{"delta": {"content": "lo"}}]}

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"openai": {"api_key": "sk-test"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="openai", explicit_model="gpt-4.1")
    )

    chunks = [
        chunk
        async for chunk in gateway.stream_chat(
            resolution, [{"role": "user", "content": "hi"}]
        )
    ]

    assert chunks == ["hel", "lo"]


@pytest.mark.asyncio
async def test_stream_chat_generic_completion_dict_yields_message_content() -> None:
    def fake_chat_api_call(**_kwargs):
        return {"choices": [{"message": {"content": "complete dict"}}]}

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"openai": {"api_key": "sk-test"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="openai", explicit_model="gpt-4.1")
    )

    chunks = [
        chunk
        async for chunk in gateway.stream_chat(
            resolution, [{"role": "user", "content": "hi"}]
        )
    ]

    assert chunks == ["complete dict"]


def test_normalize_generic_provider_response_shapes() -> None:
    unsupported = "Provider returned an unsupported response shape."
    no_content = "Provider returned no assistant content."

    class IterableSdkResponse:
        def __iter__(self):
            yield {"content": "do not dump"}

    assert list(
        ConsoleProviderGateway.normalize_provider_response({"content": "body"})
    ) == ["body"]
    assert list(
        ConsoleProviderGateway.normalize_provider_response(
            {"choices": [{"message": {"content": "choice"}}]}
        )
    ) == ["choice"]
    assert list(
        ConsoleProviderGateway.normalize_provider_response(
            {"generated_text": "generated"}
        )
    ) == ["generated"]
    assert list(
        ConsoleProviderGateway.normalize_provider_response([{"content": "do not dump"}])
    ) == [unsupported]
    assert list(
        ConsoleProviderGateway.normalize_provider_response(
            ({"content": "do not dump"},)
        )
    ) == [unsupported]
    assert list(ConsoleProviderGateway.normalize_provider_response(b"hello \xff")) == [
        "hello \ufffd"
    ]
    assert list(ConsoleProviderGateway.normalize_provider_response(iter(()))) == [
        no_content
    ]
    assert list(
        ConsoleProviderGateway.normalize_provider_response(
            {"unexpected": {"secret": "do not dump"}}
        )
    ) == [unsupported]
    assert list(
        ConsoleProviderGateway.normalize_provider_response(IterableSdkResponse())
    ) == [unsupported]


def test_stream_signal_privacy_has_one_private_event_and_a_public_usage_payload() -> (
    None
):
    # Explicit opt-in: the dataclass default is False (review finding I1) --
    # this test exercises begin_exchange/record_exchange_content/
    # close_exchange and needs capture actually happening to prove the
    # privacy claim.
    signals = gateway_module.ConsoleProviderStreamSignals(exchange_capture_enabled=True)

    signal_fields = dataclasses.fields(signals)
    assert [item.name for item in signal_fields] == [
        "_synthetic_fallback",
        "model_retry_callback",
        "usage_payload",
        "completed_usage_payloads",
        "_active_usage_payloads",
        "_usage_lock",
        "run_tag",
        "exchange_capture_enabled",
        "capture_detail",
        "completed_exchanges",
        "_active_exchanges",
        "_exchange_lock",
    ]
    assert isinstance(signals._synthetic_fallback, threading.Event)
    assert signals.__class__.__slots__ == (
        "_synthetic_fallback",
        "model_retry_callback",
        "usage_payload",
        "completed_usage_payloads",
        "_active_usage_payloads",
        "_usage_lock",
        "run_tag",
        "exchange_capture_enabled",
        "capture_detail",
        "completed_exchanges",
        "_active_exchanges",
        "_exchange_lock",
    )
    assert not hasattr(signals, "__dict__")
    assert signals.synthetic_fallback_emitted is False
    with pytest.raises(AttributeError):
        signals.synthetic_fallback_emitted = True
    assert signals.usage_payload is None
    assert signals.completed_usage_payloads == []
    assert signals.usage_payloads() == []

    # Content-free repr: usage payloads are provider-reported token counts,
    # not transcript text, but they are still per-request data that has no
    # business landing in a log line, so every field stays repr=False. The
    # exchange-capture fields follow the same rule -- `run_tag` (an opaque
    # uuid) and `exchange_capture_enabled` (a bool) are harmless and stay
    # visible, but `completed_exchanges`/`_active_exchanges` hold raw
    # request/response text and stay repr=False.
    rendered = repr(signals)
    assert rendered == (
        f"ConsoleProviderStreamSignals(run_tag={signals.run_tag!r}, "
        "exchange_capture_enabled=True)"
    )
    signals.record_usage_payload({"prompt_tokens": 4242})
    call = signals.new_usage_call()
    call.begin_exchange(
        provider="anthropic", model="m", endpoint=None,
        request={"messages_payload": [{"role": "user", "content": "SENSITIVE_EXCHANGE_TEXT"}]},
        omitted_keys=("api_key",),
    )
    call.record_exchange_content("SENSITIVE_EXCHANGE_TEXT")
    call.close_exchange()
    assert repr(signals) == (
        f"ConsoleProviderStreamSignals(run_tag={signals.run_tag!r}, "
        "exchange_capture_enabled=True)"
    )
    assert "SENSITIVE_EXCHANGE_TEXT" not in repr(signals)
    for governed_text in (
        NO_PROVIDER_CONTENT_COPY,
        UNSUPPORTED_PROVIDER_RESPONSE_COPY,
        "provider output body",
        "credential-secret",
        "raw exception detail",
        "retrieval evidence text",
        "INITIAL_BODY_SENTINEL_TASK_553_15",
        "REPAIRED_BODY_SENTINEL_TASK_553_15",
        "EVIDENCE_SENTINEL_TASK_553_15",
        "SOURCE_IDENTITY_SENTINEL_TASK_553_15",
        "LOCATOR_SENTINEL_TASK_553_15",
        "FULL_REPAIR_PROMPT_SENTINEL_TASK_553_15",
        "PROVIDER_EXCEPTION_SENTINEL_TASK_553_15",
    ):
        assert governed_text.lower() not in rendered.lower()


@pytest.mark.parametrize(
    ("response_factory", "expected"),
    [
        (lambda: {"content": ""}, NO_PROVIDER_CONTENT_COPY),
        (lambda: iter(()), NO_PROVIDER_CONTENT_COPY),
        (
            lambda: iter(({"unexpected": {"secret": "do not expose"}},)),
            UNSUPPORTED_PROVIDER_RESPONSE_COPY,
        ),
        (
            lambda: [{"content": "unsupported list body"}],
            UNSUPPORTED_PROVIDER_RESPONSE_COPY,
        ),
        (lambda: "data: [DONE]", NO_PROVIDER_CONTENT_COPY),
    ],
)
def test_synthetic_fallback_stream_signal_is_set_before_copy_is_observed(
    response_factory,
    expected,
) -> None:
    signals = gateway_module.ConsoleProviderStreamSignals()
    normalized = ConsoleProviderGateway.normalize_provider_response(
        response_factory(),
        signals=signals,
    )

    assert signals.synthetic_fallback_emitted is False
    assert next(normalized) == expected
    assert signals.synthetic_fallback_emitted is True


def test_synthetic_fallback_stream_signal_marks_only_when_iterable_reaches_junk() -> (
    None
):
    signals = gateway_module.ConsoleProviderStreamSignals()
    normalized = ConsoleProviderGateway.normalize_provider_response(
        iter(("real answer", {"unexpected": "junk"})),
        signals=signals,
    )

    assert next(normalized) == "real answer"
    assert signals.synthetic_fallback_emitted is False
    assert next(normalized) == UNSUPPORTED_PROVIDER_RESPONSE_COPY
    assert signals.synthetic_fallback_emitted is True


@pytest.mark.parametrize(
    "real_answer",
    [NO_PROVIDER_CONTENT_COPY, UNSUPPORTED_PROVIDER_RESPONSE_COPY],
)
def test_real_answer_equal_to_fallback_copy_does_not_mark_stream_signal(
    real_answer,
) -> None:
    signals = gateway_module.ConsoleProviderStreamSignals()

    chunks = list(
        ConsoleProviderGateway.normalize_provider_response(
            {"choices": [{"message": {"content": real_answer}}]},
            signals=signals,
        )
    )

    assert chunks == [real_answer]
    assert signals.synthetic_fallback_emitted is False


@pytest.mark.asyncio
async def test_stream_signal_is_set_before_async_synthetic_fallback_chunk() -> None:
    def fake_chat_api_call(**_kwargs):
        return {"choices": [{"message": {"content": ""}}]}

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"openai": {"api_key": "sk-test"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="openai", explicit_model="gpt-4.1")
    )
    signals = gateway_module.ConsoleProviderStreamSignals()
    stream = gateway.stream_chat(
        resolution,
        [{"role": "user", "content": "hi"}],
        signals=signals,
    )

    assert signals.synthetic_fallback_emitted is False
    assert await anext(stream) == NO_PROVIDER_CONTENT_COPY
    assert signals.synthetic_fallback_emitted is True
    await stream.aclose()


@pytest.mark.asyncio
async def test_stream_signal_omission_preserves_yielded_types_and_text() -> None:
    def fake_chat_api_call(**_kwargs):
        return iter(("hel", {"unexpected": "junk"}, "lo"))

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"openai": {"api_key": "sk-test"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="openai", explicit_model="gpt-4.1")
    )

    chunks = [
        chunk
        async for chunk in gateway.stream_chat(
            resolution,
            [{"role": "user", "content": "hi"}],
        )
    ]

    assert chunks == ["hel", UNSUPPORTED_PROVIDER_RESPONSE_COPY, "lo"]
    assert [type(chunk) for chunk in chunks] == [str, str, str]


@pytest.mark.asyncio
async def test_synthetic_fallback_suppression_leaves_stream_signal_unset() -> None:
    def fake_chat_api_call(**_kwargs):
        return {"choices": [{"message": {}}]}

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"groq": {"api_key": "sk-test"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider="groq", explicit_model="llama3-groq", streaming=False
        )
    )
    signals = gateway_module.ConsoleProviderStreamSignals()

    with pytest.raises(ChatProviderError):
        _ = [
            item
            async for item in gateway.stream_chat(
                resolution,
                [{"role": "user", "content": "hi"}],
                tools=TOOLS,
                signals=signals,
            )
        ]

    assert signals.synthetic_fallback_emitted is False


def test_normalize_generic_provider_response_dict_precedence() -> None:
    payload = {
        "choices": [
            {
                "delta": {"content": "delta"},
                "message": {"content": "message"},
                "text": "choice text",
            }
        ],
        "message": {"content": "top message"},
        "content": "content",
        "text": "text",
        "response": "response",
        "generated_text": "generated",
    }

    assert list(ConsoleProviderGateway.normalize_provider_response(payload)) == [
        "delta"
    ]
    assert list(
        ConsoleProviderGateway.normalize_provider_response(
            {
                "choices": [{"message": {"content": "message"}, "text": "choice text"}],
                "message": {"content": "top message"},
                "content": "content",
                "text": "text",
                "response": "response",
                "generated_text": "generated",
            }
        )
    ) == ["message"]
    assert list(
        ConsoleProviderGateway.normalize_provider_response(
            {
                "choices": [{"text": "choice text"}],
                "message": {"content": "top message"},
                "content": "content",
                "text": "text",
                "response": "response",
                "generated_text": "generated",
            }
        )
    ) == ["choice text"]
    assert list(
        ConsoleProviderGateway.normalize_provider_response(
            {
                "message": {"content": "top message"},
                "content": "content",
                "text": "text",
                "response": "response",
                "generated_text": "generated",
            }
        )
    ) == ["top message"]
    assert list(
        ConsoleProviderGateway.normalize_provider_response(
            {
                "content": "content",
                "text": "text",
                "response": "response",
                "generated_text": "generated",
            }
        )
    ) == ["content"]
    assert list(
        ConsoleProviderGateway.normalize_provider_response(
            {"text": "text", "response": "response", "generated_text": "generated"}
        )
    ) == ["text"]
    assert list(
        ConsoleProviderGateway.normalize_provider_response(
            {"response": "response", "generated_text": "generated"}
        )
    ) == ["response"]
    assert list(
        ConsoleProviderGateway.normalize_provider_response(
            {"generated_text": "generated"}
        )
    ) == ["generated"]


def test_normalize_google_gemini_candidates_response() -> None:
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "OK"}],
                    "role": "model",
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 5,
            "candidatesTokenCount": 1,
            "totalTokenCount": 6,
        },
    }

    assert list(ConsoleProviderGateway.normalize_provider_response(payload)) == ["OK"]


def test_safe_provider_error_copy_redacts_secret_like_values() -> None:
    copy = safe_provider_error_copy(
        "openai",
        RuntimeError(
            "Authorization: Bearer sk-1234567890abcdef "
            "https://user:secret@example.test/v1 password=hunter2 token=abc123"
        ),
    )

    assert "sk-1234567890abcdef" not in copy
    assert "Bearer" not in copy
    assert "user:secret@" not in copy
    assert "hunter2" not in copy
    assert "abc123" not in copy
    assert "openai" in copy


def test_safe_provider_error_copy_classifies_provider_exceptions() -> None:
    cases = [
        (ChatAuthenticationError(), "authentication failed"),
        (ChatRateLimitError(), "rate limit exceeded"),
        (ChatBadRequestError(), "bad request"),
        (ChatConfigurationError(), "configuration error"),
        (ChatProviderError(), "provider unavailable"),
        (RuntimeError("boom"), "unexpected provider error"),
    ]

    for exc, category in cases:
        copy = safe_provider_error_copy("openai", exc)
        assert f"Provider error from openai: {category}." in copy
        status_code = getattr(exc, "status_code", None)
        if status_code is not None:
            assert f"Status: {status_code}." in copy
        else:
            assert "Status:" not in copy


def test_safe_provider_error_copy_includes_status_code_when_available() -> None:
    copy = safe_provider_error_copy("openai", ChatProviderError(status_code=503))

    assert copy == "Provider error from openai: provider unavailable. Status: 503."


@pytest.mark.asyncio
async def test_stream_chat_generic_sse_string_chunks_yield_content_only() -> None:
    def fake_chat_api_call(**_kwargs):
        yield 'data: {"choices":[{"delta":{"content":"hel"}}]}'
        yield 'data: {"choices":[{"delta":{"content":"lo"}}]}'
        yield "data: [DONE]"

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"openai": {"api_key": "sk-test"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="openai", explicit_model="gpt-4.1")
    )

    chunks = [
        chunk
        async for chunk in gateway.stream_chat(
            resolution, [{"role": "user", "content": "hi"}]
        )
    ]

    assert chunks == ["hel", "lo"]


@pytest.mark.asyncio
async def test_stream_chat_generic_sse_byte_chunks_yield_content_only() -> None:
    def fake_chat_api_call(**_kwargs):
        yield b'data: {"choices":[{"delta":{"content":"bytes"}}]}'
        yield b"data: [DONE]"

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"openai": {"api_key": "sk-test"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="openai", explicit_model="gpt-4.1")
    )

    chunks = [
        chunk
        async for chunk in gateway.stream_chat(
            resolution, [{"role": "user", "content": "hi"}]
        )
    ]

    assert chunks == ["bytes"]


@pytest.mark.asyncio
async def test_stream_chat_generic_cancel_ignores_late_chunks() -> None:
    gate = threading.Event()

    def fake_chat_api_call(**_kwargs):
        yield "first"
        gate.wait(timeout=1)
        yield "late"

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"openai": {"api_key": "sk-test"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="openai", explicit_model="m")
    )
    stream = gateway.stream_chat(resolution, [{"role": "user", "content": "hi"}])

    assert await anext(stream) == "first"
    await stream.aclose()
    gate.set()


@pytest.mark.asyncio
async def test_stream_chat_generic_provider_error_raises_sanitized_exception() -> None:
    def fake_chat_api_call(**_kwargs):
        raise RuntimeError("Authorization: Bearer sk-1234567890abcdef")

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"openai": {"api_key": "sk-test"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="openai", explicit_model="gpt-4.1")
    )

    with pytest.raises(ChatProviderError) as exc_info:
        _ = [
            chunk
            async for chunk in gateway.stream_chat(
                resolution, [{"role": "user", "content": "hi"}]
            )
        ]

    message = str(exc_info.value)
    assert message == "Provider error from openai: unexpected provider error."
    assert "sk-1234567890abcdef" not in message
    assert "Bearer" not in message


@pytest.mark.asyncio
async def test_stream_bad_request_names_model_and_offers_picker_recovery() -> None:
    def fake_chat_api_call(**_kwargs):
        raise ChatBadRequestError(
            "retired model; Authorization: Bearer SECRET-CANARY",
            provider="anthropic",
        )

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {
            "api_settings": {"anthropic": {"api_key": "test-key"}}
        },
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider="anthropic",
            explicit_model="claude-3-haiku-20240307",
        )
    )

    with pytest.raises(ChatProviderError) as exc_info:
        _ = [
            chunk
            async for chunk in gateway.stream_chat(
                resolution,
                [{"role": "user", "content": "hi"}],
            )
        ]

    message = str(exc_info.value)
    assert "Provider error from anthropic" in message
    assert "claude-3-haiku-20240307" in message
    assert "Confirm the model is still available" in message
    assert "choose another model from the model picker" in message
    assert "Status: 400" in message
    assert "SECRET-CANARY" not in message


@pytest.mark.asyncio
async def test_stream_chat_generic_sse_error_raises_sanitized_exception() -> None:
    def fake_chat_api_call(**_kwargs):
        yield 'data: {"error":{"message":"Authorization: Bearer sk-1234567890abcdef"}}'

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"openai": {"api_key": "sk-test"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="openai", explicit_model="gpt-4.1")
    )

    with pytest.raises(ChatProviderError) as exc_info:
        _ = [
            chunk
            async for chunk in gateway.stream_chat(
                resolution, [{"role": "user", "content": "hi"}]
            )
        ]

    message = str(exc_info.value)
    assert message == "Provider error from openai: unexpected provider error."
    assert "sk-1234567890abcdef" not in message
    assert "Bearer" not in message


@pytest.mark.asyncio
async def test_stream_chat_generic_sse_byte_error_raises_sanitized_exception() -> None:
    def fake_chat_api_call(**_kwargs):
        yield b'data: {"error":{"message":"Authorization: Bearer sk-1234567890abcdef"}}'

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"openai": {"api_key": "sk-test"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="openai", explicit_model="gpt-4.1")
    )

    with pytest.raises(ChatProviderError) as exc_info:
        _ = [
            chunk
            async for chunk in gateway.stream_chat(
                resolution, [{"role": "user", "content": "hi"}]
            )
        ]

    message = str(exc_info.value)
    assert message == "Provider error from openai: unexpected provider error."
    assert "sk-1234567890abcdef" not in message
    assert "Bearer" not in message


@pytest.mark.asyncio
async def test_gateway_closes_owned_http_client():
    gateway = ConsoleProviderGateway()

    assert gateway.http_client.is_closed is False

    await gateway.aclose()

    assert gateway.http_client.is_closed is True


@pytest.mark.asyncio
async def test_gateway_does_not_close_injected_http_client():
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200))
    )
    gateway = ConsoleProviderGateway(http_client=client)

    await gateway.aclose()

    assert client.is_closed is False
    await client.aclose()


def test_aclose_does_not_let_a_later_loop_adopt_a_previously_claimed_client():
    """Review finding: `aclose()` clears `_loop_clients` and resets
    `_client_loop` to `None`, but never touched `self.http_client` itself,
    so the "unclaimed init client" escape hatch in `_active_http_client()`
    -- which used to key off `self._client_loop is None` -- could not tell
    "never claimed by any loop" apart from "was claimed and released by a
    loop that is now gone". Force the reliable version of the hazard: loop
    A claims a client via a plain ``asyncio.run()`` call, which closes loop
    A the instant it returns (mirrors a `console_agent_bridge` per-turn
    loop that finished on its own, with `aclose()` never having run on it).
    When `aclose()` later runs on an unrelated loop B, A's client can't even
    be scheduled for closing (its owning loop is already gone), so it is
    left with `is_closed is False`. A subsequent loop C must never adopt
    that leftover client -- doing so reuses a client whose httpx/httpcore
    connection-pool primitives are already bound to loop A, reintroducing
    the cross-loop binding failure this per-loop cache exists to eliminate.

    Uses three separate ``asyncio.run()`` calls (each spins up and tears
    down its own loop) rather than ``@pytest.mark.asyncio`` -- a loop
    cannot run another loop nested inside it, and the whole point here is
    three DISTINCT loops, the first already closed before the second ever
    starts.
    """
    gateway = ConsoleProviderGateway()

    async def claim() -> tuple[httpx.AsyncClient, asyncio.AbstractEventLoop]:
        return gateway._active_http_client(), asyncio.get_running_loop()

    client_a, loop_a = asyncio.run(claim())

    # Sanity: loop A really did claim the client, and `asyncio.run()` has
    # already closed loop A by the time control returns here.
    assert gateway._client_loop is loop_a
    assert gateway.http_client is client_a
    assert loop_a.is_closed()

    # `aclose()` now runs on loop B -- unrelated to loop A, and gone by the
    # time it returns too. It cannot schedule a close of A's client (A is
    # already closed), so A's client is left open; that alone is fine (its
    # own finalizer reclaims the sockets). What must NOT happen is a later
    # loop adopting it.
    asyncio.run(gateway.aclose())
    assert client_a.is_closed is False

    async def reclaim() -> httpx.AsyncClient:
        return gateway._active_http_client()

    new_client = asyncio.run(reclaim())

    assert new_client is not client_a, (
        "a client already bound to loop A's (now-closed) httpx/httpcore "
        "internals must never be adopted for a different loop"
    )

    asyncio.run(gateway.aclose())


# About the gateway's REAL owned client, so it opts out of the autouse
# offline-client guard (Tests/conftest.py, task-15111). Constructs a client;
# never connects.
@pytest.mark.owned_http_client
@pytest.mark.asyncio
async def test_owned_http_client_uses_generous_generation_read_timeout():
    """The owned client must not cap slow local generations at the old 30s."""
    gateway = ConsoleProviderGateway()
    try:
        timeout = gateway.http_client.timeout
        assert timeout.read == GENERATION_READ_TIMEOUT_SECONDS
        assert timeout.read >= 120
        assert timeout.write >= 120
        assert timeout.pool >= 120
        assert timeout.connect is not None and timeout.connect <= 30
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
async def test_llamacpp_probes_use_short_per_request_timeout():
    """Readiness probes stay snappy even though generation reads are long."""
    seen: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, dict(request.extensions.get("timeout", {}))))
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(200, json={"data": [{"id": "model-a"}]})

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            timeout=GENERATION_READ_TIMEOUT_SECONDS,
        )
    )

    explicit = await gateway.resolve_llamacpp(
        LlamaCppProviderConfig(explicit_model="m")
    )
    discovered = await gateway.resolve_llamacpp(LlamaCppProviderConfig())

    assert explicit.ready is True
    assert discovered.model == "model-a"
    assert [path for path, _ in seen] == ["/health", "/v1/models"]
    for path, timeout in seen:
        assert timeout.get("connect") == PROBE_TIMEOUT_SECONDS, path
        assert timeout.get("read") == PROBE_TIMEOUT_SECONDS, path


@pytest.mark.asyncio
async def test_llamacpp_generation_calls_keep_client_level_timeout():
    """Generation requests inherit the client timeout, not the probe override."""
    client_timeout = 222.0
    seen: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, dict(request.extensions.get("timeout", {}))))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "slow answer"}}]},
        )

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            timeout=client_timeout,
        )
    )

    completion = await gateway.complete_llamacpp_chat(
        base_url="http://127.0.0.1:9099",
        model="m",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert completion == "slow answer"
    assert [path for path, _ in seen] == ["/v1/chat/completions"]
    assert seen[0][1].get("read") == client_timeout
    assert seen[0][1].get("read") != PROBE_TIMEOUT_SECONDS


class _JSONOKHandler(http.server.BaseHTTPRequestHandler):
    """Minimal local HTTP server: real sockets, real httpcore connection pool.

    A ``httpx.MockTransport`` does not reproduce the loop-binding bug below
    (it never touches httpcore's real ``AsyncConnectionPool``, so no
    loop-bound lock/event is ever created) -- only genuine socket traffic
    does, which is why this fixture spins up a real (if tiny) HTTP server
    instead. ``protocol_version`` must be HTTP/1.1 (``BaseHTTPRequestHandler``
    defaults to 1.0, which closes the connection after every response and
    happens to sidestep the pool-level lock reuse this test targets) -- real
    llama.cpp servers speak keep-alive HTTP/1.1, which is what actually
    reproduced the live crash this regression test is pinned to.
    """

    protocol_version = "HTTP/1.1"

    def do_GET(self):  # noqa: N802 -- BaseHTTPRequestHandler naming
        body = b'{"data": [{"id": "model-a"}]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # noqa: D102 -- silence default stderr logging
        pass


class _DeepBacklogHTTPServer(http.server.ThreadingHTTPServer):
    # The stdlib default listen backlog is 5; the concurrent-swap test
    # barrier-releases 6 threads into fresh cold connections every round,
    # so refused/queued connects under load showed up as probe timeouts
    # (de-flake pass 2026-07-17).
    request_queue_size = 32


_LOOPBACK_LISTENER_PERMISSION_SKIP_REASON = (
    "loopback listener unavailable: permission denied"
)


@pytest.fixture
def local_http_server():
    try:
        server = _DeepBacklogHTTPServer(("127.0.0.1", 0), _JSONOKHandler)
    except PermissionError:
        pytest.skip(_LOOPBACK_LISTENER_PERMISSION_SKIP_REASON)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_local_http_server_permission_denied_skips_with_capability_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Classify listener permission denial as an explicit capability skip.

    Args:
        monkeypatch: Pytest fixture used to replace listener construction.
    """

    def deny_listener(*_args, **_kwargs):
        raise PermissionError("sandbox denied loopback bind")

    monkeypatch.setitem(globals(), "_DeepBacklogHTTPServer", deny_listener)

    with pytest.raises(pytest.skip.Exception) as exc_info:
        next(local_http_server.__wrapped__())

    assert str(exc_info.value) == _LOOPBACK_LISTENER_PERMISSION_SKIP_REASON


def test_local_http_server_non_permission_oserror_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep non-permission listener failures actionable instead of skipping.

    Args:
        monkeypatch: Pytest fixture used to replace listener construction.
    """

    def fail_listener(*_args, **_kwargs):
        raise OSError("address resources exhausted")

    def fail_if_skipped(reason: str) -> None:
        pytest.fail(f"unexpected capability skip: {reason}")

    monkeypatch.setitem(globals(), "_DeepBacklogHTTPServer", fail_listener)
    monkeypatch.setattr(pytest, "skip", fail_if_skipped)

    with pytest.raises(OSError, match="address resources exhausted"):
        next(local_http_server.__wrapped__())


# Real owned client AND a real socket: the whole point is httpx's per-loop
# connection-pool binding against a server this test starts itself on numeric
# loopback only. The fixture skips explicitly when the host denies listener
# construction (Tests/conftest.py, task-15111).
@pytest.mark.owned_http_client
@pytest.mark.loopback_network
def test_owned_http_client_survives_agent_bridge_style_loop_swap(local_http_server):
    """Regression (Task 8 live gate): every agent turn crashed against a real
    llama.cpp server with ``RuntimeError: <asyncio.locks.Event ...> is bound
    to a different event loop``. Root cause: the gateway's OWNED httpx
    client was reused verbatim across the app's main event loop (readiness
    probes, awaited in-place) and the agent bridge's per-turn
    ``asyncio.run()`` worker-thread loop (``console_agent_bridge.
    _StreamingModelAdapter.chat_call``) -- httpx/httpcore bind their
    internal connection-pool lock/event objects to whichever loop first
    touches them, so a second, concurrently-running loop reusing the same
    client always raised. This drives the exact same two-loop shape: a
    background thread keeps a loop alive indefinitely (like the Textual app
    loop) while a fresh ``asyncio.run()`` (like the agent bridge) reuses the
    same gateway afterward.
    """
    gateway = ConsoleProviderGateway()

    async def probe() -> bool:
        return await gateway._is_reachable(local_http_server)

    main_loop = asyncio.new_event_loop()
    main_loop_ready = threading.Event()

    def run_main_loop() -> None:
        asyncio.set_event_loop(main_loop)
        main_loop_ready.set()
        main_loop.run_forever()

    main_thread = threading.Thread(target=run_main_loop, daemon=True)
    main_thread.start()
    main_loop_ready.wait(timeout=2)
    try:
        # First use: a readiness probe awaited on the (still-running) main
        # loop -- binds the owned client's internal locks to `main_loop`.
        first = asyncio.run_coroutine_threadsafe(probe(), main_loop).result(timeout=5)
        assert first is True

        # Second use: the agent bridge's worker thread bridges via a BRAND
        # NEW asyncio.run() loop while `main_loop` is still alive elsewhere.
        # Before the fix this raised RuntimeError("... is bound to a
        # different event loop") on every single agent turn.
        second = asyncio.run(probe())
        assert second is True
    finally:
        main_loop.call_soon_threadsafe(main_loop.stop)
        main_thread.join(timeout=2)


def test_injected_http_client_is_never_swapped_across_loops():
    """Injected clients (test doubles / callers that own their own client)
    must never be silently replaced -- only the gateway's OWNED client is
    loop-swapped."""
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200))
    )
    gateway = ConsoleProviderGateway(http_client=client)

    async def active_client_identity() -> int:
        return id(gateway._active_http_client())

    first = asyncio.run(active_client_identity())
    second = asyncio.run(active_client_identity())

    assert first == second == id(client)
    asyncio.run(client.aclose())


def test_active_http_client_first_touch_adopts_the_unclaimed_init_client(monkeypatch):
    """TASK-1064 item 1: the client built in ``__init__`` is not yet bound to
    (has not made a request on) any event loop, so the FIRST loop to call
    ``_active_http_client()`` adopts it directly into the per-loop cache
    instead of discarding it and building a fresh one -- construction should
    not waste a connection, and there is nothing to close since no loop has
    touched it yet. Directly supersedes the old single-slot design's
    behavior (PR #629 Fix 1(b)), where the first touch unconditionally
    swapped in a brand-new client and scheduled a close of the original."""
    gateway = ConsoleProviderGateway()
    original_client = gateway.http_client
    scheduled: list[tuple[int, object]] = []

    def fake_schedule(client, loop):
        scheduled.append((id(client), loop))

    monkeypatch.setattr(
        ConsoleProviderGateway,
        "_schedule_stale_client_close",
        staticmethod(fake_schedule),
    )

    async def touch() -> httpx.AsyncClient:
        return gateway._active_http_client()

    adopted = asyncio.run(touch())

    assert adopted is original_client, (
        "the first loop to touch the gateway must adopt the unclaimed "
        "init-time client rather than discarding it for a fresh one"
    )
    assert scheduled == [], (
        "adopting an unclaimed client must not schedule a close of it -- "
        "nothing was ever using it"
    )


def test_active_http_client_creation_is_mutually_exclusive_across_threads():
    """PR #629 Fix 1(a) (Gemini HIGH x2 + Qodo-8), preserved across the move
    to a per-loop cache (TASK-1064 item 1): building a NEW per-loop cache
    entry must be a single atomic critical section guarded by one lock, not
    two independently-racy reads/writes -- otherwise two concurrent callers
    on different not-yet-cached loops could each decide the cache is empty
    and both build+insert, or interleave with the cache's other bookkeeping.
    Proven deterministically here (no reliance on GIL scheduling luck):
    thread A is parked *inside* client construction via a monkeypatched,
    blocking ``_new_owned_http_client``, and thread B's concurrent call for
    a DIFFERENT loop must provably fail to complete while A is still in
    flight -- only completing once A releases and the lock is free.

    The gateway is primed with one throwaway touch first so that both
    threads' loops are genuinely new to the cache and both take the
    ``_new_owned_http_client`` creation path -- the very first touch ever is
    now an adopt-in-place of the unclaimed ``__init__`` client (see
    ``test_active_http_client_first_touch_adopts_the_unclaimed_init_client``)
    and would not exercise this path.
    """
    gateway = ConsoleProviderGateway()
    priming_loop = asyncio.new_event_loop()
    try:

        async def prime() -> None:
            gateway._active_http_client()

        priming_loop.run_until_complete(prime())
    finally:
        priming_loop.close()

    original_new_client = ConsoleProviderGateway._new_owned_http_client
    entered = threading.Event()
    release = threading.Event()

    def blocking_new_client():
        # Only the FIRST call (thread A's) blocks -- a concurrent second
        # call (thread B's) that is *not* actually serialized by a lock
        # would sail straight through this on its own turn and finish its
        # creation well before thread A ever releases, which is exactly the
        # unlocked-race behavior this test must catch.
        if not entered.is_set():
            entered.set()
            release.wait(timeout=5)
        return original_new_client()

    ConsoleProviderGateway._new_owned_http_client = staticmethod(blocking_new_client)
    loop_a = asyncio.new_event_loop()
    loop_b = asyncio.new_event_loop()
    thread_a: threading.Thread | None = None
    thread_b: threading.Thread | None = None
    second_done = threading.Event()
    try:

        def call_a() -> None:
            async def go() -> None:
                gateway._active_http_client()

            loop_a.run_until_complete(go())

        thread_a = threading.Thread(target=call_a)
        thread_a.start()
        assert entered.wait(timeout=5), "thread A must have entered creation"

        def call_b() -> None:
            async def go() -> None:
                gateway._active_http_client()

            loop_b.run_until_complete(go())
            second_done.set()

        thread_b = threading.Thread(target=call_b)
        thread_b.start()

        # Thread A is still parked inside its creation -- give thread B
        # ample opportunity to race ahead if creation were not actually
        # serialized by a lock.
        premature = second_done.wait(timeout=0.5)
        assert premature is False, (
            "a concurrent cache-entry creation completed while another "
            "thread's creation was still in flight -- the critical section "
            "is not atomic"
        )
        release.set()
        assert second_done.wait(timeout=5)
    finally:
        # Always unblock thread A and drain both threads before touching
        # the loops, whether or not the assertions above passed -- an
        # early failure must not leave a loop "running" (from the other
        # thread's still-in-flight run_until_complete) when we try to
        # close it.
        release.set()
        if thread_a is not None:
            thread_a.join(timeout=5)
        if thread_b is not None:
            thread_b.join(timeout=5)
        # Re-wrap in `staticmethod(...)`: plain-function reassignment onto
        # the class would otherwise bind `self` as an implicit first
        # argument on the next instance access, breaking every other test
        # in this module that constructs a gateway afterward.
        ConsoleProviderGateway._new_owned_http_client = staticmethod(
            original_new_client
        )
        loop_a.close()
        loop_b.close()


def test_aclose_closes_current_loop_client_and_schedules_others(monkeypatch):
    """The per-loop cache can hold multiple live clients at once (one per
    loop that has touched the gateway); ``aclose()`` must close the calling
    loop's own client directly and hand off every OTHER cached loop's client
    to ``_schedule_stale_client_close`` -- never await, and never directly
    close, a client bound to a loop it is not currently running on. This is
    the non-leaking-close guarantee (PR #629 Fix 1(b)) restated for a cache
    that can hold more than one entry."""
    gateway = ConsoleProviderGateway()
    other_loop = asyncio.new_event_loop()
    try:

        async def touch() -> None:
            gateway._active_http_client()

        other_loop.run_until_complete(touch())
        other_client = gateway._loop_clients[other_loop]
        assert other_client.is_closed is False

        scheduled: list[tuple[int, object]] = []

        def fake_schedule(client, loop):
            scheduled.append((id(client), loop))

        monkeypatch.setattr(
            ConsoleProviderGateway,
            "_schedule_stale_client_close",
            staticmethod(fake_schedule),
        )

        async def close_from_new_loop() -> None:
            await gateway.aclose()

        asyncio.run(close_from_new_loop())

        assert scheduled == [(id(other_client), other_loop)], (
            "aclose() must hand off every other cached loop's client to "
            "_schedule_stale_client_close, keyed to its own loop"
        )
        # `fake_schedule` never actually closed it -- confirms aclose() did
        # not itself await/close a client bound to a different loop.
        assert other_client.is_closed is False
    finally:
        other_loop.run_until_complete(other_client.aclose())
        other_loop.close()


# Same as above: real owned client + this test's own `local_http_server`.
@pytest.mark.owned_http_client
@pytest.mark.loopback_network
def test_active_http_client_concurrent_swap_never_leaves_client_bound_to_wrong_loop(
    local_http_server,
    monkeypatch,
):
    """PR #629 Fix 1(a) (Gemini HIGH x2 + Qodo-8): the check-and-swap of
    ``http_client``/``_client_loop`` was not atomic, so concurrent callers
    from different threads/loops (e.g. the app loop's readiness probe
    racing the agent worker thread's per-turn loop) could interleave the
    read-then-write and leave the client bound to one loop while
    ``_client_loop`` records a different one -- the next probe on the
    recorded loop then reuses a client bound elsewhere and crashes with
    "bound to a different event loop". This hammers many persistent loops
    (each in its own OS thread, lined up on a barrier every round so they
    all race the swap concurrently) against the gateway's single owned
    client and asserts every single real request against a local server
    succeeds -- a mismatch manifests as a genuine RuntimeError out of
    httpx/httpcore, not just a stale internal-state assertion.

    De-flake (2026-07-17, ~1% repro): the observed failures were probe
    TIMEOUTS (`_is_reachable` swallowing an ``httpx`` timeout under the
    test's own 6-thread cold-connection stampede), NOT the loop-binding
    RuntimeError this guards against. The production probe timeout (5s) is
    tuned for one readiness probe, not 120 barrier-synchronized cold
    connects -- widen it for this test only so a genuine loop-binding
    regression (which fails instantly) stays the only failure mode.
    """
    import tldw_chatbook.Chat.console_provider_gateway as gateway_module

    monkeypatch.setattr(gateway_module, "PROBE_TIMEOUT_SECONDS", 20.0)
    gateway = ConsoleProviderGateway()
    thread_count = 6
    rounds = 20
    barrier = threading.Barrier(thread_count)
    loops: list[asyncio.AbstractEventLoop] = []
    ready_events: list[threading.Event] = []
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def run_loop_thread(
        loop: asyncio.AbstractEventLoop, ready: threading.Event
    ) -> None:
        asyncio.set_event_loop(loop)
        ready.set()
        loop.run_forever()

    loop_threads = []
    for _ in range(thread_count):
        loop = asyncio.new_event_loop()
        ready = threading.Event()
        loops.append(loop)
        ready_events.append(ready)
        thread = threading.Thread(
            target=run_loop_thread, args=(loop, ready), daemon=True
        )
        loop_threads.append(thread)
        thread.start()
    for ready in ready_events:
        assert ready.wait(timeout=2)

    def hammer(loop: asyncio.AbstractEventLoop) -> None:
        for _ in range(rounds):
            try:
                # Kept >= the (test-widened) probe timeout: a slower-than-
                # barrier probe would break the barrier for every thread.
                barrier.wait(timeout=30)
            except threading.BrokenBarrierError as exc:
                # A broken barrier means the concurrency scenario stopped
                # executing as designed — record it as a FAILURE instead of
                # returning quietly with fewer rounds (Qodo #680-3).
                with errors_lock:
                    errors.append(exc)
                return
            try:
                future = asyncio.run_coroutine_threadsafe(
                    gateway._is_reachable(local_http_server), loop
                )
                assert future.result(timeout=30) is True
            except BaseException as exc:  # noqa: BLE001 -- collected, asserted below
                with errors_lock:
                    errors.append(exc)

    workers = [
        threading.Thread(target=hammer, args=(loop,), daemon=True) for loop in loops
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=60)

    for loop in loops:
        loop.call_soon_threadsafe(loop.stop)
    for thread in loop_threads:
        thread.join(timeout=2)

    assert errors == []


def test_concurrent_live_loops_never_close_each_others_client():
    """TASK-1064 item 1 -- genuine concurrency, NOT sequential ``asyncio.run()``
    calls. A sequential two-loop probe does not discriminate: the first loop
    is already closed by the time the second one runs, so both the fixed and
    the unfixed single-slot code pass it. Two real OS threads each run their
    own ``asyncio.run()`` loop and are barrier-synchronized so both loops are
    genuinely alive at the same wall-clock moment while each resolves
    ``_active_http_client()`` on the SAME shared gateway instance -- exactly
    the overlap the gateway's own docstrings describe: a readiness probe
    awaited on the app's own event loop racing an agent-runtime generation
    call bridged from a worker thread's fresh ``asyncio.run()``. Under the
    old single-slot ``http_client``/``_client_loop`` cache, the second
    thread's touch treats the first thread's still-in-flight client as
    "stale" and schedules ``aclose()`` of it on the first thread's own
    (still-running) loop -- which actually executes it, closing a client the
    first thread is still using.
    """
    gateway = ConsoleProviderGateway()
    barrier = threading.Barrier(2)
    obtained: dict[str, httpx.AsyncClient] = {}
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def run(name: str) -> None:
        async def go() -> None:
            client = gateway._active_http_client()
            obtained[name] = client
            barrier.wait(timeout=5)
            # Keep this loop alive and pumping so a cross-loop `aclose()`
            # scheduled onto it via `run_coroutine_threadsafe` (the bug)
            # actually gets a chance to execute, exactly like a real
            # in-flight request would still be holding the client open.
            for _ in range(50):
                await asyncio.sleep(0.01)
                if client.is_closed:
                    break

        try:
            asyncio.run(go())
        except BaseException as exc:  # noqa: BLE001 -- collected, asserted below
            with errors_lock:
                errors.append(exc)

    thread_a = threading.Thread(target=run, args=("a",))
    thread_b = threading.Thread(target=run, args=("b",))
    thread_a.start()
    thread_b.start()
    thread_a.join(timeout=10)
    thread_b.join(timeout=10)

    assert errors == [], f"unexpected errors from worker threads: {errors!r}"
    assert "a" in obtained and "b" in obtained, "both loops must obtain a client"
    assert obtained["a"] is not obtained["b"], (
        "two live loops must never share the same owned http client"
    )
    assert obtained["a"].is_closed is False, (
        "loop A's client was closed while loop B was concurrently alive and "
        "touching the shared gateway -- a live loop must never close "
        "another live loop's client"
    )
    assert obtained["b"].is_closed is False, (
        "loop B's client was closed while loop A was concurrently alive and "
        "touching the shared gateway -- a live loop must never close "
        "another live loop's client"
    )


def _sse(payload):
    return "data: " + json.dumps(payload)


def _delta_fragment(index, call_id=None, name=None, arguments=None):
    frag = {"index": index, "function": {}}
    if call_id is not None:
        frag["id"] = call_id
        frag["type"] = "function"
    if name is not None:
        frag["function"]["name"] = name
    if arguments is not None:
        frag["function"]["arguments"] = arguments
    return {"choices": [{"delta": {"tool_calls": [frag]}}]}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "d",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


async def _collect(gateway, resolution, tools=None):
    items = []
    async for chunk in gateway.stream_chat(
        resolution, [{"role": "user", "content": "q"}], tools=tools
    ):
        items.append(chunk)
    return items


@pytest.mark.asyncio
async def test_stream_accumulates_sse_tool_call_fragments() -> None:
    """OpenAI streaming: id/name on the first fragment, arguments split
    across fragments -> ONE merged ProviderToolCalls yielded last."""
    script = iter(
        [
            _sse(_delta_fragment(0, call_id="c9", name="calculator")),
            _sse(_delta_fragment(0, arguments='{"expres')),
            _sse(_delta_fragment(0, arguments='sion": "2+2"}')),
            "data: [DONE]",
        ]
    )

    def fake_chat_api_call(**_kwargs):
        return script

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"groq": {"api_key": "sk-test"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider="groq", explicit_model="llama3-groq", streaming=True
        )
    )

    items = await _collect(gateway, resolution, tools=TOOLS)

    calls = [i for i in items if isinstance(i, ProviderToolCalls)]
    assert len(calls) == 1 and items[-1] is calls[0]
    (call,) = calls[0].tool_calls
    assert call == {
        "id": "c9",
        "type": "function",
        "function": {"name": "calculator", "arguments": '{"expression": "2+2"}'},
    }
    assert not any(
        isinstance(i, str) and i.strip() for i in items[:-1]
    )  # no copy leaked


@pytest.mark.asyncio
async def test_non_streaming_message_tool_calls_surface() -> None:
    """resolution.streaming False: chat_api_call returns the full dict;
    message.tool_calls surfaces as ProviderToolCalls, content as text."""
    response = {
        "choices": [
            {
                "message": {
                    "content": "Checking.",
                    "tool_calls": [
                        {
                            "id": "n1",
                            "type": "function",
                            "function": {"name": "calculator", "arguments": "{}"},
                        }
                    ],
                }
            }
        ]
    }

    def fake_chat_api_call(**_kwargs):
        return response

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"groq": {"api_key": "sk-test"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider="groq", explicit_model="llama3-groq", streaming=False
        )
    )

    items = await _collect(gateway, resolution, tools=TOOLS)

    assert "Checking." in [i for i in items if isinstance(i, str)]
    (ptc,) = [i for i in items if isinstance(i, ProviderToolCalls)]
    assert ptc.tool_calls[0]["id"] == "n1"


@pytest.mark.asyncio
async def test_no_tools_requested_is_byte_identical() -> None:
    """Same fragment script WITHOUT tools=: no ProviderToolCalls, no new
    strings -- the delta-only chunks stay silently dropped as today."""
    script = iter(
        [
            _sse(_delta_fragment(0, call_id="c9", name="calculator")),
            _sse(_delta_fragment(0, arguments='{"expres')),
            _sse(_delta_fragment(0, arguments='sion": "2+2"}')),
            "data: [DONE]",
        ]
    )

    def fake_chat_api_call(**_kwargs):
        return script

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"groq": {"api_key": "sk-test"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider="groq", explicit_model="llama3-groq", streaming=True
        )
    )

    items = await _collect(gateway, resolution, tools=None)

    assert all(isinstance(i, str) for i in items)
    assert UNSUPPORTED_PROVIDER_RESPONSE_COPY not in items


@pytest.mark.asyncio
async def test_tools_none_raw_dict_tool_call_chunk_keeps_baseline_copy() -> None:
    """Regression (task-243 review): a raw DICT streaming chunk (not an SSE
    string, unlike ``test_no_tools_requested_is_byte_identical`` above) that
    carries only ``delta.tool_calls`` with no content, and with ``tools``
    NOT passed, must stay byte-identical to the pre-native-tools baseline --
    ``_content_from_provider_mapping`` has no tool-call awareness in that
    codepath, so the chunk falls through to ``_UNSUPPORTED_RESPONSE`` like
    any other unrecognized dict shape, and ``normalize_provider_response``
    surfaces it as ``UNSUPPORTED_PROVIDER_RESPONSE_COPY`` in the stream.
    Mapping-level ``tool_calls`` guards previously short-circuited this to a
    silent drop instead, which changed ``tools=None`` output."""

    def fake_chat_api_call(**_kwargs):
        yield "hel"
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "calculator", "arguments": "{}"},
                            }
                        ]
                    }
                }
            ]
        }
        yield {"choices": [{"delta": {"content": "lo"}}]}

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"openai": {"api_key": "sk-test"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="openai", explicit_model="gpt-4.1")
    )

    chunks = [
        chunk
        async for chunk in gateway.stream_chat(
            resolution, [{"role": "user", "content": "hi"}]
        )
    ]

    assert chunks == ["hel", UNSUPPORTED_PROVIDER_RESPONSE_COPY, "lo"]


@pytest.mark.asyncio
async def test_tool_call_only_stream_yields_no_fallback_copy() -> None:
    """A tools= run whose stream carries ONLY tool-call fragments must not
    inject NO_PROVIDER_CONTENT_COPY / UNSUPPORTED copy into the text
    stream (that copy would be echoed into agent history)."""
    script = iter(
        [
            _sse(_delta_fragment(0, call_id="c9", name="calculator")),
            _sse(_delta_fragment(0, arguments='{"expres')),
            _sse(_delta_fragment(0, arguments='sion": "2+2"}')),
            "data: [DONE]",
        ]
    )

    def fake_chat_api_call(**_kwargs):
        return script

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"groq": {"api_key": "sk-test"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider="groq", explicit_model="llama3-groq", streaming=True
        )
    )

    items = await _collect(gateway, resolution, tools=TOOLS)

    texts = [i for i in items if isinstance(i, str)]
    assert NO_PROVIDER_CONTENT_COPY not in texts
    assert UNSUPPORTED_PROVIDER_RESPONSE_COPY not in texts


@pytest.mark.asyncio
async def test_tools_run_with_neither_content_nor_calls_raises_instead_of_silent_empty() -> (
    None
):
    """PR #648 review Minor 1: a tools= turn whose provider response carries
    NEITHER visible content NOR tool-calls must surface as a provider error,
    not complete as a silent empty turn. On the fence path the same junk
    response surfaces diagnostic copy as the answer; in tools mode that copy
    is filtered from agent history, so without this guard a misbehaving
    provider's junk 200-body becomes an indistinguishable empty RUN_DONE."""
    response = {"choices": [{"message": {}}]}  # junk: no content, no tool_calls

    def fake_chat_api_call(**_kwargs):
        return response

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"groq": {"api_key": "sk-test"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider="groq", explicit_model="llama3-groq", streaming=False
        )
    )

    with pytest.raises(ChatProviderError):
        await _collect(gateway, resolution, tools=TOOLS)


@pytest.mark.asyncio
async def test_tools_run_with_real_content_and_no_calls_stays_a_normal_answer() -> None:
    """Guard scope check: a tools= turn that answers with plain text (no tool
    calls) is a perfectly normal final answer and must NOT raise."""
    response = {"choices": [{"message": {"content": "Just an answer."}}]}

    def fake_chat_api_call(**_kwargs):
        return response

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"groq": {"api_key": "sk-test"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider="groq", explicit_model="llama3-groq", streaming=False
        )
    )

    items = await _collect(gateway, resolution, tools=TOOLS)

    assert items == ["Just an answer."]


@pytest.mark.asyncio
async def test_tool_call_fragments_out_of_index_order_emit_in_index_order() -> None:
    """PR #648 review: the provider's index field defines batch order; when
    index-1 fragments arrive before index-0, the merged ProviderToolCalls
    must still be ordered [0, 1]."""
    script = iter(
        [
            _sse(_delta_fragment(1, call_id="c1", name="get_current_datetime")),
            _sse(_delta_fragment(1, arguments="{}")),
            _sse(_delta_fragment(0, call_id="c0", name="calculator")),
            _sse(_delta_fragment(0, arguments='{"expression": "2+2"}')),
            "data: [DONE]",
        ]
    )

    def fake_chat_api_call(**_kwargs):
        return script

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"groq": {"api_key": "sk-test"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider="groq", explicit_model="llama3-groq", streaming=True
        )
    )

    items = await _collect(gateway, resolution, tools=TOOLS)

    (ptc,) = [i for i in items if isinstance(i, ProviderToolCalls)]
    assert [c["id"] for c in ptc.tool_calls] == ["c0", "c1"]
    assert [c["function"]["name"] for c in ptc.tool_calls] == [
        "calculator",
        "get_current_datetime",
    ]


def test_tool_call_accumulator_preserves_extra_fragment_keys() -> None:
    """task-266: provider-specific extra keys on tool-call fragments (e.g.
    Gemini 3 google_thought_signature) must survive the merge — the request
    converter has to echo them back verbatim."""
    from tldw_chatbook.Chat.console_provider_gateway import _ToolCallAccumulator

    acc = _ToolCallAccumulator()
    acc.feed_payload(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "calculator", "arguments": "{}"},
                                "google_thought_signature": "sig-x",
                            }
                        ]
                    }
                }
            ]
        }
    )
    (call,) = acc.calls()
    assert call["google_thought_signature"] == "sig-x"
    assert call["function"]["name"] == "calculator"
    # PR #662 review: falsy-but-present ALLOW-LISTED extras survive verbatim
    # (None drops); unknown extra keys are NOT forwarded (PR #662 final
    # review: open-ended passthrough let any provider inject echoed keys).
    acc.feed_payload(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "google_thought_signature": "",
                                "arbitrary_extra": "nope",
                                "none_extra": None,
                            }
                        ]
                    }
                }
            ]
        }
    )
    (call,) = acc.calls()
    assert call["google_thought_signature"] == ""
    assert "arbitrary_extra" not in call
    assert "none_extra" not in call


class _CloseTrackingIterator:
    def __init__(
        self,
        items: list[object],
        *,
        failure: BaseException | None = None,
        close_failure: BaseException | None = None,
    ) -> None:
        self._items = iter(items)
        self._failure = failure
        self._close_failure = close_failure
        self.close_calls = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._items)
        except StopIteration:
            if self._failure is not None:
                failure, self._failure = self._failure, None
                raise failure
            raise

    def close(self) -> None:
        self.close_calls += 1
        if self._close_failure is not None:
            raise self._close_failure


class _RecordingAccumulator:
    def __init__(self, failure: BaseException | None = None) -> None:
        self.failure = failure
        self.payloads: list[object] = []

    def feed_payload(self, payload: object) -> None:
        if self.failure is not None:
            raise self.failure
        self.payloads.append(payload)


class _BlockingLateIterator:
    def __init__(self, item: object) -> None:
        self.item = item
        self.next_entered = threading.Event()
        self.release_next = threading.Event()
        self.next_calls = 0
        self.close_calls = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.next_calls += 1
        self.next_entered.set()
        self.release_next.wait(timeout=5)
        return self.item

    def close(self) -> None:
        self.close_calls += 1
        self.release_next.set()


class _CloseableMapping(dict):
    def __init__(self, *args, close_failure: BaseException | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.close_calls = 0
        self.close_failure = close_failure

    def close(self) -> None:
        self.close_calls += 1
        if self.close_failure is not None:
            raise self.close_failure


class _BlockingAccumulator:
    def __init__(self) -> None:
        self.feed_entered = threading.Event()
        self.release_feed = threading.Event()
        self.payloads: list[object] = []

    def feed_payload(self, payload: object) -> None:
        self.feed_entered.set()
        self.release_feed.wait(timeout=5)
        self.payloads.append(payload)


def test_tee_tool_calls_explicit_close_stops_future_iteration() -> None:
    underlying = _CloseTrackingIterator([{"content": "late"}])
    accumulator = _RecordingAccumulator()
    tee = gateway_module._tee_tool_calls(underlying, accumulator)

    tee.close()

    with pytest.raises(StopIteration):
        next(tee)
    assert underlying.close_calls == 1
    assert accumulator.payloads == []


def test_tee_tool_calls_discards_item_returned_after_concurrent_close() -> None:
    payload = {"choices": [{"delta": {"content": "late"}}]}
    underlying = _BlockingLateIterator(payload)
    accumulator = _RecordingAccumulator()
    tee = gateway_module._tee_tool_calls(underlying, accumulator)
    outcomes: list[object] = []

    def consume() -> None:
        try:
            outcomes.append(next(tee))
        except BaseException as exc:  # noqa: BLE001 - asserted below
            outcomes.append(exc)

    consumer = threading.Thread(target=consume)
    consumer.start()
    assert underlying.next_entered.wait(timeout=1)

    tee.close()
    consumer.join(timeout=1)

    assert not consumer.is_alive()
    assert len(outcomes) == 1 and isinstance(outcomes[0], StopIteration)
    assert underlying.close_calls == 1
    assert accumulator.payloads == []


def test_tee_tool_calls_close_is_prompt_while_accumulator_feed_is_blocked() -> None:
    payload = {"choices": [{"delta": {"content": "late"}}]}
    underlying = _CloseTrackingIterator([payload])
    accumulator = _BlockingAccumulator()
    tee = gateway_module._tee_tool_calls(underlying, accumulator)
    outcomes: list[object] = []
    close_finished = threading.Event()

    def consume() -> None:
        try:
            outcomes.append(next(tee))
        except BaseException as exc:  # noqa: BLE001 - asserted below
            outcomes.append(exc)

    def close() -> None:
        tee.close()
        close_finished.set()

    consumer = threading.Thread(target=consume)
    closer = threading.Thread(target=close)
    consumer.start()
    assert accumulator.feed_entered.wait(timeout=1)
    closer.start()
    try:
        assert close_finished.wait(timeout=0.5)
        assert underlying.close_calls == 1
    finally:
        accumulator.release_feed.set()
        consumer.join(timeout=1)
        closer.join(timeout=1)

    assert not consumer.is_alive()
    assert not closer.is_alive()
    assert len(outcomes) == 1 and isinstance(outcomes[0], StopIteration)
    # The feed was already running when close sealed the tee, so it may finish
    # its own work; the sealed item must still never be returned.
    assert accumulator.payloads == [payload]
    tee.close()
    assert underlying.close_calls == 1


@pytest.mark.parametrize("close_stage", ["decode", "feed"])
def test_tee_tool_calls_reentrant_close_does_not_deadlock(
    close_stage,
    monkeypatch,
) -> None:
    payload = {"choices": [{"delta": {"content": "late"}}]}
    underlying = _CloseTrackingIterator([payload])
    accumulator = _RecordingAccumulator()
    tee_holder: dict[str, object] = {}

    if close_stage == "decode":

        def close_during_decode(item):
            tee_holder["tee"].close()
            return item

        monkeypatch.setattr(gateway_module, "_decode_stream_item", close_during_decode)
    else:

        def close_during_feed(item):
            tee_holder["tee"].close()
            accumulator.payloads.append(item)

        accumulator.feed_payload = close_during_feed

    tee = gateway_module._tee_tool_calls(underlying, accumulator)
    tee_holder["tee"] = tee
    outcomes: list[object] = []

    def consume() -> None:
        try:
            outcomes.append(next(tee))
        except BaseException as exc:  # noqa: BLE001 - asserted below
            outcomes.append(exc)

    consumer = threading.Thread(target=consume, daemon=True)
    consumer.start()
    consumer.join(timeout=0.5)

    assert not consumer.is_alive()
    assert len(outcomes) == 1 and isinstance(outcomes[0], StopIteration)
    assert underlying.close_calls == 1
    assert accumulator.payloads == ([] if close_stage == "decode" else [payload])


@pytest.mark.parametrize("failure_stage", ["decode", "feed"])
def test_tee_tool_calls_closes_once_on_decode_or_accumulator_failure(
    failure_stage: str,
    monkeypatch,
) -> None:
    underlying = _CloseTrackingIterator([{"content": "chunk"}])
    accumulator = _RecordingAccumulator(
        ValueError("feed-primary") if failure_stage == "feed" else None
    )
    if failure_stage == "decode":

        def fail_decode(_item):
            raise ValueError("decode-primary")

        monkeypatch.setattr(gateway_module, "_decode_stream_item", fail_decode)
    tee = gateway_module._tee_tool_calls(underlying, accumulator)

    with pytest.raises(ValueError, match=f"{failure_stage}-primary"):
        next(tee)
    assert underlying.close_calls == 1
    tee.close()
    assert underlying.close_calls == 1


def test_tee_tool_calls_closes_mapping_when_accumulator_fails() -> None:
    response = _CloseableMapping(
        {"choices": []},
        close_failure=KeyboardInterrupt("close-secondary"),
    )
    accumulator = _RecordingAccumulator(ValueError("feed-primary"))

    caught: BaseException | None = None
    try:
        gateway_module._tee_tool_calls(response, accumulator)
    except BaseException as exc:  # noqa: BLE001 - primary identity asserted below
        caught = exc

    assert isinstance(caught, ValueError)
    assert str(caught) == "feed-primary"
    assert response.close_calls == 1


def test_tee_tool_calls_provider_error_survives_keyboard_interrupt_from_close() -> None:
    provider_error = ValueError("provider-primary")
    underlying = _CloseTrackingIterator(
        [],
        failure=provider_error,
        close_failure=KeyboardInterrupt("close-secondary"),
    )
    tee = gateway_module._tee_tool_calls(underlying, _RecordingAccumulator())

    caught: BaseException | None = None
    try:
        next(tee)
    except BaseException as exc:  # noqa: BLE001 - primary identity asserted below
        caught = exc

    assert caught is provider_error
    assert underlying.close_calls == 1
    tee.close()
    assert underlying.close_calls == 1


def test_tee_tool_calls_closes_underlying_iterator_once() -> None:
    payload = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ]
                }
            }
        ]
    }

    exhausted = _CloseTrackingIterator([payload])
    exhausted_accumulator = gateway_module._ToolCallAccumulator()
    exhausted_tee = gateway_module._tee_tool_calls(
        exhausted,
        exhausted_accumulator,
    )
    assert list(exhausted_tee) == [payload]
    assert exhausted_accumulator.calls()[0]["id"] == "call-1"
    assert exhausted.close_calls == 1
    exhausted_tee.close()
    assert exhausted.close_calls == 1

    provider_failure = RuntimeError("provider-primary-error")
    failed = _CloseTrackingIterator(
        [],
        failure=provider_failure,
        close_failure=RuntimeError("close-secondary-error"),
    )
    failed_tee = gateway_module._tee_tool_calls(
        failed,
        gateway_module._ToolCallAccumulator(),
    )
    with pytest.raises(RuntimeError, match="provider-primary-error"):
        next(failed_tee)
    assert failed.close_calls == 1
    failed_tee.close()
    assert failed.close_calls == 1

    caller_closed = _CloseTrackingIterator([payload, payload])
    caller_closed_tee = gateway_module._tee_tool_calls(
        caller_closed,
        gateway_module._ToolCallAccumulator(),
    )
    assert next(caller_closed_tee) == payload
    caller_closed_tee.close()
    caller_closed_tee.close()
    assert caller_closed.close_calls == 1


@pytest.mark.asyncio
async def test_tools_run_real_answer_equal_to_fallback_copy_survives() -> None:
    """Review minor m4: a REAL model answer that happens to equal the
    fallback copy string must flow through in tools mode — suppression now
    happens at generation (provenance), not by string equality."""
    response = {"choices": [{"message": {"content": NO_PROVIDER_CONTENT_COPY}}]}

    def fake_chat_api_call(**_kwargs):
        return response

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"groq": {"api_key": "sk-test"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider="groq", explicit_model="llama3-groq", streaming=False
        )
    )

    items = await _collect(gateway, resolution, tools=TOOLS)

    assert items == [NO_PROVIDER_CONTENT_COPY]  # it IS the model's answer here


# ---- system-row extraction (PR #1112 Qodo finding 3) ----


def _bare_resolution(execution_key: str = "anthropic") -> ConsoleProviderResolution:
    """Minimal ready resolution for direct `_chat_api_kwargs` calls."""
    return ConsoleProviderResolution(
        provider="anthropic",
        base_url="",
        model="claude-x",
        ready=True,
        execution_key=execution_key,
        api_key="k",
        streaming=False,
    )


def test_chat_api_kwargs_extracts_leading_system_rows_to_system_message() -> None:
    """Leading system rows leave the payload and ride `system_message`.

    Anthropic/Gemini adapters only accept system content via their dedicated
    parameter and reject (or drop) `role="system"` rows in the message
    array, so the Console's system prompt / folded greeting must be
    extracted here or those providers never see it.
    """
    messages = [
        {"role": "system", "content": "Stay in character."},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]

    kwargs = ConsoleProviderGateway._chat_api_kwargs(_bare_resolution(), messages)

    assert kwargs["system_message"] == "Stay in character."
    assert [m["role"] for m in kwargs["messages_payload"]] == ["user", "assistant"]


def test_chat_api_kwargs_joins_multiple_leading_system_rows() -> None:
    """Contiguous leading system rows concatenate into one system_message."""
    messages = [
        {"role": "system", "content": "A."},
        {"role": "system", "content": "B."},
        {"role": "user", "content": "hi"},
    ]

    kwargs = ConsoleProviderGateway._chat_api_kwargs(_bare_resolution(), messages)

    assert kwargs["system_message"] == "A.\n\nB."
    assert [m["role"] for m in kwargs["messages_payload"]] == ["user"]


def test_chat_api_kwargs_without_system_rows_omits_system_message() -> None:
    """No leading system rows: payload passes through, no system_message key."""
    messages = [{"role": "user", "content": "hi"}]

    kwargs = ConsoleProviderGateway._chat_api_kwargs(_bare_resolution(), messages)

    assert "system_message" not in kwargs
    assert kwargs["messages_payload"] == messages


def test_chat_api_kwargs_preserves_project_marker_for_native_grouping_only() -> None:
    from tldw_chatbook.Chat.console_project_instructions import EPHEMERAL_ORIGIN_KEY

    row = {
        "role": "user",
        "content": "context",
        EPHEMERAL_ORIGIN_KEY: "project_instructions",
    }
    for endpoint in ("anthropic", "google"):
        kwargs = ConsoleProviderGateway._chat_api_kwargs(
            _bare_resolution(endpoint), [row]
        )
        assert kwargs["messages_payload"][0][EPHEMERAL_ORIGIN_KEY]

    kwargs = ConsoleProviderGateway._chat_api_kwargs(_bare_resolution("openai"), [row])
    assert EPHEMERAL_ORIGIN_KEY in kwargs["messages_payload"][0]
    assert row[EPHEMERAL_ORIGIN_KEY] == "project_instructions"
    assert kwargs["messages_payload"] == [row]


def test_chat_api_kwargs_system_message_is_byte_stable_across_turns() -> None:
    """The extracted system_message must be BYTE-identical turn over turn.

    Anthropic prompt caching matches on an exact byte prefix (tools ->
    system -> messages), so any drift in the extracted system string -- a
    changed join, an interpolated timestamp, a re-ordered row -- silently
    invalidates the cache and re-pays the 1.25x write premium on every send.
    This is the seam the cost-ticker spec names: the gateway's
    leading-system-row extraction, not the controller's verbatim prompt.

    Note the extraction is normalizing, not verbatim: rows are stripped and
    joined with "\\n\\n". That is fine for caching precisely because it is
    deterministic -- the same rows always produce the same bytes.
    """
    system_rows = [
        {"role": "system", "content": "You are terse.\n\nAnswer in one line."},
        {"role": "system", "content": "Never use emoji."},
    ]
    turn_1 = system_rows + [{"role": "user", "content": "first question"}]
    turn_2 = turn_1 + [
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
    ]

    kwargs_1 = ConsoleProviderGateway._chat_api_kwargs(_bare_resolution(), turn_1)
    kwargs_2 = ConsoleProviderGateway._chat_api_kwargs(_bare_resolution(), turn_2)

    assert kwargs_1["system_message"] == kwargs_2["system_message"]
    assert kwargs_1["system_message"].encode() == kwargs_2["system_message"].encode()
    # and the history prefix itself is untouched by the extraction
    assert (
        kwargs_2["messages_payload"][: len(kwargs_1["messages_payload"])]
        == (kwargs_1["messages_payload"])
    )


# ---- per-turn cache_control opt-in (Console-only) ----


def test_chat_api_kwargs_forwards_configured_anthropic_base_url() -> None:
    """Confirm a configured Anthropic api_base_url reaches the primary send path's kwargs."""
    resolution = ConsoleProviderResolution(
        provider="anthropic",
        base_url="https://proxy.example.test/v1",
        model="claude-x",
        ready=True,
        execution_key="anthropic",
        api_key="k",
        streaming=False,
    )

    kwargs = ConsoleProviderGateway._chat_api_kwargs(
        resolution, [{"role": "user", "content": "hi"}]
    )

    assert kwargs["api_base_url"] == "https://proxy.example.test/v1"


@pytest.mark.parametrize("provider", ["mistral", "mistralai"])
def test_all_primary_mistral_kwargs_paths_forward_resolved_base(provider) -> None:
    resolution = ConsoleProviderResolution(
        provider=provider,
        base_url=f"https://{provider}.example.test/v1",
        model="mistral-model",
        ready=True,
        execution_key=provider,
        api_key="mistral-test-key",
        streaming=False,
    )
    messages = [{"role": "user", "content": "hi"}]
    gateway = ConsoleProviderGateway()
    prepared = gateway.prepare_chat_request(resolution, messages)

    assert gateway._chat_api_kwargs(resolution, messages)["api_base_url"] == (
        resolution.base_url
    )
    assert (
        gateway._chat_api_kwargs_from_prepared(
            resolution,
            prepared,
        )["api_base_url"]
        == resolution.base_url
    )


@pytest.mark.parametrize(
    ("provider", "execution_key"),
    [
        ("Custom OpenAI API", "custom-openai-api"),
        ("custom_openai_api_2", "custom-openai-api-2"),
    ],
)
def test_all_primary_custom_kwargs_paths_forward_resolved_chat_url(
    provider: str,
    execution_key: str,
) -> None:
    chat_url = f"https://{execution_key}.example.test/proxy/v1/chat/completions"
    resolution = ConsoleProviderResolution(
        provider=provider,
        base_url=chat_url,
        model="custom-model",
        ready=True,
        execution_key=execution_key,
        api_key="custom-test-credential",
        streaming=False,
    )
    messages = [{"role": "user", "content": "hi"}]
    gateway = ConsoleProviderGateway()
    prepared = gateway.prepare_chat_request(resolution, messages)

    assert gateway._chat_api_kwargs(resolution, messages)["api_base_url"] == chat_url
    assert (
        gateway._chat_api_kwargs_from_prepared(resolution, prepared)["api_base_url"]
        == chat_url
    )


@pytest.mark.parametrize("execution_key", ["custom-openai-api", "custom-openai-api-2"])
def test_all_primary_custom_kwargs_paths_preserve_explicit_keyless_decision(
    execution_key: str,
) -> None:
    resolution = ConsoleProviderResolution(
        provider=execution_key,
        base_url=f"https://{execution_key}.example.test/proxy/v1/chat/completions",
        model="custom-model",
        ready=True,
        execution_key=execution_key,
        api_key=None,
        streaming=False,
    )
    messages = [{"role": "user", "content": "hi"}]
    gateway = ConsoleProviderGateway()
    prepared = gateway.prepare_chat_request(resolution, messages)

    for kwargs in (
        gateway._chat_api_kwargs(resolution, messages),
        gateway._chat_api_kwargs_from_prepared(resolution, prepared),
    ):
        assert "api_key" not in kwargs
        assert kwargs["api_key_resolved"] is True


def test_chat_api_kwargs_omits_api_base_url_for_unpinned_provider() -> None:
    """Providers without an established pin keep their existing kwargs."""
    resolution = ConsoleProviderResolution(
        provider="openai",
        base_url="https://proxy.example.test/v1",
        model="gpt-4.1",
        ready=True,
        execution_key="openai",
        api_key="k",
        streaming=False,
    )

    kwargs = ConsoleProviderGateway._chat_api_kwargs(
        resolution, [{"role": "user", "content": "hi"}]
    )

    assert "api_base_url" not in kwargs


def test_chat_api_kwargs_omits_prompt_caching_for_non_anthropic() -> None:
    """`prompt_caching=None` is stripped, so non-Anthropic kwargs are
    unchanged from before prompt caching existed."""
    resolution = ConsoleProviderResolution(
        provider="openai",
        base_url="",
        model="gpt-4.1",
        ready=True,
        execution_key="openai",
        api_key="k",
        streaming=False,
    )

    kwargs = ConsoleProviderGateway._chat_api_kwargs(
        resolution, [{"role": "user", "content": "hi"}]
    )

    assert "prompt_caching" not in kwargs


@pytest.mark.asyncio
async def test_anthropic_resolution_forwards_prompt_caching_opt_in() -> None:
    """Console sends are multi-turn, so they opt into the per-turn
    breakpoint; one-shot callers of `chat_with_anthropic` never do."""
    calls: list[dict] = []

    def fake_chat_api_call(**kwargs):
        calls.append(kwargs)
        return "done"

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"anthropic": {"api_key": "k"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider="anthropic",
            explicit_model="claude-sonnet-4-6",
            streaming=False,
        )
    )

    assert resolution.prompt_caching is True

    _ = [
        chunk
        async for chunk in gateway.stream_chat(
            resolution, [{"role": "user", "content": "hi"}]
        )
    ]

    assert calls[0]["api_endpoint"] == "anthropic"
    assert calls[0]["prompt_caching"] is True


@pytest.mark.asyncio
async def test_anthropic_resolution_respects_caching_kill_switch() -> None:
    """`[caching] anthropic_enabled = false` turns the opt-in off at the
    gateway too (the provider's own kill-switch is the second line)."""
    calls: list[dict] = []

    def fake_chat_api_call(**kwargs):
        calls.append(kwargs)
        return "done"

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {
            "api_settings": {"anthropic": {"api_key": "k"}},
            "caching": {"anthropic_enabled": False},
        },
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider="anthropic",
            explicit_model="claude-sonnet-4-6",
            streaming=False,
        )
    )

    assert resolution.prompt_caching is False

    _ = [
        chunk
        async for chunk in gateway.stream_chat(
            resolution, [{"role": "user", "content": "hi"}]
        )
    ]

    # falsy either way: the per-turn marker is off
    assert not calls[0].get("prompt_caching")


@pytest.mark.asyncio
async def test_anthropic_resolution_respects_kill_switch_in_load_settings_shape() -> (
    None
):
    """The live Console's config_provider is `load_settings()` output, which
    nests the raw TOML under COMPREHENSIVE_CONFIG_RAW and never projects
    `[caching]` to the top level the way it does `api_settings` (Qodo
    finding, PR #1239): a plain `app_config.get("caching")` always misses
    and the kill-switch silently reads as always-on. Pin resolution against
    exactly that shape."""
    gateway = ConsoleProviderGateway(
        config_provider=lambda: {
            "api_settings": {"anthropic": {"api_key": "k"}},
            "COMPREHENSIVE_CONFIG_RAW": {"caching": {"anthropic_enabled": False}},
        },
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider="anthropic",
            explicit_model="claude-sonnet-4-6",
            streaming=False,
        )
    )

    assert resolution.prompt_caching is False


@pytest.mark.asyncio
async def test_non_anthropic_resolution_has_no_prompt_caching_flag() -> None:
    gateway = ConsoleProviderGateway(
        config_provider=lambda: {
            "api_settings": {"openai": {"api_key": "sk-test-placeholder-not-a-key"}}
        },
        chat_api_call_fn=lambda **_kwargs: "done",
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider="openai", explicit_model="gpt-4.1", streaming=False
        )
    )

    assert resolution.prompt_caching is None


# ---- task-2114: configured api_base_url reaches the real posted URL ----


def _fake_anthropic_message_response() -> dict:
    return {
        "id": "msg_test",
        "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


class _CapturedURLSession:
    """Minimal `requests.Session` stand-in recording only the posted URL --
    a narrower cousin of `Tests/Chat/test_chat_functions.py::_CapturedSession`
    (that file already owns the full request-capture fixture; this one only
    needs the URL for the assertion below)."""

    def __init__(self, captured: dict) -> None:
        self._captured = captured

    def __enter__(self):
        return self

    def __exit__(self, *_exc_info):
        return False

    def mount(self, *_args, **_kwargs) -> None:
        return None

    def post(
        self,
        url,
        *,
        headers=None,
        json=None,
        stream=False,
        timeout=None,
        allow_redirects=None,
    ):
        self._captured["url"] = url
        return _FakeAnthropicPostResponse()


class _FakeAnthropicPostResponse:
    status_code = 200
    text = "{}"

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return _fake_anthropic_message_response()


class _CapturedMistralSession:
    def __init__(self, calls: list[tuple[str, str]]) -> None:
        self._calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *_exc_info):
        return False

    def mount(self, *_args, **_kwargs) -> None:
        return None

    def post(self, url, *, headers=None, json=None, stream=False, timeout=None):
        self._calls.append((url, (headers or {}).get("Authorization", "")))
        return _FakeMistralPostResponse()


class _FakeMistralPostResponse:
    status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {},
        }


class _CapturedCustomSession:
    def __init__(self, calls: list[tuple[str, str]]) -> None:
        self._calls = calls

    def mount(self, *_args, **_kwargs) -> None:
        return None

    def post(self, url, *, headers=None, json=None, stream=False, timeout=None):
        self._calls.append((url, (headers or {}).get("Authorization", "")))
        return _FakeMistralPostResponse()


@pytest.mark.asyncio
async def test_console_send_keeps_each_mistral_credential_on_its_own_endpoint(
    monkeypatch,
) -> None:
    from tldw_chatbook.LLM_Calls import LLM_API_Calls

    class RuntimeConfigSnapshotStub:
        def __init__(self, values) -> None:
            self.values = values

    config = {
        "api_settings": {
            "mistral": {
                "api_key": "legacy-mistral-test-key",
                "api_base_url": "https://legacy-mistral.example.test/v1",
                "model": "legacy-model",
            },
            "mistralai": {
                "api_key": "catalog-mistral-test-key",
                "api_base_url": "https://catalog-mistral.example.test/v1",
                "model": "catalog-model",
            },
        }
    }
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        LLM_API_Calls,
        "create_default_session",
        lambda: _CapturedMistralSession(calls),
    )
    monkeypatch.setattr(
        LLM_API_Calls,
        "get_runtime_config_snapshot",
        lambda: RuntimeConfigSnapshotStub(config),
    )
    gateway = ConsoleProviderGateway(
        config_provider=lambda: config,
        environ={},
    )

    for provider in ("mistral", "mistralai"):
        resolution = await gateway.resolve_for_send(
            ConsoleProviderSelection(provider=provider, streaming=False)
        )
        assert resolution.ready is True
        assert [
            chunk
            async for chunk in gateway.stream_chat(
                resolution,
                [{"role": "user", "content": "hello"}],
            )
        ] == ["ok"]

    assert calls == [
        (
            "https://legacy-mistral.example.test/v1/chat/completions",
            "Bearer legacy-mistral-test-key",
        ),
        (
            "https://catalog-mistral.example.test/v1/chat/completions",
            "Bearer catalog-mistral-test-key",
        ),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("builder", ["primary", "prepared"])
@pytest.mark.parametrize("legacy_variant", ["stale_credential", "keyless", "absent"])
@pytest.mark.parametrize(
    ("provider", "owner", "configured_endpoint", "expected_chat_url"),
    [
        (
            "Custom OpenAI API",
            "custom",
            "https://owner-custom.example.test/proxy",
            "https://owner-custom.example.test/proxy/v1/chat/completions",
        ),
        (
            "custom_openai_api_2",
            "custom_2",
            "https://owner-custom-2.example.test/proxy/v1/chat/completions",
            "https://owner-custom-2.example.test/proxy/v1/chat/completions",
        ),
    ],
)
async def test_console_send_keeps_custom_endpoint_and_credential_paired(
    monkeypatch,
    builder: str,
    legacy_variant: str,
    provider: str,
    owner: str,
    configured_endpoint: str,
    expected_chat_url: str,
) -> None:
    from tldw_chatbook.LLM_Calls import LLM_API_Calls_Local

    class RuntimeConfigSnapshotStub:
        def __init__(self, values) -> None:
            self.values = values

    owner_credential = f"{owner}-owner-credential"
    config = {
        "api_settings": {
            owner: {
                "api_key": owner_credential,
                "api_url": configured_endpoint,
                "model": f"{owner}-model",
            }
        }
    }
    legacy_endpoint = "https://legacy-fallback.example.test/v1/chat/completions"
    legacy_values = (
        {
            "api_settings": {
                "custom": {
                    "api_url": legacy_endpoint,
                    "model": "legacy-model",
                },
            },
            "custom_openai_api_2": {
                "api_ip": legacy_endpoint,
                "model": "legacy-model",
            },
        }
        if legacy_variant != "absent"
        else {}
    )
    if legacy_variant == "stale_credential":
        legacy_values["api_settings"]["custom"]["api_key"] = "stale-credential"
        legacy_values["custom_openai_api_2"]["api_key"] = "stale-credential"
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        LLM_API_Calls_Local,
        "create_default_session",
        lambda: _CapturedCustomSession(calls),
    )
    monkeypatch.setattr(
        LLM_API_Calls_Local,
        "get_runtime_config_snapshot",
        lambda: RuntimeConfigSnapshotStub(legacy_values),
    )
    monkeypatch.setattr(
        LLM_API_Calls_Local,
        "load_settings",
        lambda: legacy_values,
    )
    gateway = ConsoleProviderGateway(config_provider=lambda: config, environ={})
    if builder == "primary":
        gateway._chat_api_kwargs_from_prepared = lambda resolution, request: (
            gateway._chat_api_kwargs(
                resolution,
                [dict(message) for message in request.messages_payload],
            )
        )

    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider=provider, streaming=False)
    )
    assert resolution.ready is True
    assert resolution.base_url == expected_chat_url
    assert [
        chunk
        async for chunk in gateway.stream_chat(
            resolution,
            [{"role": "user", "content": "hello"}],
        )
    ] == ["ok"]

    assert calls == [(expected_chat_url, f"Bearer {owner_credential}")]
    assert legacy_endpoint not in {url for url, _credential in calls}


@pytest.mark.asyncio
@pytest.mark.parametrize("builder", ["primary", "prepared"])
@pytest.mark.parametrize("legacy_credential", ["stale-legacy-credential", None])
@pytest.mark.parametrize(
    ("provider", "owner", "configured_endpoint", "expected_chat_url"),
    [
        (
            "Custom OpenAI API",
            "custom",
            "https://keyless-custom.example.test/proxy",
            "https://keyless-custom.example.test/proxy/v1/chat/completions",
        ),
        (
            "custom_openai_api_2",
            "custom_2",
            "https://keyless-custom-2.example.test/proxy/v1/chat/completions",
            "https://keyless-custom-2.example.test/proxy/v1/chat/completions",
        ),
    ],
)
async def test_console_keyless_custom_send_never_falls_back_to_legacy_credential(
    monkeypatch,
    caplog,
    builder: str,
    legacy_credential: str | None,
    provider: str,
    owner: str,
    configured_endpoint: str,
    expected_chat_url: str,
) -> None:
    from tldw_chatbook.Chat import Chat_Functions
    from tldw_chatbook.LLM_Calls import LLM_API_Calls_Local

    class RuntimeConfigSnapshotStub:
        def __init__(self, values) -> None:
            self.values = values

    config = {
        "api_settings": {
            owner: {
                "api_url": configured_endpoint,
                "model": f"{owner}-model",
            }
        }
    }
    legacy_endpoint = "https://legacy-fallback.example.test/v1/chat/completions"
    legacy_values = {
        "api_settings": {
            "custom": {"api_url": legacy_endpoint, "model": "legacy-model"}
        },
        "custom_openai_api_2": {
            "api_ip": legacy_endpoint,
            "model": "legacy-model",
        },
    }
    if legacy_credential is not None:
        legacy_values["api_settings"]["custom"]["api_key"] = legacy_credential
        legacy_values["custom_openai_api_2"]["api_key"] = legacy_credential
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        LLM_API_Calls_Local,
        "create_default_session",
        lambda: _CapturedCustomSession(calls),
    )
    monkeypatch.setattr(
        LLM_API_Calls_Local,
        "get_runtime_config_snapshot",
        lambda: RuntimeConfigSnapshotStub(legacy_values),
    )
    monkeypatch.setattr(
        LLM_API_Calls_Local,
        "load_settings",
        lambda: legacy_values,
    )
    gateway = ConsoleProviderGateway(config_provider=lambda: config, environ={})
    if builder == "primary":
        gateway._chat_api_kwargs_from_prepared = lambda resolution, request: (
            gateway._chat_api_kwargs(
                resolution,
                [dict(message) for message in request.messages_payload],
            )
        )

    loguru_messages: list[str] = []
    sink_id = Chat_Functions.logger.add(
        lambda message: loguru_messages.append(str(message)),
        level="DEBUG",
    )
    try:
        resolution = await gateway.resolve_for_send(
            ConsoleProviderSelection(provider=provider, streaming=False)
        )
        assert resolution.ready is True
        assert resolution.api_key is None
        assert [
            chunk
            async for chunk in gateway.stream_chat(
                resolution,
                [{"role": "user", "content": "hello"}],
            )
        ] == ["ok"]
    finally:
        Chat_Functions.logger.remove(sink_id)

    assert calls == [(expected_chat_url, "")]
    if legacy_credential is not None:
        assert legacy_credential not in repr(calls)
        assert legacy_credential not in caplog.text
        assert legacy_credential not in "".join(loguru_messages)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "owner", "env_var"),
    [
        ("Custom OpenAI API", "custom", "CUSTOM_API_KEY"),
        ("custom_openai_api_2", "custom_2", "CUSTOM_2_API_KEY"),
    ],
)
async def test_console_persisted_explicit_keyless_ignores_saved_env_and_legacy_keys(
    monkeypatch,
    provider: str,
    owner: str,
    env_var: str,
) -> None:
    from tldw_chatbook.LLM_Calls import LLM_API_Calls_Local

    class RuntimeConfigSnapshotStub:
        def __init__(self, values) -> None:
            self.values = values

    endpoint = f"https://{owner}.keyless.example.test/v1/chat/completions"
    config = {
        "api_settings": {
            owner: {
                "api_url": endpoint,
                "model": "keyless-model",
                "credential_source": "none",
                "api_key": "saved-chat-canary",
                "api_key_env_var": env_var,
            }
        }
    }
    legacy_values = {
        "api_settings": {
            "custom": {
                "api_url": endpoint,
                "api_key": "legacy-chat-canary",
            }
        },
        "custom_openai_api_2": {
            "api_ip": endpoint,
            "api_key": "legacy-chat-canary",
        },
    }
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        LLM_API_Calls_Local,
        "create_default_session",
        lambda: _CapturedCustomSession(calls),
    )
    monkeypatch.setattr(
        LLM_API_Calls_Local,
        "get_runtime_config_snapshot",
        lambda: RuntimeConfigSnapshotStub(legacy_values),
    )
    monkeypatch.setattr(LLM_API_Calls_Local, "load_settings", lambda: legacy_values)
    gateway = ConsoleProviderGateway(
        config_provider=lambda: config,
        environ={env_var: "environment-chat-canary"},
    )

    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider=provider, streaming=False)
    )
    chunks = [
        chunk
        async for chunk in gateway.stream_chat(
            resolution,
            [{"role": "user", "content": "hello"}],
        )
    ]

    assert resolution.ready is True
    assert resolution.api_key is None
    assert resolution.api_key_source is None
    assert chunks == ["ok"]
    assert calls == [(endpoint, "")]
    rendered = repr((resolution, calls))
    assert "saved-chat-canary" not in rendered
    assert "environment-chat-canary" not in rendered
    assert "legacy-chat-canary" not in rendered


def test_custom_adapter_fallback_honors_persisted_explicit_keyless(monkeypatch):
    from tldw_chatbook.LLM_Calls import LLM_API_Calls_Local

    class RuntimeConfigSnapshotStub:
        values = {
            "api_settings": {
                "custom": {
                    "api_url": "https://adapter-keyless.example/v1/chat/completions",
                    "model": "adapter-model",
                    "credential_source": "none",
                    "api_key": "saved-adapter-canary",
                    "api_key_env_var": "CUSTOM_API_KEY",
                }
            }
        }

    calls: list[tuple[str, str]] = []
    monkeypatch.setenv("CUSTOM_API_KEY", "environment-adapter-canary")
    monkeypatch.setattr(
        LLM_API_Calls_Local,
        "create_default_session",
        lambda: _CapturedCustomSession(calls),
    )
    monkeypatch.setattr(
        LLM_API_Calls_Local,
        "get_runtime_config_snapshot",
        lambda: RuntimeConfigSnapshotStub(),
    )

    result = LLM_API_Calls_Local.chat_with_custom_openai(
        [{"role": "user", "content": "hello"}],
        streaming=False,
    )

    assert result["choices"][0]["message"]["content"] == "ok"
    assert calls == [
        ("https://adapter-keyless.example/v1/chat/completions", "")
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["llama_cpp", "local_llamacpp"])
async def test_console_persisted_explicit_keyless_llamacpp_sends_no_authorization(
    provider: str,
) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            content=(
                b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                b"data: [DONE]\n\n"
            ),
        )

    endpoint = "http://127.0.0.1:19090"
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    gateway = ConsoleProviderGateway(
        http_client=client,
        config_provider=lambda: {
            "api_settings": {
                provider: {
                    "api_url": endpoint,
                    "model": "keyless-model",
                    "credential_source": "none",
                    "api_key": "saved-llama-chat-canary",
                    "api_key_env_var": "LLAMA_CPP_API_KEY",
                }
            }
        },
        environ={"LLAMA_CPP_API_KEY": "environment-llama-chat-canary"},
    )
    try:
        resolution = await gateway.resolve_for_send(
            ConsoleProviderSelection(
                provider=provider,
                base_url=endpoint,
                explicit_model="keyless-model",
            )
        )
        chunks = [
            chunk
            async for chunk in gateway.stream_chat(
                resolution,
                [{"role": "user", "content": "hello"}],
            )
        ]
    finally:
        await client.aclose()

    assert resolution.ready is True
    assert resolution.api_key is None
    assert chunks == ["ok"]
    assert [request.method for request in requests] == ["GET", "POST"]
    assert all("Authorization" not in request.headers for request in requests)


@pytest.mark.asyncio
async def test_console_llamacpp_explicit_stored_source_reaches_probe_and_chat():
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            content=(
                b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                b"data: [DONE]\n\n"
            ),
        )

    endpoint = "http://127.0.0.1:19091"
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    gateway = ConsoleProviderGateway(
        http_client=client,
        config_provider=lambda: {
            "api_settings": {
                "llama_cpp": {
                    "api_url": endpoint,
                    "model": "authenticated-model",
                    "credential_source": "stored",
                    "api_key": "stored-llama-request-canary",
                    "api_key_env_var": "LLAMA_CPP_API_KEY",
                }
            }
        },
        environ={"LLAMA_CPP_API_KEY": "ignored-llama-environment-canary"},
    )
    try:
        resolution = await gateway.resolve_for_send(
            ConsoleProviderSelection(
                provider="llama_cpp",
                base_url=endpoint,
                explicit_model="authenticated-model",
            )
        )
        chunks = [
            chunk
            async for chunk in gateway.stream_chat(
                resolution,
                [{"role": "user", "content": "hello"}],
            )
        ]
    finally:
        await client.aclose()

    assert resolution.ready is True
    assert resolution.api_key_source == "config:api_settings.llama_cpp.api_key"
    assert chunks == ["ok"]
    assert [request.headers.get("Authorization") for request in requests] == [
        "Bearer stored-llama-request-canary",
        "Bearer stored-llama-request-canary",
    ]
    assert "stored-llama-request-canary" not in repr(resolution)


@pytest.mark.asyncio
async def test_console_send_honors_configured_anthropic_base_url(monkeypatch) -> None:
    """Confirm the real gateway-to-adapter chain posts to a configured Anthropic api_base_url.

    Drives the real gateway -> ``chat_api_call`` -> ``chat_with_anthropic``
    chain (no ``chat_api_call_fn`` stand-in) on the primary Console send
    path, not just the auxiliary/one-shot path.

    Args:
        monkeypatch: Pytest fixture used to stub the outgoing HTTP session.
    """
    from tldw_chatbook.LLM_Calls import LLM_API_Calls

    captured: dict = {}
    monkeypatch.setattr(
        LLM_API_Calls, "create_default_session", lambda: _CapturedURLSession(captured)
    )

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {
            "api_settings": {
                "anthropic": {
                    "api_key": "k",
                    "api_base_url": "https://proxy.example.test/v1",
                }
            }
        },
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider="anthropic",
            explicit_model="claude-sonnet-4-6",
            streaming=False,
        )
    )
    assert resolution.base_url == "https://proxy.example.test/v1"

    _ = [
        chunk
        async for chunk in gateway.stream_chat(
            resolution, [{"role": "user", "content": "hi"}]
        )
    ]

    assert captured["url"] == "https://proxy.example.test/v1/messages"


@pytest.mark.asyncio
async def test_console_send_default_anthropic_url_unchanged_when_unconfigured(
    monkeypatch,
) -> None:
    """Confirm the default Anthropic endpoint is unchanged when api_base_url is not configured.

    Args:
        monkeypatch: Pytest fixture used to stub the outgoing HTTP session.
    """
    from tldw_chatbook.LLM_Calls import LLM_API_Calls

    captured: dict = {}
    monkeypatch.setattr(
        LLM_API_Calls, "create_default_session", lambda: _CapturedURLSession(captured)
    )

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"anthropic": {"api_key": "k"}}},
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider="anthropic",
            explicit_model="claude-sonnet-4-6",
            streaming=False,
        )
    )

    _ = [
        chunk
        async for chunk in gateway.stream_chat(
            resolution, [{"role": "user", "content": "hi"}]
        )
    ]

    assert captured["url"] == "https://api.anthropic.com/v1/messages"


@pytest.mark.asyncio
async def test_stream_chat_records_usage_payload_from_sse_chunk() -> None:
    usage_line = (
        'data: {"object": "chat.completion.chunk", "choices": [], '
        '"usage": {"prompt_tokens": 100, "completion_tokens": 20}}'
    )

    def fake_chat_api_call(**_kwargs):
        yield 'data: {"choices": [{"delta": {"content": "hi"}}]}'
        yield usage_line

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"openai": {"api_key": "sk-test"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="openai", explicit_model="gpt-4.1")
    )
    signals = ConsoleProviderStreamSignals()

    chunks = [
        chunk
        async for chunk in gateway.stream_chat(
            resolution, [{"role": "user", "content": "hi"}], signals=signals
        )
    ]

    assert chunks == ["hi"]  # usage chunk yields no text
    # The call ended, so its payload was closed out of the in-flight slot and
    # into the completed list -- one provider call, one billable payload.
    assert signals.usage_payload is None
    assert signals.usage_payloads() == [{"prompt_tokens": 100, "completion_tokens": 20}]


@pytest.mark.asyncio
async def test_stream_chat_merges_split_usage_payloads() -> None:
    # Anthropic emits input-side usage at message_start and output at end.
    # Both belong to ONE call, so they must still key-merge into ONE payload.
    def fake_chat_api_call(**_kwargs):
        yield (
            'data: {"choices": [], "usage": {"input_tokens": 3571, '
            '"cache_read_input_tokens": 6656}}'
        )
        yield 'data: {"choices": [{"delta": {"content": "hi"}}]}'
        yield 'data: {"choices": [], "usage": {"output_tokens": 727}}'

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"anthropic": {"api_key": "k"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider="anthropic", explicit_model="claude-sonnet-4-6"
        )
    )
    signals = ConsoleProviderStreamSignals()
    _ = [
        chunk
        async for chunk in gateway.stream_chat(
            resolution, [{"role": "user", "content": "hi"}], signals=signals
        )
    ]
    assert signals.usage_payloads() == [
        {
            "input_tokens": 3571,
            "cache_read_input_tokens": 6656,
            "output_tokens": 727,
        }
    ]


@pytest.mark.asyncio
async def test_usage_accumulates_per_call_without_leaking_stale_cache_fields() -> None:
    """Two provider calls on one signals object must not key-merge.

    Regression for the final-review F2 finding: an agent turn makes N calls
    through the SAME signals object. Merging call 2's ``prompt_tokens`` on top
    of call 1's ``prompt_tokens_details.cached_tokens`` made the second call
    look 100% cached (uncached_input=0) and fabricated a 4096-token cache
    read that was never billed.
    """
    calls = iter(
        (
            (
                'data: {"choices": [{"delta": {"content": "a"}}]}',
                'data: {"choices": [], "usage": {"prompt_tokens": 5000, '
                '"completion_tokens": 10, '
                '"prompt_tokens_details": {"cached_tokens": 4096}}}',
            ),
            (
                'data: {"choices": [{"delta": {"content": "b"}}]}',
                'data: {"choices": [], "usage": {"prompt_tokens": 900, '
                '"completion_tokens": 30}}',
            ),
        )
    )

    def fake_chat_api_call(**_kwargs):
        yield from next(calls)

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"openai": {"api_key": "sk-test"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="openai", explicit_model="gpt-4.1")
    )
    signals = ConsoleProviderStreamSignals()
    for _ in range(2):
        _ = [
            chunk
            async for chunk in gateway.stream_chat(
                resolution, [{"role": "user", "content": "hi"}], signals=signals
            )
        ]

    # Two separate payloads, each intact -- no cross-call key merge.
    assert signals.usage_payloads() == [
        {
            "prompt_tokens": 5000,
            "completion_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 4096},
        },
        {"prompt_tokens": 900, "completion_tokens": 30},
    ]

    # And the normalized, summed buckets are the honest bill: call 1 is
    # 904 uncached + 4096 cached, call 2 is 900 uncached + 0 cached.
    total = None
    for payload in signals.usage_payloads():
        usage = ProviderUsage.from_provider_payload(
            payload, provider="openai", model="gpt-4.1"
        )
        total = usage if total is None else total.plus(usage)
    assert total.uncached_input == 1804
    assert total.cache_read == 4096
    assert total.output == 40


@pytest.mark.asyncio
async def test_non_streaming_mapping_response_records_usage() -> None:
    def fake_chat_api_call(**_kwargs):
        return {
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        }

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"openai": {"api_key": "sk-test"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="openai", explicit_model="gpt-4.1")
    )
    signals = ConsoleProviderStreamSignals()
    chunks = [
        chunk
        async for chunk in gateway.stream_chat(
            resolution, [{"role": "user", "content": "hi"}], signals=signals
        )
    ]
    assert chunks == ["hello"]
    assert signals.usage_payloads() == [{"prompt_tokens": 10, "completion_tokens": 2}]


@pytest.mark.asyncio
async def test_qwencloud_responses_terminal_usage_reaches_console_signals_without_copy() -> (
    None
):
    usage = {
        "input_tokens": 9,
        "input_tokens_details": {"cached_tokens": 2},
        "output_tokens": 3,
        "output_tokens_details": {"reasoning_tokens": 1},
        "total_tokens": 12,
    }
    terminal = {
        "choices": [{"delta": {"content": ""}, "finish_reason": "stop"}],
        "usage": usage,
    }

    def fake_chat_api_call(**_kwargs):
        return iter(
            (
                {"choices": [{"delta": {"content": "answer"}}]},
                terminal,
            )
        )

    gateway = ConsoleProviderGateway(chat_api_call_fn=fake_chat_api_call)
    resolution = ConsoleProviderResolution(
        provider="QwenCloud",
        base_url="https://workspace.example.test/compatible-mode/v1",
        model="qwen3.8-max",
        ready=True,
        readiness_key="qwencloud",
        execution_key="qwencloud",
        api_key="qwen-test-key",
        streaming=True,
        api_mode="responses",
    )
    signals = ConsoleProviderStreamSignals()

    chunks = [
        chunk
        async for chunk in gateway.stream_chat(
            resolution,
            [{"role": "user", "content": "hi"}],
            signals=signals,
        )
    ]

    assert chunks == ["answer"]
    assert signals.synthetic_fallback_emitted is False
    assert signals.usage_payloads() == [usage]
    normalized = ProviderUsage.from_provider_payload(
        signals.usage_payloads()[0],
        provider="qwencloud",
        model="qwen3.8-max",
    )
    assert normalized is not None
    assert normalized.uncached_input == 7
    assert normalized.cache_read == 2
    assert normalized.output == 3


class _FirstNextBlockingCloseTrackingIterator:
    def __init__(self) -> None:
        self.next_entered = threading.Event()
        self.released = threading.Event()
        self.closed = threading.Event()
        self._state_lock = threading.Lock()
        self.next_calls = 0
        self.close_calls = 0

    def __iter__(self):
        return self

    def __next__(self):
        with self._state_lock:
            self.next_calls += 1
        self.next_entered.set()
        self.released.wait(timeout=5)
        raise StopIteration

    def close(self) -> None:
        with self._state_lock:
            self.close_calls += 1
        self.closed.set()
        self.released.set()


@pytest.mark.asyncio
async def test_gateway_cancellation_before_qwencloud_iterator_retention_closes_without_iteration() -> (
    None
):
    provider_call_entered = threading.Event()
    allow_provider_return = threading.Event()
    provider_iterator = _FirstNextBlockingCloseTrackingIterator()

    def delayed_chat_api_call(**_kwargs):
        provider_call_entered.set()
        allow_provider_return.wait(timeout=5)
        return provider_iterator

    gateway = ConsoleProviderGateway(chat_api_call_fn=delayed_chat_api_call)
    resolution = ConsoleProviderResolution(
        provider="QwenCloud",
        base_url="https://workspace.example.test/compatible-mode/v1",
        model="qwen3.8-max",
        ready=True,
        readiness_key="qwencloud",
        execution_key="qwencloud",
        api_key="qwen-test-key",
        streaming=True,
        api_mode="responses",
    )
    stream = gateway.stream_chat(
        resolution,
        [{"role": "user", "content": "hi"}],
    )
    pending = asyncio.create_task(anext(stream))

    try:
        assert await asyncio.to_thread(provider_call_entered.wait, 1)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

        allow_provider_return.set()
        for _ in range(100):
            if (
                provider_iterator.closed.is_set()
                or provider_iterator.next_entered.is_set()
            ):
                break
            await asyncio.sleep(0.01)

        assert provider_iterator.next_calls == 0
        assert provider_iterator.closed.is_set()
        assert provider_iterator.close_calls == 1
        await stream.aclose()
        assert provider_iterator.close_calls == 1
    finally:
        allow_provider_return.set()
        if provider_iterator.next_entered.is_set():
            provider_iterator.released.set()
        if not pending.done():
            pending.cancel()
        await stream.aclose()


@pytest.mark.asyncio
async def test_gateway_cancellation_after_retention_before_normalization_does_not_iterate(
    monkeypatch,
) -> None:
    normalization_entered = threading.Event()
    allow_normalization = threading.Event()
    provider_iterator = _FirstNextBlockingCloseTrackingIterator()
    gateway = ConsoleProviderGateway(
        chat_api_call_fn=lambda **_kwargs: provider_iterator
    )
    original_normalize = gateway.normalize_provider_response

    def paused_normalize(response, *, suppress_fallback_copy=False, signals=None):
        normalization_entered.set()
        allow_normalization.wait(timeout=5)
        return original_normalize(
            response,
            suppress_fallback_copy=suppress_fallback_copy,
            signals=signals,
        )

    monkeypatch.setattr(gateway, "normalize_provider_response", paused_normalize)
    resolution = ConsoleProviderResolution(
        provider="QwenCloud",
        base_url="https://workspace.example.test/compatible-mode/v1",
        model="qwen3.8-max",
        ready=True,
        readiness_key="qwencloud",
        execution_key="qwencloud",
        api_key="qwen-test-key",
        streaming=True,
        api_mode="responses",
    )
    stream = gateway.stream_chat(
        resolution,
        [{"role": "user", "content": "hi"}],
    )
    pending = asyncio.create_task(anext(stream))

    try:
        assert await asyncio.to_thread(normalization_entered.wait, 1)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert provider_iterator.closed.wait(timeout=1)
        assert provider_iterator.close_calls == 1

        allow_normalization.set()
        for _ in range(20):
            if provider_iterator.next_entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert provider_iterator.next_calls == 0
        await stream.aclose()
        assert provider_iterator.close_calls == 1
    finally:
        allow_normalization.set()
        provider_iterator.released.set()
        if not pending.done():
            pending.cancel()
        await stream.aclose()


class _BlockingCloseTrackingIterator:
    def __init__(self) -> None:
        self._first = True
        self.blocked = threading.Event()
        self.released = threading.Event()
        self.closed = threading.Event()
        self._close_lock = threading.Lock()
        self.close_calls = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._first:
            self._first = False
            return {"choices": [{"delta": {"content": "partial"}}]}
        self.blocked.set()
        self.released.wait(timeout=5)
        raise StopIteration

    def close(self) -> None:
        with self._close_lock:
            self.close_calls += 1
        self.closed.set()
        self.released.set()


@pytest.mark.asyncio
async def test_gateway_cancellation_closes_qwencloud_iterator() -> None:
    provider_iterator = _BlockingCloseTrackingIterator()
    gateway = ConsoleProviderGateway(
        chat_api_call_fn=lambda **_kwargs: provider_iterator
    )
    resolution = ConsoleProviderResolution(
        provider="QwenCloud",
        base_url="https://workspace.example.test/compatible-mode/v1",
        model="qwen3.8-max",
        ready=True,
        readiness_key="qwencloud",
        execution_key="qwencloud",
        api_key="qwen-test-key",
        streaming=True,
        api_mode="responses",
    )
    stream = gateway.stream_chat(
        resolution,
        [{"role": "user", "content": "hi"}],
    )

    pending: asyncio.Task[object] | None = None
    try:
        assert await anext(stream) == "partial"
        for _ in range(100):
            if provider_iterator.blocked.is_set():
                break
            await asyncio.sleep(0.01)
        assert provider_iterator.blocked.is_set()

        pending = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

        for _ in range(100):
            if provider_iterator.closed.is_set():
                break
            await asyncio.sleep(0.01)
        assert provider_iterator.closed.is_set()
        assert provider_iterator.close_calls == 1

        await stream.aclose()
        assert provider_iterator.close_calls == 1
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
        provider_iterator.released.set()
        await stream.aclose()


@pytest.mark.asyncio
async def test_stream_without_usage_leaves_signals_none() -> None:
    def fake_chat_api_call(**_kwargs):
        yield "plain text"

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"openai": {"api_key": "sk-test"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="openai", explicit_model="gpt-4.1")
    )
    signals = ConsoleProviderStreamSignals()
    _ = [
        chunk
        async for chunk in gateway.stream_chat(
            resolution, [{"role": "user", "content": "hi"}], signals=signals
        )
    ]
    assert signals.usage_payload is None
    assert signals.usage_payloads() == []


def _auxiliary_resolution(**overrides) -> ConsoleProviderResolution:
    values = {
        "provider": "OpenAI",
        "base_url": "https://api.example.test/v1?token=ENDPOINT-CANARY",
        "model": "gpt-test",
        "ready": True,
        "readiness_key": "openai",
        "execution_key": "openai",
        "api_key": "API-KEY-CANARY",
        "temperature": 0.2,
        "top_p": 0.8,
        "min_p": 0.03,
        "top_k": 17,
        "max_tokens": 999,
        "seed": 42,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.2,
        "reasoning_effort": "high",
        "reasoning_summary": "auto",
        "verbosity": "low",
        "thinking_effort": "medium",
        "thinking_budget_tokens": 2048,
        "streaming": True,
    }
    values.update(overrides)
    return ConsoleProviderResolution(**values)


def _auxiliary_request(**overrides) -> AuxiliaryCompletionRequest:
    values = {
        "resolution": _auxiliary_resolution(),
        "messages": (
            {"role": "system", "content": "OPTIMIZER-CANARY"},
            {"role": "user", "content": "USER-CANARY"},
        ),
        "response_format": {"type": "json_object"},
        "max_output_tokens": 321,
    }
    values.update(overrides)
    return AuxiliaryCompletionRequest(**values)


def test_auxiliary_request_is_frozen_and_copies_nested_input() -> None:
    message = {"role": "user", "content": "BLOCK-CANARY"}
    required = ["rewritten_prompt"]
    response_format = {
        "type": "json_schema",
        "json_schema": {"required": required},
    }

    request = AuxiliaryCompletionRequest(
        resolution=_auxiliary_resolution(),
        messages=(message,),
        response_format=response_format,
        max_output_tokens=10,
    )
    message["role"] = "assistant"
    required.append("MUTATED")

    assert request.messages[0]["role"] == "user"
    assert request.messages[0]["content"] == "BLOCK-CANARY"
    assert request.response_format == {
        "type": "json_schema",
        "json_schema": {"required": ("rewritten_prompt",)},
    }
    with pytest.raises(dataclasses.FrozenInstanceError):
        request.max_output_tokens = 11  # type: ignore[misc]


def test_auxiliary_request_preserves_exact_text_and_freezes_json_sequences() -> None:
    content = "\n  Preserve this spacing exactly.  \t"
    enum_values = ["alpha", {"nested": [True, None, 3, 1.25]}]

    request = AuxiliaryCompletionRequest(
        resolution=_auxiliary_resolution(),
        messages=({"role": "user", "content": content},),
        response_format={"schema": {"enum": enum_values}},
        max_output_tokens=10,
    )
    enum_values[1]["nested"].append("MUTATED")

    assert request.messages[0]["content"] == content
    assert request.response_format == {
        "schema": {"enum": ("alpha", {"nested": (True, None, 3, 1.25)})}
    }


@pytest.mark.parametrize(
    "response_format",
    [
        {"schema": {"bad": {"set-value"}}},
        {"schema": {"bad": object()}},
        {"schema": {"bad": b"bytes"}},
        {"schema": {"bad": range(2)}},
        {"schema": {"bad": float("nan")}},
        {"schema": {"bad": float("inf")}},
        {1: "non-string-key"},
    ],
)
def test_auxiliary_request_rejects_nested_non_json_values(response_format) -> None:
    with pytest.raises((TypeError, ValueError)):
        AuxiliaryCompletionRequest(
            resolution=_auxiliary_resolution(),
            messages=({"role": "user", "content": "x"},),
            response_format=response_format,
            max_output_tokens=10,
        )


def test_auxiliary_contract_repr_omits_sensitive_request_and_response_fields() -> None:
    request = _auxiliary_request()
    result = AuxiliaryCompletionResult(
        provider="OpenAI",
        model="gpt-test",
        text="RESPONSE-CANARY",
    )

    rendered = repr((request, result))

    assert "ENDPOINT-CANARY" not in rendered
    assert "API-KEY-CANARY" not in rendered
    assert "OPTIMIZER-CANARY" not in rendered
    assert "USER-CANARY" not in rendered
    assert "RESPONSE-CANARY" not in rendered


@pytest.mark.parametrize(
    "overrides",
    [
        {"resolution": _auxiliary_resolution(ready=False)},
        {"resolution": _auxiliary_resolution(model="")},
        {"messages": []},
        {"messages": ({"role": "user"},)},
        {"messages": ({"role": "", "content": "x"},)},
        {"messages": ({"role": "user", "content": ["image"]},)},
        {"max_output_tokens": 0},
        {"max_output_tokens": MAX_AUXILIARY_OUTPUT_TOKENS + 1},
        {"sensitive": False},
    ],
)
def test_auxiliary_request_rejects_invalid_contract(overrides) -> None:
    values = {
        "resolution": _auxiliary_resolution(),
        "messages": ({"role": "user", "content": "x"},),
        "response_format": None,
        "max_output_tokens": 10,
    }
    values.update(overrides)

    with pytest.raises((TypeError, ValueError)):
        AuxiliaryCompletionRequest(**values)


def test_auxiliary_output_cap_matches_prompt_improvement_application_limit() -> None:
    assert MAX_AUXILIARY_OUTPUT_TOKENS == 16_384


def test_auxiliary_output_cap_accepts_boundary_and_rejects_one_over() -> None:
    request = _auxiliary_request(max_output_tokens=16_384)

    assert request.max_output_tokens == 16_384
    with pytest.raises(ValueError, match="between 1 and 16384"):
        _auxiliary_request(max_output_tokens=16_385)


@pytest.mark.asyncio
async def test_auxiliary_completion_is_one_shot_nonstreaming_and_tool_free() -> None:
    calls: list[dict[str, object]] = []
    sensitive_states: list[bool] = []

    def fake_chat_api_call(**kwargs):
        sensitive_states.append(is_sensitive_llm_request())
        calls.append(kwargs)
        return '  {"kind":"prompt_rewrite","rewritten_prompt":"Better"}\n'

    gateway = ConsoleProviderGateway(chat_api_call_fn=fake_chat_api_call)

    result = await gateway.complete_auxiliary(_auxiliary_request())

    assert result == AuxiliaryCompletionResult(
        provider="OpenAI",
        model="gpt-test",
        text='  {"kind":"prompt_rewrite","rewritten_prompt":"Better"}\n',
    )
    assert len(calls) == 1
    assert sensitive_states == [True]
    call = calls[0]
    assert call == {
        "api_endpoint": "openai",
        "api_base_url": "https://api.example.test/v1?token=ENDPOINT-CANARY",
        "system_message": "OPTIMIZER-CANARY",
        "messages_payload": [{"role": "user", "content": "USER-CANARY"}],
        "api_key": "API-KEY-CANARY",
        "model": "gpt-test",
        "streaming": False,
        "temp": 0.2,
        "topp": 0.8,
        "maxp": 0.8,
        "topk": 17,
        "minp": 0.03,
        "max_tokens": 321,
        "seed": 42,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.2,
        "reasoning_effort": "high",
        "reasoning_summary": "auto",
        "verbosity": "low",
        "thinking_effort": "medium",
        "thinking_budget_tokens": 2048,
        "response_format": {"type": "json_object"},
    }
    assert not ({"tools", "tool_choice", "stop", "history", "images"} & call.keys())
    assert is_sensitive_llm_request() is False


@pytest.mark.asyncio
async def test_auxiliary_completion_preserves_exact_empty_string() -> None:
    gateway = ConsoleProviderGateway(chat_api_call_fn=lambda **_kwargs: "")

    result = await gateway.complete_auxiliary(_auxiliary_request())

    assert result.text == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "unsupported",
    [None, (), [], iter(["chunk"]), {"unexpected": "shape"}],
)
async def test_auxiliary_completion_rejects_unsupported_response_shapes(
    unsupported,
) -> None:
    gateway = ConsoleProviderGateway(chat_api_call_fn=lambda **_kwargs: unsupported)

    with pytest.raises(ChatProviderError, match="unsupported auxiliary response"):
        await gateway.complete_auxiliary(_auxiliary_request())


@pytest.mark.asyncio
async def test_auxiliary_completion_accepts_standard_provider_mapping_exactly() -> None:
    gateway = ConsoleProviderGateway(
        chat_api_call_fn=lambda **_kwargs: {
            "choices": [{"message": {"content": " exact mapping text \n"}}],
            "usage": {
                "prompt_tokens": 13,
                "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 3},
            },
        }
    )

    result = await gateway.complete_auxiliary(_auxiliary_request())

    assert result.text == " exact mapping text \n"
    assert result.usage is not None
    assert result.usage.uncached_input == 10
    assert result.usage.cache_read == 3
    assert result.usage.output == 5


@pytest.mark.asyncio
async def test_auxiliary_completion_redacts_provider_exception_and_resets_context() -> (
    None
):
    def fail(**_kwargs):
        assert is_sensitive_llm_request() is True
        raise RuntimeError("EXCEPTION-CANARY")

    gateway = ConsoleProviderGateway(chat_api_call_fn=fail)

    with pytest.raises(ChatProviderError) as exc_info:
        await gateway.complete_auxiliary(_auxiliary_request())

    assert "EXCEPTION-CANARY" not in str(exc_info.value)
    assert is_sensitive_llm_request() is False


@pytest.mark.asyncio
async def test_auxiliary_completion_ignores_injected_raw_error_formatter() -> None:
    def fail(**_kwargs):
        raise RuntimeError("EXCEPTION-CANARY")

    gateway = ConsoleProviderGateway(
        chat_api_call_fn=fail,
        safe_error_copy=lambda _provider, exc: str(exc),
    )

    with pytest.raises(ChatProviderError) as exc_info:
        await gateway.complete_auxiliary(_auxiliary_request())

    assert "EXCEPTION-CANARY" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_auxiliary_completion_cancellation_starts_no_second_call_and_resets() -> (
    None
):
    started = threading.Event()
    release = threading.Event()
    calls = 0
    observed: list[bool] = []

    def blocking(**_kwargs):
        nonlocal calls
        calls += 1
        observed.append(is_sensitive_llm_request())
        started.set()
        release.wait(timeout=2)
        observed.append(is_sensitive_llm_request())
        return "late"

    gateway = ConsoleProviderGateway(chat_api_call_fn=blocking)
    task = asyncio.create_task(gateway.complete_auxiliary(_auxiliary_request()))
    await asyncio.to_thread(started.wait, 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    await asyncio.sleep(0.05)

    assert calls == 1
    assert observed == [True, True]
    assert is_sensitive_llm_request() is False


@pytest.mark.asyncio
async def test_auxiliary_direct_llama_is_nonstreaming_exact_and_sensitive() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["payload"] = json.loads(request.content)
        captured["sensitive"] = is_sensitive_llm_request()
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": " llama exact \n"}}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    gateway = ConsoleProviderGateway(http_client=client)
    request = _auxiliary_request(
        resolution=_auxiliary_resolution(
            provider="llama_cpp",
            execution_key="llama_cpp",
            readiness_key="llama_cpp",
            base_url="http://127.0.0.1:9099/v1",
        )
    )

    result = await gateway.complete_auxiliary(request)

    assert result.text == " llama exact \n"
    assert captured["url"] == "http://127.0.0.1:9099/v1/chat/completions"
    assert captured["sensitive"] is True
    # Auxiliary requests inherit session thinking settings (ADR-066): level
    # via chat_template_kwargs, budget via top-level
    # reasoning_budget_tokens.
    assert captured["payload"] == {
        "model": "gpt-test",
        "messages": [
            {"role": "system", "content": "OPTIMIZER-CANARY"},
            {"role": "user", "content": "USER-CANARY"},
        ],
        "stream": False,
        "temperature": 0.2,
        "top_p": 0.8,
        "min_p": 0.03,
        "top_k": 17,
        "max_tokens": 321,
        "seed": 42,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.2,
        "chat_template_kwargs": {"reasoning_effort": "high"},
        "reasoning_budget_tokens": 2048,
    }
    await client.aclose()


@pytest.mark.asyncio
async def test_auxiliary_direct_llama_rejects_malformed_completion_shape() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json={"unexpected": "shape"})
        )
    )
    gateway = ConsoleProviderGateway(http_client=client)
    request = _auxiliary_request(
        resolution=_auxiliary_resolution(
            provider="llama_cpp",
            execution_key="llama_cpp",
            readiness_key="llama_cpp",
            base_url="http://127.0.0.1:9099/v1",
        )
    )

    with pytest.raises(ChatProviderError):
        await gateway.complete_auxiliary(request)

    await client.aclose()


# -- PR3a-1 Task 6b (audit F5, first half) ---------------------------------
#
# `aclose()`'s stale sweep skips only loops that are already `is_closed()`.
# A fleet child owns a `_ModelCallLifeline` -- an event loop plus the one
# thread driving `run_forever` -- for as long as the CHILD lives, which
# (PR3a-1) can be well past the turn that spawned it. That loop is running,
# not closed, so the sweep handed its client to `_schedule_stale_client_
# close`, which closes the pool ON THE CHILD'S OWN LOOP, mid-request.
#
# Not fleet-introduced -- `run_reply` is dispatched via `asyncio.to_thread`,
# which survives Task cancellation, so `on_unmount`'s `aclose()` (called
# AFTER `controller.shutdown()`, which only cancels Tasks) could already
# reach the primary. What this PR changes is the population: one loop
# becomes one per live child.


def _running_child_loop() -> tuple[asyncio.AbstractEventLoop, threading.Thread]:
    """A `_ModelCallLifeline`-shaped loop: `run_forever` on its own thread."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    return loop, thread


def test_aclose_does_not_close_a_still_running_childs_client():
    gateway = ConsoleProviderGateway()
    child_loop, child_thread = _running_child_loop()

    async def touch() -> httpx.AsyncClient:
        return gateway._active_http_client()

    try:
        child_client = asyncio.run_coroutine_threadsafe(touch(), child_loop).result(5)
        assert child_client.is_closed is False

        async def close_from_the_app_loop() -> None:
            await gateway.aclose()

        asyncio.run(close_from_the_app_loop())

        # The scheduled close runs on the CHILD's loop, so give that loop
        # real time to execute it -- asserting immediately would pass
        # against the bug by simply outrunning it.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not child_client.is_closed:
            time.sleep(0.02)

        assert child_client.is_closed is False, (
            "aclose() closed a live child's connection pool on the child's "
            "own loop, mid-request"
        )
        # ... and the child keeps the SAME pool, rather than silently
        # rebuilding one per call for the rest of its life.
        again = asyncio.run_coroutine_threadsafe(touch(), child_loop).result(5)
        assert again is child_client
    finally:
        try:
            asyncio.run_coroutine_threadsafe(child_client.aclose(), child_loop).result(
                5
            )
        except Exception:  # noqa: BLE001 -- teardown best-effort
            pass
        child_loop.call_soon_threadsafe(child_loop.stop)
        child_thread.join(5)
        child_loop.close()


def test_aclose_still_sweeps_a_finished_turns_idle_loop():
    """The control: a per-turn loop that has STOPPED (its lifeline shut it
    down) is still swept -- narrowing the sweep to skip live loops must not
    turn it into a no-op."""
    gateway = ConsoleProviderGateway()
    idle_loop = asyncio.new_event_loop()

    async def touch() -> None:
        gateway._active_http_client()

    idle_loop.run_until_complete(touch())
    idle_client = gateway._loop_clients[idle_loop]
    assert idle_client.is_closed is False
    assert idle_loop.is_running() is False

    scheduled: list = []
    original = ConsoleProviderGateway._schedule_stale_client_close
    try:
        ConsoleProviderGateway._schedule_stale_client_close = staticmethod(
            lambda client, loop: scheduled.append((client, loop))
        )

        async def close_from_the_app_loop() -> None:
            await gateway.aclose()

        asyncio.run(close_from_the_app_loop())
    finally:
        ConsoleProviderGateway._schedule_stale_client_close = original
        idle_loop.run_until_complete(idle_client.aclose())
        idle_loop.close()

    assert scheduled == [(idle_client, idle_loop)]


class TestSignalsExchangeCapture:
    @staticmethod
    def _begin(call, label="hi"):
        call.begin_exchange(
            provider="anthropic", model="m", endpoint=None,
            request={"messages_payload": [{"role": "user", "content": label}]},
            omitted_keys=("api_key",),
        )

    def test_per_call_boundaries_never_merge(self):
        aggregate = ConsoleProviderStreamSignals(exchange_capture_enabled=True)
        call0 = aggregate.new_usage_call()
        self._begin(call0, "call0")
        call0.record_exchange_content("hel")
        call0.record_exchange_content("lo")
        call0.close_exchange()
        call1 = aggregate.new_usage_call()
        self._begin(call1, "call1")
        call1.record_exchange_content("again")
        call1.close_exchange()
        captures = aggregate.exchange_captures()
        assert [c.seq for c in captures] == [0, 1]
        assert captures[0].response["content"] == "hello"
        assert captures[1].response["content"] == "again"
        assert captures[0].run_tag == captures[1].run_tag == aggregate.run_tag

    @pytest.mark.parametrize(
        "chunks",
        [
            ("data:image/png;base64,", "QUJD" * 450, "QUJD" * 450, "QUJD" * 450),
            ("QUJD" * 450, "QUJD" * 450, "QUJD" * 450),
        ],
        ids=["split-data-uri", "split-plain-base64"],
    )
    def test_final_aggregate_stubs_binary_split_across_small_chunks(self, chunks):
        assert all(len(chunk) < 4096 for chunk in chunks)
        aggregate = ConsoleProviderStreamSignals(
            exchange_capture_enabled=True,
            capture_detail=CaptureDetail.FULL,
        )
        call = aggregate.new_usage_call()
        self._begin(call)
        for chunk in chunks:
            call.record_exchange_content(chunk)
        call.close_exchange()

        content = aggregate.exchange_captures()[0].response["content"]

        assert content.startswith("[")
        assert "sha256:" in content
        assert "QUJD" not in content

    def test_call_views_inherit_one_frozen_capture_detail(self):
        aggregate = ConsoleProviderStreamSignals(
            exchange_capture_enabled=True,
            capture_detail=CaptureDetail.FULL,
        )

        assert aggregate.new_usage_call().capture_detail is CaptureDetail.FULL
        assert aggregate.new_usage_call().capture_detail is CaptureDetail.FULL

    def test_exchange_capture_keeps_frozen_detail_and_shared_budget(self):
        aggregate = ConsoleProviderStreamSignals(
            exchange_capture_enabled=True,
            capture_detail=CaptureDetail.FULL,
        )
        call = aggregate.new_usage_call()
        budget = CaptureBudget(limit_bytes=4096)
        request, omitted = build_request_capture(
            {"messages_payload": [{"role": "user", "content": "hello"}]},
            capture_detail=call.capture_detail,
            budget=budget,
        )
        call.begin_exchange(
            provider="anthropic",
            model="m",
            endpoint=None,
            request=request,
            omitted_keys=omitted,
            capture_budget=budget,
        )
        call.record_exchange_content("world")
        call.close_exchange()

        capture = aggregate.exchange_captures()[0]
        assert capture.capture_detail is CaptureDetail.FULL
        assert capture.response["content"] == "world"

    def test_in_flight_projection_is_idempotent_and_accumulation_is_bounded(self):
        aggregate = ConsoleProviderStreamSignals(exchange_capture_enabled=True)
        call = aggregate.new_usage_call()
        budget = CaptureBudget(limit_bytes=180)
        call.begin_exchange(
            provider="anthropic", model="m", endpoint=None,
            request={}, omitted_keys=(), capture_budget=budget,
        )
        for _ in range(20):
            call.record_exchange_content("x" * 40)
            call.record_exchange_tool_calls([
                {"id": "t", "function": {"name": "n", "arguments": "y" * 40}}
            ])

        first = aggregate.exchange_captures()
        used = budget.used_bytes
        second = aggregate.exchange_captures()

        assert first == second
        assert budget.used_bytes == used
        flight = aggregate._active_exchanges[call._token]
        assert len("".join(flight["content"]).encode()) <= budget.limit_bytes
        assert len(str(flight["tool_calls"]).encode()) <= budget.limit_bytes
        assert first[0].response["truncation_inventory"]

    def test_in_flight_tail_reports_stopped(self):
        aggregate = ConsoleProviderStreamSignals(exchange_capture_enabled=True)
        call = aggregate.new_usage_call()
        self._begin(call)
        call.record_exchange_content("part")
        captures = aggregate.exchange_captures()
        assert len(captures) == 1
        assert captures[0].status == "stopped"
        assert captures[0].response["content"] == "part"

    def test_close_moves_never_copies(self):
        aggregate = ConsoleProviderStreamSignals(exchange_capture_enabled=True)
        call = aggregate.new_usage_call()
        self._begin(call)
        call.close_exchange()
        call.close_exchange()  # second close is a no-op
        assert len(aggregate.exchange_captures()) == 1

    def test_tool_calls_recorded(self):
        aggregate = ConsoleProviderStreamSignals(exchange_capture_enabled=True)
        call = aggregate.new_usage_call()
        self._begin(call)
        call.record_exchange_tool_calls([{"id": "t1", "function": {"name": "get_time"}}])
        call.close_exchange()
        assert aggregate.exchange_captures()[0].response["tool_calls"][0]["id"] == "t1"

    def test_tool_calls_recorded_are_deep_not_aliased(self):
        """Review finding M9: ``record_exchange_tool_calls`` used to
        shallow-copy (``dict(c)``), leaving the nested ``function`` dict
        aliased to the caller's live object until the flush reaches
        ``close_exchange``/``_flight_capture`` seconds later on a real
        turn -- mutating the ORIGINAL dict the caller passed in must never
        be visible in the already-recorded capture."""
        aggregate = ConsoleProviderStreamSignals(exchange_capture_enabled=True)
        call = aggregate.new_usage_call()
        self._begin(call)
        live_call = {"id": "t1", "function": {"name": "get_time", "arguments": "{}"}}
        call.record_exchange_tool_calls([live_call])
        # Mutate the caller's own object AFTER recording -- the nested
        # `function` dict, not just the top-level one.
        live_call["function"]["name"] = "MUTATED_AFTER_RECORD"
        live_call["function"]["arguments"] = "MUTATED_AFTER_RECORD"
        call.close_exchange()
        recorded = aggregate.exchange_captures()[0].response["tool_calls"][0]
        assert recorded["function"]["name"] == "get_time"
        assert recorded["function"]["arguments"] == "{}"

    def test_close_attaches_this_calls_normalized_usage(self):
        aggregate = ConsoleProviderStreamSignals(exchange_capture_enabled=True)
        call = aggregate.new_usage_call()
        self._begin(call)
        call.record_usage_payload({"prompt_tokens": 10, "completion_tokens": 5})
        call.close_exchange()
        cap = aggregate.exchange_captures()[0]
        assert cap.usage_json is not None
        from tldw_chatbook.Chat.provider_usage import ProviderUsage
        usage = ProviderUsage.from_json(cap.usage_json)
        assert usage is not None and usage.total_tokens == 15

    def test_disabled_records_nothing(self):
        aggregate = ConsoleProviderStreamSignals(exchange_capture_enabled=False)
        call = aggregate.new_usage_call()
        self._begin(call)
        call.record_exchange_content("x")
        call.close_exchange()
        assert aggregate.exchange_captures() == []

    def test_mutate_scoped_exchange_swallows_exceptions(self):
        """Review finding M4: the low-level mutate call must never raise --
        it is called from THREE sites inside ``_stream_generic_chat``'s
        worker ``try``, whose ``except BaseException`` would otherwise
        relabel a capture-bookkeeping bug as a fabricated provider error."""
        aggregate = ConsoleProviderStreamSignals(exchange_capture_enabled=True)
        call = aggregate.new_usage_call()
        self._begin(call)

        class _BoomList:
            def extend(self, items):
                raise RuntimeError("boom")

        # Corrupt the in-flight record directly to force extend() to raise.
        aggregate._active_exchanges[call._token]["content"] = _BoomList()

        call.record_exchange_content("more text")  # must not raise

    def test_complete_scoped_exchange_swallows_exceptions(self, monkeypatch):
        """Review finding M4: close_exchange's own implementation
        (_complete_scoped_exchange) must never raise -- it is called from
        stream_chat's `finally` AND twice inside `_stream_generic_chat`'s
        worker `try`/`except`. Patches the CONSUMER namespace (the module's
        own `_flight_capture` reference) so the raise happens inside the
        method's try block."""
        aggregate = ConsoleProviderStreamSignals(exchange_capture_enabled=True)
        call = aggregate.new_usage_call()
        self._begin(call)

        def raising_flight_capture(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(gateway_module, "_flight_capture", raising_flight_capture)

        call.close_exchange()  # must not raise
        assert aggregate.exchange_captures() == []

    def test_run_tags_differ_across_signals_objects(self):
        assert ConsoleProviderStreamSignals().run_tag != ConsoleProviderStreamSignals().run_tag


class TestGatewayExchangeCapture:
    @staticmethod
    def _resolution() -> ConsoleProviderResolution:
        return ConsoleProviderResolution(
            provider="openai",
            base_url="https://proxy.example.test/v1",
            model="gpt-4.1",
            ready=True,
            execution_key="openai",
            api_key="k",
            streaming=False,
        )

    @staticmethod
    async def _drain(gen):
        return [chunk async for chunk in gen]

    @pytest.mark.asyncio
    async def test_one_capture_per_call_with_request_and_response(self):
        calls = []

        def fake_chat_api_call(**kwargs):
            calls.append(kwargs)
            return {"choices": [{"message": {"content": "pong"}}]}

        gateway = ConsoleProviderGateway(chat_api_call_fn=fake_chat_api_call)
        signals = ConsoleProviderStreamSignals(exchange_capture_enabled=True)
        resolution = self._resolution()
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "ping"},
        ]
        await self._drain(gateway.stream_chat(resolution, messages, signals=signals))
        await self._drain(gateway.stream_chat(resolution, messages, signals=signals))
        # The fake provider actually received the built kwargs -- not just a
        # dead collection sitting unused.
        assert len(calls) == 2
        assert calls[0]["model"] == "gpt-4.1"
        captures = signals.exchange_captures()
        assert len(captures) == 2
        assert captures[0].status == "complete"
        assert captures[0].request["system_message"] == "sys"
        assert captures[0].request["messages_payload"] == [
            {"role": "user", "content": "ping"}
        ]
        assert "api_key" not in captures[0].request
        assert "api_key" in captures[0].omitted_keys
        assert captures[0].response["content"] == "pong"
        # Review finding M3 control: real provider output is never
        # mislabeled as synthesized fallback copy.
        assert captures[0].response["synthetic_fallback"] is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("detail", "expected"),
        [
            (CaptureDetail.SAFE, "[project instruction body omitted by capture policy"),
            (CaptureDetail.FULL, "project-body"),
        ],
    )
    async def test_generic_capture_threads_detail_to_semantic_request_builder(
        self,
        detail,
        expected,
    ):
        calls = []

        def fake_chat_api_call(**kwargs):
            calls.append(kwargs)
            return {"choices": [{"message": {"content": "pong"}}]}

        signals = ConsoleProviderStreamSignals(
            exchange_capture_enabled=True,
            capture_detail=detail,
        )
        await self._drain(
            ConsoleProviderGateway(chat_api_call_fn=fake_chat_api_call).stream_chat(
                self._resolution(),
                [
                    {
                        "role": "system",
                        "content": "project-body",
                        gateway_module.EPHEMERAL_ORIGIN_KEY: "project_instructions",
                    },
                    {"role": "user", "content": "q"},
                ],
                signals=signals,
            )
        )

        captured = signals.exchange_captures()[0]
        system_content = captured.request["system_message"]
        if detail is CaptureDetail.SAFE:
            assert system_content.startswith(expected)
        else:
            assert system_content == expected
            assert captured.request["messages_payload"] == calls[0]["messages_payload"]
            assert captured.request["system_message"] == calls[0]["system_message"]
        assert captured.capture_detail is detail

    @pytest.mark.asyncio
    async def test_synthetic_fallback_copy_is_stamped_not_silently_recorded(self):
        """Review finding M3: NO_PROVIDER_CONTENT_COPY is locally
        synthesized UI copy, not provider output -- the empty-response turn
        a user opens the inspector to debug must not show that copy as if
        the model said it. The capture stamps response["synthetic_
        fallback"] instead."""

        def fake_chat_api_call(**kwargs):
            return {"choices": [{"message": {"content": ""}}]}

        gateway = ConsoleProviderGateway(chat_api_call_fn=fake_chat_api_call)
        signals = ConsoleProviderStreamSignals(exchange_capture_enabled=True)
        items = await self._drain(
            gateway.stream_chat(
                self._resolution(),
                [{"role": "user", "content": "q"}],
                signals=signals,
            )
        )
        assert items == [NO_PROVIDER_CONTENT_COPY]
        captures = signals.exchange_captures()
        assert len(captures) == 1
        assert captures[0].response["content"] == NO_PROVIDER_CONTENT_COPY
        assert captures[0].response["synthetic_fallback"] is True

    @pytest.mark.asyncio
    async def test_transcript_output_byte_identical_with_capture(self):
        def fake_chat_api_call(**kwargs):
            return {"choices": [{"message": {"content": "exact bytes"}}]}

        resolution = self._resolution()
        messages = [{"role": "user", "content": "q"}]
        with_signals = await self._drain(
            ConsoleProviderGateway(chat_api_call_fn=fake_chat_api_call).stream_chat(
                resolution, messages,
                signals=ConsoleProviderStreamSignals(exchange_capture_enabled=True),
            )
        )
        without = await self._drain(
            ConsoleProviderGateway(chat_api_call_fn=fake_chat_api_call).stream_chat(
                resolution, messages, signals=None
            )
        )
        assert with_signals == without

    @pytest.mark.asyncio
    async def test_provider_error_closes_capture_as_error(self):
        def fake_chat_api_call(**kwargs):
            raise RuntimeError("boom")

        gateway = ConsoleProviderGateway(chat_api_call_fn=fake_chat_api_call)
        signals = ConsoleProviderStreamSignals(exchange_capture_enabled=True)
        # Narrowed from bare Exception: a capture-code explosion must not be
        # mistaken for the provider error this test actually targets.
        with pytest.raises(ChatProviderError):
            await self._drain(
                gateway.stream_chat(
                    self._resolution(),
                    [{"role": "user", "content": "q"}],
                    signals=signals,
                )
            )
        captures = signals.exchange_captures()
        assert len(captures) == 1 and captures[0].status == "error"

    @pytest.mark.asyncio
    async def test_disabled_signals_capture_nothing(self):
        def fake_chat_api_call(**kwargs):
            return {"choices": [{"message": {"content": "pong"}}]}

        gateway = ConsoleProviderGateway(chat_api_call_fn=fake_chat_api_call)
        signals = ConsoleProviderStreamSignals(exchange_capture_enabled=False)
        await self._drain(
            gateway.stream_chat(
                self._resolution(),
                [{"role": "user", "content": "q"}],
                signals=signals,
            )
        )
        assert signals.exchange_captures() == []

    @pytest.mark.asyncio
    async def test_disabled_capture_never_calls_build_request_capture(
        self, monkeypatch
    ):
        """Review finding I1: ``begin_exchange``'s own early-return guard
        only skips STORING the capture -- with capture off, the caller must
        never even CALL ``build_request_capture`` (a full ``json.dumps`` of
        ``messages_payload`` plus a recursive ``stub_binary_strings`` walk)
        in the first place. Patches the CONSUMER namespace and proves the
        patch took with a call counter, same idiom as
        ``test_never_break_send_when_build_request_capture_raises``."""
        call_count = {"n": 0}
        original = gateway_module.build_request_capture

        def counting_build_request_capture(kwargs):
            call_count["n"] += 1
            return original(kwargs)

        monkeypatch.setattr(
            gateway_module, "build_request_capture", counting_build_request_capture
        )

        def fake_chat_api_call(**kwargs):
            return {"choices": [{"message": {"content": "pong"}}]}

        gateway = ConsoleProviderGateway(chat_api_call_fn=fake_chat_api_call)
        signals = ConsoleProviderStreamSignals(exchange_capture_enabled=False)
        items = await self._drain(
            gateway.stream_chat(
                self._resolution(),
                [{"role": "user", "content": "q"}],
                signals=signals,
            )
        )
        assert items == ["pong"]
        assert call_count["n"] == 0
        assert signals.exchange_captures() == []

    @pytest.mark.asyncio
    async def test_not_ready_resolution_emits_no_phantom_capture(self):
        """The early ``not resolution.ready`` return never reaches the
        worker, so no exchange ever begins -- confirm the finally's
        close_exchange() no-ops instead of fabricating an empty capture."""

        def fake_chat_api_call(**kwargs):
            pytest.fail("chat_api_call must not run for a not-ready resolution")

        gateway = ConsoleProviderGateway(chat_api_call_fn=fake_chat_api_call)
        signals = ConsoleProviderStreamSignals(exchange_capture_enabled=True)
        not_ready = dataclasses.replace(self._resolution(), ready=False)
        await self._drain(
            gateway.stream_chat(
                not_ready, [{"role": "user", "content": "q"}], signals=signals
            )
        )
        assert signals.exchange_captures() == []

    @pytest.mark.asyncio
    async def test_no_content_no_tool_calls_closes_capture_as_error(self):
        """The 'Provider returned no content and no tool calls' route is a
        real send failure (PR #648 review Minor 1) and must close its
        exchange as 'error', not the finally's default 'complete'."""

        def fake_chat_api_call(**kwargs):
            return {"choices": [{"message": {"content": ""}}]}

        gateway = ConsoleProviderGateway(chat_api_call_fn=fake_chat_api_call)
        signals = ConsoleProviderStreamSignals(exchange_capture_enabled=True)
        with pytest.raises(ChatProviderError):
            await self._drain(
                gateway.stream_chat(
                    self._resolution(),
                    [{"role": "user", "content": "q"}],
                    tools=TOOLS,
                    signals=signals,
                )
            )
        captures = signals.exchange_captures()
        assert len(captures) == 1 and captures[0].status == "error"

    @pytest.mark.asyncio
    async def test_consumer_abort_mid_stream_closes_capture_as_stopped(self):
        """A user Stop/cancel mid-stream (consumer calls aclose()) must
        close the exchange as 'stopped', keeping the partial content that
        was already recorded -- never silently upgraded to 'complete'."""

        class _BlockAfterFirstChunk:
            """Yields one chunk, then blocks until close() releases it --
            deterministically pins the worker mid-stream so the second
            chunk can never reach the queue before the consumer aborts."""

            def __init__(self) -> None:
                self._state = 0
                self._released = threading.Event()

            def __iter__(self):
                return self

            def __next__(self):
                if self._state == 0:
                    self._state = 1
                    return {"choices": [{"delta": {"content": "he"}}]}
                self._released.wait(timeout=5)
                raise StopIteration

            def close(self) -> None:
                self._released.set()

        iterator = _BlockAfterFirstChunk()

        def fake_chat_api_call(**kwargs):
            return iterator

        gateway = ConsoleProviderGateway(chat_api_call_fn=fake_chat_api_call)
        signals = ConsoleProviderStreamSignals(exchange_capture_enabled=True)
        gen = gateway.stream_chat(
            self._resolution(), [{"role": "user", "content": "q"}], signals=signals
        )
        first = await anext(gen)
        assert first == "he"
        await gen.aclose()

        captures = signals.exchange_captures()
        assert len(captures) == 1
        assert captures[0].status == "stopped"
        assert captures[0].response["content"] == "he"

    @pytest.mark.asyncio
    async def test_openai_stop_closes_transport_through_gateway_boundary(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Console cancellation closes the real OpenAI adapter without a tail yield.

        HTTP alone is isolated: this drives the production gateway,
        ``chat_api_call``, and ``chat_with_openai`` generator. Blocking capture
        publication after the first provider delta leaves that generator
        suspended at its content yield, so cancelling the public async stream
        deterministically exercises the same ``GeneratorExit`` cleanup path as
        Console Stop.
        """

        class _StreamingResponse:
            status_code = 200
            text = ""

            def __init__(self) -> None:
                self.closed = threading.Event()

            def __bool__(self) -> bool:
                return True

            def raise_for_status(self) -> None:
                return None

            def iter_lines(self, *, decode_unicode: bool):
                assert decode_unicode is True
                yield 'data: {"choices": [{"delta": {"content": "he"}}]}'
                raise AssertionError("stopped stream requested another HTTP chunk")

            def close(self) -> None:
                self.closed.set()

        response = _StreamingResponse()

        def fake_post(_session, *_args, **_kwargs):
            return response

        monkeypatch.setattr("requests.Session.post", fake_post)

        signals = ConsoleProviderStreamSignals(exchange_capture_enabled=True)
        content_recorded = threading.Event()
        release_worker = threading.Event()
        original_record = ConsoleProviderStreamSignals._mutate_scoped_exchange

        def blocking_record(self, token, key, items):
            original_record(self, token, key, items)
            content_recorded.set()
            if not release_worker.wait(timeout=5):
                raise TimeoutError("test did not release provider worker")

        monkeypatch.setattr(
            ConsoleProviderStreamSignals,
            "_mutate_scoped_exchange",
            blocking_record,
        )
        stream = ConsoleProviderGateway().stream_chat(
            dataclasses.replace(self._resolution(), streaming=True),
            [{"role": "user", "content": "q"}],
            signals=signals,
        )
        pending = asyncio.create_task(anext(stream))
        try:
            assert await asyncio.to_thread(content_recorded.wait, 5)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending

            assert response.closed.wait(timeout=1)
            captures = signals.exchange_captures()
            assert len(captures) == 1
            assert captures[0].status == "stopped"
            assert captures[0].response["content"] == "he"
        finally:
            release_worker.set()
            if not pending.done():
                pending.cancel()
            await stream.aclose()

    @pytest.mark.asyncio
    async def test_native_tool_calls_recorded_in_capture(self):
        response = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "t1",
                                "type": "function",
                                "function": {
                                    "name": "get_time",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    }
                }
            ]
        }

        def fake_chat_api_call(**kwargs):
            return response

        gateway = ConsoleProviderGateway(chat_api_call_fn=fake_chat_api_call)
        signals = ConsoleProviderStreamSignals(exchange_capture_enabled=True)
        items = await self._drain(
            gateway.stream_chat(
                self._resolution(),
                [{"role": "user", "content": "q"}],
                tools=TOOLS,
                signals=signals,
            )
        )
        (ptc,) = [i for i in items if isinstance(i, ProviderToolCalls)]
        captures = signals.exchange_captures()
        assert len(captures) == 1
        assert captures[0].response["tool_calls"] == list(ptc.tool_calls)

    @pytest.mark.asyncio
    async def test_never_break_send_when_build_request_capture_raises(
        self, monkeypatch
    ):
        """A capture-path bug (begin_exchange's own try/except) must never
        block a send. Patches the CONSUMER namespace -- the gateway
        module's imported binding, which is what `worker()` actually calls
        -- and proves the patch took with a call counter."""
        call_count = {"n": 0}

        def raising_build_request_capture(kwargs, **_options):
            call_count["n"] += 1
            raise RuntimeError("capture exploded")

        monkeypatch.setattr(
            gateway_module, "build_request_capture", raising_build_request_capture
        )

        def fake_chat_api_call(**kwargs):
            return {"choices": [{"message": {"content": "pong"}}]}

        gateway = ConsoleProviderGateway(chat_api_call_fn=fake_chat_api_call)
        signals = ConsoleProviderStreamSignals(exchange_capture_enabled=True)
        items = await self._drain(
            gateway.stream_chat(
                self._resolution(),
                [{"role": "user", "content": "q"}],
                signals=signals,
            )
        )
        assert items == ["pong"]
        assert call_count["n"] == 1
        assert signals.exchange_captures() == []


class TestLlamaCppExchangeCapture:
    @staticmethod
    def _resolution(*, streaming: bool) -> ConsoleProviderResolution:
        return ConsoleProviderResolution(
            provider="llama_cpp",
            base_url="http://127.0.0.1:9099",
            model="m",
            ready=True,
            execution_key="llama_cpp",
            api_key="local-secret",
            streaming=streaming,
        )

    @pytest.mark.asyncio
    async def test_llamacpp_capture_is_wire_literal_and_keyless(self, monkeypatch):
        import json as _json

        gateway = ConsoleProviderGateway()
        streamed = ["hel", "lo"]

        async def fake_stream(self, **kwargs):
            for chunk in streamed:
                yield chunk

        monkeypatch.setattr(ConsoleProviderGateway, "stream_llamacpp_chat", fake_stream)
        aggregate = ConsoleProviderStreamSignals(exchange_capture_enabled=True)
        resolution = self._resolution(streaming=True)
        out = [
            c
            async for c in gateway.stream_chat(
                resolution, [{"role": "user", "content": "q"}], signals=aggregate
            )
        ]
        assert out == streamed
        captures = aggregate.exchange_captures()
        assert len(captures) == 1
        wire = captures[0].request["wire_payload"]
        assert wire["messages"][-1]["content"] == "q"
        assert captures[0].response["content"] == "hello"
        # resolution.api_key rides stream_llamacpp_chat's kwargs (headers),
        # never the wire body -- the capture must contain no trace of it.
        assert "local-secret" not in _json.dumps(captures[0].request)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("detail", "expected"),
        [
            (CaptureDetail.SAFE, "[project instruction body omitted by capture policy"),
            (CaptureDetail.FULL, "project-body"),
        ],
    )
    async def test_llamacpp_capture_applies_frozen_detail_to_wire_payload(
        self,
        monkeypatch,
        detail,
        expected,
    ):
        async def fake_stream(self, **kwargs):
            yield "ok"

        monkeypatch.setattr(ConsoleProviderGateway, "stream_llamacpp_chat", fake_stream)
        signals = ConsoleProviderStreamSignals(
            exchange_capture_enabled=True,
            capture_detail=detail,
        )
        messages = [
            {
                "role": "system",
                "content": "project-body",
                gateway_module.EPHEMERAL_ORIGIN_KEY: "project_instructions",
            },
            {"role": "user", "content": "q"},
        ]

        _ = [
            item
            async for item in ConsoleProviderGateway().stream_chat(
                self._resolution(streaming=True), messages, signals=signals
            )
        ]

        captured = signals.exchange_captures()[0]
        content = captured.request["wire_payload"]["messages"][0]["content"]
        if detail is CaptureDetail.SAFE:
            assert content.startswith(expected)
        else:
            assert content == expected
        assert captured.capture_detail is detail

    @pytest.mark.asyncio
    @pytest.mark.parametrize("detail", [CaptureDetail.SAFE, CaptureDetail.FULL])
    async def test_llamacpp_wire_capture_sanitizes_credentials_and_short_binary(
        self, monkeypatch, detail
    ):
        async def fake_stream(self, **kwargs):
            yield "ok"

        monkeypatch.setattr(ConsoleProviderGateway, "stream_llamacpp_chat", fake_stream)
        signals = ConsoleProviderStreamSignals(
            exchange_capture_enabled=True, capture_detail=detail
        )
        messages = [{
            "role": "user",
            "content": [{
                "api_key": "secret", "access_token": "token",
                "client_secret": "hidden", "data": "QUJD",
                "image_url": "data:image/png;base64,QUJD",
            }],
        }]
        _ = [item async for item in ConsoleProviderGateway().stream_chat(
            self._resolution(streaming=True), messages, signals=signals
        )]

        serialized = json.dumps(signals.exchange_captures()[0].request)
        assert "secret" not in serialized
        assert "token" not in serialized
        assert "hidden" not in serialized
        assert "QUJD" not in serialized
        assert "sha256:" in serialized

    @pytest.mark.asyncio
    async def test_llamacpp_non_streaming_capture_is_wire_literal_and_keyless(
        self, monkeypatch
    ):
        import json as _json

        gateway = ConsoleProviderGateway()

        async def fake_complete(self, **kwargs):
            return "done"

        monkeypatch.setattr(
            ConsoleProviderGateway, "complete_llamacpp_chat", fake_complete
        )
        aggregate = ConsoleProviderStreamSignals(exchange_capture_enabled=True)
        resolution = self._resolution(streaming=False)
        out = [
            c
            async for c in gateway.stream_chat(
                resolution, [{"role": "user", "content": "q"}], signals=aggregate
            )
        ]
        assert out == ["done"]
        captures = aggregate.exchange_captures()
        assert len(captures) == 1
        wire = captures[0].request["wire_payload"]
        assert wire["messages"][-1]["content"] == "q"
        assert wire["stream"] is False
        assert captures[0].response["content"] == "done"
        assert "local-secret" not in _json.dumps(captures[0].request)

    @pytest.mark.asyncio
    async def test_llamacpp_stream_to_complete_fallback_gets_its_own_capture(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """task-19324: the stream->complete retry is a SECOND HTTP request.

        It is issued inside ``stream_llamacpp_chat``, below the seam that
        captures ``stream_chat``'s own call, so before this it never got a
        capture and the Inspector showed one row for a turn that really
        made two calls -- understating what was sent on exactly the
        degraded turn a user opens the Inspector to look at.

        Drives the REAL ``stream_llamacpp_chat`` (only the HTTP layer and
        the non-streaming retry are faked) so the fallback genuinely fires.
        """
        import json as _json

        class _EmptyStreamResponse:
            def raise_for_status(self):
                return None

            async def aiter_lines(self):
                # A stream that opens fine and yields no content is exactly
                # what triggers the non-streaming retry.
                return
                yield  # pragma: no cover - generator marker

        class _StreamCtx:
            async def __aenter__(self):
                return _EmptyStreamResponse()

            async def __aexit__(self, *exc):
                return False

        class _FakeClient:
            def stream(self, *args, **kwargs):
                return _StreamCtx()

        gateway = ConsoleProviderGateway()
        monkeypatch.setattr(
            ConsoleProviderGateway,
            "_active_http_client",
            lambda self: _FakeClient(),
        )

        async def fake_complete(self, **kwargs):
            return "recovered text"

        monkeypatch.setattr(
            ConsoleProviderGateway, "complete_llamacpp_chat", fake_complete
        )

        aggregate = ConsoleProviderStreamSignals(exchange_capture_enabled=True)
        resolution = self._resolution(streaming=True)
        out = [
            c
            async for c in gateway.stream_chat(
                resolution,
                [{"role": "user", "content": {
                    "text": "q", "api_key": "retry-secret", "data": "QUJD"
                }}],
                signals=aggregate,
            )
        ]
        assert out == ["recovered text"]

        captures = aggregate.exchange_captures()
        assert len(captures) == 2, (
            "the streaming call and its non-streaming retry are two HTTP "
            f"requests and must be two captures, got {len(captures)}"
        )
        retry = [c for c in captures if "retry_of" in c.request]
        assert len(retry) == 1
        retry_capture = retry[0]
        assert retry_capture.request["wire_payload"]["stream"] is False
        retry_content = retry_capture.request["wire_payload"]["messages"][-1]["content"]
        assert retry_content["text"] == "q"
        assert "api_key" not in retry_content
        assert "sha256:" in retry_content["data"]
        assert retry_capture.response["content"] == "recovered text"
        # Same keyless guarantee the sibling captures hold.
        assert "local-secret" not in _json.dumps(retry_capture.request)
        assert "retry-secret" not in _json.dumps(retry_capture.request)

    @pytest.mark.asyncio
    async def test_llamacpp_failed_fallback_capture_records_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _EmptyStreamResponse:
            def raise_for_status(self):
                return None

            async def aiter_lines(self):
                return
                yield  # pragma: no cover - generator marker

        class _StreamCtx:
            async def __aenter__(self):
                return _EmptyStreamResponse()

            async def __aexit__(self, *_exc):
                return False

        class _FakeClient:
            def stream(self, *_args, **_kwargs):
                return _StreamCtx()

        gateway = ConsoleProviderGateway()
        monkeypatch.setattr(
            ConsoleProviderGateway,
            "_active_http_client",
            lambda self: _FakeClient(),
        )
        event = ProviderThinkingDelta(
            text="captured",
            provider="llama_cpp",
            model="m",
            protocol="chat_completions",
            source_format="start_anchored_think",
        )

        async def fake_complete(self, **_kwargs):
            return gateway_module._LocalCompletionResult(
                items=(event,), capture_failed=True
            )

        monkeypatch.setattr(
            ConsoleProviderGateway, "complete_llamacpp_chat", fake_complete
        )
        aggregate = ConsoleProviderStreamSignals(exchange_capture_enabled=True)
        stream = gateway.stream_chat(
            self._resolution(streaming=True),
            [{"role": "user", "content": "q"}],
            signals=aggregate,
        )

        assert await anext(stream) is event
        with pytest.raises(ProviderThinkingCaptureError):
            await anext(stream)

        retry = [
            capture
            for capture in aggregate.exchange_captures()
            if "retry_of" in capture.request
        ]
        assert len(retry) == 1
        assert retry[0].status == "error"

    @pytest.mark.asyncio
    async def test_llamacpp_stream_to_complete_fallback_emits_retry_signal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _EmptyStreamResponse:
            def raise_for_status(self):
                return None

            async def aiter_lines(self):
                return
                yield  # pragma: no cover

        class _StreamCtx:
            async def __aenter__(self):
                return _EmptyStreamResponse()

            async def __aexit__(self, *_exc):
                return False

        class _FakeClient:
            def stream(self, *_args, **_kwargs):
                return _StreamCtx()

        gateway = ConsoleProviderGateway()
        monkeypatch.setattr(
            ConsoleProviderGateway,
            "_active_http_client",
            lambda self: _FakeClient(),
        )

        retries: list[str] = []

        async def fake_complete(self, **_kwargs):
            assert retries == ["model_retry"]
            return "recovered"

        monkeypatch.setattr(
            ConsoleProviderGateway, "complete_llamacpp_chat", fake_complete
        )
        signals = ConsoleProviderStreamSignals(
            model_retry_callback=lambda: retries.append("model_retry")
        )
        out = [
            chunk
            async for chunk in gateway.stream_chat(
                self._resolution(streaming=True),
                [{"role": "user", "content": "q"}],
                signals=signals,
            )
        ]
        assert out == ["recovered"]
        assert retries == ["model_retry"]

        def failing_callback() -> None:
            raise RuntimeError("capture callback failed")

        out_with_failed_capture = [
            chunk
            async for chunk in gateway.stream_chat(
                self._resolution(streaming=True),
                [{"role": "user", "content": "q"}],
                signals=ConsoleProviderStreamSignals(
                    model_retry_callback=failing_callback
                ),
            )
        ]
        assert out_with_failed_capture == ["recovered"]

    @pytest.mark.asyncio
    async def test_llamacpp_non_streaming_abort_after_first_item_keeps_recorded_content(
        self, monkeypatch
    ):
        """A consumer that takes the single non-streaming item then closes
        the generator (Stop/cancel) throws GeneratorExit at the suspended
        `yield completion`. Content must already be recorded before that
        yield -- recording after it would be skipped by the abort and the
        resulting 'stopped' tail capture would show empty content even
        though the text was genuinely delivered."""
        gateway = ConsoleProviderGateway()

        async def fake_complete(self, **kwargs):
            return "done"

        monkeypatch.setattr(
            ConsoleProviderGateway, "complete_llamacpp_chat", fake_complete
        )
        aggregate = ConsoleProviderStreamSignals(exchange_capture_enabled=True)
        resolution = self._resolution(streaming=False)
        gen = gateway.stream_chat(
            resolution, [{"role": "user", "content": "q"}], signals=aggregate
        )
        first = await anext(gen)
        assert first == "done"
        await gen.aclose()

        captures = aggregate.exchange_captures()
        assert len(captures) == 1
        assert captures[0].status == "stopped"
        assert captures[0].response["content"] == "done"

    @pytest.mark.asyncio
    async def test_llamacpp_non_streaming_http_failure_closes_capture_as_error(
        self, monkeypatch
    ):
        """Review finding M1: an HTTP failure in the non-streaming llama.cpp
        branch must close the exchange as "error" -- left to the outer
        `finally`, ``completed`` would still be False and it would close as
        "stopped" instead, misreporting a real send failure as a
        user-initiated stop (the generic path already does this correctly
        via its own explicit ``close_exchange(status="error")``)."""
        gateway = ConsoleProviderGateway()

        async def failing_complete(self, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(
            ConsoleProviderGateway, "complete_llamacpp_chat", failing_complete
        )
        aggregate = ConsoleProviderStreamSignals(exchange_capture_enabled=True)
        resolution = self._resolution(streaming=False)
        with pytest.raises(RuntimeError):
            async for _ in gateway.stream_chat(
                resolution, [{"role": "user", "content": "q"}], signals=aggregate
            ):
                pass

        captures = aggregate.exchange_captures()
        assert len(captures) == 1
        assert captures[0].status == "error"

    @pytest.mark.asyncio
    async def test_llamacpp_streaming_http_failure_closes_capture_as_error(
        self, monkeypatch
    ):
        """Review finding M1: same for the streaming branch -- a mid-stream
        HTTP failure must close as "error", keeping whatever partial content
        was already recorded before the failure (contrast with a consumer
        abort, which still closes "stopped" -- see the abort test above)."""
        gateway = ConsoleProviderGateway()

        async def failing_stream(self, **kwargs):
            yield "he"
            raise RuntimeError("boom")

        monkeypatch.setattr(
            ConsoleProviderGateway, "stream_llamacpp_chat", failing_stream
        )
        aggregate = ConsoleProviderStreamSignals(exchange_capture_enabled=True)
        resolution = self._resolution(streaming=True)
        collected = []
        with pytest.raises(RuntimeError):
            async for chunk in gateway.stream_chat(
                resolution, [{"role": "user", "content": "q"}], signals=aggregate
            ):
                collected.append(chunk)

        assert collected == ["he"]
        captures = aggregate.exchange_captures()
        assert len(captures) == 1
        assert captures[0].status == "error"
        assert captures[0].response["content"] == "he"

    @pytest.mark.asyncio
    async def test_disabled_capture_never_builds_wire_payload_for_capture(
        self, monkeypatch
    ):
        """Review finding I1: with capture off, the llama.cpp branch must
        not build a SECOND wire payload purely for capture
        (``build_llamacpp_chat_payload`` at the capture call site) -- the
        real send's own payload build is bypassed here too (the fake
        ``stream_llamacpp_chat`` never calls it), so zero calls proves the
        capture-only build never ran."""
        call_count = {"n": 0}
        original = gateway_module.build_llamacpp_chat_payload

        def counting_build(*args, **kwargs):
            call_count["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(
            gateway_module, "build_llamacpp_chat_payload", counting_build
        )

        gateway = ConsoleProviderGateway()
        streamed = ["hel", "lo"]

        async def fake_stream(self, **kwargs):
            for chunk in streamed:
                yield chunk

        monkeypatch.setattr(ConsoleProviderGateway, "stream_llamacpp_chat", fake_stream)
        aggregate = ConsoleProviderStreamSignals(exchange_capture_enabled=False)
        resolution = self._resolution(streaming=True)
        out = [
            c
            async for c in gateway.stream_chat(
                resolution, [{"role": "user", "content": "q"}], signals=aggregate
            )
        ]
        assert out == streamed
        assert call_count["n"] == 0
        assert aggregate.exchange_captures() == []
