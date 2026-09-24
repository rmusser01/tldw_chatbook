"""Custom-endpoint engine swap (ADR-179 Phase 2, Task 6).

Stage coverage in this file:

- Stage 1 (engine extensions, no gateway touching): the ``CUSTOM_HOSTED``
  registry record, the dispatch entry + ``CUSTOM_PROVIDER_PARAM_MAP``
  (legacy-surface parity), the engine closure's extended parameter surface
  with ``payload_flags`` gating, the ``api_settings.custom`` settings
  fallbacks (section > record defaults; explicit kwargs win), the ADR-066
  custom-row reasoning composition (``reasoning_effort`` verbatim, budget
  accepted-and-dropped), and the continuation ``_PAIRINGS`` additions.
- Stage 2 (shared ``CUSTOM_OPENAI_EXECUTION_KEYS``): see
  ``test_custom_openai_execution_keys_constant.py``.
- Stage 3 (gateway swap + kill switch): appended below.
"""
from __future__ import annotations

import inspect
from typing import get_args

import pytest

from tldw_chatbook.Chat.Chat_Deps import ChatBadRequestError
from tldw_chatbook.Chat.Chat_Functions import API_CALL_HANDLERS, PROVIDER_PARAM_MAP
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationProvider,
    ContinuationRound,
    ContinuationCall,
    ProviderContinuationCheckpoint,
    dump_provider_continuation_json,
    parse_provider_continuation_json,
)
from tldw_chatbook.LLM_Calls.hosted_provider_engine import (
    HostedProviderResolution,
    build_hosted_chat_handler,
    build_hosted_chat_payload,
    resolve_hosted_request,
)
from tldw_chatbook.provider_registry import (
    CUSTOM_HOSTED,
    RECORDS_BY_KEY,
    TOGETHER,
)


def _resolution(**overrides: object) -> HostedProviderResolution:
    values: dict[str, object] = {
        "provider": "custom-hosted",
        "model": "test-model",
        "api_key": "",
        "base_url": "https://custom.example/v1",
        "timeout": 120.0,
        "retries": 1,
        "retry_delay": 1.0,
        "streaming": False,
    }
    values.update(overrides)
    return HostedProviderResolution(**values)  # type: ignore[arg-type]


def _messages() -> list[dict[str, object]]:
    return [{"role": "user", "content": "hi"}]


# --- Stage 1: registry record shape ---


def test_custom_hosted_record_shape() -> None:
    record = CUSTOM_HOSTED
    assert record.key == "custom-hosted"
    assert record.config_key == "Custom-hosted"
    assert record.classification == "local"
    assert record.engine_driven is True
    assert record.default_base_url is None
    assert record.auth_scheme == "bearer_optional"
    assert record.tolerant_response_extras is True
    assert record.native_tools is True
    assert record.auto_refresh is False
    assert record.api_key_env_var is None
    assert record.api_key_env_candidates == ()
    assert record.choice_allowances == frozenset({"logprobs", "stop_reason"})
    assert dict(record.settings_defaults) == {
        "streaming": False,
        "max_tokens": 4096,
        "timeout": 120,
        "retries": 1,
        "retry_delay": 1.0,
    }
    assert record.defaults_settings_section == "custom"
    assert RECORDS_BY_KEY["custom-hosted"] is record


def test_custom_hosted_payload_flags_carry_the_custom_surface() -> None:
    flags = set(CUSTOM_HOSTED.payload_flags)
    assert {
        "temperature",
        "top_p",
        "min_p",
        "top_k",
        "max_tokens",
        "stop",
        "response_format",
        "seed",
        "n",
        "user",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "thinking_budget_tokens",
    } <= flags


def test_custom_hosted_supports_reasoning_effort() -> None:
    assert CUSTOM_HOSTED.reasoning_effort is True


# --- Stage 1: dispatch + param map parity ---


def test_custom_hosted_dispatch_entry_registered() -> None:
    handler = API_CALL_HANDLERS.get("custom-hosted")
    assert callable(handler)


def test_named_custom_slots_untouched() -> None:
    from tldw_chatbook.LLM_Calls.LLM_API_Calls_Local import (
        chat_with_custom_openai,
        chat_with_custom_openai_2,
    )

    assert API_CALL_HANDLERS["custom-openai-api"] is chat_with_custom_openai
    assert API_CALL_HANDLERS["custom-openai-api-2"] is chat_with_custom_openai_2


def test_custom_provider_param_map_equals_legacy_surface() -> None:
    legacy = PROVIDER_PARAM_MAP["custom-openai-api"]
    ours = PROVIDER_PARAM_MAP["custom-hosted"]
    assert set(ours) == set(legacy)
    assert ours == legacy


# --- Stage 1: closure signature / kwargs ---


def test_engine_closure_accepts_the_custom_param_surface() -> None:
    handler = build_hosted_chat_handler(CUSTOM_HOSTED)
    params = set(inspect.signature(handler).parameters)
    expected = {
        "input_data",
        "model",
        "api_key",
        "system_message",
        "temp",
        "maxp",
        "minp",
        "topk",
        "streaming",
        "max_tokens",
        "tools",
        "custom_prompt_arg",
        "api_base_url",
        "tool_choice",
        "stop",
        "response_format",
        "user",
        "user_identifier",
        "seed",
        "n",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "reasoning_effort",
        "thinking_budget_tokens",
        "provider_continuations",
        "request_timeout",
        "request_retries",
        "request_retry_delay",
        "api_key_resolved",
    }
    assert expected <= params, sorted(expected - params)


def test_custom_surface_params_flow_into_payload() -> None:
    payload = build_hosted_chat_payload(
        CUSTOM_HOSTED,
        resolution=_resolution(),
        messages_payload=_messages(),
        temperature=0.3,
        top_p=0.9,
        min_p=0.05,
        top_k=40,
        seed=7,
        n=1,
        presence_penalty=0.1,
        frequency_penalty=0.2,
        logit_bias={"1234": -1.5},
        logprobs=True,
        top_logprobs=5,
        user="tester",
    )
    assert payload["temperature"] == 0.3
    assert payload["top_p"] == 0.9
    assert payload["min_p"] == 0.05
    assert payload["top_k"] == 40
    assert payload["seed"] == 7
    assert payload["n"] == 1
    assert payload["presence_penalty"] == 0.1
    assert payload["frequency_penalty"] == 0.2
    assert payload["logit_bias"] == {"1234": -1.5}
    assert payload["logprobs"] is True
    assert payload["top_logprobs"] == 5
    assert payload["user"] == "tester"


def test_top_logprobs_requires_logprobs() -> None:
    with pytest.raises(ChatBadRequestError):
        build_hosted_chat_payload(
            CUSTOM_HOSTED,
            resolution=_resolution(),
            messages_payload=_messages(),
            top_logprobs=5,
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("min_p", 1.5),
        ("top_k", -1),
        ("presence_penalty", 3.0),
        ("frequency_penalty", "high"),
        ("logit_bias", {"1234": "banana"}),
        ("logprobs", "yes"),
        ("top_logprobs", 21),
        ("thinking_budget_tokens", 0),
    ],
)
def test_custom_surface_validators_fail_closed(name: str, value: object) -> None:
    with pytest.raises(ChatBadRequestError):
        build_hosted_chat_payload(
            CUSTOM_HOSTED,
            resolution=_resolution(),
            messages_payload=_messages(),
            **{name: value},
        )


def test_flagless_records_keep_raise_on_supplied() -> None:
    resolution = _resolution(
        provider="together",
        base_url="https://api.together.xyz/v1",
        timeout=90.0,
        retries=3,
        retry_delay=5.0,
    )
    with pytest.raises(ChatBadRequestError):
        build_hosted_chat_payload(
            TOGETHER,
            resolution=resolution,  # type: ignore[arg-type]
            messages_payload=_messages(),
            min_p=0.05,
        )
    with pytest.raises(ChatBadRequestError):
        build_hosted_chat_payload(
            TOGETHER,
            resolution=resolution,  # type: ignore[arg-type]
            messages_payload=_messages(),
            thinking_budget_tokens=1024,
        )


# --- Stage 1: settings fallback precedence ---


def test_section_tuned_values_flow_into_resolution() -> None:
    resolution = resolve_hosted_request(
        CUSTOM_HOSTED,
        explicit_base_url="https://custom.example/v1",
        app_config={
            "api_settings": {
                "custom": {
                    "model": "tuned-model",
                    "temperature": 0.11,
                    "top_p": 0.22,
                    "top_k": 33,
                    "min_p": 0.44,
                    "max_tokens": 512,
                    "seed": 9,
                    "stop": ["\n\n"],
                    "response_format": {"type": "json_object"},
                    "streaming": False,
                    "api_timeout": 33,
                    "api_retries": 4,
                    "api_retry_delay": 2.5,
                }
            }
        },
        environ={},
    )
    assert resolution.model == "tuned-model"
    assert resolution.temperature == 0.11
    assert resolution.top_p == 0.22
    assert resolution.top_k == 33
    assert resolution.min_p == 0.44
    assert resolution.max_tokens == 512
    assert resolution.seed == 9
    assert resolution.stop == ["\n\n"]
    assert resolution.response_format == {"type": "json_object"}
    assert resolution.streaming is False
    assert resolution.timeout == 33
    assert resolution.retries == 4
    assert resolution.retry_delay == 2.5


def test_absent_section_yields_record_defaults() -> None:
    resolution = resolve_hosted_request(
        CUSTOM_HOSTED,
        explicit_base_url="https://custom.example/v1",
        app_config={"api_settings": {}},
        environ={},
    )
    assert resolution.streaming is False
    assert resolution.max_tokens == 4096
    assert resolution.timeout == 120
    assert resolution.retries == 1
    assert resolution.retry_delay == 1.0
    # No legacy temperature default ships on the record: an absent section
    # leaves the sampler unset (the payload omits it) instead of forcing 0.7.
    assert resolution.temperature is None
    assert resolution.top_p is None
    assert resolution.model == ""


def test_legacy_fallback_spellings_are_read() -> None:
    resolution = resolve_hosted_request(
        CUSTOM_HOSTED,
        explicit_base_url="https://custom.example/v1",
        app_config={
            "api_settings": {
                "custom": {
                    "temp": 0.66,
                    "maxp": 0.77,
                    "topk": 12,
                    "minp": 0.88,
                }
            }
        },
        environ={},
    )
    assert resolution.temperature == 0.66
    assert resolution.top_p == 0.77
    assert resolution.top_k == 12
    assert resolution.min_p == 0.88


def test_section_values_flow_into_payload_and_explicit_kwargs_win() -> None:
    resolution = resolve_hosted_request(
        CUSTOM_HOSTED,
        explicit_base_url="https://custom.example/v1",
        explicit_model="m",
        app_config={
            "api_settings": {
                "custom": {"temperature": 0.11, "max_tokens": 512, "seed": 9}
            }
        },
        environ={},
    )
    from_settings = build_hosted_chat_payload(
        CUSTOM_HOSTED,
        resolution=resolution,
        messages_payload=_messages(),
    )
    assert from_settings["temperature"] == 0.11
    assert from_settings["max_tokens"] == 512
    assert from_settings["seed"] == 9

    explicit_wins = build_hosted_chat_payload(
        CUSTOM_HOSTED,
        resolution=resolution,
        messages_payload=_messages(),
        temperature=0.9,
        max_tokens=64,
    )
    assert explicit_wins["temperature"] == 0.9
    assert explicit_wins["max_tokens"] == 64


def test_payload_carries_record_default_max_tokens_when_section_silent() -> None:
    resolution = resolve_hosted_request(
        CUSTOM_HOSTED,
        explicit_base_url="https://custom.example/v1",
        explicit_model="m",
        app_config={"api_settings": {}},
        environ={},
    )
    payload = build_hosted_chat_payload(
        CUSTOM_HOSTED,
        resolution=resolution,
        messages_payload=_messages(),
    )
    assert payload["max_tokens"] == 4096
    assert payload["stream"] is False
    assert "temperature" not in payload


# --- Stage 1: ADR-066 reasoning composition (custom row) ---


def test_reasoning_effort_composed_verbatim() -> None:
    payload = build_hosted_chat_payload(
        CUSTOM_HOSTED,
        resolution=_resolution(),
        messages_payload=_messages(),
        reasoning_effort="high",
    )
    # ADR-066 custom row: top-level OpenAI-style reasoning_effort, verbatim.
    assert payload["reasoning_effort"] == "high"


def test_thinking_budget_tokens_accepted_and_dropped() -> None:
    without_budget = build_hosted_chat_payload(
        CUSTOM_HOSTED,
        resolution=_resolution(),
        messages_payload=_messages(),
    )
    with_budget = build_hosted_chat_payload(
        CUSTOM_HOSTED,
        resolution=_resolution(),
        messages_payload=_messages(),
        reasoning_effort="low",
        thinking_budget_tokens=2048,
    )
    # ADR-066 custom row: budget dropped (strict OpenAI proxies may reject
    # llama.cpp-specific fields); effort stays verbatim.
    assert with_budget["reasoning_effort"] == "low"
    assert "reasoning_budget_tokens" not in with_budget
    assert "chat_template_kwargs" not in with_budget
    assert with_budget == {**without_budget, "reasoning_effort": "low"}


# --- Stage 1: continuation pairings ---


def test_continuation_pairings_gain_the_four_keys() -> None:
    literal = set(get_args(ContinuationProvider))
    assert {"custom-hosted", "together", "fireworks", "cerebras"} <= literal


def test_custom_hosted_continuation_round_trip() -> None:
    checkpoint = ProviderContinuationCheckpoint(
        schema_version=1,
        checkpoint_revision=1,
        provider="custom-hosted",
        protocol="chat_completions",
        model="test-model",
        api_base_url="https://custom.example/v1",
        state="active",
        rounds=(
            ContinuationRound(
                assistant_content="",
                reasoning_blocks=(),
                calls=(
                    ContinuationCall(
                        call_id="call-1",
                        name="get_time",
                        arguments="{}",
                        state="pending",
                    ),
                ),
            ),
        ),
    )
    parsed = parse_provider_continuation_json(
        dump_provider_continuation_json(checkpoint)
    )
    assert parsed is not None
    assert parsed.provider == "custom-hosted"


# --- Stage 3: gateway identity-site swap + kill switch ---


def _entry_config(
    *, console: dict[str, object] | None = None
) -> dict[str, object]:
    config: dict[str, object] = {
        "custom_endpoints": {
            "paid": {
                "display_name": "Paid",
                "family": "openai_compatible",
                "base_url": "https://api.example.com/v1",
            }
        }
    }
    if console is not None:
        config["console"] = console
    return config


def _llama_entry_config(
    *, console: dict[str, object] | None = None
) -> dict[str, object]:
    config: dict[str, object] = {
        "custom_endpoints": {
            "local": {
                "display_name": "Local",
                "family": "llama_cpp",
                "base_url": "https://llama.example",
            }
        }
    }
    if console is not None:
        config["console"] = console
    return config


async def _resolve(
    config: dict[str, object], *, provider: str = "custom-ep:paid"
) -> object:
    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderGateway,
        ConsoleProviderSelection,
    )

    gateway = ConsoleProviderGateway(config_provider=lambda: config, environ={})
    return await gateway._resolve_for_send_unclassified(
        ConsoleProviderSelection(provider=provider, explicit_model="m")
    )


def test_family_execution_key_is_unchanged() -> None:
    from tldw_chatbook.Chat.custom_endpoint_registry import family_execution_key

    assert family_execution_key("openai_compatible") == "custom"
    assert family_execution_key("llama_cpp") == "llama_cpp"
    assert family_execution_key("ollama") == "ollama"


@pytest.mark.asyncio
async def test_swap_on_by_default_routes_to_the_engine_key() -> None:
    resolved = await _resolve(_entry_config())
    assert resolved.ready is True
    assert resolved.execution_key == "custom-hosted"
    assert resolved.readiness_key == "custom"
    assert resolved.selected_provider == "custom-ep:paid"
    assert "api.example.com/v1" in resolved.base_url


@pytest.mark.asyncio
async def test_swap_explicitly_on_routes_to_the_engine_key() -> None:
    resolved = await _resolve(
        _entry_config(console={"custom_endpoints_use_engine": True})
    )
    assert resolved.execution_key == "custom-hosted"


@pytest.mark.asyncio
async def test_swap_off_keeps_the_legacy_execution_key() -> None:
    resolved = await _resolve(
        _entry_config(console={"custom_endpoints_use_engine": False})
    )
    assert resolved.ready is True
    assert resolved.execution_key == "custom-openai-api"


@pytest.mark.asyncio
async def test_malformed_switch_value_fails_open_to_the_default() -> None:
    resolved = await _resolve(
        _entry_config(console={"custom_endpoints_use_engine": "false"})
    )
    assert resolved.execution_key == "custom-hosted"


@pytest.mark.asyncio
async def test_non_openai_compatible_family_never_swaps() -> None:
    resolved = await _resolve(
        _llama_entry_config(console={"custom_endpoints_use_engine": True}),
        provider="custom-ep:local",
    )
    assert resolved.execution_key == "llama_cpp"


@pytest.mark.asyncio
async def test_saved_session_identity_byte_identical_both_directions() -> None:
    # The persisted/session-facing identity of a custom-ep provider must
    # not change with the swap: same readiness key, same selected provider
    # (the saved-session spelling), same endpoint/model/credential fields
    # and same canonical connection identity -- only the execution key
    # differs.
    from tldw_chatbook.Chat.provider_endpoint_contract import (
        canonical_connection_identity,
    )

    swapped = await _resolve(_entry_config())
    legacy = await _resolve(
        _entry_config(console={"custom_endpoints_use_engine": False})
    )
    assert swapped.execution_key != legacy.execution_key
    for field in (
        "provider",
        "readiness_key",
        "selected_provider",
        "base_url",
        "model",
        "api_key",
        "api_key_source",
        "visible_copy",
    ):
        assert getattr(swapped, field) == getattr(legacy, field), field
    assert canonical_connection_identity(
        swapped.readiness_key, swapped.base_url
    ) == canonical_connection_identity(legacy.readiness_key, legacy.base_url)


def test_finish_policy_resolver_returns_the_engine_policy() -> None:
    from tldw_chatbook.Chat.console_provider_gateway import resolve_finish_policy
    from tldw_chatbook.LLM_Calls.hosted_provider_engine import (
        HostedPresetFinishPolicy,
    )

    policy = resolve_finish_policy("custom-hosted")
    assert isinstance(policy, HostedPresetFinishPolicy)


# --- Stage 3: canned transport (keyless + keyed, non-streaming only) ---


_SUCCESS_BODY = {
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ]
}


def _canned_session(
    monkeypatch: pytest.MonkeyPatch,
) -> "_RecordingSession":
    import tldw_chatbook.LLM_Calls.hosted_chat as hosted_chat
    from Tests.LLM_Calls.test_qwencloud import (
        _RecordingSession,
        _TransportResponse,
    )

    session = _RecordingSession(_TransportResponse(dict(_SUCCESS_BODY)))
    monkeypatch.setattr(hosted_chat, "create_default_session", lambda: session)
    return session


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_keyed_custom_hosted_send_uses_forwarded_url_and_bearer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.Chat.Chat_Functions import chat_api_call

    session = _canned_session(monkeypatch)
    response = chat_api_call(
        api_endpoint="custom-hosted",
        messages_payload=[{"role": "user", "content": "hi"}],
        api_key="stored-entry-key",
        api_key_resolved=True,
        api_base_url="https://api.example.com/v1",
        model="m",
        streaming=False,
    )
    assert response["choices"][0]["message"]["content"] == "ok"
    post = session.posts[0]
    assert post["url"].startswith("https://api.example.com/v1/")
    assert post["url"].endswith("/chat/completions")
    assert post["headers"]["Authorization"] == "Bearer stored-entry-key"
    payload = post["json"]
    assert payload["model"] == "m"
    assert payload["stream"] is False
    assert payload["max_tokens"] == 4096  # legacy default flows through


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_keyless_custom_hosted_send_sends_no_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.Chat.Chat_Functions import chat_api_call

    session = _canned_session(monkeypatch)
    response = chat_api_call(
        api_endpoint="custom-hosted",
        messages_payload=[{"role": "user", "content": "hi"}],
        api_key_resolved=True,
        api_base_url="https://keyless.example/v1",
        model="m",
        streaming=False,
    )
    assert response["choices"][0]["message"]["content"] == "ok"
    post = session.posts[0]
    assert post["url"].startswith("https://keyless.example/v1/")
    assert post["headers"].get("Authorization") is None


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_custom_hosted_send_composes_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.Chat.Chat_Functions import chat_api_call

    session = _canned_session(monkeypatch)
    chat_api_call(
        api_endpoint="custom-hosted",
        messages_payload=[{"role": "user", "content": "hi"}],
        api_key="stored-entry-key",
        api_key_resolved=True,
        api_base_url="https://api.example.com/v1",
        model="m",
        streaming=False,
        reasoning_effort="high",
        thinking_budget_tokens=2048,
        minp=0.05,
        topk=40,
    )
    payload = session.posts[0]["json"]
    # ADR-066 custom row: effort verbatim; budget dropped; samplers flow.
    assert payload["reasoning_effort"] == "high"
    assert "reasoning_budget_tokens" not in payload
    assert "chat_template_kwargs" not in payload
    assert payload["min_p"] == 0.05
    assert payload["top_k"] == 40
