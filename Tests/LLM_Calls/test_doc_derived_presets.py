"""Doc-derived inference-cloud presets (TASK-33126).

SambaNova, NVIDIA NIM, DeepInfra, Nebius Token Factory, Novita AI and
MiniMax ship as engine records derived from each provider's PUBLIC
documentation (read 2026-09-27) -- no keys, no captured fixtures. The
response bodies below are built from the shapes those docs publish, so a
record's allowances are pinned to documented fields and anything else still
fails closed. Also pinned: the two streamed-usage record flags the docs
forced (``stream_include_usage``, ``stream_usage_optional``) and that every
pre-existing preset's payload is unchanged by them.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest

from tldw_chatbook.Chat.Chat_Deps import ChatProviderError
from tldw_chatbook.LLM_Calls import hosted_provider_engine
from tldw_chatbook.LLM_Calls.hosted_chat import HostedChatProtocolError
from tldw_chatbook.LLM_Calls.hosted_chat_streaming import SSERecord
from tldw_chatbook.LLM_Calls.hosted_provider_engine import (
    HostedProviderResolution,
    HostedProviderStream,
    build_hosted_chat_payload,
)
from tldw_chatbook.provider_registry import RECORDS_BY_KEY

EXPECTED: dict[str, dict[str, Any]] = {
    "sambanova": {
        "config_key": "SambaNova",
        "base_url": "https://api.sambanova.ai/v1",
        "env_var": "SAMBANOVA_API_KEY",
        "models_url": "https://api.sambanova.ai/v1/models",
        "response": frozenset(),
        "choice": frozenset({"logprobs"}),
        "message": frozenset({"reasoning", "channel"}),
        "include_usage": True,
        "usage_optional": False,
        "reasoning": "ignored",
        "extra_body": {},
    },
    "nvidia": {
        "config_key": "NVIDIA",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "env_var": "NVIDIA_API_KEY",
        "models_url": "https://integrate.api.nvidia.com/v1/models",
        "response": frozenset(),
        "choice": frozenset(),
        "message": frozenset(),
        "include_usage": False,
        "usage_optional": True,
        "reasoning": "proprietary",
        "extra_body": {},
    },
    "deepinfra": {
        "config_key": "DeepInfra",
        "base_url": "https://api.deepinfra.com/v1/openai",
        "env_var": "DEEPINFRA_API_KEY",
        "models_url": "https://api.deepinfra.com/v1/openai/models",
        "response": frozenset({"service_tier"}),
        "choice": frozenset(),
        "message": frozenset(),
        "include_usage": False,
        "usage_optional": False,
        "reasoning": "ignored",
        "extra_body": {},
    },
    "nebius": {
        "config_key": "Nebius",
        "base_url": "https://api.tokenfactory.nebius.com/v1",
        "env_var": "NEBIUS_API_KEY",
        "models_url": "https://api.tokenfactory.nebius.com/v1/models",
        "response": frozenset({"service_tier"}),
        "choice": frozenset({"logprobs"}),
        "message": frozenset(),
        "include_usage": True,
        "usage_optional": False,
        "reasoning": "proprietary",
        "extra_body": {},
    },
    "novita": {
        "config_key": "Novita",
        "base_url": "https://api.novita.ai/openai/v1",
        "env_var": "NOVITA_API_KEY",
        "models_url": "https://api.novita.ai/openai/v1/models",
        "response": frozenset(),
        "choice": frozenset(),
        "message": frozenset(),
        "include_usage": True,
        "usage_optional": False,
        "reasoning": "proprietary",
        "extra_body": {"separate_reasoning": True},
    },
    "minimax": {
        "config_key": "MiniMax",
        "base_url": "https://api.minimax.io/v1",
        "env_var": "MINIMAX_API_KEY",
        "models_url": None,  # no documented /models route
        "response": frozenset(
            {
                "base_resp",
                "input_sensitive",
                "input_sensitive_type",
                "output_sensitive",
                "output_sensitive_type",
            }
        ),
        "choice": frozenset(),
        "message": frozenset({"name", "audio_content"}),
        "include_usage": True,
        "usage_optional": False,
        "reasoning": "proprietary",
        "extra_body": {"reasoning_split": True},
    },
}
KEYS = tuple(EXPECTED)
EXISTING_ENGINE_KEYS = ("databricks", "together", "fireworks", "cerebras", "custom-hosted")
_LLM_CALLS = Path(__file__).resolve().parents[2] / "tldw_chatbook" / "LLM_Calls"
_USAGE = {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}


def _resolution(record: Any, *, streaming: bool) -> HostedProviderResolution:
    return HostedProviderResolution(
        provider=record.key,
        model="doc-model",
        api_key="secret",
        base_url=record.default_base_url or "https://engine.invalid/v1",
        timeout=10.0,
        retries=0,
        retry_delay=0.0,
        streaming=streaming,
    )


def _replay(
    monkeypatch: pytest.MonkeyPatch,
    record: Any,
    *,
    body: dict[str, Any] | None = None,
    stream_events: list[str] | None = None,
) -> Any:
    """Run one canned response through the engine's real handler path."""
    streaming = stream_events is not None
    resolution = _resolution(record, streaming=streaming)
    monkeypatch.setattr(
        hosted_provider_engine,
        "resolve_hosted_request",
        lambda _record, **_kwargs: resolution,
    )
    if streaming:
        records = iter([SSERecord(event=None, data=data) for data in stream_events])
        monkeypatch.setattr(hosted_provider_engine, "owned_json_post", lambda **_kw: records)
    else:
        monkeypatch.setattr(hosted_provider_engine, "owned_json_post", lambda **_kw: body)
    handler = hosted_provider_engine.build_hosted_chat_handler(record)
    return handler(
        input_data=[{"role": "user", "content": "Say ok."}],
        api_key="secret",
        streaming=streaming,
    )


def _body(*, top: dict[str, Any] | None = None, choice: dict[str, Any] | None = None,
          message: dict[str, Any] | None = None, finish: str = "stop") -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "doc-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok", **(message or {})},
                "finish_reason": finish,
                **(choice or {}),
            }
        ],
        "usage": dict(_USAGE),
        **(top or {}),
    }


def _chunk(delta: dict[str, Any], *, finish: str | None = None,
           usage: dict[str, Any] | None = None) -> str:
    import json

    event: dict[str, Any] = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "doc-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if usage is not None:
        event["usage"] = usage
    return json.dumps(event)



# --- record shape: every value traceable to the docs (see registry comments) ---


@pytest.mark.parametrize("key", KEYS)
def test_record_matches_its_documented_contract(key: str) -> None:
    """Every record field matches the value its provider documents.

    Args:
        key: Registry key of one doc-derived preset.
    """
    record = RECORDS_BY_KEY[key]
    expected = EXPECTED[key]
    assert record.engine_driven is True
    assert record.auth_scheme == "bearer"
    assert record.tolerant_response_extras is False
    assert record.native_tools is True
    assert record.config_key == expected["config_key"]
    assert record.default_base_url == expected["base_url"]
    assert record.api_key_env_var == expected["env_var"]
    assert record.api_key_env_candidates == (expected["env_var"],)
    assert record.response_allowances == expected["response"]
    assert record.choice_allowances == expected["choice"]
    assert record.message_allowances == expected["message"]
    assert record.stream_include_usage is expected["include_usage"]
    assert record.stream_usage_optional is expected["usage_optional"]
    assert record.reasoning_disposition == expected["reasoning"]
    assert dict(record.extra_body_fields) == expected["extra_body"]
    assert record.auto_refresh is (expected["models_url"] is not None)
    assert (record.discovery_route is None) is (expected["models_url"] is None)
    assert record.status_envelope_key == ("base_resp" if key == "minimax" else None)
    assert "model" not in record.settings_defaults


def test_minimax_never_borrows_the_openai_key() -> None:
    """MiniMax's OpenAI-SDK quickstart repoints OPENAI_API_KEY; the record
    must not walk it, or an OpenAI credential would be sent to MiniMax."""
    assert "OPENAI_API_KEY" not in RECORDS_BY_KEY["minimax"].api_key_env_candidates


def test_no_per_provider_module_ships() -> None:
    """The presets are data only: no ``LLM_Calls/<provider>*.py`` module."""
    for key in KEYS:
        assert not list(_LLM_CALLS.glob(f"{key}*.py")), key


@pytest.mark.parametrize("key", KEYS)
def test_dispatch_and_param_map_route_through_the_engine(key: str) -> None:
    """Each preset dispatches through the engine with the engine param map.

    Args:
        key: Registry key of one doc-derived preset.
    """
    from tldw_chatbook.Chat.Chat_Functions import (
        API_CALL_HANDLERS,
        ENGINE_PROVIDER_PARAM_MAP,
        PROVIDER_PARAM_MAP,
    )

    assert callable(API_CALL_HANDLERS[key])
    assert PROVIDER_PARAM_MAP[key] is ENGINE_PROVIDER_PARAM_MAP


# --- config tables and discovery ---


@pytest.mark.parametrize("key", KEYS)
def test_config_tables_mirror_the_record(key: str) -> None:
    """The shipped ``[api_settings]`` table and ``[providers]`` seed match the record.

    Args:
        key: Registry key of one doc-derived preset.
    """
    from tldw_chatbook.config import CONFIG_TOML_CONTENT

    record = RECORDS_BY_KEY[key]
    parsed = tomllib.loads(CONFIG_TOML_CONTENT)
    assert parsed["api_settings"][key] == dict(record.settings_defaults) | {
        "api_base_url": record.default_base_url
    }
    seeds = parsed["providers"][record.config_key]
    if record.auto_refresh:
        assert seeds == []
    else:
        # MiniMax: seeded from the documented ``model`` enum instead.
        assert "MiniMax-M2.7" in seeds and len(seeds) == len(set(seeds))


@pytest.mark.parametrize("key", [k for k in KEYS if EXPECTED[k]["models_url"]])
def test_default_urls_discover_at_the_documented_models_route(key: str) -> None:
    """Discovery accepts each default URL and derives the documented models URL.

    Args:
        key: Registry key of a preset with a models route.
    """
    from tldw_chatbook.LLM_Provider_Catalog.openai_compatible_model_discovery import (
        build_models_url,
        supports_openai_compatible_model_discovery,
    )

    base = EXPECTED[key]["base_url"]
    assert supports_openai_compatible_model_discovery(key, base) is True
    assert build_models_url(base, key) == EXPECTED[key]["models_url"]


@pytest.mark.parametrize("identity", ["minimax", "MiniMax"])
def test_seeded_only_minimax_refuses_discovery(identity: str) -> None:
    """MiniMax documents no models route, so discovery never probes one.

    Args:
        identity: Provider identity spelling as a caller may pass it.
    """
    from tldw_chatbook.LLM_Provider_Catalog.openai_compatible_model_discovery import (
        supports_openai_compatible_model_discovery,
    )

    assert supports_openai_compatible_model_discovery(identity, EXPECTED["minimax"]["base_url"]) is False


# --- payloads: the streamed-usage request flag ---


@pytest.mark.parametrize("key", KEYS + EXISTING_ENGINE_KEYS)
def test_stream_options_only_where_the_record_asks(key: str) -> None:
    """``stream_options`` rides only streaming payloads of records that ask.

    Args:
        key: Registry key of a doc-derived or pre-existing engine preset.
    """
    record = RECORDS_BY_KEY[key]
    messages = [{"role": "user", "content": "hi"}]
    streamed = build_hosted_chat_payload(
        record, resolution=_resolution(record, streaming=True),
        messages_payload=messages, streaming=True,
    )
    blocking = build_hosted_chat_payload(
        record, resolution=_resolution(record, streaming=False),
        messages_payload=messages, streaming=False,
    )
    assert "stream_options" not in blocking
    if record.stream_include_usage:
        assert streamed["stream_options"] == {"include_usage": True}
    else:
        assert "stream_options" not in streamed


def test_existing_presets_keep_their_payload_contract() -> None:
    """The new flags default off for every preset that predates them."""
    for key in EXISTING_ENGINE_KEYS:
        record = RECORDS_BY_KEY[key]
        assert record.stream_include_usage is False, key
        assert record.stream_usage_optional is False, key
        assert record.status_envelope_key is None, key


@pytest.mark.parametrize("key", ["novita", "minimax"])
def test_reasoning_split_fields_ride_every_payload(key: str) -> None:
    """Novita/MiniMax always ask for reasoning in its own field.

    Args:
        key: Registry key of a preset with reasoning-split body fields.
    """
    record = RECORDS_BY_KEY[key]
    payload = build_hosted_chat_payload(
        record, resolution=_resolution(record, streaming=False),
        messages_payload=[{"role": "user", "content": "hi"}], streaming=False,
    )
    for field, value in EXPECTED[key]["extra_body"].items():
        assert payload[field] == value


# --- responses: documented extras parse, undocumented ones fail closed ---

_DOCUMENTED_BODIES: dict[str, dict[str, Any]] = {
    "sambanova": _body(choice={"logprobs": None}),
    "nvidia": _body(message={"reasoning_content": "thinking"}),
    "deepinfra": _body(
        top={"service_tier": "default"},
    ),
    "nebius": _body(
        top={"service_tier": "default"},
        choice={"logprobs": None},
        message={"reasoning_content": "thinking"},
    ),
    "novita": _body(message={"reasoning_content": "thinking"}),
    "minimax": _body(
        top={
            "base_resp": {"status_code": 0, "status_msg": ""},
            "input_sensitive": False,
            "input_sensitive_type": 0,
            "output_sensitive": False,
            "output_sensitive_type": 0,
        },
        message={"reasoning_content": "thinking", "name": "MiniMax AI", "audio_content": ""},
    ),
}
_DOCUMENTED_BODIES["deepinfra"]["usage"]["estimated_cost"] = 0.0000268


@pytest.mark.parametrize("key", KEYS)
def test_documented_response_shape_parses(key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A body carrying every documented extra parses into a normal turn.

    Args:
        key: Registry key of one doc-derived preset.
        monkeypatch: Replaces resolution and transport with canned values.
    """
    result = _replay(monkeypatch, RECORDS_BY_KEY[key], body=_DOCUMENTED_BODIES[key])
    assert result.terminal_turn.text == "ok"
    assert result.terminal_turn.finish_reason == "stop"


@pytest.mark.parametrize("key", KEYS)
def test_undocumented_top_level_key_still_fails_closed(
    key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control: a field no doc lists is still rejected.

    Args:
        key: Registry key of one doc-derived preset.
        monkeypatch: Replaces resolution and transport with canned values.
    """
    body = _body(top={"undocumented_extra": 1})
    with pytest.raises((HostedChatProtocolError, ChatProviderError)):
        _replay(monkeypatch, RECORDS_BY_KEY[key], body=body)


@pytest.mark.parametrize("key", ["nebius", "minimax"])
def test_content_filter_finish_is_a_provider_error(
    key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A documented ``content_filter`` finish is a 502, not a partial reply.

    Args:
        key: Registry key of a preset that documents ``content_filter``.
        monkeypatch: Replaces resolution and transport with canned values.
    """
    with pytest.raises(ChatProviderError) as excinfo:
        _replay(monkeypatch, RECORDS_BY_KEY[key], body=_body(finish="content_filter"))
    assert excinfo.value.status_code == 502


# --- MiniMax status envelope: an error status never becomes a reply ---

_MINIMAX_ERROR = {"status_code": 1002, "status_msg": "rate limit: prompt text echoed here"}


def test_minimax_error_status_with_valid_choices_is_a_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A nonzero ``base_resp`` wins over otherwise valid choices, and the
    provider-authored status text never reaches the error message.

    Args:
        monkeypatch: Replaces resolution and transport with canned values.
    """
    body = _body(top={"base_resp": _MINIMAX_ERROR})
    with pytest.raises(ChatProviderError) as excinfo:
        _replay(monkeypatch, RECORDS_BY_KEY["minimax"], body=body)
    assert excinfo.value.status_code == 502
    assert "1002" in str(excinfo.value)
    assert "echoed" not in str(excinfo.value)


def test_minimax_error_status_in_a_stream_event_is_a_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same check runs on every stream event, not just bodies.

    Args:
        monkeypatch: Replaces resolution and transport with canned values.
    """
    import json

    event = json.loads(_chunk({"role": "assistant", "content": "ok"}))
    event["base_resp"] = _MINIMAX_ERROR
    stream = _replay(
        monkeypatch, RECORDS_BY_KEY["minimax"], stream_events=[json.dumps(event), "[DONE]"]
    )
    with pytest.raises(ChatProviderError) as excinfo:
        list(stream)
    assert "1002" in str(excinfo.value)


@pytest.mark.parametrize("envelope", ["oops", {"status_code": "0"}, {"status_code": True}])
def test_minimax_malformed_status_envelope_fails_closed(
    envelope: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A status envelope that is not ``{status_code: int}`` is rejected.

    Args:
        envelope: A malformed ``base_resp`` value.
        monkeypatch: Replaces resolution and transport with canned values.
    """
    with pytest.raises(ChatProviderError):
        _replay(monkeypatch, RECORDS_BY_KEY["minimax"], body=_body(top={"base_resp": envelope}))


# --- streams: usage requested, usage optional, delta-only reasoning ---


def test_sambanova_stream_with_delta_reasoning_and_trailing_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SambaNova's delta-only ``reasoning``/``channel`` are tolerated and the
    trailing usage chunk is kept.

    Args:
        monkeypatch: Replaces resolution and transport with canned values.
    """
    import json

    events = [
        _chunk({"role": "assistant", "reasoning": "hmm", "channel": "analysis"}),
        _chunk({"content": "ok"}),
        _chunk({}, finish="stop"),
        json.dumps({"id": "chatcmpl-1", "object": "chat.completion.chunk",
                    "created": 1, "model": "doc-model", "choices": [], "usage": _USAGE}),
        "[DONE]",
    ]
    stream = _replay(monkeypatch, RECORDS_BY_KEY["sambanova"], stream_events=events)
    assert isinstance(stream, HostedProviderStream)
    list(stream)
    assert stream.terminal_turn.text == "ok"
    assert stream.terminal_turn.usage == _USAGE


def test_nvidia_stream_without_usage_completes(monkeypatch: pytest.MonkeyPatch) -> None:
    """NVIDIA's chunk schema has no usage; its stream still completes.

    Args:
        monkeypatch: Replaces resolution and transport with canned values.
    """
    events = [_chunk({"role": "assistant", "content": "ok"}), _chunk({}, finish="stop"), "[DONE]"]
    stream = _replay(monkeypatch, RECORDS_BY_KEY["nvidia"], stream_events=events)
    list(stream)
    assert stream.terminal_turn.text == "ok"
    assert stream.terminal_turn.usage is None


def test_usage_optional_still_requires_a_finish_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Usage-optional relaxes usage only; a missing finish reason still fails.

    Args:
        monkeypatch: Replaces resolution and transport with canned values.
    """
    events = [_chunk({"role": "assistant", "content": "ok"}), "[DONE]"]
    stream = _replay(monkeypatch, RECORDS_BY_KEY["nvidia"], stream_events=events)
    with pytest.raises(HostedChatProtocolError):
        list(stream)


@pytest.mark.parametrize("key", [k for k in KEYS if not EXPECTED[k]["usage_optional"]])
def test_strict_records_still_reject_a_stream_without_usage(
    key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every other preset still requires streamed usage.

    Args:
        key: Registry key of a preset without ``stream_usage_optional``.
        monkeypatch: Replaces resolution and transport with canned values.
    """
    events = [_chunk({"role": "assistant", "content": "ok"}), _chunk({}, finish="stop"), "[DONE]"]
    stream = _replay(monkeypatch, RECORDS_BY_KEY[key], stream_events=events)
    with pytest.raises(HostedChatProtocolError):
        list(stream)


def test_deepinfra_usage_on_the_finish_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    """docs.deepinfra.com/chat/streaming: usage rides the finish chunk.

    Args:
        monkeypatch: Replaces resolution and transport with canned values.
    """
    events = [
        _chunk({"role": "assistant", "content": "ok"}),
        _chunk({}, finish="stop", usage=dict(_USAGE, estimated_cost=0.1)),
        "[DONE]",
    ]
    stream = _replay(monkeypatch, RECORDS_BY_KEY["deepinfra"], stream_events=events)
    list(stream)
    assert stream.terminal_turn.usage["total_tokens"] == 4
