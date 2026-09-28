"""Top-15 OpenRouter model-maker presets (TASK-33350).

Xiaomi MiMo, Tencent TokenHub (Hy4), ByteDance Seed (BytePlus ModelArk) and
StepFun ship as engine records built from each provider's public
documentation (read 2026-09-28) plus unauthenticated probes of each host.
The bodies below use the shapes those docs publish: every documented extra
parses, anything else still fails closed. Also pinned: MiMo's ``api-key``
header through the real engine transport, TokenHub's mid-stream error frame,
and the ``repetition_truncation`` finish both document.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

import pytest

from Tests.LLM_Calls.test_doc_derived_presets import (
    EXISTING_ENGINE_KEYS,
    _USAGE,
    _body,
    _chunk,
    _replay,
    _resolution,
)
from Tests.LLM_Calls.test_qwencloud import _RecordingSession, _TransportResponse
from tldw_chatbook.Chat.Chat_Deps import ChatProviderError
from tldw_chatbook.LLM_Calls import hosted_chat, hosted_provider_engine
from tldw_chatbook.LLM_Calls.hosted_chat import HostedChatProtocolError
from tldw_chatbook.LLM_Calls.hosted_provider_engine import build_hosted_chat_payload
from tldw_chatbook.provider_registry import RECORDS_BY_KEY

EXPECTED: dict[str, dict[str, Any]] = {
    "mimo": {
        "config_key": "MiMo",
        "base_url": "https://api.xiaomimimo.com/v1",
        "env_vars": ("MIMO_API_KEY",),
        "auth_scheme": "api_key_header",
        "native_tools": True,
        "models_url": None,
        "response": frozenset(),
        "choice": frozenset(),
        "message": frozenset(),
        "include_usage": True,
        "usage_optional": True,
        "reasoning": "proprietary",
        "extra_terminal": {"repetition_truncation"},
        "provider_errors": {"content_filter"},
        "error_frame_key": None,
    },
    "tokenhub": {
        "config_key": "TokenHub",
        "base_url": "https://tokenhub-intl.tencentcloudmaas.com/v1",
        "env_vars": ("TOKENHUB_API_KEY",),
        "auth_scheme": "bearer",
        "native_tools": True,
        "models_url": "https://tokenhub-intl.tencentcloudmaas.com/v1/models",
        "response": frozenset({"search_info"}),
        "choice": frozenset({"logprobs"}),
        "message": frozenset({"refusal"}),
        "include_usage": True,
        "usage_optional": False,
        "reasoning": "proprietary",
        "extra_terminal": {"repetition_truncation"},
        "provider_errors": {"content_filter"},
        "error_frame_key": "error",
    },
    "byteplus": {
        "config_key": "BytePlus",
        "base_url": "https://ark.ap-southeast.bytepluses.com/api/v3",
        "env_vars": ("ARK_API_KEY",),
        "auth_scheme": "bearer",
        "native_tools": True,
        "models_url": None,
        "response": frozenset({"service_tier"}),
        "choice": frozenset(),
        "message": frozenset({"encrypted_content"}),
        "include_usage": True,
        "usage_optional": True,
        "reasoning": "proprietary",
        "extra_terminal": set(),
        "provider_errors": {"content_filter"},
        "error_frame_key": None,
    },
    "stepfun": {
        "config_key": "StepFun",
        "base_url": "https://api.stepfun.ai/v1",
        "env_vars": ("STEPFUN_API_KEY",),
        "auth_scheme": "bearer",
        "native_tools": False,
        "models_url": "https://api.stepfun.ai/v1/models",
        "response": frozenset(),
        "choice": frozenset(),
        "message": frozenset({"reasoning"}),
        "include_usage": True,
        "usage_optional": False,
        "reasoning": "ignored",
        "extra_terminal": set(),
        "provider_errors": set(),
        "error_frame_key": None,
    },
}
KEYS = tuple(EXPECTED)
_LLM_CALLS = Path(__file__).resolve().parents[2] / "tldw_chatbook" / "LLM_Calls"
_DEFAULT_TERMINAL = {"stop", "tool_calls", "length"}


# --- record shape: every value traceable to the docs (see registry comments) ---


@pytest.mark.parametrize("key", KEYS)
def test_record_matches_its_documented_contract(key: str) -> None:
    """Every record field matches the value its provider documents.

    Args:
        key: Registry key of one model-maker preset.
    """
    record = RECORDS_BY_KEY[key]
    expected = EXPECTED[key]
    assert record.engine_driven is True
    assert record.tolerant_response_extras is False
    assert record.config_key == expected["config_key"]
    assert record.default_base_url == expected["base_url"]
    assert record.api_key_env_var == expected["env_vars"][0]
    assert record.api_key_env_candidates == expected["env_vars"]
    assert record.auth_scheme == expected["auth_scheme"]
    assert record.native_tools is expected["native_tools"]
    assert record.response_allowances == expected["response"]
    assert record.choice_allowances == expected["choice"]
    assert record.message_allowances == expected["message"]
    assert record.stream_include_usage is expected["include_usage"]
    assert record.stream_usage_optional is expected["usage_optional"]
    assert record.reasoning_disposition == expected["reasoning"]
    assert set(record.finish_terminal) == _DEFAULT_TERMINAL | expected["extra_terminal"]
    assert set(record.finish_provider_errors) == expected["provider_errors"]
    assert record.error_frame_key == expected["error_frame_key"]
    assert record.auto_refresh is (expected["models_url"] is not None)
    assert (record.discovery_route is None) is (expected["models_url"] is None)
    assert "model" not in record.settings_defaults


def test_no_per_provider_module_ships() -> None:
    """The presets are data only: no ``LLM_Calls/<provider>*.py`` module."""
    for key in KEYS:
        assert not list(_LLM_CALLS.glob(f"{key}*.py")), key


@pytest.mark.parametrize("key", KEYS)
def test_dispatch_and_param_map_route_through_the_engine(key: str) -> None:
    """Each preset dispatches through the engine with the engine param map.

    Args:
        key: Registry key of one model-maker preset.
    """
    from tldw_chatbook.Chat.Chat_Functions import (
        API_CALL_HANDLERS,
        ENGINE_PROVIDER_PARAM_MAP,
        PROVIDER_PARAM_MAP,
    )

    assert callable(API_CALL_HANDLERS[key])
    assert PROVIDER_PARAM_MAP[key] is ENGINE_PROVIDER_PARAM_MAP


@pytest.mark.parametrize("key", KEYS)
def test_config_tables_mirror_the_record(key: str) -> None:
    """The shipped ``[api_settings]`` table and ``[providers]`` seed match the record.

    Args:
        key: Registry key of one model-maker preset.
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
        assert seeds and len(seeds) == len(set(seeds))


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


@pytest.mark.parametrize("key", [k for k in KEYS if not EXPECTED[k]["models_url"]])
def test_seeded_only_presets_refuse_discovery(key: str) -> None:
    """MiMo and BytePlus have no usable models route; discovery never probes one.

    Args:
        key: Registry key of a seeded-only preset.
    """
    from tldw_chatbook.LLM_Provider_Catalog.openai_compatible_model_discovery import (
        supports_openai_compatible_model_discovery,
    )

    assert supports_openai_compatible_model_discovery(key, EXPECTED[key]["base_url"]) is False


# --- requests: auth header, streamed usage, byte-identity of older presets ---


def test_mimo_sends_its_key_as_an_api_key_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """Through the real engine handler, MiMo's key rides ``api-key``, never Bearer.

    Args:
        monkeypatch: Replaces resolution and the HTTP session with fakes.
    """
    record = RECORDS_BY_KEY["mimo"]
    resolution = _resolution(record, streaming=False)
    monkeypatch.setattr(
        hosted_provider_engine, "resolve_hosted_request", lambda _record, **_kw: resolution
    )
    session = _RecordingSession(_TransportResponse(_body()))
    monkeypatch.setattr(hosted_chat, "create_default_session", lambda: session)
    handler = hosted_provider_engine.build_hosted_chat_handler(record)
    result = handler(
        input_data=[{"role": "user", "content": "Say ok."}], api_key="secret", streaming=False
    )
    assert result.terminal_turn.text == "ok"
    headers = session.posts[0]["headers"]
    assert headers["api-key"] == "secret"
    assert "Authorization" not in headers


@pytest.mark.parametrize("key", KEYS + EXISTING_ENGINE_KEYS)
def test_stream_options_only_where_the_record_asks(key: str) -> None:
    """``stream_options`` rides only streaming payloads of records that ask.

    Args:
        key: Registry key of a model-maker or pre-existing engine preset.
    """
    record = RECORDS_BY_KEY[key]
    messages = [{"role": "user", "content": "hi"}]
    streamed = build_hosted_chat_payload(
        record, resolution=_resolution(record, streaming=True),
        messages_payload=messages, streaming=True,
    )
    if record.stream_include_usage:
        assert streamed["stream_options"] == {"include_usage": True}
    else:
        assert "stream_options" not in streamed


def test_existing_presets_keep_their_contract() -> None:
    """The new record fields default off for every preset that predates them."""
    for key in EXISTING_ENGINE_KEYS:
        record = RECORDS_BY_KEY[key]
        assert record.error_frame_key is None, key
        assert record.auth_scheme in {"bearer", "bearer_optional"}, key


# --- responses: documented extras parse, undocumented ones fail closed ---

_DOCUMENTED_BODIES: dict[str, dict[str, Any]] = {
    "mimo": _body(message={"reasoning_content": "thinking"}),
    "tokenhub": _body(
        top={"search_info": None},
        choice={"logprobs": None},
        message={"reasoning_content": "thinking", "refusal": None},
    ),
    "byteplus": _body(
        top={"service_tier": "default"},
        message={"reasoning_content": "thinking", "encrypted_content": "opaque-blob"},
    ),
    "stepfun": _body(message={"reasoning": "thinking"}),
}


@pytest.mark.parametrize("key", KEYS)
def test_documented_response_shape_parses(key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A body carrying every documented extra parses into a normal turn.

    Args:
        key: Registry key of one model-maker preset.
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
        key: Registry key of one model-maker preset.
        monkeypatch: Replaces resolution and transport with canned values.
    """
    with pytest.raises((HostedChatProtocolError, ChatProviderError)):
        _replay(monkeypatch, RECORDS_BY_KEY[key], body=_body(top={"undocumented_extra": 1}))


@pytest.mark.parametrize("key", ["mimo", "tokenhub"])
def test_repetition_truncation_is_a_normal_end(key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """MiMo and TokenHub document ``repetition_truncation`` as a truncated finish.

    Args:
        key: Registry key of a preset that documents it.
        monkeypatch: Replaces resolution and transport with canned values.
    """
    result = _replay(monkeypatch, RECORDS_BY_KEY[key], body=_body(finish="repetition_truncation"))
    assert result.terminal_turn.finish_reason == "repetition_truncation"


@pytest.mark.parametrize("key", ["mimo", "tokenhub", "byteplus"])
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


@pytest.mark.parametrize("field", ["reasoning_details", "search_results"])
def test_tokenhub_array_extras_fail_closed(field: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """TokenHub's array-valued extras stay out of the allowance (documented as rejected).

    Args:
        field: An array-valued message field TokenHub can send.
        monkeypatch: Replaces resolution and transport with canned values.
    """
    body = _body(message={field: [{"type": "x"}]})
    with pytest.raises(ChatProviderError):
        _replay(monkeypatch, RECORDS_BY_KEY["tokenhub"], body=body)


# --- streams ---


def test_tokenhub_mid_stream_error_frame_is_a_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare ``{"error": {...}}`` frame after content is a provider error whose
    message never copies the provider's text.

    Args:
        monkeypatch: Replaces resolution and transport with canned values.
    """
    events = [
        _chunk({"role": "assistant", "content": "partial"}),
        json.dumps({"error": {"type": "upstream_error", "message": "prompt text echoed"}}),
        "[DONE]",
    ]
    stream = _replay(monkeypatch, RECORDS_BY_KEY["tokenhub"], stream_events=events)
    with pytest.raises(ChatProviderError) as excinfo:
        list(stream)
    assert excinfo.value.status_code == 502
    assert "echoed" not in str(excinfo.value)


def test_error_frames_stay_protocol_errors_without_the_record_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Records without ``error_frame_key`` keep failing closed as before.

    Args:
        monkeypatch: Replaces resolution and transport with canned values.
    """
    events = [json.dumps({"error": {"type": "x"}}), "[DONE]"]
    stream = _replay(monkeypatch, RECORDS_BY_KEY["stepfun"], stream_events=events)
    with pytest.raises(HostedChatProtocolError):
        list(stream)


def test_tokenhub_stream_with_trailing_usage_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    """TokenHub's documented stream: finish on the last content chunk, then a
    ``choices: []`` usage chunk (only when usage is requested).

    Args:
        monkeypatch: Replaces resolution and transport with canned values.
    """
    usage = dict(_USAGE, prompt_tokens_details={"cached_tokens": 0},
                 completion_tokens_details={"reasoning_tokens": 0})
    events = [
        _chunk({"role": "assistant", "reasoning_content": "hm"}),
        _chunk({"content": "ok"}, finish="stop"),
        json.dumps({"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1,
                    "model": "hy4-preview", "choices": [], "usage": usage}),
        "[DONE]",
    ]
    stream = _replay(monkeypatch, RECORDS_BY_KEY["tokenhub"], stream_events=events)
    list(stream)
    assert stream.terminal_turn.text == "ok"
    assert stream.terminal_turn.usage["total_tokens"] == 4


@pytest.mark.parametrize("key", ["mimo", "byteplus"])
def test_usage_optional_streams_complete_without_usage(
    key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MiMo and BytePlus request usage but tolerate its absence.

    Args:
        key: Registry key of a usage-optional preset.
        monkeypatch: Replaces resolution and transport with canned values.
    """
    events = [_chunk({"role": "assistant", "content": "ok"}), _chunk({}, finish="stop"), "[DONE]"]
    stream = _replay(monkeypatch, RECORDS_BY_KEY[key], stream_events=events)
    list(stream)
    assert stream.terminal_turn.usage is None


def test_stepfun_stream_drops_its_reasoning_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """StepFun's own ``reasoning`` delta field is tolerated and dropped.

    Args:
        monkeypatch: Replaces resolution and transport with canned values.
    """
    events = [
        _chunk({"role": "assistant", "reasoning": "hm"}),
        _chunk({"content": "ok"}, finish="stop"),
        json.dumps({"id": "c", "object": "chat.completion.chunk", "created": 1,
                    "model": "step-3.7-flash", "choices": [], "usage": _USAGE}),
        "[DONE]",
    ]
    stream = _replay(monkeypatch, RECORDS_BY_KEY["stepfun"], stream_events=events)
    frames = list(stream)
    assert stream.terminal_turn.text == "ok"
    assert all("reasoning" not in json.dumps(frame.get("choices", [])) for frame in frames)


# --- StepFun's env-var name (Qodo, PR #2889) ---


def _stepfun_config(env_var: str) -> dict[str, Any]:
    record = RECORDS_BY_KEY["stepfun"]
    table = dict(record.settings_defaults) | {
        "api_base_url": record.default_base_url,
        "api_key_env_var": env_var,
    }
    return {"api_settings": {"stepfun": table}, "providers": {"StepFun": ["step-3.7-flash"]}}


def test_stepfun_sample_env_name_works_via_api_key_env_var() -> None:
    """Setting ``api_key_env_var = "STEP_API_KEY"`` makes both Console readiness
    and the engine use StepFun's own sample name.
    """
    from tldw_chatbook.Chat.provider_readiness import get_provider_readiness

    environ = {"STEP_API_KEY": "sk-stepfun-sample-name"}
    config = _stepfun_config("STEP_API_KEY")
    readiness = get_provider_readiness("stepfun", config, environ=environ)
    assert readiness.ready is True
    resolution = hosted_provider_engine.resolve_hosted_request(
        RECORDS_BY_KEY["stepfun"], app_config=config, environ=environ
    )
    assert resolution.api_key == "sk-stepfun-sample-name"


def test_stepfun_default_table_does_not_silently_accept_the_bare_alias() -> None:
    """Negative control: with the shipped table, ``STEP_API_KEY`` alone is not
    a credential for readiness OR the engine (the two can no longer disagree).
    """
    from tldw_chatbook.Chat.provider_readiness import get_provider_readiness

    environ = {"STEP_API_KEY": "sk-stepfun-sample-name"}
    config = _stepfun_config("STEPFUN_API_KEY")
    assert get_provider_readiness("stepfun", config, environ=environ).ready is False
    with pytest.raises(Exception):
        hosted_provider_engine.resolve_hosted_request(
            RECORDS_BY_KEY["stepfun"], app_config=config, environ=environ
        )
