"""Gateway and host presets from the Hermes / oh-my-pi comparison (TASK-33351).

Thirteen OpenAI Chat Completions providers that NousResearch/hermes-agent
and/or can1357/oh-my-pi support: Vercel AI Gateway, ZenMux, Kilo Gateway,
SiliconFlow, Baseten, GMI Cloud, Ollama Cloud, Upstage, Arcee AI, Baidu
Qianfan, Nous Research, Venice and Meta's Model API. Each record is built
from the provider's public documentation (read 2026-09-28) plus
unauthenticated probes; the bodies below use the shapes those docs publish,
so every documented extra parses and anything else still fails closed.
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
from tldw_chatbook.Chat.Chat_Deps import ChatProviderError
from tldw_chatbook.LLM_Calls.hosted_chat import HostedChatProtocolError
from tldw_chatbook.LLM_Calls.hosted_provider_engine import build_hosted_chat_payload
from tldw_chatbook.provider_registry import RECORDS_BY_KEY

_EXCLUDE = {"reasoning": {"exclude": True}}

# key -> (config key, base URL, env var, models URL or None, tools, reasoning)
EXPECTED: dict[str, dict[str, Any]] = {
    "vercel": dict(config_key="Vercel", base="https://ai-gateway.vercel.sh/v1", env="AI_GATEWAY_API_KEY",
                   models="https://ai-gateway.vercel.sh/v1/models", tools=True, reasoning="ignored",
                   extra_body=_EXCLUDE, message={"reasoning"}, include_usage=True, usage_optional=True),
    "zenmux": dict(config_key="ZenMux", base="https://zenmux.ai/api/v1", env="ZENMUX_API_KEY",
                   models="https://zenmux.ai/api/v1/models", tools=True, reasoning="ignored",
                   extra_body=_EXCLUDE, response={"service_tier"},
                   message={"refusal", "reasoning", "reasoning_content"}, errors={"content_filter"},
                   include_usage=True),
    "kilo": dict(config_key="Kilo", base="https://api.kilo.ai/api/gateway", env="KILO_API_KEY",
                 models="https://api.kilo.ai/api/gateway/models", tools=True, reasoning="ignored",
                 response={"error"}, errors={"error"}),
    "siliconflow": dict(config_key="SiliconFlow", base="https://api.siliconflow.com/v1", env="SILICONFLOW_API_KEY",
                        models="https://api.siliconflow.com/v1/models", tools=True, reasoning="proprietary",
                        terminal={"eos"}, include_usage=True, usage_optional=True),
    "baseten": dict(config_key="Baseten", base="https://inference.baseten.co/v1", env="BASETEN_API_KEY",
                    models="https://inference.baseten.co/v1/models", tools=True, reasoning="proprietary",
                    choice={"stop_reason", "logprobs"}, include_usage=True),
    "gmi": dict(config_key="GMI", base="https://api.gmi-serving.com/v1", env="GMI_API_KEY",
                models="https://api.gmi-serving.com/v1/models", tools=True, reasoning="ignored"),
    "ollama_cloud": dict(config_key="OllamaCloud", base="https://ollama.com/v1", env="OLLAMA_API_KEY",
                         models="https://ollama.com/v1/models", tools=True, reasoning="ignored",
                         response={"timings"}, message={"reasoning"}, include_usage=True),
    "upstage": dict(config_key="Upstage", base="https://api.upstage.ai/v1", env="UPSTAGE_API_KEY",
                    models=None, tools=True, reasoning="ignored", choice={"logprobs"},
                    message={"reasoning"}, include_usage=True, usage_optional=True),
    "arcee": dict(config_key="Arcee", base="https://api.arcee.ai/api/v1", env="ARCEE_API_KEY",
                  models="https://api.arcee.ai/api/v1/models", tools=True, reasoning="proprietary",
                  include_usage=True, usage_optional=True),
    "qianfan": dict(config_key="Qianfan", base="https://qianfan.baidubce.com/v2", env="QIANFAN_API_KEY",
                    models=None, tools=True, reasoning="proprietary", response={"search_results"},
                    choice={"flag", "ban_round"}, errors={"content_filter"}, include_usage=True),
    "nous": dict(config_key="Nous", base="https://inference-api.nousresearch.com/v1", env="NOUS_API_KEY",
                 models="https://inference-api.nousresearch.com/v1/models", tools=False, reasoning="ignored",
                 include_usage=True, usage_optional=True),
    "venice": dict(config_key="Venice", base="https://api.venice.ai/api/v1", env="VENICE_API_KEY",
                   models="https://api.venice.ai/api/v1/models", tools=True, reasoning="proprietary",
                   response={"cost", "prompt_logprobs", "venice_parameters"}, choice={"stop_reason", "logprobs"},
                   message={"refusal", "thought_signature"}, errors={"content_filter"}, include_usage=True),
    "meta": dict(config_key="Meta", base="https://api.meta.ai/v1", env="META_API_KEY",
                 models="https://api.meta.ai/v1/models", tools=True, reasoning="ignored",
                 response={"service_tier"}, choice={"logprobs"}, message={"refusal"},
                 errors={"content_filter"}, include_usage=True),
}
KEYS = tuple(EXPECTED)
_LLM_CALLS = Path(__file__).resolve().parents[2] / "tldw_chatbook" / "LLM_Calls"
_DEFAULT_TERMINAL = {"stop", "tool_calls", "length"}


# --- record shape: every value traceable to the docs (see registry comments) ---


@pytest.mark.parametrize("key", KEYS)
def test_record_matches_its_documented_contract(key: str) -> None:
    """Every record field matches the value its provider documents.

    Args:
        key: Registry key of one gateway/host preset.
    """
    record = RECORDS_BY_KEY[key]
    e = EXPECTED[key]
    assert record.engine_driven is True
    assert record.auth_scheme == "bearer"
    assert record.tolerant_response_extras is False
    assert record.config_key == e["config_key"]
    assert record.default_base_url == e["base"]
    assert record.api_key_env_var == e["env"]
    assert record.api_key_env_candidates == (e["env"],)
    assert record.native_tools is e["tools"]
    assert record.reasoning_disposition == e["reasoning"]
    assert dict(record.extra_body_fields) == e.get("extra_body", {})
    assert record.response_allowances == frozenset(e.get("response", set()))
    assert record.choice_allowances == frozenset(e.get("choice", set()))
    assert record.message_allowances == frozenset(e.get("message", set()))
    assert set(record.finish_terminal) == _DEFAULT_TERMINAL | e.get("terminal", set())
    assert set(record.finish_provider_errors) == e.get("errors", set())
    assert record.stream_include_usage is e.get("include_usage", False)
    assert record.stream_usage_optional is e.get("usage_optional", False)
    assert record.auto_refresh is (e["models"] is not None)
    assert (record.discovery_route is None) is (e["models"] is None)
    assert "model" not in record.settings_defaults


def test_meta_never_reads_the_generic_model_api_key() -> None:
    """Meta's SDKs read ``MODEL_API_KEY``; that name is too generic to send to Meta by default."""
    assert "MODEL_API_KEY" not in RECORDS_BY_KEY["meta"].api_key_env_candidates


def test_no_per_provider_module_ships() -> None:
    """The presets are data only: no ``LLM_Calls/<provider>*.py`` module."""
    for key in KEYS:
        assert not list(_LLM_CALLS.glob(f"{key}*.py")), key


@pytest.mark.parametrize("key", KEYS)
def test_dispatch_and_param_map_route_through_the_engine(key: str) -> None:
    """Each preset dispatches through the engine with the engine param map.

    Args:
        key: Registry key of one gateway/host preset.
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
        key: Registry key of one gateway/host preset.
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


@pytest.mark.parametrize("key", [k for k in KEYS if EXPECTED[k]["models"]])
def test_default_urls_discover_at_the_documented_models_route(key: str) -> None:
    """Discovery accepts each default URL and derives the documented models URL.

    Args:
        key: Registry key of a preset with a models route.
    """
    from tldw_chatbook.LLM_Provider_Catalog.openai_compatible_model_discovery import (
        build_models_url,
        supports_openai_compatible_model_discovery,
    )

    base = EXPECTED[key]["base"]
    assert supports_openai_compatible_model_discovery(key, base) is True
    assert build_models_url(base, key) == EXPECTED[key]["models"]


@pytest.mark.parametrize("key", [k for k in KEYS if not EXPECTED[k]["models"]])
def test_seeded_only_presets_refuse_discovery(key: str) -> None:
    """Upstage and Qianfan document no models route; discovery never probes one.

    Args:
        key: Registry key of a seeded-only preset.
    """
    from tldw_chatbook.LLM_Provider_Catalog.openai_compatible_model_discovery import (
        supports_openai_compatible_model_discovery,
    )

    assert supports_openai_compatible_model_discovery(key, EXPECTED[key]["base"]) is False


# --- requests ---


@pytest.mark.parametrize("key", KEYS + EXISTING_ENGINE_KEYS)
def test_stream_options_only_where_the_record_asks(key: str) -> None:
    """``stream_options`` rides only streaming payloads of records that ask.

    Args:
        key: Registry key of a gateway/host or pre-existing engine preset.
    """
    record = RECORDS_BY_KEY[key]
    streamed = build_hosted_chat_payload(
        record, resolution=_resolution(record, streaming=True),
        messages_payload=[{"role": "user", "content": "hi"}], streaming=True,
    )
    if record.stream_include_usage:
        assert streamed["stream_options"] == {"include_usage": True}
    else:
        assert "stream_options" not in streamed


@pytest.mark.parametrize("key", ["vercel", "zenmux"])
def test_gateways_ask_to_exclude_reasoning(key: str) -> None:
    """Vercel/ZenMux reasoning includes a ``reasoning_details`` array the strict
    parser rejects, so every request asks the gateway to leave reasoning out.

    Args:
        key: Registry key of a gateway that documents ``reasoning.exclude``.
    """
    record = RECORDS_BY_KEY[key]
    payload = build_hosted_chat_payload(
        record, resolution=_resolution(record, streaming=False),
        messages_payload=[{"role": "user", "content": "hi"}], streaming=False,
    )
    assert payload["reasoning"] == {"exclude": True}


# --- responses: documented extras parse, undocumented ones fail closed ---

_DOCUMENTED_BODIES: dict[str, dict[str, Any]] = {
    "vercel": _body(message={"reasoning": None}),
    "zenmux": _body(top={"service_tier": "default"},
                    message={"refusal": None, "reasoning": None, "reasoning_content": None}),
    "kilo": _body(),
    "siliconflow": _body(message={"reasoning_content": "thinking"}),
    "baseten": _body(choice={"stop_reason": None, "logprobs": None},
                     message={"reasoning_content": "thinking"}),
    "gmi": _body(),
    "ollama_cloud": _body(top={"timings": {"prompt_n": 3, "predicted_n": 1}},
                          message={"reasoning": "thinking"}),
    "upstage": _body(choice={"logprobs": None}, message={"reasoning": "thinking"}),
    "arcee": _body(message={"reasoning_content": "thinking"}),
    "qianfan": _body(top={"search_results": {"hits": [{"index": 1, "url": "https://a", "title": "t"}]}},
                     choice={"flag": 0, "ban_round": 0}, message={"reasoning_content": "thinking"}),
    "nous": _body(),
    "venice": _body(top={"cost": {"usd": 0.00042, "diem": 0}, "prompt_logprobs": None,
                         "venice_parameters": {"web_search_citations": []}},
                    choice={"stop_reason": None, "logprobs": None},
                    message={"reasoning_content": None, "refusal": None, "thought_signature": None}),
    "meta": _body(top={"service_tier": "default"}, choice={"logprobs": None},
                  message={"refusal": None, "reasoning_content": ""}),
}


@pytest.mark.parametrize("key", KEYS)
def test_documented_response_shape_parses(key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A body carrying every documented extra parses into a normal turn.

    Args:
        key: Registry key of one gateway/host preset.
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
        key: Registry key of one gateway/host preset.
        monkeypatch: Replaces resolution and transport with canned values.
    """
    with pytest.raises((HostedChatProtocolError, ChatProviderError)):
        _replay(monkeypatch, RECORDS_BY_KEY[key], body=_body(top={"undocumented_extra": 1}))


@pytest.mark.parametrize("key", ["vercel", "zenmux", "venice"])
def test_reasoning_details_arrays_still_fail_closed(key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The documented ``reasoning_details`` array stays outside every allowance.

    Args:
        key: Registry key of a provider that documents the array.
        monkeypatch: Replaces resolution and transport with canned values.
    """
    with pytest.raises(ChatProviderError):
        _replay(monkeypatch, RECORDS_BY_KEY[key],
                body=_body(message={"reasoning_details": [{"type": "reasoning.text", "text": "t"}]}))


@pytest.mark.parametrize("key", ["zenmux", "qianfan", "venice", "meta"])
def test_content_filter_finish_is_a_provider_error(key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A documented ``content_filter`` finish is a 502, not a partial reply.

    Args:
        key: Registry key of a preset that documents ``content_filter``.
        monkeypatch: Replaces resolution and transport with canned values.
    """
    with pytest.raises(ChatProviderError) as excinfo:
        _replay(monkeypatch, RECORDS_BY_KEY[key], body=_body(finish="content_filter"))
    assert excinfo.value.status_code == 502


def test_siliconflow_eos_finish_is_a_normal_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """SiliconFlow documents ``eos`` alongside stop/length/tool_calls.

    Args:
        monkeypatch: Replaces resolution and transport with canned values.
    """
    result = _replay(monkeypatch, RECORDS_BY_KEY["siliconflow"], body=_body(finish="eos"))
    assert result.terminal_turn.finish_reason == "eos"


# --- streams ---


def test_kilo_mid_stream_error_frame_is_a_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Kilo's documented failure frame (top-level ``error`` + ``finish_reason:
    "error"``) becomes a provider error, never a truncated reply.

    Args:
        monkeypatch: Replaces resolution and transport with canned values.
    """
    events = [
        _chunk({"role": "assistant", "content": "partial"}),
        json.dumps({"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1,
                    "model": "doc-model", "error": {"message": "Provider disconnected", "code": 502},
                    "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}]}),
        "[DONE]",
    ]
    stream = _replay(monkeypatch, RECORDS_BY_KEY["kilo"], stream_events=events)
    with pytest.raises(ChatProviderError) as excinfo:
        list(stream)
    assert excinfo.value.status_code == 502


def test_kilo_stream_with_trailing_usage_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Kilo injects include_usage itself: a ``choices: []`` usage chunk ends the stream.

    Args:
        monkeypatch: Replaces resolution and transport with canned values.
    """
    events = [
        _chunk({"role": "assistant", "content": "ok"}, finish="stop"),
        json.dumps({"id": "chatcmpl-1", "object": "chat.completion.chunk", "usage": _USAGE, "choices": []}),
        "[DONE]",
    ]
    stream = _replay(monkeypatch, RECORDS_BY_KEY["kilo"], stream_events=events)
    list(stream)
    assert stream.terminal_turn.usage == _USAGE


@pytest.mark.parametrize("key", [k for k in KEYS if EXPECTED[k].get("usage_optional")])
def test_usage_optional_streams_complete_without_usage(key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Providers whose streamed usage is unconfirmed tolerate its absence.

    Args:
        key: Registry key of a usage-optional preset.
        monkeypatch: Replaces resolution and transport with canned values.
    """
    events = [_chunk({"role": "assistant", "content": "ok"}), _chunk({}, finish="stop"), "[DONE]"]
    stream = _replay(monkeypatch, RECORDS_BY_KEY[key], stream_events=events)
    list(stream)
    assert stream.terminal_turn.usage is None


@pytest.mark.parametrize("key", [k for k in KEYS if not EXPECTED[k].get("usage_optional")])
def test_strict_records_still_reject_a_stream_without_usage(key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Providers that document streamed usage still require it.

    Args:
        key: Registry key of a preset without ``stream_usage_optional``.
        monkeypatch: Replaces resolution and transport with canned values.
    """
    events = [_chunk({"role": "assistant", "content": "ok"}), _chunk({}, finish="stop"), "[DONE]"]
    stream = _replay(monkeypatch, RECORDS_BY_KEY[key], stream_events=events)
    with pytest.raises(HostedChatProtocolError):
        list(stream)
