"""Follow-up presets from the Hermes / oh-my-pi comparison (TASK-33505..33509).

Azure OpenAI (v1 API), W&B Inference (CoreWeave), Cloudflare Workers AI,
OpenCode Zen and Command Code, plus the small engine capabilities they
needed: a ``max_completion_tokens`` spelling, provider annotation frames in
streams, empty-list allowlisted values, and optional config-sourced headers.
Records are built from public documentation (read 2026-09-29) plus
unauthenticated probes; the Azure shapes below come from published captures
(see the AZURE registry comment), so every documented extra parses and
anything else still fails closed.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import replace
from typing import Any

import pytest

from Tests.LLM_Calls.test_doc_derived_presets import (
    _USAGE,
    _body,
    _chunk,
    _replay,
    _resolution,
)
from tldw_chatbook.Chat.Chat_Deps import ChatConfigurationError, ChatProviderError
from tldw_chatbook.Chat.provider_readiness import get_provider_readiness
from tldw_chatbook.LLM_Calls import hosted_provider_engine
from tldw_chatbook.LLM_Calls.hosted_chat import (
    HostedChatProtocolError,
    _level_allowance_value_is_valid,
)
from tldw_chatbook.LLM_Calls.hosted_provider_engine import (
    build_hosted_chat_payload,
    resolve_hosted_request,
)
from tldw_chatbook.provider_registry import ALL_RECORDS, RECORDS_BY_KEY

# key -> (config key, base URL or None, env var, auth, models route?, headers)
EXPECTED: dict[str, dict[str, Any]] = {
    "azure": dict(config_key="Azure", base=None, env="AZURE_OPENAI_API_KEY",
                  auth="api_key_header", discovers=False, headers={}),
    "wandb": dict(config_key="WandB", base="https://api.inference.wandb.ai/v1", env="WANDB_API_KEY",
                  auth="bearer", discovers=True, headers={"OpenAI-Project": "project"}),
    "cloudflare": dict(config_key="Cloudflare", base=None, env="CLOUDFLARE_API_TOKEN",
                       auth="bearer", discovers=False, headers={"cf-aig-gateway-id": "gateway_id"}),
    "opencode_zen": dict(config_key="OpenCodeZen", base="https://opencode.ai/zen/v1", env="OPENCODE_API_KEY",
                         auth="bearer", discovers=False, headers={}),
    "commandcode": dict(config_key="CommandCode", base="https://api.commandcode.ai/provider/v1",
                        env="COMMANDCODE_API_KEY", auth="bearer", discovers=False, headers={}),
}
KEYS = tuple(EXPECTED)
_AZURE_HOST = "https://my-resource.openai.azure.com"
_CF_URL = "https://api.cloudflare.com/client/v4/accounts/0123456789abcdef/ai/v1"
_TOOL = {
    "type": "function",
    "function": {"name": "add", "description": "Add.", "parameters": {"type": "object"}},
}


def _record_resolution(key: str, *, streaming: bool = False) -> Any:
    record = RECORDS_BY_KEY[key]
    resolution = _resolution(record, streaming=streaming)
    base = {"azure": f"{_AZURE_HOST}/openai/v1", "cloudflare": _CF_URL}.get(key)
    return replace(resolution, base_url=base) if base else resolution


# --- record shape and wiring ---


@pytest.mark.parametrize("key", KEYS)
def test_record_matches_its_documented_contract(key: str) -> None:
    """Args:
    key: Registry key of one follow-up preset.
    """
    record = RECORDS_BY_KEY[key]
    e = EXPECTED[key]
    assert record.engine_driven is True
    assert record.tolerant_response_extras is False
    assert record.config_key == e["config_key"]
    assert record.default_base_url == e["base"]
    assert record.api_key_env_var == e["env"]
    assert record.api_key_env_candidates == (e["env"],)
    assert record.auth_scheme == e["auth"]
    assert dict(record.config_headers) == e["headers"]
    assert record.auto_refresh is e["discovers"]
    assert (record.discovery_route is None) is (not e["discovers"])
    assert "model" not in record.settings_defaults


@pytest.mark.parametrize("key", KEYS)
def test_dispatch_readiness_and_catalog_lists_carry_the_preset(key: str) -> None:
    """Args:
    key: Registry key of one follow-up preset.
    """
    from tldw_chatbook.Agents.native_tools import NATIVE_TOOLS_PROVIDERS
    from tldw_chatbook.Chat.Chat_Functions import (
        API_CALL_HANDLERS,
        ENGINE_PROVIDER_PARAM_MAP,
        PROVIDER_PARAM_MAP,
    )
    from tldw_chatbook.Chat.console_session_settings import (
        CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS,
    )
    from tldw_chatbook.Chat.provider_catalog import PROVIDER_DISPLAY_NAMES
    from tldw_chatbook.Chat.provider_readiness import PROVIDERS_REQUIRING_API_KEY_KEYS

    assert callable(API_CALL_HANDLERS[key])
    assert PROVIDER_PARAM_MAP[key] is ENGINE_PROVIDER_PARAM_MAP
    assert key in NATIVE_TOOLS_PROVIDERS
    assert key in CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS
    assert key in PROVIDERS_REQUIRING_API_KEY_KEYS
    assert PROVIDER_DISPLAY_NAMES[key] == RECORDS_BY_KEY[key].display_name


@pytest.mark.parametrize("key", KEYS)
def test_config_tables_mirror_the_record(key: str) -> None:
    """The shipped ``[api_settings]`` table and ``[providers]`` seed match the record.

    Args:
        key: Registry key of one follow-up preset.
    """
    from tldw_chatbook.config import CONFIG_TOML_CONTENT

    record = RECORDS_BY_KEY[key]
    parsed = tomllib.loads(CONFIG_TOML_CONTENT)
    shipped_url = {"api_base_url": record.default_base_url} if record.default_base_url else {}
    assert parsed["api_settings"][key] == dict(record.settings_defaults) | shipped_url
    seeds = parsed["providers"][record.config_key]
    if key in ("azure", "wandb"):
        assert seeds == []  # deployment names / discovery
    else:
        assert seeds and len(seeds) == len(set(seeds))
    assert not any(seed.startswith("claude") for seed in seeds)


def test_only_wandb_discovers_models() -> None:
    from tldw_chatbook.LLM_Provider_Catalog.openai_compatible_model_discovery import (
        build_models_url,
        supports_openai_compatible_model_discovery,
    )

    base = EXPECTED["wandb"]["base"]
    assert supports_openai_compatible_model_discovery("wandb", base) is True
    assert build_models_url(base, "wandb") == "https://api.inference.wandb.ai/v1/models"
    for key, base in (("azure", f"{_AZURE_HOST}/openai/v1"), ("cloudflare", _CF_URL),
                      ("opencode_zen", EXPECTED["opencode_zen"]["base"]),
                      ("commandcode", EXPECTED["commandcode"]["base"])):
        assert supports_openai_compatible_model_discovery(key, base) is False, key


# --- per-account base URLs (Azure, Cloudflare) ---


def test_azure_resource_host_gets_the_v1_path_and_cloudflare_url_is_kept() -> None:
    azure = resolve_hosted_request(
        RECORDS_BY_KEY["azure"], explicit_api_key="secret", explicit_model="my-deployment",
        app_config={"api_settings": {"azure": {"api_base_url": _AZURE_HOST}}}, environ={},
    )
    assert azure.base_url == f"{_AZURE_HOST}/openai/v1"
    cloudflare = resolve_hosted_request(
        RECORDS_BY_KEY["cloudflare"], explicit_api_key="secret", explicit_model="@cf/openai/gpt-oss-120b",
        app_config={"api_settings": {"cloudflare": {"api_base_url": _CF_URL}}}, environ={},
    )
    assert cloudflare.base_url == _CF_URL


@pytest.mark.parametrize(("provider", "key", "reason", "example"), [
    ("Azure", "azure", "Missing resource URL", "openai.azure.com"),
    ("Cloudflare", "cloudflare", "Missing account URL", "/accounts/<account-id>/ai/v1"),
])
def test_readiness_asks_for_the_per_account_url(provider: str, key: str, reason: str, example: str) -> None:
    """Args:
    provider: Config display key.
    key: Registry key.
    reason: Expected blocking reason.
    example: A fragment of the example URL the recovery copy must show.
    """
    missing = get_provider_readiness(provider, {"api_settings": {key: {"api_key": "sk-canary-1234"}}}, environ={})
    assert missing.ready is False
    assert missing.reason == reason
    assert example in missing.recovery
    url = _AZURE_HOST if key == "azure" else _CF_URL
    ready = get_provider_readiness(
        provider, {"api_settings": {key: {"api_key": "sk-canary-1234", "api_base_url": url}}}, environ={}
    )
    assert ready.ready is True


# --- engine: max_completion_tokens ---


def test_azure_sends_max_completion_tokens_and_others_keep_max_tokens() -> None:
    azure = build_hosted_chat_payload(
        RECORDS_BY_KEY["azure"], resolution=_record_resolution("azure"),
        messages_payload=[{"role": "user", "content": "hi"}], max_tokens=128,
    )
    assert azure["max_completion_tokens"] == 128 and "max_tokens" not in azure
    for record in ALL_RECORDS:
        if record.engine_driven and record.key != "azure":
            assert record.max_tokens_key is None, record.key


# --- engine: optional headers from config ---


@pytest.mark.parametrize(("key", "setting", "header"), [
    ("wandb", "project", "OpenAI-Project"),
    ("cloudflare", "gateway_id", "cf-aig-gateway-id"),
])
def test_config_header_is_sent_only_when_set(key: str, setting: str, header: str) -> None:
    """Args:
    key: Registry key.
    setting: The ``api_settings`` field holding the header value.
    header: The header name the provider documents.
    """
    record = RECORDS_BY_KEY[key]
    base = {"api_base_url": _CF_URL} if key == "cloudflare" else {}

    def resolve(extra: dict[str, Any]) -> Any:
        return resolve_hosted_request(
            record, explicit_api_key="secret", explicit_model="m",
            app_config={"api_settings": {key: base | extra}}, environ={},
        )

    assert resolve({setting: "team/project"}).extra_headers == {header: "team/project"}
    assert resolve({}).extra_headers == {}
    assert resolve({setting: "   "}).extra_headers == {}
    for bad in ("a\r\nX-Injected: 1", 7, "x" * 257):
        with pytest.raises(ChatConfigurationError):
            resolve({setting: bad})


def test_config_headers_reach_the_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    record = RECORDS_BY_KEY["wandb"]
    resolution = replace(_record_resolution("wandb"), extra_headers={"OpenAI-Project": "team/project"})
    monkeypatch.setattr(hosted_provider_engine, "resolve_hosted_request", lambda _r, **_k: resolution)
    captured: dict[str, Any] = {}

    def fake_post(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _body()

    monkeypatch.setattr(hosted_provider_engine, "owned_json_post", fake_post)
    hosted_provider_engine.build_hosted_chat_handler(record)(
        input_data=[{"role": "user", "content": "hi"}], api_key="secret", streaming=False,
    )
    assert captured["config"].extra_headers == {"OpenAI-Project": "team/project"}


def test_no_other_preset_sends_config_headers() -> None:
    declared = {r.key for r in ALL_RECORDS if r.config_headers}
    assert declared == {"wandb", "cloudflare"}


# --- engine: empty-list allowlisted values ---


def test_allowlisted_values_may_be_an_empty_list_but_not_a_filled_one() -> None:
    assert _level_allowance_value_is_valid([]) is True
    assert _level_allowance_value_is_valid([{"type": "url_citation"}]) is False


# --- Azure: content-filter annotations (captured shapes) ---

_AZURE_FILTER = {"hate": {"filtered": False, "severity": "safe"}}


def test_azure_captured_response_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    body = _body(
        top={"prompt_filter_results": [{"prompt_index": 0, "content_filter_results": _AZURE_FILTER}]},
        choice={"content_filter_results": _AZURE_FILTER, "logprobs": None},
        message={"refusal": None, "annotations": []},
    )
    result = _replay(monkeypatch, RECORDS_BY_KEY["azure"], body=body)
    assert result.terminal_turn.text == "ok"


def test_azure_filled_annotations_still_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    body = _body(message={"annotations": [{"type": "url_citation"}]})
    with pytest.raises((HostedChatProtocolError, ChatProviderError)):
        _replay(monkeypatch, RECORDS_BY_KEY["azure"], body=body)


def _azure_stream() -> list[str]:
    def event(**fields: Any) -> str:
        return json.dumps({"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1,
                           "model": "gpt-4.1", **fields})

    choice = {"index": 0, "content_filter_results": {}, "logprobs": None, "finish_reason": None}
    return [
        json.dumps({"choices": [], "created": 0, "id": "", "model": "", "object": "",
                    "prompt_filter_results": [{"prompt_index": 0, "content_filter_results": _AZURE_FILTER}]}),
        event(obfuscation="x1", system_fingerprint="fp_1",
              choices=[{**choice, "delta": {"role": "assistant", "content": "", "refusal": None}}]),
        event(obfuscation="x2", choices=[{**choice, "content_filter_results": _AZURE_FILTER,
                                          "delta": {"content": "ok"}}]),
        event(choices=[{**choice, "delta": {}, "finish_reason": "stop"}]),
        event(choices=[], usage=_USAGE, latency_checkpoint={"t": 1}, routing={"region": "x"}),
        "[DONE]",
    ]


def test_azure_captured_stream_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = _replay(monkeypatch, RECORDS_BY_KEY["azure"], stream_events=_azure_stream())
    list(stream)
    assert stream.terminal_turn.text == "ok"
    assert stream.terminal_turn.usage == _USAGE


def test_annotation_frame_needs_the_record_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Negative control: without ``stream_annotation_key`` the leading frame is misplaced."""
    record = replace(RECORDS_BY_KEY["azure"], stream_annotation_key=None)
    stream = _replay(monkeypatch, record, stream_events=_azure_stream())
    with pytest.raises(HostedChatProtocolError):
        list(stream)


def test_azure_content_filter_finish_is_a_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ChatProviderError) as excinfo:
        _replay(monkeypatch, RECORDS_BY_KEY["azure"], body=_body(finish="content_filter"))
    assert excinfo.value.status_code == 502


# --- documented extras of the other presets ---


@pytest.mark.parametrize(("key", "body"), [
    ("wandb", _body(message={"reasoning": None})),
    ("cloudflare", _body(choice={"logprobs": None}, message={"refusal": None, "reasoning_content": "t"})),
    ("opencode_zen", _body(top={"cost": "0.00012"}, message={"reasoning_content": "t"})),
    ("commandcode", _body(message={"refusal": None, "annotations": []})),
])
def test_documented_response_shape_parses(key: str, body: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Args:
    key: Registry key.
    body: A response carrying the provider's documented extras.
    monkeypatch: Replaces resolution and transport with canned values.
    """
    record = RECORDS_BY_KEY[key]
    resolution = _record_resolution(key)
    monkeypatch.setattr(hosted_provider_engine, "resolve_hosted_request", lambda _r, **_k: resolution)
    monkeypatch.setattr(hosted_provider_engine, "owned_json_post", lambda **_k: body)
    result = hosted_provider_engine.build_hosted_chat_handler(record)(
        input_data=[{"role": "user", "content": "hi"}], api_key="secret", streaming=False,
    )
    assert result.terminal_turn.text == "ok"


@pytest.mark.parametrize("key", KEYS)
def test_undocumented_top_level_key_still_fails_closed(key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Args:
    key: Registry key.
    monkeypatch: Replaces resolution and transport with canned values.
    """
    with pytest.raises((HostedChatProtocolError, ChatProviderError)):
        _replay(monkeypatch, RECORDS_BY_KEY[key], body=_body(top={"undocumented_extra": 1}))


def test_commandcode_streams_usage_without_asking(monkeypatch: pytest.MonkeyPatch) -> None:
    """Command Code always streams usage, so no stream_options is sent and usage is required."""
    record = RECORDS_BY_KEY["commandcode"]
    payload = build_hosted_chat_payload(
        record, resolution=_record_resolution("commandcode", streaming=True),
        messages_payload=[{"role": "user", "content": "hi"}], streaming=True, tools=[_TOOL],
    )
    assert "stream_options" not in payload
    events = [_chunk({"role": "assistant", "content": "ok"}), _chunk({}, finish="stop"), "[DONE]"]
    with pytest.raises((HostedChatProtocolError, ChatProviderError)):
        list(_replay(monkeypatch, record, stream_events=events))
