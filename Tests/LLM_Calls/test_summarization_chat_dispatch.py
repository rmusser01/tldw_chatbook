"""TASK-34895: the chat-dispatch core for summarization.

`_summarize_via_chat_api` routes the OpenAI-wire cloud summarizers through
`chat_api_call` (the unified chat provider dispatch) instead of per-provider
HTTP copies. These tests pin the core's contract per spec: endpoint/model/
prompt-assembly forwarding, the legacy in-band error strings (the ingest
failure-marker contract), the dict-summary short-circuit, and the SSE-to-
text-chunk translation for streaming.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

import tldw_chatbook.LLM_Calls.Summarization_General_Lib as sgl
from tldw_chatbook.Chat.Chat_Deps import (
    ChatAPIError,
    ChatAuthenticationError,
    ChatConfigurationError,
)


class _FakeChatDispatch:
    def __init__(self, result=None, exc=None):
        self.calls: List[Dict[str, Any]] = []
        self._result = result
        self._exc = exc

    def __call__(self, api_endpoint, messages_payload, **kwargs):
        self.calls.append(
            {
                "endpoint": api_endpoint,
                "messages": messages_payload,
                "kwargs": kwargs,
            }
        )
        if self._exc is not None:
            raise self._exc
        return self._result


def _install(monkeypatch, fake: _FakeChatDispatch, cfg_key: Any = "cfg-key") -> None:
    # The core imports chat_api_call lazily from the module, so patching the
    # module attribute intercepts every call regardless of import timing.
    monkeypatch.setattr("tldw_chatbook.Chat.Chat_Functions.chat_api_call", fake)

    def fake_setting(section, key, *args, **kwargs):
        if key == "api_key":
            return cfg_key
        return args[0] if args else kwargs.get("default")

    # Key/model sourcing reads the legacy config section through this
    # module-level seam; pin both the configured-key and fallback paths.
    monkeypatch.setattr(sgl, "get_cli_setting", fake_setting)


SPEC_BY_NAME = dict(sgl._SUMMARIZER_CHAT_SPECS)


@pytest.mark.parametrize("name", list(SPEC_BY_NAME))
def test_core_forwards_prompt_model_and_defaults(monkeypatch, name):
    spec = SPEC_BY_NAME[name]
    fake = _FakeChatDispatch(
        result={"choices": [{"message": {"content": " Summary text. "}}]}
    )
    _install(monkeypatch, fake)

    out = sgl._summarize_via_chat_api(
        name, "key-1", "BODY", "Summarize.", None, None, False
    )

    assert out == "Summary text."
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["endpoint"] == spec.endpoint
    assert call["kwargs"]["model"] == spec.default_model
    assert call["kwargs"]["temp"] == spec.default_temp
    assert call["kwargs"]["streaming"] is False
    assert call["kwargs"]["system_message"] == spec.system_default
    user = call["messages"][0]["content"]
    if spec.prompt_style == "mistral":
        assert user == "Summarize.\n\n\n\nBODY "
    elif spec.prompt_style == "huggingface":
        assert user == "Summarize.\n\n\nBODY"
    elif spec.prompt_style == "template_nl4":
        assert user == "Summarize.\n\n\n\nBODY"
    elif spec.prompt_style in ("template_nl", "ollama"):
        assert user == "Summarize.\n\nBODY"
    else:
        assert user == "BODY \n\n\n\nSummarize."


def test_core_passes_explicit_temp_and_system_and_model(monkeypatch):
    fake = _FakeChatDispatch(result={"choices": [{"message": {"content": "ok"}}]})
    _install(monkeypatch, fake)

    sgl._summarize_via_chat_api("groq", None, "BODY", "cp", 0.5, "CUSTOM SYSTEM", False)

    call = fake.calls[0]
    assert call["kwargs"]["temp"] == 0.5
    assert call["kwargs"]["system_message"] == "CUSTOM SYSTEM"
    # No parameter key: the legacy-section key is resolved in the core.
    assert call["kwargs"]["api_key"] == "cfg-key"


def test_core_refuses_keyless_before_the_chat_dispatch(monkeypatch):
    fake = _FakeChatDispatch()
    _install(monkeypatch, fake, cfg_key=None)

    for name, spec in SPEC_BY_NAME.items():
        if not spec.require_key:
            continue
        assert (
            sgl._summarize_via_chat_api(name, None, "B", "cp", None, None, False)
            == spec.missing_key_message
        )
    # Optional-key locals with NO declared key proceed to the chat
    # dispatch; a declared-but-ciphertext key still refuses in-band.
    assert fake.calls == []
    # Ciphertext refusal is pinned through the real dispatch path in
    # Tests/Chat/test_encrypted_api_key_never_sent.py (_dispatch_to_api
    # strips `enc:` before the core; chat_api_call strips again).


def test_core_preserves_legacy_error_strings(monkeypatch):
    # Handler-side failures -- including auth (a valid-but-rejected key)
    # and configuration (a missing local URL/model) -- all use the generic
    # per-spec template; the pre-dispatch key resolution above is the only
    # source of the missing-key literal, so real failures report what
    # actually happened instead of masquerading as "no key".
    for exc in (ChatAuthenticationError("no key"), ChatConfigurationError("no cfg")):
        fake = _FakeChatDispatch(exc=exc)
        _install(monkeypatch, fake)
        assert sgl._summarize_via_chat_api(
            "groq", None, "B", "cp", None, None, False
        ) == (f"Groq: Error occurred while processing summary with Groq: {exc}")

    # Other provider failures use the per-spec template.
    fake = _FakeChatDispatch(exc=ChatAPIError("status 503"))
    _install(monkeypatch, fake)
    assert sgl._summarize_via_chat_api(
        "deepseek", None, "B", "cp", None, None, False
    ) == ("DeepSeek: Error occurred while processing summary: status 503")


def test_core_plain_empty_completion_returns_empty_string(monkeypatch):
    # A completion with choices but no content and no reasoning evidence
    # returns "" (the openai-family legacy behavior) so downstream
    # falsy-summary fallbacks fire instead of storing a pseudo-error.
    fake = _FakeChatDispatch(result={"choices": [{"message": {"content": ""}}]})
    _install(monkeypatch, fake)

    assert (
        sgl._summarize_via_chat_api("mistral", None, "B", "cp", None, None, False) == ""
    )


def test_core_reasoning_only_completion_returns_diagnostic(monkeypatch):
    fake = _FakeChatDispatch(
        result={
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": "", "reasoning_content": "thought " * 10},
                }
            ],
            "usage": {"completion_tokens": 4096},
        }
    )
    _install(monkeypatch, fake)

    out = sgl._summarize_via_chat_api("llama.cpp", None, "B", "cp", None, None, False)

    assert out.startswith("Llama: No choices in response data")
    assert "finish_reason=length" in out
    assert "completion_tokens=4096" in out
    assert "reasoning-only completion" in out


def test_core_dict_with_summary_short_circuits(monkeypatch):
    fake = _FakeChatDispatch()
    _install(monkeypatch, fake)

    out = sgl._summarize_via_chat_api(
        "groq", None, {"summary": "PREEXISTING"}, "cp", None, None, False
    )

    assert out == "PREEXISTING"
    assert fake.calls == []


def test_core_translates_sse_frames_to_text_chunks(monkeypatch):
    frames = [
        'data: {"choices": [{"delta": {"content": "Hel"}}]}\n\n',
        ": keep-alive\n\n",
        'data: {"choices": [{"delta": {"content": "lo"}}]}\n\n',
        "data: [DONE]\n\n",
    ]
    fake = _FakeChatDispatch(result=iter(frames))
    _install(monkeypatch, fake)

    out = sgl._summarize_via_chat_api("groq", None, "B", "cp", None, None, True)

    chunks = [c for c in out]
    assert chunks == ["Hel", "lo"]
    assert fake.calls[0]["kwargs"]["streaming"] is True


def test_core_stream_error_frame_becomes_error_chunk(monkeypatch):
    frames = [
        'data: {"choices": [{"delta": {"content": "par"}}]}\n\n',
        'data: {"error": {"message": "boom"}}\n\n',
        'data: {"choices": [{"delta": {"content": "never"}}]}\n\n',
    ]
    fake = _FakeChatDispatch(result=iter(frames))
    _install(monkeypatch, fake)

    out = sgl._summarize_via_chat_api("openrouter", None, "B", "cp", None, None, True)

    assert [c for c in out] == ["par", "Error during streaming: boom"]


def test_dispatch_routes_the_four_providers_to_the_core(monkeypatch):
    fake = _FakeChatDispatch(result={"choices": [{"message": {"content": "ok"}}]})
    _install(monkeypatch, fake)

    # The chat-dispatch spelling "mistralai" and the plain spellings all land
    # on the core via _dispatch_to_api.
    for api_name in ("groq", "deepseek", "mistral", "mistralai", "openrouter"):
        result = sgl._dispatch_to_api(
            "Body text.", "Summarize.", api_name, "key-1", 0.7, None, False
        )
        assert result == "ok", api_name
        assert fake.calls, api_name

    endpoints = [c["endpoint"] for c in fake.calls]
    assert endpoints == ["groq", "deepseek", "mistral", "mistral", "openrouter"]


def test_dispatch_core_failure_stays_in_band(monkeypatch):
    fake = _FakeChatDispatch(exc=ChatAPIError("status 500"))
    _install(monkeypatch, fake)

    result = sgl._dispatch_to_api(
        "Body text.", "Summarize.", "groq", None, None, None, False
    )

    assert isinstance(result, str)
    assert result.startswith("Groq: Error occurred")


@pytest.mark.parametrize("spec_key", list(sgl._SUMMARIZER_CHAT_SPECS))
def test_dispatch_routes_every_spec_key(monkeypatch, spec_key):
    """Review round 1 (engine presets were unreachable): every spec key
    must have a dispatch row -- a spec without one is dead code that only
    fails at runtime with 'Invalid API Name'."""
    routed: list[str] = []

    def fake_core(key, *args, **kwargs):
        routed.append(key)
        return "ok"

    monkeypatch.setattr(sgl, "_summarize_via_chat_api", fake_core)

    result = sgl._dispatch_to_api(
        "Body text.", "Summarize.", spec_key, "key-1", 0.7, None, False
    )

    assert result == "ok"
    assert routed == [spec_key]


_LEGACY_MATCHING_MISSING_KEY = {
    # Providers whose LEGACY keyless literal the Web_Scraping failure
    # detector recognized (an "Error:" prefix or an "error occurred"
    # clause). Most legacy literals ("API Key Not Provided...") were never
    # matched -- preserving them verbatim is behavior parity, not a
    # regression; these must keep matching.
    "openai",
    "openrouter",
    "huggingface",
}
# The custom endpoints' legacy failure phrasing ("Failed to process chat
# response. Status code: ...") also never matched the markers -- parity.
_TEMPLATE_MARKER_EXEMPT = {"custom-openai-api", "custom-openai-api-2"}


@pytest.mark.parametrize("spec_key", list(sgl._SUMMARIZER_CHAT_SPECS))
def test_failure_strings_match_the_failure_markers_where_legacy_did(spec_key):
    """Pin the Web_Scraping failure-marker contract: where a legacy string
    was recognized by _is_summary_failure, the core's must be too; where
    legacy stored the failure text verbatim, parity is preserved. Plain
    empty completions return "" (falsy), which the detector's callers
    treat as failure separately."""
    from tldw_chatbook.Web_Scraping.WebSearch_APIs import _is_summary_failure

    spec = sgl._SUMMARIZER_CHAT_SPECS[spec_key]
    if spec_key in _LEGACY_MATCHING_MISSING_KEY:
        assert _is_summary_failure(spec.missing_key_message), spec_key
    if spec_key not in _TEMPLATE_MARKER_EXEMPT:
        assert _is_summary_failure(spec.error_template.format(detail="status 503")), (
            spec_key
        )
