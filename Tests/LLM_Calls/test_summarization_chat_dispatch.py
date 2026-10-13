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


SPEC_BY_NAME = {
    "groq": sgl._SUMMARIZER_CHAT_SPECS["groq"],
    "deepseek": sgl._SUMMARIZER_CHAT_SPECS["deepseek"],
    "mistral": sgl._SUMMARIZER_CHAT_SPECS["mistral"],
    "openrouter": sgl._SUMMARIZER_CHAT_SPECS["openrouter"],
}


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
    assert (
        call["kwargs"]["system_message"]
        == "You are a helpful AI assistant who does whatever the user requests."
    )
    user = call["messages"][0]["content"]
    if spec.prompt_first:
        assert user == "Summarize.\n\n\n\nBODY "
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

    for name in SPEC_BY_NAME:
        spec = SPEC_BY_NAME[name]
        assert (
            sgl._summarize_via_chat_api(name, None, "B", "cp", None, None, False)
            == spec.missing_key_message
        )
    assert fake.calls == []
    # Ciphertext refusal is pinned through the real dispatch path in
    # Tests/Chat/test_encrypted_api_key_never_sent.py (_dispatch_to_api
    # strips `enc:` before the core; chat_api_call strips again).


def test_core_preserves_legacy_error_strings(monkeypatch):
    # Missing key: authentication/configuration failures become the legacy
    # in-band literal.
    for exc in (ChatAuthenticationError("no key"), ChatConfigurationError("no cfg")):
        fake = _FakeChatDispatch(exc=exc)
        _install(monkeypatch, fake)
        assert (
            sgl._summarize_via_chat_api("groq", None, "B", "cp", None, None, False)
            == "Groq: API Key Not Provided/Found in Config file or is empty"
        )

    # Other provider failures use the per-spec template.
    fake = _FakeChatDispatch(exc=ChatAPIError("status 503"))
    _install(monkeypatch, fake)
    assert sgl._summarize_via_chat_api(
        "deepseek", None, "B", "cp", None, None, False
    ) == ("DeepSeek: Error occurred while processing summary: status 503")


def test_core_empty_extraction_returns_legacy_literal(monkeypatch):
    fake = _FakeChatDispatch(result={"choices": [{"message": {"content": ""}}]})
    _install(monkeypatch, fake)

    assert (
        sgl._summarize_via_chat_api("mistral", None, "B", "cp", None, None, False)
        == "Mistral: Summary not available"
    )


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
