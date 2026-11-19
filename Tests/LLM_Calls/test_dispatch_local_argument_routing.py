"""TASK-33641.10: ``analyze`` hands each provider every argument by name.

TASK-34980 (filed as 34812) pinned three local routes in ``_dispatch_to_api``
that bound values to the wrong parameters by position (KoboldCpp: the
streaming flag onto ``kobold_api_ip``; TabbyAPI: the system message onto
``api_key``; Ollama: the streaming flag onto ``model``). TASK-34895 then
replaced the per-provider summarizers with one chat-dispatch core, so the
functions those pins patched are gone. What must still hold: through
``analyze``, the key, temperature, system message and streaming flag a caller
passes reach each chat handler under the parameter meant for them.
"""

from __future__ import annotations

import inspect

import pytest

import tldw_chatbook.LLM_Calls.Summarization_General_Lib as sgl
from tldw_chatbook.Chat import Chat_Functions

#: Distinct values, so a swap of any two arguments shows.
_ARGS = {
    "api_key": "fixed-key",
    "temp": 0.25,
    "system_message": "fixed system",
}


@pytest.fixture(autouse=True)
def _config_defaults(monkeypatch):
    """The model and temperature settings the core reads come back as their
    defaults, so no test here reads a profile."""

    def setting(_section, _key=None, default=None):
        return default

    monkeypatch.setattr(sgl, "get_cli_setting", setting)


def _analyze(api_name: str, streaming: bool):
    result = sgl.analyze(
        api_name=api_name,
        input_data="fixed text",
        custom_prompt_arg="fixed prompt",
        streaming=streaming,
        **_ARGS,
    )
    return result if isinstance(result, str) else "".join(result)


@pytest.mark.parametrize("streaming", [False, True], ids=["plain", "streaming"])
@pytest.mark.parametrize(
    ("api_name", "endpoint", "temp_parameter"),
    [
        ("kobold", "koboldcpp", "temp"),
        ("ollama", "ollama", "temperature"),
    ],
)
def test_kobold_and_ollama_receive_each_argument_under_its_own_parameter(
    monkeypatch, api_name, endpoint, temp_parameter, streaming
):
    """AC#1: bound against the real chat handler's signature."""
    real = Chat_Functions.API_CALL_HANDLERS[endpoint]
    signature = inspect.signature(real)
    bound: list[dict] = []

    def handler(**kwargs):
        bound.append(dict(signature.bind(**kwargs).arguments))
        if kwargs.get("streaming"):
            return iter(['data: {"choices": [{"delta": {"content": "SUMMARY"}}]}\n\n'])
        return {"choices": [{"message": {"content": "SUMMARY"}}]}

    monkeypatch.setitem(Chat_Functions.API_CALL_HANDLERS, endpoint, handler)

    assert _analyze(api_name, streaming) == "SUMMARY"

    (arguments,) = bound
    assert arguments["api_key"] == _ARGS["api_key"]
    assert arguments[temp_parameter] == _ARGS["temp"]
    assert arguments["system_message"] == _ARGS["system_message"]
    assert arguments["streaming"] is streaming
    assert "fixed text" in arguments["input_data"][0]["content"]
    assert "fixed prompt" in arguments["input_data"][0]["content"]


@pytest.mark.parametrize("api_name", sorted(sgl._SUMMARIZER_CHAT_SPECS))
def test_every_dispatch_name_hands_each_argument_to_its_own_parameter(
    monkeypatch, api_name
):
    """AC#2: a call in ``_dispatch_to_api`` that binds an argument to a
    parameter of another name fails here, for every provider it routes."""
    calls: list[dict] = []

    def chat_api_call(endpoint, messages, **kwargs):
        calls.append({"endpoint": endpoint, "messages": messages, **kwargs})
        return {"choices": [{"message": {"content": "SUMMARY"}}]}

    monkeypatch.setattr(Chat_Functions, "chat_api_call", chat_api_call)

    assert _analyze(api_name, False) == "SUMMARY"

    (call,) = calls
    assert call["endpoint"] == sgl._SUMMARIZER_CHAT_SPECS[api_name].endpoint
    assert {name: call[name] for name in _ARGS} == _ARGS
    assert call["streaming"] is False
    assert "fixed text" in call["messages"][0]["content"]


def test_the_mock_provider_receives_each_argument_under_its_own_parameter(
    monkeypatch,
):
    """AC#2 for the one dispatch name that is not a chat-dispatch spec."""
    signature = inspect.signature(sgl.summarize_with_mock_llm)
    bound: list[dict] = []

    def mock(*args, **kwargs):
        bound.append(dict(signature.bind(*args, **kwargs).arguments))
        return "SUMMARY"

    monkeypatch.setattr(sgl, "summarize_with_mock_llm", mock)

    assert _analyze("mock-llm", False) == "SUMMARY"
    assert bound == [
        {
            "text_to_summarize": "fixed text",
            "custom_prompt_arg": "fixed prompt",
            "streaming": False,
            **_ARGS,
        }
    ]
