# Tests/LLM_Provider_Catalog/test_local_openai_compatible_provider_name.py
"""Regression: custom-openai-api / local-llm handlers must reach the shared
OpenAI-compatible helper with a *string* provider_name.

Both handlers passed ``cfg.capitalize()`` where ``cfg`` is the settings
DICT, so every call raised ``AttributeError`` before any request was made —
the providers were unusable end-to-end (surfaced by the task-243
native-tools live gate, which routes through custom-openai-api).
"""
from unittest.mock import patch

from tldw_chatbook.LLM_Calls import LLM_API_Calls_Local as local_calls

_MESSAGES = [{"role": "user", "content": "hi"}]


def _run_handler_capturing_kwargs(handler, settings_payload):
    captured = {}

    def fake_helper(**kwargs):
        captured.update(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    with patch.object(local_calls, "_chat_with_openai_compatible_local_server",
                      side_effect=fake_helper), \
         patch.object(local_calls, "settings", settings_payload):
        result = handler(input_data=_MESSAGES, model="test-model")
    return captured, result


def test_custom_openai_passes_string_provider_name():
    captured, result = _run_handler_capturing_kwargs(
        local_calls.chat_with_custom_openai,
        {"api_settings": {"custom": {"api_url": "http://127.0.0.1:9/v1"}}})
    assert isinstance(captured["provider_name"], str)
    assert captured["provider_name"] == "Custom OpenAI"
    assert result["choices"][0]["message"]["content"] == "ok"


def test_local_llm_passes_string_provider_name():
    captured, result = _run_handler_capturing_kwargs(
        local_calls.chat_with_local_llm,
        {"local-llm": {"api_ip": "http://127.0.0.1:9/v1"}})
    assert isinstance(captured["provider_name"], str)
    assert captured["provider_name"] == "Local-LLM"
    assert result["choices"][0]["message"]["content"] == "ok"
