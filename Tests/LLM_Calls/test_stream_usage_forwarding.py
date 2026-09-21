"""Streaming token-usage conversion for Gemini and Cohere (TASK-32805.2).

Unit tests for the pure OpenAI-usage translators. The end-to-end assertions
(usage forwarded on a real stream) live in
``Tests/Chat/test_cohere_native_tools.py`` and
``Tests/Chat/test_google_native_tools.py`` -- they go through ``chat_api_call``,
which touches the storage-admission path, so they run in CI rather than a clean
worktree; these cover the conversion locally.
"""

from tldw_chatbook.LLM_Calls.LLM_API_Calls import (
    _cohere_usage_to_openai,
    _gemini_usage_to_openai,
)


def test_cohere_usage_prefers_actual_tokens_over_billed_units():
    usage = {
        "billed_units": {"input_tokens": 100, "output_tokens": 200},
        "tokens": {"input_tokens": 11, "output_tokens": 5},
    }
    assert _cohere_usage_to_openai(usage) == {
        "prompt_tokens": 11,
        "completion_tokens": 5,
        "total_tokens": 16,
    }


def test_cohere_usage_falls_back_to_billed_units():
    assert _cohere_usage_to_openai(
        {"billed_units": {"input_tokens": 7, "output_tokens": 3}}
    ) == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}


def test_cohere_usage_is_none_when_absent():
    assert _cohere_usage_to_openai({}) is None
    assert _cohere_usage_to_openai(None) is None


def test_gemini_usage_maps_openai_fields():
    assert _gemini_usage_to_openai(
        {"promptTokenCount": 12, "candidatesTokenCount": 4, "totalTokenCount": 16}
    ) == {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16}


def test_gemini_usage_defaults_total_when_absent():
    assert _gemini_usage_to_openai(
        {"promptTokenCount": 12, "candidatesTokenCount": 4}
    ) == {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16}


def test_gemini_usage_is_none_when_absent():
    assert _gemini_usage_to_openai({}) is None
    assert _gemini_usage_to_openai(None) is None
