"""Opt-in live validation for provider thinking/reasoning API contracts.

These tests intentionally skip unless both the provider API key and an explicit
thinking-capable model id are supplied. They are for local release validation,
not default CI.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from tldw_chatbook.LLM_Calls import LLM_API_Calls


pytestmark = [pytest.mark.integration, pytest.mark.optional, pytest.mark.slow]


def _required_env(*names: str) -> dict[str, str]:
    values = {name: os.environ.get(name, "").strip() for name in names}
    missing = [name for name, value in values.items() if not value]
    if missing:
        pytest.skip("Set " + ", ".join(missing) + " to run live provider validation.")
    return values


def _assistant_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    assert isinstance(choices, list) and choices, response
    message = choices[0].get("message")
    assert isinstance(message, dict), response
    content = message.get("content")
    assert isinstance(content, str), response
    return content.strip()


def test_live_openai_reasoning_model_accepts_responses_reasoning(monkeypatch):
    env = _required_env("OPENAI_API_KEY", "TLDW_LIVE_OPENAI_REASONING_MODEL")
    base_url = os.environ.get(
        "TLDW_LIVE_OPENAI_API_BASE_URL",
        "https://api.openai.com/v1",
    )
    reasoning_effort = os.environ.get("TLDW_LIVE_OPENAI_REASONING_EFFORT", "low")
    reasoning_summary = os.environ.get("TLDW_LIVE_OPENAI_REASONING_SUMMARY", "none")
    verbosity = os.environ.get("TLDW_LIVE_OPENAI_VERBOSITY", "").strip() or None

    monkeypatch.setattr(
        LLM_API_Calls,
        "load_settings",
        lambda: {"openai_api": {"api_base_url": base_url}},
    )

    response = LLM_API_Calls.chat_with_openai(
        input_data=[
            {
                "role": "user",
                "content": "Return one short sentence confirming OpenAI reasoning works.",
            }
        ],
        api_key=env["OPENAI_API_KEY"],
        model=env["TLDW_LIVE_OPENAI_REASONING_MODEL"],
        streaming=False,
        max_tokens=96,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        verbosity=verbosity,
    )

    assert response["object"] == "chat.completion"
    assert response["model"]
    assert _assistant_text(response)


def test_live_anthropic_thinking_model_accepts_thinking(monkeypatch):
    env = _required_env("ANTHROPIC_API_KEY", "TLDW_LIVE_ANTHROPIC_THINKING_MODEL")
    base_url = os.environ.get(
        "TLDW_LIVE_ANTHROPIC_API_BASE_URL",
        "https://api.anthropic.com/v1",
    )
    thinking_effort = os.environ.get("TLDW_LIVE_ANTHROPIC_THINKING_EFFORT", "low")
    thinking_budget_tokens = os.environ.get(
        "TLDW_LIVE_ANTHROPIC_THINKING_BUDGET_TOKENS",
        "",
    ).strip()

    monkeypatch.setattr(
        LLM_API_Calls,
        "load_settings",
        lambda: {"anthropic_api": {"api_base_url": base_url}},
    )

    response = LLM_API_Calls.chat_with_anthropic(
        input_data=[
            {
                "role": "user",
                "content": "Return one short sentence confirming Anthropic thinking works.",
            }
        ],
        api_key=env["ANTHROPIC_API_KEY"],
        model=env["TLDW_LIVE_ANTHROPIC_THINKING_MODEL"],
        streaming=False,
        temp=0.2,
        topp=0.8,
        topk=40,
        max_tokens=4096,
        thinking_effort=thinking_effort,
        thinking_budget_tokens=int(thinking_budget_tokens)
        if thinking_budget_tokens
        else None,
    )

    assert response["object"] == "chat.completion"
    assert response["model"]
    assert _assistant_text(response)
