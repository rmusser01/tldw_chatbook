"""THROWAWAY LIVE CHECK (task-18803 Step 3): production chat path, real keys.

Seam-level through `chat_api_call` (the production dispatcher) -> the real
provider handlers -> the real APIs. Uncommitted; run explicitly. Keys come
from the repo-root *-api-key.txt files (agent-use per project memory).
Config isolation comes from Tests/conftest.py's bootstrap scratch
TLDW_CONFIG_PATH/HOME.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tldw_chatbook.Chat.Chat_Functions import chat_api_call

_REPO_ROOT = Path("/Users/macbook-dev/Documents/GitHub/tldw_chatbook")
_MESSAGES = [{"role": "user", "content": "Reply with exactly: OK"}]


def _key(name: str) -> str:
    return (_REPO_ROOT / name).read_text().strip()


def _content(response: object) -> str:
    assert isinstance(response, dict), f"non-dict response: {response!r}"
    choices = response.get("choices")
    assert choices, f"no choices in response: {response!r}"
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    print(f"\nLIVE model={response.get('model')} id={response.get('id')} "
          f"finish={choices[0].get('finish_reason')} content={content!r}")
    return content


def test_live_gpt5_no_reasoning_effort_completes():
    """The headline fix: gpt-5 + max_tokens, NO reasoning effort."""
    response = chat_api_call(
        api_endpoint="openai",
        messages_payload=_MESSAGES,
        api_key=_key("openai-api-key.txt"),
        model="gpt-5",
        streaming=False,
        max_tokens=900,
    )
    assert _content(response)


def test_live_gpt5_with_reasoning_effort_completes():
    response = chat_api_call(
        api_endpoint="openai",
        messages_payload=_MESSAGES,
        api_key=_key("openai-api-key.txt"),
        model="gpt-5",
        streaming=False,
        reasoning_effort="low",
        max_tokens=900,
    )
    assert _content(response)


def test_live_gpt4o_control_completes():
    response = chat_api_call(
        api_endpoint="openai",
        messages_payload=_MESSAGES,
        api_key=_key("openai-api-key.txt"),
        model="gpt-4o",
        streaming=False,
        max_tokens=64,
    )
    assert _content(response)


def test_live_moonshot_kimi_k26_effort_with_sampling_dropped():
    """Exercises BOTH new Moonshot predicates live: reasoning_effort=medium
    on a non-k3 kimi id (old code raised client-side) while the explicit
    temperature is dropped for the versioned kimi family (wire 400s on it)."""
    response = chat_api_call(
        api_endpoint="moonshot",
        messages_payload=_MESSAGES,
        api_key=_key("moonshot-api-key.txt"),
        model="kimi-k2.6",
        streaming=False,
        reasoning_effort="medium",
        temp=0.4,
        max_tokens=900,
    )
    assert _content(response)
