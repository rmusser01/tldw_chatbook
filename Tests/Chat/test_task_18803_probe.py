"""THROWAWAY PROBE (task-18803 Step 1): reproduce the builder-side claims.

Captures the payloads the production builders emit TODAY (pre-fix) so each
finding is recorded before any code change. Not committed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import tldw_chatbook.LLM_Calls.LLM_API_Calls as llm_calls
from tldw_chatbook.Chat.Chat_Deps import ChatBadRequestError
from tldw_chatbook.LLM_Calls.LLM_API_Calls import chat_with_openai
from tldw_chatbook.LLM_Calls.moonshot import (
    MoonshotResolution,
    build_moonshot_chat_payload,
)
from tldw_chatbook.LLM_Calls.zai import ZAIResolution, build_zai_chat_payload

SCRATCH = Path(
    "/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/"
    "e54dd542-19ea-4a66-bd45-b5fee1b5ec4c/scratchpad"
)


class _FakeResponse:
    status_code = 200

    def __init__(self, body: dict):
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._body


@pytest.fixture
def captured_payloads(monkeypatch):
    payloads: list[dict] = []

    class _FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def mount(self, *_args, **_kwargs) -> None:
            return None

        def post(self, url, headers=None, json=None, timeout=None, **_kwargs):
            payloads.append({"url": url, "payload": json})
            return _FakeResponse(
                {
                    "id": "chatcmpl_1",
                    "model": json.get("model", ""),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {},
                }
            )

    monkeypatch.setattr(llm_calls.requests, "Session", _FakeSession)
    return payloads


_MESSAGES = [{"role": "user", "content": "Reply with exactly: OK"}]


def test_finding2_gpt5_no_effort_emits_max_tokens(captured_payloads):
    """Finding 2 headline: no reasoning effort configured -> max_tokens."""
    for model in ("gpt-5", "o3", "o4-mini", "gpt-5.1"):
        chat_with_openai(
            input_data=_MESSAGES,
            api_key="test-key",
            model=model,
            streaming=False,
            max_tokens=64,
        )
        entry = captured_payloads[-1]
        print(f"\nFINDING2 {model} url={entry['url']}")
        print(f"FINDING2 {model} payload={json.dumps(entry['payload'], sort_keys=True)}")
    # Dump the exact gpt-5 built payload for the wire confirmation.
    gpt5 = captured_payloads[0]["payload"]
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "task18803_gpt5_built_payload.json").write_text(json.dumps(gpt5))


def test_finding1_family_miss_emits_sampling(captured_payloads):
    """Finding 1: a family outside the hand tuple gets temperature/top_p."""
    for model in ("gpt-6", "o5", "o5-mini"):
        chat_with_openai(
            input_data=_MESSAGES,
            api_key="test-key",
            model=model,
            streaming=False,
            max_tokens=64,
        )
        entry = captured_payloads[-1]
        print(f"\nFINDING1 {model} payload={json.dumps(entry['payload'], sort_keys=True)}")
    print(
        "FINDING1 tuple-covered gpt-5 suppression: "
        f"_is_openai_reasoning_model('gpt-5')={llm_calls._is_openai_reasoning_model('gpt-5')} "
        f"gpt-6={llm_calls._is_openai_reasoning_model('gpt-6')} "
        f"o5={llm_calls._is_openai_reasoning_model('o5')}"
    )


def _moonshot_resolution(model: str) -> MoonshotResolution:
    return MoonshotResolution(
        provider="moonshot",
        model=model,
        api_key="sk-test",
        base_url="https://api.moonshot.ai/v1",
        timeout=60.0,
        retries=0,
        retry_delay=0.0,
        streaming=False,
    )


def test_finding3_moonshot_kimi_sampling_silently_dropped():
    payload = build_moonshot_chat_payload(
        resolution=_moonshot_resolution("kimi-k3"),
        messages_payload=_MESSAGES,
        temperature=0.4,
        top_p=0.9,
        n=1,
        presence_penalty=0.1,
        frequency_penalty=0.1,
    )
    print(f"\nFINDING3 kimi-k3 payload={json.dumps(payload, sort_keys=True)}")
    control = build_moonshot_chat_payload(
        resolution=_moonshot_resolution("moonshot-v1-8k"),
        messages_payload=_MESSAGES,
        temperature=0.4,
        top_p=0.9,
        n=1,
        presence_penalty=0.1,
        frequency_penalty=0.1,
    )
    print(f"FINDING3 moonshot-v1-8k payload={json.dumps(control, sort_keys=True)}")


def test_finding4_moonshot_effort_pinned_to_literal_kimi_k3():
    ok = build_moonshot_chat_payload(
        resolution=_moonshot_resolution("kimi-k3"),
        messages_payload=_MESSAGES,
        reasoning_effort="high",
    )
    print(f"\nFINDING4 kimi-k3 effort payload={json.dumps(ok, sort_keys=True)}")
    for model in ("kimi-k3-turbo", "kimi-k4", "kimi-latest"):
        with pytest.raises(ChatBadRequestError) as excinfo:
            build_moonshot_chat_payload(
                resolution=_moonshot_resolution(model),
                messages_payload=_MESSAGES,
                reasoning_effort="high",
            )
        print(f"FINDING4 {model} -> {type(excinfo.value).__name__}: {excinfo.value}")


def _zai_resolution(model: str) -> ZAIResolution:
    return ZAIResolution(
        provider="zai",
        model=model,
        api_key="sk-test",
        base_url="https://api.z.ai/api/paas/v4",
        timeout=60.0,
        retries=0,
        retry_delay=0.0,
        streaming=False,
    )


def test_finding4b_zai_effort_pinned_to_literal_glm_5_2():
    ok = build_zai_chat_payload(
        resolution=_zai_resolution("glm-5.2"),
        messages_payload=_MESSAGES,
        reasoning_effort="medium",
    )
    print(f"\nFINDING4B glm-5.2 effort payload={json.dumps(ok, sort_keys=True)}")
    for model in ("glm-5.3", "glm-6", "glm-5.2-air"):
        with pytest.raises(ChatBadRequestError) as excinfo:
            build_zai_chat_payload(
                resolution=_zai_resolution(model),
                messages_payload=_MESSAGES,
                reasoning_effort="medium",
            )
        print(f"FINDING4B {model} -> {type(excinfo.value).__name__}: {excinfo.value}")


def test_adjacent_zai_thinking_unconditional():
    plain = build_zai_chat_payload(
        resolution=_zai_resolution("glm-5.2"),
        messages_payload=_MESSAGES,
    )
    other = build_zai_chat_payload(
        resolution=_zai_resolution("glm-4.6"),
        messages_payload=_MESSAGES,
    )
    print(f"\nADJACENT glm-5.2 thinking={json.dumps(plain.get('thinking'))}")
    print(f"ADJACENT glm-4.6 thinking={json.dumps(other.get('thinking'))}")
