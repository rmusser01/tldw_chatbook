"""task-404: reasoning-model requests must omit unsupported sampling params.

OpenAI reasoning models (o-series, gpt-5 family) reject `temperature` and
`top_p` with HTTP 400 on both the Chat Completions and Responses APIs.
`chat_with_openai` used to inject config-backed defaults for both on every
request, so ANY call routed to a reasoning model failed. These tests pin the
payload shape at the HTTP seam for both request branches.
"""

import pytest

import tldw_chatbook.LLM_Calls.LLM_API_Calls as llm_calls
from tldw_chatbook.LLM_Calls.LLM_API_Calls import chat_with_openai


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
    """Swap ``requests.Session`` for a fake that records posted payloads."""
    payloads: list[dict] = []

    class _FakeSession:
        """Stands in for ``requests.Session`` in the non-streaming send path."""

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def mount(self, *_args, **_kwargs) -> None:
            return None

        def post(self, url, headers=None, json=None, timeout=None, **_kwargs):
            payloads.append({"url": url, "payload": json})
            if url.rstrip("/").endswith("responses"):
                return _FakeResponse(
                    {"id": "resp_1", "output_text": "ok", "output": [], "usage": {}}
                )
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


_MESSAGES = [{"role": "user", "content": "hi"}]


def test_responses_branch_omits_temperature_and_top_p(captured_payloads):
    chat_with_openai(
        input_data=_MESSAGES,
        api_key="test-key",
        model="gpt-5-mini",
        streaming=False,
        reasoning_effort="low",
    )
    payload = captured_payloads[-1]["payload"]
    assert "temperature" not in payload
    assert "top_p" not in payload
    assert payload["input"] == _MESSAGES
    assert payload["reasoning"] == {"effort": "low"}


def test_reasoning_model_on_chat_completions_omits_sampling(captured_payloads):
    chat_with_openai(
        input_data=_MESSAGES,
        api_key="test-key",
        model="o3-mini",
        streaming=False,
    )
    payload = captured_payloads[-1]["payload"]
    assert "temperature" not in payload
    assert "top_p" not in payload
    assert payload["messages"] == _MESSAGES


def test_explicit_sampling_on_reasoning_model_is_dropped(captured_payloads):
    chat_with_openai(
        input_data=_MESSAGES,
        api_key="test-key",
        model="gpt-5-mini",
        streaming=False,
        reasoning_effort="low",
        temp=0.9,
        maxp=0.5,
    )
    payload = captured_payloads[-1]["payload"]
    assert "temperature" not in payload
    assert "top_p" not in payload


def test_non_reasoning_model_keeps_default_sampling(captured_payloads):
    chat_with_openai(
        input_data=_MESSAGES,
        api_key="test-key",
        model="gpt-4o-mini",
        streaming=False,
    )
    payload = captured_payloads[-1]["payload"]
    # Today's behavior preserved: config/hardcoded defaults still included.
    assert payload["temperature"] == pytest.approx(0.7)
    assert payload["top_p"] == pytest.approx(0.95)
    assert payload["messages"] == _MESSAGES


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("o1", True),
        ("o1-mini", True),
        ("o3", True),
        ("o3-mini-2025-01-31", True),
        ("o4-mini", True),
        ("gpt-5", True),
        ("gpt-5-mini", True),
        ("gpt-5.1", True),
        ("GPT-5-NANO", True),
        ("gpt-4o-mini", False),
        ("gpt-4.1", False),
        ("olmo-7b", False),
        ("o365-copilot", False),
    ],
)
def test_is_openai_reasoning_model_boundaries(model, expected):
    assert llm_calls._is_openai_reasoning_model(model) is expected
