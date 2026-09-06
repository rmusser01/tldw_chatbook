"""task-404 / TASK-18803: modern-model requests must fit the accepted surface.

OpenAI reasoning models (o-series, gpt-5 family) reject `temperature` and
`top_p` with HTTP 400 on both the Chat Completions and Responses APIs, and
their chat-completions surface rejects the classic `max_tokens` cap in favor
of `max_completion_tokens` (probe-verified in TASK-18802/18803). Both facts
are consulted from `model_capabilities` predicates rather than the
hand-maintained name lists the builder used to carry. These tests pin the
payload shape at the HTTP seam for both request branches.
"""

import pytest

import tldw_chatbook.LLM_Calls.LLM_API_Calls as llm_calls
from tldw_chatbook.LLM_Calls.LLM_API_Calls import chat_with_openai
from tldw_chatbook.model_capabilities import openai_model_rejects_sampling_params


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

    monkeypatch.setattr(llm_calls, "create_default_session", _FakeSession)
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


@pytest.mark.parametrize("model", ["gpt-5", "o3", "o4-mini", "gpt-5.1"])
def test_no_effort_modern_model_uses_max_completion_tokens(captured_payloads, model):
    """TASK-18803 headline: with NO reasoning effort configured these models
    fall through to chat-completions, where the classic ``max_tokens`` cap is
    HTTP 400 ``unsupported_parameter`` (probe-verified with the exact built
    gpt-5 payload). The builder must emit ``max_completion_tokens``."""
    chat_with_openai(
        input_data=_MESSAGES,
        api_key="test-key",
        model=model,
        streaming=False,
        max_tokens=64,
    )
    entry = captured_payloads[-1]
    assert entry["url"].rstrip("/").endswith("/chat/completions")
    payload = entry["payload"]
    assert payload["max_completion_tokens"] == 64
    assert "max_tokens" not in payload
    assert "max_output_tokens" not in payload


def test_gpt_5_6_no_effort_token_cap_unchanged(captured_payloads):
    """Control: gpt-5.6 already got ``max_completion_tokens`` pre-fix."""
    chat_with_openai(
        input_data=_MESSAGES,
        api_key="test-key",
        model="gpt-5.6-terra",
        streaming=False,
        max_tokens=64,
    )
    payload = captured_payloads[-1]["payload"]
    assert payload["max_completion_tokens"] == 64
    assert "max_tokens" not in payload


@pytest.mark.parametrize("model", ["gpt-4o", "gpt-4o-mini", "gpt-4.1"])
def test_legacy_model_keeps_max_tokens_and_sampling(captured_payloads, model):
    """AC #4 control: currently-working models keep their exact payload."""
    chat_with_openai(
        input_data=_MESSAGES,
        api_key="test-key",
        model=model,
        streaming=False,
        max_tokens=64,
    )
    payload = captured_payloads[-1]["payload"]
    assert payload["max_tokens"] == 64
    assert "max_completion_tokens" not in payload
    assert payload["temperature"] == pytest.approx(0.7)
    assert payload["top_p"] == pytest.approx(0.95)


def test_responses_branch_uses_max_output_tokens(captured_payloads):
    """Control: a configured reasoning effort still routes to /responses
    with ``max_output_tokens``."""
    chat_with_openai(
        input_data=_MESSAGES,
        api_key="test-key",
        model="gpt-5",
        streaming=False,
        reasoning_effort="low",
        max_tokens=64,
    )
    entry = captured_payloads[-1]
    assert entry["url"].rstrip("/").endswith("/responses")
    payload = entry["payload"]
    assert payload["max_output_tokens"] == 64
    assert "max_tokens" not in payload
    assert "max_completion_tokens" not in payload


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
def test_openai_sampling_predicate_boundaries(model, expected):
    """The builder consults the TASK-18802 predicate; its boundaries carry
    the exact rows the retired ``_is_openai_reasoning_model`` tuple pinned."""
    assert openai_model_rejects_sampling_params(model) is expected
