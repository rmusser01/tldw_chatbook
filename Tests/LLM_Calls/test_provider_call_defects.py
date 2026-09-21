"""Four provider-call defects fixed in TASK-32805.4.

Each test drives the real function/module with the HTTP layer mocked; none
touches storage, so they run without the ADR-126 admission binding.
"""

import types
from unittest.mock import Mock

from tldw_chatbook.LLM_Calls import Local_Summarization_Lib as lsl
from tldw_chatbook.LLM_Calls import Summarization_General_Lib as sgl
from tldw_chatbook.LLM_Calls import hosted_chat as hc
from tldw_chatbook.LLM_Calls import qwencloud as qc


class FakeResponse:
    def __init__(self, *, status_code=200, json_data=None, lines=None):
        self.status_code = status_code
        self._json = json_data or {}
        self._lines = lines or []
        self.text = ""
        self.headers = {}

    def json(self):
        return self._json

    def raise_for_status(self):
        return None

    def close(self):
        return None

    def iter_lines(self):
        for line in self._lines:
            yield line


class FakeSession:
    def __init__(self, response):
        self._response = response
        self.posts = 0

    def mount(self, *_a, **_k):
        pass

    def post(self, *_a, **_k):
        self.posts += 1
        return self._response

    def close(self):
        pass


# --- Defect 1: summarize_with_vllm crashed on an explicit api_key ---
def test_vllm_explicit_key_returns_a_summary(monkeypatch):
    """`loaded_config_data` was bound only on the no-key branch, so an explicit
    key raised UnboundLocalError at the retry-count reads."""
    cfg = {
        "vllm_api": {
            "model": "m",
            "api_ip": "http://vllm.invalid/v1/chat/completions",
            "max_tokens": 64,
            "api_retries": 0,
            "api_retry_delay": 0,
            "api_key": "cfg-key",
            "temperature": 0.7,
        }
    }
    monkeypatch.setattr(lsl, "load_settings", lambda *a, **k: cfg)
    resp = FakeResponse(json_data={"choices": [{"message": {"content": " THE SUMMARY "}}]})
    monkeypatch.setattr(lsl, "create_default_session", lambda: FakeSession(resp))

    result = lsl.summarize_with_vllm("explicit-key", "some text", "Summarize.")

    assert isinstance(result, str), result
    assert "loaded_config_data" not in result, result
    assert result.strip() == "THE SUMMARY", result


# --- Defect 3: streaming summarizers re-emitted the whole summary at the end ---
def test_custom_openai_stream_yields_the_summary_once(monkeypatch):
    """A trailing `yield collected_messages` doubled the text for any consumer
    that joins the chunks."""
    cfg = {
        "custom_openai_api": {
            "api_key": "k",
            "model": "m",
            "max_tokens": 64,
            "api_ip": "http://custom.invalid/v1/chat/completions",
            "api_retries": 0,
            "api_retry_delay": 0,
            "temperature": 0.7,
        }
    }
    monkeypatch.setattr(lsl, "load_settings", lambda *a, **k: cfg)
    lines = [
        b'data: {"choices":[{"delta":{"content":"Hello "}}]}',
        b'data: {"choices":[{"delta":{"content":"world"}}]}',
        b"data: [DONE]",
    ]
    monkeypatch.setattr(
        lsl, "create_default_session", lambda: FakeSession(FakeResponse(lines=lines))
    )

    result = lsl.summarize_with_custom_openai(
        "k", "some text", "Summarize.", streaming=True
    )

    assert not isinstance(result, str), "streaming must return an iterator"
    assert "".join(result) == "Hello world"  # not "Hello worldHello world"


# --- Defect 2: summarize_with_anthropic posted through a bare requests.post ---
def test_anthropic_posts_through_the_session(monkeypatch):
    """The retry adapter is mounted on a session; the request must go through it
    (bare requests.post bypasses the configured api_retries)."""
    session = FakeSession(
        FakeResponse(json_data={"content": [{"type": "text", "text": "ANTHROPIC SUMMARY"}]})
    )
    # Keep the call off the config/storage path (the ADR-126 admission gate
    # fires from a real config read in a clean worktree).
    monkeypatch.setattr(
        sgl, "get_cli_setting", lambda _section, _key, default=None: default
    )
    monkeypatch.setattr(sgl, "requests_verify", lambda: True)
    monkeypatch.setattr(sgl, "create_default_session", lambda: session)

    def _boom(*_a, **_k):
        raise AssertionError("bare requests.post used; the retry adapter is bypassed")

    monkeypatch.setattr(sgl.requests, "post", _boom)

    result = sgl.summarize_with_anthropic("key", "some text", "Summarize.")

    assert result == "ANTHROPIC SUMMARY", result
    assert session.posts == 1


# --- Defect 4: a provider Retry-After header was slept uncapped on a worker ---
def test_hosted_chat_retry_after_is_clamped():
    delay = hc._retry_delay(
        Mock(headers={"Retry-After": "99999999"}), attempt=0, retry_delay=1.0
    )
    assert 0.0 <= delay <= hc._MAX_RETRY_AFTER_SECONDS


def test_qwencloud_retry_after_is_clamped():
    policy = Mock()
    policy.get_retry_after.return_value = 99999999.0
    nxt = Mock()
    nxt.get_backoff_time.return_value = 0.0
    policy.increment.return_value = nxt
    resp = types.SimpleNamespace(status_code=429, headers={"Retry-After": "99999999"})

    _next, delay = qc._advance_retry_policy(policy, api_url="http://q.invalid", response=resp)

    assert 0.0 <= delay <= qc._MAX_RETRY_AFTER_SECONDS
