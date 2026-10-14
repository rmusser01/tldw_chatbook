"""Four provider-call defects fixed in TASK-32805.4.

Each test drives the real function/module with the HTTP layer mocked; none
touches storage, so they run without the ADR-126 admission binding.

TASK-34895: the per-provider summarizer functions defects 1-3 lived in are
deleted (summarization rides ``chat_api_call`` now), so those pins moved to
the chat handlers the dispatch rides -- same defect classes, same faked-wire
style. Defect 2 folded into the existing chat-side pin (see the comment at
its old location below).
"""

import types
from unittest.mock import Mock

from tldw_chatbook.LLM_Calls import LLM_API_Calls_Local as local
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

    def iter_lines(self, **_kwargs):
        yield from self._lines


class FakeSession:
    def __init__(self, response):
        self._response = response
        self.posts = 0
        self.headers_seen = []

    def mount(self, *_a, **_k):
        pass

    def post(self, *_a, headers=None, **_k):
        self.posts += 1
        self.headers_seen.append(headers or {})
        return self._response

    def close(self):
        pass


# --- Defect 1: an explicit api_key must survive the handler's config reads
# (originally: summarize_with_vllm raised UnboundLocalError on the
# explicit-key branch because `loaded_config_data` was bound only when the
# key was absent; retargeted to chat_with_vllm, the handler the
# summarization dispatch now rides) ---
def test_vllm_explicit_key_returns_a_summary(monkeypatch):
    cfg = {
        "api_settings": {
            "vllm_api": {
                "model": "m",
                "api_url": "http://vllm.invalid/v1/chat/completions",
                "max_tokens": 64,
                "api_retries": 0,
                "api_retry_delay": 0,
                "api_key": "cfg-key",
                "temperature": 0.7,
            }
        }
    }
    monkeypatch.setattr(
        local, "get_runtime_config_snapshot", lambda: types.SimpleNamespace(values=cfg)
    )
    session = FakeSession(
        FakeResponse(json_data={"choices": [{"message": {"content": " THE SUMMARY "}}]})
    )
    monkeypatch.setattr(local, "create_default_session", lambda: session)

    result = local.chat_with_vllm(
        [{"role": "user", "content": "some text"}], api_key="explicit-key", model="m"
    )

    assert result["choices"][0]["message"]["content"].strip() == "THE SUMMARY", result
    # The explicit key -- not the config's -- is the credential that
    # reaches the wire (the config reads must not have crashed or shadowed
    # it, the original defect's two faces).
    assert session.headers_seen[0]["Authorization"] == "Bearer explicit-key"


# --- Defect 3: a streamed summary must reach the consumer exactly once
# (originally: summarize_with_custom_openai's trailing
# `yield collected_messages` doubled the joined text; retargeted to the
# chat_with_custom_openai wire the summarization dispatch now rides) ---
def test_custom_openai_stream_yields_the_summary_once(monkeypatch):
    cfg = {
        "api_settings": {
            "custom": {
                "api_key": "k",
                "model": "m",
                "max_tokens": 64,
                "api_url": "http://custom.invalid/v1/chat/completions",
                "api_retries": 0,
                "api_retry_delay": 0,
                "temperature": 0.7,
            }
        },
        "custom_openai_api": {"api_key": "k"},
    }
    monkeypatch.setattr(
        local, "get_runtime_config_snapshot", lambda: types.SimpleNamespace(values=cfg)
    )
    lines = [
        'data: {"choices":[{"delta":{"content":"Hello "}}]}',
        'data: {"choices":[{"delta":{"content":"world"}}]}',
        "data: [DONE]",
    ]
    monkeypatch.setattr(
        local, "create_default_session", lambda: FakeSession(FakeResponse(lines=lines))
    )
    monkeypatch.setattr(
        sgl,
        "get_cli_setting",
        lambda section, key, default=None: cfg.get(section, {}).get(key, default),
    )

    stream = sgl._summarize_via_chat_api(
        "custom-openai-api", "explicit-key", "some text", "Summarize.", None, None, True
    )
    result = "".join(stream)

    assert result == "Hello world"  # not "Hello worldHello world"


# --- Defect 2: summarize_with_anthropic posted through a bare requests.post
# (bypassing the session's retry adapter). Deleted with the function in the
# TASK-34895 chat-dispatch cutover: the summarization path now rides
# chat_with_anthropic, whose session-transport contract is pinned chat-side
# in Tests/Chat/test_anthropic_native_tools.py (patches requests.Session.post
# and drives the real chat_api_call dispatch -- a bare requests.post would
# bypass that patch and fail every test there). ---


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

    _next, delay = qc._advance_retry_policy(
        policy, api_url="http://q.invalid", response=resp
    )

    assert 0.0 <= delay <= qc._MAX_RETRY_AFTER_SECONDS
