from unittest.mock import Mock, patch

from tldw_chatbook.Chat.Chat_Functions import chat_api_call


def _spy_histograms(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        "tldw_chatbook.LLM_Calls.LLM_API_Calls.log_histogram",
        lambda name, value, **kw: calls.__setitem__(name, value),
    )
    return calls


def _mock_post(resp):
    m = Mock()
    m.json.return_value = resp
    m.status_code = 200
    m.raise_for_status = Mock()
    return m


def test_anthropic_logs_cache_metrics(monkeypatch):
    calls = _spy_histograms(monkeypatch)
    resp = {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-x",
        "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 10, "output_tokens": 5,
            "cache_read_input_tokens": 100, "cache_creation_input_tokens": 20,
        },
    }
    with patch("requests.Session.post", return_value=_mock_post(resp)):
        chat_api_call(
            "anthropic", messages_payload=[{"role": "user", "content": "hi"}],
            api_key="k", model="claude-3-opus-20240229", streaming=False,
        )
    assert calls["anthropic_api_cache_read_input_tokens"] == 100
    assert calls["anthropic_api_cache_creation_input_tokens"] == 20
    # existing metrics still fire
    assert calls["anthropic_api_input_tokens"] == 10


def test_anthropic_cache_metrics_absent_fields_zero(monkeypatch):
    calls = _spy_histograms(monkeypatch)
    resp = {
        "id": "m", "type": "message", "role": "assistant", "model": "claude-x",
        "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
        "usage": {"input_tokens": 3, "output_tokens": 2},  # no cache fields
    }
    with patch("requests.Session.post", return_value=_mock_post(resp)):
        chat_api_call(
            "anthropic", messages_payload=[{"role": "user", "content": "hi"}],
            api_key="k", model="claude-3-opus-20240229", streaming=False,
        )
    assert calls["anthropic_api_cache_read_input_tokens"] == 0
    assert calls["anthropic_api_cache_creation_input_tokens"] == 0


def test_openai_logs_cached_tokens(monkeypatch):
    calls = _spy_histograms(monkeypatch)
    resp = {
        "id": "cmpl", "object": "chat.completion", "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                     "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
            "prompt_tokens_details": {"cached_tokens": 8},
        },
    }
    with patch("requests.Session.post", return_value=_mock_post(resp)):
        chat_api_call(
            "openai", messages_payload=[{"role": "user", "content": "hi"}],
            api_key="k", model="gpt-4o", streaming=False,
        )
    assert calls["openai_api_cached_tokens"] == 8


def test_openai_cached_tokens_absent_zero(monkeypatch):
    calls = _spy_histograms(monkeypatch)
    resp = {
        "id": "cmpl", "object": "chat.completion", "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    with patch("requests.Session.post", return_value=_mock_post(resp)):
        chat_api_call(
            "openai", messages_payload=[{"role": "user", "content": "hi"}],
            api_key="k", model="gpt-4o", streaming=False,
        )
    assert calls["openai_api_cached_tokens"] == 0
