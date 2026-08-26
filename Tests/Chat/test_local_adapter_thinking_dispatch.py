"""Adapter-path thinking dispatch: param maps + shared local builder."""

from unittest.mock import patch

from tldw_chatbook.Chat.Chat_Functions import chat_api_call

COVERED_KEYS = [
    "llama_cpp",
    "local_llamacpp",
    "local_llamafile",
    "local-llm",
    "vllm",
    "local_vllm",
    "local_mlx_lm",
    "custom-openai-api",
    "custom-openai-api-2",
]


def test_param_maps_forward_thinking_fields(monkeypatch):
    captured = {}

    def fake_handler(**kwargs):
        captured.update(kwargs)
        return "ok"

    import tldw_chatbook.Chat.Chat_Functions as cf

    for key in COVERED_KEYS:
        captured.clear()
        monkeypatch.setitem(cf.API_CALL_HANDLERS, key, fake_handler)
        chat_api_call(
            api_endpoint=key,
            messages_payload=[{"role": "user", "content": "hi"}],
            api_key=None,
            reasoning_effort="low",
            thinking_budget_tokens=2048,
        )
        assert captured.get("reasoning_effort") == "low", key
        assert captured.get("thinking_budget_tokens") == 2048, key


def test_shared_builder_composes_llama_wire_format():
    from tldw_chatbook.LLM_Calls.LLM_API_Calls_Local import (
        _chat_with_openai_compatible_local_server,
    )

    posted = {}

    class FakeResponse:
        status_code = 200
        text = '{"choices":[{"message":{"content":"ok"}}]}'

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

        def raise_for_status(self):
            return None

    class FakeSession:
        def __init__(self):
            self.adapters = None

        def mount(self, *a, **k):
            pass

        def post(self, url, json=None, headers=None, timeout=None):
            posted["url"] = url
            posted["payload"] = json
            return FakeResponse()

        def close(self):
            pass

    with patch(
        "tldw_chatbook.LLM_Calls.LLM_API_Calls_Local.create_default_session",
        return_value=FakeSession(),
    ):
        _chat_with_openai_compatible_local_server(
            api_base_url="http://127.0.0.1:8080",
            model_name="qwen",
            input_data=[{"role": "user", "content": "hi"}],
            streaming=False,
            reasoning_effort="low",
            thinking_budget_tokens=2048,
            thinking_wire_key="llama_cpp",
        )

    payload = posted["payload"]
    assert payload["chat_template_kwargs"] == {"reasoning_effort": "low"}
    assert payload["reasoning_budget_tokens"] == 2048


def test_shared_builder_composes_vllm_dual_placement():
    # Same harness as above; only assertions differ. FakeResponse mirrors the
    # llama harness surface (the builder reads .status_code for metrics).
    posted = {}

    class FakeResponse:
        status_code = 200
        text = '{"choices":[{"message":{"content":"ok"}}]}'

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

        def raise_for_status(self):
            return None

    class FakeSession:
        def mount(self, *a, **k):
            pass

        def post(self, url, json=None, headers=None, timeout=None):
            posted["payload"] = json
            return FakeResponse()

        def close(self):
            pass

    with patch(
        "tldw_chatbook.LLM_Calls.LLM_API_Calls_Local.create_default_session",
        return_value=FakeSession(),
    ):
        from tldw_chatbook.LLM_Calls.LLM_API_Calls_Local import (
            _chat_with_openai_compatible_local_server,
        )

        _chat_with_openai_compatible_local_server(
            api_base_url="http://127.0.0.1:8000",
            model_name="qwen",
            input_data=[{"role": "user", "content": "hi"}],
            streaming=False,
            reasoning_effort="medium",
            thinking_budget_tokens=2048,
            thinking_wire_key="vllm",
        )

    payload = posted["payload"]
    assert payload["reasoning_effort"] == "medium"
    assert payload["chat_template_kwargs"] == {"reasoning_effort": "medium"}
    assert "reasoning_budget_tokens" not in payload
