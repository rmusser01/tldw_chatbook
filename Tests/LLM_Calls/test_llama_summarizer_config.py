"""llama.cpp summarization configuration reads at the analyze seam.

task-17382: ``summarize_with_llama`` read ``loaded_config_data["llama_api"]``
-- a section the loader has never built -- so every call raised KeyError
before contacting a server and returned an error STRING the deep-search
caller stored as evidence. TASK-34895 cut the local summarizers over to
the unified chat dispatch (``analyze`` -> ``_summarize_via_chat_api`` ->
``chat_api_call``); the per-provider functions are gone. What stays pinned
here is the seam the deep-search pipeline actually calls: the modern-table
token budget (task-17382) reaching the request and the reasoning-only
empty-completion diagnostic (task-17384) surfacing through ``analyze``.
"""


class _FakeResponse:
    """The shape llama-server's /v1/chat/completions ACTUALLY returns.

    The first version of this fake returned llama.cpp's NATIVE
    ``{"content": ...}`` shape, which is what the function parsed -- so these
    tests passed while every real chunk summarization came back "Llama: No
    choices in response data" (observed live). A fake that agrees with the
    code instead of with the server proves nothing.
    """

    status_code = 200

    def raise_for_status(self):
        return None

    def __init__(self, payload=None):
        self.text = ""
        self._payload = payload or {
            "choices": [{"message": {"role": "assistant", "content": " SUMMARY "}}]
        }

    def json(self):
        return self._payload


def _settings(
    *, modern_url=None, legacy_ip="http://127.0.0.1:8080/v1/chat/completions"
):
    """A settings snapshot shaped like the real loader's: `llama_cpp_api`
    exists, `llama_api` does NOT."""
    settings = {
        "llama_cpp_api": {
            "api_key": "",
            "api_ip": legacy_ip,
            "temperature": 0.7,
            "max_tokens": 4096,
            "streaming": False,
            "api_retries": 3,
            "api_retry_delay": 5,
        },
        "api_settings": {"llama_cpp": {}},
    }
    if modern_url:
        settings["api_settings"]["llama_cpp"]["api_url"] = modern_url
    return settings


def test_budget_and_diagnostic_reach_the_public_analyze_boundary(monkeypatch):
    """Qodo (PR 1774): go through `analyze`, which is the seam the
    deep-search pipeline actually calls, so the budget resolution and the
    reasoning-only diagnostic are verified where a caller meets them rather
    than one layer in."""
    from tldw_chatbook.LLM_Calls import LLM_API_Calls_Local as _local
    from tldw_chatbook.LLM_Calls import Summarization_General_Lib as _general
    from tldw_chatbook.LLM_Calls.Summarization_General_Lib import analyze
    from tldw_chatbook.Web_Scraping.WebSearch_APIs import _is_summary_failure

    settings = _settings()
    settings["api_settings"]["llama_cpp"]["max_tokens"] = 16384
    # The chat handler resolves the URL from the modern table only (the
    # legacy api_ip fallback lived in the retired LS resolution).
    settings["api_settings"]["llama_cpp"]["api_url"] = (
        "http://modern.invalid/v1/chat/completions"
    )

    seen = {}

    class _Session:
        def mount(self, *_a, **_k):
            pass

        def post(self, url, headers=None, json=None, stream=False, **kwargs):
            seen["max_tokens"] = json["max_tokens"]
            return _FakeResponse(
                {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "reasoning_content": "thought " * 400,
                            },
                        }
                    ],
                    "usage": {"completion_tokens": 16384},
                }
            )

    # TASK-34895: llama_cpp routes through the chat handler; the transport
    # and config seams now live on LLM_API_Calls_Local.
    monkeypatch.setattr(
        _local, "create_pooled_default_session", lambda **_kw: _Session()
    )

    class _Snapshot:
        values = settings

    monkeypatch.setattr(_local, "get_runtime_config_snapshot", lambda: _Snapshot)

    # The chat-dispatch core resolves model/key from the legacy top-level
    # sections through sgl.get_cli_setting.
    def _fake_get_cli_setting(section, key, *args, **kwargs):
        default = args[0] if args else kwargs.get("default")
        return settings.get(section, {}).get(key, default)

    monkeypatch.setattr(_general, "get_cli_setting", _fake_get_cli_setting)
    # TASK-33001.14: the [api_settings] key rung reads the loaded settings.
    monkeypatch.setattr(_general, "load_settings", lambda: settings)

    result = analyze(
        input_data="a chunk of packed evidence",
        custom_prompt_arg="Summarize.",
        api_name="llama_cpp",
        api_key=None,
        temp=0.3,
        system_message=None,
        streaming=False,
    )

    assert seen["max_tokens"] == 16384, seen
    assert _is_summary_failure(result), result
    assert "reasoning" in result.lower() and "length" in result.lower(), result
