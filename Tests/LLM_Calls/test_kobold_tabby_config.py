"""Kobold and TabbyAPI summarizer configuration at the analyze seam.

task-17383: both summarizers indexed loader sections that were never built
(``api_keys``, ``local_api_ip``, ``models``) -- like llama.cpp before
task-17382 they raised before contacting a server and returned an error
STRING that callers could store as evidence. TASK-34895 cut the local
summarizers over to the unified chat dispatch (``analyze`` ->
``_summarize_via_chat_api`` -> ``chat_api_call``); the per-provider
functions are gone. What stays pinned here is the seam the deep-search
pipeline actually calls: the modern-table endpoint resolution and the
plain-string (task-17387 / task-32805.3) non-streaming return.
"""


class FakeResponse:
    """Shapes the SERVERS actually return, not the shape the code wants.

    Kobold's native generate endpoint answers ``{"results": [{"text": ...}]}``;
    TabbyAPI is OpenAI-compatible and answers with ``choices[].message``. A
    single fake asserting one shape would have passed while the other provider
    stayed broken -- the failure mode task-17382 was built on.
    """

    status_code = 200

    def __init__(self, payload=None):
        self.text = ""
        self._payload = (
            payload
            if payload is not None
            else {
                # Kobold native generate
                "results": [{"text": " SUMMARY "}],
                # OpenAI-compatible envelope -- TabbyAPI validates every one of
                # these keys before reading the content, so a fake carrying only
                # "choices" passes nothing.
                "id": "cmpl-test",
                "created": 0,
                "model": "test-model",
                "object": "chat.completion",
                "usage": {"completion_tokens": 3},
                "choices": [{"message": {"role": "assistant", "content": " SUMMARY "}}],
            }
        )

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


def drain(result):
    """Run a summarizer to completion and return its final value."""
    if isinstance(result, str):
        return result
    chunks = []
    while True:
        try:
            chunks.append(next(result))
        except StopIteration as stop:
            return stop.value if stop.value is not None else "".join(chunks)


def settings(**overrides):
    """A snapshot shaped like the real loader's: the modern per-provider table
    and the legacy sections exist; `api_keys`/`local_api_ip`/`models` do NOT."""
    base = {
        "api_settings": {
            "koboldcpp": {"api_url": "http://kobold.invalid/v1/chat/completions"},
            "tabbyapi": {"api_url": "http://tabby.invalid/v1/chat/completions"},
            # chat_with_tabbyapi reads the "tabby_api" spelling from the
            # api_settings table (LLM_API_Calls_Local.py) -- both live.
            "tabby_api": {
                "api_url": "http://tabby.invalid/v1/chat/completions",
                "model": "tabby-model",
            },
        },
        "kobold_api": {
            "api_ip": "http://legacy-kobold.invalid/api/v1/generate",
            "api_key": "legacy-kobold-key",
            "api_retries": 0,
            "api_retry_delay": 0,
            "streaming": False,
            "temperature": 0.7,
            "max_tokens": 64,
        },
        "tabby_api": {
            "api_ip": "http://legacy-tabby.invalid/v1/chat/completions",
            "api_key": "legacy-tabby-key",
            "model": "legacy-model",
            "api_retries": 0,
            "api_retry_delay": 0,
            "streaming": False,
            "temperature": 0.7,
            "max_tokens": 64,
        },
    }
    for key, value in overrides.items():
        base["api_settings"].setdefault(key, {}).update(value)
    return base


def test_configuration_reaches_the_public_analyze_boundary(monkeypatch):
    """Qodo (PR 1788): go through `analyze`, the seam the deep-search
    pipeline actually uses, so the resolution is verified where a caller
    meets it."""
    from tldw_chatbook.LLM_Calls import LLM_API_Calls_Local as _local
    from tldw_chatbook.LLM_Calls import Summarization_General_Lib as _general
    from tldw_chatbook.LLM_Calls.Summarization_General_Lib import analyze

    seen = {}

    class Session:
        def mount(self, *_a, **_k):
            pass

        def post(self, url, headers=None, json=None, stream=False, **_kw):
            seen["url"] = url
            return FakeResponse()

    # TASK-34895: tabbyapi routes through the chat handler; the transport
    # and config seams now live on LLM_API_Calls_Local.
    monkeypatch.setattr(
        _local, "create_pooled_default_session", lambda **_kw: Session()
    )

    class _Snapshot:
        values = settings()

    monkeypatch.setattr(_local, "get_runtime_config_snapshot", lambda: _Snapshot)

    # The chat-dispatch core resolves model/key from the legacy top-level
    # sections through sgl.get_cli_setting.
    _fake_settings = settings()

    def _fake_get_cli_setting(section, key, *args, **kwargs):
        default = args[0] if args else kwargs.get("default")
        return _fake_settings.get(section, {}).get(key, default)

    monkeypatch.setattr(_general, "get_cli_setting", _fake_get_cli_setting)
    # TASK-33001.14: the [api_settings] key rung reads the loaded settings.
    monkeypatch.setattr(_general, "load_settings", lambda: _fake_settings)

    result = drain(
        analyze(
            input_data="a chunk of packed evidence",
            custom_prompt_arg="Summarize.",
            api_name="tabbyapi",
            api_key=None,
            temp=0.3,
            system_message=None,
            streaming=False,
        )
    )

    # The configuration resolution is what this asserts: the request went to
    # the endpoint the modern table names, through the public seam.
    assert seen["url"] == "http://tabby.invalid/v1/chat/completions", seen
    # task-17387 / task-32805.3 landed: the summarizers are no longer generator
    # functions, so `analyze` hands a non-streaming caller the summary string
    # directly (not a generator surfaced as "Unexpected result type").
    assert isinstance(result, str), result
    assert "SUMMARY" in result, result
