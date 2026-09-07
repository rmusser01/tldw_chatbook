"""Kobold and TabbyAPI summarizer configuration resolution (task-17383).

NOTE ON SHAPE: both functions are GENERATOR functions -- a top-level `yield` in
their streaming branches makes the whole function one -- so calling them returns
a generator and the body runs only when it is iterated. That is a separate,
larger defect (a non-streaming caller receives a generator where a summary
belongs, and the deep-search pipeline would store it as evidence); it is filed
on its own because fixing it re-attributes ~20 diagnostics to nested functions
and rewrites a security-reviewed ledger. These tests therefore drive the
functions the way they currently behave, and still prove the configuration
resolution.

Both indexed sections the loader has never built -- `api_keys`,
`local_api_ip`, `models` -- so like llama.cpp before task-17382 they raised
before contacting a server and returned an error STRING that callers could
store as evidence. Both providers DO have complete modern `api_settings`
entries and complete legacy sections; these tests pin that they read those.
"""

import pytest

from tldw_chatbook.LLM_Calls import Local_Summarization_Lib as lib


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
        self._payload = payload if payload is not None else {
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

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


@pytest.fixture
def captured_post(monkeypatch):
    captured = {}

    class FakeSession:
        def mount(self, *_args, **_kwargs):
            pass

        def post(self, url, headers=None, json=None, stream=False, **_kw):
            captured["url"] = url
            captured["headers"] = headers or {}
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr(lib, "create_default_session", lambda: FakeSession())
    return captured


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


@pytest.mark.parametrize(
    "summarize,expected_url",
    [
        (
            lambda: lambda: lib.summarize_with_kobold("some text", None, "Summarize."),
            "http://kobold.invalid/v1/chat/completions",
        ),
        (
            lambda: lambda: lib.summarize_with_tabbyapi("some text", "Summarize."),
            "http://tabby.invalid/v1/chat/completions",
        ),
    ],
    ids=["kobold", "tabby"],
)
def test_summarizer_reaches_its_server_via_the_modern_entry(
    monkeypatch, captured_post, summarize, expected_url
):
    """The whole defect: no KeyError, a real request, a real summary."""
    monkeypatch.setattr(lib, "load_settings", lambda: settings())

    result = drain(summarize()())

    # Kobold strips its summary and TabbyAPI does not; that difference is not
    # what this test is about.
    assert result.strip() == "SUMMARY", result
    assert captured_post["url"] == expected_url


@pytest.mark.parametrize(
    "summarize,expected_url",
    [
        (
            lambda: lambda: lib.summarize_with_kobold("some text", None, "Summarize."),
            "http://legacy-kobold.invalid/api/v1/generate",
        ),
        (
            lambda: lambda: lib.summarize_with_tabbyapi("some text", "Summarize."),
            "http://legacy-tabby.invalid/v1/chat/completions",
        ),
    ],
    ids=["kobold", "tabby"],
)
def test_summarizer_falls_back_to_its_legacy_section(
    monkeypatch, captured_post, summarize, expected_url
):
    conf = settings()
    conf["api_settings"] = {"koboldcpp": {}, "tabbyapi": {}}
    monkeypatch.setattr(lib, "load_settings", lambda: conf)

    drain(summarize()())

    assert captured_post["url"] == expected_url


def test_tabby_reads_the_credential_env_var_the_config_names(
    monkeypatch, captured_post
):
    """`api_key_env_var` is this repo's existing convention (the media and
    settings windows both honour it), and it is tabbyapi's only credential
    field in the modern table."""
    conf = settings(tabbyapi={"api_key_env_var": "TABBY_TEST_KEY"})
    conf["tabby_api"]["api_key"] = ""
    monkeypatch.setattr(lib, "load_settings", lambda: conf)
    monkeypatch.setenv("TABBY_TEST_KEY", "env-tabby-key")

    drain(lib.summarize_with_tabbyapi("some text", "Summarize."))

    assert "env-tabby-key" in str(captured_post["headers"])


def test_missing_sections_do_not_raise(monkeypatch, captured_post):
    """A config with neither modern nor legacy entries must fail legibly, not
    with a KeyError that becomes an error string stored as evidence."""
    monkeypatch.setattr(lib, "load_settings", lambda: {"api_settings": {}})

    result = drain(lib.summarize_with_kobold("some text", None, "Summarize."))

    from tldw_chatbook.Web_Scraping.WebSearch_APIs import _is_summary_failure

    assert isinstance(result, str)
    assert _is_summary_failure(result), result
    assert "KeyError" not in result


def test_kobold_streaming_uses_the_openai_compatible_endpoint(monkeypatch):
    """Qodo (PR 1788): the streaming branch parses OpenAI SSE, but both the
    modern api_url and the legacy api_ip point at Kobold's NATIVE generate
    endpoint -- only `api_streaming_ip` is OpenAI-compatible. Preferring the
    modern url sent streaming to the native endpoint and yielded nothing."""
    conf = settings()
    conf["api_settings"]["koboldcpp"]["api_url"] = "http://kobold.invalid/api/v1/generate"
    conf["kobold_api"]["api_ip"] = "http://kobold.invalid/api/v1/generate"
    conf["kobold_api"]["api_streaming_ip"] = "http://kobold.invalid/v1/chat/completions"
    monkeypatch.setattr(lib, "load_settings", lambda: conf)

    seen = {}

    class StreamResponse:
        status_code = 200
        text = ""

        def iter_lines(self):
            yield b'data: {"choices":[{"delta":{"content":"chunk"}}]}'
            yield b"data: [DONE]"

        def json(self):
            return {}

        def raise_for_status(self):
            return None

    class StreamSession:
        def mount(self, *_a, **_k):
            pass

        def post(self, url, **_kw):
            seen["url"] = url
            return StreamResponse()

    monkeypatch.setattr(lib, "create_default_session", lambda: StreamSession())

    drain(lib.summarize_with_kobold("text", None, "Summarize.", streaming=True))

    assert seen["url"] == "http://kobold.invalid/v1/chat/completions", seen


def test_kobold_non_streaming_uses_the_native_endpoint(monkeypatch, captured_post):
    """The complement: the non-streaming branch parses `results[].text`, so it
    must go to the NATIVE generate endpoint. Shaped like the real config, where
    both the modern api_url and the legacy api_ip are native."""
    conf = settings()
    conf["api_settings"]["koboldcpp"]["api_url"] = "http://kobold.invalid/api/v1/generate"
    conf["kobold_api"]["api_ip"] = "http://kobold.invalid/api/v1/generate"
    conf["kobold_api"]["api_streaming_ip"] = "http://kobold.invalid/v1/chat/completions"
    monkeypatch.setattr(lib, "load_settings", lambda: conf)

    drain(lib.summarize_with_kobold("text", None, "Summarize."))

    assert captured_post["url"] == "http://kobold.invalid/api/v1/generate"


def test_configuration_reaches_the_public_analyze_boundary(monkeypatch):
    """Qodo (PR 1788): the other tests call the summarizers directly. This one
    goes through `analyze`, the seam the deep-search pipeline actually uses, so
    the resolution is verified where a caller meets it."""
    from tldw_chatbook.LLM_Calls.Summarization_General_Lib import analyze

    monkeypatch.setattr(lib, "load_settings", lambda: settings())
    seen = {}

    class Session:
        def mount(self, *_a, **_k):
            pass

        def post(self, url, headers=None, json=None, stream=False, **_kw):
            seen["url"] = url
            return FakeResponse()

    monkeypatch.setattr(lib, "create_default_session", lambda: Session())

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
    # The VALUE a caller gets back is still unusable, and deliberately so: these
    # functions are generators (task-17387), so `analyze` hands back something
    # a non-streaming caller cannot treat as a summary. Pinned here rather than
    # asserted away, so this test starts telling the truth about the return
    # value the moment task-17387 lands.
    assert result.strip() == "", (
        "task-17387 fixed? then assert the summary here instead: " + repr(result)
    )
