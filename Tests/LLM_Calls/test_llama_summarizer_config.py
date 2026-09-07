"""llama.cpp summarization configuration reads (task-17382).

`summarize_with_llama` read `loaded_config_data["llama_api"]`, a section that
has never existed -- the loader builds `llama_cpp_api`. Every call therefore
raised KeyError before contacting a server and returned an error STRING,
which the deep-search caller stored as a result's evidence content.
"""

import pytest

from tldw_chatbook.LLM_Calls import Local_Summarization_Lib as lib


class _FakeResponse:
    """The shape llama-server's /v1/chat/completions ACTUALLY returns.

    The first version of this fake returned llama.cpp's NATIVE
    ``{"content": ...}`` shape, which is what the function parsed -- so these
    tests passed while every real chunk summarization came back "Llama: No
    choices in response data" (observed live). A fake that agrees with the
    code instead of with the server proves nothing.
    """

    status_code = 200

    def __init__(self, payload=None):
        self.text = ""
        self._payload = payload or {
            "choices": [{"message": {"role": "assistant", "content": " SUMMARY "}}]
        }

    def json(self):
        return self._payload


@pytest.fixture
def captured_post(monkeypatch):
    """Capture the POST instead of performing it; the summary path must get
    far enough to make a request at all."""
    captured = {}

    class _FakeSession:
        def mount(self, *_args, **_kwargs):
            pass

        def post(self, url, headers=None, json=None, stream=False):
            captured["url"] = url
            captured["json"] = json
            return _FakeResponse()

    monkeypatch.setattr(lib, "create_default_session", lambda: _FakeSession())
    return captured


def _settings(*, modern_url=None, legacy_ip="http://127.0.0.1:8080/v1/chat/completions"):
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


def test_summarize_with_llama_does_not_need_a_nonexistent_config_section(
    monkeypatch, captured_post
):
    """The whole defect: no KeyError, and a real summary comes back."""
    monkeypatch.setattr(lib, "load_settings", lambda: _settings())

    result = lib.summarize_with_llama(
        input_data="Retrieval augmented generation combines retrieval and generation.",
        custom_prompt="Summarize in one sentence.",
        api_key=None,
        temp=None,
        system_message=None,
        streaming=False,
    )

    assert result == "SUMMARY"
    assert "url" in captured_post, "no request was ever made"


def test_summarize_with_llama_prefers_the_modern_api_url(monkeypatch, captured_post):
    """Runs point local providers at their endpoint through
    `api_settings.llama_cpp.api_url` (what the baseline recorder primes and
    the chat handler reads). The summarizer must follow the same routing
    instead of the legacy default, or it posts where nothing is listening."""
    monkeypatch.setattr(
        lib, "load_settings", lambda: _settings(modern_url="http://127.0.0.1:9191/v1/chat/completions")
    )

    result = lib.summarize_with_llama(
        input_data="text to summarize",
        custom_prompt="Summarize.",
        api_key=None,
    )

    assert result == "SUMMARY"
    assert captured_post["url"] == "http://127.0.0.1:9191/v1/chat/completions"


def test_summarize_with_llama_falls_back_to_the_legacy_ip(monkeypatch, captured_post):
    """With no modern entry, the historical key still routes the request."""
    monkeypatch.setattr(
        lib, "load_settings", lambda: _settings(legacy_ip="http://127.0.0.1:8080/v1/chat/completions")
    )

    lib.summarize_with_llama(input_data="text", custom_prompt="Summarize.", api_key=None)

    assert captured_post["url"] == "http://127.0.0.1:8080/v1/chat/completions"


@pytest.mark.parametrize(
    "configured,expected",
    [
        # A run priming a local endpoint stores a BASE url; posting it raw
        # returned 404 "File Not Found" from llama-server (observed live).
        ("http://127.0.0.1:9191/v1", "http://127.0.0.1:9191/v1/chat/completions"),
        # A full endpoint must not gain a second copy of the path.
        (
            "http://127.0.0.1:9191/v1/chat/completions",
            "http://127.0.0.1:9191/v1/chat/completions",
        ),
        # Bare host:port is a documented shape for these keys.
        ("127.0.0.1:9191", "http://127.0.0.1:9191/v1/chat/completions"),
        # Trailing slash must not double up.
        ("http://127.0.0.1:9191/", "http://127.0.0.1:9191/v1/chat/completions"),
    ],
)
def test_summarize_with_llama_normalizes_the_endpoint(
    monkeypatch, captured_post, configured, expected
):
    """task-17382: the summarizer POSTs directly, so whatever shape the config
    holds has to become the chat-completions endpoint exactly once -- the same
    normalization the chat handler applies via normalize_llamacpp_base_url."""
    monkeypatch.setattr(lib, "load_settings", lambda: _settings(modern_url=configured))

    lib.summarize_with_llama(input_data="text", custom_prompt="Summarize.", api_key=None)

    assert captured_post["url"] == expected


def test_summarize_with_llama_parses_the_openai_response_shape(monkeypatch):
    """It posts to /v1/chat/completions, so it must read choices[].message."""
    monkeypatch.setattr(lib, "load_settings", lambda: _settings())

    class _Session:
        def mount(self, *_a, **_k):
            pass

        def post(self, *_a, **_k):
            return _FakeResponse()

    monkeypatch.setattr(lib, "create_default_session", lambda: _Session())

    assert lib.summarize_with_llama("text", "Summarize.", api_key=None) == "SUMMARY"


def test_summarize_with_llama_still_parses_the_native_completion_shape(monkeypatch):
    """llama.cpp's native endpoint returns a top-level content; a config
    pointing at one must keep working."""
    monkeypatch.setattr(lib, "load_settings", lambda: _settings())

    class _Session:
        def mount(self, *_a, **_k):
            pass

        def post(self, *_a, **_k):
            return _FakeResponse({"content": " NATIVE "})

    monkeypatch.setattr(lib, "create_default_session", lambda: _Session())

    assert lib.summarize_with_llama("text", "Summarize.", api_key=None) == "NATIVE"


def test_summarize_with_llama_reports_an_unusable_payload(monkeypatch):
    """Neither shape present: return a failure the deep-search guard catches
    rather than something that could be stored as evidence."""
    monkeypatch.setattr(lib, "load_settings", lambda: _settings())

    class _Session:
        def mount(self, *_a, **_k):
            pass

        def post(self, *_a, **_k):
            return _FakeResponse({"unexpected": True})

    monkeypatch.setattr(lib, "create_default_session", lambda: _Session())

    result = lib.summarize_with_llama("text", "Summarize.", api_key=None)

    from tldw_chatbook.Web_Scraping.WebSearch_APIs import _is_summary_failure

    assert _is_summary_failure(result), result


# --- token budget and empty-content reporting (task-17384) --------------------
# Chunk summarization failed with "No choices in response data" while
# per-result calls on the same path succeeded. Captured live: the model spends
# its budget on reasoning_content (4028 of 4096 completion tokens on a real
# 6000-char chunk, leaving 465 chars of content), so a chunk that reasons a
# little longer returns EMPTY content. The summarizer was reading max_tokens
# from the legacy section only, never the modern table a run primes -- the same
# split that sent its requests to the wrong port.


def test_summarize_with_llama_prefers_the_modern_max_tokens(monkeypatch, captured_post):
    """A run priming a local endpoint sets the budget in the modern table; the
    summarizer must spend it rather than the legacy default."""
    settings = _settings()
    settings["api_settings"]["llama_cpp"]["max_tokens"] = 16384
    monkeypatch.setattr(lib, "load_settings", lambda: settings)

    lib.summarize_with_llama("text", "Summarize.", api_key=None)

    assert captured_post["json"]["max_tokens"] == 16384


def test_summarize_with_llama_falls_back_to_the_legacy_max_tokens(
    monkeypatch, captured_post
):
    """With no modern entry the historical key still governs the budget."""
    monkeypatch.setattr(lib, "load_settings", lambda: _settings())

    lib.summarize_with_llama("text", "Summarize.", api_key=None)

    assert captured_post["json"]["max_tokens"] == 4096


def test_empty_content_failure_names_the_actual_cause(monkeypatch):
    """The old message blamed missing choices; the real cause is a completion
    that spent its budget on reasoning. A caller reading the run's warnings
    should be able to tell those apart."""
    monkeypatch.setattr(lib, "load_settings", lambda: _settings())

    class _Session:
        def mount(self, *_a, **_k):
            pass

        def post(self, *_a, **_k):
            return _FakeResponse(
                {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"role": "assistant", "content": "",
                                        "reasoning_content": "thinking..." * 50},
                        }
                    ],
                    "usage": {"completion_tokens": 4096},
                }
            )

    monkeypatch.setattr(lib, "create_default_session", lambda: _Session())

    result = lib.summarize_with_llama("text", "Summarize.", api_key=None)

    from tldw_chatbook.Web_Scraping.WebSearch_APIs import _is_summary_failure

    assert _is_summary_failure(result), result
    lowered = result.lower()
    assert "length" in lowered and "reasoning" in lowered, result


def test_budget_and_diagnostic_reach_the_public_analyze_boundary(monkeypatch):
    """Qodo (PR 1774): the other tests call summarize_with_llama directly. This
    one goes through `analyze`, which is the seam the deep-search pipeline
    actually calls, so the budget resolution and the reasoning-only diagnostic
    are verified where a caller meets them rather than one layer in."""
    from tldw_chatbook.LLM_Calls.Summarization_General_Lib import analyze
    from tldw_chatbook.Web_Scraping.WebSearch_APIs import _is_summary_failure

    settings = _settings()
    settings["api_settings"]["llama_cpp"]["max_tokens"] = 16384
    monkeypatch.setattr(lib, "load_settings", lambda: settings)

    seen = {}

    class _Session:
        def mount(self, *_a, **_k):
            pass

        def post(self, url, headers=None, json=None, stream=False):
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

    monkeypatch.setattr(lib, "create_default_session", lambda: _Session())

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
