"""llama.cpp summarization configuration reads (task-17382).

`summarize_with_llama` read `loaded_config_data["llama_api"]`, a section that
has never existed -- the loader builds `llama_cpp_api`. Every call therefore
raised KeyError before contacting a server and returned an error STRING,
which the deep-search caller stored as a result's evidence content.
"""

import pytest

from tldw_chatbook.LLM_Calls import Local_Summarization_Lib as lib


class _FakeResponse:
    status_code = 200

    def __init__(self):
        self.text = ""

    def json(self):
        return {"content": " SUMMARY "}


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

    monkeypatch.setattr(lib.requests, "Session", lambda: _FakeSession())
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
