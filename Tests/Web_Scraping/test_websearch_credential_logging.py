"""TASK-19552: the Google Custom Search API key must never reach a log
record, at any level, regardless of whether the request succeeds or fails.

Two independent leak vectors existed in `search_web_google`
(`tldw_chatbook/Web_Scraping/WebSearch_APIs.py`), both on the DEFAULT
search engine path (`search_engine` defaults to "google"):

  1. `logger.info(f"Prepared parameters for Google Search: {params}")`
     formatted the whole `params` dict, including `params["key"]` -- the
     API key -- at INFO, on every successful call.
  2. `logger.error(f"Error during API request: {str(re)}")` formatted a
     `requests` exception's `str()`. Because this engine (unlike every
     other one in the file) sends `key` as a URL query parameter rather
     than a header, `requests`'s own `HTTPError`/`ConnectionError` text
     embeds the full request URL -- including the key -- via
     `response.url`. This fires on exactly the failure modes a user hits
     debugging a bad/expired key (401/403/429/connection failure).

Both are fixed at the point of formatting: vector 1 via an explicit
ALLOWLIST of safe param keys (`SAFE_GOOGLE_SEARCH_PARAM_KEYS`), vector 2 via
`Utils.log_sanitizer.sanitize_string`, which carries a dedicated
`AIza[A-Za-z0-9_-]{35}` pattern for Google API keys. Each test below proves
the leak is real for the exact runtime value in play (not a hardcoded
string) as a methodology check, then asserts the actual emitted log
records never carry it, while the non-credential diagnostic value survives.
"""

import logging
from urllib.parse import urlencode

import pytest
import requests as real_requests
from loguru import logger as loguru_logger

from tldw_chatbook.Logging_Config import _forward_loguru_to_standard
from tldw_chatbook.Web_Scraping import WebSearch_APIs

# Shaped to match log_sanitizer's Google-API-key pattern (AIza + 35 chars)
# so the redaction path under test actually engages -- an arbitrarily
# shaped secret would NOT be caught by that specific pattern, which is a
# real, disclosed limitation of vector 2's fix (see the task's Implementation
# Notes). Real Google API keys are uniformly AIza-prefixed, so this is the
# shape that matters for this engine.
_SENTINEL_SUFFIX = "TASK19552SENTINELKEYNOTREAL00011234"
assert len(_SENTINEL_SUFFIX) == 35, len(_SENTINEL_SUFFIX)
SENTINEL_KEY = "AIza" + _SENTINEL_SUFFIX


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.headers = {"Content-Type": "application/json"}
        self.text = ""

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise real_requests.exceptions.HTTPError(f"status {self.status_code}")


class _FakeRequests:
    """Stands in for the module's `requests` import; records the call."""

    exceptions = real_requests.exceptions

    def __init__(self, payload, status_code=200):
        self.calls = []
        self._payload = payload
        self._status = status_code

    def get(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return _FakeResponse(self._payload, self._status)


class _HTTPErrorRequests:
    """Raises the SAME exception `requests` itself would raise for a failed
    GET -- built from a real `requests.models.Response` so the message
    format (including the URL) is authentic, not hand-rolled."""

    exceptions = real_requests.exceptions

    def __init__(self):
        self.calls = []

    def get(self, url, params=None, **kwargs):
        full_url = f"{url}?{urlencode(params or {})}"
        self.calls.append({"url": full_url, "params": params})
        resp = real_requests.models.Response()
        resp.status_code = 403
        resp.reason = "Forbidden"
        resp.url = full_url
        resp.raise_for_status()  # raises the real requests.exceptions.HTTPError


def _set_key(monkeypatch, key, value):
    monkeypatch.setitem(WebSearch_APIs.loaded_config_data["search_engines"], key, value)


@pytest.fixture
def loguru_caplog(caplog):
    """Bridge loguru records into caplog (the repo's standard pattern --
    see Tests/Chat/test_chat_api_call_endpoint_redaction.py)."""
    sink_id = loguru_logger.add(_forward_loguru_to_standard, level="DEBUG")
    caplog.set_level(logging.DEBUG)
    try:
        yield caplog
    finally:
        loguru_logger.remove(sink_id)


_GOOGLE_PAYLOAD = {"items": [{"title": "G Title", "link": "https://g.example/", "snippet": "g snippet"}]}


# ---------------------------------------------------------------------------
# Unit-level: the allowlist helper itself
# ---------------------------------------------------------------------------


def test_safe_search_params_for_log_drops_key_keeps_allowlisted_and_unknown():
    """The allowlist drops the credential AND any key it doesn't recognize
    -- an unknown/future param key is safe by default, not exposed by
    default (the property Utils/sensitive_llm_logging.py documents)."""
    params = {
        "q": "hello",
        "key": SENTINEL_KEY,
        "cx": "cx123",
        "num": 5,
        "some_future_field_nobody_allowlisted_yet": "x",
    }
    safe = WebSearch_APIs._safe_search_params_for_log(
        params, WebSearch_APIs.SAFE_GOOGLE_SEARCH_PARAM_KEYS
    )
    assert safe == {"q": "hello", "cx": "cx123", "num": 5}
    assert "key" not in safe
    assert "some_future_field_nobody_allowlisted_yet" not in safe


# ---------------------------------------------------------------------------
# Vector 1: the params-dict INFO log on the success path
# ---------------------------------------------------------------------------


def test_google_success_path_never_logs_the_key(monkeypatch, loguru_caplog):
    _set_key(monkeypatch, "google_search_api_key", SENTINEL_KEY)
    _set_key(monkeypatch, "google_search_engine_id", "cx123")
    fake = _FakeRequests(_GOOGLE_PAYLOAD)
    monkeypatch.setattr(WebSearch_APIs, "requests", fake)

    WebSearch_APIs.search_web_google("cherry cake")

    # Baseline / methodology check: the dict that was ACTUALLY sent on the
    # wire really does carry the key, and formatting that same dict (as the
    # old code did) would leak it -- proves this test can detect the leak.
    assert fake.calls, "search_web_google made no request"
    sent_params = fake.calls[0]["params"]
    assert sent_params["key"] == SENTINEL_KEY
    assert SENTINEL_KEY in str(sent_params)

    # Actual: no emitted log record contains the key.
    messages = [r.getMessage() for r in loguru_caplog.records]
    assert not any(SENTINEL_KEY in m for m in messages), messages

    # The diagnostic must still be useful -- not merely deleted.
    prepared = [m for m in messages if "Prepared parameters for Google Search" in m]
    assert prepared, "expected the params-preparation log line"
    assert "cherry cake" in prepared[0]
    assert "cx123" in prepared[0]


# ---------------------------------------------------------------------------
# Vector 2: the exception-text ERROR log on a failed request
# ---------------------------------------------------------------------------


def test_google_http_error_never_logs_the_key_via_exception_text(monkeypatch, loguru_caplog):
    _set_key(monkeypatch, "google_search_api_key", SENTINEL_KEY)
    _set_key(monkeypatch, "google_search_engine_id", "cx123")
    fake = _HTTPErrorRequests()
    monkeypatch.setattr(WebSearch_APIs, "requests", fake)

    with pytest.raises(real_requests.exceptions.HTTPError) as excinfo:
        WebSearch_APIs.search_web_google("cherry cake")

    # Baseline / methodology check: requests' OWN exception text really
    # does embed the key for this call shape (GET + key as a query param)
    # -- proves this test can detect the leak, and explains why Google
    # (unlike header-auth engines such as Bing) needs this second fix.
    assert SENTINEL_KEY in str(excinfo.value)

    # Actual: no emitted log record contains the key.
    messages = [r.getMessage() for r in loguru_caplog.records]
    assert not any(SENTINEL_KEY in m for m in messages), messages

    # The diagnostic must still be useful -- the error line survives,
    # just without the credential.
    error_lines = [m for m in messages if "Error during API request" in m]
    assert error_lines, "expected the RequestException error log line"
    assert "403" in error_lines[0] or "Forbidden" in error_lines[0]


def test_google_value_error_never_logs_the_key_via_exception_text(monkeypatch, loguru_caplog):
    """Defense in depth for the `except ValueError` branch.

    `response.json()` raises `requests.exceptions.JSONDecodeError` on
    malformed content -- which is ALSO a `ValueError` subclass, so a
    real-world bad-JSON response is caught by this function's FIRST except
    clause (`except ValueError as ve`), not the `RequestException`/generic
    `Exception` clauses covered above. A naturally occurring JSON-decode
    message never embeds the request URL, so this test injects the
    sentinel into a ValueError message directly (a synthetic, not a
    naturally-occurring, message) to prove this branch is *also* sanitized
    -- not merely the two branches a JSONDecodeError happens not to reach.
    """
    _set_key(monkeypatch, "google_search_api_key", SENTINEL_KEY)
    _set_key(monkeypatch, "google_search_engine_id", "cx123")

    class _BadJSONResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            raise ValueError(f"synthetic decode failure embedding key={SENTINEL_KEY}")

    class _BadJSONRequests:
        exceptions = real_requests.exceptions

        def get(self, url, params=None, **kwargs):
            return _BadJSONResponse()

    monkeypatch.setattr(WebSearch_APIs, "requests", _BadJSONRequests())

    with pytest.raises(ValueError) as excinfo:
        WebSearch_APIs.search_web_google("cherry cake")

    assert SENTINEL_KEY in str(excinfo.value)

    messages = [r.getMessage() for r in loguru_caplog.records]
    assert not any(SENTINEL_KEY in m for m in messages), messages
    error_lines = [m for m in messages if "Configuration error" in m]
    assert error_lines, "expected the ValueError error log line"
