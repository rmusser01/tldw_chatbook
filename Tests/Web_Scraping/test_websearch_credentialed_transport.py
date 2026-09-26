"""TASK-32894: credentialed search-provider calls must not follow redirects.

`requests.sessions.SessionRedirectMixin.rebuild_auth` strips **only**
`Authorization` across a cross-origin hop. Every other credential header
these backends use -- `X-Subscription-Token` (Brave), `X-API-KEY` (Serper),
`x-api-key` (Exa), `Ocp-Apim-Subscription-Key` (Bing) -- and Google's
credential, which rides in the *query string*, are forwarded verbatim to
whatever a 30x names. Bing's endpoint is config-supplied and unvalidated,
so the redirect target need not be the vendor's.

The same eight calls were also the only outbound HTTP in `Web_Scraping/`
with no response size cap, while the package defines and uses
`MAX_FETCH_BYTES_PAGE` everywhere else.

No test here touches the network: `WebSearch_APIs.requests` is replaced by
a recorder.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from tldw_chatbook.Web_Scraping import WebSearch_APIs

MODULE_PATH = Path(WebSearch_APIs.__file__)

#: Credential-carrying backends and the minimum settings each one reads.
#: Bing and Google are covered by the census below instead -- their config
#: branching is long enough that stubbing it would test the stub.
CREDENTIALED_BACKENDS = {
    "brave": (
        lambda: WebSearch_APIs.search_web_brave("q", "US", "en", "en", 5),
        {"brave_search_api_key": "brave-key"},
    ),
    "kagi": (
        lambda: WebSearch_APIs.search_web_kagi("q"),
        {"kagi_search_api_key": "kagi-key"},
    ),
    "serper": (
        lambda: WebSearch_APIs.search_web_serper("q"),
        {"serper_search_api_key": "serper-key"},
    ),
    "exa": (
        lambda: WebSearch_APIs.search_web_exa("q"),
        {"exa_search_api_key": "exa-key"},
    ),
    "tavily": (
        lambda: WebSearch_APIs.search_web_tavily("q"),
        {"tavily_search_api_key": "tavily-key"},
    ),
    "yandex": (
        lambda: WebSearch_APIs.search_web_yandex("q"),
        {
            "yandex_search_api_key": "yandex-key",
            "yandex_search_folder_id": "folder",
        },
    ),
}


class _Response:
    """Models the shape the capped reader needs, not just `.json()`."""

    def __init__(self, payload, status_code=200, is_redirect=False):
        self.status_code = status_code
        self.is_redirect = is_redirect
        self.headers = {"Content-Type": "application/json"}
        self._body = json.dumps(payload).encode()
        self.text = self._body.decode()

    def iter_content(self, chunk_size=65536):
        for start in range(0, len(self._body), chunk_size):
            yield self._body[start : start + chunk_size]

    @property
    def content(self):
        return self._body

    def json(self):
        return json.loads(self._body)

    def close(self):
        return None

    def raise_for_status(self):
        if self.status_code >= 400:
            raise WebSearch_APIs.requests.exceptions.HTTPError(
                f"status {self.status_code}"
            )


class _Recorder:
    """Stands in for the module's `requests` import AND for a Session."""

    def __init__(self, payload=None, is_redirect=False, body_bytes=None):
        self.calls = []
        self._payload = payload if payload is not None else {"rawData": ""}
        self._is_redirect = is_redirect
        self._body_bytes = body_bytes

    exceptions = None  # bound in the fixture to the real exceptions module

    def _record(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        response = _Response(self._payload, is_redirect=self._is_redirect)
        if self._body_bytes is not None:
            response._body = self._body_bytes
        return response

    def get(self, url, **kwargs):
        return self._record("get", url, **kwargs)

    def post(self, url, **kwargs):
        return self._record("post", url, **kwargs)

    def Session(self):
        return self

    def mount(self, *args, **kwargs):
        return None

    verify = True


@pytest.fixture
def recorder(monkeypatch):
    rec = _Recorder()
    rec.exceptions = WebSearch_APIs.requests.exceptions
    monkeypatch.setattr(WebSearch_APIs, "requests", rec)
    monkeypatch.setattr(WebSearch_APIs, "requests_verify", lambda: True)
    return rec


def _stub_settings(monkeypatch, values):
    monkeypatch.setattr(
        WebSearch_APIs,
        "initialize_config",
        lambda: {"search_engines": dict(values)},
    )


@pytest.mark.parametrize("backend", sorted(CREDENTIALED_BACKENDS))
def test_credentialed_call_refuses_redirects(backend, monkeypatch, recorder):
    call, settings = CREDENTIALED_BACKENDS[backend]
    _stub_settings(monkeypatch, settings)
    try:
        call()
    except Exception as error:  # parsing of the stub payload is not under test
        if not recorder.calls:
            raise AssertionError(f"{backend} never issued a request: {error}")
    assert recorder.calls, f"{backend} never issued a request; the test proved nothing"
    for record in recorder.calls:
        assert record.get("allow_redirects") is False, (
            f"{backend} follows redirects on a credentialed request: {record}"
        )


def test_a_redirect_on_a_credentialed_call_is_refused(monkeypatch):
    rec = _Recorder(is_redirect=True)
    rec.exceptions = WebSearch_APIs.requests.exceptions
    monkeypatch.setattr(WebSearch_APIs, "requests", rec)
    monkeypatch.setattr(WebSearch_APIs, "requests_verify", lambda: True)
    _stub_settings(monkeypatch, {"exa_search_api_key": "exa-key"})
    with pytest.raises(Exception):
        WebSearch_APIs.search_web_exa("q")


def test_an_oversized_response_is_refused_rather_than_buffered(monkeypatch):
    from tldw_chatbook.Utils.egress import EgressFetchError

    rec = _Recorder(body_bytes=b"x" * (WebSearch_APIs.MAX_SEARCH_RESPONSE_BYTES + 1))
    rec.exceptions = WebSearch_APIs.requests.exceptions
    monkeypatch.setattr(WebSearch_APIs, "requests", rec)
    monkeypatch.setattr(WebSearch_APIs, "requests_verify", lambda: True)
    _stub_settings(monkeypatch, {"exa_search_api_key": "exa-key"})
    with pytest.raises(EgressFetchError):
        WebSearch_APIs.search_web_exa("q")


# ---------------------------------------------------------------------------
# The census: no credentialed backend may go back to a bare requests call.
# ---------------------------------------------------------------------------

#: The two uncredentialed backends. DuckDuckGo's HTML endpoint and a SearX
#: instance carry no API key, so `rebuild_auth` has nothing to leak; they
#: are named here so the exemption is a decision on the record rather than
#: a gap in the sweep.
_UNCREDENTIALED_CALLERS = frozenset({"search_web_duckduckgo", "search_web_searx"})


def _bare_http_calls() -> dict[str, list[int]]:
    """`requests.get/post` or `<session>.get/post` calls, by enclosing function."""
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    found: dict[str, list[int]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Call):
                continue
            func = inner.func
            if not isinstance(func, ast.Attribute) or func.attr not in {"get", "post"}:
                continue
            value = func.value
            if isinstance(value, ast.Name) and value.id in {"requests", "session"}:
                found.setdefault(node.name, []).append(inner.lineno)
    return found


def test_no_credentialed_backend_issues_a_bare_requests_call():
    offenders = {
        name: lines
        for name, lines in _bare_http_calls().items()
        if name not in _UNCREDENTIALED_CALLERS
    }
    assert offenders == {}, (
        "these functions reach the network without the capped, "
        f"redirect-refusing helper: {offenders}"
    )
