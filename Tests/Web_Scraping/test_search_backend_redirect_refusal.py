"""Credentialed search-provider calls do not follow redirects.

Tier-2 review S14, P2 [D1/D4]: `WebSearch_APIs.py` is the only outbound
HTTP in `Web_Scraping/` that does not route through `Utils/egress.py`;
every provider call uses bare `requests` with `allow_redirects` at its
`True` default.

`requests.sessions.SessionRedirectMixin.rebuild_auth` strips **only**
`Authorization` on a cross-host hop. Four backends authenticate with a
CUSTOM header, which requests has no notion of and forwards verbatim to
whatever host the redirect names:

* Bing   -- `Ocp-Apim-Subscription-Key`, and its `search_url` is
             **config-supplied** (`bing_search_api_url`), with no
             URL-shape validation in `search_backend_settings.BACKENDS`
* Brave  -- `X-Subscription-Token`
* Serper -- `X-API-KEY`
* Exa    -- `x-api-key`

A search API has no legitimate reason to redirect, so the minimum correct
fix is to refuse rather than follow. The backends whose credential rides
in `Authorization` (Kagi, Tavily, Yandex) are already covered by
`rebuild_auth`, and the two uncredentialed ones (DuckDuckGo's HTML
endpoint, a user-configured SearX instance) may legitimately redirect --
so they are deliberately not changed here.

Asserted against the sentinel key VALUE reaching a redirect target is not
possible without a live redirect, so these assert the kwarg that makes it
impossible, at every credentialed call site, by AST census as well as by
driving each function.
"""

from __future__ import annotations

import ast
import inspect
from typing import Any

import pytest
import requests as real_requests

from tldw_chatbook.Web_Scraping import WebSearch_APIs

_PAYLOAD = {"web": {"results": []}, "organic": [], "results": [], "webPages": {}}


class FakeResponse:
    status_code = 200
    headers = {"Content-Type": "application/json"}
    text = ""
    content = b"{}"

    def json(self) -> dict:
        return _PAYLOAD

    def raise_for_status(self) -> None:
        return None


class RecordingRequests:
    """Stands in for the module's `requests`, recording every call."""

    exceptions = real_requests.exceptions

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        return FakeResponse()

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        return FakeResponse()

    def Session(self) -> "RecordingSession":  # noqa: N802 - mirrors requests
        return RecordingSession(self)


class RecordingSession:
    def __init__(self, parent: RecordingRequests) -> None:
        self._parent = parent
        self.verify = True

    def mount(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        return self._parent.get(url, **kwargs)


_SETTINGS = {
    "search_engines": {
        "bing_search_api_url": "https://api.bing.microsoft.invalid/v7.0/search",
        "bing_search_api_key": "SENTINEL",
        "brave_search_api_key": "SENTINEL",
        "serper_search_api_key": "SENTINEL",
        "exa_search_api_key": "SENTINEL",
        "search_result_max": 5,
    }
}


@pytest.fixture
def recording(monkeypatch: pytest.MonkeyPatch) -> RecordingRequests:
    fake = RecordingRequests()
    monkeypatch.setattr(WebSearch_APIs, "requests", fake)
    # Never read the user's real config (and never dial out).
    monkeypatch.setattr(WebSearch_APIs, "initialize_config", lambda: _SETTINGS)
    monkeypatch.setattr(WebSearch_APIs, "requests_verify", lambda: True)
    return fake


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            lambda: WebSearch_APIs.search_web_bing("q", bing_api_key="SENTINEL"),
            id="bing",
        ),
        pytest.param(
            lambda: WebSearch_APIs.search_web_brave(
                "q", "US", "en", "en-US", 5, brave_api_key="SENTINEL"
            ),
            id="brave",
        ),
        pytest.param(
            lambda: WebSearch_APIs.search_web_serper("q"),
            id="serper",
        ),
        pytest.param(
            lambda: WebSearch_APIs.search_web_exa("q"),
            id="exa",
        ),
    ],
)
def test_a_credentialed_provider_call_refuses_to_follow_a_redirect(
    call, recording: RecordingRequests
) -> None:
    call()

    assert recording.calls, "no request was issued"
    for issued in recording.calls:
        assert issued.get("allow_redirects") is False, (
            "this call carries a custom API-key header that requests' "
            f"rebuild_auth does not strip: {sorted(issued.get('headers') or {})}"
        )


#: Source-level census, so a NEW credentialed backend cannot be added
#: without the kwarg (the per-call tests above only cover what exists).
_CREDENTIALED = (
    "search_web_bing",
    "search_web_brave",
    "search_web_serper",
    "search_web_exa",
)


def _request_calls(func_name: str) -> list[ast.Call]:
    source = inspect.getsource(getattr(WebSearch_APIs, func_name))
    tree = ast.parse(source.lstrip())
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"get", "post"}
        and any(kw.arg == "headers" for kw in node.keywords)
    ]


@pytest.mark.parametrize("func_name", _CREDENTIALED)
def test_every_credentialed_call_site_passes_the_kwarg(func_name: str) -> None:
    calls = _request_calls(func_name)
    assert calls, f"{func_name}: no headers-carrying request call found"
    for node in calls:
        kwarg = next(
            (kw for kw in node.keywords if kw.arg == "allow_redirects"), None
        )
        assert kwarg is not None, f"{func_name}:{node.lineno} has no allow_redirects"
        assert isinstance(kwarg.value, ast.Constant) and kwarg.value.value is False
