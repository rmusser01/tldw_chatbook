"""TASK-19735: the Tavily API key must not sit in the request-parameter dict.

This one is **latent, not a live leak**, and is sized as such. At this
branch's base `search_web_tavily` built:

    payload = {"api_key": tavily_api_key, "query": ..., "max_results": ...}

Nothing logged `payload`, and because the key travelled in the POST *body*
the exception path was clean too -- `str(e)` on a `requests` exception
carries the URL, not the body. The hazard was prospective: a maintainer
adding `logger.debug(f"payload: {payload}")` while debugging writes the key
to disk, with no local signal that it is dangerous. Every sibling backend in
the file keeps its credential in a `headers` dict, which reads as "do not
log this"; this one did not.

Per the owner's standing ruling (durable over clever) the fix makes the
hazard structurally impossible rather than relying on a future reviewer:
Tavily accepts `Authorization: Bearer <key>`, so the credential moves to
`headers` like every sibling.

The already-clean properties are **pinned, not assumed** (the task's fourth
AC): the error string returned on a request exception is asserted to be
credential-free, and the sibling backends' shapes are asserted by an AST
census rather than recorded in prose that can go stale.
"""

from __future__ import annotations

import ast
import inspect
import json
from typing import Any, Dict, List

import pytest
import requests as real_requests

from tldw_chatbook.Web_Scraping import WebSearch_APIs

SENTINEL_KEY = "tvly-TASK19735-SENTINEL-NOT-A-REAL-KEY"

_TAVILY_PAYLOAD = {
    "results": [
        {"title": "T", "url": "https://t.example/", "content": "c", "score": 0.5}
    ],
    "answer": None,
}


class _RecordingRequests:
    """Stands in for the module's `requests` import; records the POST."""

    exceptions = real_requests.exceptions

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any):
        self.calls.append({"url": url, **kwargs})
        return _FakeResponse(_TAVILY_PAYLOAD)


class _FakeResponse:
    def __init__(self, payload: Dict[str, Any], status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.headers = {"Content-Type": "application/json"}
        self.text = ""

    def json(self) -> Dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise real_requests.exceptions.HTTPError(f"status {self.status_code}")


@pytest.fixture
def tavily_key(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setitem(
        WebSearch_APIs.loaded_config_data["search_engines"],
        "tavily_search_api_key",
        SENTINEL_KEY,
    )
    return SENTINEL_KEY


@pytest.fixture
def recording_requests(monkeypatch: pytest.MonkeyPatch) -> _RecordingRequests:
    fake = _RecordingRequests()
    monkeypatch.setattr(WebSearch_APIs, "requests", fake)
    return fake


def _request_parameters(call: Dict[str, Any]) -> Dict[str, Any]:
    """The loggable request-parameter object actually sent."""
    if "json" in call:
        return dict(call["json"])
    body = call.get("data")
    if isinstance(body, (bytes, str)):
        return json.loads(body)
    return dict(body or {})


# ---------------------------------------------------------------------------
# The fix
# ---------------------------------------------------------------------------


def test_tavily_request_parameters_hold_no_credential(
    tavily_key: str, recording_requests: _RecordingRequests
) -> None:
    """A debug log of the request parameters cannot disclose the key.

    Asserted against the sentinel VALUE, not against the key name: a fix
    that merely renamed `api_key` would leave the credential in the dict.
    """
    WebSearch_APIs.search_web_tavily("cherry cake", result_count=3)

    assert recording_requests.calls, "no request was issued"
    parameters = _request_parameters(recording_requests.calls[0])
    rendered = json.dumps(parameters)
    assert SENTINEL_KEY not in rendered, (
        f"the Tavily key is inside the request-parameter object: {parameters}"
    )
    # ...and the ordinary parameters are still there, so the assertion above
    # is not passing because the request was gutted.
    assert parameters.get("query") == "cherry cake"
    assert parameters.get("max_results") == 3


def test_tavily_credential_travels_in_the_headers(
    tavily_key: str, recording_requests: _RecordingRequests
) -> None:
    """It still authenticates -- via the transport every sibling uses."""
    WebSearch_APIs.search_web_tavily("cherry cake")

    headers = recording_requests.calls[0]["headers"]
    rendered = " ".join(f"{k}: {v}" for k, v in headers.items())
    assert SENTINEL_KEY in rendered, f"the key reaches no transport at all: {headers}"
    assert headers.get("Authorization") == f"Bearer {SENTINEL_KEY}"


def test_tavily_optional_domain_filters_still_reach_the_request(
    tavily_key: str, recording_requests: _RecordingRequests
) -> None:
    WebSearch_APIs.search_web_tavily(
        "cake", site_whitelist=["a.example"], site_blacklist=["b.example"]
    )
    parameters = _request_parameters(recording_requests.calls[0])
    assert parameters["include_domains"] == ["a.example"]
    assert parameters["exclude_domains"] == ["b.example"]


# ---------------------------------------------------------------------------
# The already-clean properties, pinned rather than assumed
# ---------------------------------------------------------------------------


def test_tavily_error_string_carries_no_credential(
    tavily_key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The returned error text is the one string a user actually sees.

    Built from a real `requests.models.Response` so the message format --
    including the URL `requests` embeds -- is authentic rather than
    hand-rolled.
    """

    class _FailingRequests:
        exceptions = real_requests.exceptions

        def post(self, url: str, **kwargs: Any):
            response = real_requests.models.Response()
            response.status_code = 401
            response.reason = "Unauthorized"
            response.url = url
            response.raise_for_status()

    monkeypatch.setattr(WebSearch_APIs, "requests", _FailingRequests())
    result = WebSearch_APIs.search_web_tavily("cherry cake")

    assert isinstance(result, str)
    assert "error searching for content" in result
    assert SENTINEL_KEY not in result, f"the error string leaked the key: {result}"


# ---------------------------------------------------------------------------
# Sibling-backend census (the task's fifth AC, in executable form)
# ---------------------------------------------------------------------------

#: Request-parameter object names -- the dicts a debugging maintainer
#: reaches for when adding a log line.
_PARAMETER_DICT_NAMES = frozenset({"payload", "params", "data", "body"})

#: The one documented exception. `search_web_google` sends its credential as
#: a URL query parameter (`params["key"]`), which is the engine's own API
#: contract; TASK-19552 addressed the disclosure that shape caused by fixing
#: the two places it was FORMATTED (an INFO log of `params`, and a
#: `str(exception)` carrying the request URL). Recorded here so the census
#: stays honest instead of silently green.
_KNOWN_PARAMETER_CREDENTIALS = {("search_web_google", "params")}


def _credential_carrying_parameter_dicts() -> set[tuple[str, str]]:
    """Every `search_web_*` function that puts an api-key into a param dict.

    Detects both shapes: a dict literal assigned to a parameter-object name
    with an api-key-ish value, and a later `params["key"] = api_key`
    subscript assignment.
    """
    tree = ast.parse(inspect.getsource(WebSearch_APIs))
    found: set[tuple[str, str]] = set()

    def _is_credential(node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return "api_key" in node.id.lower()
        if isinstance(node, ast.Subscript):
            key = getattr(node.slice, "value", None)
            return isinstance(key, str) and "api_key" in key.lower()
        return False

    for func in ast.walk(tree):
        if not isinstance(func, ast.FunctionDef) or not func.name.startswith(
            "search_web_"
        ):
            continue
        for node in ast.walk(func):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                name = None
                if isinstance(target, ast.Name):
                    name = target.id
                elif isinstance(target, ast.Subscript) and isinstance(
                    target.value, ast.Name
                ):
                    name = target.value.id
                if name not in _PARAMETER_DICT_NAMES:
                    continue
                values = (
                    node.value.values
                    if isinstance(node.value, ast.Dict)
                    else [node.value]
                )
                if any(_is_credential(v) for v in values):
                    found.add((func.name, name))
    return found


def test_no_search_backend_hides_a_credential_in_its_parameter_dict() -> None:
    """The census, executable so it cannot go stale.

    Expected clean (credential in `headers`): bing
    (`Ocp-Apim-Subscription-Key`), brave (`X-Subscription-Token`), serper
    (`X-API-KEY`), exa (`x-api-key`), kagi (`Authorization: Bot ...`),
    yandex (`Authorization: Api-Key ...`). searx/duckduckgo/baidu carry no
    credential at all.
    """
    assert _credential_carrying_parameter_dicts() == _KNOWN_PARAMETER_CREDENTIALS


def test_the_census_detector_actually_detects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Methodology check: the AST walk finds the shape it claims to find.

    Without this, a detector that silently matched nothing would make the
    census above pass forever.
    """
    source = (
        "def search_web_fake(q):\n"
        "    fake_api_key = 'x'\n"
        "    payload = {'api_key': fake_api_key, 'query': q}\n"
        "    return payload\n"
    )
    monkeypatch.setattr(inspect, "getsource", lambda _m: source)
    assert _credential_carrying_parameter_dicts() == {("search_web_fake", "payload")}
