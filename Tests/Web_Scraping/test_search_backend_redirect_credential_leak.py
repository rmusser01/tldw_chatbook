"""task-19862: the four custom-header search backends never leak their key.

TASK-32894 already routed Bing/Brave/Serper/Exa (plus the four
``Authorization``-based backends) through
``WebSearch_APIs._credentialed_search_request``, which passes
``allow_redirects=False`` and refuses a 30x with ``EgressFetchError``.
Its own tests assert the KWARG via a recording fake -- strong against a
regression at the transport, but they never drive a real redirect, so
they cannot show the sentinel key failing to arrive at the second host.

These tests close that evidence gap for the four sites task-19862 named,
in the TASK-19557 idiom: transport is faked at
``requests.adapters.HTTPAdapter.send`` -- the layer just above the real
socket -- so ``requests``' own redirect machinery
(``Session.resolve_redirects``, ``rebuild_method``, ``rebuild_auth``,
``rebuild_headers``) still runs for real. A first host answers with a
30x whose ``Location`` names a second host; the second host records what
it receives. Only the actual network I/O is replaced; no test here makes
a real network call.

Credential per backend (each a synthetic sentinel, never a real key):

* Bing   -- ``Ocp-Apim-Subscription-Key``, endpoint config-supplied
* Brave  -- ``X-Subscription-Token``
* Serper -- ``X-API-KEY``
* Exa    -- ``x-api-key``

``requests`` strips only ``Authorization``/``Cookie`` across a
cross-origin hop, so all four headers would otherwise be forwarded
verbatim to whatever host the redirect names.

Born-red (mutation check): flipping the transport's
``allow_redirects=False`` back to ``True`` makes every case below fail
by showing the sentinel key delivered to the second host.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

import pytest
import requests
import requests.adapters

from tldw_chatbook.Utils.egress import EgressFetchError
from tldw_chatbook.Web_Scraping import WebSearch_APIs

_EVIL_HOST = "evil.example"
_EVIL_LOCATION = f"https://{_EVIL_HOST}/steal"
#: A well-formed per-provider success body the (unfixed) followed request
#: would return -- the silent-disclosure shape the refusal must prevent.
_EVIL_BODY = b'{"results": [], "organic": [], "web": {"results": []}}'

#: (id, first-hop host, key header name, sentinel value, callable that
#: drives the backend, extra ``search_engines`` settings). Brave/Serper/
#: Exa endpoints are hardcoded literals in the module, so their "first
#: hop" is that literal host; Bing's endpoint is config-supplied, so the
#: stub settings point it at the test's first-hop host.
_SITES: dict[str, dict[str, Any]] = {
    "bing": {
        "good_host": "good.example",
        "key_header": "ocp-apim-subscription-key",
        "sentinel": "sentinel-bing-subscription-key-must-never-leak",
        "settings": {
            "bing_search_api_url": "https://good.example/v7.0/search",
            "search_result_max": 5,
        },
        "call": lambda settings: WebSearch_APIs.search_web_bing(
            "q", bing_api_key=settings["bing_search_api_key"]
        ),
        "settings_key": "bing_search_api_key",
    },
    "brave": {
        "good_host": "api.search.brave.com",
        "key_header": "x-subscription-token",
        "sentinel": "sentinel-brave-subscription-token-must-never-leak",
        "settings": {},
        "call": lambda settings: WebSearch_APIs.search_web_brave(
            "q", "US", "en", "en-US", 5, brave_api_key=settings["brave_api_key"]
        ),
        "settings_key": "brave_api_key",
    },
    "serper": {
        "good_host": "google.serper.dev",
        "key_header": "x-api-key",
        "sentinel": "sentinel-serper-api-key-must-never-leak",
        "settings": {},
        "call": lambda settings: WebSearch_APIs.search_web_serper("q"),
        "settings_key": "serper_search_api_key",
    },
    "exa": {
        "good_host": "api.exa.ai",
        "key_header": "x-api-key",
        "sentinel": "sentinel-exa-api-key-must-never-leak",
        "settings": {},
        "call": lambda settings: WebSearch_APIs.search_web_exa("q"),
        "settings_key": "exa_search_api_key",
    },
}


class _FakeRaw:
    """Just enough of a urllib3 raw response for ``iter_content``.

    Only the followed-redirect path (the mutation check, or a regression)
    ever iterates the FINAL response's body through
    ``_credentialed_search_request``'s capped reader; the refused 30x is
    rejected before iteration. ``_original_response`` is deliberately
    absent so requests' cookie extraction skips it.
    """

    def __init__(self, body: bytes) -> None:
        self._body = body
        self._offset = 0
        self.closed = False

    def read(self, amount: int = -1) -> bytes:
        if amount is None or amount < 0:
            chunk = self._body[self._offset :]
        else:
            chunk = self._body[self._offset : self._offset + amount]
        self._offset += len(chunk)
        return chunk

    def stream(self, chunk_size: int, decode_content: bool = True):
        while True:
            chunk = self.read(chunk_size)
            if not chunk:
                break
            yield chunk

    def close(self) -> None:
        self.closed = True


def _fake_response(
    status_code: int,
    headers: dict[str, str],
    body: bytes = b"",
    *,
    close_calls: list[int] | None = None,
    request: requests.PreparedRequest | None = None,
) -> requests.Response:
    """Build a bare ``requests.Response`` double that ``close()`` safely.

    Mirrors the TASK-19557 double: ``_content_consumed = True`` makes an
    unpatched ``.close()`` a safe no-op rather than an ``AttributeError``
    a broad ``except`` would swallow, and ``request`` must be set because
    ``Session.send``'s redirect peek-ahead dereferences
    ``response.request.url`` inside ``rebuild_auth``. ``raw`` is a
    minimal reader so the capped body iteration works on the followed
    path. When ``close_calls`` is given, every ``.close()`` is recorded
    there -- the signal the closure assertion checks.
    """
    resp = requests.Response()
    resp.status_code = status_code
    resp.headers = requests.structures.CaseInsensitiveDict(headers or {})
    resp._content = body
    resp._content_consumed = True
    resp.encoding = "utf-8"
    resp.raw = _FakeRaw(body)
    resp.request = request
    if close_calls is not None:
        original_close = resp.close

        def _counting_close() -> None:
            close_calls.append(1)
            original_close()

        resp.close = _counting_close
    return resp


def _install_redirecting_adapter(
    monkeypatch: pytest.MonkeyPatch,
    seen: dict[str, dict[str, object]],
    *,
    good_host: str,
    redirect_status: int,
) -> list[int]:
    """Patch ``HTTPAdapter.send`` so ``good_host`` 30x's to ``evil.example``.

    Bing builds a ``requests.Session`` with retry adapters; the other
    three go through the module-level ``requests`` -- both dispatch
    through an ``HTTPAdapter`` instance, so patching the class method
    intercepts every path without knowing which one a backend takes.

    Returns:
        A list that accumulates one entry per ``.close()`` call on the
        redirect response from ``good_host``.
    """
    redirect_close_calls: list[int] = []

    def _fake_send(self, request, **kwargs):  # noqa: ANN001 - requests API
        host = urlsplit(request.url).netloc
        seen[host] = {
            "method": request.method,
            "headers": {k.lower(): v for k, v in request.headers.items()},
        }
        if host == good_host:
            return _fake_response(
                redirect_status,
                {"Location": _EVIL_LOCATION},
                close_calls=redirect_close_calls,
                request=request,
            )
        if host == _EVIL_HOST:
            return _fake_response(
                200,
                {"Content-Type": "application/json"},
                _EVIL_BODY,
                request=request,
            )
        raise AssertionError(f"unexpected host in test transport: {host}")

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", _fake_send)
    return redirect_close_calls


def _install_stub_settings(
    monkeypatch: pytest.MonkeyPatch, site: dict[str, Any]
) -> dict[str, str]:
    """Stub ``initialize_config``/``requests_verify`` and return settings."""
    values = dict(site["settings"])
    values[site["settings_key"]] = site["sentinel"]
    settings = {"search_engines": values}
    monkeypatch.setattr(WebSearch_APIs, "initialize_config", lambda: settings)
    # Never read the guarded config loader (and never dial out).
    monkeypatch.setattr(WebSearch_APIs, "requests_verify", lambda: True)
    return values


@pytest.mark.parametrize("redirect_status", [302, 303, 307])
@pytest.mark.parametrize("site_id", sorted(_SITES))
def test_search_site_refuses_redirect_and_never_leaks_its_key(
    monkeypatch: pytest.MonkeyPatch, site_id: str, redirect_status: int
) -> None:
    """A 30x from the provider endpoint must not be followed.

    For the POST-based backends (Serper, Exa) a 302/303 additionally
    converts the POST to a GET before the re-issue -- the same silent
    method conversion task-19862's KoboldAI case calls out. All three
    redirect statuses are driven for every site.
    """
    site = _SITES[site_id]
    seen: dict[str, dict[str, object]] = {}
    values = _install_stub_settings(monkeypatch, site)
    redirect_close_calls = _install_redirecting_adapter(
        monkeypatch,
        seen,
        good_host=site["good_host"],
        redirect_status=redirect_status,
    )

    raised: Exception | None = None
    try:
        site["call"](values)
    except EgressFetchError as exc:
        raised = exc
    except Exception as exc:  # noqa: BLE001 - classified below
        raised = None
        pytest.fail(
            f"{site_id}: expected EgressFetchError on redirect refusal, "
            f"got {type(exc).__name__}: {exc}"
        )

    # Sanity: the first hop really did carry the credential.
    assert site["good_host"] in seen
    good_headers = seen[site["good_host"]]["headers"]
    assert good_headers.get(site["key_header"]) == site["sentinel"], (
        f"{site_id}: first hop did not carry its credential header "
        f"({site['key_header']}); the test proved nothing"
    )

    # Load-bearing: the credential -- and any request at all -- must never
    # reach the redirect target.
    assert _EVIL_HOST not in seen, (
        f"{site_id} request was re-issued to the redirect target "
        f"{_EVIL_HOST!r} as {seen.get(_EVIL_HOST, {}).get('method')!r} "
        f"carrying headers {seen.get(_EVIL_HOST, {}).get('headers')!r}"
    )

    # The refusal must be loud, not a silent follow.
    assert raised is not None, (
        f"{site_id}: expected EgressFetchError on redirect refusal; the "
        "call returned without one"
    )
    assert "redirect" in str(raised).lower()

    # The refusal message must not echo the attacker-controlled Location
    # header (TASK-19321/19552/19557 exception-text rule).
    refusal_text = str(raised)
    assert _EVIL_HOST not in refusal_text and "/steal" not in refusal_text

    # The refused response must be explicitly closed by the refusal path
    # (the helper's ``finally``), in addition to requests' own
    # redirect-peek close: exactly 2, never 1 (see the sibling Anthropic
    # module's verified rationale for why >= 1 proves nothing).
    assert len(redirect_close_calls) == 2, (
        f"{site_id}: expected the refusal path's own response.close() in "
        f"addition to requests' resolve_redirects() peek close; observed "
        f"{len(redirect_close_calls)} close() call(s)"
    )
