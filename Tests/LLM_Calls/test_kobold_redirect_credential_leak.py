"""task-19862: KoboldAI's ``X-Api-Key`` must never survive a redirect.

``chat_with_kobold`` (``LLM_Calls/LLM_API_Calls_Local.py``) POSTs to a
**user-configured** URL (``[api_settings.koboldcpp].api_url`` or the
``api_base_url`` argument) carrying the API key in the custom
``X-Api-Key`` header. ``requests`` strips only ``Authorization``/``Cookie``
on a cross-host redirect -- it has no notion of ``X-Api-Key`` -- so with
``requests``' default ``allow_redirects=True`` a 302/303-serving endpoint
made the client convert the POST to GET and re-issue it to whatever host
the ``Location`` header named, still carrying the key verbatim. KoboldAI
was the worst of task-19862's five sites precisely because the
destination is user-configured: a redirecting endpoint is reachable by
ordinary misconfiguration, not only a compromised vendor.

The fix under test: ``allow_redirects=False`` on the ``session.post`` and
an explicit, closing, ``ChatProviderError`` refusal of any 3xx before the
body is parsed -- the same convention TASK-19557 shipped for
``chat_with_anthropic``/``chat_with_google`` and TASK-32894 for the
search backends.

Born-red: reverting ``allow_redirects=False`` (and the 3xx refusal) makes
every test below fail by showing the sentinel key delivered to the
cross-origin host -- for 302/303 as a GET, the method conversion that a
307-only test cannot demonstrate.

Transport is faked at ``requests.adapters.HTTPAdapter.send`` -- the layer
just above the real socket -- so ``requests``' own redirect/session
machinery (``Session.resolve_redirects``, ``rebuild_method``,
``rebuild_auth``, ``rebuild_headers``) still runs for real; only the
actual network I/O is replaced. No test in this module makes a real
network call.
"""

from __future__ import annotations

from urllib.parse import urlsplit

import pytest
import requests
import requests.adapters

import tldw_chatbook.LLM_Calls.LLM_API_Calls_Local as llm_local
from tldw_chatbook.Chat.Chat_Deps import ChatProviderError

# Synthetic sentinel -- never a real credential.
_SENTINEL_KEY = "sentinel-kobold-x-api-key-must-never-leak"

# ``chat_with_kobold`` builds its session through
# ``Utils.egress.create_default_session``, whose default-timeout and
# TLS-trust reads go through the guarded config loader; under the per-test
# sandbox redirect that admission fails closed with RecoveryRequired
# (same signature as ``test_hosted_chat.py`` et al., see
# ``Tests/conftest.py``). This module drives the real provider function,
# so it opts each node into the bootstrap profile instead of faking the
# config getters -- no config seam is patched, the production session
# construction runs for real.
pytestmark = pytest.mark.bootstrap_profile

_GOOD_HOST = "good.example"
_EVIL_HOST = "evil.example"
_EVIL_LOCATION = f"https://{_EVIL_HOST}/steal"
#: A body the (unfixed) followed request would happily turn into a
#: successful generation -- the silent-disclosure shape the fix must turn
#: into a loud error.
_EVIL_BODY = b'{"results": [{"text": "stolen"}]}'


class _FakeRaw:
    """Just enough of a urllib3 raw response for ``iter_content``.

    Only the followed-redirect path (the mutation check, or a regression)
    ever iterates the FINAL response's body; the refused 30x is closed
    before that. ``_original_response`` is deliberately absent so
    requests' cookie extraction skips it.
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

    Mirrors the TASK-19557 double (``test_anthropic_redirect_credential_
    leak.py``): ``_content_consumed = True`` so an unpatched ``.close()``
    is a safe no-op rather than an ``AttributeError`` on ``raw = None``
    that the production code's own ``except`` would swallow. ``request``
    must be set because ``Session.send``'s redirect peek-ahead goes
    through ``rebuild_auth``, which dereferences ``response.request.url``.

    When ``close_calls`` is given, ``.close`` is wrapped so every call is
    recorded -- the signal the closure assertion checks. See the counting
    rationale in the sibling Anthropic module: ``Session.send`` peeks one
    ``resolve_redirects`` step even under ``allow_redirects=False``, and
    that peek's bookkeeping closes a genuine redirect response exactly
    once by itself, so the refusal site's own ``response.close()`` call
    makes the observed count exactly 2.
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
    redirect_status: int,
) -> list[int]:
    """Patch ``HTTPAdapter.send`` so ``good.example`` 30x's to ``evil.example``.

    ``chat_with_kobold`` builds its own ``requests.Session`` (via
    ``Utils.egress.create_default_session``) and mounts retry adapters on
    it; patching the class method intercepts the dispatch regardless.

    Returns:
        A list that accumulates one entry per ``.close()`` call on the
        redirect response from ``good.example`` -- the caller asserts
        against this to confirm the refusal path released the connection.
    """
    redirect_close_calls: list[int] = []

    def _fake_send(self, request, **kwargs):  # noqa: ANN001 - requests API
        host = urlsplit(request.url).netloc
        seen[host] = {
            "method": request.method,
            "headers": {k.lower(): v for k, v in request.headers.items()},
        }
        if host == _GOOD_HOST:
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


def _call_kobold() -> object:
    return llm_local.chat_with_kobold(
        input_data=[{"role": "user", "content": "hi"}],
        api_key=_SENTINEL_KEY,
        api_base_url=f"https://{_GOOD_HOST}/api/v1/generate",
    )


@pytest.mark.parametrize("redirect_status", [302, 303, 307])
def test_kobold_refuses_redirect_and_never_leaks_x_api_key(
    monkeypatch: pytest.MonkeyPatch, redirect_status: int
) -> None:
    """A 302/303/303-answer from the endpoint must not be followed.

    302 and 303 are the load-bearing parameters: on those, ``requests``
    converts the POST to GET before re-issuing, so following the redirect
    silently turns a key-bearing generation POST into a key-bearing GET
    against an attacker-named host. A 307-only test would keep the method
    and could not demonstrate that conversion (task-19862 AC).
    """
    seen: dict[str, dict[str, object]] = {}
    redirect_close_calls = _install_redirecting_adapter(
        monkeypatch, seen, redirect_status=redirect_status
    )

    raised: Exception | None = None
    try:
        result = _call_kobold()
    except ChatProviderError as exc:
        raised = exc

    # Sanity: the first hop really did carry the credential.
    assert _GOOD_HOST in seen
    good_headers = seen[_GOOD_HOST]["headers"]
    assert good_headers.get("x-api-key") == _SENTINEL_KEY

    # Load-bearing, checked independently of the exception: the credential
    # -- and any request at all -- must never reach the second host. On
    # 302/303 a followed redirect re-issues as a GET (the method
    # conversion the fix exists to prevent), on 307 as the original POST;
    # both carry the key verbatim, which is why the refusal is absolute.
    assert _EVIL_HOST not in seen, (
        f"KoboldAI request was re-issued to the redirect target "
        f"{_EVIL_HOST!r} as {seen.get(_EVIL_HOST, {}).get('method')!r} "
        f"carrying headers {seen.get(_EVIL_HOST, {}).get('headers')!r}"
    )

    # If the redirect HAD been followed (mutation check / regression), the
    # attacker host's answer would silently become the model's generation.
    if raised is None:
        assert "stolen" not in str(result), (
            "the redirect was followed and the attacker host's answer "
            "became the model's generation"
        )

    # The refusal must be loud, not a silent follow.
    assert raised is not None, (
        "expected ChatProviderError on redirect refusal; the call "
        "returned without one"
    )
    assert "redirect" in str(raised).lower()

    # The refusal message must not echo the attacker-controlled Location
    # header (TASK-19321/19552/19557 exception-text rule).
    refusal_text = str(raised)
    assert _EVIL_HOST not in refusal_text and "/steal" not in refusal_text

    # The refused response must be explicitly closed by the refusal site,
    # in addition to requests' own redirect-peek close (== 2, not >= 1:
    # the library alone always contributes exactly one -- see the sibling
    # Anthropic module's verified rationale).
    assert len(redirect_close_calls) == 2, (
        f"expected the refusal site's own response.close() in addition "
        f"to requests' resolve_redirects() peek close; observed "
        f"{len(redirect_close_calls)} close() call(s)"
    )
