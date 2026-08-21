"""task-19557: ``x-api-key`` must never survive a cross-origin redirect.

Both Anthropic call sites (``chat_with_anthropic`` in ``LLM_API_Calls.py``
and ``summarize_with_anthropic`` in ``Summarization_General_Lib.py``) send
the API key in the custom ``x-api-key`` header over ``requests``.
``requests`` strips only ``Authorization``/``Cookie`` on a cross-host
redirect -- it has no notion of ``x-api-key`` -- so a redirecting or
compromised endpoint could otherwise capture the real key verbatim.

Fixed by passing ``allow_redirects=False`` on every ``session.post``/
``requests.post`` call that carries the header and explicitly refusing any
3xx response before the body is processed, mirroring the already-shipped
``x-goog-api-key`` fix for Google (``LLM_API_Calls.py``
``chat_with_google``, task-686/lines ~3528-3549).

Born-red: reverting either ``allow_redirects=False`` guard (and the
explicit 3xx check that follows it) makes the corresponding test below fail
by showing the sentinel key delivered to the cross-origin host.

Transport is faked at ``requests.adapters.HTTPAdapter.send`` -- the layer
just above the real socket -- so ``requests``' own redirect/session
machinery (``Session.resolve_redirects``, ``rebuild_auth``,
``rebuild_headers``) still runs for real; only the actual network I/O is
replaced. No test in this module makes a real network call.
"""

from __future__ import annotations

from urllib.parse import urlsplit

import pytest
import requests
import requests.adapters

import tldw_chatbook.LLM_Calls.LLM_API_Calls as llm_api_calls
import tldw_chatbook.LLM_Calls.Summarization_General_Lib as summarization_lib
from tldw_chatbook.Chat.Chat_Deps import ChatProviderError

# Synthetic sentinel -- never a real credential.
_SENTINEL_KEY = "sentinel-test-x-api-key-must-never-leak"


def _fake_response(
    status_code: int, headers: dict[str, str], body: bytes = b""
) -> requests.Response:
    resp = requests.Response()
    resp.status_code = status_code
    resp.headers = requests.structures.CaseInsensitiveDict(headers or {})
    resp._content = body
    resp.encoding = "utf-8"
    return resp


def _install_redirecting_adapter(
    monkeypatch: pytest.MonkeyPatch,
    seen_headers_by_host: dict[str, dict[str, str]],
    *,
    good_host: str,
    ok_body: bytes,
) -> None:
    """Patch ``HTTPAdapter.send`` so ``good_host`` 302s to ``evil.example``.

    Both fixed call sites build their own ``requests.Session`` (or, for
    ``summarize_with_anthropic``, let ``requests.post`` build one
    internally) but ultimately dispatch through a ``requests.adapters.
    HTTPAdapter`` instance either way -- a custom-mounted one or the
    library's own default. Patching the class method intercepts both
    without needing to know which path a given call takes.
    """

    def _fake_send(self, request, **kwargs):
        host = urlsplit(request.url).netloc
        seen_headers_by_host[host] = {
            k.lower(): v for k, v in request.headers.items()
        }
        if host == good_host:
            return _fake_response(
                302, {"Location": "https://evil.example/steal"}
            )
        if host == "evil.example":
            return _fake_response(200, {"Content-Type": "application/json"}, ok_body)
        raise AssertionError(f"unexpected host in test transport: {host}")

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", _fake_send)


def test_chat_with_anthropic_refuses_cross_origin_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_headers_by_host: dict[str, dict[str, str]] = {}
    _install_redirecting_adapter(
        monkeypatch,
        seen_headers_by_host,
        good_host="good.example",
        ok_body=b'{"content": [{"type": "text", "text": "stolen"}]}',
    )

    raised: Exception | None = None
    try:
        llm_api_calls.chat_with_anthropic(
            input_data=[{"role": "user", "content": "hi"}],
            model="claude-test",
            api_key=_SENTINEL_KEY,
            streaming=False,
            api_base_url="https://good.example",
        )
    except ChatProviderError as exc:
        raised = exc

    # Sanity: the first hop really did carry the credential.
    assert "good.example" in seen_headers_by_host
    assert seen_headers_by_host["good.example"].get("x-api-key") == _SENTINEL_KEY

    # Load-bearing, checked independently of whether the call raised: the
    # credential must never reach the cross-origin host.
    evil_headers = seen_headers_by_host.get("evil.example", {})
    assert "x-api-key" not in evil_headers, (
        f"x-api-key leaked to the cross-origin redirect target: {evil_headers}"
    )

    # And the redirect must actually be refused, not merely have its
    # credential incidentally dropped by some other mechanism.
    assert raised is not None, "expected ChatProviderError on redirect refusal"


def test_summarize_with_anthropic_refuses_cross_origin_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_headers_by_host: dict[str, dict[str, str]] = {}
    # summarize_with_anthropic's endpoint is a hardcoded literal
    # ("https://api.anthropic.com/v1/messages"), not configurable, so the
    # "good" host for this test is that literal host rather than a
    # caller-chosen one.
    _install_redirecting_adapter(
        monkeypatch,
        seen_headers_by_host,
        good_host="api.anthropic.com",
        ok_body=b'{"content": [{"type": "text", "text": "stolen"}]}',
    )

    result = summarization_lib.summarize_with_anthropic(
        api_key=_SENTINEL_KEY,
        input_data="Some text to summarize.",
        custom_prompt_arg="Summarize this.",
        streaming=False,
        max_retries=1,
    )

    # Sanity: the first hop really did carry the credential.
    assert "api.anthropic.com" in seen_headers_by_host
    assert (
        seen_headers_by_host["api.anthropic.com"].get("x-api-key") == _SENTINEL_KEY
    )

    # Load-bearing, checked independently of the returned message: the
    # credential must never reach the cross-origin host.
    evil_headers = seen_headers_by_host.get("evil.example", {})
    assert "x-api-key" not in evil_headers, (
        f"x-api-key leaked to the cross-origin redirect target: {evil_headers}"
    )

    # The function reports failure by returning a string rather than
    # raising (its established convention -- see the "API Key Not
    # Provided"/"Network error" returns elsewhere in the same function);
    # a successful summary would not mention a redirect.
    assert isinstance(result, str)
    assert "redirect" in result.lower()
