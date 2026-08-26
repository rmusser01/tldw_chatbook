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

Qodo round 2 finding: ``summarize_with_anthropic``'s refusal branch
returned an error string on a 3xx WITHOUT calling ``response.close()`` --
a leaked ``requests`` connection on every refused redirect (the redirect
this task's own fix makes reachable). Fixed by closing before returning,
same as the other two refusal sites (``chat_with_anthropic``,
``chat_with_google``).

Both tests below assert an explicit-close count of exactly 2, not merely
that SOME close happened: ``requests.Session.send()`` calls
``resolve_redirects(..., yield_requests=True)`` to populate
``Response._next`` even under ``allow_redirects=False``, and that
generator's own bookkeeping closes any response with a redirect
``Location`` regardless of what the calling code does -- so a plain
``>= 1``/``is_closed`` check would have passed even with
``summarize_with_anthropic``'s ``response.close()`` call missing (this
was verified directly, not assumed, before the assertion was written this
way -- see ``_fake_response``'s docstring).

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
    status_code: int,
    headers: dict[str, str],
    body: bytes = b"",
    *,
    close_calls: list[int] | None = None,
    request: requests.PreparedRequest | None = None,
) -> requests.Response:
    """Build a bare ``requests.Response`` double that ``close()`` safely.

    ``_content_consumed = True`` matters, not just cosmetically: a bare
    ``requests.Response()`` defaults ``raw = None`` and
    ``_content_consumed = False``, so an unpatched ``.close()`` call hits
    ``self.raw.close()`` -> ``AttributeError`` on ``None``. That crash was
    getting silently absorbed by the production code's own outer
    ``except Exception`` and re-wrapped as the SAME ``ChatProviderError``
    the earlier version of this test asserted on -- so the test passed
    whether or not the refusal path's ``close()`` call actually worked,
    catching neither the missing call (Summarization) nor a broken one.
    Marking the body already-consumed (true to how this double is built --
    directly assigning ``_content``, not via a real stream) makes
    ``close()`` a safe, meaningful no-op instead.

    When ``close_calls`` is given, ``.close`` is wrapped so every call is
    recorded there -- the actual signal these tests check, since "an
    exception of the right type came back" does not prove the connection
    was released.

    Verified (not assumed): ``requests.Session.send()`` calls
    ``next(self.resolve_redirects(r, request, yield_requests=True, ...))``
    even when ``allow_redirects=False`` -- to populate ``Response._next``
    for a caller who wants ``response.next()`` -- and that generator's
    first iteration ALREADY calls ``resp.close()`` on any response that
    has a redirect ``Location``, regardless of ``allow_redirects``. So a
    genuine redirect response gets exactly one ``close()`` call from
    ``requests`` itself no matter what the calling code does -- an
    ``is``-it-closed check, or an ``>= 1`` count, cannot tell "the
    refusal site's own explicit close ran" from "requests closed it
    anyway". With the fix, the count is 2 (library + the refusal site's
    own call); without it, 1.
    """
    resp = requests.Response()
    resp.status_code = status_code
    resp.headers = requests.structures.CaseInsensitiveDict(headers or {})
    resp._content = body
    resp._content_consumed = True
    resp.encoding = "utf-8"
    # A real adapter sets this in `HTTPAdapter.build_response`; a hand-rolled
    # `send` double has to do it explicitly. Not cosmetic: `Session.send`
    # peeks ahead through `resolve_redirects` -> `rebuild_auth`, which
    # dereferences `response.request.url` to decide whether to strip
    # `Authorization` across a host change. With `.request` left as None that
    # raises `AttributeError` INSIDE requests, before the production refusal
    # can be observed -- and the production code's own `except Exception`
    # then swallows it and reports it as a generic failure, so the test
    # appears to exercise the redirect path while actually never reaching it.
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
    seen_headers_by_host: dict[str, dict[str, str]],
    *,
    good_host: str,
    ok_body: bytes,
) -> list[int]:
    """Patch ``HTTPAdapter.send`` so ``good_host`` 302s to ``evil.example``.

    Both fixed call sites build their own ``requests.Session`` (or, for
    ``summarize_with_anthropic``, let ``requests.post`` build one
    internally) but ultimately dispatch through a ``requests.adapters.
    HTTPAdapter`` instance either way -- a custom-mounted one or the
    library's own default. Patching the class method intercepts both
    without needing to know which path a given call takes.

    Returns:
        A list that accumulates one entry per ``.close()`` call on the
        302 response from ``good_host`` -- the caller asserts against
        this to confirm the refusal path actually released the
        connection, not merely that it returned/raised.
    """
    redirect_close_calls: list[int] = []

    def _fake_send(self, request, **kwargs):
        host = urlsplit(request.url).netloc
        seen_headers_by_host[host] = {
            k.lower(): v for k, v in request.headers.items()
        }
        if host == good_host:
            return _fake_response(
                302,
                {"Location": "https://evil.example/steal"},
                close_calls=redirect_close_calls,
                request=request,
            )
        if host == "evil.example":
            return _fake_response(
                200,
                {"Content-Type": "application/json"},
                ok_body,
                request=request,
            )
        raise AssertionError(f"unexpected host in test transport: {host}")

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", _fake_send)
    return redirect_close_calls


def test_chat_with_anthropic_refuses_cross_origin_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``chat_with_anthropic`` must not forward ``x-api-key`` to a redirect target.

    At base (``allow_redirects`` unset, ``requests``' own default ``True``),
    ``requests`` follows the 302 and re-sends every header -- including
    ``x-api-key``, which neither ``requests`` nor httpx strips on a
    cross-host hop -- to ``evil.example`` verbatim. Also pins that the
    refusal path closes the redirect response rather than leaking the
    connection (verified correct here; see ``summarize_with_anthropic``'s
    sibling test for the refusal site that Qodo round 2 found did NOT).
    """
    seen_headers_by_host: dict[str, dict[str, str]] = {}
    redirect_close_calls = _install_redirecting_adapter(
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

    # Qodo round 2: the refusal path must EXPLICITLY release the
    # connection, not merely rely on requests' own incidental close.
    # requests.Session.send() always calls resolve_redirects() once (even
    # under allow_redirects=False, to populate Response._next), and that
    # generator's own bookkeeping closes a genuine redirect response
    # regardless of what chat_with_anthropic does -- so a redirect
    # response is closed exactly ONCE by the library alone. Two calls
    # means the refusal site's own `response.close()` also ran.
    assert len(redirect_close_calls) == 2, (
        f"expected the refusal site's own response.close() call in "
        f"addition to requests' own resolve_redirects() close; observed "
        f"{len(redirect_close_calls)} close() call(s)"
    )


def test_summarize_with_anthropic_refuses_cross_origin_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``summarize_with_anthropic`` must not forward ``x-api-key`` either.

    Same defect as ``chat_with_anthropic``, different failure convention:
    this function reports a redirect refusal by RETURNING an error string
    rather than raising, so the credential-leak assertion is checked
    independently of that return value.

    Also pins the Qodo round 2 finding: this refusal branch returned
    WITHOUT calling ``response.close()`` -- a real leaked ``requests``
    connection on every refused redirect, undetected by the original
    version of this test (which asserted only on headers and the returned
    string, never on whether the response was released).
    """
    seen_headers_by_host: dict[str, dict[str, str]] = {}
    # summarize_with_anthropic's endpoint is a hardcoded literal
    # ("https://api.anthropic.com/v1/messages"), not configurable, so the
    # "good" host for this test is that literal host rather than a
    # caller-chosen one.
    redirect_close_calls = _install_redirecting_adapter(
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

    # Qodo round 2 -- the load-bearing addition: the refusal path must
    # EXPLICITLY release the connection, not merely rely on requests' own
    # incidental close. Same mechanism as chat_with_anthropic's sibling
    # assertion above: requests' own resolve_redirects() peek-ahead
    # closes a genuine redirect response once regardless of what this
    # function does, so >= 1 would pass even with the response.close()
    # call removed (verified: it did, before this assertion was
    # tightened to == 2). Two calls means summarize_with_anthropic's own
    # close ran too.
    assert len(redirect_close_calls) == 2, (
        f"expected summarize_with_anthropic's own response.close() call "
        f"in addition to requests' own resolve_redirects() close; "
        f"observed {len(redirect_close_calls)} close() call(s) -- this "
        f"is the exact Qodo round 2 finding: the refusal branch returned "
        f"without closing the response"
    )
