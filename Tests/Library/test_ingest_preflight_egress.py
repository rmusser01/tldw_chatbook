"""TASK-19556 (a): the library ingest pre-flight is not a port-scan oracle.

The defect, confirmed live at this branch's base
(`tldw_chatbook/Library/ingest_preflight.py:15,221`):

* `_probe_url` issued a bare `urllib.request.urlopen` HEAD. `urlopen`
  auto-follows redirects and consults nothing -- there was no reference to
  `Utils/egress.py` anywhere in the module.
* `library_screen.handle_library_ingest_path_changed` arms an **0.8 s
  debounce timer on every keystroke**, so the probe fired while the user
  was still typing, before any deliberate "import this" action.
* `analyze_path` then returned three *distinguishable* outcomes for the
  probed host -- refused (an `error`), answered-{code} (a `warning`), or
  clean (a type group and `total_files == 1`). Pasting a link therefore
  drove an attacker-readable probe of the user's internal network: the
  difference between "10.0.0.5:8080 refused" and "10.0.0.5:8080 answered
  403" is visible in the summary the UI renders.

The fix has three parts, and each is pinned below:

1. **No probe by default.** The typing-debounced path performs no network
   request at all; the URL is classified by name, exactly as a local path
   is classified by `stat`. A user who wants link checking opts in with
   `[library] ingest_url_preflight_probe = true`.
2. **Policy-routed when enabled.** The probe consults `check_url_or_raise`
   with *no* trusted origins before any transport call, and follows no
   redirects, so it cannot be walked into internal space.
3. **One collapsed outcome.** Every URL the policy declines -- private,
   loopback, link-local, CGNAT, metadata, bad scheme, DNS failure --
   produces the identical advisory note, so the outcomes cannot be
   differenced.

`validate_url` also now runs *before* any of this, satisfying "no network
request at all before URL validation".
"""

from __future__ import annotations

from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import BaseHandler

import pytest
from loguru import logger

from Tests import network_guard
from tldw_chatbook.Library import ingest_preflight
from tldw_chatbook.Library.ingest_preflight import analyze_path
from tldw_chatbook.Library.ingest_types import PreflightResult

#: RFC1918 host + a port an attacker would want to probe. Numeric on
#: purpose: the network guard blocks/records `connect`, not name
#: resolution, so a numeric target proves a *connection* was attempted
#: rather than merely a DNS lookup.
INTERNAL_URL = "http://10.255.255.1:8080/report.pdf"

#: Three internal targets that an oracle would let a caller tell apart.
INTERNAL_TARGETS = (
    "http://10.255.255.1:8080/a.pdf",
    "http://10.255.255.2:9200/b.pdf",
    "http://192.168.77.9:22/c.pdf",
)


def _observable(result: PreflightResult) -> tuple[Any, ...]:
    """Everything about a pre-flight result an attacker could difference.

    Deliberately includes the *rendered* fields rather than a summary flag:
    the oracle in this defect was readable straight off the ingest summary
    (an error line vs. a "could not be confirmed" warning vs. a clean
    "1 file" echo), so equality has to cover all three.

    Args:
        result: The pre-flight result to reduce.

    Returns:
        A hashable tuple of the user-visible outcome.
    """
    return (
        tuple(result.errors),
        tuple(sorted(w.get("label", "") for w in result.warnings)),
        tuple(sorted(result.type_groups)),
        result.total_files,
        result.path_invalid,
    )


class _FakeOpen:
    """Stands in for `OpenerDirector.open` -- the seam BOTH the base's
    `urlopen` and any `build_opener(...)` variant funnel through.

    Patching here (rather than a module-level `urlopen` name) means the
    same test body exercises the code before and after the fix, so a red
    run is red for the defect and not for a missing test seam.
    """

    def __init__(self, responder: Callable[[str], Any]):
        self.responder = responder
        self.urls: list[str] = []

    def __call__(self, fullurl, data=None, timeout=None):  # noqa: ANN001
        # Bound as a plain class attribute, so no implicit `self` arrives:
        # `opener.open(...)` yields this instance, called with urlopen's own
        # positional `(url, data, timeout)`.
        url = getattr(fullurl, "full_url", fullurl)
        self.urls.append(url)
        return self.responder(url)


class _Answered:
    """A minimal context-manager HTTP response."""

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _responder_for(url: str):
    """Give each internal target a *different* real-world reaction.

    - `:8080` refuses the connection (closed port)
    - `:9200` answers 403 (an internal Elasticsearch behind auth)
    - `:22` answers cleanly (something is listening and happy)
    """
    if ":8080" in url:
        raise URLError(ConnectionRefusedError(61, "Connection refused"))
    if ":9200" in url:
        raise HTTPError(url, 403, "Forbidden", {}, None)
    return _Answered()


@pytest.fixture
def fake_transport(monkeypatch: pytest.MonkeyPatch) -> _FakeOpen:
    """Intercept every urllib open, with per-target reactions."""
    fake = _FakeOpen(_responder_for)
    monkeypatch.setattr("urllib.request.OpenerDirector.open", fake)
    return fake


@pytest.fixture
def probe_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opt this test into the (default-off) network probe."""
    monkeypatch.setattr(
        ingest_preflight, "url_probe_enabled", lambda: True, raising=False
    )


# ---------------------------------------------------------------------------
# 1. Reachability: the typing path must not touch the network at all
# ---------------------------------------------------------------------------


def test_typing_preflight_makes_no_connection_to_an_internal_address() -> None:
    """The defect's core evidence: at base this records a real `connect`.

    Nothing is stubbed here on purpose. The suite's process-wide guard
    (`Tests/network_guard.py`) turns an outbound `connect` into a recorded
    `BlockedNetworkAccess`; `_probe_url`'s `except Exception` swallows it,
    which is exactly why the *record* -- not an exception -- is the
    assertion. At base the record holds an attempt on 10.255.255.1:8080.
    """
    analyze_path(INTERNAL_URL)
    attempts = list(network_guard.drain_blocked_attempts())
    assert attempts == [], (
        "pre-flight attempted a connection to an internal address while the "
        f"user was typing: {attempts}"
    )


def test_typing_preflight_still_classifies_the_url_without_probing() -> None:
    """Dropping the probe must not drop the useful part of the summary."""
    result = analyze_path("https://example.com/lecture.mp4")
    network_guard.drain_blocked_attempts()
    assert result.errors == []
    assert result.source_is_url is True
    assert result.total_files == 1
    assert result.type_groups == {"audio_video": ["https://example.com/lecture.mp4"]}


# ---------------------------------------------------------------------------
# 2. Outcome vocabulary: internal targets must be indistinguishable
# ---------------------------------------------------------------------------


def test_typing_path_outcomes_are_identical_for_every_internal_target(
    fake_transport: _FakeOpen,
) -> None:
    """Refused / 403 / answered must reduce to ONE observable outcome."""
    outcomes = {_observable(analyze_path(url)) for url in INTERNAL_TARGETS}
    network_guard.drain_blocked_attempts()
    assert len(outcomes) == 1, (
        f"pre-flight distinguishes internal targets: {sorted(outcomes)}"
    )
    assert fake_transport.urls == [], (
        f"pre-flight reached the transport for internal targets: {fake_transport.urls}"
    )


def test_enabled_probe_still_collapses_every_internal_outcome(
    probe_enabled: None, fake_transport: _FakeOpen
) -> None:
    """Opting in must not reopen the oracle.

    With the probe enabled the egress policy declines all three targets
    *before* the transport, so the collapse is structural: there is no
    per-target reaction left to observe.
    """
    outcomes = {_observable(analyze_path(url)) for url in INTERNAL_TARGETS}
    network_guard.drain_blocked_attempts()
    assert len(outcomes) == 1, (
        f"enabled probe distinguishes internal targets: {sorted(outcomes)}"
    )
    assert fake_transport.urls == [], (
        "enabled probe reached the transport for a policy-declined target: "
        f"{fake_transport.urls}"
    )


def test_enabled_probe_declines_metadata_endpoint_like_any_other_internal_host(
    probe_enabled: None, fake_transport: _FakeOpen
) -> None:
    """A cloud metadata endpoint is not a distinguishable outcome either."""
    # Same path/extension on both, so the comparison isolates the PROBE
    # outcome from the name-based type classification (which is derived
    # from the URL the user typed and reveals nothing about the host).
    metadata = _observable(analyze_path("http://169.254.169.254/a.pdf"))
    private = _observable(analyze_path(INTERNAL_TARGETS[0]))
    network_guard.drain_blocked_attempts()
    assert metadata == private
    assert fake_transport.urls == []


# ---------------------------------------------------------------------------
# 3. Ordering: validation precedes any network request
# ---------------------------------------------------------------------------


def test_malformed_url_is_rejected_without_any_transport_call(
    probe_enabled: None, fake_transport: _FakeOpen
) -> None:
    """`validate_url` runs first, so a credential-bearing URL never flies.

    `http://user:pass@host/` is a URL `is_http_url` accepts and
    `validate_url` refuses; at base the probe fired first and would have
    put the embedded credential on the wire.
    """
    result = analyze_path("http://user:secret@example.com/doc.pdf")
    network_guard.drain_blocked_attempts()
    assert result.errors, "a URL that fails validate_url must be refused"
    assert result.path_invalid is True
    assert fake_transport.urls == []


# ---------------------------------------------------------------------------
# 4. The enabled probe is genuinely policy-routed (not merely silent)
# ---------------------------------------------------------------------------


def test_enabled_probe_reaches_the_transport_for_an_allowed_public_host(
    probe_enabled: None,
    fake_transport: _FakeOpen,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The opt-in probe still works: policy allows, transport is reached.

    Without this, "no transport call" above would pass trivially for a
    probe that had simply been deleted. The egress policy is stubbed to
    *allow* rather than left to resolve DNS for real.
    """
    monkeypatch.setattr(ingest_preflight, "check_url_or_raise", lambda *a, **k: None)
    result = analyze_path("https://example.com/document.pdf")
    network_guard.drain_blocked_attempts()
    assert fake_transport.urls == ["https://example.com/document.pdf"]
    assert result.errors == []
    assert "pdf" in result.type_groups


def test_enabled_probe_asks_the_policy_without_trusting_the_typed_host(
    probe_enabled: None,
    fake_transport: _FakeOpen,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An automatic probe never seeds trust from its own input URL.

    `Utils/egress.py`'s module contract: "Shared pipeline code must NEVER
    auto-trust its own input URL". Self-trusting here would make the check
    a no-op for precisely the private hosts this task is about.
    """
    seen: list[tuple[tuple, dict]] = []

    def _record(*args, **kwargs):
        seen.append((args, kwargs))

    monkeypatch.setattr(ingest_preflight, "check_url_or_raise", _record)
    analyze_path("http://10.255.255.1:8080/a.pdf")
    network_guard.drain_blocked_attempts()
    assert seen, "the probe did not consult the egress policy"
    trusted = seen[0][1].get("trusted_origins", frozenset())
    assert not trusted, f"probe self-trusted its own input: {trusted}"


# ---------------------------------------------------------------------------
# 5. The debounce wiring itself (TASK-19556 AC 3)
# ---------------------------------------------------------------------------


def test_the_typing_debounce_forbids_probing_even_when_it_is_enabled() -> None:
    """A keystroke pause is not a request to contact a host.

    The config gate alone would leave a user who opted the probe in being
    made to hit the host on every 0.8 s pause, which is the behaviour the
    task asks to be reviewed rather than merely defaulted away. The typing
    timer therefore passes `allow_probe=False` unconditionally; the
    deliberate triggers (blur, Enter, Browse..., the retry button) leave it
    at its default and let the config decide.

    Driven against the unbound method with a stand-in `self` so the
    assertion is about the wiring, not about mounting a 34k-line screen.
    """
    from types import SimpleNamespace

    from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_INGEST_MEDIA
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    calls: list[tuple[str, dict]] = []
    stand_in = SimpleNamespace(
        _ingest_state=SimpleNamespace(
            path_debounce_timer=object(),
            form=SimpleNamespace(path="  http://10.255.255.1:8080/x  "),
        ),
        _library_selected_row_id=LIBRARY_ROW_INGEST_MEDIA,
        _trigger_library_ingest_preflight=(
            lambda path, **kwargs: calls.append((path, kwargs))
        ),
    )

    LibraryScreen._run_debounced_library_ingest_preflight(stand_in)

    assert calls == [("http://10.255.255.1:8080/x", {"allow_probe": False})]


def test_the_deliberate_retry_trigger_does_not_forbid_probing() -> None:
    """The contrast: an explicit re-check is allowed to consult the config.

    Without this, `allow_probe=False` everywhere would satisfy the test
    above by simply deleting the feature.
    """
    from types import SimpleNamespace

    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    calls: list[tuple[str, dict]] = []
    stand_in = SimpleNamespace(
        _trigger_library_ingest_preflight=(
            lambda path, **kwargs: calls.append((path, kwargs))
        ),
    )

    LibraryScreen._trigger_preflight(stand_in, "http://example.com/a.pdf")

    assert calls == [("http://example.com/a.pdf", {})]


def test_the_worker_translates_allow_probe_into_the_analyze_path_keyword(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The middle link of the chain, which nothing else pins.

    (Independent review) `_run_debounced_...` -> `_trigger_...(allow_probe=
    False)` is pinned above, and `analyze_path(probe_url=False)`'s behaviour
    is pinned at the top of this module -- but the step that JOINS them,
    `_run_library_ingest_preflight`'s `probe_url=None if allow_probe else
    False`, was covered by neither. Rewriting it to a bare `probe_url=None`
    left `Tests/Library/`, `Tests/UI/test_library_ingest_*` and the census
    entirely green (509 passed, only the pre-existing red), while silently
    putting an opted-in user back on a probe per 0.8 s typing pause -- the
    exact behaviour AC 3 exists to prevent.

    Driven against the undecorated function (`@work(thread=True)`'s
    `__wrapped__`) with a stand-in `self`, matching the two tests above:
    the assertion is about the keyword translation, not about running a
    worker.
    """
    from types import SimpleNamespace

    from tldw_chatbook.UI.Screens import library_screen as library_screen_module
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    seen: list[dict] = []
    monkeypatch.setattr(
        library_screen_module,
        "analyze_path",
        lambda path, **kwargs: seen.append(kwargs) or PreflightResult(
            type_groups={}, warnings=[], errors=[], total_size=0,
            truncated=False, total_files=0,
        ),
    )
    run = LibraryScreen._run_library_ingest_preflight.__wrapped__
    stand_in = SimpleNamespace(
        _annotate_preflight_duplicates=lambda result: result,
        _apply_library_ingest_preflight_result=lambda *a: None,
        app=SimpleNamespace(call_from_thread=lambda fn, *a: fn(*a)),
    )

    run(stand_in, INTERNAL_URL, 1, False)
    assert seen[-1].get("probe_url") is False, (
        "the while-typing worker must forbid probing outright, not defer to "
        f"the config gate: {seen[-1]}"
    )

    run(stand_in, INTERNAL_URL, 2, True)
    assert seen[-1].get("probe_url") is None, (
        "a deliberate trigger must defer to the config gate, not force a "
        f"probe and not forbid one: {seen[-1]}"
    )


# ---------------------------------------------------------------------------
# 6. The probe follows no redirects (TASK-19556's second stated hardening)
# ---------------------------------------------------------------------------


class _FakeRedirectingHTTPHandler(BaseHandler):
    """A real `urllib` handler that answers 302 once, then 200.

    Deliberately NOT the `fake_transport` fixture above: that one patches
    `OpenerDirector.open`, which is the very method whose handler chain
    performs redirect following -- so it can never observe whether the
    chain follows one. This substitutes the *transport* handler instead
    and leaves `OpenerDirector.open` real.
    """

    opened: list[str] = []

    def http_open(self, req):  # noqa: ANN001
        import io
        from email.message import Message
        from urllib.response import addinfourl

        url = req.full_url
        type(self).opened.append(url)
        headers = Message()
        if url == PUBLIC_REDIRECTOR:
            headers["Location"] = INTERNAL_REDIRECT_TARGET
            response = addinfourl(io.BytesIO(b""), headers, url, 302)
            response.msg = "Found"
            return response
        response = addinfourl(io.BytesIO(b""), headers, url, 200)
        response.msg = "OK"
        return response


#: A public host the egress policy would allow, answering a 302 that points
#: at an internal one. The base's bare `urlopen` followed it, so the check
#: that had just passed on the public target was walked into private space.
PUBLIC_REDIRECTOR = "http://public.example.test/doc.pdf"
INTERNAL_REDIRECT_TARGET = "http://10.0.0.5:8080/"


def test_the_probe_does_not_follow_a_redirect_into_internal_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(Independent review) Nothing pinned this hardening.

    Reverting `_open_probe` to `build_opener()` -- i.e. dropping
    `_NoRedirectHandler`, restoring `urlopen`'s own redirect following --
    left all of `Tests/Library/` plus the census green (2252 passed, 0
    failed), because every other test in this module patches
    `OpenerDirector.open` and therefore never reaches the handler chain.

    A 302 must surface as an `HTTPError` about the host the user typed,
    and the `Location` target must never be opened.
    """
    import urllib.request
    from urllib.request import Request

    _FakeRedirectingHTTPHandler.opened = []
    monkeypatch.setattr(
        urllib.request, "HTTPHandler", _FakeRedirectingHTTPHandler, raising=True
    )

    with pytest.raises(HTTPError) as excinfo:
        ingest_preflight._open_probe(Request(PUBLIC_REDIRECTOR, method="HEAD"))

    assert excinfo.value.code == 302
    assert _FakeRedirectingHTTPHandler.opened == [PUBLIC_REDIRECTOR], (
        "the probe followed the redirect into internal space: "
        f"{_FakeRedirectingHTTPHandler.opened}"
    )
    network_guard.drain_blocked_attempts()


def test_the_probe_reports_a_redirect_as_an_answered_status_not_an_error(
    probe_enabled: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The user-visible half: a 3xx reads as "the site answered 302".

    That is an outcome about the public host the user typed and nothing
    else -- it must not become a clean result (which would imply the probe
    verified something it never fetched).
    """
    import urllib.request

    _FakeRedirectingHTTPHandler.opened = []
    monkeypatch.setattr(
        urllib.request, "HTTPHandler", _FakeRedirectingHTTPHandler, raising=True
    )
    monkeypatch.setattr(ingest_preflight, "check_url_or_raise", lambda *a, **k: None)

    result = analyze_path(PUBLIC_REDIRECTOR)
    network_guard.drain_blocked_attempts()

    assert result.errors == []
    assert [w.get("label") for w in result.warnings] == ["Could not check the link"]
    assert "302" in result.warnings[0]["hint"]
    assert _FakeRedirectingHTTPHandler.opened == [PUBLIC_REDIRECTOR]


# ---------------------------------------------------------------------------
# 7. Privacy: the raw URL (credentials, query-string secrets) must never
#    reach a log line -- including when policy EVALUATION itself blows up,
#    not just when it declines.
# ---------------------------------------------------------------------------

#: A URL carrying both an embedded credential and a query-string secret --
#: exactly what `validate_url` exists to keep out of `analyze_path`'s
#: network path, but `_probe_url` is reachable directly (and defends in
#: depth regardless of caller).
_CREDENTIAL_URL = (
    "http://sneaky_user:hunter2@10.255.255.1/report?api_key=SECRET_TOKEN_VALUE"
)


def test_policy_check_failure_does_not_log_the_credential_or_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The `except Exception` fallback around the policy check must not log
    more than the `EgressBlockedError` branch immediately above it does.

    That sibling branch is careful on purpose: `EgressBlockedError.__str__`
    renders `Utils/egress.py`'s `_log_origin(url)` -- a credential- and
    query-free `scheme://host[:port]` label -- never the raw URL. This
    exercises the *other* except clause: `check_url_or_raise` raising
    something other than `EgressBlockedError` (a bug in policy evaluation
    itself, not a decline), with a URL carrying an embedded credential and
    a query-string secret.

    Channel: this module's `logger` is `loguru`'s global singleton
    (`from loguru import logger`, no reassignment) -- confirmed by reading
    `ingest_preflight.py` directly, not assumed. A `logger.add(...)` sink
    captures the RENDERED message text, the same channel a real deployment
    writes to, so a leak here is a leak there.
    """

    def _boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("config lookup exploded")

    monkeypatch.setattr(ingest_preflight, "check_url_or_raise", _boom)

    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(message.record["message"]), level="DEBUG"
    )
    try:
        result = ingest_preflight._probe_url(_CREDENTIAL_URL)
    finally:
        logger.remove(sink_id)

    assert result.note == ingest_preflight._UNVERIFIABLE_NOTE

    rendered = "\n".join(messages)
    assert rendered, "expected the policy-check-failed line to be logged at all"
    for secret in ("sneaky_user", "hunter2", "SECRET_TOKEN_VALUE", "api_key"):
        assert secret not in rendered, f"credential/secret leaked into log: {rendered!r}"
    assert "/report" not in rendered, f"URL path leaked into log: {rendered!r}"
    assert _CREDENTIAL_URL not in rendered, f"raw URL leaked into log: {rendered!r}"
    # Still diagnostic: the exception type distinguishes "policy evaluation
    # itself failed" from an ordinary EgressBlockedError decline.
    assert "RuntimeError" in rendered
