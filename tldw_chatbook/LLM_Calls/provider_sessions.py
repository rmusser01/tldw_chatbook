"""Per-thread provider HTTP session registry (ADR-222).

``requests.Session`` is not thread-safe, and every provider call site used
to build a fresh session per call — paying TCP + TLS handshake per turn
with zero keep-alive reuse. This module caches one session per
``(key, thread)``: the factory (``create_default_session`` or a closure
over it) runs once per key per thread, and later calls on that thread
reuse the warmed connection pool.

Thread model: every LLM/summarization call in this app runs on the event
loop's **default ThreadPoolExecutor** — Textual 8.x thread workers are
dispatched via ``loop.run_in_executor(None, ...)`` (``textual/worker.py``),
the same executor ``asyncio.to_thread`` uses — so per-thread caching is
per-pooled-worker-thread caching: consecutive turns reuse the pool.

Lifecycle (ADR-222 §3): a registry session lives for its thread's
lifetime. There is deliberately no per-worker close hook — workers share
pooled threads, so closing at worker exit would destroy the reuse this
registry exists to create. ``close_all_for_current_thread`` exists for
test isolation and future explicit teardown; process exit bounds the rest.

See ``backlog/decisions/222-provider-http-session-reuse.md`` for the key
granularity rules (what config-derived values must appear in a key) and
the call sites that intentionally keep per-call sessions.
"""

from __future__ import annotations

__all__ = [
    "close_all_for_current_thread",
    "close_session",
    "default_timeout_fragment",
    "get_session",
    "trust_setting_fragment",
]

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import requests

_local = threading.local()


def trust_setting_fragment() -> str:
    """Stable key fragment for the current TLS trust setting (ADR-222 §2).

    Call sites that rely on their session's factory-baked ``verify``
    (``create_default_session()`` applies ``requests_verify()`` at
    construction) include this fragment in the registry key, so a trust
    policy change produces a new key — and a new session — instead of
    silently reusing one that still verifies with the old policy.

    The read is best-effort by design: ``requests_verify()`` funnels into
    the guarded config bootstrap, which cold or unreconciled test
    sandboxes refuse. In that state the fragment degrades to a constant,
    which is harmless — such sandboxes patch out the session factory, and
    in production the factory's own identical read (inside
    ``create_default_session``) surfaces any real bootstrap error exactly
    as the pre-ADR-222 per-call sessions did.
    """
    try:
        from tldw_chatbook.Utils.tls_trust import requests_verify

        return repr(requests_verify())
    except Exception:  # noqa: BLE001 - see docstring; never gates a call
        return "unresolved"


def default_timeout_fragment() -> str:
    """Stable key fragment for the config-driven default session timeout.

    Only needed by call sites that omit an explicit ``timeout=`` and rely
    on ``DefaultTimeoutSession``'s construction-time default; the fragment
    keeps a ``[web_security]`` timeout change producing a new key (and a
    new session) instead of reusing one with the old default baked in.
    Best-effort for the same reason as :func:`trust_setting_fragment`.
    """
    try:
        from tldw_chatbook.Utils.egress import default_session_timeout

        return repr(default_session_timeout())
    except Exception:  # noqa: BLE001 - see docstring; never gates a call
        return "unresolved"


def _sessions_for_current_thread() -> dict[str, requests.Session]:
    sessions = getattr(_local, "sessions", None)
    if sessions is None:
        sessions = {}
        _local.sessions = sessions
    return sessions


def _best_effort_close(session: requests.Session) -> None:
    try:
        session.close()
    except Exception:  # noqa: BLE001, S110 - teardown must never mask the happy path
        pass


def get_session(key: str, factory: Callable[[], requests.Session]) -> requests.Session:
    """Return this thread's session for ``key``, building it once.

    Args:
        key: Registry key. Call sites use ``(provider, base_url)`` plus any
            config-derived value the factory bakes into the session that the
            call relies on (TLS trust value, retry-adapter budget, default
            timeout) so a settings change produces a new key — and a new
            session — instead of silently reusing a stale one.
        factory: Builds the session on first use for this thread. Must be
            resolved through the calling module's ``create_default_session``
            global at call time so the existing test seam keeps working.

    Returns:
        The cached session for ``(key, this thread)``; never a session
        built on another thread.

    The cookie jar is cleared on every hit: before ADR-222 every call used
    a fresh session with an empty jar, so a provider ``Set-Cookie`` never
    reached the next call. Clearing preserves that behaviour while keeping
    within-call (retry-attempt) jar semantics unchanged.
    """
    sessions = _sessions_for_current_thread()
    session = sessions.get(key)
    if session is None:
        session = factory()
        sessions[key] = session
    else:
        # Test fakes may not subclass requests.Session; the clear is
        # best-effort on anything cookie-less.
        cookies = getattr(session, "cookies", None)
        if cookies is not None:
            cookies.clear()
    return session


def close_all_for_current_thread() -> None:
    """Close and drop every session registered for the calling thread.

    Sessions built by other threads are untouched — each thread owns its
    registry exclusively. Intended for test isolation and explicit
    teardown paths; normal callers never need this (see module docstring).
    """
    sessions = getattr(_local, "sessions", None)
    if not sessions:
        return
    sessions_before = list(sessions.values())
    sessions.clear()
    for session in sessions_before:
        _best_effort_close(session)


def close_session(key: str) -> None:
    """Close and drop one key's session for the calling thread, if present.

    Surgical cleanup for paths that know a specific endpoint's session is
    no longer valid (e.g. an auth or base-url change); absent keys are a
    no-op.
    """
    sessions = getattr(_local, "sessions", None)
    if not sessions:
        return
    session = sessions.pop(key, None)
    if session is not None:
        _best_effort_close(session)
