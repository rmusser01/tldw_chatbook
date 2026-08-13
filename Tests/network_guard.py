"""Process-wide network egress guard for the test suite (task-15111).

Why this exists
---------------
`Tests/UI`'s Console suites were making **real TCP connections to
``127.0.0.1:8080`` and ``127.0.0.1:11434``** on every test that mounted the
Console chat screen with an unconfigured provider. The mechanism:
``chat_screen._maybe_start_console_local_discovery`` starts a worker that calls
``Chat.local_server_discovery.discover_local_servers``, whose candidate list
*always* begins with the two well-known localhost defaults regardless of
config, and ``probe_models_endpoint`` builds a real ``httpx.AsyncClient`` when
none is injected. A developer with a llama.cpp/Ollama/audio.cpp server on
those ports therefore ran **different behaviour** from CI (the probe answers,
a "detected server" affordance appears on the setup card) and nothing told
either of them. Measured on this machine with an ``audiocpp`` server on 8080:
one record-only run logged hundreds of such connects across ``Tests/UI``.

Design
------
* The patch is installed at ``Tests/conftest.py`` **import** time and defaults
  to BLOCKED, so collection, module import, fixture setup/teardown and any
  worker thread that outlives a test are all covered — not just the window a
  function-scoped fixture happens to span.
* Only ``AF_INET``/``AF_INET6`` egress is blocked. ``AF_UNIX`` is local IPC
  (system libraries use it) and is not what this guard is about.
* The raised error subclasses ``OSError`` so that client libraries treat it as
  an ordinary connection failure and degrade down their already-tested
  "endpoint unreachable" path instead of exploding somewhere unrelated.
* Because that degradation means a broad ``except Exception`` can *swallow*
  the block (``local_server_discovery._get_models_payload`` does exactly
  that), every blocked attempt is also **recorded**. The autouse fixture in
  ``Tests/conftest.py`` fails the test at teardown on a non-empty record, so
  the guard cannot be silently absorbed by the code under test.

Opting in
---------
A test that genuinely needs a socket marks itself
``@pytest.mark.allow_network`` (registered in ``pyproject.toml``). The
``live`` marker — real paid external APIs, already gated behind
``--run-live`` — is treated as an implicit opt-in.
"""

from __future__ import annotations

import socket
import threading
from typing import Any

__all__ = [
    "BlockedNetworkAccess",
    "blocked_attempts",
    "drain_blocked_attempts",
    "install",
    "is_allowed",
    "set_allowed",
]


class BlockedNetworkAccess(OSError):
    """Raised when a test attempts network egress without opting in.

    Subclasses ``OSError`` deliberately: ``httpx``/``requests``/``urllib`` all
    translate an ``OSError`` from ``connect()`` into their own connection
    error, so a guarded test exercises the same code path it would take
    against a dead port rather than an unrelated crash.
    """


#: ``(call, address)`` pairs recorded since the last drain. Consulted by the
#: autouse fixture so a swallowed block still fails its test.
_blocked_attempts: list[tuple[str, str]] = []

#: Egress is denied unless a test explicitly opted in.
_allowed = False

_installed = False

_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex
_real_create_connection = socket.create_connection
_real_sendto = socket.socket.sendto
_real_socketpair = socket.socketpair

_socketpair_state = threading.local()

_INET_FAMILIES = frozenset({socket.AF_INET, socket.AF_INET6})


def is_allowed() -> bool:
    """Return whether network egress is currently permitted."""
    return _allowed


def set_allowed(allowed: bool) -> None:
    """Permit or deny network egress process-wide.

    Args:
        allowed: ``True`` to let connections through (opt-in tests only).
    """
    global _allowed
    _allowed = bool(allowed)


def blocked_attempts() -> tuple[tuple[str, str], ...]:
    """Return the blocked egress attempts recorded so far."""
    return tuple(_blocked_attempts)


def drain_blocked_attempts() -> tuple[tuple[str, str], ...]:
    """Return the recorded blocked attempts and clear the record."""
    recorded = tuple(_blocked_attempts)
    _blocked_attempts.clear()
    return recorded


def _describe(address: Any) -> str:
    """Return a short, stable description of a socket address."""
    if isinstance(address, tuple) and len(address) >= 2:
        return f"{address[0]}:{address[1]}"
    return repr(address)


def _deny(call: str, address: Any) -> BlockedNetworkAccess:
    """Record a blocked attempt and build the error to raise.

    Args:
        call: Name of the socket API that was intercepted.
        address: The address the caller tried to reach.

    Returns:
        The ``BlockedNetworkAccess`` the caller should raise.
    """
    described = _describe(address)
    _blocked_attempts.append((call, described))
    return BlockedNetworkAccess(
        f"network access blocked in tests: {call}({described}). "
        "Tests must not touch a live endpoint — stub the client seam, or mark "
        "the test @pytest.mark.allow_network if it genuinely needs a socket "
        "(see Tests/network_guard.py, task-15111)."
    )


def _should_block(family: Any) -> bool:
    """Return whether egress on this address family is guarded."""
    return not (_allowed or _inside_socketpair()) and family in _INET_FAMILIES


def _inside_socketpair() -> bool:
    """Return whether this thread is executing the real socketpair call."""
    return getattr(_socketpair_state, "depth", 0) > 0


def _guarded_socketpair(*args: Any, **kwargs: Any):  # noqa: ANN401
    """Call socketpair while allowing only its current-thread bootstrap connect."""
    prior_depth = getattr(_socketpair_state, "depth", 0)
    _socketpair_state.depth = prior_depth + 1
    try:
        return _real_socketpair(*args, **kwargs)
    finally:
        _socketpair_state.depth = prior_depth


def _guarded_connect(self: socket.socket, address: Any):  # noqa: ANN401
    if _should_block(self.family):
        raise _deny("socket.connect", address)
    return _real_connect(self, address)


def _guarded_connect_ex(self: socket.socket, address: Any):  # noqa: ANN401
    if _should_block(self.family):
        raise _deny("socket.connect_ex", address)
    return _real_connect_ex(self, address)


def _guarded_sendto(self: socket.socket, *args: Any, **kwargs: Any):  # noqa: ANN401
    if _should_block(self.family):
        # sendto() is the one egress path that never calls connect().
        raise _deny("socket.sendto", args[-1] if args else None)
    return _real_sendto(self, *args, **kwargs)


def _guarded_create_connection(address: Any, *args: Any, **kwargs: Any):  # noqa: ANN401
    if not _allowed and not _inside_socketpair():
        raise _deny("socket.create_connection", address)
    return _real_create_connection(address, *args, **kwargs)


def install() -> None:
    """Patch the socket egress points. Idempotent."""
    global _installed
    if _installed:
        return
    socket.socket.connect = _guarded_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = _guarded_connect_ex  # type: ignore[method-assign]
    socket.socket.sendto = _guarded_sendto  # type: ignore[method-assign]
    socket.create_connection = _guarded_create_connection  # type: ignore[assignment]
    socket.socketpair = _guarded_socketpair  # type: ignore[assignment]
    _installed = True
