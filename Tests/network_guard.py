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
* That record is process-global and drained at *every* teardown, so an attempt
  made by a thread outliving its test lands on an unrelated later test. Each
  attempt therefore also records the **thread** that made it, and
  ``describe_blocked_attempts`` says so in the failure message when it is not
  the main thread (task-21592).

Opting in
---------
A test that owns a loopback listener marks itself
``@pytest.mark.loopback_network``. That mode accepts only numeric IPv4
``127.0.0.0/8`` and IPv6 ``::1`` destinations, so even DNS resolution stays
outside the permitted boundary. Tests that genuinely need unrestricted
sockets use ``@pytest.mark.allow_network``. The ``live`` marker — real paid
external APIs, already gated behind ``--run-live`` — is an implicit
unrestricted opt-in.
"""

from __future__ import annotations

import ipaddress
import socket
import threading
from collections.abc import Sequence
from enum import Enum
from typing import Any

__all__ = [
    "BlockedNetworkAccess",
    "NetworkMode",
    "blocked_attempts",
    "describe_blocked_attempts",
    "drain_blocked_attempt_threads",
    "drain_blocked_attempts",
    "install",
    "is_allowed",
    "is_loopback_destination",
    "is_loopback_only",
    "resolve_mode",
    "set_allowed",
    "set_mode",
]


class BlockedNetworkAccess(OSError):
    """Raised when a test attempts network egress without opting in.

    Subclasses ``OSError`` deliberately: ``httpx``/``requests``/``urllib`` all
    translate an ``OSError`` from ``connect()`` into their own connection
    error, so a guarded test exercises the same code path it would take
    against a dead port rather than an unrelated crash.
    """


class NetworkMode(str, Enum):
    """Process-wide socket policy selected by the active pytest marker."""

    BLOCKED = "blocked"
    LOOPBACK_ONLY = "loopback-only"
    ALLOW_ALL = "allow-all"


#: ``(call, address)`` pairs recorded since the last drain. Consulted by the
#: autouse fixture so a swallowed block still fails its test.
_blocked_attempts: list[tuple[str, str]] = []

#: Names of the threads that made the recorded attempts, in the same order and
#: drained alongside them. Kept as a parallel list rather than a third tuple
#: element so the published ``(call, address)`` shape stays unpacking-compatible
#: for the tests that already consume it.
#:
#: TASK-21592: this record is process-global and the autouse fixture asserts it
#: empty at EVERY test's teardown, so an attempt made by a thread that outlives
#: the test that started it is charged to whichever unrelated test happens to
#: be finishing. That is exactly how ``huggingface_hub``'s five-retry backoff
#: produced 10-15 teardown errors on innocent, passing ``Tests/Library`` nodes
#: whose ids varied run to run. The thread name is the one piece of provenance
#: available at record time, and it is enough to tell "this test did it" from
#: "something older did it".
_blocked_attempt_threads: list[str] = []

#: Held across BOTH appends in ``_deny`` and across every read/drain of the two
#: lists. Each ``list.append`` is individually atomic under the GIL, but the
#: pair is not: two threads denying at once could interleave as
#: append(attempt A) / append(attempt B) / append(thread B) / append(thread A),
#: leaving the provenance record positionally swapped -- and provenance is the
#: whole point of the second list (TASK-21592). Worker-thread egress is the
#: case it was added for, so concurrent ``_deny`` is the expected operating
#: condition, not an exotic one. The drains take it too, so a teardown cannot
#: snapshot one list either side of a concurrent append.
_blocked_attempt_lock = threading.Lock()

#: Egress is denied unless a test explicitly selects a narrower or wider mode.
_mode = NetworkMode.BLOCKED

_installed = False

_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex
_real_create_connection = socket.create_connection
_real_sendto = socket.socket.sendto
_real_socketpair = socket.socketpair

_socketpair_state = threading.local()

_INET_FAMILIES = frozenset({socket.AF_INET, socket.AF_INET6})
_IPV4_LOOPBACK_NETWORK = ipaddress.ip_network("127.0.0.0/8")
_IPV6_LOOPBACK_ADDRESS = ipaddress.IPv6Address("::1")


def is_allowed() -> bool:
    """Return whether unrestricted network egress is currently permitted."""
    return _mode is NetworkMode.ALLOW_ALL


def is_loopback_only() -> bool:
    """Return whether only numeric loopback destinations are permitted."""
    return _mode is NetworkMode.LOOPBACK_ONLY


def set_mode(mode: NetworkMode) -> None:
    """Set the process-wide network policy."""
    global _mode
    if not isinstance(mode, NetworkMode):
        raise TypeError("mode must be a NetworkMode")
    _mode = mode


def resolve_mode(*, allow_all: bool, loopback_only: bool) -> NetworkMode:
    """Resolve marker flags to one unambiguous network policy."""
    if allow_all and loopback_only:
        raise ValueError(
            "loopback_network conflicts with allow_network/live; select one "
            "network policy"
        )
    if allow_all:
        return NetworkMode.ALLOW_ALL
    if loopback_only:
        return NetworkMode.LOOPBACK_ONLY
    return NetworkMode.BLOCKED


def set_allowed(allowed: bool) -> None:
    """Permit or deny network egress process-wide.

    Args:
        allowed: ``True`` to let connections through (opt-in tests only).
    """
    set_mode(NetworkMode.ALLOW_ALL if allowed else NetworkMode.BLOCKED)


def blocked_attempts() -> tuple[tuple[str, str], ...]:
    """Return the blocked egress attempts recorded so far."""
    with _blocked_attempt_lock:
        return tuple(_blocked_attempts)


def drain_blocked_attempts() -> tuple[tuple[str, str], ...]:
    """Return the recorded blocked attempts and clear the record.

    Also clears the parallel thread-name record, so
    ``drain_blocked_attempt_threads`` must be called *before* this if both are
    wanted for the same batch.
    """
    with _blocked_attempt_lock:
        recorded = tuple(_blocked_attempts)
        _blocked_attempts.clear()
        _blocked_attempt_threads.clear()
    return recorded


def describe_blocked_attempts(
    attempts: Sequence[tuple[str, str]],
    threads: Sequence[str] = (),
) -> str:
    """Describe recorded attempts, naming any thread that is not the main one.

    A non-main thread means the attempt may have been made by a worker that
    outlived the test that started it, in which case the test being failed is a
    bystander (TASK-21592). Saying so in the message is the difference between
    a diagnosable failure and an unattributable flake.

    Args:
        attempts: The drained ``(call, address)`` records.
        threads: The thread names from
            :func:`drain_blocked_attempt_threads`, positionally aligned with
            ``attempts``. May be shorter or empty.

    Returns:
        A one-line description, empty when there are no attempts.
    """
    if not attempts:
        return ""
    padded = list(threads) + ["thread unknown"] * (len(attempts) - len(threads))
    detail = ", ".join(
        f"{call} -> {address} [{thread}]"
        for (call, address), thread in zip(attempts, padded)
    )
    main = threading.main_thread().name
    foreign = sorted({name for name in threads if name != main})
    if not foreign:
        return detail
    return (
        f"{detail} — NOTE: recorded on {', '.join(foreign)}, not {main}, so the "
        "attempt may belong to an EARLIER test whose worker outlived it; this "
        "record is process-global and is drained at every teardown (TASK-21592)"
    )


def drain_blocked_attempt_threads() -> tuple[str, ...]:
    """Return the thread names behind the recorded attempts, and clear them.

    Positionally aligned with :func:`blocked_attempts`. Read this first, then
    :func:`drain_blocked_attempts`.

    Returns:
        One thread name per currently recorded attempt.
    """
    with _blocked_attempt_lock:
        recorded = tuple(_blocked_attempt_threads)
        _blocked_attempt_threads.clear()
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
    with _blocked_attempt_lock:
        _blocked_attempts.append((call, described))
        _blocked_attempt_threads.append(threading.current_thread().name)
    return BlockedNetworkAccess(
        f"network access blocked in tests: {call}({described}). "
        "Tests must not touch a live endpoint — stub the client seam, use "
        "@pytest.mark.loopback_network for an owned numeric loopback listener, "
        "or use @pytest.mark.allow_network for unrestricted sockets "
        "(see Tests/network_guard.py, task-15111)."
    )


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


def is_loopback_destination(family: Any, address: Any) -> bool:
    """Classify a numeric destination without resolving a hostname."""
    if family not in _INET_FAMILIES:
        return False
    if not isinstance(address, tuple) or len(address) < 2:
        return False
    host = address[0]
    if not isinstance(host, str):
        return False
    try:
        parsed = ipaddress.ip_address(host)
    except ValueError:
        return False
    if family == socket.AF_INET:
        return (
            isinstance(parsed, ipaddress.IPv4Address)
            and parsed in _IPV4_LOOPBACK_NETWORK
        )
    return (
        isinstance(parsed, ipaddress.IPv6Address)
        and parsed == _IPV6_LOOPBACK_ADDRESS
    )


def _create_connection_is_loopback(address: Any) -> bool:
    return is_loopback_destination(
        socket.AF_INET, address
    ) or is_loopback_destination(socket.AF_INET6, address)


def _should_block(family: Any, address: Any) -> bool:
    """Return whether this address-family destination violates the policy."""
    if (
        family not in _INET_FAMILIES
        or _inside_socketpair()
        or _mode is NetworkMode.ALLOW_ALL
    ):
        return False
    if _mode is NetworkMode.LOOPBACK_ONLY:
        return not is_loopback_destination(family, address)
    return True


def _guarded_connect(self: socket.socket, address: Any):
    if _should_block(self.family, address):
        raise _deny("socket.connect", address)
    return _real_connect(self, address)


def _guarded_connect_ex(self: socket.socket, address: Any):
    if _should_block(self.family, address):
        raise _deny("socket.connect_ex", address)
    return _real_connect_ex(self, address)


def _guarded_sendto(self: socket.socket, *args: Any, **kwargs: Any):
    address = args[-1] if args else None
    if _should_block(self.family, address):
        # sendto() is the one egress path that never calls connect().
        raise _deny("socket.sendto", address)
    return _real_sendto(self, *args, **kwargs)


def _guarded_create_connection(address: Any, *args: Any, **kwargs: Any):
    if not _inside_socketpair() and (
        _mode is NetworkMode.BLOCKED
        or (
            _mode is NetworkMode.LOOPBACK_ONLY
            and not _create_connection_is_loopback(address)
        )
    ):
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
