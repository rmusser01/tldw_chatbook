"""Non-blocking Ollama probe (task-15473).

`_probe_local_server` (`UI/Screens/llm_screen.py`) used to be a blocking
`socket.create_connection(..., timeout=0.25)`, called directly from the
Models screen's periodic status timer (`LLMManagementWindow._update_
ollama_api_state`, driven by `set_interval(3.0, ...)` and also run once,
unconditionally, right after mount). ECONNREFUSED resolved instantly, but a
genuinely unresponsive ("blackholed") port -- the firewalled/container
setups the audit called out -- froze the WHOLE event loop for up to the
full 250ms per tick, since a synchronous socket call on the loop thread
blocks every other task in the process, not just this one.

This file pins two things against REAL sockets, no mocking of asyncio
itself:
  1. The async replacement (`asyncio.wait_for(asyncio.open_connection(...),
     timeout=0.25)`) preserves the exact up/refused/timeout semantics.
  2. The event loop stays responsive while the probe is in flight against a
     genuinely unresponsive address -- the actual bug this fixes.

`test_llm_screen_ollama_ux_unchanged.py` (companion file) covers AC#1's
other half: that the Models screen's button-gating UX is unchanged by the
async conversion, with the probe itself patched out (no network needed
there).
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time

import pytest

from tldw_chatbook.UI.Screens.llm_screen import _probe_local_server

# Every test in this file opens a real socket (or deliberately tries to
# reach an unresponsive one) -- opt out of the autouse network-egress guard
# for the whole module (Tests/conftest.py, task-15111).
pytestmark = pytest.mark.allow_network

#: A private, non-routed ("black hole") address: nothing answers the SYN and
#: nothing sends back an ICMP unreachable either, so a connect attempt hangs
#: for the full OS-level timeout instead of failing fast. Verified in this
#: sandbox (no route configured) to hang for the requested duration rather
#: than raise "network unreachable" immediately -- the same failure mode the
#: audit measured for a firewalled/container Ollama setup. The task's own
#: evidence note acknowledges a true blackhole is hard to simulate portably;
#: this address is the standard, widely-used stand-in for it and the tests
#: below tolerate an environment where it instead fails fast (see the
#: timing assertions).
_BLACKHOLE_HOST = "10.255.255.1"
_BLACKHOLE_PORT = 11434


def _real_listener() -> tuple[socket.socket, int]:
    """A real TCP listener on an ephemeral loopback port."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    return server, server.getsockname()[1]


def _closed_port() -> int:
    """A port nothing is listening on -- connecting refuses immediately."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    port = server.getsockname()[1]
    server.close()
    return port


@pytest.mark.asyncio
async def test_probe_reports_up_against_a_real_listener() -> None:
    """Up = connectable, same as the old blocking probe."""
    server, port = _real_listener()
    accepted: list[socket.socket] = []

    def _accept() -> None:
        try:
            conn, _addr = server.accept()
            accepted.append(conn)
        except OSError:
            pass

    thread = threading.Thread(target=_accept, daemon=True)
    thread.start()
    try:
        result = await _probe_local_server("127.0.0.1", port)
    finally:
        thread.join(timeout=2)
        for conn in accepted:
            conn.close()
        server.close()

    assert result is True


@pytest.mark.asyncio
async def test_probe_reports_down_on_a_refused_connection() -> None:
    """Down = refused, same as the old blocking probe (instant either way)."""
    port = _closed_port()
    result = await _probe_local_server("127.0.0.1", port)
    assert result is False


@pytest.mark.asyncio
async def test_probe_caps_at_the_configured_timeout_against_an_unresponsive_address() -> None:
    """Down = timeout, capped at the same 0.25s the blocking probe used.

    Best-effort against a real network condition (see `_BLACKHOLE_HOST`'s
    docstring): the behavioural assertion (`result is False`) holds whether
    the environment hangs or fails fast; the timing assertion only checks
    an upper bound generous enough to tolerate either.
    """
    start = time.monotonic()
    result = await _probe_local_server(_BLACKHOLE_HOST, _BLACKHOLE_PORT)
    elapsed = time.monotonic() - start

    assert result is False
    assert elapsed < 2.0, (
        f"probe should cap near 0.25s even against an unresponsive address, "
        f"took {elapsed:.3f}s"
    )


@pytest.mark.asyncio
async def test_probe_never_calls_the_blocking_socket_primitives(monkeypatch) -> None:
    """Structural guard: the async probe must never fall back to a
    synchronous connect -- that is the exact regression this task fixes.
    """

    def _forbidden(*args, **kwargs):
        raise AssertionError(
            "the async probe must never call the blocking "
            "socket.create_connection"
        )

    monkeypatch.setattr(socket, "create_connection", _forbidden)

    server, port = _real_listener()
    accepted: list[socket.socket] = []

    def _accept() -> None:
        try:
            conn, _addr = server.accept()
            accepted.append(conn)
        except OSError:
            pass

    thread = threading.Thread(target=_accept, daemon=True)
    thread.start()
    try:
        up = await _probe_local_server("127.0.0.1", port)
    finally:
        thread.join(timeout=2)
        for conn in accepted:
            conn.close()
        server.close()
    assert up is True

    down = await _probe_local_server("127.0.0.1", _closed_port())
    assert down is False


@pytest.mark.asyncio
async def test_event_loop_stays_responsive_during_an_unresponsive_probe() -> None:
    """The actual bug: a heartbeat task must keep ticking while the probe
    awaits a genuinely unresponsive connect.

    A synchronous equivalent run on the loop thread starves everything else
    for the whole wait -- measured directly (see the module docstring and
    the task's audit) at ~1 heartbeat tick in the same ~0.25s window versus
    dozens for the async version. This test only asserts the async side,
    since production no longer contains the blocking code path to compare
    against live.
    """
    heartbeats = 0
    stop = asyncio.Event()

    async def _heartbeat() -> None:
        nonlocal heartbeats
        while not stop.is_set():
            heartbeats += 1
            await asyncio.sleep(0.005)

    hb_task = asyncio.create_task(_heartbeat())
    try:
        result = await _probe_local_server(_BLACKHOLE_HOST, _BLACKHOLE_PORT)
    finally:
        stop.set()
        await hb_task

    assert result is False
    # A loop frozen by a blocking call would land at ~1 (the tick already
    # in flight when the block started); a healthy async wait comfortably
    # clears double digits over a ~0.25s window at a 5ms heartbeat period.
    assert heartbeats >= 10, (
        f"event loop looks starved during the probe: only {heartbeats} "
        "heartbeat ticks landed"
    )
