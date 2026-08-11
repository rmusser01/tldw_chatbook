"""The test suite's network egress guard actually bites (task-15111).

Background: `Tests/UI`'s Console suites were opening real TCP connections to
`127.0.0.1:8080` / `127.0.0.1:11434` on every test that mounted the chat screen
with an unconfigured provider (see `Tests/network_guard.py` for the mechanism).
A guard that can be reintroduced silently is no guard, so these tests pin the
guard's own behaviour: egress is refused by default, the refusal is *recorded*
so a `except Exception` in the code under test cannot absorb it, and the
explicit opt-in works.
"""

from __future__ import annotations

import socket
import threading

import pytest

from Tests import network_guard


def _drain_expecting(*, at_least: int = 1) -> tuple[tuple[str, str], ...]:
    """Consume the blocked-attempt record and assert something was recorded.

    The autouse `_no_network_io` fixture fails any test that leaves a
    non-empty record, so a test that *deliberately* trips the guard must
    consume its own evidence.

    Args:
        at_least: Minimum number of blocked attempts expected.

    Returns:
        The drained ``(call, address)`` records.
    """
    recorded = network_guard.drain_blocked_attempts()
    assert len(recorded) >= at_least, (
        f"guard recorded {len(recorded)} blocked attempts, expected >= {at_least}"
    )
    return recorded


def test_guard_is_installed_and_denies_by_default() -> None:
    """The patch is in place from conftest import time, denied by default."""
    assert socket.socket.connect is not network_guard._real_connect
    assert socket.create_connection is not network_guard._real_create_connection
    assert network_guard.is_allowed() is False


def test_create_connection_to_the_llamacpp_default_port_is_refused() -> None:
    """`127.0.0.1:8080` is the exact address the Console discovery probe used.

    This is the address that made the suite environment-dependent: on a
    machine running llama.cpp/Ollama/audio.cpp it answered, and the Console
    setup card grew a "detected server" affordance CI never sees.
    """
    with pytest.raises(network_guard.BlockedNetworkAccess) as excinfo:
        socket.create_connection(("127.0.0.1", 8080), timeout=1)

    assert "127.0.0.1:8080" in str(excinfo.value)
    recorded = _drain_expecting()
    assert ("socket.create_connection", "127.0.0.1:8080") in recorded


def test_socket_connect_is_refused_for_loopback_and_remote_alike() -> None:
    """Both the app's loopback ports and ordinary remote hosts are blocked."""
    for address in (("127.0.0.1", 11434), ("93.184.216.34", 80)):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(network_guard.BlockedNetworkAccess):
                sock.connect(address)
        finally:
            sock.close()

    recorded = _drain_expecting(at_least=2)
    assert {address for _call, address in recorded} == {
        "127.0.0.1:11434",
        "93.184.216.34:80",
    }


def test_connect_ex_is_refused_too() -> None:
    """`connect_ex` returns an errno instead of raising, so it needs its own patch."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(network_guard.BlockedNetworkAccess):
            sock.connect_ex(("127.0.0.1", 8080))
    finally:
        sock.close()

    _drain_expecting()


def test_blocked_access_is_an_oserror_so_clients_degrade_normally() -> None:
    """httpx/requests/urllib translate an OSError into their own connect error.

    That is deliberate: a guarded probe should walk the same
    "endpoint unreachable" path it would take against a dead port, not blow up
    somewhere unrelated. It is also exactly why the record exists -- see
    `test_a_swallowed_block_still_fails_the_test`.
    """
    assert issubclass(network_guard.BlockedNetworkAccess, OSError)


def test_a_swallowed_block_still_fails_the_test() -> None:
    """A caller that eats the error cannot hide the escape.

    `local_server_discovery._get_models_payload` really does `except
    Exception: return None, ...`, so the raise alone is invisible. The record
    is what the autouse fixture asserts on.
    """
    try:
        socket.create_connection(("127.0.0.1", 8080), timeout=1)
    except Exception:  # noqa: BLE001 - reproducing the swallowing caller
        pass

    recorded = _drain_expecting()
    assert recorded[0][1] == "127.0.0.1:8080"


def test_unix_domain_sockets_are_not_blocked(tmp_path, monkeypatch) -> None:
    """AF_UNIX is local IPC used by system libraries, not network egress."""
    # Bound by a RELATIVE name from inside tmp_path: macOS caps sun_path at
    # ~104 bytes and pytest's absolute tmp_path already exceeds it.
    monkeypatch.chdir(tmp_path)
    sock_path = "guard.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(sock_path)
        server.listen(1)
        client.connect(sock_path)  # must not raise
    finally:
        client.close()
        server.close()

    assert network_guard.blocked_attempts() == ()


@pytest.mark.allow_network
def test_opted_in_test_may_open_a_real_socket() -> None:
    """`@pytest.mark.allow_network` lifts the denial for the marked test only.

    Connects to a listener this test owns, so the opt-in is proven without
    depending on anything outside the process.
    """
    assert network_guard.is_allowed() is True

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    accepted: list[socket.socket] = []

    def _accept() -> None:
        conn, _addr = server.accept()
        accepted.append(conn)

    thread = threading.Thread(target=_accept, daemon=True)
    thread.start()
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=5):
            thread.join(timeout=5)
    finally:
        for conn in accepted:
            conn.close()
        server.close()

    assert accepted, "opted-in test could not complete a real connection"
    assert network_guard.blocked_attempts() == ()


def test_denial_is_restored_after_an_opted_in_test() -> None:
    """The opt-in is per test; the next test is denied again.

    Ordering-dependent by construction: it only means anything when it runs
    after `test_opted_in_test_may_open_a_real_socket` in the same file.
    """
    assert network_guard.is_allowed() is False
    with pytest.raises(network_guard.BlockedNetworkAccess):
        socket.create_connection(("127.0.0.1", 8080), timeout=1)
    _drain_expecting()
