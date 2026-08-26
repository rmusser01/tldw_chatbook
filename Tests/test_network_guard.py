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
import sys
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
    except Exception:  # noqa: BLE001, S110 - reproduces the swallowing caller
        pass

    recorded = _drain_expecting()
    assert recorded[0][1] == "127.0.0.1:8080"


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="AF_UNIX unavailable")
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


@pytest.mark.skipif(sys.platform != "win32", reason="Windows socketpair fallback")
def test_guarded_socketpair_exchanges_data_without_weakening_family_denial() -> None:
    """Catch a socketpair bootstrap path that requires clearing guarded families."""
    protected_families = network_guard._INET_FAMILIES

    left, right = socket.socketpair()
    try:
        left.sendall(b"x")
        assert right.recv(1) == b"x"
    finally:
        left.close()
        right.close()

    assert network_guard._INET_FAMILIES is protected_families
    assert network_guard.blocked_attempts() == ()


def test_socketpair_exemption_does_not_escape_to_another_thread(monkeypatch) -> None:
    """Catch a process-global socketpair exemption that leaks concurrent egress."""
    entered = threading.Event()
    release = threading.Event()
    worker_errors: list[BaseException] = []
    real_socketpair = getattr(network_guard, "_real_socketpair", socket.socketpair)

    def held_socketpair(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        entered.set()
        assert release.wait(timeout=5), "test did not release held socketpair"
        return real_socketpair(*args, **kwargs)

    monkeypatch.setattr(network_guard, "_real_socketpair", held_socketpair, raising=False)

    def run_socketpair() -> None:
        try:
            left, right = socket.socketpair()
            left.close()
            right.close()
        except BaseException as exc:  # pragma: no cover - assertion aid
            worker_errors.append(exc)

    worker = threading.Thread(target=run_socketpair)
    worker.start()
    try:
        assert entered.wait(timeout=5), "guarded socketpair did not enter wrapper"
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(network_guard.BlockedNetworkAccess):
                client.connect(("127.0.0.1", 8080))
        finally:
            client.close()
        assert ("socket.connect", "127.0.0.1:8080") in _drain_expecting()
    finally:
        release.set()
        worker.join(timeout=5)

    assert not worker.is_alive()
    assert not worker_errors


def test_socketpair_exception_restores_network_denial(monkeypatch) -> None:
    """Catch an exception path that leaves the dynamic socketpair exemption set."""
    class SocketpairFailure(RuntimeError):
        pass

    def raising_socketpair(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise SocketpairFailure("socketpair failed")

    monkeypatch.setattr(
        network_guard, "_real_socketpair", raising_socketpair, raising=False
    )
    with pytest.raises(SocketpairFailure, match="socketpair failed"):
        socket.socketpair()

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(network_guard.BlockedNetworkAccess):
            client.connect(("127.0.0.1", 8080))
    finally:
        client.close()
    assert ("socket.connect", "127.0.0.1:8080") in _drain_expecting()


def test_nested_socketpair_restores_outer_dynamic_exemption(monkeypatch) -> None:
    """Catch nested socketpair cleanup that resets rather than restores depth."""
    inner_pair = (object(), object())
    outer_pair = (object(), object())
    calls = 0
    permitted_connects: list[tuple[str, int]] = []

    def fake_connect(_sock, address):  # noqa: ANN001
        permitted_connects.append(address)
        return None

    def recursive_socketpair(*args, **kwargs):  # noqa: ANN002, ANN003
        nonlocal calls
        calls += 1
        if calls == 2:
            return inner_pair

        assert socket.socketpair(*args, **kwargs) == inner_pair
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client.connect(("127.0.0.1", 8080))
        finally:
            client.close()
        return outer_pair

    monkeypatch.setattr(network_guard, "_real_connect", fake_connect)
    monkeypatch.setattr(network_guard, "_real_socketpair", recursive_socketpair)

    assert socket.socketpair() == outer_pair
    assert permitted_connects == [("127.0.0.1", 8080)]
    assert network_guard.blocked_attempts() == ()

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(network_guard.BlockedNetworkAccess):
            client.connect(("127.0.0.1", 8080))
    finally:
        client.close()
    assert ("socket.connect", "127.0.0.1:8080") in _drain_expecting()


def test_create_connection_is_exempt_only_inside_socketpair_scope(monkeypatch) -> None:
    """Catch create_connection omitting or leaking its socketpair exemption."""
    connection_sentinel = object()
    socketpair_sentinels = (object(), object())
    calls: list[tuple[tuple[str, int], tuple[object, ...], dict[str, object]]] = []

    def fake_create_connection(address, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        calls.append((address, args, kwargs))
        return connection_sentinel

    def fake_socketpair(*_args, **_kwargs):  # noqa: ANN002, ANN003
        result = socket.create_connection(("127.0.0.1", 8080), timeout=1)
        assert result is connection_sentinel
        assert network_guard.blocked_attempts() == ()
        return socketpair_sentinels

    monkeypatch.setattr(
        network_guard, "_real_create_connection", fake_create_connection
    )
    monkeypatch.setattr(network_guard, "_real_socketpair", fake_socketpair)

    assert socket.socketpair() == socketpair_sentinels
    assert calls == [(("127.0.0.1", 8080), (), {"timeout": 1})]
    assert network_guard.blocked_attempts() == ()

    with pytest.raises(network_guard.BlockedNetworkAccess):
        socket.create_connection(("127.0.0.1", 8080), timeout=1)
    assert ("socket.create_connection", "127.0.0.1:8080") in _drain_expecting()


def test_repeated_install_keeps_one_guarded_socketpair_wrapper() -> None:
    """Catch repeated installation stacking socketpair wrappers."""
    wrapper = socket.socketpair
    network_guard.install()

    assert socket.socketpair is wrapper
    assert socket.socketpair is network_guard._guarded_socketpair


def test_loopback_destination_classification_is_numeric_and_family_specific() -> None:
    assert network_guard.is_loopback_destination(
        socket.AF_INET, ("127.0.0.1", 80)
    )
    assert network_guard.is_loopback_destination(
        socket.AF_INET, ("127.255.255.254", 80)
    )
    assert not network_guard.is_loopback_destination(
        socket.AF_INET, ("126.255.255.255", 80)
    )
    assert not network_guard.is_loopback_destination(
        socket.AF_INET, ("localhost", 80)
    )
    assert network_guard.is_loopback_destination(socket.AF_INET6, ("::1", 80))
    assert not network_guard.is_loopback_destination(
        socket.AF_INET6, ("::ffff:127.0.0.1", 80)
    )
    assert not network_guard.is_loopback_destination(
        socket.AF_INET6, ("localhost", 80)
    )


def test_network_mode_rejects_conflicting_loopback_and_allow_all_markers() -> None:
    with pytest.raises(ValueError, match="loopback_network.*allow_network/live"):
        network_guard.resolve_mode(allow_all=True, loopback_only=True)


@pytest.mark.loopback_network
def test_loopback_only_test_connects_owned_listener_and_blocks_remote_ip() -> None:
    assert network_guard.is_allowed() is False
    assert network_guard.is_loopback_only() is True

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    accepted: socket.socket | None = None
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=5):
            accepted, _address = server.accept()
        with pytest.raises(network_guard.BlockedNetworkAccess):
            socket.create_connection(("192.0.2.1", 80), timeout=1)
    finally:
        if accepted is not None:
            accepted.close()
        server.close()

    recorded = _drain_expecting()
    assert recorded == (("socket.create_connection", "192.0.2.1:80"),)


@pytest.mark.loopback_network
def test_loopback_only_mode_covers_connect_connect_ex_and_sendto() -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(2)
    port = server.getsockname()[1]
    clients: list[socket.socket] = []
    accepted: list[socket.socket] = []
    udp_server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_server.bind(("127.0.0.1", 0))
    udp_client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        connect_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connect_client.connect(("127.0.0.1", port))
        clients.append(connect_client)

        connect_ex_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        assert connect_ex_client.connect_ex(("127.0.0.1", port)) == 0
        clients.append(connect_ex_client)

        for _ in range(2):
            connection, _address = server.accept()
            accepted.append(connection)

        assert udp_client.sendto(b"guard", udp_server.getsockname()) == 5
        payload, _address = udp_server.recvfrom(16)
        assert payload == b"guard"

        with (
            socket.socket(socket.AF_INET, socket.SOCK_STREAM) as blocked_connect,
            pytest.raises(network_guard.BlockedNetworkAccess),
        ):
            blocked_connect.connect(("192.0.2.1", 80))
        with (
            socket.socket(socket.AF_INET, socket.SOCK_STREAM) as blocked_connect_ex,
            pytest.raises(network_guard.BlockedNetworkAccess),
        ):
            blocked_connect_ex.connect_ex(("192.0.2.1", 80))
        with pytest.raises(network_guard.BlockedNetworkAccess):
            udp_client.sendto(b"guard", ("192.0.2.1", 80))
    finally:
        udp_client.close()
        udp_server.close()
        for connection in accepted:
            connection.close()
        for client in clients:
            client.close()
        server.close()

    recorded = _drain_expecting(at_least=3)
    assert {call for call, _address in recorded} == {
        "socket.connect",
        "socket.connect_ex",
        "socket.sendto",
    }


def test_denial_is_restored_after_a_loopback_only_test() -> None:
    assert network_guard.is_allowed() is False
    assert network_guard.is_loopback_only() is False
    with pytest.raises(network_guard.BlockedNetworkAccess):
        socket.create_connection(("127.0.0.1", 8080), timeout=1)
    _drain_expecting()


# --- provenance: which thread made the attempt (TASK-21592) -----------------
#
# The blocked-attempt record is process-global and the autouse `_no_network_io`
# fixture drains it at EVERY teardown. `huggingface_hub` retries a blocked
# request five times with backoff on a worker thread, so on dev `f49956038`
# running `Tests/App` with `Tests/Library` produced ten teardown errors on
# `Tests/Library` nodes that had themselves passed, with the ids varying run to
# run (10, then 3, 7, 3 on a reduced repro). Nothing in the message said the
# attempt came from somewhere else. These pin the provenance that says so.


def test_a_blocked_attempt_records_the_thread_that_made_it() -> None:
    with pytest.raises(network_guard.BlockedNetworkAccess):
        socket.create_connection(("192.0.2.1", 80), timeout=1)

    threads = network_guard.drain_blocked_attempt_threads()
    _drain_expecting()
    assert threads == (threading.main_thread().name,)


def test_an_attempt_from_a_worker_thread_names_that_worker() -> None:
    def _trip_the_guard() -> None:
        try:
            socket.create_connection(("192.0.2.1", 80), timeout=1)
        except network_guard.BlockedNetworkAccess:
            pass

    worker = threading.Thread(target=_trip_the_guard, name="outlives-its-test")
    worker.start()
    worker.join(timeout=10)
    assert not worker.is_alive()

    threads = network_guard.drain_blocked_attempt_threads()
    attempts = _drain_expecting()
    assert threads == ("outlives-its-test",)

    message = network_guard.describe_blocked_attempts(attempts, threads)
    assert "outlives-its-test" in message
    assert "EARLIER test" in message


def test_a_main_thread_attempt_is_not_blamed_on_an_earlier_test() -> None:
    """The provenance note must not fire for the ordinary same-test case."""
    message = network_guard.describe_blocked_attempts(
        (("socket.create_connection", "192.0.2.1:80"),),
        (threading.main_thread().name,),
    )
    assert "192.0.2.1:80" in message
    assert "EARLIER test" not in message


def test_draining_the_attempts_also_clears_their_thread_names() -> None:
    """The two records must never drift out of alignment across tests."""
    with pytest.raises(network_guard.BlockedNetworkAccess):
        socket.create_connection(("192.0.2.1", 80), timeout=1)

    _drain_expecting()
    assert network_guard.drain_blocked_attempt_threads() == ()
