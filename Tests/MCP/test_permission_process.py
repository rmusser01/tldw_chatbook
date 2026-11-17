"""Causal controls for the disabled permission-process transport and mirror."""

import importlib.util
import io
import struct
import threading
from types import SimpleNamespace

import pytest


def implementation():
    assert (
        importlib.util.find_spec("tldw_chatbook.MCP.permission_process") is not None
    ), "permission command transport is not implemented"
    from tldw_chatbook.MCP import permission_process

    return permission_process


def test_large_payload_round_trip_is_chunked_without_truncation():
    impl = implementation()
    payload = {"body": "é" * 700000}
    stream = io.BytesIO()
    impl.write_message(stream, payload)
    assert len(stream.getvalue()) > 1000000
    stream.seek(0)
    assert impl.read_message(stream) == payload


@pytest.mark.parametrize("body", [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e999}'])
def test_ambiguous_json_is_refused(body):
    impl = implementation()
    with pytest.raises(impl.ProtocolError):
        impl.read_message(io.BytesIO(struct.pack("!I", len(body)) + body + b"\0\0\0\0"))


def test_changed_observation_invalidates_before_settlement_and_noop_retains_change():
    impl = implementation()
    old = object()
    fence = SimpleNamespace(token=old, observed=b"old", pending=False, uncertain=False)
    mirror = impl._PublicationMirror(fence)
    mirror.observe("61" * 32)
    assert fence.token is None
    assert not fence.pending
    mirror.settle({"digest": "61" * 32, "pending": False, "uncertain": False})
    assert fence.token is not None and fence.token is not old


def test_unchanged_observation_keeps_exact_token_and_mutation_discards_candidate():
    impl = implementation()
    old = object()
    fence = SimpleNamespace(
        token=old, observed=b"a" * 32, pending=False, uncertain=False
    )
    mirror = impl._PublicationMirror(fence)
    mirror.observe("61" * 32)
    assert fence.token is old
    mirror.observe("62" * 32)
    mirror.begin()
    assert fence.token is None and fence.pending
    mirror.settle({"digest": None, "pending": False, "uncertain": False})
    assert fence.token is None and fence.observed is None and not fence.pending


def test_reused_or_foreign_request_identity_cannot_advance_protocol():
    impl = implementation()
    identity = impl._Identity("a" * 32, "b" * 32)
    message = identity.frame("Check", {"site": "entry"})
    receiver = impl._Identity("a" * 32, "b" * 32)
    receiver.accept(message, {"Check"})
    with pytest.raises(impl.ProtocolError):
        receiver.accept(message, {"Check"})
    message["request"] = "c" * 32
    with pytest.raises(impl.ProtocolError):
        impl._Identity("a" * 32, "b" * 32).accept(message, {"Check"})


def test_nonraising_last_hold_failure_withholds_deferred_completion():
    impl = implementation()
    assert hasattr(impl, "_Command"), "command-local native retirement is missing"
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    key = (storage.os.getpid(), "permission-retirement-control")
    hold = SimpleNamespace(
        count=1,
        stop=threading.Event(),
        ready=threading.Event(),
        error=OSError("native close failed"),
    )
    hold.thread = threading.Thread(target=lambda: None)
    hold.thread.start()
    hold.thread.join()
    # A hashable hold with the real StorageLease.close non-raising path.
    hold = type("Hold", (), vars(hold))()
    with storage._lock:
        storage._holds[key] = hold
    command = impl._Command(None, None)
    try:
        with command:
            lease = storage.StorageLease(key)
            lease.close()
        assert lease not in storage._live_leases and lease._key is None
        assert hold in storage._retiring_holds
        with pytest.raises(impl.ProtocolError, match="retirement"):
            command.retirement()
    finally:
        storage._retiring_holds.discard(hold)
        storage._holds.pop(key, None)


def test_incomplete_launch_still_physically_retires_process_and_pipes():
    impl = implementation()
    assert hasattr(impl.PermissionProcess, "_retire_process")
    process = SimpleNamespace(stdin=io.BytesIO(), stdout=io.BytesIO(), poll=lambda: 0)
    broker = object.__new__(impl.PermissionProcess)
    broker.finished = threading.Event()
    broker.process, broker.tree, broker.monitor = process, None, None
    broker._retire_process()
    assert broker.retired and process.stdin.closed and process.stdout.closed


@pytest.mark.parametrize("payload", [{1: "coerced key"}, {"custom": (1, 2)}])
def test_non_json_argument_protocols_are_refused_before_serialization(payload):
    impl = implementation()
    with pytest.raises(impl.ProtocolError, match="unsupported_argument"):
        impl._command_arguments(
            "save", {"payload": payload, "expected_generation": None}
        )


def test_custom_config_mapping_is_ineligible_before_native_selection(monkeypatch):
    impl = implementation()
    from tldw_chatbook.Backup_Recovery import mcp_source_participants as sources

    class Custom(dict):
        pass

    source = object()
    config = SimpleNamespace(_CONFIG_CACHE={"paths": Custom(data_dir="selected")})
    monkeypatch.setattr(sources, "_BINDINGS", {source: SimpleNamespace(config=config)})
    with pytest.raises(impl.ProtocolError, match="config_ineligible"):
        impl._config_selection(source)
