"""Each permission command reads freshly under one exact native operation."""

import json
import sys
from collections import Counter

import pytest

from Tests.Backup_Recovery.test_raw_source_coordinator_io import local_root, permission  # noqa: F401
from tldw_chatbook.Backup_Recovery import bootstrap, generation_witnesses
from tldw_chatbook.Backup_Recovery import mcp_source_participants as sources
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.MCP import permission_store as permissions
from tldw_chatbook.Utils import windows_files

pytestmark = [
    pytest.mark.skipif(sys.platform != "win32", reason="Native Windows permission commands"),
    pytest.mark.usefixtures("local_root"),
]


@pytest.mark.parametrize("command", ["load", "get_kill_switch"])
def test_fresh_command_has_one_forwarding_scope(permission, command, request):  # noqa: F811
    source = permission
    source.set_kill_switch(True)
    source.set_kill_switch(False)
    for _ in range(3):
        assert source.get_kill_switch() is False
    wrapper = permissions.MCPPermissionStore.load.__code__
    body = permissions.MCPPermissionStore._load_locked.__wrapped__.__code__
    codes = {
        bootstrap._control_records.__code__: "control",
        bootstrap._registry.__code__: "registry",
        generation_witnesses._witnesses.__code__: "witnesses",
        windows_files._Native.open_handle.__code__: "opens",
        windows_files._Native.security.__code__: "security",
        permissions._PermissionPathFence.observe.__code__: "publication",
        body: "payload",
    }
    calls, counts = [], Counter()
    before = set(storage._live_leases)
    previous = sys.getprofile()

    def observe(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        if event != "call":
            return
        if frame.f_code is wrapper and frame.f_locals.get("source") is source:
            calls.append(frame.f_locals["function"].__name__)
        if frame.f_code in codes:
            counts[codes[frame.f_code]] += 1

    sys.setprofile(observe)
    try:
        result = getattr(source, command)()
    finally:
        sys.setprofile(previous)
    assert (result is False if command == "get_kill_switch" else result["kill_switch"] is False)
    request.node.user_properties.append(("permission_command_native_counts", json.dumps(counts)))
    assert counts["payload"] == counts["publication"] == 1
    assert set(storage._live_leases) == before
    assert not any(state.source is source for state in raw._states.values())
    assert calls == [command], calls


def test_separate_commands_read_external_policy_changes(permission):  # noqa: F811
    source = permission
    source.set_kill_switch(True)
    source.set_kill_switch(False)
    assert source.get_kill_switch() is False
    original = source.path.read_bytes()
    payload = json.loads(original)
    payload["kill_switch"] = True
    source.path.write_text(json.dumps(payload), encoding="utf-8")
    assert source.get_kill_switch() is True
    source.path.write_bytes(original)
    assert source.load()["kill_switch"] is False


@pytest.mark.parametrize("command", ["load", "get_kill_switch"])
def test_corrupt_command_keeps_original_backup_and_publication(permission, command):  # noqa: F811
    source = permission
    source.path.write_bytes(b"not JSON")
    before = set(storage._live_leases)
    result = getattr(source, command)()
    assert (result is False if command == "get_kill_switch" else result["kill_switch"] is False)
    assert not source.path.exists()
    assert source.path.with_suffix(".json.bak").read_bytes() == b"not JSON"
    assert not source._permission_publication.pending
    assert source._mcp_persistence_error is None
    assert set(storage._live_leases) == before


def test_custom_load_retains_getter_callback(permission, monkeypatch):  # noqa: F811
    calls = []

    def custom():
        calls.append(True)
        return {"kill_switch": True}

    monkeypatch.setattr(permission, "load", custom)
    assert permission.get_kill_switch() is True
    assert calls == [True]


def test_custom_publication_observer_keeps_original_load(permission, monkeypatch):  # noqa: F811
    original = permission._permission_publication.observe
    calls = []

    def observe(payload):
        calls.append(payload["kill_switch"])
        original(payload)

    monkeypatch.setattr(permission._permission_publication, "observe", observe)
    assert permission.get_kill_switch() is False
    assert calls == [False]


@pytest.mark.parametrize("command", ["load", "get_kill_switch"])
def test_read_failure_does_not_reset_or_publish_policy(permission, command):  # noqa: F811
    source = permission
    source.set_kill_switch(True)
    original = source.path.read_bytes()
    body = permissions.MCPPermissionStore._load_locked.__wrapped__.__code__
    before = set(storage._live_leases)
    previous = sys.getprofile()
    failed = []

    def observe(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        if event == "c_call" and frame.f_code is body and getattr(result, "__name__", None) == "read":
            failed.append(True)
            raise OSError("test-owned permission read failure")

    sys.setprofile(observe)
    try:
        with pytest.raises(OSError, match="test-owned permission read failure"):
            getattr(source, command)()
    finally:
        sys.setprofile(previous)
    assert failed == [True]
    assert source.path.read_bytes() == original
    assert not source.path.with_suffix(".json.bak").exists()
    assert source.get_kill_switch() is True
    assert set(storage._live_leases) == before


@pytest.mark.parametrize("change", ["control", "path", "binding", "reader"])
def test_command_drift_after_payload_refuses_before_publication(permission, monkeypatch, change):  # noqa: F811
    source = permission
    source.load()
    selected, bound = source.path, sources._BINDINGS[source]
    body = permissions.MCPPermissionStore._load_locked.__wrapped__.__code__
    publication = permissions._PermissionPathFence.observe.__code__
    marker = bootstrap.default_bootstrap_root() / "unknown-control.json"
    changed, published = [], []
    before = set(storage._live_leases)
    previous = sys.getprofile()

    def observe(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        if event == "call" and frame.f_code is publication:
            published.append(True)
        if event != "return" or frame.f_code is not body or changed:
            return
        changed.append(True)
        if change == "control":
            marker.write_bytes(b"{}")
        elif change == "path":
            source.path = selected.with_name("foreign.json")
        elif change == "binding":
            sources._BINDINGS.pop(source)
        else:
            monkeypatch.setattr(source, "_load_locked", lambda: {"kill_switch": False})

    sys.setprofile(observe)
    try:
        with pytest.raises((bootstrap.RecoveryRequired, ValueError)):
            source.get_kill_switch()
    finally:
        sys.setprofile(previous)
        source.path = selected
        sources._BINDINGS[source] = bound
        marker.unlink(missing_ok=True)
    assert changed == [True] and not published
    assert set(storage._live_leases) == before
    assert not any(state.source is source for state in raw._states.values())
