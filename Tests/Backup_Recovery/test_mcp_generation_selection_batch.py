"""Finite MCP selection shares fresh ordinary inputs only before locking."""

import inspect
import json
import sys
import threading
import time
from collections import Counter
from dataclasses import replace

import pytest

from Tests.Backup_Recovery.test_raw_source_coordinator_io import local_root, permission  # noqa: F401
from tldw_chatbook.Backup_Recovery import bootstrap, generation_witnesses
from tldw_chatbook.Backup_Recovery import mcp_source_participants as mcp_sources
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Utils import windows_files

pytestmark = [
    pytest.mark.skipif(sys.platform != "win32", reason="Native Windows observation batch"),
    pytest.mark.usefixtures("local_root"),
]


@pytest.fixture
def ordinary_permission(permission):  # noqa: F811 - imported pytest fixture
    source = permission
    source.set_kill_switch(True)
    source.set_kill_switch(False)
    assert source in mcp_sources._BINDINGS
    assert source._recovery_original_path == source.path
    for _ in range(3):
        assert source.get_kill_switch() is False
    return source


def test_permission_selection_discards_evidence_at_source_lock(ordinary_permission, request):
    source = ordinary_permission
    codes = {
        bootstrap._parse_control_records.__code__: "control",
        bootstrap._parse_registry.__code__: "registry",
        generation_witnesses._witnesses.__code__: "witnesses",
        windows_files._Native.open_handle.__code__: "opens",
        windows_files._Native.security.__code__: "security",
    }
    scope_code = inspect.unwrap(raw.scope).__code__
    acquire_code = storage.acquire_storage.__code__
    counts = {"before_lock": Counter(), "after_lock": Counter()}
    acquired = []
    before = set(storage._live_leases)
    previous = sys.getprofile()

    def observe(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        if event == "return" and frame.f_code is acquire_code and type(result) is storage.StorageLease:
            acquired.append(result)
        if event != "call" or frame.f_code not in codes:
            return
        parent = frame.f_back
        while parent is not None and parent.f_code is not scope_code:
            if parent.f_code is acquire_code:
                return
            parent = parent.f_back
        if parent is not None and parent.f_locals.get("source") is source:
            active = raw._states.get(getattr(raw._local, "operation", None))
            already_owned = active is not None and active.source is source
            phase = "after_lock" if parent.f_locals.get("locked") or already_owned else "before_lock"
            counts[phase][codes[frame.f_code]] += 1

    sys.setprofile(observe)
    try:
        assert source.get_kill_switch() is False
    finally:
        sys.setprofile(previous)
    request.node.user_properties.append(("selection_native_counts", json.dumps(counts)))
    assert len(acquired) == 2
    assert set(storage._live_leases) == before
    assert all(lease not in storage._live_leases for lease in acquired)
    assert not any(item.source is source for item in raw._mcp_preparations.values())
    assert not raw._mcp_ordinary_selections
    assert not any(state.source is source for state in raw._states.values())
    # Original source/binding checks still request each generation selection.
    assert counts["before_lock"]["witnesses"] == 7, counts
    assert (counts["before_lock"]["control"], counts["before_lock"]["registry"]) == (2, 2), counts
    # Only the new operation entry proof executes in raw.scope after locking.
    # The command's protected-open and publication proofs run after it yields.
    assert counts["after_lock"]["witnesses"] == 0, counts
    assert counts["after_lock"]["control"] == counts["after_lock"]["registry"] == 1, counts


@pytest.mark.parametrize("change", ["pending", "association", "registry", "selector", "path", "binding", "lease", "reader"])
def test_changed_prelock_selection_refuses_before_payload(ordinary_permission, monkeypatch, change):
    source = ordinary_permission
    selected, binding = source.path, mcp_sources._BINDINGS[source]
    original_bytes = selected.read_bytes()
    before = set(storage._live_leases)
    changed = []
    cleanup = []
    previous = sys.getprofile()
    code = raw._prepared_mcp_witnesses.__code__

    def observe(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        if event != "return" or frame.f_code is not code or result is not True or changed:
            return
        proof = raw._mcp_ordinary_selections[raw._mcp_preparation_local.operation]
        changed.append(change)
        if change in {"pending", "association"}:
            if change == "pending":
                marker = proof.root / ("pending-" + bootstrap._key("mcp-change") + ".json")
                record = dict(version=1, operation_id="mcp-change", namespaces=list(proof.names),
                              control_root=str(proof.root), selectors=[str(proof.selector)])
            else:
                marker = proof.root / ("activation-" + bootstrap._key(str(proof.selector)) + ".json")
                record = dict(version=1, selector=str(proof.selector), activation=dict(
                    operation_id="mcp-change", generation="g1", owners=["mcp.permissions"],
                    namespaces=list(proof.names), store_root=str(proof.root / "review")))
            marker.write_text(json.dumps(record), encoding="utf-8")
            cleanup.append(marker)
        elif change == "registry":
            path = proof.root / "admission" / "registry.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            record["entries"]["extra"] = next(iter(record["entries"].values())).copy()
            path.write_text(json.dumps(record), encoding="utf-8")
        elif change == "selector":
            monkeypatch.setenv("TLDW_CONFIG_PATH", str(selected.with_name("different-config")))
        elif change == "path":
            source.path = selected.with_name("different-permissions.json")
        elif change == "binding":
            mcp_sources._BINDINGS[source] = replace(binding)
        elif change == "lease":
            proof.preparation.lease.close()
        else:
            monkeypatch.setattr(generation_witnesses, "_paired_witnesses", lambda *args: [])

    sys.setprofile(observe)
    try:
        with pytest.raises((bootstrap.RecoveryRequired, ValueError)):
            source.get_kill_switch()
    finally:
        sys.setprofile(previous)
        source.path = selected
        mcp_sources._BINDINGS[source] = binding
        for path in cleanup:
            path.unlink(missing_ok=True)
    assert changed == [change]
    assert selected.read_bytes() == original_bytes
    assert set(storage._live_leases) == before
    assert not raw._mcp_ordinary_selections
    assert not any(item.source is source for item in raw._mcp_preparations.values())
    assert not any(state.source is source for state in raw._states.values())


def test_substituted_generation_reader_keeps_original_route(ordinary_permission, monkeypatch):
    original = generation_witnesses._paired_witnesses
    calls = []

    def read(*args):
        assert not raw._mcp_ordinary_selections
        calls.append(args[0])
        return original(*args)

    monkeypatch.setattr(generation_witnesses, "_paired_witnesses", read)
    assert ordinary_permission.get_kill_switch() is False
    assert len(calls) >= 16


def test_ordinary_selection_is_discarded_before_source_lock_wait(ordinary_permission):
    source = ordinary_permission
    locked, selected = threading.Event(), threading.Event()
    failures, markers = [], []
    before = set(storage._live_leases)
    root = bootstrap.default_bootstrap_root()

    def holder():
        try:
            with source._mcp_source_lock:
                locked.set()
                assert selected.wait(10)
                # Keep the native lock through at least one timed acquisition.
                time.sleep(0.08)
                assert not raw._mcp_ordinary_selections
                marker = root / "unknown-control.json"
                marker.write_bytes(b"{}")
                markers.append(marker)
        except BaseException as error:
            failures.append(error)

    thread = threading.Thread(target=holder)
    thread.start()
    assert locked.wait(5)
    previous = sys.getprofile()
    code = raw._finish_mcp_ordinary_selection.__code__

    def observe(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        if event == "return" and frame.f_code is code:
            assert not raw._mcp_ordinary_selections
            selected.set()

    sys.setprofile(observe)
    try:
        with pytest.raises((bootstrap.RecoveryRequired, ValueError)):
            source.get_kill_switch()
    finally:
        sys.setprofile(previous)
        selected.set()
        thread.join(10)
        for marker in markers:
            marker.unlink(missing_ok=True)
    assert not thread.is_alive() and not failures
    assert markers and set(storage._live_leases) == before
    assert not raw._mcp_ordinary_selections


def test_failed_initial_ordinary_read_is_not_retried(ordinary_permission):
    source = ordinary_permission
    marker = bootstrap.default_bootstrap_root() / "unknown-control.json"
    acquired, denied = [], []
    before = set(storage._live_leases)
    previous = sys.getprofile()
    acquire_code = storage.acquire_storage.__code__
    control_code = bootstrap._parse_control_records.__code__

    def observe(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        if event == "return" and frame.f_code is acquire_code and not acquired:
            acquired.append(result)
            marker.write_bytes(b"{}")
        elif event == "return" and frame.f_code is control_code and marker.exists():
            assert frame.f_locals["ordinary_only"]
            denied.append(result)
            marker.unlink()

    sys.setprofile(observe)
    try:
        with pytest.raises(bootstrap.RecoveryRequired, match="recovery_scope_uncertain"):
            source.get_kill_switch()
    finally:
        sys.setprofile(previous)
        marker.unlink(missing_ok=True)
    assert denied == [None]
    assert len(acquired) == 1 and set(storage._live_leases) == before
    assert not raw._mcp_ordinary_selections
