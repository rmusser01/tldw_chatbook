"""Native evidence for a complete ordinary permission read."""

import io
import json
import sys
from collections import Counter
from contextlib import contextmanager

import pytest

from Tests.Chat import test_console_async_mcp_snapshot as fixtures
from tldw_chatbook.Backup_Recovery import activation, bootstrap
from tldw_chatbook.Backup_Recovery import mcp_source_participants as sources
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.MCP.console_snapshot import _CapturedSources
from tldw_chatbook.MCP.permission_store import _PermissionPathFence
from tldw_chatbook.Utils import windows_files

local_root = fixtures.local_root
catalog_store = fixtures.catalog_store
mcp_sources = fixtures.mcp_sources
snapshot_case = fixtures.snapshot_case

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Native Windows permission reads")


@contextmanager
def _observe(callback):
    previous = sys.getprofile()

    def profile(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        callback(frame, event, result)

    sys.setprofile(profile)
    try:
        yield
    finally:
        sys.setprofile(previous)


@pytest.mark.parametrize("command", ["load", "get_kill_switch", "console"])
def test_complete_read_consolidates_native_work(snapshot_case, command, request):
    source = snapshot_case.permissions
    source.set_kill_switch(True)
    source.set_kill_switch(False)
    console = _CapturedSources(snapshot_case.service)
    source.load()
    counts, events, admitted, retired = Counter(), [], [], []
    original_leases = set(storage._live_leases)
    codes = {
        bootstrap._parse_control_records.__code__: "control",
        bootstrap._parse_registry.__code__: "registry",
        windows_files._Native.open_handle.__code__: "opens",
        windows_files._Native.security.__code__: "security",
        raw._check_parent_pins.__code__: "parent",
    }

    def observe(frame, event, result):
        code = frame.f_code
        if event == "call":
            if code in codes:
                counts[codes[code]] += 1
            if code is _PermissionPathFence.observe.__code__:
                events.append("published")
        elif event == "return":
            if code is activation._ordinary_activation_observation.__code__ and result is not None:
                caller = frame.f_back
                if caller.f_code is raw._permission_read_boundary.__code__:
                    state = raw._states[caller.f_locals["operation"]]
                    events.append(state.permission_read_stage)
                else:
                    events.append("pre-lock")
            if code is storage.acquire_storage.__code__ and result is not None:
                admitted.append(result)
            if code is raw.close_descriptor.__code__:
                fd = frame.f_locals["fd"]
                if fd in retired:
                    retired.remove(fd)
                    events.append("payload retired")
        elif event == "c_call" and getattr(result, "__name__", None) == "read":
            handle = getattr(result, "__self__", None)
            if type(handle) is io.TextIOWrapper and any(
                state.source is source and any(handle is file for file in state.files)
                for state in raw._states.values()
            ):
                events.append("payload read")
                retired.append(handle.fileno())

    call = console.read_permission_payload if command == "console" else getattr(source, command)
    with _observe(observe):
        result = call()

    request.node.user_properties.append(("permission_read_native_counts", json.dumps(counts)))
    request.node.user_properties.append(("permission_read_phases", json.dumps(events)))
    assert (result if command == "get_kill_switch" else result["kill_switch"]) is False
    assert len(admitted) == 2
    assert all(lease not in storage._live_leases for lease in admitted)
    assert set(storage._live_leases) == original_leases
    assert not any(state.source is source for state in raw._states.values())
    assert events.count("payload read") == events.count("payload retired") == 1
    assert events.count("published") == (0 if command == "console" else 1)
    assert counts["control"] <= 5 and counts["registry"] <= 5, counts
    assert counts["parent"] <= 3 and counts["opens"] < 720 and counts["security"] < 370, counts
    assert events == ["pre-lock", "pre-lock", "setup", "metadata", "payload read", "payload retired", "final"] + (
        [] if command == "console" else ["published"]
    )


@pytest.mark.parametrize(
    "change",
    ["control", "path", "appearing_file", "selector", "pause", "replaced_file", "lease_reader"],
)
def test_metadata_changes_refuse_before_payload_open(snapshot_case, monkeypatch, change):
    source = snapshot_case.permissions
    source.load()
    selected = source.path
    previous_bytes = selected.read_bytes()
    if change == "appearing_file":
        selected.unlink()
    marker = bootstrap.default_bootstrap_root() / "unknown-control.json"
    changed, opened, published, pauses, substituted = [], [], [], [], []
    original_leases = set(storage._live_leases)

    def observe(frame, event, result):
        if event == "call":
            if frame.f_code is windows_files.WindowsOS.open.__code__:
                if frame.f_locals.get("path") == selected.name:
                    opened.append(True)
            elif frame.f_code is _PermissionPathFence.observe.__code__:
                published.append(True)
        elif event == "return" and frame.f_code is sources.preflight.__code__ and not changed:
            changed.append(True)
            if change == "control":
                marker.write_bytes(b"{}")
            elif change == "path":
                source.path = selected.with_name("foreign.json")
            elif change == "appearing_file":
                selected.write_bytes(previous_bytes)
            elif change == "selector":
                monkeypatch.setenv("TLDW_CONFIG_PATH", str(selected.with_suffix(".toml")))
            elif change == "pause":
                pauses.append(storage._begin_local_pause())
            elif change == "lease_reader":
                state = frame.f_locals["state"]
                lease = state.mcp_observation_lease
                original = lease.execution_scope

                def replacement():
                    substituted.append(True)
                    return original()

                monkeypatch.setattr(lease, "execution_scope", replacement)
            else:
                replacement = selected.with_suffix(".replacement")
                replacement.write_bytes(previous_bytes)
                replacement.replace(selected)

    try:
        with _observe(observe), pytest.raises((bootstrap.RecoveryRequired, ValueError)):
            source.load()
    finally:
        source.path = selected
        marker.unlink(missing_ok=True)
        for pause in pauses:
            pause.resume()
    assert changed == [True]
    assert not opened and not published and not substituted
    assert selected.read_bytes() == previous_bytes
    assert set(storage._live_leases) == original_leases
    assert not any(state.source is source for state in raw._states.values())


def test_live_hold_rechecked_after_final_source_predicate(snapshot_case):
    source = snapshot_case.permissions
    source.load()
    previous = set(storage._live_leases)
    paused, published = [], []
    final_source_checks = []

    def observe(frame, event, result):
        if event == "call" and frame.f_code is _PermissionPathFence.observe.__code__:
            published.append(True)
        if (
            event == "return"
            and frame.f_code is sources.canonical_path.__code__
            and frame.f_back.f_code is raw._permission_read_custody.__code__
            and not paused
        ):
            state = frame.f_back.f_locals["state"]
            if state.permission_read_stage == "final":
                final_source_checks.append(True)
                if len(final_source_checks) == 2:
                    paused.append(storage._begin_local_pause())

    try:
        with _observe(observe), pytest.raises(bootstrap.RecoveryRequired):
            source.load()
    finally:
        for pause in paused:
            pause.resume()
    assert len(paused) == 1 and not published
    assert set(storage._live_leases) == previous


def test_custom_raw_custody_guard_keeps_original_refusal(snapshot_case, monkeypatch):
    source = snapshot_case.permissions
    original = raw._source_custody_check
    calls = []

    @contextmanager
    def check(operation, path=None, *, writing=False):
        state = raw._states.get(operation)
        if state is not None and state.source is source and state.active:
            calls.append(True)
            raise bootstrap.RecoveryRequired("test_custom_permission_custody")
        with original(operation, path, writing=writing) as state:
            yield state

    monkeypatch.setattr(raw, "_source_custody_check", check)
    with pytest.raises(bootstrap.RecoveryRequired, match="test_custom_permission_custody"):
        source.load()
    assert calls == [True]


def test_missing_permission_read_proves_absence_at_return(snapshot_case):
    source = snapshot_case.permissions
    source.path.unlink()
    events = []

    def observe(frame, event, result):
        if (
            event == "return"
            and frame.f_code is activation._ordinary_activation_observation.__code__
            and result is not None
        ):
            events.append("proof")
        if event == "call" and frame.f_code is _PermissionPathFence.observe.__code__:
            events.append("publication")

    source.load()  # Warm the already-absent leaf.
    source.load()
    source.load()
    with _observe(observe):
        assert source.get_kill_switch() is False
    assert events == ["proof", "proof", "proof", "proof", "publication"]


def test_uncertain_payload_close_cannot_publish_or_retire_lease(snapshot_case, monkeypatch):
    source = snapshot_case.permissions
    source.load()
    original_close = raw.os.close
    saved_publication = source._permission_publication.token
    assert saved_publication is not None
    before = set(storage._live_leases)
    injected, payload_fd = [], []
    published = []

    def fail_payload_close(fd):
        if payload_fd and fd == payload_fd[0]:
            state = raw._states[raw._local.operation]
            assert fd in state.descriptors
            injected.append((state, fd))
            raise OSError("test owned descriptor close uncertainty")
        return original_close(fd)

    def observe(frame, event, result):
        if event == "call" and frame.f_code is _PermissionPathFence.observe.__code__:
            published.append(True)
        if event == "c_call" and getattr(result, "__name__", None) == "read":
            handle = getattr(result, "__self__", None)
            if type(handle) is io.TextIOWrapper and any(
                state.source is source and any(handle is file for file in state.files)
                for state in raw._states.values()
            ):
                payload_fd.append(handle.fileno())

    with monkeypatch.context() as patch:
        patch.setattr(raw.os, "close", fail_payload_close)
        with _observe(observe), pytest.raises(bootstrap.RecoveryRequired, match="raw_resources_not_retired"):
            source.load()
    try:
        assert len(injected) == 1 and not published
        state, fd = injected[0]
        assert state.uncertain and not state.active
        assert fd in state.descriptors and not state.files
        assert set(storage._live_leases) - before == set(state.leases)
        assert source._permission_publication.uncertain
    finally:
        # The injected close had no native side effect. Retire this exact test
        # descriptor and the retained scope so it cannot pollute another case.
        for operation, state in tuple(raw._states.items()):
            if state.source is source:
                for fd in tuple(state.descriptors):
                    original_close(fd)
                    state.descriptors.remove(fd)
                state.uncertain = False
                assert raw._retire(state)
                with storage._changed:
                    del raw._states[operation]
                    storage._raw_operations.discard(operation)
                    storage._changed.notify_all()
