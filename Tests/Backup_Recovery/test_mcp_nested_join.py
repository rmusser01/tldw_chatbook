"""A nested installed MCP join repeats proofs, not native ownership."""

import os
import sys
import threading
from contextlib import contextmanager

import pytest

from Tests.Backup_Recovery import test_permission_member_admission as controls
from tldw_chatbook.Backup_Recovery import bootstrap, generation_witnesses
from tldw_chatbook.Backup_Recovery import mcp_source_participants as mcp
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.MCP.local_store import LocalMCPStore
from tldw_chatbook.MCP.permission_store import MCPPermissionStore
from tldw_chatbook.MCP.unified_context_store import UnifiedMCPContextStore

configured_source = controls.configured_source
local_root = controls.local_root


@contextmanager
def _original_join_counts():
    """Count only original calls during the explicitly bracketed empty join."""
    originals = {
        "witness": generation_witnesses._witnesses,
        "parents": raw._check_parent_pins,
    }
    codes = {function.__code__: name for name, function in originals.items()}
    counts = dict.fromkeys(originals, 0)
    thread = threading.get_ident()
    tool = next(index for index in range(6) if sys.monitoring.get_tool(index) is None)
    sys.monitoring.use_tool_id(tool, "mcp-nested-original-proofs")

    def started(code, _offset):
        if threading.get_ident() == thread:
            counts[codes[code]] += 1

    try:
        sys.monitoring.register_callback(tool, sys.monitoring.events.PY_START, started)
        for code in codes:
            sys.monitoring.set_local_events(tool, code, sys.monitoring.events.PY_START)
        yield counts
    finally:
        for code in codes:
            sys.monitoring.set_local_events(tool, code, 0)
        sys.monitoring.register_callback(tool, sys.monitoring.events.PY_START, None)
        sys.monitoring.free_tool_id(tool)
        assert generation_witnesses._witnesses is originals["witness"]
        assert raw._check_parent_pins is originals["parents"]


def _seed(source):
    if type(source) is MCPPermissionStore:
        source.set_kill_switch(True)
    else:
        source.save(source.load())
    assert source.path.is_file()
    return source.load()


def _retire_seed_startup(local_root):
    # The fixture's config import owns a shared startup reference. Retire that
    # actual lease before measurement, as the existing construction-close test
    # does, so the measured raw scope is the final native owner.
    key = (os.getpid(), str(local_root))
    startup = storage._startups.pop(key, None)
    if startup is not None:
        startup.close()
    assert key not in storage._holds


def _retired(source, observed, previous, holds):
    assert getattr(raw._local, "operation", None) is previous
    assert not any(state.source is source for state in raw._states.values())
    assert observed.leases and all(
        lease not in storage._live_leases for lease in observed.leases
    )
    assert observed.pins
    for descriptor in observed.pins:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert holds and all(
        hold.stop.is_set() and not hold.thread.is_alive() and hold.error is None
        for hold in holds
    )


@pytest.mark.parametrize(
    "source_type,leaf",
    [
        (LocalMCPStore, "local_mcp_store.json"),
        (UnifiedMCPContextStore, "unified_mcp_context.json"),
        (MCPPermissionStore, "mcp_permissions.json"),
    ],
    ids=["local", "context", "permissions"],
)
def test_installed_nested_join_uses_one_fresh_proof(
    configured_source, local_root, source_type, leaf, record_property
):
    source = source_type(configured_source.get_user_data_dir() / leaf)
    expected = _seed(source)
    before = source.path.read_bytes()
    _retire_seed_startup(local_root)
    previous = getattr(raw._local, "operation", None)
    with controls._original_admissions(source) as observed:
        with raw.scope(source, mcp.ROUTE, writing=True) as outer:
            state = raw._states[outer]
            assert state.active and state.participant is not None
            assert state.source is source and state.selected == source.path
            assert state.leases and all(
                lease in storage._live_leases for lease in state.leases
            )
            holds = tuple(state.holds)
            assert all(hold is not None for hold in holds)
            pins = tuple(state.pins.values())
            assert pins and all(os.fstat(descriptor) for descriptor in pins)
            acquisitions_before = len(observed.requests)
            with _original_join_counts() as counts:
                with raw.scope(source, mcp.ROUTE, writing=True) as joined:
                    assert joined is outer
            assert len(observed.requests) == acquisitions_before
            assert raw._states[outer] is state and state.active
            # The real original load/file/final checks remain outside the count.
            assert source.load() == expected
            assert all(lease in storage._live_leases for lease in state.leases)
        assert source.path.read_bytes() == before
    _retired(source, observed, previous, holds)
    record_property("nested_witness_calls", counts["witness"])
    record_property("nested_parent_checks", counts["parents"])
    record_property("actual_acquisitions", len(observed.requests))
    # Last oracle: original code reaches every payload/custody/retirement proof.
    assert counts == {"witness": 1, "parents": 1}, counts


@pytest.mark.parametrize("change", ["source", "profile"])
def test_nested_join_rechecks_changed_source_before_body(
    configured_source, local_root, tmp_path, change
):
    source = MCPPermissionStore(
        configured_source.get_user_data_dir() / "mcp_permissions.json"
    )
    _seed(source)
    selected = source.path
    selector = configured_source._CONFIG_CACHE_SOURCE
    before = selected.read_bytes()
    foreign = tmp_path / "foreign-policy.json"
    foreign.write_bytes(b"foreign policy must remain untouched")
    _retire_seed_startup(local_root)
    previous = getattr(raw._local, "operation", None)
    with controls._original_admissions(source) as observed:
        with raw.scope(source, mcp.ROUTE, writing=True) as outer:
            holds = tuple(raw._states[outer].holds)
            try:
                if change == "source":
                    source.path = foreign
                else:
                    configured_source._CONFIG_CACHE_SOURCE = tmp_path / "foreign.toml"
                with pytest.raises(bootstrap.RecoveryRequired):
                    with raw.scope(source, mcp.ROUTE, writing=True):
                        pytest.fail("changed source entered nested body")
            finally:
                source.path = selected
                configured_source._CONFIG_CACHE_SOURCE = selector
    _retired(source, observed, previous, holds)
    assert selected.read_bytes() == before
    assert foreign.read_bytes() == b"foreign policy must remain untouched"


def test_nested_join_cannot_upgrade_read_only_operation(configured_source, local_root):
    source = MCPPermissionStore(
        configured_source.get_user_data_dir() / "mcp_permissions.json"
    )
    _seed(source)
    before = source.path.read_bytes()
    _retire_seed_startup(local_root)
    previous = getattr(raw._local, "operation", None)
    with controls._original_admissions(source) as observed:
        with raw.scope(source, mcp.ROUTE, writing=False) as outer:
            holds = tuple(raw._states[outer].holds)
            with pytest.raises(
                bootstrap.RecoveryRequired, match="raw_path_outside_scope"
            ):
                with raw.scope(source, mcp.ROUTE, writing=True):
                    pytest.fail("read-only custody became writable")
            with raw.scope(source, mcp.ROUTE, writing=False) as joined:
                assert joined is outer
    _retired(source, observed, previous, holds)
    assert source.path.read_bytes() == before


@pytest.mark.parametrize("matches", [True, False], ids=["same", "foreign"])
def test_explicit_read_selection_keeps_original_join_fence(
    configured_source, local_root, matches
):
    source = MCPPermissionStore(
        configured_source.get_user_data_dir() / "mcp_permissions.json"
    )
    expected = _seed(source)
    selected = source.path if matches else source.path.with_name("foreign-policy.json")
    _retire_seed_startup(local_root)
    previous = getattr(raw._local, "operation", None)
    with controls._original_admissions(source) as observed:
        with raw.scope(source, mcp.ROUTE, writing=True) as outer:
            holds = tuple(raw._states[outer].holds)
            with _original_join_counts() as counts:
                if matches:
                    with raw.scope(source, mcp.ROUTE, selected_read=selected) as joined:
                        assert joined is outer
                else:
                    with pytest.raises(
                        bootstrap.RecoveryRequired, match="raw_source_selection_changed"
                    ):
                        with raw.scope(source, mcp.ROUTE, selected_read=selected):
                            pytest.fail("foreign selected read entered")
            assert source.load() == expected
    _retired(source, observed, previous, holds)
    if matches:
        assert counts == {"witness": 3, "parents": 2}, counts
    else:
        assert not selected.exists()


def test_recovered_nested_join_keeps_canonical_bytes_passive(
    configured_source, local_root, tmp_path
):
    canonical = configured_source.get_user_data_dir() / "mcp_permissions.json"
    canonical.write_bytes(b"unreviewed canonical bytes must remain untouched")
    controls._publish_recovery_generation(configured_source, tmp_path)
    source = MCPPermissionStore(canonical)
    selected = source.path
    assert selected != canonical and not selected.parent.exists()
    expected = source.load()
    _retire_seed_startup(local_root)
    previous = getattr(raw._local, "operation", None)
    with controls._original_admissions(source) as observed:
        with raw.scope(source, mcp.ROUTE, writing=True) as outer:
            state = raw._states[outer]
            holds = tuple(state.holds)
            assert state.selected == selected and state.mcp_canonical == canonical
            with _original_join_counts() as counts:
                with raw.scope(source, mcp.ROUTE, writing=True) as joined:
                    assert joined is outer
            assert source.load() == expected
    _retired(source, observed, previous, holds)
    assert canonical.read_bytes() == b"unreviewed canonical bytes must remain untouched"
    assert not selected.parent.exists()
    assert counts == {"witness": 1, "parents": 1}, counts


def test_exact_store_path_property_keeps_callback_fallback(
    configured_source, local_root, monkeypatch
):
    source = MCPPermissionStore(
        configured_source.get_user_data_dir() / "mcp_permissions.json"
    )
    expected = _seed(source)
    _retire_seed_startup(local_root)
    previous = getattr(raw._local, "operation", None)
    called_under_coordinator = []

    def selected_path(receiver):
        assert receiver is source
        called_under_coordinator.append(storage._lock._is_owned())
        return vars(receiver)["path"]

    with controls._original_admissions(source) as observed:
        with raw.scope(source, mcp.ROUTE, writing=True) as outer:
            holds = tuple(raw._states[outer].holds)
            with monkeypatch.context() as changed:
                changed.setattr(
                    type(source), "path", property(selected_path), raising=False
                )
                with _original_join_counts() as counts:
                    with raw.scope(source, mcp.ROUTE, writing=True) as joined:
                        assert joined is outer
            # Restore the descriptor before invoking original payload/cleanup.
            assert source.load() == expected
    _retired(source, observed, previous, holds)
    assert called_under_coordinator and not any(called_under_coordinator)
    assert counts == {"witness": 3, "parents": 2}, counts
