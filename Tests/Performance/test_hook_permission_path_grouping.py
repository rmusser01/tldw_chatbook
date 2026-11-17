"""Original hook readers retain one admitted group for all declared members."""

from collections import Counter
import inspect
import os
from pathlib import Path

import pytest

from Tests.private_profile import private_profile_test
from Tests.Performance.test_sensitive_db_path_grouping import _starts


def _settled_owner():
    from tldw_chatbook import config
    from tldw_chatbook.Agents.hook_permissions import HookPermissions

    config.get_user_data_dir()
    owner = HookPermissions()
    owner.snapshot()
    owner.snapshot()
    return owner


@private_profile_test
def test_original_hook_authority_read_groups_all_members_and_retires(
    request, record_property
):
    from tldw_chatbook.Agents.hook_permissions import HookPermissions
    from tldw_chatbook.Backup_Recovery import raw_participants as raw
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    owner = _settled_owner()
    reader = HookPermissions._read_state.__code__
    acquisition = storage._acquire_storage.__code__
    codes, native = {reader, acquisition}, None
    if os.name == "nt":
        from tldw_chatbook.Utils.windows_files import _Native

        native = inspect.getattr_static(_Native, "open_handle").__code__
        codes.add(native)
    counts, issued = Counter(), []

    def entered(code, offset):
        if code is acquisition:
            counts["acquisition"] += 1
        elif code is native:
            counts["native"] += 1
        elif code is reader:
            counts["reader"] += 1
            operation = getattr(raw._local, "operation", None)
            state = raw._states[operation]
            assert state.source is owner and state.route == "hook_permissions"
            issued.append((operation, state, tuple(state.leases)))

    with _starts(codes, entered):
        result = owner.authority_read()
    assert result.owner is owner
    assert counts["reader"] == 1
    assert len(issued) == 1
    operation, state, leases = issued[0]
    selected = Path(result.snapshot.store_path)
    assert state.selected == selected
    assert state.paths[:2] == (selected, selected.with_name(selected.name + ".lock"))
    assert len(state.paths) == 3
    temporary = state.paths[2]
    assert temporary.parent == selected.parent
    assert temporary.name.startswith("." + selected.name + ".")
    assert temporary.suffix == ".tmp"
    assert state.directories == (selected.parent,)
    record_property("original_hook_counts", dict(counts))
    record_property("declared_members", len(state.paths) + len(state.directories))
    record_property("hook_scope_leases", len(leases))
    # The test-only red run executes all original code and observes four leases;
    # the one-condition product change makes the exact same source issue one.
    assert len(leases) == 1
    assert operation not in raw._states and operation not in storage._raw_operations
    assert not state.pins and not state.files and not state.descriptors
    assert all(lease not in storage._live_leases for lease in leases)


@private_profile_test
def test_hook_group_companion_refusal_precedes_original_reader(request, monkeypatch):
    from tldw_chatbook.Agents.hook_permissions import default_hook_permissions_path
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery import raw_participants as raw
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    owner = _settled_owner()
    selected = Path(default_hook_permissions_path())
    companion = selected.with_name(selected.name + ".lock")
    original, attempted = storage._Acquisition.check, []
    before = set(storage._live_leases)

    def refuse(attempt, path=None):
        if path == companion:
            attempted.append(path)
            raise bootstrap.RecoveryRequired("hook_group_admission_refused")
        return original(attempt, path)

    monkeypatch.setattr(storage._Acquisition, "check", refuse)
    entered = False
    with pytest.raises(bootstrap.RecoveryRequired, match="hook_group_admission_refused"):
        with raw.scope(owner, "hook_permissions", writing=True, selected_read=selected):
            entered = True
            owner._read_state(selected)
    assert not entered and len(attempted) == 1
    assert not any(state.source is owner for state in raw._states.values())
    assert set(storage._live_leases) == before


@private_profile_test
def test_hook_group_unretired_native_owner_keeps_its_actual_lease(request, monkeypatch):
    from tldw_chatbook.Agents.hook_permissions import default_hook_permissions_path
    from tldw_chatbook.Backup_Recovery import raw_participants as raw
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    owner = _settled_owner()
    selected = Path(default_hook_permissions_path())
    original, retained = raw._retire, []

    def hold_first(state):
        if state.source is owner and not retained:
            retained.extend((operation, state) for operation, current in raw._states.items()
                            if current is state)
            return False
        return original(state)

    monkeypatch.setattr(raw, "_retire", hold_first)
    try:
        with raw.scope(owner, "hook_permissions", writing=True, selected_read=selected):
            state, _precondition, error = owner._read_state(selected)
            assert state is not None and error is None
        assert len(retained) == 1
        operation, state = retained[0]
        leases = tuple(state.leases)
        assert len(leases) == 1
        assert raw._states[operation] is state and operation in storage._raw_operations
        assert not state.active and state.pins
        assert all(lease in storage._live_leases for lease in leases)
    finally:
        monkeypatch.setattr(raw, "_retire", original)
        for operation, state in retained:
            assert original(state)
            with storage._changed:
                del raw._states[operation]
                storage._raw_operations.discard(operation)
                storage._changed.notify_all()
    assert all(lease not in storage._live_leases for lease in leases)
