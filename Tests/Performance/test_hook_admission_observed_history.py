"""Observe an actual disabled/removal edge through stock controller admission."""

import copy
import inspect
import json

import pytest
import toml

from Tests.Agents.test_hook_permissions import (
    _approve,
    _edit,
    _owner,
    hook_file as _hook_file,
)
from Tests.Performance.console_storage_unit_observer import OriginalStorageUnitObserver
from tldw_chatbook.Backup_Recovery import (
    config_participants,
    raw_participants,
    storage_admission,
)
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.DB.private_sqlite_process import HelperLease
from tldw_chatbook.Agents.hook_permissions import HookPermissions
from tldw_chatbook.Agents.run_hooks import HookLaunchRefused

pytestmark = pytest.mark.bootstrap_profile
hook_file = _hook_file


@pytest.mark.asyncio
@pytest.mark.timeout(90)
@pytest.mark.parametrize("observation", ["snapshot", "admission"])
@pytest.mark.parametrize("edge", ["master_disabled", "row_disabled", "removed"])
async def test_stock_admission_observed_edge_preserves_consent_epoch(
    hook_file, tmp_path, edge, observation
):
    original = copy.deepcopy(toml.loads(hook_file.read_text())["hooks"])
    owner = _owner()
    assert _approve(owner).ready
    old = owner.targets("PostToolUse", None)[0]
    assert owner.target_current(old)
    if edge == "master_disabled":
        _edit(hook_file, lambda section: section.update(enabled=False))
    elif edge == "row_disabled":
        _edit(hook_file, lambda section: section["hook"][0].update(enabled=False))
    else:
        _edit(hook_file, lambda section: section.update(hook=[]))

    controller = object.__new__(ConsoleChatController)
    accessor_calls = []
    controller._hook_permissions_accessor = (
        lambda: accessor_calls.append(owner) or owner
    )
    baseline_raw = set(raw_participants._states)
    baseline_operations = set(storage_admission._operations)
    baseline_leases = set(storage_admission._live_leases)
    counts = {
        "config_admissions": 0,
        "storage_admissions": 0,
        "helper_spawns": 0,
        "history_reconciles": 0,
        "permission_store_reads": 0,
    }
    observer = OriginalStorageUnitObserver(
        counts, True, lambda label: counts.__setitem__(label, counts[label] + 1)
    )
    try:
        observer.install(config_participants, storage_admission, HelperLease)
        for name, label in (
            ("_reconcile", "history_reconciles"),
            ("_read_state", "permission_store_reads"),
        ):
            function = inspect.getattr_static(HookPermissions, name)
            code = observer._pin(function)
            observer.codes[code] = label
            observer.slots.append((HookPermissions, name, function))
            observer.monitor.set_local_events(
                observer.tool,
                code,
                observer.monitor.events.PY_START | observer.monitor.events.PY_RETURN,
            )
        if observation == "snapshot":
            clear = owner.snapshot().ready
        else:
            clear = await controller.hook_admission_reason() is None
    finally:
        receipt = observer.close()
    assert clear
    assert accessor_calls == ([owner] if observation == "admission" else [])
    assert receipt["complete"] and receipt["original_source_current"]
    assert receipt["global_events"] == 0 and receipt["hooks_retired_before_inactive"]
    assert set(raw_participants._states) == baseline_raw
    assert set(storage_admission._operations) == baseline_operations
    assert set(storage_admission._live_leases) == baseline_leases

    def restore(section):
        section.clear()
        section.update(copy.deepcopy(original))

    _edit(hook_file, restore)
    current = owner.snapshot()
    refused = False
    try:
        with owner.launch_guard(old, tool_name=None):
            pass
    except HookLaunchRefused:
        refused = True
    metadata = {
        "edge": edge,
        "observation": observation,
        "accessor_calls": len(accessor_calls),
        "observation_clear": clear,
        "counts": counts,
        "observer": receipt,
        "raw_operation_lease_baselines_retained": True,
        "restored_ready": current.ready,
        "old_launch_refused": refused,
        "no_subprocess_or_hook_command_launched": True,
    }
    (tmp_path / "hook-admission-observed-history.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    try:
        # Both original retained-consent and removal semantics must survive the
        # actual admission observation, not only full review snapshot callers.
        assert current.ready is (edge != "removed")
        assert (
            refused
        ), "observed disabled/removal edge retained a queued old grant epoch"
        if edge == "removed":
            assert _approve(owner).ready
        fresh = owner.targets("PostToolUse", None)[0]
        assert fresh != old
        with owner.launch_guard(fresh, tool_name=None):
            pass
    finally:
        owner.close()
