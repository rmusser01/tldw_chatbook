"""Named stock loading retains original callback and publication boundaries."""

import inspect
import sys
from contextlib import contextmanager
from types import ModuleType


import pytest

from Tests.Backup_Recovery import test_raw_owned_permission_load_counts as count_cases
from Tests.Backup_Recovery.test_generation_witness_observation import _insert_pending
from tldw_chatbook import Backup_Recovery as backup_package
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.MCP import permission_store

configured_source = count_cases.configured_source
local_root = count_cases.local_root
installed_source_case = count_cases.installed_source_case
checked_permission_case = count_cases.checked_permission_case


class _NestedScopeRefused(RuntimeError):
    pass


@pytest.mark.parametrize("installed_source_case", ["permission"], indirect=True)
@pytest.mark.parametrize("guard", ["scope", "check", "package_scope"])
def test_custom_raw_scope_retains_original_nested_refusal(
    checked_permission_case, monkeypatch, guard
):
    case = checked_permission_case
    original = raw._scope
    assert permission_store._capture_console_owned_load(case.source) is not None
    before, leases_before = case.path.read_bytes(), set(storage._live_leases)
    entries = []

    @contextmanager
    def custom(source, route, **kwargs):
        if source is case.source:
            nested = getattr(raw._local, "operation", None) is not None
            entries.append(nested)
            if nested:
                raise _NestedScopeRefused("original custom nested scope refused")
        with original(source, route, **kwargs) as operation:
            yield operation

    with monkeypatch.context() as patch:
        if guard == "scope":
            patch.setattr(raw, "_scope", custom)
        elif guard == "package_scope":
            clone = ModuleType(raw.__name__)
            vars(clone).update(vars(raw))
            clone._scope = custom
            patch.setattr(backup_package, "raw_participants", clone)
            assert sys.modules[raw.__name__] is raw
        else:
            scope_code = inspect.unwrap(original).__code__
            original_check = raw._check

            def custom_check(operation, *args, **kwargs):
                parent = sys._getframe(1)
                state = raw._states.get(operation)
                if (
                    parent.f_code is scope_code
                    and state is not None
                    and state.source is case.source
                ):
                    nested = parent.f_locals.get("previous") is not None
                    entries.append(nested)
                    if nested:
                        raise _NestedScopeRefused(
                            "original custom nested check refused"
                        )
                return original_check(operation, *args, **kwargs)

            patch.setattr(raw, "_check", custom_check)
        with pytest.raises(_NestedScopeRefused):
            count_cases.read_checked_payload(case)
    expected = [True] if guard == "package_scope" else [False, True]
    assert entries == expected, "the custom nested source boundary was skipped"
    assert set(storage._live_leases) == leases_before
    assert not any(state.source is case.source for state in raw._states.values())
    assert case.path.read_bytes() == before
    assert raw._scope is original


@pytest.mark.parametrize("installed_source_case", ["permission"], indirect=True)
def test_property_backed_locked_loader_is_resolved_only_in_ordinary_scope(
    checked_permission_case, monkeypatch
):
    case = checked_permission_case
    original = permission_store.MCPPermissionStore._load_locked
    body = inspect.unwrap(original)
    resolutions = []
    before, leases = case.path.read_bytes(), set(storage._live_leases)

    def getter(source):
        operation = getattr(raw._local, "operation", None)
        resolutions.append((source, operation))
        return body.__get__(source, type(source))

    with monkeypatch.context() as patch:
        patch.setattr(
            permission_store.MCPPermissionStore, "_load_locked", property(getter)
        )
        assert permission_store._capture_console_owned_load(case.source) is None
        assert resolutions == [], "pure capture invoked a replaced descriptor"
        payload = count_cases.read_checked_payload(case)
    assert payload["kill_switch"] is False
    assert len(resolutions) == 1 and resolutions[0][0] is case.source
    assert resolutions[0][1] is not None and resolutions[0][1] not in raw._states
    assert set(storage._live_leases) == leases and case.path.read_bytes() == before


@pytest.mark.parametrize("installed_source_case", ["permission"], indirect=True)
@pytest.mark.parametrize("phase", ["helper_entry", "body_return"])
@pytest.mark.parametrize("change", ["source", "pending", "registry"])
def test_owned_policy_refuses_real_drift_before_effect_or_result(
    checked_permission_case, monkeypatch, local_root, phase, change
):
    case = checked_permission_case
    before, leases = case.path.read_bytes(), set(storage._live_leases)
    registry = local_root / "admission" / "registry.json"
    registry_before = registry.read_bytes()
    control = local_root.parent / "owned-load-pending-control"
    marker = local_root / ("pending-" + bootstrap._key("during-read") + ".json")
    if change == "pending":
        bootstrap.os.mkdir(control, 0o700)
        assert not marker.exists()
    helper_code = permission_store._load_in_owned_scope.__code__
    body_code = inspect.unwrap(
        permission_store.MCPPermissionStore._load_locked
    ).__code__
    mutations, operations = [], []
    previous = sys.getprofile()
    with monkeypatch.context() as patch:

        def observe(frame, event, result):
            if previous is not None:
                previous(frame, event, result)
            selected = (
                phase == "helper_entry"
                and frame.f_code is helper_code
                and event == "call"
                or phase == "body_return"
                and frame.f_code is body_code
                and event == "return"
                and isinstance(result, dict)
            )
            if not selected or mutations:
                return
            operation = getattr(raw._local, "operation", None)
            state = raw._states.get(operation)
            if state is None or state.source is not case.source:
                return
            assert state.active and operation in storage._raw_operations
            operations.append(operation)
            if change == "source":
                patch.setattr(
                    case.source, "path", case.path.with_name("retargeted.json")
                )
            elif change == "pending":
                binding = raw.mcp_sources._BINDINGS[case.source]
                _insert_pending(local_root, control, binding.profile)
                bootstrap.os.chmod(marker, 0o600)
                assert marker.is_file()
            else:
                registry.write_bytes(b"{")
                assert registry.read_bytes() == b"{"
            mutations.append(change)

        sys.setprofile(observe)
        try:
            with pytest.raises((bootstrap.RecoveryRequired, ValueError)):
                count_cases.read_checked_payload(case)
        finally:
            sys.setprofile(previous)
            if change == "registry":
                registry.write_bytes(registry_before)
            if change == "pending":
                if marker.exists():
                    bootstrap.os.unlink(marker)
                bootstrap.os.rmdir(control)
    assert mutations == [change] and len(operations) == 1
    assert (
        operations[0] not in raw._states
        and operations[0] not in storage._raw_operations
    )
    assert set(storage._live_leases) == leases
    assert case.path.read_bytes() == before
    assert not case.path.with_suffix(".json.tmp").exists()
