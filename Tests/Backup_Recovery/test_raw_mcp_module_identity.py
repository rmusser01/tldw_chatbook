"""Pending MCP custody must qualify the module that the source actually imports."""

import sys
from types import ModuleType

import pytest

from Tests.Backup_Recovery import (
    test_raw_mcp_pending_preparation_lifetimes as lifetimes,
)
from Tests.Backup_Recovery import test_raw_related_source_preparation as source_cases
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook import MCP
from tldw_chatbook.MCP import recovery_activation

configured_source = source_cases.configured_source
local_root = source_cases.local_root
installed_source_case = source_cases.installed_source_case


def _replacement_module(calls):
    replacement = ModuleType(recovery_activation.__name__)
    replacement.__dict__.update(vars(recovery_activation))
    original = recovery_activation.selected_path

    def selected(*args, **kwargs):
        calls.append(
            (args, dict(kwargs), getattr(raw._local, "pending_mcp_preparation", None))
        )
        return original(*args, **kwargs)

    replacement.selected_path = selected
    return replacement


@pytest.mark.parametrize("installed_source_case", ["permission"], indirect=True)
def test_sys_modules_replacement_keeps_original_unprepared_callback_route(
    installed_source_case, monkeypatch
):
    case = installed_source_case
    before, before_leases = case.path.read_bytes(), set(storage._live_leases)
    calls = []
    replacement = _replacement_module(calls)
    assert sys.modules[recovery_activation.__name__] is recovery_activation
    assert MCP.recovery_activation is recovery_activation

    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, recovery_activation.__name__, replacement)
        # Leave the package slot unchanged: this is the split lookup in the
        # original qualifier versus the source's full submodule import.
        assert MCP.recovery_activation is recovery_activation
        with source_cases._original_scope_acquisitions(case.source) as observed:
            assert case.read() is False

    assert observed.entries and observed.closes
    assert all(closed and removed for closed, removed in observed.closes)
    assert not observed.active_acquires and not observed.active_closes
    for operation, state, leases in observed.entries:
        assert operation not in raw._states and operation not in storage._raw_operations
        assert not state.active and not state.uncertain
        assert not state.pins and not state.files and not state.descriptors
        assert all(lease not in storage._live_leases for lease in leases)
    assert set(storage._live_leases) == before_leases
    assert case.path.read_bytes() == before
    assert sys.modules[recovery_activation.__name__] is recovery_activation
    assert MCP.recovery_activation is recovery_activation
    assert calls
    # Current RED passes an issued pending lease to a foreign selected_path.
    assert calls[0] == ((case.path,), {"retained": None}, None)
    assert all(pending is None for _args, _kwargs, pending in calls)


@pytest.mark.parametrize("installed_source_case", ["permission"], indirect=True)
def test_late_sys_modules_replacement_refuses_before_custom_callback(
    installed_source_case, monkeypatch
):
    case = installed_source_case
    before, before_leases = case.path.read_bytes(), set(storage._live_leases)
    calls = []
    replacement = _replacement_module(calls)
    refused = []

    with monkeypatch.context() as patch:

        def mutate(lease):
            preparation = raw._local.pending_mcp_preparation
            assert preparation[1] is case.source and preparation[4] is lease
            assert lease in storage._live_leases
            patch.setitem(sys.modules, recovery_activation.__name__, replacement)
            assert MCP.recovery_activation is recovery_activation

        with lifetimes._original_pending_witness(case, "after", mutate) as observed:
            try:
                case.read()
            except bootstrap.RecoveryRequired as error:
                refused.append(error)

    assert sys.modules[recovery_activation.__name__] is recovery_activation
    assert MCP.recovery_activation is recovery_activation
    assert len(observed.barriers) == 1
    assert not calls, "issued preparation invoked the substituted module callback"
    assert len(refused) == 1
    lifetimes._assert_early_retirement(case, observed, before_leases, before)
