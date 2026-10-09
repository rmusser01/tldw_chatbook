"""Pure participant registration retains the actual pre-lock refusal boundary."""

import inspect
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from Tests.Backup_Recovery import test_raw_related_source_preparation as source_cases
from Tests.Backup_Recovery.test_generation_witness_observation import _insert_pending
from tldw_chatbook.Backup_Recovery import bootstrap, mcp_source_participants as sources
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage

configured_source = source_cases.configured_source
local_root = source_cases.local_root
installed_source_case = source_cases.installed_source_case


@contextmanager
def _original_registration_boundary(case, *, mutate=None):
    """Select the real full pre-lock gate by owner locals, never by line number."""
    scope_code = inspect.unwrap(raw._scope).__code__
    gate_code = raw._participant_state.__code__
    acquire_code = storage.acquire_storage.__code__
    state_code = raw._State.__init__.__code__
    observed = SimpleNamespace(barriers=[], issued=[], states=[], lock_calls=0)
    previous = sys.getprofile()

    def scope_for(frame):
        while frame is not None:
            if (
                frame.f_code is scope_code
                and frame.f_locals.get("source") is case.source
            ):
                return frame
            frame = frame.f_back
        return None

    def observe(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        scope = scope_for(frame)
        if frame.f_code is acquire_code and event == "return" and scope is not None:
            if isinstance(result, storage.StorageLease):
                assert result in storage._live_leases
                observed.issued.append(result)
        elif (
            frame.f_code is state_code
            and event == "call"
            and frame.f_locals["source"] is case.source
        ):
            observed.states.append(frame.f_locals["self"])
        elif (
            event == "c_call"
            and getattr(result, "__self__", None) is case.source._mcp_source_lock
            and getattr(result, "__name__", None) == "acquire"
            and scope is not None
        ):
            observed.lock_calls += 1
        elif (
            frame.f_code is gate_code
            and event == "call"
            and frame.f_back.f_code is scope_code
            and scope is not None
            and scope.f_locals.get("source_lock") is case.source._mcp_source_lock
            and scope.f_locals.get("source_key") is None
            and scope.f_locals.get("state") is None
            and not observed.barriers
        ):
            attempt = scope.f_locals["attempt"]
            preparation = attempt._mcp_preparation
            assert preparation is raw._local.pending_mcp_preparation
            assert preparation[1] is case.source and preparation[3] == case.path
            assert attempt in storage._pending_acquisitions
            assert preparation[4] in storage._live_leases
            observed.barriers.append((attempt, preparation[4]))
            if mutate is not None:
                mutate()

    sys.setprofile(observe)
    try:
        yield observed
    finally:
        sys.setprofile(previous)


def _assert_retired_before_effects(case, observed, before_leases, before):
    assert observed.issued and not observed.states and observed.lock_calls == 0
    assert all(lease not in storage._live_leases for lease in observed.issued)
    assert set(storage._live_leases) == before_leases
    assert not any(state.source is case.source for state in raw._states.values())
    assert case.path.read_bytes() == before
    assert not case.path.with_suffix(".json.tmp").exists()
    if observed.barriers:
        attempt, lease = observed.barriers[0]
        assert attempt not in storage._pending_acquisitions
        assert lease in observed.issued and lease not in storage._live_leases


@pytest.mark.parametrize("installed_source_case", ["permission"], indirect=True)
@pytest.mark.parametrize("change", ["source", "pending"])
def test_retained_prelock_gate_refuses_real_change_after_registration(
    installed_source_case, monkeypatch, local_root, change
):
    case = installed_source_case
    before, before_leases = case.path.read_bytes(), set(storage._live_leases)
    binding = sources._BINDINGS[case.source]
    mutations = []
    control = local_root.parent / "registration-pending-control"
    pending = local_root / ("pending-" + bootstrap._key("during-read") + ".json")
    if change == "pending":
        bootstrap.os.mkdir(control, 0o700)
        assert not pending.exists()
    try:
        with monkeypatch.context() as patch:

            def mutate():
                if change == "source":
                    patch.setattr(
                        case.source, "path", case.path.with_name("retargeted.json")
                    )
                else:
                    _insert_pending(local_root, control, binding.profile)
                    bootstrap.os.chmod(pending, 0o600)
                    assert pending.is_file()
                mutations.append(change)

            with _original_registration_boundary(case, mutate=mutate) as observed:
                with pytest.raises((bootstrap.RecoveryRequired, ValueError)):
                    case.read()
    finally:
        if change == "pending":
            if pending.exists():
                bootstrap.os.unlink(pending)
            bootstrap.os.rmdir(control)
    assert mutations == [change] and len(observed.barriers) == 1
    _assert_retired_before_effects(case, observed, before_leases, before)


@pytest.mark.parametrize("installed_source_case", ["permission"], indirect=True)
@pytest.mark.parametrize("field", ["owner", "selected"])
def test_registration_refuses_mismatched_existing_participant_metadata(
    installed_source_case, monkeypatch, field
):
    case = installed_source_case
    participant = raw._source_participants[case.source]
    record = raw._participants[participant]
    assert record.source() is case.source and record.source_type is type(case.source)
    assert record.owner == "mcp.permissions" and record.selected == case.path
    before, before_leases = case.path.read_bytes(), set(storage._live_leases)
    replacement = (
        "mcp.context"
        if field == "owner"
        else case.path.with_name("different-selected.json")
    )
    with monkeypatch.context() as patch:
        patch.setattr(record, field, replacement)
        with _original_registration_boundary(case) as observed:
            with pytest.raises(bootstrap.RecoveryRequired):
                case.read()
    _assert_retired_before_effects(case, observed, before_leases, before)
    assert record.owner == "mcp.permissions" and record.selected == case.path
    assert raw._source_participants[case.source] is participant
