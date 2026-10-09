"""Pending canonical observation owns real leases without caching witnesses."""

import copy
import inspect
import sys
import threading
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest

from Tests.Backup_Recovery import test_raw_related_source_preparation as source_cases
from Tests.Backup_Recovery.test_generation_witness_observation import _insert_pending
from tldw_chatbook.Backup_Recovery import bootstrap, generation_witnesses
from tldw_chatbook.Backup_Recovery import mcp_source_participants as sources
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.MCP import recovery_activation

configured_source = source_cases.configured_source
local_root = source_cases.local_root
installed_source_case = source_cases.installed_source_case


def _source_scope(frame, source, code):
    while frame is not None:
        if frame.f_code is code and frame.f_locals.get("source") is source:
            return frame
        frame = frame.f_back
    return None


@contextmanager
def _original_pending_witness(case, phase, mutate):
    """Mutate only at an actual canonical witness before raw state creation."""
    scope_code = inspect.unwrap(raw._scope).__code__
    witness_code = generation_witnesses._witnesses.__code__
    acquire_code = storage.acquire_storage.__code__
    close_code = storage.StorageLease.close.__code__
    state_code = raw._State.__init__.__code__
    observed = SimpleNamespace(issued=[], closed=[], states=[], barriers=[])
    previous = sys.getprofile()
    active_closes = {}

    def observe(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        scope = _source_scope(frame, case.source, scope_code)
        if frame.f_code is acquire_code and event == "return" and scope is not None:
            if isinstance(result, storage.StorageLease):
                assert result in storage._live_leases
                observed.issued.append(result)
        elif frame.f_code is close_code:
            lease = frame.f_locals["self"]
            if event == "call" and any(
                lease is original for original in observed.issued
            ):
                active_closes[id(frame)] = lease in storage._live_leases
            elif event == "return" and id(frame) in active_closes:
                was_live = active_closes.pop(id(frame))
                assert lease not in storage._live_leases
                if was_live:
                    observed.closed.append(lease)
        elif (
            frame.f_code is state_code
            and event == "call"
            and frame.f_locals["source"] is case.source
        ):
            observed.states.append(frame.f_locals["self"])
        elif (
            frame.f_code is witness_code
            and event == ("call" if phase == "before" else "return")
            and frame.f_locals["path"] == case.path
            and scope is not None
            and scope.f_locals.get("state") is None
            and not observed.barriers
        ):
            lease = frame.f_locals["lease"]
            attempt = scope.f_locals["attempt"]
            assert attempt in storage._pending_acquisitions
            assert lease in storage._live_leases
            assert getattr(raw._local, "operation", None) is None
            if phase == "after":
                assert isinstance(result, (list, tuple))
            observed.barriers.append((attempt, lease))
            retired = mutate(lease)
            if retired is lease:
                # Recursive profiling is disabled inside this observer. Record
                # the actual mutation's positive close, not the later no-op.
                assert lease not in storage._live_leases
                observed.closed.append(lease)

    sys.setprofile(observe)
    try:
        yield observed
    finally:
        sys.setprofile(previous)


def _assert_early_retirement(case, observed, before_leases, before):
    assert len(observed.barriers) == 1 and observed.issued
    assert not observed.states, "refused preparation created a raw effect owner"
    attempt, _lease = observed.barriers[0]
    assert attempt not in storage._pending_acquisitions
    assert len({id(lease) for lease in observed.issued}) == len(observed.issued)
    assert all(lease not in storage._live_leases for lease in observed.issued)
    assert all(
        sum(closed is lease for closed in observed.closed) == 1
        for lease in observed.issued
    )
    assert set(storage._live_leases) == before_leases
    assert not any(state.source is case.source for state in raw._states.values())
    assert case.path.read_bytes() == before
    assert not case.path.with_suffix(".json.tmp").exists()


@pytest.mark.parametrize("installed_source_case", ["permission"], indirect=True)
@pytest.mark.parametrize("change", ["source_path", "config_module", "canonical"])
def test_pending_source_drift_retires_early_lease_before_raw_state(
    installed_source_case, monkeypatch, change
):
    case = installed_source_case
    before, before_leases = case.path.read_bytes(), set(storage._live_leases)
    binding = sources._BINDINGS[case.source]
    with monkeypatch.context() as patch:

        def mutate(_lease):
            if change == "source_path":
                patch.setattr(
                    case.source, "path", case.path.with_name("retargeted.json")
                )
            elif change == "config_module":
                patch.setitem(
                    sys.modules, "tldw_chatbook.config", ModuleType("displaced_config")
                )
            else:
                config = copy.deepcopy(binding.config._CONFIG_CACHE)
                config.setdefault("general", {})["users_name"] = "different_canonical"
                patch.setattr(binding.config, "_CONFIG_CACHE", config)
                assert sources.canonical_path(case.source) != case.path

        with _original_pending_witness(case, "after", mutate) as observed:
            with pytest.raises((bootstrap.RecoveryRequired, ValueError)):
                case.read()
    _assert_early_retirement(case, observed, before_leases, before)


@pytest.mark.parametrize("installed_source_case", ["permission"], indirect=True)
@pytest.mark.parametrize("phase", ["before", "after"])
@pytest.mark.parametrize("change", ["pause", "lease"])
def test_pending_witness_keeps_real_pause_and_lease_fences(
    installed_source_case, phase, change
):
    case = installed_source_case
    before, before_leases = case.path.read_bytes(), set(storage._live_leases)
    pauses = []

    def mutate(lease):
        if change == "pause":
            pauses.append(storage._begin_local_pause())
        else:
            lease.close()
            assert lease not in storage._live_leases
            return lease

    try:
        with _original_pending_witness(case, phase, mutate) as observed:
            with pytest.raises(bootstrap.RecoveryRequired):
                case.read()
    finally:
        for pause in pauses:
            pause.resume()
    _assert_early_retirement(case, observed, before_leases, before)


@pytest.mark.parametrize("installed_source_case", ["permission"], indirect=True)
def test_custom_canonical_acquisition_keeps_original_positional_route(
    installed_source_case, monkeypatch
):
    case = installed_source_case
    original = recovery_activation.acquire_storage
    before, before_leases = case.path.read_bytes(), set(storage._live_leases)
    calls, issued = [], []

    def custom(*args, **kwargs):
        calls.append((args, dict(kwargs)))
        lease = original(*args, **kwargs)
        assert lease in storage._live_leases
        issued.append(lease)
        return lease

    with monkeypatch.context() as patch:
        patch.setattr(recovery_activation, "acquire_storage", custom)
        assert case.read() is False
    assert len(calls) > 1 and len(issued) == len(calls)
    assert all(args == (case.path,) and not kwargs for args, kwargs in calls)
    assert all(lease not in storage._live_leases for lease in issued)
    assert set(storage._live_leases) == before_leases
    assert not any(state.source is case.source for state in raw._states.values())
    assert case.path.read_bytes() == before
    assert recovery_activation.acquire_storage is original


@pytest.mark.parametrize("installed_source_case", ["permission"], indirect=True)
def test_pending_preparation_rereads_changed_recovery_controls(
    installed_source_case, local_root
):
    case = installed_source_case
    before, before_leases = case.path.read_bytes(), set(storage._live_leases)
    control = local_root.parent / "new-pending-control"
    raw.os.mkdir(control, 0o700)
    target = local_root / ("pending-" + bootstrap._key("during-read") + ".json")
    assert not target.exists()

    def mutate(_lease):
        # Use the original valid native pending-record fixture. The first
        # witness body completed; a later fresh body must see this new state.
        binding = sources._BINDINGS[case.source]
        _insert_pending(local_root, control, binding.profile)
        raw.os.chmod(target, 0o600)
        assert target.is_file()

    try:
        with _original_pending_witness(case, "after", mutate) as observed:
            with pytest.raises((bootstrap.RecoveryRequired, ValueError)):
                case.read()
    finally:
        if target.exists():
            raw.os.unlink(target)
        raw.os.rmdir(control)
    _assert_early_retirement(case, observed, before_leases, before)


@pytest.mark.parametrize("installed_source_case", ["permission"], indirect=True)
def test_issued_pending_metadata_refuses_copies_foreign_actor_and_expiry(
    installed_source_case, monkeypatch
):
    case = installed_source_case
    before, before_leases = case.path.read_bytes(), set(storage._live_leases)
    issued, refusals = [], []

    def mutate(lease):
        preparation = raw._local.pending_mcp_preparation
        attempt, source, binding, canonical, actual_lease = preparation
        assert (
            source is case.source and canonical == case.path and actual_lease is lease
        )
        assert binding is sources._BINDINGS[case.source]
        assert attempt._mcp_preparation is preparation
        assert (
            raw._check_pending_mcp_preparation(preparation, source, canonical) is lease
        )
        issued.append(preparation)

        copied_tuple = tuple(list(preparation))
        assert copied_tuple is not preparation
        with monkeypatch.context() as patch:
            patch.setattr(raw._local, "pending_mcp_preparation", copied_tuple)
            with pytest.raises(bootstrap.RecoveryRequired):
                raw._check_pending_mcp_preparation(copied_tuple, source, canonical)
            refusals.append("copied_tuple")

        copied_attempt = copy.copy(attempt)
        forged = (copied_attempt, source, binding, canonical, lease)
        copied_attempt._mcp_preparation = forged
        assert copied_attempt not in storage._pending_acquisitions
        with monkeypatch.context() as patch:
            patch.setattr(raw._local, "pending_mcp_preparation", forged)
            with pytest.raises(bootstrap.RecoveryRequired):
                raw._check_pending_mcp_preparation(forged, source, canonical)
            refusals.append("copied_attempt")

        foreign_errors = []

        def foreign():
            raw._local.pending_mcp_preparation = preparation
            try:
                raw._check_pending_mcp_preparation(preparation, source, canonical)
            except BaseException as error:
                foreign_errors.append(error)
            finally:
                del raw._local.pending_mcp_preparation

        worker = threading.Thread(target=foreign)
        worker.start()
        worker.join(3)
        assert not worker.is_alive()
        assert len(foreign_errors) == 1 and isinstance(
            foreign_errors[0], bootstrap.RecoveryRequired
        )
        refusals.append("foreign_actor")
        assert raw._local.pending_mcp_preparation is preparation
        assert attempt._mcp_preparation is preparation and lease in storage._live_leases

    with _original_pending_witness(case, "before", mutate) as observed:
        assert case.read() is False
    assert len(issued) == 1 and len(observed.barriers) == 1
    assert refusals == ["copied_tuple", "copied_attempt", "foreign_actor"]
    preparation = issued[0]
    attempt, source, _binding, canonical, lease = preparation
    assert (
        attempt not in storage._pending_acquisitions
        and lease not in storage._live_leases
    )
    assert getattr(raw._local, "pending_mcp_preparation", None) is not preparation
    with monkeypatch.context() as patch:
        patch.setattr(raw._local, "pending_mcp_preparation", preparation)
        with pytest.raises(bootstrap.RecoveryRequired):
            raw._check_pending_mcp_preparation(preparation, source, canonical)
    assert observed.states
    assert all(not state.active and not state.uncertain for state in observed.states)
    assert all(
        not state.pins and not state.files and not state.descriptors
        for state in observed.states
    )
    assert all(lease not in storage._live_leases for lease in observed.issued)
    assert all(
        sum(closed is original for closed in observed.closed) == 1
        for original in observed.issued
    )
    assert set(storage._live_leases) == before_leases
    assert not any(state.source is case.source for state in raw._states.values())
    assert case.path.read_bytes() == before


def _delegating_recovery_callback(name, original, calls):
    if name == "selected_path":

        def selected(canonical, *, retained=None):
            calls.append((canonical, retained))
            return original(canonical, retained=retained)

        return selected

    @contextmanager
    def observed(path, *, retained=None):
        calls.append((path, retained))
        with original(path, retained=retained) as witnesses:
            yield witnesses

    return observed


@pytest.mark.parametrize("installed_source_case", ["permission"], indirect=True)
@pytest.mark.parametrize("callback", ["selected_path", "observed"])
def test_custom_recovery_callback_keeps_original_source_body(
    installed_source_case, monkeypatch, callback
):
    case = installed_source_case
    original = getattr(recovery_activation, callback)
    calls = []
    custom = _delegating_recovery_callback(callback, original, calls)
    before, before_leases = case.path.read_bytes(), set(storage._live_leases)
    with monkeypatch.context() as patch:
        patch.setattr(recovery_activation, callback, custom)
        with source_cases._original_scope_acquisitions(case.source) as observed:
            assert case.read() is False
    assert calls and calls[0] == (case.path, None)
    assert all(path == case.path for path, _retained in calls)
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
    assert getattr(recovery_activation, callback) is original


@pytest.mark.parametrize("installed_source_case", ["permission"], indirect=True)
@pytest.mark.parametrize("callback", ["selected_path", "observed"])
def test_late_recovery_callback_change_refuses_without_reentering_custom_body(
    installed_source_case, monkeypatch, callback
):
    case = installed_source_case
    original = getattr(recovery_activation, callback)
    calls = []
    custom = _delegating_recovery_callback(callback, original, calls)
    before, before_leases = case.path.read_bytes(), set(storage._live_leases)
    with monkeypatch.context() as patch:

        def mutate(lease):
            preparation = raw._local.pending_mcp_preparation
            assert preparation[1] is case.source and preparation[4] is lease
            assert lease in storage._live_leases
            patch.setattr(recovery_activation, callback, custom)

        with _original_pending_witness(case, "after", mutate) as observed:
            with pytest.raises(bootstrap.RecoveryRequired):
                case.read()
    assert (
        not calls
    ), "issued preparation invoked a successor callback instead of refusing"
    _assert_early_retirement(case, observed, before_leases, before)
    assert getattr(recovery_activation, callback) is original
