"""Unrun AC77 regressions using actual ordinary evidence and original owners."""
import errno
import os
import sys
from contextlib import contextmanager

import pytest

from Tests.Backup_Recovery import test_config_companion_missing_record_evidence as missing
from Tests.Backup_Recovery import test_raw_related_source_preparation as sources
from tldw_chatbook.Backup_Recovery import bootstrap, config_participants, generation_witnesses
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage

configured_source = sources.configured_source
local_root = sources.local_root
pytestmark = missing.pytestmark


def _outcome(call, problem=None):
    try:
        return "returned", call()
    except Exception as error:
        return "raised", type(error), error is problem


def _guard_frame(source):
    frame = sys._getframe(1)
    code = config_participants.companion_guard.__code__
    for _ in range(16):
        if frame is None:
            return None
        if frame.f_code is code:
            state = raw._states.get(frame.f_locals.get("operation"))
            return frame if state is not None and state.source is source else None
        frame = frame.f_back
    return None


def _operation_outcome(source, hold, problem=None):
    leases_before, count_before = set(storage._live_leases), hold.count
    with sources._original_scope_acquisitions(source) as observed:
        outcome = _outcome(lambda: missing._read(source), problem)
    # Evaluate compatibility only after the original operation cleaned up.
    assert set(storage._live_leases) == leases_before and hold.count == count_before
    assert storage._holds.get(hold.key) is hold and storage._hold_serving(hold)
    assert hold.thread.is_alive() and not hold.stop.is_set() and hold.error is None
    assert all(lease not in storage._live_leases for lease in observed.returned)
    assert all(closed and removed for closed, removed in observed.closes)
    assert not raw._states and not storage._raw_operations
    return outcome, observed


def _two_fallbacks(observed, hold):
    assert len(observed.entries) == 1 and len(observed.returned) == 2
    operation, state, leases = observed.entries[0]
    assert state.companion_guard is None and len(leases) == 2
    assert len(state.holds) == 2 and all(actual is hold for actual in state.holds)
    assert observed.closes and all(closed and removed for closed, removed in observed.closes)
    assert all(lease not in storage._live_leases for lease in leases)
    assert operation not in raw._states and operation not in storage._raw_operations


@contextmanager
def _actual_reuses(caller, hold, key):
    """Observe only the original helper's exact immediate consumer/hold/key."""
    function = storage._derived_reuse
    code, namespace = function.__code__, function.__globals__
    caller_code, caller_globals = caller.__code__, caller.__globals__
    previous, rows = sys.getprofile(), []

    def observe(frame, event, result):
        if event == "return" and frame.f_code is code:
            parent = frame.f_back
            if (frame.f_globals is namespace and parent is not None
                and parent.f_code is caller_code and parent.f_globals is caller_globals
                and frame.f_locals.get("hold") is hold and frame.f_locals.get("key") == key
                and type(result) is tuple and len(result) == 2 and result[0] is True):
                rows.append(True)
        if previous is not None:
            previous(frame, event, result)

    sys.setprofile(observe)
    try:
        yield rows
    finally:
        sys.setprofile(previous)
        assert storage._derived_reuse is function and function.__code__ is code
        assert function.__globals__ is namespace
        assert caller.__code__ is caller_code and caller.__globals__ is caller_globals


@contextmanager
def _getter_change_after_guard_observation(source, hold, problem):
    """Declare source mutation after the real SH-locked before observation."""
    function = storage._derived_before
    code, namespace = function.__code__, function.__globals__
    guard_code = config_participants.companion_guard.__code__
    binding_code = config_participants.binding.__code__
    original = source._get_effective_config_path
    previous, fired, calls = sys.getprofile(), [], []

    def changed():
        parent = sys._getframe(1)
        if (parent.f_code is binding_code and parent.f_back is not None
            and parent.f_back.f_code is guard_code):
            calls.append(True)
            raise problem
        return original()

    def observe(frame, event, result):
        if event == "return" and frame.f_code is code and not fired:
            parent = frame.f_back
            if (frame.f_globals is namespace and parent is not None
                and parent.f_code is guard_code
                and frame.f_locals.get("hold") is hold):
                state = raw._states.get(parent.f_locals.get("operation"))
                if state is not None and state.source is source:
                    # The genuine guard acquired its directory/SH lock before
                    # this exact original helper. No lock/native result is faked.
                    assert len(state.leases) == 1 and state.leases[0] in storage._live_leases
                    assert parent.f_locals["guard"]._exit_callbacks
                    source._get_effective_config_path = changed
                    fired.append(True)
        if previous is not None:
            previous(frame, event, result)

    sys.setprofile(observe)
    try:
        yield fired, calls
    finally:
        sys.setprofile(previous)
        source._get_effective_config_path = original
        assert storage._derived_before is function and function.__code__ is code
        assert function.__globals__ is namespace


def test_missing_route_keeps_held_getter_customization_outside_found_checks(
    configured_source, monkeypatch,
):
    source = configured_source
    _, selected, hold = missing._warm_missing(source)
    before = selected.read_bytes()
    problem = LookupError("declared guard-only source getter failure")
    with _getter_change_after_guard_observation(source, hold, problem) as enabled_fault:
        enabled, enabled_native = _operation_outcome(source, hold, problem)
    with monkeypatch.context() as fresh:
        fresh.setattr(storage, "_EVIDENCE_REUSE", False)
        with _getter_change_after_guard_observation(source, hold, problem) as fresh_fault:
            disabled, disabled_native = _operation_outcome(source, hold, problem)
    assert enabled_fault[0] == fresh_fault[0] == [True]
    assert selected.read_bytes() == before
    assert disabled == ("returned", "retirement")
    _two_fallbacks(disabled_native, hold)
    assert enabled == disabled
    assert enabled_fault[1] == fresh_fault[1] == []
    _two_fallbacks(enabled_native, hold)


@pytest.mark.parametrize("slot", ["_open_verified_parent", "_native_close"])
def test_engaged_missing_reader_native_alias_keeps_original_callback(
    configured_source, monkeypatch, slot,
):
    source = configured_source
    _, selected, hold = missing._warm_missing(source)
    before = selected.read_bytes()
    original, calls = getattr(bootstrap, slot), []

    def changed(*args, **kwargs):
        if _guard_frame(source) is not None:
            calls.append((len(args), tuple(sorted(kwargs))))
        return original(*args, **kwargs)

    with monkeypatch.context() as changed_slot:
        changed_slot.setattr(bootstrap, slot, changed)
        enabled, enabled_native = _operation_outcome(source, hold)
        enabled_calls = tuple(calls)
        calls.clear()
        with monkeypatch.context() as fresh:
            fresh.setattr(storage, "_EVIDENCE_REUSE", False)
            disabled, disabled_native = _operation_outcome(source, hold)
        fresh_calls = tuple(calls)
    assert getattr(bootstrap, slot) is original and selected.read_bytes() == before
    assert disabled == ("returned", "retirement") and fresh_calls
    _two_fallbacks(disabled_native, hold)
    _two_fallbacks(enabled_native, hold)
    assert enabled == disabled and enabled_calls == fresh_calls


@contextmanager
def _pause_open_retirement(authority):
    """Immediate close evidence for FDs returned by the real pause _open only."""
    open_function = type(authority)._open
    open_code = open_function.__code__
    pause_code = type(authority).pause_requested.__code__
    close = os.close
    previous, pending, closing, retired, issues = sys.getprofile(), {}, {}, [], []

    def in_pause(frame):
        for _ in range(16):
            if frame is None:
                return False
            if frame.f_code is pause_code:
                return frame.f_locals.get("self") is authority
            frame = frame.f_back
        return False

    def observe(frame, event, value):
        try:
            if event == "return" and frame.f_code is open_code and type(value) is int and in_pause(frame):
                info = os.fstat(value)
                if value in pending:
                    issues.append("fd_reused_before_observed_close")
                pending[value] = (info.st_dev, info.st_ino)
            elif value is close and event in ("c_call", "c_return", "c_exception"):
                if event == "c_call":
                    fd = frame.f_locals.get("fd")
                    if type(fd) is not int or fd not in pending:
                        args = frame.f_locals.get("args")
                        fd = args[0] if type(args) is tuple and len(args) == 1 else None
                    if type(fd) is int and fd in pending:
                        info = os.fstat(fd)
                        closing[id(frame)] = fd, (info.st_dev, info.st_ino) == pending[fd]
                elif id(frame) in closing:
                    fd, same = closing.pop(id(frame))
                    gone = False
                    if event == "c_return":
                        try:
                            os.fstat(fd)
                        except OSError as error:
                            gone = error.errno == errno.EBADF
                    retired.append((same, gone, event == "c_return"))
                    pending.pop(fd)
        except Exception as error:
            issues.append(type(error).__name__)
        finally:
            if previous is not None:
                previous(frame, event, value)

    sys.setprofile(observe)
    try:
        yield retired
    finally:
        sys.setprofile(previous)
        assert type(authority)._open is open_function and open_function.__code__ is open_code
        assert os.close is close
        assert not issues and not pending and not closing
        assert retired and all(same and gone and returned for same, gone, returned in retired)


@pytest.mark.parametrize("slot", ["_read", "_groups"])
def test_engaged_pause_derivation_keeps_existing_authority_callback(
    configured_source, monkeypatch, slot,
):
    source = configured_source
    _, selected, hold = missing._warm_missing(source)
    authority = hold.authority
    caller = type(authority).pause_requested
    key = ("pause", hold.names, str(bootstrap.effective_config_path()))
    for _ in range(3):
        assert authority.pause_requested(hold.names) is False
    with _actual_reuses(caller, hold, key) as reused:
        assert authority.pause_requested(hold.names) is False
    assert reused == [True] and hold.derived_evidence[key][0].confirmed
    before, leases_before, count_before = selected.read_bytes(), set(storage._live_leases), hold.count
    problem, calls = LookupError("declared pause-reader callback failure"), []

    def changed(*args, **kwargs):
        calls.append(True)
        raise problem

    with monkeypatch.context() as changed_slot:
        changed_slot.setattr(authority, slot, changed)
        with _pause_open_retirement(authority) as enabled_retired:
            enabled = _outcome(lambda: authority.pause_requested(hold.names), problem)
        enabled_calls = len(calls)
        with monkeypatch.context() as fresh:
            fresh.setattr(storage, "_EVIDENCE_REUSE", False)
            with _pause_open_retirement(authority) as fresh_retired:
                disabled = _outcome(lambda: authority.pause_requested(hold.names), problem)
    assert selected.read_bytes() == before and set(storage._live_leases) == leases_before
    assert hold.count == count_before and storage._holds.get(hold.key) is hold
    assert storage._hold_serving(hold) and hold.thread.is_alive()
    assert not raw._states and not storage._raw_operations
    assert disabled == ("raised", LookupError, True) and len(calls) > enabled_calls
    assert enabled == disabled and enabled_calls == 1 and len(calls) == 2


@pytest.mark.parametrize("slot", ["_source_scope_admitted", "_paired_witnesses"])
def test_engaged_generation_derivation_keeps_existing_reader_callback(
    configured_source, monkeypatch, slot,
):
    source = configured_source
    _, selected, hold = missing._warm_missing(source)
    before, leases_before, count_before = selected.read_bytes(), set(storage._live_leases), hold.count
    lease = storage.acquire_storage(selected)
    try:
        assert lease in storage._live_leases and lease._key == hold.key
        root, names = lease.execution_context(selected)
        assert root == bootstrap.default_bootstrap_root() and names == hold.names
        key = ("witness", str(bootstrap.effective_config_path()), str(selected))
        for _ in range(3):
            assert generation_witnesses._witnesses(selected, lease) == []
        with _actual_reuses(generation_witnesses._witnesses, hold, key) as reused:
            assert generation_witnesses._witnesses(selected, lease) == []
        assert reused == [True] and hold.derived_evidence[key][0].confirmed
        problem, calls = LookupError("declared generation-reader callback failure"), []

        def changed(*args, **kwargs):
            calls.append(True)
            raise problem

        with monkeypatch.context() as changed_slot:
            changed_slot.setattr(generation_witnesses, slot, changed)
            enabled = _outcome(lambda: generation_witnesses._witnesses(selected, lease), problem)
            enabled_calls = len(calls)
            with monkeypatch.context() as fresh:
                fresh.setattr(storage, "_EVIDENCE_REUSE", False)
                disabled = _outcome(lambda: generation_witnesses._witnesses(selected, lease), problem)
        assert lease in storage._live_leases and lease.execution_context(selected) == (root, names)
    finally:
        lease.close()
    assert lease not in storage._live_leases and hold.count == count_before
    assert set(storage._live_leases) == leases_before and selected.read_bytes() == before
    assert storage._holds.get(hold.key) is hold and storage._hold_serving(hold)
    assert hold.thread.is_alive() and not raw._states and not storage._raw_operations
    assert disabled == ("raised", LookupError, True) and len(calls) > enabled_calls
    assert enabled == disabled and enabled_calls == 1 and len(calls) == 2
