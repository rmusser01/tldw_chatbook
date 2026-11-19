"""Unrun single-case AC77 held source change after genuine confirmed reuse."""
import errno
import json
import os
import sys
from contextlib import contextmanager

from Tests.Backup_Recovery import test_config_companion_missing_record_evidence as missing
from Tests.Backup_Recovery import test_config_companion_review_regressions as review
from tldw_chatbook.Backup_Recovery import bootstrap, config_participants, native_files
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage

configured_source = review.configured_source
local_root = review.local_root
pytestmark = missing.pytestmark


@contextmanager
def _temporary_guard_retirement(source, hold):
    """Observe only actual yielded companion directory/registry-lock FDs."""
    authority = hold.authority
    directory = type(authority)._directory.__wrapped__
    locking = type(authority)._lock.__wrapped__
    native_directory = native_files.pinned_directory.__wrapped__
    guard = config_participants.companion_guard
    previous, close = sys.getprofile(), os.close
    rows, pending, closing, errors = [], {}, {}, []

    def in_guard(frame):
        for _ in range(16):
            if frame is None:
                return False
            if frame.f_code is guard.__code__ and frame.f_globals is guard.__globals__:
                state = raw._states.get(frame.f_locals.get("operation"))
                return state is not None and state.source is source and state.holds[-1] is hold
            frame = frame.f_back
        return False

    def observe(frame, event, value):
        try:
            code = frame.f_code
            if event == "return" and type(value) is int and in_guard(frame):
                if code is native_directory.__code__:
                    parent = frame.f_back
                    for _ in range(16):
                        if parent is None or parent.f_code is directory.__code__:
                            break
                        parent = parent.f_back
                    if (parent is None or parent.f_locals.get("self") is not authority
                            or frame.f_locals.get("root") != authority.control_root):
                        return
                    kind = "directory"
                elif code is locking.__code__ and frame.f_locals.get("self") is authority:
                    assert frame.f_locals["name"] == "registry.lock"
                    assert frame.f_locals["mode"] == config_participants.fcntl.LOCK_SH
                    assert frame.f_locals["parent"] in pending
                    kind = "lock"
                elif code is directory.__code__ and frame.f_locals.get("self") is authority:
                    assert pending[value]["kind"] == "directory"
                    pending[value]["authority_yielded"] = True
                    return
                else:
                    return
                assert value not in pending
                info = os.fstat(value)
                row = {"kind": kind, "fd": value, "identity": (info.st_dev, info.st_ino),
                       "creator": frame, "closed": False, "terminal": False,
                       "authority_yielded": kind == "lock"}
                rows.append(row)
                pending[value] = row
            elif value is close and event in ("c_call", "c_return", "c_exception"):
                if event == "c_call":
                    row = pending.get(frame.f_locals.get("fd"))
                    if row is not None and row["creator"] is frame:
                        info = os.fstat(row["fd"])
                        assert (info.st_dev, info.st_ino) == row["identity"]
                        closing[id(frame)] = row
                elif id(frame) in closing:
                    row = closing.pop(id(frame))
                    assert row["creator"] is frame and event == "c_return"
                    try:
                        os.fstat(row["fd"])
                    except OSError as error:
                        assert error.errno == errno.EBADF
                    else:
                        raise AssertionError("companion guard FD live at actual close C_return")
                    row["closed"] = True
                    del pending[row["fd"]]
            elif event == "return" and value is None:
                for row in rows:
                    if row["creator"] is frame:
                        assert row["closed"]
                        row["terminal"] = True
        except BaseException as error:
            errors.append(type(error).__name__)
        finally:
            if previous is not None:
                previous(frame, event, value)

    sys.setprofile(observe)
    try:
        yield rows
    finally:
        sys.setprofile(previous)
        assert type(authority)._directory.__wrapped__ is directory
        assert type(authority)._lock.__wrapped__ is locking
        assert native_files.pinned_directory.__wrapped__ is native_directory and os.close is close
        assert config_participants.companion_guard is guard
        assert not errors and not pending and not closing
        assert [row["kind"] for row in rows] == ["directory", "lock"]
        assert all(row["closed"] and row["terminal"] and row["authority_yielded"] for row in rows)


@contextmanager
def _records_change_after_true_reuse(source, selected, hold, problem, monkeypatch):
    """Mutate only the existing reader at the real accepted-reuse return."""
    function = storage._derived_reuse
    code, namespace = function.__code__, function.__globals__
    guard = config_participants.companion_guard
    guard_code, guard_globals = guard.__code__, guard.__globals__
    original = bootstrap._records
    key = ("companions", str(selected))
    previous = sys.getprofile()
    phase, calls, fired, errors = ["enabled"], {"enabled": [], "fresh": []}, [], []

    def changed(root):
        parent = sys._getframe(1)
        if parent.f_code is guard_code and parent.f_globals is guard_globals:
            state = raw._states.get(parent.f_locals.get("operation"))
            if state is not None and state.source is source:
                calls[phase[0]].append(key not in hold.derived_evidence)
                raise problem
        return original(root)

    with monkeypatch.context() as patch:
        def observe(frame, event, result):
            try:
                if event == "return" and frame.f_code is code and not fired:
                    parent = frame.f_back
                    if (frame.f_globals is namespace and parent is not None
                        and parent.f_code is guard_code and parent.f_globals is guard_globals
                        and frame.f_locals.get("hold") is hold and frame.f_locals.get("key") == key
                        and type(result) is tuple and len(result) == 2 and result[0] is True):
                        state = raw._states.get(parent.f_locals.get("operation"))
                        assert state is not None and state.source is source and not state.active
                        assert parent.f_locals["operation"] in storage._raw_operations
                        assert not state.pins and not state.files and not state.descriptors
                        assert len(state.leases) == 1 and state.leases[0] in storage._live_leases
                        assert state.holds[-1] is hold and parent.f_locals["guard"]._exit_callbacks
                        assert result[1] == (None, None) and hold.derived_evidence[key][0].confirmed
                        patch.setattr(bootstrap, "_records", changed)
                        fired.append(True)
            except BaseException as error:
                errors.append(type(error).__name__)
            finally:
                if previous is not None:
                    previous(frame, event, result)

        sys.setprofile(observe)
        try:
            yield {"changed": changed, "phase": phase, "calls": calls, "fired": fired, "errors": errors}
        finally:
            sys.setprofile(previous)
    assert bootstrap._records is original
    assert storage._derived_reuse is function and function.__code__ is code
    assert function.__globals__ is namespace
    assert config_participants.companion_guard is guard and guard.__code__ is guard_code


def test_engaged_missing_source_change_after_true_reuse_keeps_fresh_error(
    configured_source, monkeypatch, request,
):
    source = configured_source
    _, selected, hold = missing._warm_missing(source)
    before = selected.read_bytes()
    original = bootstrap._records
    problem = LookupError("declared companion reader change after accepted reuse")
    with _temporary_guard_retirement(source, hold) as enabled_guard:
        with _records_change_after_true_reuse(source, selected, hold, problem, monkeypatch) as observed:
            enabled, enabled_native = review._operation_outcome(source, hold, problem)
    observed["phase"][0] = "fresh"
    with monkeypatch.context() as fresh:
        fresh.setattr(storage, "_EVIDENCE_REUSE", False)
        fresh.setattr(bootstrap, "_records", observed["changed"])
        with _temporary_guard_retirement(source, hold) as fresh_guard:
            disabled, disabled_native = review._operation_outcome(source, hold, problem)
    # Every operation has retired before compatibility or eviction is evaluated.
    assert not observed["errors"] and observed["fired"] == [True]
    assert bootstrap._records is original and selected.read_bytes() == before
    assert enabled_native.returned and disabled_native.returned
    assert disabled_native.closes == []
    for label, outcome, native, guards in (("enabled", enabled, enabled_native, enabled_guard),
                                            ("fresh", disabled, disabled_native, fresh_guard)):
        request.node.user_properties.append((label, json.dumps({
            "result": outcome[0], "error_type": outcome[1].__name__ if outcome[0] == "raised" else None,
            "same_error": outcome[2] if outcome[0] == "raised" else None,
            "returned_leases": len(native.returned), "positive_closes": len(native.closes),
            "reader_calls_key_evicted": observed["calls"][label],
            "temporary_guard_positive_closes": len(guards),
        }, sort_keys=True)))
    assert disabled == ("raised", LookupError, True) and len(observed["calls"]["fresh"]) == 1
    assert enabled == disabled
    assert observed["calls"]["enabled"] == [True]
    assert len(enabled_native.returned) == len(disabled_native.returned) == 1
    assert enabled_native.closes == disabled_native.closes == []
