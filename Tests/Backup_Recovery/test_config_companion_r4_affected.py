"""Unrun focused R4 publication and actual populated-generation controls."""
import contextlib
import errno
import json
import os
import sys
import time
from contextlib import contextmanager

import pytest

from Tests.Backup_Recovery import test_config_companion_missing_record_evidence as missing
from Tests.Backup_Recovery import test_config_companion_post_reuse_drift as causal
from Tests.Backup_Recovery import test_config_companion_review_regressions as review
from tldw_chatbook.Backup_Recovery import activation, bootstrap, config_participants, control_records
from tldw_chatbook.Backup_Recovery import generation_witnesses as generation
from tldw_chatbook.Backup_Recovery import native_files
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage

configured_source = review.configured_source
local_root = review.local_root
pytestmark = missing.pytestmark


@contextmanager
def _generation_retirement(selected, lease, store):
    """Observe only original paired-witness directory yields and their creator closes."""
    witness = generation._witnesses
    paired = generation._paired_witnesses
    private = activation._private.__wrapped__
    directory = native_files.pinned_directory.__wrapped__
    manager_type = contextlib._GeneratorContextManager
    enter = manager_type.__enter__
    paths = (store.root, store._generation("r4-generation"))
    previous, close = sys.getprofile(), os.close
    rows, pending, closing, calls, errors = [], {}, {}, [], []

    def in_witness(frame):
        for _ in range(16):
            if frame is None:
                return False
            if frame.f_code is witness.__code__ and frame.f_globals is witness.__globals__:
                return frame.f_locals.get("lease") is lease and frame.f_locals.get("path") == selected
            frame = frame.f_back
        return False

    def observe(frame, event, value):
        try:
            if not in_witness(frame):
                return
            if event == "call" and frame.f_code is paired.__code__ and frame.f_globals is paired.__globals__:
                calls.append(True)
            elif event == "return" and frame.f_code is directory.__code__ and type(value) is int:
                parent = frame.f_back
                if (parent is not None and parent.f_code is enter.__code__
                        and parent.f_globals is enter.__globals__):
                    manager = parent.f_locals.get("self")
                    if type(manager) is not manager_type:
                        return
                    generator = manager.gen
                    if generator.gi_frame is not frame or generator.gi_code is not directory.__code__:
                        return
                    parent = parent.f_back
                if (parent is None or parent.f_code is not private.__code__
                        or parent.f_globals is not private.__globals__
                        or frame.f_locals.get("root") not in paths):
                    return
                assert parent.f_locals["root"] == frame.f_locals["root"]
                assert value not in pending
                info = os.fstat(value)
                row = {"fd": value, "root": str(frame.f_locals["root"]),
                       "identity": (info.st_dev, info.st_ino), "creator": frame,
                       "private_yielded": False, "closed": False, "terminal": False}
                rows.append(row)
                pending[value] = row
            elif event == "return" and frame.f_code is private.__code__ and type(value) is int:
                if frame.f_locals.get("root") in paths:
                    assert value in pending
                    pending[value]["private_yielded"] = True
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
                        raise AssertionError("paired-witness directory live at creator close C_return")
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
        yield {"rows": rows, "paired_calls": calls}
    finally:
        sys.setprofile(previous)
        assert generation._witnesses is witness and generation._paired_witnesses is paired
        assert activation._private.__wrapped__ is private
        assert native_files.pinned_directory.__wrapped__ is directory and os.close is close
        assert not errors and not pending and not closing
        assert all(row["closed"] and row["terminal"] and row["private_yielded"] for row in rows)


def _directory_counts(observed, expected):
    assert observed["paired_calls"] == [True] * (expected // 2)
    assert len(observed["rows"]) == expected
    assert all(row["closed"] and row["terminal"] and row["private_yielded"] for row in observed["rows"])


@contextmanager
def _populated_generation(source, tmp_path):
    """Use the original supported publication/maintenance route, then a real admitted lease."""
    root, selected = bootstrap.default_bootstrap_root(), source.get_cli_config_path()
    old = storage._holds[(os.getpid(), str(root))]
    startup = storage._startups.pop(old.key)
    assert startup._key == old.key and old.count == 1
    startup.close()
    assert old.stop.is_set() and not old.thread.is_alive()
    assert old.key not in storage._holds and old not in storage._retiring_holds
    authority = control_records.admission_authority(root)
    data = source.get_user_data_dir()
    assert data.is_dir()
    authority.register("profile", (selected, data))
    control = tmp_path / "r4-operation"
    control.mkdir(mode=0o700)
    control_records.register_pending(root, "r4-restore", ("profile",), control, (selected,))
    with authority.maintenance(("profile",), 3) as session:
        activation.bind_activation(root, "r4-restore", selected, "r4-generation", ("config",), session=session)
    (root / ("pending-" + bootstrap._key("r4-restore") + ".json")).unlink()
    store = activation.ActivationStore(control / "activation")
    store.approve("r4-generation", "config")
    required = store._generation("r4-generation") / "required.json"
    expected = {"operation_id": "r4-restore", "generation": "r4-generation", "owners": ["config"],
                "namespaces": ["profile"], "store_root": str(store.root)}
    assert json.loads(required.read_bytes()) == {"version": 1, "generation": "r4-generation", "owners": ["config"]}
    assert bootstrap._records(root)[1][0]["activation"] == expected
    assert bootstrap._control_records(root)[2][0]["activation"] == expected
    newest = max(time.time_ns(), selected.stat().st_ctime_ns,
                 *(p.stat().st_ctime_ns for base in (root, control) for p in base.rglob("*")))
    missing._eventually(lambda: time.time_ns() - newest >= storage._EVIDENCE_SETTLE_NS)
    before = selected.read_bytes()
    lease = storage.acquire_storage(selected)
    hold = storage._ordinary_hold(lease=lease)
    assert hold is not old and hold.names == ("profile",)
    assert lease.execution_context(selected) == (root, ("profile",))
    try:
        yield selected, lease, hold, store, expected, required
    finally:
        lease.close()
        assert lease._key is None and lease not in storage._live_leases
        assert hold.stop.is_set() and not hold.thread.is_alive()
        assert hold.key not in storage._holds and hold not in storage._retiring_holds
        assert selected.read_bytes() == before and not raw._states and not storage._raw_operations


def _engaged_generation(selected, lease, hold, store, expected, request):
    key = ("witness", str(bootstrap.effective_config_path()), str(selected))
    with _generation_retirement(selected, lease, store) as cold:
        assert generation._witnesses(selected, lease) == [expected]
    _directory_counts(cold, 2)
    for _ in range(2):
        assert generation._witnesses(selected, lease) == [expected]
    with review._actual_reuses(generation._witnesses, hold, key) as reused:
        with _generation_retirement(selected, lease, store) as warm:
            assert generation._witnesses(selected, lease) == [expected]
    assert reused == [True] and hold.derived_evidence[key][0].confirmed
    _directory_counts(warm, 0)
    request.node.user_properties.append(("populated_engagement", json.dumps({
        "generation": expected["generation"], "owners": expected["owners"],
        "namespaces": expected["namespaces"], "actual_true_reuses": reused,
        "confirmed": hold.derived_evidence[key][0].confirmed,
        "cold_paired_calls": len(cold["paired_calls"]), "cold_positive_creator_closes": len(cold["rows"]),
        "warm_paired_calls": len(warm["paired_calls"]), "warm_positive_creator_closes": len(warm["rows"]),
    }, sort_keys=True)))
    return key


def _property(request, label, result, observed):
    request.node.user_properties.append((label, json.dumps({
        "result": result[0], "error_type": result[1].__name__ if result[0] == "raised" else None,
        "paired_calls": len(observed["paired_calls"]), "positive_creator_closes": len(observed["rows"]),
        "all_closed_terminal_private_yielded": all(row["closed"] and row["terminal"] and row["private_yielded"]
                                                  for row in observed["rows"]),
    }, sort_keys=True)))


def test_populated_generation_true_reuse_equals_original_fresh(configured_source, tmp_path, monkeypatch, request):
    with _populated_generation(configured_source, tmp_path) as (selected, lease, hold, store, expected, _):
        _engaged_generation(selected, lease, hold, store, expected, request)
        with _generation_retirement(selected, lease, store) as native:
            with monkeypatch.context() as fresh:
                fresh.setattr(storage, "_EVIDENCE_REUSE", False)
                result = review._outcome(lambda: generation._witnesses(selected, lease))
        _directory_counts(native, 2)
        _property(request, "fresh_populated", result, native)
        assert result == ("returned", [expected])


def test_engaged_populated_generation_required_damage_keeps_fresh_refusal(configured_source, tmp_path, monkeypatch, request):
    with _populated_generation(configured_source, tmp_path) as (selected, lease, hold, store, expected, required):
        _engaged_generation(selected, lease, hold, store, expected, request)
        original = required.read_bytes()
        try:
            required.write_bytes(b"{")
            with _generation_retirement(selected, lease, store) as enabled_native:
                enabled = review._outcome(lambda: generation._witnesses(selected, lease))
            with monkeypatch.context() as fresh:
                fresh.setattr(storage, "_EVIDENCE_REUSE", False)
                with _generation_retirement(selected, lease, store) as fresh_native:
                    disabled = review._outcome(lambda: generation._witnesses(selected, lease))
        finally:
            required.write_bytes(original)
        for label, result, native in (("enabled", enabled, enabled_native), ("fresh", disabled, fresh_native)):
            _directory_counts(native, 2)
            _property(request, label, result, native)
        assert enabled == disabled and enabled[0] == "raised" and issubclass(enabled[1], ValueError)


@pytest.mark.parametrize("change", ["replacement", "body"])
def test_engaged_populated_generation_required_callback_keeps_fresh_dispatch(
    configured_source, tmp_path, monkeypatch, request, change,
):
    with _populated_generation(configured_source, tmp_path) as (selected, lease, hold, store, expected, _):
        key = _engaged_generation(selected, lease, hold, store, expected, request)
        original = activation.ActivationStore._required
        problem, calls = LookupError("populated required reader customization"), []

        def changed(self, parent, generation_id):
            assert self.root == store.root and generation_id == "r4-generation"
            calls.append(key not in hold.derived_evidence)
            raise problem

        with monkeypatch.context() as patch:
            if change == "replacement":
                patch.setattr(activation.ActivationStore, "_required", changed)
            else:
                patch.setattr(original, "__code__", original.__code__.replace(co_consts=original.__code__.co_consts + ("r4 body marker",)))
            with _generation_retirement(selected, lease, store) as enabled_native:
                enabled = review._outcome(lambda: generation._witnesses(selected, lease), problem)
            patch.setattr(storage, "_EVIDENCE_REUSE", False)
            with _generation_retirement(selected, lease, store) as fresh_native:
                disabled = review._outcome(lambda: generation._witnesses(selected, lease), problem)
        for label, result, native in (("enabled", enabled, enabled_native), ("fresh", disabled, fresh_native)):
            _directory_counts(native, 2)
            _property(request, label, result, native)
        assert enabled == disabled
        if change == "replacement":
            assert enabled == ("raised", LookupError, True) and calls == [True, True]
        else:
            assert enabled == ("returned", [expected]) and not calls


@contextmanager
def _restoring_records_after_true_reuse(source, selected, hold, problem, monkeypatch):
    """Restore the customized original reader inside its genuine post-reuse dispatch."""
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
                patch.setattr(bootstrap, "_records", original)
                return original(root)
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


def test_restored_post_reuse_reader_cannot_republish_custom_fresh_result(configured_source, monkeypatch, request):
    source = configured_source
    _, selected, hold = missing._warm_missing(source)
    key, before = ("companions", str(selected)), selected.read_bytes()
    original = bootstrap._records
    with causal._temporary_guard_retirement(source, hold) as guards:
        with _restoring_records_after_true_reuse(source, selected, hold, None, monkeypatch) as observed:
            result, native = review._operation_outcome(source, hold)
    # Original operation and both temporary native guard creators retired first.
    review._two_fallbacks(native, hold)
    assert not observed["errors"] and observed["fired"] == [True]
    assert observed["calls"]["enabled"] == [True]
    assert result == ("returned", "retirement") and len(guards) == 2
    assert bootstrap._records is original and selected.read_bytes() == before
    request.node.user_properties.append(("restoring_reader", json.dumps({
        "actual_true_reuse_mutation": observed["fired"], "fresh_call_key_evicted": observed["calls"]["enabled"],
        "fallback_leases": len(native.returned), "positive_raw_closes": len(native.closes),
        "temporary_guard_positive_closes": len(guards), "key_absent_after_custom_return": key not in hold.derived_evidence,
    }, sort_keys=True)))
    assert key not in hold.derived_evidence
