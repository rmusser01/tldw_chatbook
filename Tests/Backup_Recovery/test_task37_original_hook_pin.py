"""TASK-34563.37: raw pins must not collect unpublishable derived evidence."""

from collections import Counter
from contextlib import contextmanager
import errno
import inspect
import json
import os
import sys
import threading
from types import SimpleNamespace

import pytest

from Tests.Backup_Recovery import test_raw_native_retirement as retirement_cases
from Tests.Performance.console_storage_unit_observer import OriginalStorageUnitObserver
from tldw_chatbook.Agents import hook_permissions
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Utils import private_paths, windows_files

configured_source = retirement_cases.configured_source
local_root = retirement_cases.local_root


def _custody():
    with storage._lock:
        return {
            name: frozenset(getattr(storage, name))
            for name in (
                "_live_leases",
                "_pending_acquisitions",
                "_raw_operations",
                "_operations",
                "_retiring_holds",
            )
        } | {"raw_states": frozenset(raw._states)}


@contextmanager
def _original_hook_pin_work(owner, config):
    """Observe only original bodies on this actor beneath this hook owner."""
    witness = OriginalStorageUnitObserver({}, lambda: True, lambda unit: None)
    members = (
        (raw, "_pin_parent"),
        (raw, "_check"),
        (raw, "_check_parent_pins"),
        (raw, "_close_descriptor"),
        (storage, "_path_evidence"),
        (storage, "_note_derived"),
        (storage, "_ordinary_hold"),
        (storage, "_acquire_storage"),
        (private_paths, "_open_verified_parent"),
        (windows_files.WindowsOS, "stat_many_for_admission"),
        (hook_permissions.HookPermissions, "snapshot"),
        (hook_permissions.HookPermissions, "close"),
    )
    functions = {name: inspect.getattr_static(target, name) for target, name in members}
    for target, name in members:
        function = functions[name]
        witness._pin(function)
        witness.slots.append((target, name, function))
        witness.codes[function.__code__] = name
    current = inspect.getattr_static(hook_permissions.HookPermissions, "_current")
    witness._pin(current)
    root_code = witness._pin(current.__wrapped__)
    witness.slots.extend(
        (
            (hook_permissions.HookPermissions, "_current", current),
            (current, "__wrapped__", current.__wrapped__),
        )
    )
    native_open = inspect.getattr_static(windows_files._Native, "open_handle")
    native_code = witness._pin(native_open)
    witness.slots.append((windows_files._Native, "open_handle", native_open))
    codes = {name: function.__code__ for name, function in functions.items()}
    actor = threading.current_thread()
    counts, pins, active_pins, closes, failures = Counter(), [], {}, {}, []
    observed = SimpleNamespace(
        counts=counts, pins=pins, failures=failures, receipt=None
    )

    def ancestry(frame):
        pin = evidence = admission = None
        for _ in range(64):
            if frame is None:
                return None
            if pin is None and frame.f_code is codes["_pin_parent"]:
                pin = frame
            if evidence is None and frame.f_code is codes["_path_evidence"]:
                evidence = frame
            if admission is None and frame.f_code is codes["_acquire_storage"]:
                admission = frame
            if frame.f_code is root_code and frame.f_locals.get("self") is owner:
                return pin, evidence, admission
            frame = frame.f_back
        raise AssertionError("hook observation ancestry exceeded its bound")

    def started(code, offset):
        if threading.current_thread() is not actor:
            return
        try:
            frame = witness._frame(code)
            if code is native_code:
                context = ancestry(frame)
                if context is not None:
                    pin, evidence, admission = context
                    label = (
                        "pin_evidence"
                        if pin is not None and evidence is not None
                        else (
                            "admission_evidence"
                            if evidence is not None
                            else "admission"
                            if admission is not None
                            else "other"
                        )
                    )
                    counts["native_" + label] += 1
                return
            witness._start(code, offset)
            if code is codes["close"] and frame.f_locals.get("self") is owner:
                counts["owner_close"] += 1
            context = ancestry(frame)
            if context is None:
                return
            counts[witness.codes[code]] += 1
            pin, evidence, _ = context
            if code is codes["_pin_parent"]:
                assert len(pins) < 16
                state = frame.f_locals["state"]
                assert state.source is owner or state.source is config
                operation = next(
                    op for op, value in raw._states.items() if value is state
                )
                row = dict(
                    state=state,
                    operation=operation,
                    leases=tuple(state.leases),
                    holds=tuple(state.holds),
                    anchor=frame.f_locals["anchor"],
                    evidence=[],
                    notes=[],
                    eligibility_rejections=0,
                    walks=0,
                    fd=None,
                    retired=False,
                )
                pins.append(row)
                active_pins[id(frame)] = row
            elif code is codes["_path_evidence"]:
                counts[
                    "pin_evidence" if pin is not None else "admission_path_evidence"
                ] += 1
                if pin is not None:
                    assert frame.f_back is pin
            elif code is codes["_open_verified_parent"] and pin is not None:
                assert frame.f_back is pin
                active_pins[id(pin)]["walks"] += 1
            elif code is codes["_note_derived"] and pin is not None:
                row = active_pins[id(pin)]
                hold, key = frame.f_locals["hold"], frame.f_locals["key"]
                assert key == ("raw-pin", str(row["anchor"]))
                assert hold is None or any(hold is value for value in row["holds"])
                assert frame.f_back is pin
                assert frame.f_locals["entry"] is pin.f_locals["evidence"]
                previous = hold.derived_evidence.get(key) if hold is not None else None
                row["notes"].append((hold, key, frame.f_locals["entry"], previous))
            elif code is codes["_close_descriptor"]:
                state, fd = frame.f_locals["state"], frame.f_locals["fd"]
                if state.source is owner or state.source is config:
                    closes[id(frame)] = state, fd
        except BaseException as error:
            failures.append("start:" + type(error).__name__ + ":" + str(error))

    def returned(code, offset, value):
        if threading.current_thread() is not actor:
            return
        try:
            frame = witness._frame(code)
            context = ancestry(frame)
            if context is not None:
                pin, _, _ = context
                if code is codes["_path_evidence"] and pin is not None:
                    active_pins[id(pin)]["evidence"].append(value)
                elif code is codes["_ordinary_hold"] and pin is not None:
                    assert (
                        value is None
                    ), "Selected derived reuse must remain ineligible"
                    if frame.f_back.f_code is codes["_note_derived"]:
                        active_pins[id(pin)]["eligibility_rejections"] += 1
                        counts["publisher_eligibility_rejections"] += 1
                    elif frame.f_back is pin:
                        counts["producer_eligibility_rejections"] += 1
                elif code is codes["_note_derived"] and pin is not None:
                    row = active_pins[id(pin)]
                    hold, key, entry, previous = row["notes"][-1]
                    assert value is None
                    assert hold is None or hold.derived_evidence.get(key) is previous
                    if not row["evidence"]:
                        assert entry is None
                        counts["pin_collection_skipped"] += 1
                    else:
                        assert len(row["evidence"]) == 1
                        collected = row["evidence"][0]
                        if entry is collected:
                            counts["pin_evidence_forwarded"] += 1
                        else:
                            # The original pin can discard optional metadata before
                            # the publisher; prove its actual posture rejection.
                            assert entry is None and collected is not None
                            assert collected.posture[-1][1] != pin.f_locals["posture"]
                            counts["pin_evidence_discarded_by_posture"] += 1
                    counts["rejected_pin_publications"] += 1
                elif code is codes["_pin_parent"]:
                    row = active_pins.pop(id(frame))
                    row["fd"] = value
                    os.fstat(value)  # The real returned descriptor is open now.
                elif code is codes["stat_many_for_admission"]:
                    assert not frame.f_locals["opened"] and not frame.f_locals["failed"]
                    counts["retired_metadata_snapshots"] += 1
                elif code is codes["_close_descriptor"] and id(frame) in closes:
                    state, fd = closes.pop(id(frame))
                    try:
                        os.fstat(fd)
                    except OSError as error:
                        assert error.errno == errno.EBADF
                    else:
                        raise AssertionError(
                            "original close did not physically retire FD"
                        )
                    assert fd not in state.descriptors
                    counts["physical_closes"] += 1
                    for row in pins:
                        if row["state"] is state and row["fd"] == fd:
                            row["retired"] = True
        except BaseException as error:
            failures.append("return:" + type(error).__name__ + ":" + str(error))
        finally:
            witness._return(code, offset, value)

    monitor = sys.monitoring
    tool = next(
        slot
        for slot in range(5, 0, -1)
        if slot != monitor.DEBUGGER_ID and monitor.get_tool(slot) is None
    )
    monitor.use_tool_id(tool, "original-hook-pin-evidence")
    witness.tool, witness.active, witness.installed = tool, True, True
    # Native allocation attempts include expected EEXIST; count starts only.
    witness.codes[native_code] = "native_open_attempt"
    witness.registered = {
        monitor.events.PY_START: started,
        monitor.events.PY_RETURN: returned,
    }
    try:
        for event, callback in witness.registered.items():
            assert monitor.register_callback(tool, event, callback) is None
        for code in witness.codes:
            mask = monitor.events.PY_START
            if code is not native_code:
                mask |= monitor.events.PY_RETURN
            monitor.set_local_events(tool, code, mask)
        assert monitor.get_events(tool) == 0
        yield observed
    finally:
        observed.receipt = witness.close()
        assert monitor.get_tool(tool) is None
        assert not active_pins and not closes
        assert not failures, failures
        assert observed.receipt["complete"], observed.receipt


@pytest.mark.parametrize(
    "reuse_enabled",
    [
        pytest.param(
            True,
            id="windows-default",
            marks=pytest.mark.skipif(
                os.name != "nt", reason="Actual Windows eligibility"
            ),
        ),
        pytest.param(False, id="reuse-disabled"),
    ],
)
def test_warm_hook_snapshot_skips_unpublishable_pin_evidence(
    configured_source, monkeypatch, request, reuse_enabled
):
    if not reuse_enabled:
        monkeypatch.setattr(storage, "_EVIDENCE_REUSE", False)
    else:
        assert storage._EVIDENCE_REUSE is True
    monkeypatch.setattr(hook_permissions, "config", configured_source)
    owner = hook_permissions.HookPermissions()
    try:
        first, warm = owner.snapshot(), owner.snapshot()
        assert first.ready and warm.ready and warm.rows == ()
        config_path = configured_source._get_effective_config_path()
        store_path = warm.store_path
        before_bytes = config_path.read_bytes(), store_path.read_bytes()
        baseline = _custody()
        with _original_hook_pin_work(owner, configured_source) as observed:
            try:
                result = owner.snapshot()
                after = _custody()
            finally:
                owner.close()
        assert owner._closed.is_set()
        assert baseline == after == _custody()
        assert result == warm
        assert (config_path.read_bytes(), store_path.read_bytes()) == before_bytes
        assert observed.counts["owner_close"] == 1
        assert {row["state"].source for row in observed.pins} == {
            owner,
            configured_source,
        }
        for row in observed.pins:
            state = row["state"]
            assert row["walks"] == 1 and row["retired"]
            assert row["operation"] not in raw._states
            assert row["operation"] not in storage._raw_operations
            assert not state.active and not state.uncertain
            assert not state.pins and not state.files and not state.descriptors
            assert all(lease not in storage._live_leases for lease in row["leases"])
            assert len(row["notes"]) == 1
            if row["notes"][0][0] is not None:
                assert row["eligibility_rejections"] == 1
        counts = observed.counts
        assert counts["_check"] > 0 and counts["_check_parent_pins"] > 0
        assert counts["_acquire_storage"] > 0
        if os.name == "nt":
            assert counts["native_admission"] + counts["native_admission_evidence"] > 0
            assert counts["retired_metadata_snapshots"] > 0
        assert counts["physical_closes"] > 0
        assert counts["rejected_pin_publications"] == len(observed.pins)
        request.node.user_properties.append(
            (
                "original_raw_pin_evidence",
                json.dumps(
                    {
                        "counts": dict(counts),
                        "reuse_enabled": reuse_enabled,
                        "windows_native_counter": os.name == "nt",
                        "pins": len(observed.pins),
                        "same_issued_custody": True,
                        "pins_physically_retired": True,
                        "owner_closed": True,
                        "source_current": observed.receipt["original_source_current"],
                        "monitor_retired": observed.receipt[
                            "hooks_retired_before_inactive"
                        ],
                    },
                    sort_keys=True,
                ),
            )
        )
        # Causal RED comes only after actual payload, source, pin and cleanup checks.
        assert counts["pin_evidence"] == 0, dict(counts)
        if os.name == "nt":
            assert counts["native_pin_evidence"] == 0, dict(counts)
    finally:
        if not owner._closed.is_set():
            owner.close()
