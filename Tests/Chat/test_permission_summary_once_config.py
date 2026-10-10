"""AC26: live summary configuration is read only for an unconsumed round."""

import threading
from types import SimpleNamespace

import pytest

from Tests.Chat.test_permission_summary_wiring import (
    _ThreadStub,
    _armed,
    _bare_controller,
    _parked_controller,
    _payload,
    _resolution,
)
from tldw_chatbook.Chat import console_chat_controller as ccc
from tldw_chatbook.Chat import permission_summary_service as summary_service


def test_consumed_round_remount_never_reads_configuration(monkeypatch):
    ctrl, payload, mounted = _parked_controller(monkeypatch, "always")
    ctrl._pending_approval_rounds["r1"]["summary_fired"] = True
    reads = []

    def snapshot():
        reads.append(True)
        return SimpleNamespace(values={})

    # Install after construction: the original host accessor stays live.
    monkeypatch.setattr(ccc, "get_runtime_config_snapshot", snapshot)
    assert ctrl.remount_pending_approval_for_active_session() is True
    assert ctrl.remount_pending_approval_for_active_session() is True
    assert mounted == [payload, payload]
    assert _ThreadStub.started == []
    assert reads == []


def test_resolved_round_stale_payload_never_reads_configuration(monkeypatch):
    _armed(monkeypatch, "always")
    ctrl = _bare_controller()
    reads = []
    monkeypatch.setattr(
        ccc,
        "get_runtime_config_snapshot",
        lambda: (reads.append(True), SimpleNamespace(values={}))[1],
    )
    ctrl._maybe_fire_permission_summary(_payload())
    assert _ThreadStub.started == []
    assert reads == []


def test_first_no_call_consumes_round_without_another_config_read(monkeypatch):
    ctrl, payload, mounted = _parked_controller(monkeypatch, "fallback")
    reads = []

    def snapshot():
        acquired = ctrl._approval_state_lock.acquire(blocking=False)
        assert acquired, "fresh config read held the non-reentrant approval lock"
        ctrl._approval_state_lock.release()
        reads.append(True)
        return SimpleNamespace(values={})

    monkeypatch.setattr(ccc, "get_runtime_config_snapshot", snapshot)
    assert ctrl.remount_pending_approval_for_active_session() is True
    assert ctrl._pending_approval_rounds["r1"]["summary_fired"] is True
    assert ctrl.remount_pending_approval_for_active_session() is True
    assert mounted == [payload, payload]
    assert _ThreadStub.started == []
    assert reads == [True]


def test_each_new_round_reads_current_configuration(monkeypatch):
    _armed(monkeypatch, "always")
    ctrl = _bare_controller()
    current = {"mode": "off"}
    reads = []
    resolutions = []

    def snapshot():
        acquired = ctrl._approval_state_lock.acquire(blocking=False)
        assert acquired
        ctrl._approval_state_lock.release()
        values = dict(current)
        reads.append(values)
        return SimpleNamespace(values=values)

    def resolve(values):
        resolutions.append(values)
        return _resolution(values["mode"], active=values["mode"] != "off")

    monkeypatch.setattr(ccc, "get_runtime_config_snapshot", snapshot)
    monkeypatch.setattr(summary_service, "resolve_permission_summary", resolve)
    ctrl._pending_approval_rounds["r1"] = {
        "event": threading.Event(),
        "summary_fired": False,
    }
    ctrl._maybe_fire_permission_summary(_payload())
    assert _ThreadStub.started == []
    current["mode"] = "always"
    ctrl._pending_approval_rounds["r2"] = {
        "event": threading.Event(),
        "summary_fired": False,
    }
    payload = dict(_payload(), round_id="r2")
    ctrl._maybe_fire_permission_summary(payload)
    assert _ThreadStub.started == [True]
    assert reads == resolutions == [{"mode": "off"}, {"mode": "always"}]


@pytest.mark.parametrize("changed", ["consumed", "resolved"])
def test_final_locked_check_rejects_round_changed_during_config_read(
    monkeypatch, changed
):
    _armed(monkeypatch, "always")
    ctrl = _bare_controller()
    ctrl._pending_approval_rounds["r1"] = {
        "event": threading.Event(),
        "summary_fired": False,
    }
    reads = []

    def snapshot():
        acquired = ctrl._approval_state_lock.acquire(blocking=False)
        assert acquired, "fresh config read held the non-reentrant approval lock"
        try:
            if changed == "resolved":
                ctrl._pending_approval_rounds.pop("r1")
            else:
                ctrl._pending_approval_rounds["r1"]["summary_fired"] = True
        finally:
            ctrl._approval_state_lock.release()
        reads.append(True)
        return SimpleNamespace(values={})

    monkeypatch.setattr(ccc, "get_runtime_config_snapshot", snapshot)
    ctrl._maybe_fire_permission_summary(_payload())
    assert reads == [True]
    assert _ThreadStub.started == []
