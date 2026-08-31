"""ADR-080: fire-once trigger matrix + guarded delivery, no real threads."""

import threading

from tldw_chatbook.Chat import console_chat_controller as ccc
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.permission_summary_service import (
    PermissionSummaryResolution,
)


def _bare_controller():
    ctrl = object.__new__(ConsoleChatController)
    ctrl._pending_approval_rounds = {}
    ctrl._approval_state_lock = threading.Lock()
    ctrl.app = None
    ctrl.update_pending_approval_summary = None
    return ctrl


def _payload(rationales=("why",)):
    return {
        "round_id": "r1",
        "session_id": "s1",
        "calls": [{"llm_name": "t", "rationale": r} for r in rationales],
        "summary": None,
    }


def _resolution(mode, active=True):
    return PermissionSummaryResolution(mode=mode, active=active)


class _ThreadStub:
    started = []

    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        _ThreadStub.started.append(True)


def _armed(monkeypatch, mode, active=True):
    monkeypatch.setattr(
        ccc, "resolve_permission_summary", lambda cfg: _resolution(mode, active)
    )
    _ThreadStub.started = []
    monkeypatch.setattr(ccc.threading, "Thread", _ThreadStub)


def test_mode_off_never_fires(monkeypatch):
    _armed(monkeypatch, "off", active=True)
    ctrl = _bare_controller()
    ctrl._pending_approval_rounds["r1"] = {
        "event": threading.Event(), "summary_fired": False,
    }
    ctrl._maybe_fire_permission_summary(_payload())
    assert _ThreadStub.started == []
    assert ctrl._pending_approval_rounds["r1"]["summary_fired"] is True


def test_fallback_fires_only_when_a_rationale_is_missing(monkeypatch):
    _armed(monkeypatch, "fallback")
    ctrl = _bare_controller()
    ctrl._pending_approval_rounds["r1"] = {
        "event": threading.Event(), "summary_fired": False,
    }
    ctrl._maybe_fire_permission_summary(_payload(rationales=("why", "also why")))
    assert _ThreadStub.started == []  # every row explained: no call
    ctrl._maybe_fire_permission_summary(_payload(rationales=("why", "")))
    # first fire consumed the once-flag... but it was marked fired above
    # (no-call also counts as fired) -- so this must NOT start one either.
    assert _ThreadStub.started == []


def test_fallback_fires_when_missing_and_always_fires(monkeypatch):
    for mode, rationales in (("fallback", ("",)), ("always", ("why",))):
        _armed(monkeypatch, mode)
        ctrl = _bare_controller()
        ctrl._pending_approval_rounds["r1"] = {
            "event": threading.Event(), "summary_fired": False,
        }
        ctrl._maybe_fire_permission_summary(_payload(rationales))
        assert _ThreadStub.started == [True], mode


def test_delivery_drops_resolved_rounds_and_updates_live_ones():
    ctrl = _bare_controller()
    resolved = threading.Event()
    resolved.set()
    ctrl._pending_approval_rounds["r1"] = {"event": resolved}
    payload = _payload()
    seen = []
    ctrl.update_pending_approval_summary = lambda rid, text: seen.append((rid, text))
    ctrl._deliver_permission_summary("r1", payload, "sum")
    assert seen == [] and payload["summary"] is None  # dropped

    live_event = threading.Event()
    ctrl._pending_approval_rounds["r2"] = {"event": live_event}
    payload2 = _payload()
    payload2["round_id"] = "r2"
    ctrl._deliver_permission_summary("r2", payload2, "sum")
    assert payload2["summary"] == "sum"
    assert ctrl._pending_approval_rounds["r2"]["summary"] == "sum"
    assert seen == [("r2", "sum")]
