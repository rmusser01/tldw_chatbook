"""ADR-090: fire-once trigger matrix + guarded delivery, no real threads."""

import builtins

import threading
from types import SimpleNamespace

from tldw_chatbook.Chat import console_chat_controller as ccc
from tldw_chatbook.Chat import permission_summary_service as summary_service
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_interrupt_rounds import InterruptRoundHost
from tldw_chatbook.Chat.permission_summary_service import (
    PermissionSummaryResolution,
)


def _bare_controller():
    ctrl = object.__new__(ConsoleChatController)
    ctrl._pending_approval_rounds = {}
    ctrl._approval_state_lock = threading.Lock()
    # task-31384: session activation re-derives the non-approval kinds
    # through the host; a real one over this bare double stays inert
    # (no app, no setters) while the approval path under test is untouched.
    ctrl._interrupt_host = InterruptRoundHost(ctrl)
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
        summary_service,
        "resolve_permission_summary",
        lambda cfg: _resolution(mode, active),
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


def test_summary_service_import_failure_is_advisory(monkeypatch):
    ctrl = _bare_controller()
    ctrl._pending_approval_rounds["r1"] = {
        "event": threading.Event(), "summary_fired": False,
    }
    real_import = builtins.__import__

    def fail_summary_service_import(name, *args, **kwargs):
        if name == "tldw_chatbook.Chat.permission_summary_service":
            raise ImportError("summary service unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_summary_service_import)

    ctrl._maybe_fire_permission_summary(_payload())

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


def _parked_controller(monkeypatch, mode, round_id="r1"):
    """A bare controller whose round is PARKED (stored, never marshalled)."""
    _armed(monkeypatch, mode)
    ctrl = _bare_controller()
    ctrl._pending_approval_rounds[round_id] = {
        "event": threading.Event(), "summary_fired": False,
    }
    payload = _payload()
    payload["round_id"] = round_id
    ctrl._parked_approval_payloads = {round_id: payload}
    mounted = []
    ctrl.set_pending_approval = mounted.append
    ctrl.store = SimpleNamespace(active_session_id="s1")
    return ctrl, payload, mounted


def test_parked_round_fires_once_on_attach_remount(monkeypatch):
    ctrl, payload, mounted = _parked_controller(monkeypatch, "always")
    assert ctrl.remount_pending_approval_for_active_session() is True
    assert mounted == [payload]
    assert _ThreadStub.started == [True]  # never marshalled before: fires now
    # A later re-attach (second Console view attach) re-mounts but must
    # NOT re-fire the summarizer.
    assert ctrl.remount_pending_approval_for_active_session() is True
    assert mounted == [payload, payload]
    assert _ThreadStub.started == [True]
    assert ctrl._pending_approval_rounds["r1"]["summary_fired"] is True


def test_consumed_flag_prevents_refire_on_attach_remount(monkeypatch):
    ctrl, _payload_, mounted = _parked_controller(monkeypatch, "always")
    ctrl._pending_approval_rounds["r1"]["summary_fired"] = True  # consumed
    assert ctrl.remount_pending_approval_for_active_session() is True
    assert len(mounted) == 1
    assert _ThreadStub.started == []


def test_switch_session_promotes_parked_round_and_fires_once(monkeypatch):
    ctrl, payload, mounted = _parked_controller(monkeypatch, "always")
    ctrl.store.active_session_id = "s0"  # switching away from s0 onto s1
    ctrl.store.switch_session = lambda session_id: SimpleNamespace(id=session_id)
    ctrl.mark_session_visited = lambda session_id: None
    ctrl._clear_terminal_run_state = lambda **_: None
    ctrl._remount_parked_skill_install = lambda session_id: None
    ctrl._remount_parked_skill_script = lambda session_id: None
    ctrl.switch_session("s1")
    assert mounted == [payload]
    assert _ThreadStub.started == [True]
    # Switching away and back re-mounts unchanged, never re-fires.
    ctrl.switch_session("s0")
    assert mounted == [payload, None]
    assert _ThreadStub.started == [True]


def test_remount_head_helper_fires_promoted_sibling_once(monkeypatch):
    # Teardown/revocation promotion: the resolved head was unparked and its
    # round popped; `_remount_head` promotes the queued sibling r2.
    ctrl, _head, mounted = _parked_controller(monkeypatch, "always")
    parked = ctrl._parked_approval_payloads
    parked["r2"] = dict(parked["r1"], round_id="r2")
    parked.pop("r1")
    ctrl._pending_approval_rounds.pop("r1")
    ctrl._pending_approval_rounds["r2"] = {
        "event": threading.Event(), "summary_fired": False,
    }

    class _InlineApp:
        @staticmethod
        def call_from_thread(fn, *args):
            fn(*args)  # run the UI-thread callback inline

    ctrl.app = _InlineApp()
    ctrl._remount_head(parked, ctrl.set_pending_approval, "s1")
    assert mounted == [parked["r2"]]
    assert _ThreadStub.started == [True]
    # A second promotion pass (e.g. another sibling resolving) must not
    # re-fire for r2: its once-flag was consumed above.
    ctrl._remount_head(parked, ctrl.set_pending_approval, "s1")
    assert mounted == [parked["r2"], parked["r2"]]
    assert _ThreadStub.started == [True]


def test_remount_head_helper_skips_non_mcp_payloads(monkeypatch):
    # The skill bridges share `_remount_head`; their payloads' round ids
    # are unknown to `_pending_approval_rounds`, so nothing may fire.
    ctrl, _payload_, _mounted = _parked_controller(monkeypatch, "always")
    skill_payload = {"round_id": "skill-1", "session_id": "s1", "calls": []}
    parked = {"skill-1": skill_payload}
    ctrl._pending_approval_rounds.pop("r1")  # no MCP round armed at all

    class _InlineApp:
        @staticmethod
        def call_from_thread(fn, *args):
            fn(*args)

    ctrl.app = _InlineApp()
    mounted = []
    ctrl.set_pending_approval = mounted.append
    ctrl._remount_head(parked, ctrl.set_pending_approval, "s1")
    assert mounted == [skill_payload]
    assert _ThreadStub.started == []
