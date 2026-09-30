"""Moved Hook bodies preserve canonical class binding and live patch seams."""

from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat import console_chat_controller as controller_owner
from tldw_chatbook.Chat import console_run_hooks
from tldw_chatbook.Chat.console_interrupt_rounds import _sibling_approval_refusals
from tldw_chatbook.UI.Console_Modules import hooks, wiring
from tldw_chatbook.UI.Screens import chat_screen as screen_owner

pytestmark = pytest.mark.bootstrap_profile


def test_moved_hook_bodies_keep_canonical_class_and_bound_methods(monkeypatch):
    controller = controller_owner.ConsoleChatController
    screen = screen_owner.ChatScreen
    for name in (
        "_hook_admission_reason",
        "_notify_run_hook_approval",
        "_run_hooks_engine",
    ):
        assert getattr(controller, name) is getattr(console_run_hooks, name)
    for name in (
        "_request_console_hooks_review",
        "_open_console_hooks_review",
        "_refresh_console_hooks",
        "_apply_console_hooks_state",
        "_dispatch_console_draft_send",
    ):
        assert getattr(screen, name) is getattr(hooks, name)
    assert (
        screen._build_console_workbench_state
        is wiring.build_console_workbench_projection
    )
    marker = object()
    monkeypatch.setattr(screen_owner, "Button", marker)
    calls = []
    button = SimpleNamespace(set_class=lambda *args: calls.append(args))
    view = SimpleNamespace(
        _sync_console_control_bar=lambda: None,
        query_one=lambda selector, kind: button if kind is marker else None,
    )
    screen._apply_console_hooks_state.__get__(view)(SimpleNamespace(pending_count=1))
    assert calls == [(True, "hooks-attention")]


def test_sibling_refusal_uses_the_live_canonical_review_decision(monkeypatch):
    assert controller_owner._sibling_approval_refusals is _sibling_approval_refusals
    rows = [
        SimpleNamespace(call_id="approved", llm_name="same"),
        SimpleNamespace(call_id="unanswered", llm_name="same"),
    ]
    decisions = {"approved": "approve_once"}
    marker = object()
    calls = []
    monkeypatch.setattr(controller_owner, "_review_decision", lambda *args: marker)
    result = _sibling_approval_refusals(
        rows,
        lambda row: decisions.get(row.call_id),
        decisions,
        lambda row: ("approve_once",),
        lambda row, timed_out: calls.append((row.call_id, timed_out)),
    )
    assert result == {"unanswered": marker}
    assert calls == [("unanswered", False)]


def test_resize_tolerates_a_partial_workspace_grid():
    from textual.events import Resize
    from textual.geometry import Size

    calls = []

    def query(selector):
        calls.append(selector)
        if selector == "#console-workspace-grid":
            return object()
        raise screen_owner.QueryError("rail is not mounted")

    view = SimpleNamespace(
        _last_console_workspace_width_band=None,
        app=SimpleNamespace(focused=None),
        query_one=query,
    )
    screen_owner.ChatScreen._adapt_console_workspace_to_width(
        view, Resize(Size(120, 40), Size(120, 40))
    )
    assert calls == ["#console-workspace-grid", "#console-left-rail"]
