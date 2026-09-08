"""Switcher decisions without a Textual app or persistence services."""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from textual.containers import Vertical
from textual.widgets import Button, Input

from tldw_chatbook.Chat.console_switcher_state import SwitcherMode
from tldw_chatbook.UI.Console_Modules import conversation_token_preparation, workspace
from tldw_chatbook.Widgets.Console.console_session_switcher_modal import (
    ConsoleSessionSwitcherModal,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("active, expected", [("other", "first"), ("second", "second")])
@pytest.mark.parametrize("missing", [False, True])
async def test_reuse_validates_saved_identity_before_selecting_runtime(
    monkeypatch, active, expected, missing
):
    # A runtime whose ID equals the saved ID is deliberately unrelated.
    sessions = [
        SimpleNamespace(id="native:other", persisted_conversation_id="unrelated"),
        SimpleNamespace(id="first", persisted_conversation_id="native:other"),
        SimpleNamespace(id="second", persisted_conversation_id="native:other"),
    ]
    store = SimpleNamespace(active_session_id=active, sessions=lambda: sessions)

    async def load(_app, target):
        assert target == "native:other"
        return None if missing else {"conversation": {"id": target}, "messages": []}

    async def open_runtime(target):
        store.active_session_id = target.removeprefix("native:")
        return True

    owner = SimpleNamespace(
        app_instance=SimpleNamespace(notify=Mock()),
        _ensure_console_chat_store=lambda: store,
        _capture_console_draft_switch_snapshot=lambda: None,
        _restore_console_session_after_failed_open=AsyncMock(),
        open_console_workspace_conversation=open_runtime,
    )
    monkeypatch.setattr(workspace, "load_console_conversation_tree", load)
    result = await workspace.ConsoleWorkspaceController._resume_console_workspace_conversation(
        owner, "native:other", reuse_existing=True
    )
    assert result is (not missing)
    assert store.active_session_id == (active if missing else expected)


@pytest.mark.parametrize(
    "mode, widened, expected",
    [
        (SwitcherMode.ACTIVE, False, "Active"),
        (SwitcherMode.ACTIVE, True, "History"),
        (SwitcherMode.HISTORY, False, "History"),
        (SwitcherMode.CHARACTER_CHATS, False, "Character chats"),
    ],
)
def test_mode_title_and_selected_class_agree_without_app(mode, widened, expected):
    widgets = {
        "#console-switcher-active-mode": Button(),
        "#console-switcher-history-mode": Button(),
        "#console-switcher-character-mode": Button(),
        "#console-switcher-query": Input(),
        "#console-switcher-modal": Vertical(),
    }
    owner = SimpleNamespace(
        query_one=lambda selector, _kind: widgets[selector],
        _active_results=(),
        _mode=mode,
        _widened_to_history=widened,
        _activation_in_flight=False,
    )
    ConsoleSessionSwitcherModal._update_mode_controls(owner)
    assert f"[{expected}]" in widgets["#console-switcher-modal"].border_title
    selected = [
        str(widget.label)
        for widget in widgets.values()
        if isinstance(widget, Button)
        and widget.has_class("console-switcher-mode-current")
    ]
    assert selected == ["Active (0)" if expected == "Active" else expected]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["scope", "cancel", "presentation"])
async def test_warm_open_scope_error_is_nonfatal_but_activation_failures_restore(
    monkeypatch, failure
):
    target = SimpleNamespace(id="target")
    store = SimpleNamespace(active_session_id="prior", sessions=lambda: [target])
    shown = []
    notices = []

    def switch(session_id):
        store.active_session_id = session_id

    async def refresh(_session):
        if failure == "scope":
            raise RuntimeError("synthetic scope failure")
        if failure == "cancel":
            raise asyncio.CancelledError()

    async def present():
        if failure == "presentation":
            raise RuntimeError("synthetic presentation failure")
        shown.append(store.active_session_id)

    async def restore(_store, prior):
        store.active_session_id = prior

    owner = SimpleNamespace(
        _screen=None,
        app_instance=SimpleNamespace(
            notify=lambda *args, **kwargs: notices.append(args)
        ),
        _console_session_id_for_workspace_conversation=lambda _id: "target",
        _ensure_chat_controller_fn=lambda: SimpleNamespace(
            store=store, switch_session=switch
        ),
        _capture_console_draft_switch_snapshot=lambda: None,
        _set_active_workspace_for_console_session=lambda _id: None,
        _refresh_console_effective_scope_and_sync=refresh,
        _sync_console_chat_core_state=lambda: None,
        _sync_native_console_chat_ui_fn=present,
        _sync_temporary_chip_fn=lambda: None,
        _focus_composer_if_needed_fn=lambda **kwargs: None,
        _refresh_console_conversation_browser_after_selection=AsyncMock(),
        _restore_console_session_after_failed_open=restore,
    )
    monkeypatch.setattr(
        conversation_token_preparation, "prepare_conversation_tokens", AsyncMock()
    )
    open_conversation = (
        workspace.ConsoleWorkspaceController.open_console_workspace_conversation
    )
    if failure == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await open_conversation(owner, "native:target")
    else:
        result = await open_conversation(owner, "native:target")
        assert result is (True if failure == "scope" else None)
    assert store.active_session_id == ("target" if failure == "scope" else "prior")
    assert shown == (["target"] if failure == "scope" else [])
    assert bool(notices) is (failure == "presentation")


@pytest.mark.asyncio
async def test_queued_reconciliation_does_not_allocate_coroutine_before_start():
    queued = []
    refreshed = []

    async def refresh(query):
        refreshed.append(query)

    owner = SimpleNamespace(
        _closed=False,
        _profile_authority="profile",
        _authority_token="token",
        _active_projection_generation=1,
        _activation_in_flight=False,
        _mode=SwitcherMode.HISTORY,
        query_one=lambda *_args: SimpleNamespace(value="Exact"),
        run_worker=lambda work, **_kwargs: queued.append(work),
        _refresh_results=refresh,
    )
    ConsoleSessionSwitcherModal.reconcile_active_results(
        owner,
        (),
        profile_authority="profile",
        authority_token="token",
        projection_generation=2,
    )
    assert len(queued) == 1
    work = queued.pop()
    # A cancelled queued worker never starts its work. Close the old eager
    # implementation only on RED so this regression itself does not leak it.
    if inspect.iscoroutine(work):
        work.close()
        pytest.fail("queued reconciliation allocated an unawaited coroutine")
    assert inspect.iscoroutinefunction(work), "Textual requires async-callable work"
    assert refreshed == []
    await work()
    assert refreshed == ["Exact"]
