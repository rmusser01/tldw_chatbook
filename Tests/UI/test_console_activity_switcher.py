"""Production-shaped Active/History projection contracts for Ctrl+K."""

from __future__ import annotations

import asyncio
from threading import Event
from types import SimpleNamespace

import pytest
from textual.widgets import Button, Input, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from Tests.UI.test_console_workspace_controller import _workspace_controller
from tldw_chatbook.Chat.console_switcher_state import (
    ActivityGroup,
    CapturedReceipt,
    ConsoleSwitcherEntry,
    ConsoleSwitcherHistoryPage,
    ConsoleSwitcherTarget,
    SwitcherTargetKind,
    UnavailableSessionNotice,
)
from tldw_chatbook.Widgets.Console.console_session_switcher_modal import (
    SEARCH_DEBOUNCE_SECONDS,
    ConsoleSessionSwitcherModal,
    ConsoleSwitcherChoice,
)
from tldw_chatbook.Workspaces.conversation_browser_state import (
    ConsoleConversationBrowserInputRow,
)


def _native_row(session_id: str = "session-1") -> ConsoleConversationBrowserInputRow:
    return ConsoleConversationBrowserInputRow(
        row_key=f"native:{session_id}",
        conversation_id=None,
        native_session_id=session_id,
        title="Live agent work",
        scope_type="workspace",
        workspace_id="workspace-1",
        workspace_label="Workspace 1",
        status="active session",
        selected=True,
        source_kind="native",
        updated_sort="2026-08-23T12:00:00+00:00",
        run_marker="[*]",
    )


class _ReceiptSnapshot:
    def __init__(self, *, state: str = "ready") -> None:
        self._state = state

    def unseen_snapshot(self):
        return ()

    def hydration_state(self):
        return self._state


def _projection_controller(app):
    controller = _workspace_controller(app_instance=app)
    controller._native_console_browser_rows = lambda _current=None: [_native_row()]
    controller._membership_console_browser_rows = lambda _current=None: []
    return controller


@pytest.mark.asyncio
async def test_active_is_immediate_while_bounded_history_is_blocked():
    entered = Event()
    release = Event()

    def list_conversations(**_kwargs):
        entered.set()
        assert release.wait(5)
        return {"items": [], "pagination": {"total": 0}}

    app = SimpleNamespace(
        console_runtime=SimpleNamespace(
            profile_authority="profile-a",
            authority_token="runtime-a",
            activity_receipts=_ReceiptSnapshot(),
        ),
        local_chat_conversation_service=SimpleNamespace(
            list_conversations=list_conversations
        ),
    )
    controller = _projection_controller(app)

    history = asyncio.create_task(
        controller.load_console_session_switcher_history(
            query="", offset=0, limit=50
        )
    )
    assert await asyncio.to_thread(entered.wait, 5)

    active = controller.console_session_switcher_active_entries()

    assert len(active) == 1
    assert isinstance(active[0], ConsoleSwitcherEntry)
    assert active[0].group is ActivityGroup.WORKING
    release.set()
    assert (await history).entries == ()


@pytest.mark.asyncio
async def test_history_uses_one_all_local_bounded_page_with_explicit_targets():
    calls: list[dict[str, object]] = []

    def list_conversations(**kwargs):
        calls.append(kwargs)
        items = [
            {
                "id": f"conversation-{index}",
                "title": f"Conversation {index}",
                "scope_type": "workspace",
                "workspace_id": "workspace-1",
                "state": "in-progress",
                "last_modified": "2026-08-23T12:00:00+00:00",
            }
            for index in range(70)
        ]
        return {"items": items, "pagination": {"total": 70}}

    app = SimpleNamespace(
        console_runtime=SimpleNamespace(
            profile_authority="profile-a",
            authority_token="runtime-a",
            activity_receipts=_ReceiptSnapshot(),
        ),
        local_chat_conversation_service=SimpleNamespace(
            list_conversations=list_conversations
        ),
    )
    controller = _projection_controller(app)

    page = await controller.load_console_session_switcher_history(
        query="release", offset=0, limit=500
    )

    assert len(page.entries) == 50
    assert page.total == 70
    assert page.has_more is True
    assert calls == [
        {
            "query": "release",
            "scope_type": "all",
            "limit": 50,
            "offset": 0,
        }
    ]
    assert all(entry.target is not None for entry in page.entries)
    assert all(entry.row_key.startswith("conversation:profile-a:") for entry in page.entries)


@pytest.mark.asyncio
async def test_receipt_degradation_leaves_open_active_and_history_available():
    def list_conversations(**_kwargs):
        return {
            "items": [
                {
                    "id": "saved-1",
                    "title": "Saved conversation",
                    "scope_type": "global",
                    "last_modified": "2026-08-22T12:00:00+00:00",
                }
            ],
            "pagination": {"total": 1},
        }

    receipts = _ReceiptSnapshot(state="degraded")
    app = SimpleNamespace(
        console_runtime=SimpleNamespace(
            profile_authority="profile-a",
            authority_token="runtime-a",
            activity_receipts=receipts,
        ),
        local_chat_conversation_service=SimpleNamespace(
            list_conversations=list_conversations
        ),
    )
    controller = _projection_controller(app)
    tree_before = controller._workspace_tree_search
    pages_before = dict(controller._workspace_page_attempts)

    active = controller.console_session_switcher_active_entries()
    history = await controller.load_console_session_switcher_history(
        query="saved", offset=0, limit=50
    )

    assert receipts.hydration_state() == "degraded"
    assert active and active[0].title == "Live agent work"
    assert [entry.title for entry in history.entries] == ["Saved conversation"]
    assert controller._workspace_tree_search is tree_before
    assert controller._workspace_page_attempts == pages_before


def _active_entry(
    key: str,
    title: str,
    *,
    session_id: str,
    group: ActivityGroup = ActivityGroup.OTHER_OPEN,
) -> ConsoleSwitcherEntry:
    target = ConsoleSwitcherTarget(
        kind=SwitcherTargetKind.NATIVE_SESSION,
        profile_authority="profile-a",
        authority_token="runtime-a",
        session_id=session_id,
        conversation_id=None,
        scope_type="workspace",
        workspace_id="workspace-1",
    )
    return ConsoleSwitcherEntry(
        row_key=key,
        title=title,
        subtitle="OPEN AGENT · Workspace 1 · now",
        native_session_id=session_id,
        conversation_id=None,
        scope_type="workspace",
        workspace_id="workspace-1",
        is_active=False,
        section=group.value,
        state_label="OPEN AGENT",
        target=target,
        group=group,
        activity_state="other-open",
    )


def _history_entry(key: str, title: str) -> ConsoleSwitcherEntry:
    conversation_id = key.removeprefix("conversation:")
    target = ConsoleSwitcherTarget(
        kind=SwitcherTargetKind.PERSISTED_CONVERSATION,
        profile_authority="profile-a",
        authority_token="runtime-a",
        session_id=None,
        conversation_id=conversation_id,
        scope_type="global",
        workspace_id=None,
    )
    return ConsoleSwitcherEntry(
        row_key=key,
        title=title,
        subtitle="SAVED CHAT · Today",
        native_session_id=None,
        conversation_id=conversation_id,
        scope_type="global",
        workspace_id=None,
        is_active=False,
        section="Today",
        state_label="SAVED CHAT",
        target=target,
    )


class _ActivitySwitcherApp(ConsolidatedCSSApp):
    def __init__(
        self,
        *,
        active_results=(),
        history_loader=None,
        preferred_native_session_id: str | None = None,
    ) -> None:
        super().__init__()
        self.active_results = tuple(active_results)
        self.history_loader = history_loader
        self.preferred_native_session_id = preferred_native_session_id
        self.result: ConsoleSwitcherChoice | None | str = "unset"

    async def on_mount(self) -> None:
        await self.push_screen(
            ConsoleSessionSwitcherModal(
                active_results=self.active_results,
                history_loader=self.history_loader,
                preferred_native_session_id=self.preferred_native_session_id,
                profile_authority="profile-a",
                authority_token="runtime-a",
                active_projection_generation=7,
            ),
            callback=self._capture,
        )

    def _capture(self, result) -> None:  # type: ignore[no-untyped-def]
        self.result = result


@pytest.mark.asyncio
async def test_modal_opens_on_active_without_loading_history():
    calls: list[tuple[str, int, int]] = []

    async def load_history(*, query: str, offset: int, limit: int):
        calls.append((query, offset, limit))
        return ConsoleSwitcherHistoryPage((), offset, limit, 0)

    app = _ActivitySwitcherApp(
        active_results=(_active_entry("session:one", "Live deploy", session_id="one"),),
        history_loader=load_history,
    )
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.pause()
        assert isinstance(app.focused, Input)
        assert app.screen._entries, (
            app.screen._request_generation,
            app.screen._query_pending,
            app.screen._rendered_query,
            app.screen._active_results,
        )
        assert "Live deploy" in str(
            app.screen.query_one(".console-switcher-result", Button).label
        )
        assert "Active (1) — selected" in str(
            app.screen.query_one("#console-switcher-active-mode", Button).label
        )
        assert calls == []


@pytest.mark.asyncio
async def test_zero_active_matches_widens_to_history_and_f3_retains_query():
    calls: list[tuple[str, int, int]] = []

    async def load_history(*, query: str, offset: int, limit: int):
        calls.append((query, offset, limit))
        return ConsoleSwitcherHistoryPage(
            (_history_entry("conversation:migration", "Migration notes"),),
            offset,
            limit,
            1,
        )

    app = _ActivitySwitcherApp(
        active_results=(_active_entry("session:one", "Live deploy", session_id="one"),),
        history_loader=load_history,
    )
    async with app.run_test(size=(90, 30)) as pilot:
        query = app.screen.query_one("#console-switcher-query", Input)
        query.value = "migration"
        await pilot.pause(SEARCH_DEBOUNCE_SECONDS + 0.1)
        assert "Migration notes" in str(
            app.screen.query_one(".console-switcher-result", Button).label
        )
        status = app.screen.query_one("#console-switcher-status", Static)
        assert "No active matches — showing History" in str(status.renderable)
        assert calls == [("migration", 0, 50)]

        await pilot.press("f3")
        await pilot.pause()
        assert query.value == "migration"
        assert "History — selected" in str(
            app.screen.query_one("#console-switcher-history-mode", Button).label
        )


@pytest.mark.asyncio
async def test_f2_requires_a_focused_native_result():
    app = _ActivitySwitcherApp(
        active_results=(_active_entry("session:one", "Live deploy", session_id="one"),)
    )
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.press("f2")
        await pilot.pause()
        assert app.result == "unset"
        assert "Focus an open agent result" in str(
            app.screen.query_one("#console-switcher-feedback", Static).renderable
        )

        await pilot.press("down")
        await pilot.press("f2")
        await pilot.pause()
    assert isinstance(app.result, ConsoleSwitcherChoice)
    assert app.result.kind == "rename"
    assert app.result.entry.native_session_id == "one"


@pytest.mark.asyncio
async def test_unavailable_search_result_requires_two_enter_presses():
    notice = UnavailableSessionNotice(
        stable_result_key="unavailable-session:profile-a:gone",
        profile_authority="profile-a",
        session_id="gone",
        group=ActivityGroup.WAITING_FOR_YOU,
        latest_at=None,
        receipts=(CapturedReceipt("activity-1", "failed"),),
        primary_status="failed",
    )
    app = _ActivitySwitcherApp(active_results=(notice,))
    async with app.run_test(size=(90, 30)) as pilot:
        query = app.screen.query_one("#console-switcher-query", Input)
        query.value = "unavailable"
        await pilot.press("enter")
        await pilot.pause()
        assert app.result == "unset"
        assert isinstance(app.focused, Button)
        assert app.focused.has_class("console-switcher-result")

        await pilot.press("enter")
        await pilot.pause()
    assert isinstance(app.result, ConsoleSwitcherChoice)
    assert app.result.kind == "mark_seen"
    assert app.result.entry is notice


@pytest.mark.asyncio
async def test_history_page_buttons_keep_pages_bounded():
    calls: list[tuple[str, int, int]] = []

    async def load_history(*, query: str, offset: int, limit: int):
        calls.append((query, offset, limit))
        return ConsoleSwitcherHistoryPage(
            tuple(
                _history_entry(
                    f"conversation:{offset + index}", f"Conversation {offset + index}"
                )
                for index in range(2)
            ),
            offset,
            limit,
            52,
        )

    app = _ActivitySwitcherApp(history_loader=load_history)
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.press("f3")
        await pilot.pause()
        assert len(app.screen.query(".console-switcher-result")) == 2
        await pilot.click("#console-switcher-next-page")
        await pilot.pause()
        assert calls == [("", 0, 50), ("", 50, 50)]
        page = app.screen.query_one("#console-switcher-page-status", Static)
        assert "51–52 of 52" in str(page.renderable)


@pytest.mark.asyncio
async def test_pending_exact_query_cannot_activate_an_old_active_row():
    entered = asyncio.Event()
    release = asyncio.Event()

    async def load_history(*, query: str, offset: int, limit: int):
        assert query == "migration"
        entered.set()
        await release.wait()
        return ConsoleSwitcherHistoryPage(
            (_history_entry("conversation:migration", "Migration notes"),),
            offset,
            limit,
            1,
        )

    app = _ActivitySwitcherApp(
        active_results=(_active_entry("session:one", "Live deploy", session_id="one"),),
        history_loader=load_history,
    )
    async with app.run_test(size=(90, 30)) as pilot:
        app.screen.query_one("#console-switcher-query", Input).value = "migration"
        submission = asyncio.create_task(pilot.press("enter"))
        await asyncio.wait_for(entered.wait(), timeout=2)
        assert app.result == "unset"
        release.set()
        await submission
        await pilot.pause()
    assert isinstance(app.result, ConsoleSwitcherChoice)
    assert app.result.entry.title == "Migration notes"


@pytest.mark.asyncio
async def test_closing_during_history_load_drops_the_late_commit():
    entered = asyncio.Event()
    release = asyncio.Event()

    async def load_history(*, query: str, offset: int, limit: int):
        entered.set()
        await release.wait()
        return ConsoleSwitcherHistoryPage(
            (_history_entry("conversation:late", "Late row"),),
            offset,
            limit,
            1,
        )

    app = _ActivitySwitcherApp(history_loader=load_history)
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.press("f3")
        await asyncio.wait_for(entered.wait(), timeout=2)
        await pilot.press("escape")
        await pilot.pause()
        assert app.result is None
        assert not isinstance(app.screen, ConsoleSessionSwitcherModal)
        release.set()
        await pilot.pause()
        assert app.result is None


@pytest.mark.asyncio
async def test_live_reorder_retains_focus_by_stable_payload_identity():
    first = _active_entry("session:one", "Agent one", session_id="one")
    second = _active_entry("session:two", "Agent two", session_id="two")
    app = _ActivitySwitcherApp(active_results=(first, second))
    async with app.run_test(size=(90, 30)) as pilot:
        buttons = list(app.screen.query(".console-switcher-result"))
        buttons[1].focus()
        await pilot.pause()
        app.screen.reconcile_active_results(
            (second, first),
            profile_authority="profile-a",
            authority_token="runtime-a",
            projection_generation=8,
        )
        await pilot.pause()
        focused = app.focused
        assert isinstance(focused, Button)
        payload = app.screen._payload_by_widget_id[focused.id]
        assert payload is second


@pytest.mark.parametrize("size", [(52, 20), (72, 35), (120, 50)])
@pytest.mark.asyncio
async def test_switchboard_chrome_and_literal_state_fit_terminal(size):
    app = _ActivitySwitcherApp(
        active_results=tuple(
            _active_entry(
                f"session:{index}",
                f"Agent {index}",
                session_id=str(index),
            )
            for index in range(20)
        )
    )
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        modal = app.screen.query_one("#console-switcher-modal")
        cancel = app.screen.query_one("#console-switcher-cancel", Button)
        status = app.screen.query_one("#console-switcher-status", Static)
        assert modal.region.height <= 35
        assert modal.region.x >= 0 and modal.region.right <= app.size.width
        assert modal.region.y >= 0 and modal.region.bottom <= app.size.height
        assert cancel.region.bottom <= modal.content_region.bottom
        assert "Active (20) — selected" in str(
            app.screen.query_one("#console-switcher-active-mode", Button).label
        )
        assert "Selection:" in str(status.renderable)
