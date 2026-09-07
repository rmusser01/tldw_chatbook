"""Production-shaped Active/History projection contracts for Ctrl+K."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from threading import Event
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from textual.widgets import Button, Input, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from Tests.UI.test_console_workspace_controller import _workspace_controller
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.console_switcher_state import (
    ActivityGroup,
    CapturedReceipt,
    ConsoleSwitcherEntry,
    ConsoleSwitcherHistoryPage,
    ConsoleSwitcherTarget,
    SwitcherTargetKind,
    UnavailableSessionNotice,
    filter_console_active_results,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Widgets.Console.console_session_switcher_modal import (
    ACTIVE_PROJECTION_POLL_SECONDS,
    SEARCH_DEBOUNCE_SECONDS,
    ConsoleSessionSwitcherModal,
    ConsoleSwitcherChoice,
)
import tldw_chatbook.UI.Console_Modules.workspace as workspace_module
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
    controller._native_console_switcher_rows = lambda _cached=(): [_native_row()]
    controller._membership_console_browser_rows = lambda _current=None: []
    return controller


@pytest.fixture
def history_db(tmp_path):
    database = CharactersRAGDB(tmp_path / "history.sqlite", "test-client")
    yield database
    database.close_connection()


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
        controller.load_console_session_switcher_history(query="", offset=0, limit=50)
    )
    assert await asyncio.to_thread(entered.wait, 5)

    active = controller.console_session_switcher_active_entries()

    assert len(active) == 1
    assert isinstance(active[0], ConsoleSwitcherEntry)
    assert active[0].group is ActivityGroup.WORKING
    release.set()
    assert (await history).entries == ()


def test_active_projection_never_calls_full_workspace_membership_lister():
    app = SimpleNamespace(
        console_runtime=SimpleNamespace(
            profile_authority="profile-a",
            authority_token="runtime-a",
            activity_receipts=_ReceiptSnapshot(),
        )
    )
    controller = _projection_controller(app)
    controller._membership_console_browser_rows = lambda _current=None: (
        _ for _ in ()
    ).throw(AssertionError("Active must not scan every workspace membership"))

    active = controller.console_session_switcher_active_entries()

    assert [entry.title for entry in active] == ["Live agent work"]


def test_active_projection_reads_open_sessions_without_any_persistence_reader():
    session = SimpleNamespace(
        id="session-1",
        title="Memory-only agent",
        workspace_id="workspace-1",
        persisted_conversation_id="conversation-1",
        updated_at="2026-08-23T12:00:00+00:00",
    )
    store = SimpleNamespace(active_session_id="session-1", sessions=lambda: (session,))
    app = SimpleNamespace(
        console_runtime=SimpleNamespace(
            profile_authority="profile-a",
            authority_token="runtime-a",
            activity_receipts=_ReceiptSnapshot(),
        )
    )

    def persistence_forbidden(*_args, **_kwargs):
        raise AssertionError("Active projection must stay memory-only")

    controller = _workspace_controller(
        app_instance=app,
        current_chat_store_accessor=lambda: store,
        current_chat_controller_accessor=lambda: None,
        fleet_unseen_ids_accessor=persistence_forbidden,
    )
    controller._native_console_browser_rows = persistence_forbidden
    controller._membership_console_browser_rows = persistence_forbidden
    controller._starred_console_conversation_ids = persistence_forbidden
    controller._console_browser_workspace_labels = persistence_forbidden
    controller._console_browser_workspace_records = persistence_forbidden

    active = controller.console_session_switcher_active_entries()

    assert [entry.title for entry in active] == ["Memory-only agent"]
    assert active[0].workspace_label == "workspace-1"


@pytest.mark.asyncio
async def test_history_loader_revalidates_query_before_storage_search():
    calls: list[dict[str, object]] = []

    def list_conversations(**kwargs):
        calls.append(kwargs)
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

    page = await controller.load_console_session_switcher_history(
        query="x" * 513,
        offset=0,
        limit=50,
    )

    assert calls == []
    assert page.entries == ()
    assert page.error == "Search is limited to 512 characters."


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
            "query_terms": ("release",),
            "query_workspace_ids_by_term": ((),),
            "query_include_global_scope_by_term": (False,),
            "limit": 50,
            "offset": 0,
        }
    ]
    assert all(entry.target is not None for entry in page.entries)
    assert all(
        entry.row_key.startswith("conversation:profile-a:") for entry in page.entries
    )


@pytest.mark.asyncio
async def test_history_workspace_filter_resolves_labels_before_storage_search():
    calls: list[dict[str, object]] = []

    def list_conversations(**kwargs):
        calls.append(kwargs)
        return {
            "items": [
                {
                    "id": "roleplay-portrait",
                    "title": "Portrait scene",
                    "scope_type": "workspace",
                    "workspace_id": "workspace-roleplay",
                    "state": "in-progress",
                    "last_modified": "2026-08-23T12:00:00+00:00",
                }
            ],
            "pagination": {"total": 1},
        }

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
    controller._console_browser_workspace_labels = lambda: {
        "workspace-roleplay": "Roleplay Tavern",
        "workspace-research": "Research Lab",
    }

    page = await controller.load_console_session_switcher_history(
        query="workspace:roleplay portrait", offset=0, limit=50
    )

    assert [entry.title for entry in page.entries] == ["Portrait scene"]
    assert calls == [
        {
            "query": "portrait",
            "scope_type": "all",
            "workspace_ids": ("workspace-roleplay",),
            "query_terms": ("portrait",),
            "query_workspace_ids_by_term": ((),),
            "query_include_global_scope_by_term": (False,),
            "limit": 50,
            "offset": 0,
        }
    ]


@pytest.mark.asyncio
async def test_history_multiword_workspace_filter_consumes_matching_label_words():
    calls: list[dict[str, object]] = []

    def list_conversations(**kwargs):
        calls.append(kwargs)
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
    controller._console_browser_workspace_labels = lambda: {
        "workspace-roleplay": "Roleplay Tavern",
        "workspace-research": "Research Lab",
    }

    await controller.load_console_session_switcher_history(
        query="workspace:Roleplay Tavern", offset=0, limit=50
    )

    assert calls == [
        {
            "query": "",
            "scope_type": "all",
            "workspace_ids": ("workspace-roleplay",),
            "limit": 50,
            "offset": 0,
        }
    ]


@pytest.mark.asyncio
async def test_history_multiword_workspace_filter_preserves_preceding_title_text():
    calls: list[dict[str, object]] = []

    def list_conversations(**kwargs):
        calls.append(kwargs)
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
    controller._console_browser_workspace_labels = lambda: {
        "workspace-roleplay": "Roleplay Tavern",
    }

    await controller.load_console_session_switcher_history(
        query="portrait workspace:Roleplay Tavern", offset=0, limit=50
    )

    assert calls == [
        {
            "query": "portrait",
            "scope_type": "all",
            "workspace_ids": ("workspace-roleplay",),
            "query_terms": ("portrait",),
            "query_workspace_ids_by_term": ((),),
            "query_include_global_scope_by_term": (False,),
            "limit": 50,
            "offset": 0,
        }
    ]


@pytest.mark.asyncio
async def test_history_plain_workspace_and_unicode_text_search_end_to_end(history_db):
    roleplay_id = history_db.add_conversation(
        {
            "title": "An unrelated landscape",
            "scope_type": "workspace",
            "workspace_id": "workspace-roleplay",
        }
    )
    mixed_metadata_id = history_db.add_conversation(
        {
            "title": "Portrait scene",
            "scope_type": "workspace",
            "workspace_id": "workspace-roleplay",
        }
    )
    unicode_title_id = history_db.add_conversation({"title": "Straße release"})
    unicode_message_id = history_db.add_conversation({"title": "Localization notes"})
    history_db.add_message(
        {
            "conversation_id": unicode_message_id,
            "sender": "user",
            "content": "Straße evidence",
        }
    )
    app = SimpleNamespace(
        console_runtime=SimpleNamespace(
            profile_authority="profile-a",
            authority_token="runtime-a",
            activity_receipts=_ReceiptSnapshot(),
        ),
        local_chat_conversation_service=ChatConversationService(history_db),
    )
    controller = _projection_controller(app)
    controller._console_browser_workspace_labels = lambda: {
        "workspace-roleplay": "Roleplay Tavern"
    }

    workspace_page = await controller.load_console_session_switcher_history(
        query="Roleplay Tavern", offset=0, limit=50
    )
    mixed_page = await controller.load_console_session_switcher_history(
        query="portrait roleplay", offset=0, limit=50
    )
    title_page = await controller.load_console_session_switcher_history(
        query="Straße release", offset=0, limit=50
    )
    message_page = await controller.load_console_session_switcher_history(
        query="Straße evidence", offset=0, limit=50
    )

    assert {entry.conversation_id for entry in workspace_page.entries} == {
        roleplay_id,
        mixed_metadata_id,
    }
    assert [entry.conversation_id for entry in mixed_page.entries] == [
        mixed_metadata_id
    ]
    assert [entry.conversation_id for entry in title_page.entries] == [unicode_title_id]
    assert [entry.conversation_id for entry in message_page.entries] == [
        unicode_message_id
    ]


@pytest.mark.asyncio
async def test_history_chats_workspace_filter_includes_global_and_default_scopes():
    calls: list[dict[str, object]] = []

    def list_conversations(**kwargs):
        calls.append(kwargs)
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
    controller._console_browser_workspace_labels = lambda: {
        workspace_module.DEFAULT_WORKSPACE_ID: "Chats",
        "workspace-roleplay": "Roleplay Tavern",
    }

    await controller.load_console_session_switcher_history(
        query="workspace:chats", offset=0, limit=50
    )

    assert calls == [
        {
            "query": "",
            "scope_type": "all",
            "workspace_ids": (workspace_module.DEFAULT_WORKSPACE_ID,),
            "include_global_scope": True,
            "limit": 50,
            "offset": 0,
        }
    ]


@pytest.mark.asyncio
async def test_history_saved_alias_is_not_forwarded_as_title_search():
    calls: list[dict[str, object]] = []

    def list_conversations(**kwargs):
        calls.append(kwargs)
        return {
            "items": [
                {
                    "id": "saved-1",
                    "title": "Release plan",
                    "scope_type": "global",
                    "state": "in-progress",
                    "last_modified": "2026-08-23T12:00:00+00:00",
                }
            ],
            "pagination": {"total": 1},
        }

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
        query="is:saved", offset=0, limit=50
    )

    assert [entry.title for entry in page.entries] == ["Release plan"]
    assert calls == [
        {
            "query": "",
            "scope_type": "all",
            "limit": 50,
            "offset": 0,
        }
    ]


@pytest.mark.asyncio
async def test_history_loader_groups_missing_timezone_by_host_local_date(monkeypatch):
    class BoundaryDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            instant = cls(2026, 9, 3, 1, 0, tzinfo=timezone.utc)
            return (
                instant.astimezone(tz)
                if tz is not None
                else instant.replace(tzinfo=None)
            )

    def list_conversations(**_kwargs):
        return {
            "items": [
                {
                    "id": "recent-local-today",
                    "title": "Recent local conversation",
                    "scope_type": "global",
                    "last_modified": "2026-09-02T23:00:00+00:00",
                }
            ],
            "pagination": {"total": 1},
        }

    monkeypatch.setattr(workspace_module, "datetime", BoundaryDatetime)
    monkeypatch.setattr(
        workspace_module,
        "resolve_console_history_timezone",
        lambda _configured: ZoneInfo("America/Los_Angeles"),
    )
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
        query="", offset=0, limit=50
    )

    assert page.entries[0].section == "Today"


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
        receipt_state: str = "ready",
        active_projection_loader=None,
        authority_snapshot=None,
    ) -> None:
        super().__init__()
        self.active_results = tuple(active_results)
        self.history_loader = history_loader
        self.preferred_native_session_id = preferred_native_session_id
        self.receipt_state = receipt_state
        self.active_projection_loader = active_projection_loader
        self.authority_snapshot = authority_snapshot
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
                activity_receipt_state=self.receipt_state,
                active_projection_loader=self.active_projection_loader,
                authority_snapshot=self.authority_snapshot,
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
    async with app.run_test(size=(52, 20)) as pilot:
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
        active_mode = app.screen.query_one("#console-switcher-active-mode", Button)
        history_mode = app.screen.query_one("#console-switcher-history-mode", Button)
        assert str(active_mode.label) == "Active (1)"
        assert active_mode.has_class("console-switcher-mode-current")
        assert not history_mode.has_class("console-switcher-mode-current")
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
        assert "History matches" in str(status.renderable)
        assert app.screen.query_one("#console-switcher-history-mode", Button).has_class(
            "console-switcher-mode-current"
        )
        assert not app.screen.query_one(
            "#console-switcher-active-mode", Button
        ).has_class("console-switcher-mode-current")
        assert calls == [("migration", 0, 50)]

        await pilot.press("f3")
        await pilot.pause()
        assert query.value == "migration"
        history_mode = app.screen.query_one("#console-switcher-history-mode", Button)
        assert str(history_mode.label) == "History"
        assert history_mode.has_class("console-switcher-mode-current")


@pytest.mark.asyncio
async def test_oversized_query_is_rejected_before_history_search():
    calls: list[str] = []

    async def load_history(*, query: str, offset: int, limit: int):
        calls.append(query)
        return ConsoleSwitcherHistoryPage((), offset, limit, 0)

    app = _ActivitySwitcherApp(history_loader=load_history)
    async with app.run_test(size=(90, 30)) as pilot:
        committed = await app.screen._refresh_results("x" * 513, reset_page=True)
        await pilot.pause()

        assert committed is False
        assert calls == []
        assert "512 characters" in str(
            app.screen.query_one("#console-switcher-feedback", Static).renderable
        )


@pytest.mark.asyncio
async def test_blank_enter_consequence_names_the_exact_mru_destination():
    app = _ActivitySwitcherApp(
        active_results=(
            replace(
                _active_entry(
                    "session:current",
                    "Current design review",
                    session_id="current",
                    group=ActivityGroup.CURRENT,
                ),
                is_active=True,
                state_label="CURRENT",
            ),
            _active_entry(
                "session:mru",
                "Release approval",
                session_id="mru",
                group=ActivityGroup.WAITING_FOR_YOU,
            ),
        ),
        preferred_native_session_id="mru",
    )
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.pause()
        status = str(
            app.screen.query_one("#console-switcher-status", Static).renderable
        )
        assert status == "Enter switches to: Release approval · 2 of 2"
        assert "F2: rename" in str(
            app.screen.query_one("#console-switcher-hints", Static).renderable
        )
        assert isinstance(app.focused, Input)


@pytest.mark.asyncio
async def test_result_rows_use_left_aligned_state_and_distilled_metadata():
    entry = replace(
        _active_entry(
            "session:finished",
            "Regression triage",
            session_id="finished",
            group=ActivityGroup.NEW_RESULTS,
        ),
        state_label="FINISHED · UNSEEN",
        activity_state="done",
        workspace_label="Quality",
        multiplicity=1,
        subtitle=(
            "FINISHED · UNSEEN · CONSOLE TAB · Quality · open session · now · +1"
        ),
    )
    app = _ActivitySwitcherApp(
        active_results=(entry,), preferred_native_session_id="finished"
    )
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.pause()
        button = app.screen.query_one(".console-switcher-result", Button)
        lines = str(button.label).splitlines()
        assert lines[0].startswith("▸ FINISHED · UNSEEN")
        assert lines[1].startswith("  Quality · Console tab")
        assert "2 updates" in lines[1]
        assert "+1" not in str(button.label)
        assert button.styles.content_align_horizontal == "left"


@pytest.mark.asyncio
async def test_plain_language_search_guidance_fits_without_filter_grammar():
    app = _ActivitySwitcherApp(
        active_results=(_active_entry("session:one", "Live deploy", session_id="one"),)
    )
    async with app.run_test(size=(72, 35)) as pilot:
        await pilot.pause()
        query = app.screen.query_one("#console-switcher-query", Input)
        assert "waiting" in query.placeholder
        assert "running" in query.placeholder
        assert "is:" not in query.placeholder
        assert "workspace:<" not in query.placeholder


@pytest.mark.asyncio
async def test_single_result_switchboard_sizes_to_content_below_ceiling():
    app = _ActivitySwitcherApp(
        active_results=(_active_entry("session:one", "Live deploy", session_id="one"),)
    )
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        modal = app.screen.query_one("#console-switcher-modal")
        assert modal.region.height < 25


@pytest.mark.asyncio
async def test_switchboard_surface_follows_terminal_theme():
    app = _ActivitySwitcherApp(
        active_results=(_active_entry("session:one", "Live deploy", session_id="one"),)
    )
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.pause()
        modal = app.screen.query_one("#console-switcher-modal")
        dark_background = modal.styles.background
        app.theme = "textual-light"
        await pilot.pause()
        assert modal.styles.background != dark_background


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
        authority_token="runtime-a",
        session_id="gone",
        group=ActivityGroup.WAITING_FOR_YOU,
        latest_at=None,
        receipts=(CapturedReceipt(activity_id="activity-1", status="failed"),),
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
async def test_unavailable_pointer_action_requires_explicit_confirm_button():
    notice = UnavailableSessionNotice(
        stable_result_key="unavailable-session:profile-a:gone",
        profile_authority="profile-a",
        authority_token="runtime-a",
        session_id="gone",
        group=ActivityGroup.WAITING_FOR_YOU,
        latest_at=None,
        receipts=(CapturedReceipt(activity_id="activity-1", status="failed"),),
        primary_status="failed",
    )
    app = _ActivitySwitcherApp(active_results=(notice,))
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.click(".console-switcher-result")
        await pilot.pause()
        assert app.result == "unset"
        assert (
            "again"
            in str(
                app.screen.query_one("#console-switcher-status", Static).renderable
            ).lower()
        )
        confirm = app.screen.query_one("#console-switcher-confirm-mark-seen", Button)
        assert confirm.display

        await pilot.click("#console-switcher-confirm-mark-seen")
        await pilot.pause()
    assert isinstance(app.result, ConsoleSwitcherChoice)
    assert app.result.kind == "mark_seen"
    assert app.result.entry is notice


@pytest.mark.asyncio
async def test_home_end_and_page_keys_move_the_explicit_candidate():
    app = _ActivitySwitcherApp(
        active_results=tuple(
            _active_entry(f"session:{index}", f"Agent {index}", session_id=str(index))
            for index in range(20)
        )
    )
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.press("end")
        await pilot.pause()
        assert app.screen._candidate_index == 19

        await pilot.press("home")
        await pilot.pause()
        assert app.screen._candidate_index == 0

        await pilot.press("pagedown")
        await pilot.pause()
        assert app.screen._candidate_index > 0

        await pilot.press("pageup")
        await pilot.pause()
        assert app.screen._candidate_index == 0


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
    async with app.run_test(size=(52, 20)) as pilot:
        await pilot.press("f3")
        await pilot.pause()
        assert len(app.screen.query(".console-switcher-result")) == 2
        await pilot.click("#console-switcher-next-page")
        await pilot.pause()
        assert calls == [("", 0, 50), ("", 50, 50)]
        page = app.screen.query_one("#console-switcher-page-status", Static)
        assert "51–52 of 52" in str(page.renderable)


@pytest.mark.asyncio
async def test_active_page_buttons_reach_rows_beyond_first_fifty():
    unavailable = UnavailableSessionNotice(
        stable_result_key="unavailable-session:profile-a:gone",
        profile_authority="profile-a",
        authority_token="runtime-a",
        session_id="gone",
        group=ActivityGroup.OTHER_OPEN,
        latest_at=None,
        receipts=(CapturedReceipt(activity_id="activity-gone", status="failed"),),
        primary_status="failed",
    )
    active = tuple(
        _active_entry(f"session:{index}", f"Agent {index}", session_id=str(index))
        for index in range(50)
    ) + (unavailable,)
    app = _ActivitySwitcherApp(active_results=active)

    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.pause()
        page = app.screen.query_one("#console-switcher-page-status", Static)
        assert "1–50 of 51" in str(page.renderable)

        await pilot.click("#console-switcher-next-page")
        await pilot.pause()

        assert "Session unavailable" in str(
            app.screen.query_one(".console-switcher-result", Button).label
        )
        assert "51–51 of 51" in str(page.renderable)


@pytest.mark.asyncio
async def test_active_reconcile_moves_to_page_containing_still_focused_result():
    original = tuple(
        _active_entry(f"session:{index}", f"Agent {index}", session_id=str(index))
        for index in range(60)
    )
    app = _ActivitySwitcherApp(active_results=original)

    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.click("#console-switcher-next-page")
        await pilot.pause()
        focused_entry = original[55]
        app.screen._result_buttons()[5].focus()
        await pilot.pause()
        prepended = tuple(
            _active_entry(
                f"session:prepended-{index}",
                f"Prepended agent {index}",
                session_id=f"prepended-{index}",
            )
            for index in range(50)
        )

        app.screen.reconcile_active_results(
            prepended + original,
            profile_authority="profile-a",
            authority_token="runtime-a",
            projection_generation=8,
        )
        await pilot.pause()

        assert app.screen._page_offset == 100
        assert app.screen._payload_by_widget_id[app.focused.id] is focused_entry
        assert "101–110 of 110" in str(
            app.screen.query_one("#console-switcher-page-status", Static).renderable
        )
        assert "no longer available" not in str(
            app.screen.query_one("#console-switcher-status", Static).renderable
        )


@pytest.mark.asyncio
async def test_active_reconcile_returns_to_first_page_when_focused_result_disappears(
    monkeypatch,
):
    original = tuple(
        _active_entry(f"session:{index}", f"Agent {index}", session_id=str(index))
        for index in range(60)
    )
    app = _ActivitySwitcherApp(active_results=original)

    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.click("#console-switcher-next-page")
        await pilot.pause()
        app.screen._result_buttons()[5].focus()
        await pilot.pause()
        notices: list[str] = []
        monkeypatch.setattr(
            app.screen,
            "notify",
            lambda message, **_kwargs: notices.append(str(message)),
        )

        app.screen.reconcile_active_results(
            tuple(entry for entry in original if entry is not original[55]),
            profile_authority="profile-a",
            authority_token="runtime-a",
            projection_generation=8,
        )
        await pilot.pause()

        feedback = str(
            app.screen.query_one("#console-switcher-status", Static).renderable
        )
        assert app.screen._page_offset == 0
        assert (
            feedback == "The selected result is no longer available — selection moved."
        )
        assert notices == [feedback]


@pytest.mark.asyncio
async def test_degraded_activity_status_is_visible_and_clears_after_retry():
    entry = _active_entry("session:one", "Agent one", session_id="one")
    app = _ActivitySwitcherApp(active_results=(entry,), receipt_state="degraded")

    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.pause()
        status = app.screen.query_one("#console-switcher-receipt-state", Static)
        assert status.display
        assert "Local activity updates unavailable" in str(status.renderable)

        app.screen.reconcile_active_results(
            (entry,),
            profile_authority="profile-a",
            authority_token="runtime-a",
            projection_generation=8,
            activity_receipt_state="ready",
        )
        await pilot.pause()
        assert status.display is False


def test_semantic_aliases_share_predicates_and_current_is_destination_identity():
    current_running = replace(
        _active_entry("session:current", "Current runner", session_id="current"),
        is_active=True,
        group=ActivityGroup.WORKING,
        activity_state="running",
    )
    queued = replace(
        _active_entry("session:queued", "Queued agent", session_id="queued"),
        group=ActivityGroup.WORKING,
        activity_state="queued",
    )
    saved = replace(
        _history_entry("conversation:saved", "Saved chat"),
        group=ActivityGroup.CURRENT,
    )
    saved_open = replace(
        _active_entry("session:saved-open", "Saved open", session_id="saved-open"),
        conversation_id="saved-open-conversation",
    )
    unavailable = UnavailableSessionNotice(
        stable_result_key="unavailable-session:profile-a:gone",
        profile_authority="profile-a",
        authority_token="runtime-a",
        session_id="gone",
        group=ActivityGroup.WAITING_FOR_YOU,
        latest_at=None,
        receipts=(CapturedReceipt(activity_id="activity-gone", status="failed"),),
        primary_status="failed",
    )
    results = (current_running, queued, saved, saved_open, unavailable)

    for query in ("current", "is:current"):
        assert filter_console_active_results(results, query) == (current_running,)
    for query in (
        "working",
        "is:working",
        "running",
        "is:running",
        "queued",
        "is:queued",
    ):
        assert filter_console_active_results(results, query) == (
            current_running,
            queued,
        )
    for query in ("open", "is:open"):
        assert filter_console_active_results(results, query) == (
            current_running,
            queued,
            saved_open,
        )
    for query in ("saved", "is:saved"):
        assert filter_console_active_results(results, query) == (saved, saved_open)
    for query in ("unavailable", "is:unavailable"):
        assert filter_console_active_results(results, query) == (unavailable,)


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


@pytest.mark.asyncio
async def test_open_modal_polls_and_reconciles_controller_projection_changes():
    first = _active_entry("session:one", "Agent one", session_id="one")
    second = _active_entry("session:two", "Agent two", session_id="two")
    state = {"results": (first,), "generation": 7, "receipt_state": "ready"}

    def load_active():
        return (
            state["results"],
            "profile-a",
            "runtime-a",
            state["generation"],
            state["receipt_state"],
        )

    app = _ActivitySwitcherApp(
        active_results=(first,), active_projection_loader=load_active
    )
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.pause()
        state["results"] = (second, first)
        await pilot.pause(ACTIVE_PROJECTION_POLL_SECONDS + 0.1)

        labels = [str(button.label) for button in app.screen._result_buttons()]
        assert any("Agent two" in label for label in labels)
        active_mode = app.screen.query_one("#console-switcher-active-mode", Button)
        assert str(active_mode.label) == "Active (2)"
        assert active_mode.has_class("console-switcher-mode-current")


@pytest.mark.asyncio
async def test_projection_change_during_history_load_retries_without_stuck_pending():
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = {"count": 0}
    entry = _active_entry("session:one", "Agent one", session_id="one")
    state = {"generation": 7}

    async def load_history(*, query: str, offset: int, limit: int):
        calls["count"] += 1
        if calls["count"] == 1:
            entered.set()
            await release.wait()
        return ConsoleSwitcherHistoryPage(
            (_history_entry("conversation:saved", "Saved chat"),),
            offset,
            limit,
            1,
        )

    def load_active():
        return (
            (entry,),
            "profile-a",
            "runtime-a",
            state["generation"],
            "ready",
        )

    def authority_snapshot():
        return "profile-a", "runtime-a", state["generation"]

    app = _ActivitySwitcherApp(
        active_results=(entry,),
        history_loader=load_history,
        active_projection_loader=load_active,
        authority_snapshot=authority_snapshot,
    )
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.press("f3")
        await asyncio.wait_for(entered.wait(), timeout=2)
        state["generation"] = 8
        await pilot.pause(ACTIVE_PROJECTION_POLL_SECONDS + 0.1)
        release.set()
        await pilot.pause(0.2)

        assert app.screen._query_pending is False
        assert "Saved chat" in str(
            app.screen.query_one(".console-switcher-result", Button).label
        )
        assert "Searching History" not in str(
            app.screen.query_one("#console-switcher-status", Static).renderable
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("remove_unavailable", [False, True])
async def test_reconcile_moves_focus_when_selected_result_disappears(
    remove_unavailable: bool,
):
    first = _active_entry("session:one", "Agent one", session_id="one")
    unavailable = UnavailableSessionNotice(
        stable_result_key="unavailable-session:profile-a:gone",
        profile_authority="profile-a",
        authority_token="runtime-a",
        session_id="gone",
        group=ActivityGroup.OTHER_OPEN,
        latest_at=None,
        receipts=(CapturedReceipt(activity_id="activity-gone", status="failed"),),
        primary_status="failed",
    )
    third = _active_entry("session:three", "Agent three", session_id="three")
    middle = (
        unavailable
        if remove_unavailable
        else _active_entry("session:two", "Agent two", session_id="two")
    )
    app = _ActivitySwitcherApp(active_results=(first, middle, third))

    async with app.run_test(size=(90, 30)) as pilot:
        buttons = app.screen._result_buttons()
        buttons[1].focus()
        await pilot.pause()
        app.screen.reconcile_active_results(
            (first, third),
            profile_authority="profile-a",
            authority_token="runtime-a",
            projection_generation=8,
        )
        await pilot.pause()

        focused = app.focused
        assert isinstance(focused, Button)
        assert app.screen._payload_by_widget_id[focused.id] is third
        assert "no longer available" in str(
            app.screen.query_one("#console-switcher-status", Static).renderable
        )


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
        assert (
            str(app.screen.query_one("#console-switcher-active-mode", Button).label)
            == "Active (20)"
        )
        assert "Enter switches to:" in str(status.renderable)
