"""Exact Library inspection preparation must not consume the source visit."""

from __future__ import annotations

import asyncio

import pytest

from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.Character_Chat.character_conversation_navigation import (
    UnresolvedConversationKey,
)
from tldw_chatbook.Chat.chat_conversation_scope_service import (
    ChatConversationScopeService,
)
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Constants import LIBRARY_NAV_CONTEXT_CHARACTER_INSPECTION
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Navigation.character_conversation_navigation import (
    LibraryUnavailableConversationInspection,
    RoleplayReturnTarget,
    serialize_library_unavailable_inspection,
)
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen


def _context(db, conversation_id="exact"):
    return {
        LIBRARY_NAV_CONTEXT_CHARACTER_INSPECTION: serialize_library_unavailable_inspection(
            LibraryUnavailableConversationInspection(
                UnresolvedConversationKey(db.get_local_authority_id(), conversation_id),
                RoleplayReturnTarget.console_context_character(),
            )
        )
    }


@pytest.fixture
def library(tmp_path):
    owner = _build_test_app()
    db = CharactersRAGDB(tmp_path / "inspection.sqlite", client_id="inspection")
    db.add_conversation({"id": "exact", "title": "Exact local inspection"})
    owner.chachanotes_db = db
    owner.chat_conversation_scope_service = ChatConversationScopeService(
        local_service=ChatConversationService(db), server_service=None
    )
    screen = LibraryScreen(owner)
    yield owner, screen, db
    db.close()


@pytest.mark.asyncio
async def test_cold_inspection_prepares_display_neutrally_then_commits_exact_once(
    library,
):
    _, screen, db = library
    before = (
        screen._library_selected_row_id,
        screen._selected_conversation_id,
        screen._conversations_state.reader_state,
    )
    prepared = await screen.prepare_character_inspection(
        _context(db), is_current=lambda: True
    )
    released = []
    try:
        assert prepared is not None
        prepared.release = lambda: released.append("own")
        assert (
            screen._library_selected_row_id,
            screen._selected_conversation_id,
            screen._conversations_state.reader_state,
        ) == before
        assert prepared.is_current()
        assert screen.commit_character_inspection(prepared)
        assert screen._selected_conversation_id == "exact"
        assert screen._conversations_state.reader_state.selected_id == "exact"
        assert (
            screen._navigation_controller.character_route.route.unresolved.conversation_id
            == "exact"
        )
        assert screen._pending_library_character_navigation is None
        assert not screen.commit_character_inspection(prepared)
    finally:
        if prepared is not None:
            prepared.discard()
            prepared.discard()
    assert released == ["own"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["database", "service", "generation", "cancel", "authority"]
)
async def test_inspection_rejects_changed_owner_after_real_locator_without_display_mutation(
    library, change, monkeypatch
):
    owner, screen, db = library
    service = owner.chat_conversation_scope_service
    locate = service.locate_conversation_page
    entered, release = asyncio.Event(), asyncio.Event()
    current = True
    before = (
        screen._library_selected_row_id,
        screen._selected_conversation_id,
        screen._conversations_state.reader_state,
    )

    async def held(*args, **kwargs):
        result = await locate(*args, **kwargs)
        entered.set()
        await release.wait()
        return result

    monkeypatch.setattr(service, "locate_conversation_page", held)
    task = asyncio.create_task(
        screen.prepare_character_inspection(_context(db), is_current=lambda: current)
    )
    try:
        await asyncio.wait_for(entered.wait(), 2)
        if change == "database":
            owner.chachanotes_db = object()
        elif change == "service":
            owner.chat_conversation_scope_service = object()
        elif change == "generation":
            screen._library_navigation_context_generation += 1
        elif change == "authority":
            monkeypatch.setattr(db, "get_local_authority_id", lambda: "different")
        else:
            current = False
    finally:
        release.set()
    assert await task is None
    assert (
        screen._library_selected_row_id,
        screen._selected_conversation_id,
        screen._conversations_state.reader_state,
    ) == before


@pytest.mark.asyncio
async def test_missing_inspection_does_not_replace_existing_selection(library):
    _, screen, db = library
    assert (
        await screen.prepare_character_inspection(
            _context(db, "missing"), is_current=lambda: True
        )
        is None
    )
    assert screen._selected_conversation_id != "missing"


@pytest.mark.asyncio
@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize(
    "cancel",
    [
        False,
        True,
        "replace-visit",
        "commit-false",
        "projection-accept",
        "projection-cancel",
        "projection-authority",
    ],
)
async def test_switcher_waits_for_real_library_admission_and_cancellation(
    library, monkeypatch, warm, cancel
):
    from types import SimpleNamespace
    from typing import ClassVar

    from textual.containers import VerticalScroll
    from textual.screen import Screen
    from textual.widgets import Button, Input

    from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
    from Tests.UI.test_console_character_context import _controller
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Character_Chat.character_conversation_navigation import (
        CharacterConversationPage,
        CharacterConversationRow,
        UnavailableCharacterReason,
    )
    from tldw_chatbook.Chat.console_switcher_state import SwitcherMode
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    from tldw_chatbook.Widgets.Console.console_session_switcher_modal import (
        ACTIVE_PROJECTION_POLL_SECONDS,
        ConsoleSessionSwitcherModal,
    )

    owner, target, db = library

    class AdmissionApp(ConsolidatedCSSApp):
        CSS_PATH: ClassVar[list[str]] = [str(path) for path in APP_STYLESHEETS]
        _dispatch_screen_navigation = TldwCli._dispatch_screen_navigation
        handle_screen_navigation = TldwCli.handle_screen_navigation
        _screen_navigation_lock = TldwCli._screen_navigation_lock
        _handle_screen_navigation_locked = TldwCli._handle_screen_navigation_locked
        _complete_screen_navigation = TldwCli._complete_screen_navigation
        _navigation_outgoing_screen = TldwCli._navigation_outgoing_screen
        _dismiss_navigation_overlays = TldwCli._dismiss_navigation_overlays
        _navigation_target_owns_stack = TldwCli._navigation_target_owns_stack
        _navigation_overlay_awaiter_pending = staticmethod(
            TldwCli._navigation_overlay_awaiter_pending
        )
        _notify_navigation_failure = TldwCli._notify_navigation_failure
        _MAX_NAVIGATION_OVERLAY_DISMISSALS = TldwCli._MAX_NAVIGATION_OVERLAY_DISMISSALS

        def __init__(self):
            super().__init__()
            self.app_config = owner.app_config
            self.screen_state_store = owner.screen_state_store
            self.current_tab = "chat"
            self._initial_screen_pushed = True
            self._current_runtime_identity = owner._current_runtime_identity
            self._resolve_screen_navigation_target = lambda _name: (
                "library",
                "library",
                LibraryScreen,
            )
            self._ensure_screen_owned_css = lambda _name: None
            self._reusable_navigation_screen = lambda *_args: target
            self._clear_focus_if_leaving_console = lambda _name: None

        async def on_mount(self):
            self.install_screen(target, "admission-library")
            if warm:
                await self.push_screen(target)
                await self.pop_screen()
            await self.push_screen(Screen())

    app = AdmissionApp()
    console = SimpleNamespace(
        _character_context=_controller(database_accessor=lambda: db),
        post_message=app.post_message,
    )
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0
    projection = {"generation": 0, "receipt": "ready", "profile": ""}

    def active_projection():
        return (
            (),
            projection["profile"],
            "",
            projection["generation"],
            projection["receipt"],
        )

    accepts = cancel in (False, "projection-accept")
    service = owner.chat_conversation_scope_service
    locate = service.locate_conversation_page

    async def held(*args, **kwargs):
        nonlocal calls
        calls += 1
        value = await locate(*args, **kwargs)
        entered.set()
        await release.wait()
        return value

    async with app.run_test(size=(120, 50)) as pilot:
        monkeypatch.setattr(service, "locate_conversation_page", held)
        if cancel == "commit-false":
            monkeypatch.setattr(
                target, "commit_character_inspection", lambda _prepared: False
            )
        row = CharacterConversationRow.unavailable(
            UnresolvedConversationKey(db.get_local_authority_id(), "exact"),
            reason=UnavailableCharacterReason.MISSING_CARD,
            character_label="Historical",
            title="Exact local inspection",
            last_modified="2026-09-03T12:00:00Z",
            created_at="2026-09-01T00:00:00Z",
        )

        async def loader(**_kwargs):
            return CharacterConversationPage((row,), 1, None, 1)

        async def recover(entry, **kwargs):
            return await ChatScreen._open_console_character_library(
                console, entry, **kwargs
            )

        modal = ConsoleSessionSwitcherModal(
            character_loader=loader,
            character_open_library=recover,
            initial_mode=SwitcherMode.CHARACTER_CHATS,
            initial_character_query="needle",
            active_projection_loader=active_projection
            if str(cancel).startswith("projection-")
            else None,
        )
        await app.push_screen(modal)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        committed = modal._committed_character_result
        scroll_y = modal.query_one("#console-switcher-results", VerticalScroll).scroll_y
        modal.query_one("#console-switcher-recovery", Button).press()
        try:
            await asyncio.wait_for(entered.wait(), 3)
            assert app.screen is modal
            assert modal.query_one("#console-switcher-query", Input).value == "needle"
            assert (
                modal._committed_character_result.unresolved.conversation_id == "exact"
            )
            assert modal.query_one("#console-switcher-query", Input).disabled
            if str(cancel).startswith("projection-"):
                projection.update(generation=1, receipt="degraded")
                if cancel == "projection-authority":
                    projection["profile"] = "changed-profile"
                await pilot.pause(ACTIVE_PROJECTION_POLL_SECONDS + 0.1)
                if cancel != "projection-authority":
                    assert modal._active_projection_generation == 1
            if cancel in (True, "projection-cancel"):
                await pilot.press("escape")
                assert app.screen is modal
            elif cancel == "replace-visit":
                await modal.dismiss(None)
                replacement = ConsoleSessionSwitcherModal(
                    character_loader=loader, initial_mode=SwitcherMode.CHARACTER_CHATS
                )
                await app.push_screen(replacement)
        except BaseException:
            app.workers.cancel_group(app, "screen-navigation")
            raise
        finally:
            release.set()
        for _ in range(60):
            await pilot.pause(0.05)
            if cancel in ("replace-visit", "projection-authority"):
                if modal._activation_task.done():
                    break
                continue
            if (
                not accepts
                and modal.query_one("#console-switcher-recovery", Button).display
                and not modal._activation_in_flight
            ):
                break
            if (
                accepts
                and isinstance(app.screen, LibraryScreen)
                and app.screen._selected_conversation_id == "exact"
            ):
                break
        if cancel == "projection-authority":
            assert app.screen is not modal
            assert not isinstance(app.screen, LibraryScreen)
            assert target._selected_conversation_id != "exact"
        elif cancel == "replace-visit":
            assert app.screen is replacement
            assert target._selected_conversation_id != "exact"
        elif not accepts:
            assert app.screen is modal
            assert modal._activation_task.done()
            assert not modal._activation_in_flight
            assert not modal.query_one("#console-switcher-query", Input).disabled
            assert modal._committed_character_result is committed
            assert (
                modal.query_one("#console-switcher-results", VerticalScroll).scroll_y
                == scroll_y
            )
            assert modal.query_one("#console-switcher-query", Input).value == "needle"
            assert (
                modal._committed_character_result.unresolved.conversation_id == "exact"
            )
        else:
            assert isinstance(app.screen, LibraryScreen)
            await pilot.pause()
            assert app.screen._selected_conversation_id == "exact"
            assert app.screen._conversations_state.reader_state.selected_id == "exact"
            assert app.screen._pending_library_character_navigation is None
        assert calls == 1
        if cancel == "projection-cancel":
            await pilot.press("escape")
            assert app.screen is not modal


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "veto", ["file", "note", "prompt", "skill", "cancel-save", "exception"]
)
async def test_retained_library_save_veto_preserves_display_and_disposes_only_own_lease(
    library, monkeypatch, veto
):
    from types import SimpleNamespace

    from Tests.UI.consolidated_css import ConsolidatedCSSApp
    from tldw_chatbook.Library.library_notes_session import NoteFlushOutcomeKind

    _, screen, db = library
    app = ConsolidatedCSSApp()
    released = []
    current = True

    def acquire(kind):
        assert kind == "source"
        return lambda: released.append("own")

    async def file_flush():
        return veto != "file"

    async def note_flush():
        nonlocal current
        if veto == "exception":
            raise RuntimeError("save failed")
        if veto == "cancel-save":
            current = False
        return SimpleNamespace(
            kind=NoteFlushOutcomeKind.PERMITTED if veto != "note" else None
        )

    async def prompt_flush():
        return veto != "prompt"

    async def skill_flush():
        return veto != "skill"

    async with app.run_test(size=(120, 50)) as pilot:
        await app.push_screen(screen)
        await pilot.pause()
        before = (
            screen._library_selected_row_id,
            screen._selected_conversation_id,
            screen._conversations_state.reader_state,
        )
        monkeypatch.setattr(screen, "_flush_active_file_notes", file_flush)
        monkeypatch.setattr(screen, "_acquire_file_notes_transition", acquire)
        monkeypatch.setattr(screen, "_flush_library_note_save", note_flush)
        monkeypatch.setattr(screen, "_flush_library_prompt_save", prompt_flush)
        monkeypatch.setattr(screen, "_flush_library_skill_save", skill_flush)
        if veto == "exception":
            with pytest.raises(RuntimeError, match="save failed"):
                await screen.prepare_character_inspection(
                    _context(db), is_current=lambda: current
                )
        else:
            assert (
                await screen.prepare_character_inspection(
                    _context(db), is_current=lambda: current
                )
                is None
            )
        assert (
            screen._library_selected_row_id,
            screen._selected_conversation_id,
            screen._conversations_state.reader_state,
        ) == before
        assert released == ([] if veto == "file" else ["own"])
