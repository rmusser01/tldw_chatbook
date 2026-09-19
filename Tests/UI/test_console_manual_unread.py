"""Manual reminder lifecycle through a mounted Console and real SQLite marks."""

import asyncio

from textual.widgets import Button

from Tests.UI.test_console_left_rail import make_console_pilot
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.conversation_local_marks_service import (
    ConversationLocalMarksService,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Widgets.Console.console_conversation_action_menu import (
    ConsoleConversationActionMenu,
)


async def test_current_unread_survives_repaint_and_clears_after_tab_revisit(tmp_path):
    async with make_console_pilot(production_styles=True) as pilot:
        screen = pilot.app.screen
        db = CharactersRAGDB(str(tmp_path / "marks.sqlite"), client_id="unread")
        marks = ConversationLocalMarksService(db)
        screen.app_instance.chachanotes_db = db
        screen.app_instance.conversation_local_marks_service = marks
        conversations = ChatConversationService(db)
        store = screen._ensure_console_chat_store()
        a = next(
            item for item in store.sessions() if item.id == store.active_session_id
        )
        a.persisted_conversation_id = conversations.create_conversation(title="A")
        b = store.create_session(title="B", activate=False)
        b.persisted_conversation_id = conversations.create_conversation(title="B")
        await screen._sync_native_console_chat_ui()
        screen._workspace._sync_console_workspace_context()
        await pilot.pause()
        # The real menu dispatch must write the same durable reminder checked below.
        opener = next(
            button
            for button in screen.query(".console-conversation-actions")
            if button.conversation_id == a.persisted_conversation_id
        )
        opener.press()
        await pilot.pause()
        menu = screen.query_one(ConsoleConversationActionMenu)
        menu.query_one("#console-conversation-action-mark_unread", Button).press()
        async with asyncio.timeout(5):
            while not await asyncio.to_thread(
                marks.unread_token, a.persisted_conversation_id
            ):
                await asyncio.sleep(0.01)
        await screen._sync_native_console_chat_ui()
        await screen._session._activate_native_console_session(a.id)
        assert marks.unread_token(a.persisted_conversation_id) is not None
        await screen._session._activate_native_console_session(b.id)
        await screen._session._activate_native_console_session(a.id)
        async with asyncio.timeout(5):
            while await asyncio.to_thread(
                marks.unread_token, a.persisted_conversation_id
            ):
                await asyncio.sleep(0.01)
        assert screen._console_visible_send_session_id() == a.id


import pytest


@pytest.mark.parametrize(
    "change", ["remark", "profile", "service", "navigation", "unpainted", "wrong_chat"]
)
async def test_late_read_acknowledgement_cannot_clear_a_new_or_unseen_reminder(
    tmp_path, monkeypatch, change
):
    async with make_console_pilot(production_styles=True) as pilot:
        screen = pilot.app.screen
        db = CharactersRAGDB(str(tmp_path / "marks.sqlite"), client_id="unread")
        marks = ConversationLocalMarksService(db)
        screen.app_instance.conversation_local_marks_service = marks
        conversations = ChatConversationService(db)
        store = screen._ensure_console_chat_store()
        active = next(
            item for item in store.sessions() if item.id == store.active_session_id
        )
        active.persisted_conversation_id = conversations.create_conversation(
            title="Reminder"
        )
        cid = active.persisted_conversation_id
        await asyncio.to_thread(marks.mark_unread, cid)
        visit = await screen._session.begin_manual_read_visit(
            active.id, allow_current=True
        )
        assert visit is not None
        monkeypatch.setattr(
            screen._session, "_painted_session_accessor", lambda: active.id
        )
        if change == "remark":
            await asyncio.to_thread(marks.mark_unread, cid)
        elif change == "profile":
            monkeypatch.setattr(
                screen._session, "_manual_profile_key", lambda: "another-profile"
            )
        elif change == "service":
            screen.app_instance.conversation_local_marks_service = (
                ConversationLocalMarksService(db)
            )
        elif change == "navigation":
            await screen._session.begin_manual_read_visit(active.id)
        elif change == "unpainted":
            monkeypatch.setattr(
                screen._session, "_painted_session_accessor", lambda: None
            )
        else:
            store.create_session(title="Elsewhere", activate=True)
        assert not await screen._session._acknowledge_manual_read_visit(visit)
        assert marks.unread_token(cid) is not None


async def test_duplicate_tabs_and_cancelled_activation_preserve_manual_unread(tmp_path):
    async with make_console_pilot(production_styles=True) as pilot:
        screen = pilot.app.screen
        db = CharactersRAGDB(str(tmp_path / "marks.sqlite"), client_id="unread")
        marks = ConversationLocalMarksService(db)
        screen.app_instance.conversation_local_marks_service = marks
        conversations = ChatConversationService(db)
        store = screen._ensure_console_chat_store()
        a = next(
            item for item in store.sessions() if item.id == store.active_session_id
        )
        a.persisted_conversation_id = conversations.create_conversation(
            title="Same chat"
        )
        duplicate = store.create_session(title="Duplicate tab", activate=False)
        duplicate.persisted_conversation_id = a.persisted_conversation_id
        marks.mark_unread(a.persisted_conversation_id)
        await screen._session._activate_native_console_session(duplicate.id)
        await pilot.pause()
        assert marks.unread_token(a.persisted_conversation_id) is not None
        b = store.create_session(title="Other", activate=False)
        await screen._session._activate_native_console_session(b.id)
        await screen._session._activate_native_console_session(
            a.id, activate_if=lambda: False
        )
        await pilot.pause()
        assert store.active_session_id == b.id
        assert marks.unread_token(a.persisted_conversation_id) is not None


@pytest.mark.parametrize("size", [(80, 24), (120, 40), (160, 48)])
@pytest.mark.parametrize("ascii_mode", [False, True])
async def test_compact_row_keeps_action_geometry_and_keyboard_focus(
    tmp_path, size, ascii_mode
):
    from tldw_chatbook.Widgets.glyph_fallback import set_ascii_glyph_mode

    set_ascii_glyph_mode(ascii_mode)
    try:
        async with make_console_pilot(size=size, production_styles=True) as pilot:
            screen = pilot.app.screen
            db = CharactersRAGDB(str(tmp_path / "marks.sqlite"), client_id="unread")
            marks = ConversationLocalMarksService(db)
            screen.app_instance.conversation_local_marks_service = marks
            conversations = ChatConversationService(db)
            store = screen._ensure_console_chat_store()
            active = next(
                item for item in store.sessions() if item.id == store.active_session_id
            )
            active.title = "日本語 [literal] " * 8
            active.persisted_conversation_id = conversations.create_conversation(
                title=active.title
            )
            await screen._sync_native_console_chat_ui()
            screen._workspace._sync_console_workspace_context()
            await pilot.pause()
            if not screen.query_one("#console-left-rail").display:
                screen.action_toggle_console_context_rail()
                await pilot.pause()
            row = next(
                button
                for button in screen.query(".console-conversation-compact-row")
                if button.conversation_id == active.persisted_conversation_id
            )
            opener = next(
                button
                for button in screen.query(".console-conversation-actions")
                if button.conversation_id == active.persisted_conversation_id
            )
            row.scroll_visible(animate=False, immediate=True)
            await pilot.pause()
            assert row.region.height == 1
            assert row.region.right <= opener.region.x
            initial = opener.region
            marks.mark_unread(active.persisted_conversation_id)
            screen._workspace._sync_console_workspace_context()
            async with asyncio.timeout(5):
                while True:
                    await pilot.pause()
                    opener = next(
                        button
                        for button in screen.query(".console-conversation-actions")
                        if button.conversation_id == active.persisted_conversation_id
                    )
                    if ("[unread]" if ascii_mode else "✉") in str(opener.label):
                        break
            assert (opener.region.x, opener.region.width, opener.region.height) == (
                initial.x,
                initial.width,
                initial.height,
            )
            row = next(
                button
                for button in screen.query(".console-conversation-compact-row")
                if button.conversation_id == active.persisted_conversation_id
            )
            row.focus()
            await pilot.pause()
            assert screen.focused is row, repr(screen.focused)
            await pilot.press("m")
            async with asyncio.timeout(5):
                while not screen.query(ConsoleConversationActionMenu):
                    await pilot.pause()
            menu = screen.query_one(ConsoleConversationActionMenu)
            assert menu.target.conversation_id == active.persisted_conversation_id
            assert menu.region.bottom <= screen.region.bottom
            await pilot.press("escape")
            await pilot.pause()
            assert screen.focused.id == row.id
            from tldw_chatbook.Widgets.Console.console_composer_bar import (
                ConsoleComposerBar,
            )

            composer = screen.query_one(ConsoleComposerBar)
            composer.load_draft("")
            composer.focus()
            await pilot.press("m")
            await pilot.pause()
            assert composer.draft_text() == "m"
            assert not screen.query(ConsoleConversationActionMenu)
    finally:
        set_ascii_glyph_mode(False)
