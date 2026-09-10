"""Archive recovery through the current permanent Library reader and real SQLite."""

import pytest
from textual.widgets import Button, Input

from Tests.UI.test_library_shell import (
    LibraryHarness,
    _active_library_screen,
    _build_test_app,
    _wait_for_library_shell,
)
from tldw_chatbook.Constants import LIBRARY_NAV_CONTEXT_MODE


async def wait_until(pilot, predicate, debug=None):
    for _ in range(150):
        await pilot.pause(0.03)
        if predicate():
            return
    assert predicate(), debug() if debug else "condition not reached"


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(100, 30), (160, 44)])
async def test_real_saved_body_search_archive_restore_resume_and_undo(size, tmp_path):
    from tldw_chatbook.Chat.chat_conversation_scope_service import (
        ChatConversationScopeService,
    )
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(tmp_path / "library-archive.db", client_id="library-review")
    cid = db.add_conversation({"title": "Research notes"})
    db.add_message(
        {
            "conversation_id": cid,
            "sender": "user",
            "content": "x" * 7996 + " needle evidence",
        }
    )
    app = _build_test_app()
    local = ChatConversationService(db)
    app.local_chat_conversation_service = local
    app.chat_conversation_scope_service = ChatConversationScopeService(
        local_service=local, server_service=None
    )
    app.console_runtime = None
    registry = app.workspace_registry_service
    registry.link_membership(
        registry.ensure_default_workspace().workspace_id,
        item_type="conversation",
        item_id=cid,
        title="Research notes",
    )
    calls = []
    app.resume_console_conversation = lambda identity: calls.append(
        ("resume", identity)
    )
    app.open_chat_with_handoff = lambda payload, **kw: calls.append(
        ("source", payload.source_id)
    )
    host = LibraryHarness(app)
    try:
        async with host.run_test(size=size) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            screen.apply_navigation_context(
                {
                    LIBRARY_NAV_CONTEXT_MODE: "conversations",
                    "conversation_archive_scope": "active",
                    "conversation_query": "needle",
                }
            )
            await wait_until(
                pilot,
                lambda: (
                    screen._conversations_state.reader_state.loaded_actions_eligible
                ),
                lambda: (
                    screen._conversations_state.total,
                    screen._conversations_state.reader_state,
                    screen._conversations_state.error,
                ),
            )
            assert screen._conversations_state.total == 1
            assert (
                screen.query_one("#library-conversations-filter", Input).value
                == "needle"
            )
            reader = screen.query_one("#library-conversation-reader")
            assert "needle evidence" in reader.state.messages[0].text
            find = screen.query_one("#library-conversation-reader-find", Input)
            find.value = "needle"
            find.focus()
            await pilot.press("enter")
            await wait_until(
                pilot,
                lambda: bool(screen._conversations_state.reader_state.find_matches),
            )
            assert (
                screen._conversations_state.reader_state.find_matches[0].message_offset
                == 7997
            )
            archive = screen.query_one("#library-conversation-archive", Button)
            archive.press()
            await wait_until(pilot, lambda: host.screen is not screen)
            host.screen.query_one("#confirm-button", Button).press()
            recovery = screen._conversation_recovery()
            await wait_until(
                pilot,
                lambda: (
                    bool(recovery.receipt_versions)
                    and not screen._conversations_state.loading
                    and host.screen is screen
                ),
            )
            assert local.get_conversation_archive_states([cid]) == {cid: True}
            assert screen._conversations_state.total == 0
            await wait_until(
                pilot,
                lambda: bool(screen.query("#library-conversations-view-archived")),
            )
            screen.query_one("#library-conversations-view-archived", Button).press()
            await wait_until(
                pilot,
                lambda: (
                    screen._conversations_state.total == 1
                    and screen._conversations_state.reader_state.loaded_actions_eligible
                ),
            )
            await pilot.pause(0.3)
            assert screen._conversations_state.requested_query == "needle"
            resume = screen.query_one("#library-conversation-open-console", Button)
            assert str(resume.label) == "Restore and resume"
            source = screen.query_one("#library-conversation-use-source", Button)
            assert resume in screen.focus_chain
            assert source in screen.focus_chain, (
                source.disabled,
                source.tooltip,
                reader.loaded_metadata,
                screen._library_workspace_depth_state(),
            )
            resume.focus()
            await pilot.press("enter")
            await wait_until(pilot, lambda: bool(calls))
            assert calls == [("resume", cid)]
            source.press()
            await wait_until(pilot, lambda: len(calls) == 2)
            assert calls[-1] == ("source", cid)
            (tmp_path / f"library-archive-port-{size[0]}.svg").write_text(
                host.export_screenshot()
            )
            await recovery.undo()
            await wait_until(pilot, lambda: not screen._conversations_state.loading)
            assert local.get_conversation_archive_states([cid]) == {cid: False}
            assert screen._conversations_state.total == 0
            assert not recovery.receipt_versions
            screen.query_one("#library-conversations-scope-all", Button).press()
            await wait_until(
                pilot,
                lambda: (
                    screen._conversations_state.total == 1
                    and not screen._conversations_state.loading
                ),
            )
            assert screen._conversation_recovery().scope == "all"
            assert screen._conversations_state.requested_query == "needle"
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_find_previous_and_next_cycle_canonical_reader_matches(widget_pilot):
    from textual.widgets import Static

    from tldw_chatbook.Library.library_conversation_reader_state import (
        ConversationMessageView,
        ConversationReaderState,
        set_conversation_find_query,
    )
    from tldw_chatbook.Widgets.Library.library_conversation_reader import (
        LibraryConversationReader,
    )

    state = ConversationReaderState(
        selected_id="c",
        selected_version=2,
        loaded_id="c",
        loaded_version=2,
        loaded_generation=1,
        generation=1,
        complete=True,
        message_total=2,
        messages=(
            ConversationMessageView("m1", "user", "2026-09-10", "r1", 6, "needle"),
            ConversationMessageView("m2", "assistant", "2026-09-10", "r2", 6, "needle"),
        ),
    )
    state = set_conversation_find_query(state, "needle")
    async with await widget_pilot(
        LibraryConversationReader, state=state, id="reader"
    ) as pilot:
        reader = pilot.app.query_one("#reader", LibraryConversationReader)
        reader.query_one("#library-conversation-find-next", Button).press()
        await pilot.pause()
        assert "Match 1 of 2" in str(
            reader.query_one("#library-conversation-find-position", Static).renderable
        )
        reader.query_one("#library-conversation-find-next", Button).press()
        await pilot.pause()
        assert "message 2" in str(
            reader.query_one("#library-conversation-find-position", Static).renderable
        )
        reader.query_one("#library-conversation-find-previous", Button).press()
        await pilot.pause()
        assert "message 1" in str(
            reader.query_one("#library-conversation-find-position", Static).renderable
        )
        assert reader.state.loaded_id == "c"
