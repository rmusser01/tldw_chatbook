"""Compact Buddy replies retain visible recovery and close controls."""

import pytest
from textual.widgets import Button, TextArea

from Tests.UI.test_buddy_conversation_modal import Harness
from tldw_chatbook.Persona_Buddy.interaction import BuddyBinding
from tldw_chatbook.UI.Navigation.buddy_conversation import open_buddy_conversation


@pytest.mark.asyncio
@pytest.mark.parametrize("available", [True, False])
async def test_compact_conversation_keeps_reply_and_recovery_actions_visible(available):
    app = Harness()
    async with app.run_test(size=(60, 20)) as pilot:
        binding = (
            BuddyBinding.for_session(app.target)
            if available
            else BuddyBinding(kind="conversation", target_id="missing")
        )
        underlying = app.screen
        modal = open_buddy_conversation(app, binding)
        await pilot.pause()
        for button in modal.query("#buddy-actions Button"):
            assert button.region.bottom <= 20
            assert button.region.right <= 60
            assert button.region.y >= 0
        assert modal.query_one("#buddy-reply", TextArea).region.height >= 3
        modal.query_one("#buddy-close", Button).press()
        await pilot.pause()
        assert app.screen is underlying


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(60, 20), (80, 24), (120, 40)])
async def test_transcript_opens_latest_follows_only_at_end_and_off_speech_is_compact(
    size,
):
    from textual.containers import VerticalScroll

    from tldw_chatbook.Chat.console_chat_store import ConsoleMessageRole

    app = Harness()
    for n in range(40):
        app.store.append_message(
            app.target.id,
            role=ConsoleMessageRole.USER,
            content=f"Message {n}\nReading line {n}",
        )
    async with app.run_test(size=size) as pilot:
        modal = open_buddy_conversation(
            app, BuddyBinding.for_session(app.target), allow_voice=False
        )
        await pilot.pause()
        body = modal.query_one("#buddy-conversation-body", VerticalScroll)
        assert body.max_scroll_y > 0
        assert body.scroll_y == body.max_scroll_y
        assert not modal.query_one("#buddy-speech-actions").display
        assert modal.query_one("#buddy-speech-status").region.height == 1
        assert body.region.height >= 3
        body.scroll_home(animate=False)
        await pilot.pause()
        app.store.append_message(
            app.target.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="New reply\nSecond line",
        )
        modal.refresh_projection()
        await pilot.pause()
        assert body.scroll_y == 0
        assert "new updates" in str(modal.query_one("#buddy-latest", Button).label)
        assert await pilot.click("#buddy-latest")
        await pilot.pause()
        assert body.scroll_y == body.max_scroll_y
        app.store.append_message(
            app.target.id, role=ConsoleMessageRole.ASSISTANT, content="Next reply"
        )
        modal.refresh_projection()
        await pilot.pause()
        assert body.scroll_y == body.max_scroll_y
        assert not modal.query("#buddy-mic")
        for button in modal.query("#buddy-actions Button"):
            assert button.region.bottom <= size[1]


@pytest.mark.asyncio
async def test_empty_transcript_displays_no_messages_yet():
    app = Harness()
    async with app.run_test(size=(80, 24)) as pilot:
        modal = open_buddy_conversation(app, BuddyBinding.for_session(app.target))
        await pilot.pause()
        assert "No messages yet" in str(modal.query_one("#buddy-transcript").render())


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(60, 20), (80, 24), (120, 40)])
async def test_pending_decision_is_reachable_while_reading_history(size):
    import asyncio

    from textual.containers import VerticalScroll

    from Tests.UI.test_buddy_conversation_modal import _pending_call, until
    from tldw_chatbook.Chat.console_chat_store import ConsoleMessageRole
    from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard

    app = Harness()
    for n in range(40):
        app.store.append_message(
            app.target.id, role=ConsoleMessageRole.USER, content=f"Message {n}"
        )
    async with app.run_test(size=size) as pilot:
        modal = open_buddy_conversation(
            app, BuddyBinding.for_session(app.target), allow_voice=False
        )
        await pilot.pause()
        body = modal.query_one("#buddy-conversation-body", VerticalScroll)
        body.scroll_home(animate=False)
        pending = asyncio.create_task(
            asyncio.to_thread(
                app.controller.request_mcp_approvals,
                [_pending_call()],
                session_id=app.target.id,
            )
        )
        try:
            await until(lambda: app.controller._interrupt_host.pending_total() == 1)
            modal.refresh_projection()
            await pilot.pause()
            assert body.scroll_y == 0
            assert modal.query_one("#buddy-pending", Button).display
            assert await pilot.click("#buddy-pending")
            await pilot.pause()
            card = modal.query_one(ChatApprovalCard)
            assert card.region.y < body.region.bottom
            assert card.region.bottom > body.region.y
            assert isinstance(app.focused, Button)
            assert app.focused.region.bottom <= body.region.bottom
            assert app.focused.region.y >= body.region.y
            # Coordinates alone can pass while a 1fr ancestor clips the action.
            for ancestor in app.focused.ancestors:
                if ancestor is body:
                    break
                assert ancestor.content_region.contains_region(app.focused.region)
            deny = card.query_one(".approval-row-fast-deny", Button)
            assert deny.region.right <= body.content_region.right
        finally:
            if not pending.done():
                app.controller.begin_shutdown()
                await pending
