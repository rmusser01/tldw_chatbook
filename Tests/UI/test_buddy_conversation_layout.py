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
