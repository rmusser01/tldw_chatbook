"""Workspace Buddy interaction leaves the underlying destination in place."""

from types import SimpleNamespace

import pytest
from textual.app import App
from textual.widgets import Button, Input

from tldw_chatbook.Persona_Buddy.inbox import BuddyInboxEntry
from tldw_chatbook.Persona_Buddy.interaction import BuddyBinding
from tldw_chatbook.Persona_Buddy.speech import BuddySpeechQueue


def entry(key="first"):
    return BuddyInboxEntry(
        key,
        "Research",
        "results",
        "Response ready",
        BuddyBinding(kind="conversation", target_id="live"),
        (key,),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(100, 40), (60, 20)])
async def test_workspace_open_is_read_only_and_actions_fit(size):
    from tldw_chatbook.Widgets.Persona_Widgets.buddy_workspace_modal import (
        BuddyWorkspaceModal,
    )

    opened, seen = [], []

    async def snapshot():
        return "Research workspace", (entry(),)

    app = App()

    async def play(*_):
        pytest.fail("Opening the inbox must not start speech")

    queue = BuddySpeechQueue(play)
    speech = SimpleNamespace(queue=queue, notice="", enabled=True, needs_consent=True)
    async with app.run_test(size=size) as pilot:
        underlying = app.screen
        modal = BuddyWorkspaceModal(
            snapshot=snapshot,
            open_entry=opened.append,
            acknowledge=seen.append,
            speech=speech,
        )
        app.push_screen(modal)
        await pilot.pause()
        assert seen == []
        assert not list(modal.query(Input))
        assert "voice" not in " ".join(
            str(b.label).lower() for b in modal.query(Button)
        )
        actions = modal.query_one("#buddy-inbox-actions")
        assert actions.region.bottom <= size[1]
        assert modal.query_one("#buddy-inbox-list").region.height >= 1
        speech_confirm = modal.query_one("#buddy-speech-confirm", Button)
        assert speech_confirm.region.right <= size[0]
        assert speech_confirm.region.bottom <= size[1]
        modal.query_one("#buddy-speech-pause", Button).press()
        await pilot.pause()
        assert queue.state.paused
        assert seen == []
        modal.query_one("#buddy-inbox-open", Button).press()
        await pilot.pause()
        assert opened == [entry()]
        assert seen == []
        modal.query_one("#buddy-inbox-seen", Button).press()
        await pilot.pause()
        assert seen == [entry()]
        modal.query_one("#buddy-inbox-close", Button).press()
        await pilot.pause()
        assert app.screen is underlying


@pytest.mark.asyncio
async def test_refresh_preserves_selected_result_and_focus_without_acknowledging():
    from tldw_chatbook.Widgets.Persona_Widgets.buddy_workspace_modal import (
        BuddyWorkspaceModal,
    )

    rows = [entry()]

    async def snapshot():
        return "Research", tuple(rows)

    seen = []
    app = App()
    async with app.run_test() as pilot:
        modal = BuddyWorkspaceModal(
            snapshot=snapshot, open_entry=lambda _: None, acknowledge=seen.append
        )
        app.push_screen(modal)
        await pilot.pause()
        modal.query_one("#buddy-inbox-close", Button).focus()
        await pilot.pause()
        rows.insert(0, entry("new"))
        await modal.refresh_inbox()
        assert modal.query_one("#buddy-inbox-close").has_focus
        assert modal.selected_entry() == entry()
        assert seen == []
