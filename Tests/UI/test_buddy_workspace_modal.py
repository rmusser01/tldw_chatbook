"""Workspace Buddy interaction leaves the underlying destination in place."""

from types import SimpleNamespace

import pytest
from textual.widgets import Button, Input

from Tests.UI.consolidated_css import ConsolidatedCSSApp as App
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


@pytest.mark.asyncio
async def test_failed_refresh_blocks_enter_highlight_and_acknowledgement_until_recovery():
    from textual.widgets import OptionList

    from tldw_chatbook.Widgets.Persona_Widgets.buddy_workspace_modal import (
        BuddyWorkspaceModal,
    )

    failed = False
    opened, seen = [], []

    async def snapshot():
        if failed:
            raise ValueError("Workspace unavailable; retrying.")
        return "Research", (entry(),)

    app = App()
    async with app.run_test() as pilot:
        modal = BuddyWorkspaceModal(
            snapshot=snapshot, open_entry=opened.append, acknowledge=seen.append
        )
        app.push_screen(modal)
        await pilot.pause()
        failed = True
        await modal.refresh_inbox()
        listing = modal.query_one(OptionList)
        listing.focus()
        await pilot.press("enter")
        modal._sync_actions()
        assert modal.query_one("#buddy-inbox-open", Button).disabled
        await modal._mark_seen(entry())
        assert opened == seen == []
        failed = False
        await modal.refresh_inbox()
        await pilot.press("enter")
        assert opened == [entry()]


# --- TASK-35028: the two pre-gate poll writes are equality-gated --------------


def _spy_updates(widget):
    """Wrap ``widget.update`` so each call is recorded."""
    calls = []
    original_update = widget.update

    def counting_update(*args, **kwargs):
        calls.append(args)
        return original_update(*args, **kwargs)

    widget.update = counting_update
    return calls


@pytest.mark.asyncio
async def test_healthy_poll_repaints_title_and_error_only_on_change():
    """The 1 s self-poll rewrote 'Buddy · <title>' and '' every tick before
    the entries gate (TASK-34991); TASK-35028 gates each on its own copy."""
    from textual.widgets import Static

    from tldw_chatbook.Widgets.Persona_Widgets.buddy_workspace_modal import (
        BuddyWorkspaceModal,
    )

    title_holder = ["Research workspace"]

    async def snapshot():
        return title_holder[0], ()

    app = App()
    async with app.run_test(size=(100, 40)) as pilot:
        modal = BuddyWorkspaceModal(
            snapshot=snapshot, open_entry=lambda e: None, acknowledge=lambda e: None
        )
        app.push_screen(modal)
        await pilot.pause()
        title = modal.query_one("#buddy-inbox-title", Static)
        error = modal.query_one("#buddy-inbox-error", Static)
        title_calls = _spy_updates(title)
        error_calls = _spy_updates(error)
        # The mount's own refresh already painted both rows and consumed the
        # gates; reset them so the baseline here is a real first paint.
        modal._last_inbox_title = None
        modal._last_inbox_error = None
        await modal.refresh_inbox()
        await pilot.pause()
        assert len(title_calls) == 1 and len(error_calls) == 1
        painted_once_title, painted_once_error = len(title_calls), len(error_calls)
        await modal.refresh_inbox()
        await modal.refresh_inbox()
        assert len(title_calls) == painted_once_title, (
            "unchanged title copy must not repaint"
        )
        assert len(error_calls) == painted_once_error, (
            "unchanged (empty) error line must not repaint"
        )
        title_holder[0] = "Renamed workspace"
        await modal.refresh_inbox()
        await pilot.pause()
        assert len(title_calls) == painted_once_title + 1
        assert "Buddy · Renamed workspace" in str(title.renderable)


@pytest.mark.asyncio
async def test_persistent_snapshot_failure_rewrites_the_error_only_once():
    """A failing snapshot re-raises every tick; the error arm is gated too."""
    from textual.widgets import Static

    from tldw_chatbook.Widgets.Persona_Widgets.buddy_workspace_modal import (
        BuddyWorkspaceModal,
    )

    class Boom(ValueError):
        pass

    async def snapshot():
        raise Boom("storage is gone")

    app = App()
    async with app.run_test(size=(100, 40)) as pilot:
        modal = BuddyWorkspaceModal(
            snapshot=snapshot, open_entry=lambda e: None, acknowledge=lambda e: None
        )
        app.push_screen(modal)
        await pilot.pause()
        error = modal.query_one("#buddy-inbox-error", Static)
        error_calls = _spy_updates(error)
        # The mount's own refresh already wrote the failure and consumed the
        # gate; reset it so the baseline here is a real first paint.
        modal._last_inbox_error = None
        await modal.refresh_inbox()
        await pilot.pause()
        assert len(error_calls) == 1
        await modal.refresh_inbox()
        await modal.refresh_inbox()
        assert len(error_calls) == 1, "the same failure message must not repaint"
        assert "storage is gone" in str(error.renderable)
