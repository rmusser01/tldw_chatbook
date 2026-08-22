"""Physical-message and production-CSS tests for Add from files."""

from __future__ import annotations

from dataclasses import replace

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Static

from tldw_chatbook.Library.library_notes_lasting_sync_state import (
    LastingSyncReview,
    LastingSyncReviewRow,
    initial_lasting_sync_snapshot,
)
from tldw_chatbook.Widgets.Library.library_notes_add_from_files_canvas import (
    LibraryNotesAddFromFilesCanvas,
)
from tldw_chatbook.app import TldwCli

pytestmark = pytest.mark.asyncio


class _Host(App[None]):
    CSS_PATH = TldwCli.CSS_PATH

    def __init__(self, snapshot) -> None:
        super().__init__()
        self.snapshot = snapshot
        self.messages: list[object] = []

    def compose(self) -> ComposeResult:
        yield LibraryNotesAddFromFilesCanvas(self.snapshot)

    def on_library_notes_add_from_files_canvas_relationship_requested(
        self, message
    ) -> None:
        self.messages.append(message)

    def on_library_notes_add_from_files_canvas_setup_changed(self, message) -> None:
        self.messages.append(message)

    def on_library_notes_add_from_files_canvas_folder_requested(self, message) -> None:
        self.messages.append(message)

    def on_library_notes_add_from_files_canvas_check_requested(self, message) -> None:
        self.messages.append(message)

    def on_library_notes_add_from_files_canvas_apply_requested(self, message) -> None:
        self.messages.append(message)

    def on_library_notes_add_from_files_canvas_page_requested(self, message) -> None:
        self.messages.append(message)

    def on_library_notes_add_from_files_canvas_back_requested(self, message) -> None:
        self.messages.append(message)

    def on_library_notes_add_from_files_canvas_attention_choice_requested(
        self, message
    ) -> None:
        self.messages.append(message)

    def on_library_notes_add_from_files_canvas_activate_requested(
        self, message
    ) -> None:
        self.messages.append(message)


def _frame(app: App[None]) -> str:
    return "\n".join(strip.text for strip in app.screen._compositor.render_strips())


async def test_chooser_explains_relationship_before_any_folder_input_and_posts_messages() -> (
    None
):
    app = _Host(initial_lasting_sync_snapshot())
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        assert not app.query("Input")
        assert "Later changes" in _frame(app)
        assert await pilot.click("#notes-add-import-once")
        assert await pilot.click("#notes-add-keep-synced")
        await pilot.pause()

    assert [message.relationship for message in app.messages] == [
        "import_once",
        "keep_synced",
    ]


async def test_configure_keeps_server_option_readable_but_disabled_and_preserves_brackets() -> (
    None
):
    snapshot = replace(
        initial_lasting_sync_snapshot(lasting_available=True), phase="configure"
    )
    app = _Host(snapshot)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        server = app.query_one("#notes-sync-destination-server", Button)
        assert server.disabled is True
        assert "Unavailable - server sync-folder capability not installed" in _frame(
            app
        )
        name = app.query_one("#notes-sync-display-name", Input)
        name.focus()
        await pilot.press(*"Research[2026]")
        await pilot.pause()

    changed = [message for message in app.messages if hasattr(message, "field")]
    assert changed[-1].value == "Research[2026]"


async def test_configure_posts_folder_direction_and_local_destination_messages() -> (
    None
):
    snapshot = replace(
        initial_lasting_sync_snapshot(lasting_available=True), phase="configure"
    )
    app = _Host(snapshot)
    async with app.run_test(size=(80, 28)) as pilot:
        assert await pilot.click("#notes-sync-folder-choose")
        assert await pilot.click("#notes-sync-direction-folder-to-notes")
        assert await pilot.click("#notes-sync-destination-local")
        await pilot.pause()
        painted = _frame(app)

    values = [
        (message.field, message.value)
        for message in app.messages
        if hasattr(message, "field")
    ]
    assert ("direction", "folder_to_notes") in values
    assert ("destination", "local") in values
    assert any(type(message).__name__ == "FolderRequested" for message in app.messages)
    assert "Local Library notes (selected)" in painted
    assert "Local destination ID" not in painted
    assert "✓ ⇄ Both ways" in painted


@pytest.mark.parametrize("phase", ("checking", "activating"))
async def test_running_phases_show_honest_wait_status_without_dead_cancel(
    phase: str,
) -> None:
    snapshot = replace(
        initial_lasting_sync_snapshot(lasting_available=True),
        phase=phase,
        status_line=f"{phase.title()}…",
    )
    app = _Host(snapshot)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        assert phase.title() in _frame(app)
        assert not app.query("#notes-sync-cancel")
        wait = app.query_one("#notes-sync-wait-status", Static)
        assert "Wait for the current step" in str(wait.renderable)


async def test_receipt_keeps_durable_status_and_back_visible() -> None:
    snapshot = replace(
        initial_lasting_sync_snapshot(lasting_available=True),
        phase="receipt",
        status_line="Finished.",
        receipt_line="1 applied · durable receipt recorded",
    )
    app = _Host(snapshot)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        assert "durable receipt" in _frame(app)
        assert app.query_one("#notes-sync-back", Button)


async def test_review_renders_effect_choices_paging_and_posts_physical_messages() -> (
    None
):
    review = LastingSyncReview(
        root_id="root-1",
        observation_token="c" * 64,
        safe_count=1,
        attention_count=1,
        rows=(
            LastingSyncReviewRow(
                "bind-1", "safe", "Update a Library note", action_id="act-1"
            ),
            LastingSyncReviewRow(
                "bind-2",
                "attention",
                "Both file and note changed",
                ("Keep file", "Keep note", "Keep both"),
            ),
        ),
        page=1,
        page_count=2,
        next_action="Apply reviewed",
    )
    app = _Host(
        replace(
            initial_lasting_sync_snapshot(lasting_available=True),
            phase="review",
            review=review,
            status_line="Review changes.",
        )
    )
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        assert "Keep file" in _frame(app)
        attention = app.query_one("#notes-sync-attention-1-0", Button)
        assert attention.disabled is True
        assert "unavailable in this release" in str(attention.tooltip)
        assert any(
            "Conflict and deletion choices are unavailable in this release"
            in str(copy.renderable)
            for copy in app.query(".library-disabled-reason")
        )
        await pilot.click(attention)
        assert app.query_one("#notes-sync-apply", Button).disabled is True
        next_page = app.query_one("#notes-sync-page-next", Button)
        next_page.scroll_visible(immediate=True)
        await pilot.pause()
        assert next_page in app.screen._compositor.visible_widgets
        assert await pilot.click(next_page)
        await pilot.pause()

    assert any(type(message).__name__ == "PageRequested" for message in app.messages)
    assert not any(
        type(message).__name__ == "AttentionChoiceRequested" for message in app.messages
    )


async def test_safe_review_apply_posts_only_visible_reviewed_action_ids() -> None:
    review = LastingSyncReview(
        root_id="root-1",
        observation_token="c" * 64,
        safe_count=1,
        rows=(
            LastingSyncReviewRow(
                "bind-1", "safe", "Update a Library note", action_id="act-1"
            ),
        ),
    )
    app = _Host(
        replace(
            initial_lasting_sync_snapshot(lasting_available=True),
            phase="review",
            review=review,
        )
    )
    async with app.run_test(size=(60, 20)) as pilot:
        assert await pilot.click("#notes-sync-apply")
        await pilot.pause()

    next(
        message
        for message in app.messages
        if type(message).__name__ == "ApplyRequested"
    )


async def test_activation_review_posts_distinct_activate_message() -> None:
    review = LastingSyncReview(
        root_id="root-1",
        observation_token="d" * 64,
        activation=True,
    )
    app = _Host(
        replace(
            initial_lasting_sync_snapshot(lasting_available=True),
            phase="review",
            review=review,
        )
    )
    async with app.run_test(size=(60, 20)) as pilot:
        assert await pilot.click("#notes-sync-activate")
        await pilot.pause()

    assert any(
        type(message).__name__ == "ActivateRequested" for message in app.messages
    )


async def test_long_deletion_choices_stack_and_remain_visible_at_60x20() -> None:
    review = LastingSyncReview(
        root_id="root-1",
        observation_token="e" * 64,
        attention_count=1,
        rows=(
            LastingSyncReviewRow(
                "bind-1",
                "attention",
                "One side was deleted",
                (
                    "Restore missing side",
                    "Delete/archive counterpart",
                    "Disconnect item",
                ),
            ),
        ),
    )
    app = _Host(
        replace(
            initial_lasting_sync_snapshot(lasting_available=True),
            phase="review",
            review=review,
        )
    )
    async with app.run_test(size=(60, 20)) as pilot:
        last = app.query_one("#notes-sync-attention-0-2", Button)
        last.scroll_visible(immediate=True)
        await pilot.pause()
        assert last.disabled is True
        assert last in app.screen._compositor.visible_widgets
        assert last.region.right <= 60


async def test_stale_review_replaces_apply_with_check_again() -> None:
    review = LastingSyncReview(stale=True, next_action="Check again")
    app = _Host(
        replace(
            initial_lasting_sync_snapshot(lasting_available=True),
            phase="review",
            review=review,
            status_line="The review is stale.",
        )
    )
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        assert not app.query("#notes-sync-apply")
        assert app.query_one("#notes-sync-check-again", Button)


async def test_empty_review_does_not_claim_hidden_scroll_content() -> None:
    app = _Host(
        replace(
            initial_lasting_sync_snapshot(lasting_available=True),
            phase="review",
        )
    )
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        assert not app.query("#notes-sync-fold-hint")


async def test_same_mode_snapshot_sync_retains_live_input_identity() -> None:
    snapshot = replace(
        initial_lasting_sync_snapshot(lasting_available=True), phase="configure"
    )
    app = _Host(snapshot)
    async with app.run_test(size=(70, 22)) as pilot:
        await pilot.pause()
        canvas = app.query_one(LibraryNotesAddFromFilesCanvas)
        before = app.query_one("#notes-sync-display-name", Input)
        before.value = "Typed draft"
        updated = replace(
            snapshot,
            setup=replace(
                snapshot.setup,
                direction="folder_to_notes",
                validation_message="Choose a folder.",
            ),
            status_line="Folder selected.",
        )
        canvas.sync_state(updated)
        await pilot.pause()
        after = app.query_one("#notes-sync-display-name", Input)

        assert after is before
        assert after.value == "Typed draft"
        assert "Choose a folder." in _frame(app)
        assert (
            app.query_one("#notes-sync-direction-folder-to-notes", Button).label.plain
            == "✓ → Folder to Notes"
        )


async def test_configure_canvas_is_contained_and_initial_focus_is_safe_at_60x20() -> (
    None
):
    snapshot = replace(
        initial_lasting_sync_snapshot(lasting_available=True), phase="configure"
    )
    app = _Host(snapshot)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        canvas = app.query_one(LibraryNotesAddFromFilesCanvas)
        region = canvas.region
        assert region.x >= 0 and region.y >= 0
        assert region.right <= 60 and region.bottom <= 20
        assert "Keep a folder synced" in _frame(app)
        hint = app.query_one("#notes-sync-fold-hint", Static)
        assert "Additional setup content is scrollable" in str(hint.renderable)
        assert "above" not in str(hint.renderable).casefold()
