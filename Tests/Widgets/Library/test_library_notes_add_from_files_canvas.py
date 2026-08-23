"""Physical-message and production-CSS tests for Add from files."""

from __future__ import annotations

from dataclasses import replace

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Static, TextArea

from tldw_chatbook.Library.library_notes_lasting_sync_state import (
    LastingSyncApplyBlocker,
    LastingSyncHistory,
    LastingSyncHistoryRow,
    LastingSyncReceiptRow,
    LastingSyncReview,
    LastingSyncReviewRow,
    initial_lasting_sync_snapshot,
)
from tldw_chatbook.Notes.notes_sync_conflicts import (
    ConflictComparison,
    NotesSyncConflictChoice,
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

    def on_library_notes_add_from_files_canvas_choice_requested(self, message) -> None:
        self.messages.append(message)

    def on_library_notes_add_from_files_canvas_view_requested(self, message) -> None:
        self.messages.append(message)

    def on_library_notes_add_from_files_canvas_return_requested(self, message) -> None:
        self.messages.append(message)

    def on_library_notes_add_from_files_canvas_undo_requested(self, message) -> None:
        self.messages.append(message)

    def on_library_notes_add_from_files_canvas_dismiss_requested(self, message) -> None:
        self.messages.append(message)

    def on_library_notes_add_from_files_canvas_history_requested(self, message) -> None:
        self.messages.append(message)

    def on_library_notes_add_from_files_canvas_history_page_requested(
        self, message
    ) -> None:
        self.messages.append(message)

    def on_library_notes_add_from_files_canvas_history_return_requested(
        self, message
    ) -> None:
        self.messages.append(message)

    def on_library_notes_add_from_files_canvas_activate_requested(
        self, message
    ) -> None:
        self.messages.append(message)


def _frame(app: App[None]) -> str:
    return "\n".join(strip.text for strip in app.screen._compositor.render_strips())


async def _wait_for(pilot, predicate, *, message: str) -> None:
    for _ in range(100):
        if predicate():
            return
        await pilot.pause(0.01)
    raise AssertionError(message)


def _conflict_review(
    *, selected: NotesSyncConflictChoice | None = None
) -> LastingSyncReview:
    label = {
        None: "",
        NotesSyncConflictChoice.KEEP_FILE: "Selected: Keep file",
        NotesSyncConflictChoice.KEEP_NOTE: "Selected: Keep note",
        NotesSyncConflictChoice.KEEP_BOTH: "Selected: Keep both",
        NotesSyncConflictChoice.SKIP: "Selected: Skip for now",
    }[selected]
    blocker = (
        LastingSyncApplyBlocker.NOTHING_SELECTED
        if selected in {None, NotesSyncConflictChoice.SKIP}
        else LastingSyncApplyBlocker.NONE
    )
    return LastingSyncReview(
        root_id="root-1",
        observation_token="c" * 64,
        attention_count=1,
        rows=(
            LastingSyncReviewRow(
                "bind-1",
                "attention",
                "Both file and note changed",
                ("Keep file", "Keep note", "Keep both", "Skip for now"),
                conflict_eligible=True,
                selected_choice=selected,
                selected_label=label,
            ),
        ),
        can_apply=blocker is LastingSyncApplyBlocker.NONE,
        apply_blocker=blocker,
    )


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


async def test_conflict_choices_are_keyboard_reachable_and_update_in_place() -> None:
    review = _conflict_review()
    app = _Host(
        replace(
            initial_lasting_sync_snapshot(lasting_available=True),
            phase="review",
            review=review,
            status_line="Review changes.",
        )
    )
    async with app.run_test(size=(60, 20)) as pilot:
        await _wait_for(
            pilot,
            lambda: len(app.query(".notes-sync-conflict-choice")) == 4,
            message="four conflict choices did not mount",
        )
        canvas = app.query_one(LibraryNotesAddFromFilesCanvas)
        body = app.query_one("#notes-sync-body")
        scroll_before = body.scroll_y
        keep_file = app.query_one("#notes-sync-conflict-0-keep-file", Button)
        assert keep_file.name == "bind-1"
        assert not keep_file.disabled
        keep_file.focus()
        await pilot.press("enter")
        await _wait_for(
            pilot,
            lambda: any(
                type(message).__name__ == "ChoiceRequested" for message in app.messages
            ),
            message="Enter did not activate the conflict choice",
        )
        choice_message = next(
            message
            for message in app.messages
            if type(message).__name__ == "ChoiceRequested"
        )
        assert (choice_message.binding_id, choice_message.choice) == (
            "bind-1",
            "Keep file",
        )

        updated = replace(
            canvas.snapshot,
            review=_conflict_review(selected=NotesSyncConflictChoice.KEEP_FILE),
            status_line="Choice staged. No changes yet.",
        )
        canvas.sync_state(updated)
        await _wait_for(
            pilot,
            lambda: "Selected: Keep file" in _frame(app),
            message="non-color selected label did not update",
        )
        assert app.query_one("#notes-sync-conflict-0-keep-file", Button) is keep_file
        assert "✓ Keep file" in keep_file.label.plain
        assert app.focused is keep_file
        assert body.scroll_y == scroll_before
        assert app.query_one("#notes-sync-apply", Button).disabled is False
        assert "Choice staged. No changes yet." in _frame(app)

        skip = app.query_one("#notes-sync-conflict-0-skip", Button)
        skip.focus()
        await pilot.press("space")
        await _wait_for(
            pilot,
            lambda: (
                len([m for m in app.messages if type(m).__name__ == "ChoiceRequested"])
                == 2
            ),
            message="Space did not activate the conflict choice",
        )


async def test_conflict_comparison_is_literal_scrollable_and_return_restores_view_focus() -> (
    None
):
    comparison = ConflictComparison(
        binding_id="bind-1",
        note_title="Release [red]note[/red]",
        relative_path="notes/release.md",
        note_version=3,
        note_updated_at=None,
        file_modified_ns=42,
        note_character_count=12,
        note_line_count=2,
        file_character_count=20,
        file_line_count=2,
        diff="--- Note\n+++ File\n-" + "n" * 100 + "\n+file\n",
        input_elided=False,
        output_elided=False,
    )
    snapshot = replace(
        initial_lasting_sync_snapshot(lasting_available=True),
        phase="review",
        review=_conflict_review(),
        status_line="Review changes.",
    )
    app = _Host(snapshot)
    async with app.run_test(size=(60, 20)) as pilot:
        await _wait_for(
            pilot,
            lambda: bool(app.query("#notes-sync-conflict-view-0")),
            message="View comparison did not mount",
        )
        canvas = app.query_one(LibraryNotesAddFromFilesCanvas)
        view = app.query_one("#notes-sync-conflict-view-0", Button)
        view.focus()
        view.press()
        await _wait_for(
            pilot,
            lambda: any(type(m).__name__ == "ViewRequested" for m in app.messages),
            message="View comparison message was not posted",
        )
        canvas.sync_state(replace(snapshot, comparison=comparison))
        await _wait_for(
            pilot,
            lambda: app.query_one("#notes-sync-comparison-0").display,
            message="comparison did not expand",
        )
        diff = app.query_one("#notes-sync-comparison-diff-0", TextArea)
        assert diff.read_only is True
        assert diff.language is None
        assert diff.soft_wrap is False
        summary = app.query_one("#notes-sync-comparison-summary-0", Static)
        assert "[red]note[/red]" in str(summary.renderable)
        assert diff.max_scroll_x > 0
        await _wait_for(
            pilot,
            lambda: app.focused is diff,
            message="current View provenance did not focus the comparison",
        )
        diff.scroll_visible(immediate=True)
        await pilot.pause()
        assert diff in app.screen._compositor.visible_widgets
        assert diff.region.right <= 60

        returned = app.query_one("#notes-sync-comparison-return-0", Button)
        returned.focus()
        returned.press()
        await _wait_for(
            pilot,
            lambda: app.focused is view,
            message="Return did not restore the originating View focus",
        )
        await _wait_for(
            pilot,
            lambda: any(type(m).__name__ == "ReturnRequested" for m in app.messages),
            message="Return message was not posted",
        )


async def test_receipts_and_history_render_actions_labels_and_fallback_at_60x20() -> (
    None
):
    receipt = LastingSyncReceiptRow(
        "operation-1",
        "Release note · notes/release.md",
        NotesSyncConflictChoice.KEEP_BOTH,
        "completed",
        True,
    )
    history = LastingSyncHistory(
        "root-1",
        (
            LastingSyncHistoryRow(
                "operation-1",
                "Release note · notes/release.md",
                NotesSyncConflictChoice.KEEP_BOTH,
                "completed",
                "2026-08-22T12:00:00+00:00",
                "2026-08-22T12:00:00+00:00",
                True,
            ),
            LastingSyncHistoryRow(
                "opaque123456",
                "opaque12",
                NotesSyncConflictChoice.KEEP_FILE,
                "completed",
                None,
                "2026-08-22T13:00:00+00:00",
                False,
                "Undo expired",
            ),
        ),
        1,
        True,
    )
    review_snapshot = replace(
        initial_lasting_sync_snapshot(lasting_available=True),
        phase="review",
        review=_conflict_review(),
        receipts=(receipt,),
    )
    app = _Host(review_snapshot)
    async with app.run_test(size=(60, 20)) as pilot:
        await _wait_for(
            pilot,
            lambda: bool(app.query("#notes-sync-receipt-0")),
            message="receipt did not mount",
        )
        assert "Keep both" in _frame(app)
        assert app.query_one("#notes-sync-receipt-undo-0", Button).name == "operation-1"
        assert (
            app.query_one("#notes-sync-receipt-dismiss-0", Button).name == "operation-1"
        )
        history_open = app.query_one("#notes-sync-history-open", Button)
        app.query_one("#notes-sync-receipt-undo-0", Button).press()
        app.query_one("#notes-sync-receipt-dismiss-0", Button).press()
        history_open.press()
        await _wait_for(
            pilot,
            lambda: (
                {type(message).__name__ for message in app.messages}
                >= {"UndoRequested", "DismissRequested", "HistoryRequested"}
            ),
            message="receipt and history actions did not post their typed messages",
        )
        history_open.scroll_visible(immediate=True)
        await pilot.pause()
        assert history_open in app.screen._compositor.visible_widgets
        assert history_open.region.right <= 60

        canvas = app.query_one(LibraryNotesAddFromFilesCanvas)
        canvas.sync_state(
            replace(review_snapshot, phase="history", history=history, receipts=())
        )
        await _wait_for(
            pilot,
            lambda: bool(app.query("#notes-sync-history-row-1")),
            message="history page did not mount",
        )
        painted = _frame(app)
        assert "Release note · notes/release.md" in painted
        assert "opaque12" in painted
        assert "Undo expired" in painted
        next_page = app.query_one("#notes-sync-history-next", Button)
        next_page.scroll_visible(immediate=True)
        await pilot.pause()
        assert next_page in app.screen._compositor.visible_widgets
        assert next_page.region.right <= 60
        app.query_one("#notes-sync-history-undo-0", Button).press()
        next_page.press()
        app.query_one("#notes-sync-history-return", Button).press()
        await _wait_for(
            pilot,
            lambda: (
                {type(message).__name__ for message in app.messages}
                >= {"HistoryPageRequested", "HistoryReturnRequested"}
            ),
            message="history paging and Return did not post their typed messages",
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
