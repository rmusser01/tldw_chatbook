"""Render and physical-message contracts for the Notes import-once canvas."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Button, Input, Static

from tldw_chatbook.Library.library_note_import_state import (
    LibraryNoteImportItemSnapshot,
    LibraryNoteImportSnapshot,
)
from tldw_chatbook.Widgets.Library.library_note_import_canvas import (
    LibraryNoteImportCanvas,
)
from tldw_chatbook.app import TldwCli

pytestmark = pytest.mark.asyncio


def _snapshot(**changes: object) -> LibraryNoteImportSnapshot:
    snapshot = LibraryNoteImportSnapshot(
        phase="select",
        selected_names=(),
        selection_kind="",
        destination="",
        status_line="Choose one or more files, or one folder.",
        preview_items=(),
        page=1,
        page_count=1,
        can_check=False,
        check_disabled_reason="Choose a source first.",
        can_import=False,
        import_disabled_reason="Check the selection before importing.",
    )
    return replace(snapshot, **changes)


def _item(**changes: object) -> LibraryNoteImportItemSnapshot:
    item = LibraryNoteImportItemSnapshot(
        item_id="item-1",
        name="draft.md",
        classification="new",
        action="create_new",
    )
    return replace(item, **changes)


def _plain(widget: Static) -> str:
    return getattr(widget.renderable, "plain", str(widget.renderable))


class _CanvasApp(App[None]):
    def __init__(self, snapshot: LibraryNoteImportSnapshot) -> None:
        super().__init__()
        self.snapshot = snapshot
        self.messages: list[object] = []

    def compose(self) -> ComposeResult:
        yield LibraryNoteImportCanvas(self.snapshot, id="import-canvas")

    def on_library_note_import_canvas_add_source_requested(
        self, message: LibraryNoteImportCanvas.AddSourceRequested
    ) -> None:
        self.messages.append(message)

    def on_library_note_import_canvas_destination_changed(
        self, message: LibraryNoteImportCanvas.DestinationChanged
    ) -> None:
        self.messages.append(message)

    def on_library_note_import_canvas_check_requested(
        self, message: LibraryNoteImportCanvas.CheckRequested
    ) -> None:
        self.messages.append(message)

    def on_library_note_import_canvas_collision_choice_requested(
        self, message: LibraryNoteImportCanvas.CollisionChoiceRequested
    ) -> None:
        self.messages.append(message)

    def on_library_note_import_canvas_collision_name_changed(
        self, message: LibraryNoteImportCanvas.CollisionNameChanged
    ) -> None:
        self.messages.append(message)

    def on_library_note_import_canvas_item_action_requested(
        self, message: LibraryNoteImportCanvas.ItemActionRequested
    ) -> None:
        self.messages.append(message)

    def on_library_note_import_canvas_item_choice_requested(
        self, message: LibraryNoteImportCanvas.ItemChoiceRequested
    ) -> None:
        self.messages.append(message)

    def on_library_note_import_canvas_uncertain_match_confirmed(
        self, message: LibraryNoteImportCanvas.UncertainMatchConfirmed
    ) -> None:
        self.messages.append(message)

    def on_library_note_import_canvas_import_requested(
        self, message: LibraryNoteImportCanvas.ImportRequested
    ) -> None:
        self.messages.append(message)

    def on_library_note_import_canvas_cancel_requested(
        self, message: LibraryNoteImportCanvas.CancelRequested
    ) -> None:
        self.messages.append(message)

    def on_library_note_import_canvas_retry_requested(
        self, message: LibraryNoteImportCanvas.RetryRequested
    ) -> None:
        self.messages.append(message)

    def on_library_note_import_canvas_page_requested(
        self, message: LibraryNoteImportCanvas.PageRequested
    ) -> None:
        self.messages.append(message)


async def test_file_selection_accumulates_literal_names_and_requires_destination() -> (
    None
):
    snapshot = _snapshot(
        selected_names=("draft[1].md", "notes.md"),
        selection_kind="files",
        status_line="2 files selected.",
        can_check=False,
        check_disabled_reason="Choose a Notes destination.",
    )
    app = _CanvasApp(snapshot)

    async with app.run_test(size=(80, 28)) as pilot:
        await pilot.pause()
        assert _plain(app.query_one("#note-import-source-summary", Static)) == (
            "2 files selected: draft[1].md, notes.md"
        )
        assert app.query_one("#note-import-add-source", Button).label.plain == (
            "Add another file"
        )
        assert app.query_one("#note-import-destination-label", Static)
        assert app.query_one("#note-import-destination", Input).value == ""

        check = app.query_one("#note-import-check", Button)
        assert check.disabled is True
        assert "Choose a Notes destination" in str(check.tooltip)
        assert "unavailable" in check.label.plain


async def test_folder_selection_is_exclusive_and_needs_no_destination() -> None:
    app = _CanvasApp(
        _snapshot(
            selected_names=("Research[2026]",),
            selection_kind="folder",
            status_line="1 folder selected.",
            can_check=True,
            check_disabled_reason="",
        )
    )

    async with app.run_test(size=(70, 24)) as pilot:
        await pilot.pause()
        assert not app.query("#note-import-add-source")
        assert not app.query("#note-import-destination")
        assert "Research[2026]" in _plain(
            app.query_one("#note-import-source-summary", Static)
        )
        assert app.query_one("#note-import-check", Button).disabled is False


async def test_select_controls_post_typed_physical_messages() -> None:
    app = _CanvasApp(
        _snapshot(
            selected_names=("draft.md",),
            selection_kind="files",
            destination="Inbox",
            can_check=True,
            check_disabled_reason="",
        )
    )

    async with app.run_test(size=(80, 28)) as pilot:
        assert await pilot.click("#note-import-add-source")
        destination = app.query_one("#note-import-destination", Input)
        destination.focus()
        await pilot.press("end", "space", "2")
        await pilot.pause()
        assert await pilot.click("#note-import-check")
        await pilot.pause()

    assert isinstance(app.messages[0], LibraryNoteImportCanvas.AddSourceRequested)
    changed = next(
        message
        for message in reversed(app.messages)
        if isinstance(message, LibraryNoteImportCanvas.DestinationChanged)
    )
    assert changed.destination == "Inbox 2"
    assert isinstance(app.messages[-1], LibraryNoteImportCanvas.CheckRequested)


async def test_checking_state_names_work_and_offers_cancel() -> None:
    app = _CanvasApp(
        _snapshot(
            phase="checking",
            status_line="◌ Checking 18 selected files…",
        )
    )

    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        status = _plain(app.query_one("#note-import-status", Static))
        assert status == "◌ Checking 18 selected files…"
        assert (
            app.query_one("#note-import-cancel", Button).label.plain == "Cancel check"
        )
        assert not app.query("#note-import-check")


@pytest.mark.parametrize(
    ("classification", "label"),
    (
        ("new", "New"),
        ("unchanged_repeat", "Unchanged repeat"),
        ("changed_repeat", "Changed repeat"),
        ("uncertain_match", "Uncertain match"),
        ("unsupported", "Unsupported"),
        ("failed", "Failed"),
    ),
)
async def test_review_groups_every_planner_classification(
    classification: str, label: str
) -> None:
    app = _CanvasApp(
        _snapshot(
            phase="review",
            status_line="Review 1 item before import.",
            preview_items=(_item(classification=classification),),
        )
    )

    async with app.run_test(size=(80, 28)) as pilot:
        await pilot.pause()
        assert label in _plain(app.query_one(".note-import-group-heading", Static))
        assert "draft.md" in _plain(app.query_one(".note-import-item-name", Static))


async def test_collision_review_exposes_three_explicit_choices_and_name_input() -> None:
    app = _CanvasApp(
        _snapshot(
            phase="review",
            collision_kind="root",
            collision_name="Work",
            collision_choice="",
            collision_reason="Choose how to handle the existing Work folder.",
        )
    )

    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        assert app.query_one("#note-import-collision-use-existing", Button)
        assert app.query_one("#note-import-collision-unique", Button)
        assert app.query_one("#note-import-collision-rename", Button)
        assert app.query_one("#note-import-collision-name", Input)
        reason = _plain(app.query_one("#note-import-collision-reason", Static))
        assert "Choose how" in reason


async def test_collision_controls_post_choice_and_proposed_name() -> None:
    app = _CanvasApp(
        _snapshot(
            phase="review",
            collision_kind="root",
            collision_name="Work",
            collision_reason="Choose how to handle the existing Work folder.",
        )
    )

    async with app.run_test(size=(80, 30)) as pilot:
        assert await pilot.click("#note-import-collision-unique")
        collision_name = app.query_one("#note-import-collision-name", Input)
        collision_name.focus()
        await pilot.press("end", "space", "2")
        await pilot.pause()

    choice = next(
        message
        for message in app.messages
        if isinstance(message, LibraryNoteImportCanvas.CollisionChoiceRequested)
    )
    name = next(
        message
        for message in reversed(app.messages)
        if isinstance(message, LibraryNoteImportCanvas.CollisionNameChanged)
    )
    assert choice.choice == "unique_sibling"
    assert name.name == "Work 2"


async def test_uncertain_update_requires_explicit_match_confirmation() -> None:
    item = _item(
        classification="uncertain_match",
        action="create_new",
        can_update=False,
        uncertain=True,
        confirmed=False,
        reason="Possible match; confirm before updating.",
    )
    app = _CanvasApp(_snapshot(phase="review", preview_items=(item,)))

    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        update = app.query_one("#note-import-action-item-1-update", Button)
        assert update.disabled is True
        assert "Confirm the match" in str(update.tooltip)
        assert await pilot.click("#note-import-confirm-item-1")
        await pilot.pause()

    message = app.messages[-1]
    assert isinstance(message, LibraryNoteImportCanvas.UncertainMatchConfirmed)
    assert message.item_id == "item-1"


async def test_update_choices_are_independent_and_post_item_scoped_messages() -> None:
    item = _item(
        classification="changed_repeat",
        action="update_existing",
        can_update=True,
        replace_content=True,
        add_membership=False,
    )
    app = _CanvasApp(_snapshot(phase="review", preview_items=(item,)))

    async with app.run_test(size=(80, 34)) as pilot:
        await pilot.pause()
        replace_button = app.query_one("#note-import-replace-item-1", Button)
        membership_button = app.query_one("#note-import-membership-item-1", Button)
        assert replace_button.label.plain.startswith("✓")
        assert membership_button.label.plain.startswith("○")
        assert await pilot.click(replace_button)
        assert await pilot.click(membership_button)
        await pilot.pause()

    choices = [
        message
        for message in app.messages
        if isinstance(message, LibraryNoteImportCanvas.ItemChoiceRequested)
    ]
    assert [
        (message.item_id, message.choice, message.enabled) for message in choices
    ] == [
        ("item-1", "replace_content", False),
        ("item-1", "add_membership", True),
    ]


async def test_review_pages_are_bounded_and_navigation_posts_direction() -> None:
    app = _CanvasApp(
        _snapshot(
            phase="review",
            preview_items=(_item(),),
            page=2,
            page_count=4,
        )
    )

    async with app.run_test(size=(70, 28)) as pilot:
        await pilot.pause()
        assert _plain(app.query_one("#note-import-page", Static)) == "Page 2 of 4"
        assert await pilot.click("#note-import-page-next")
        await pilot.pause()

    message = app.messages[-1]
    assert isinstance(message, LibraryNoteImportCanvas.PageRequested)
    assert message.delta == 1


async def test_import_gate_carries_disabled_reason_in_label_and_tooltip() -> None:
    app = _CanvasApp(
        _snapshot(
            phase="review",
            can_import=False,
            import_disabled_reason="Resolve the folder collision first.",
        )
    )

    async with app.run_test(size=(70, 24)) as pilot:
        await pilot.pause()
        import_button = app.query_one("#note-import-import", Button)
        assert import_button.disabled is True
        assert import_button.label.plain == "Import selected items unavailable"
        assert import_button.tooltip == "Resolve the folder collision first."


async def test_review_action_and_import_post_typed_physical_messages() -> None:
    app = _CanvasApp(
        _snapshot(
            phase="review",
            preview_items=(_item(),),
            can_import=True,
            import_disabled_reason="",
        )
    )

    async with app.run_test(size=(70, 26)) as pilot:
        assert await pilot.click("#note-import-action-item-1-skip")
        assert await pilot.click("#note-import-import")
        await pilot.pause()

    action = next(
        message
        for message in app.messages
        if isinstance(message, LibraryNoteImportCanvas.ItemActionRequested)
    )
    assert (action.item_id, action.action) == ("item-1", "skip")
    assert isinstance(app.messages[-1], LibraryNoteImportCanvas.ImportRequested)


async def test_importing_shows_bounded_progress_and_cooperative_cancel() -> None:
    app = _CanvasApp(
        _snapshot(
            phase="importing",
            status_line="Importing notes…",
            progress_completed=7,
            progress_total=12,
            progress_detail="7 imported · 1 skipped · 0 failed",
        )
    )

    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        assert _plain(app.query_one("#note-import-progress", Static)) == (
            "7 of 12 complete · 7 imported · 1 skipped · 0 failed"
        )
        assert await pilot.click("#note-import-cancel")
        await pilot.pause()

    assert isinstance(app.messages[-1], LibraryNoteImportCanvas.CancelRequested)


async def test_receipt_is_truthful_about_partial_cancel_and_retryable_failures() -> (
    None
):
    app = _CanvasApp(
        _snapshot(
            phase="receipt",
            status_line="Import stopped after the current item.",
            receipt_line="7 imported · 2 updated · 1 skipped · 2 failed",
            receipt_detail="Partial completion. Finished items were not rolled back.",
            retryable_failures=2,
        )
    )

    async with app.run_test(size=(65, 24)) as pilot:
        await pilot.pause()
        assert "2 failed" in _plain(app.query_one("#note-import-receipt", Static))
        assert "not rolled back" in _plain(
            app.query_one("#note-import-receipt-detail", Static)
        )
        assert await pilot.click("#note-import-retry")
        await pilot.pause()

    assert isinstance(app.messages[-1], LibraryNoteImportCanvas.RetryRequested)


async def test_dynamic_user_names_are_plain_text_not_markup() -> None:
    app = _CanvasApp(
        _snapshot(
            selected_names=("[bold]not-bold[/bold].md",),
            selection_kind="files",
        )
    )

    async with app.run_test(size=(70, 24)) as pilot:
        await pilot.pause()
        source = app.query_one("#note-import-source-summary", Static)
        assert source._render_markup is False
        assert "[bold]not-bold[/bold].md" in _plain(source)


async def test_valid_opaque_item_id_never_becomes_a_dom_identifier() -> None:
    item = _item(item_id="item:1.2")
    app = _CanvasApp(_snapshot(phase="review", preview_items=(item,)))

    async with app.run_test(size=(70, 24)) as pilot:
        await pilot.pause()
        skip = next(
            button
            for button in app.query(".note-import-item-action")
            if isinstance(button, Button)
            if button.name == "item:1.2:skip"
        )
        skip.press()
        await pilot.pause()

    message = app.messages[-1]
    assert isinstance(message, LibraryNoteImportCanvas.ItemActionRequested)
    assert (message.item_id, message.action) == ("item:1.2", "skip")


class _ProductionCssCanvasApp(App[None]):
    CSS_PATH = TldwCli.CSS_PATH

    def __init__(self, snapshot: LibraryNoteImportSnapshot) -> None:
        super().__init__()
        self.snapshot = snapshot

    def compose(self) -> ComposeResult:
        with Container(id="library-main"):
            with Container(id="library-canvas", classes="library-notes-compact"):
                yield LibraryNoteImportCanvas(
                    self.snapshot,
                    id="library-note-import-canvas",
                )


async def test_review_is_scrollable_and_paints_next_action_at_60_columns() -> None:
    items = tuple(
        _item(item_id=f"item-{index}", name=f"draft [{index}].md")
        for index in range(1, 9)
    )
    snapshot = _snapshot(
        phase="review",
        status_line="Review 8 items before import.",
        preview_items=items,
        can_import=True,
        import_disabled_reason="",
    )
    app = _ProductionCssCanvasApp(snapshot)

    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        canvas = app.query_one(LibraryNoteImportCanvas)
        assert canvas.allow_vertical_scroll is True
        assert canvas.virtual_size.height > canvas.container_size.height

        import_button = app.query_one("#note-import-import", Button)
        import_button.scroll_visible(animate=False, force=True, immediate=True)
        await pilot.pause()
        visible = app.screen._compositor.visible_widgets
        assert import_button in visible
        frame = app.export_screenshot()
        assert "Import selected items" in frame.replace("&#160;", " ")
        assert import_button.region.x + import_button.region.width <= 60


async def test_canvas_module_has_no_planner_storage_or_filesystem_dependencies() -> (
    None
):
    source = Path(
        "tldw_chatbook/Widgets/Library/library_note_import_canvas.py"
    ).read_text(encoding="utf-8")

    assert "note_import_planner" not in source
    assert "note_import_receipts" not in source
    assert "sqlite" not in source.lower()
    assert "pathlib" not in source
    assert "open(" not in source


async def test_library_widgets_package_exports_import_canvas() -> None:
    from tldw_chatbook.Widgets import Library as library_widgets

    assert library_widgets.LibraryNoteImportCanvas is LibraryNoteImportCanvas
    assert "LibraryNoteImportCanvas" in library_widgets.__all__
