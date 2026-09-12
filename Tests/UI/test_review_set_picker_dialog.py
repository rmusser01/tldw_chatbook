"""Contract tests for the Library review-set picker dialog (task-28243).

The dialog is dumb: it renders pre-computed ``(set_id, name, progress_label,
active)`` rows (built by ``build_picker_rows``) and dismisses with an
``(action, set_id)`` decision -- ``("open", id)`` to resume/switch,
``("dismiss", id)`` to soft-delete, or ``None`` on cancel. All service work
happens in the screen's worker, never here.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from tldw_chatbook.Widgets.Library.library_review_set_picker import (
    LibraryReviewSetPickerDialog,
)

ROWS = [
    ("s1", "All media", "2 of 5 · 1 reviewed", True),
    ("s2", "pdf items", "All 3 reviewed", False),
]


class ModalHarness(App[None]):
    """Minimal host capturing the picker's typed dismissal result."""

    def __init__(self, rows: list[tuple[str, str, str, bool]]) -> None:
        super().__init__()
        self.rows = rows
        self.results: list[tuple[str, str] | None] = []

    def compose(self) -> ComposeResult:
        yield Static("Library")

    def show(self) -> None:
        self.push_screen(
            LibraryReviewSetPickerDialog(self.rows),
            callback=self.results.append,
        )


@pytest.mark.asyncio
async def test_rows_render_name_progress_and_active_marker() -> None:
    app = ModalHarness(ROWS)
    async with app.run_test(size=(100, 30)) as pilot:
        app.show()
        await pilot.pause()
        opens = list(app.screen.query(".library-review-set-open"))
        labels = [str(button.label) for button in opens]

        assert len(opens) == 2
        assert "All media" in labels[0] and "2 of 5 · 1 reviewed" in labels[0]
        assert labels[0].startswith("✓")  # the active set carries the marker
        assert "pdf items" in labels[1] and not labels[1].startswith("✓")


@pytest.mark.asyncio
async def test_picking_a_set_dismisses_with_open_decision() -> None:
    app = ModalHarness(ROWS)
    async with app.run_test(size=(100, 30)) as pilot:
        app.show()
        await pilot.pause()
        opens = list(app.screen.query(".library-review-set-open"))
        opens[1].press()
        await pilot.pause()

        assert app.results == [("open", "s2")]


@pytest.mark.asyncio
async def test_dismiss_button_dismisses_with_dismiss_decision() -> None:
    app = ModalHarness(ROWS)
    async with app.run_test(size=(100, 30)) as pilot:
        app.show()
        await pilot.pause()
        dismisses = list(app.screen.query(".library-review-set-dismiss"))
        assert len(dismisses) == 2
        dismisses[0].press()
        await pilot.pause()

        assert app.results == [("dismiss", "s1")]


@pytest.mark.asyncio
async def test_empty_rows_show_empty_copy_and_close_returns_none() -> None:
    app = ModalHarness([])
    async with app.run_test(size=(100, 30)) as pilot:
        app.show()
        await pilot.pause()
        copy = str(
            app.screen.query_one("#library-review-set-picker-empty", Static).renderable
        )
        assert "No saved review sets" in copy
        assert not list(app.screen.query(".library-review-set-open"))

        app.screen.query_one("#library-review-set-picker-close", Button).press()
        await pilot.pause()
        assert app.results == [None]


@pytest.mark.asyncio
async def test_read_later_action_dismisses_with_read_later_decision() -> None:
    """The picker's "Review read-later" action resolves its own decision.

    task-28244: the picker is the set hub, so the action lives in its
    actions row -- present even with no saved sets, since the read-later
    queue is independent of them.
    """
    app = ModalHarness([])
    async with app.run_test(size=(100, 30)) as pilot:
        app.show()
        await pilot.pause()
        app.screen.query_one(
            "#library-review-set-picker-read-later", Button
        ).press()
        await pilot.pause()

        assert app.results == [("read_later", "")]


@pytest.mark.asyncio
async def test_escape_cancels_with_none() -> None:
    app = ModalHarness(ROWS)
    async with app.run_test(size=(100, 30)) as pilot:
        app.show()
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

        assert app.results == [None]


@pytest.mark.asyncio
async def test_many_rows_scroll_and_close_stays_reachable() -> None:
    # Qodo #2337: rows must live in a scrolling region so a long set list
    # cannot push lower rows and the Close action past the modal's height cap.
    rows = [
        (f"s{n}", f"Set {n}", "1 of 1 · 0 reviewed", False) for n in range(40)
    ]
    app = ModalHarness(rows)
    async with app.run_test(size=(100, 30)) as pilot:
        app.show()
        await pilot.pause()
        modal = app.screen
        scroll = modal.query_one("#library-review-set-picker-rows")
        assert len(scroll.query(".library-review-set-row")) == 40
        actions = modal.query_one("#library-review-set-picker-actions")
        assert scroll not in actions.ancestors  # Close is outside the scroll

        close = modal.query_one("#library-review-set-picker-close", Button)
        close.press()
        await pilot.pause()
        assert app.results == [None]


@pytest.mark.asyncio
async def test_markup_in_set_names_renders_literally() -> None:
    # Set names derive from user search queries -- hostile markup must not
    # style the label or crash compose (home/library escape_markup lesson).
    app = ModalHarness([("s1", "[red]evil[/red]", "1 of 1 · 0 reviewed", False)])
    async with app.run_test(size=(100, 30)) as pilot:
        app.show()
        await pilot.pause()
        button = app.screen.query_one(".library-review-set-open", Button)

        assert "[red]evil[/red]" in str(button.label)
