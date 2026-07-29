"""TASK-1300: Escape must dismiss every watchlist dialog.

Pressing Escape in the Rename dialog left it open — only clicking Cancel worked.
Because the dialog is modal, a keyboard user could neither back out nor do
anything else: during the third Watchlists UAT a `Delete` click was silently
swallowed by the still-open Rename dialog, which is exactly how it presents in
normal use — the app appears to ignore you.

All five dialogs declared `BINDINGS = []`, so no Escape binding existed on any of
them. The rest of the app uses `BINDINGS = [("escape", "cancel", "Cancel")]` with
an `action_cancel` (see `Widgets/embedding_template_selector.py`).

**Escape must dismiss with the same value the Cancel button uses**, which is not
`None` everywhere: `ConfirmDeleteDialog` cancels with `False`, and a caller
testing that result must not see `None` instead.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from tldw_chatbook.UI.Watchlists_Modules.opml_dialogs import (
    ConfirmDeleteDialog,
    OpmlExportDialog,
    OpmlImportDialog,
    WatchlistNameDialog,
    WatchlistSourcePickerDialog,
)

pytestmark = pytest.mark.unit


class _Host(App):
    """Minimal host that can push a dialog and record what it dismissed with."""

    def compose(self) -> ComposeResult:
        yield Static("host")


def _make(dialog_cls):
    """Construct each dialog with the arguments it actually requires."""
    if dialog_cls is ConfirmDeleteDialog:
        return ConfirmDeleteDialog("Daily")
    if dialog_cls is WatchlistNameDialog:
        return WatchlistNameDialog(
            dialog_title="Rename watchlist",
            submit_label="Rename",
            initial_name="Daily",
            taken_names=("Security Watch",),
        )
    if dialog_cls is WatchlistSourcePickerDialog:
        return WatchlistSourcePickerDialog("Daily", [])
    if dialog_cls is OpmlExportDialog:
        return OpmlExportDialog("<opml/>")
    return OpmlImportDialog()


#: Each dialog and the value its Cancel button dismisses with.
DIALOGS = [
    (OpmlImportDialog, None),
    (OpmlExportDialog, None),
    (ConfirmDeleteDialog, False),
    (WatchlistNameDialog, None),
    (WatchlistSourcePickerDialog, None),
]


@pytest.mark.parametrize(
    "dialog_cls, cancel_value", DIALOGS, ids=lambda v: getattr(v, "__name__", str(v))
)
@pytest.mark.asyncio
async def test_escape_dismisses_the_dialog(dialog_cls, cancel_value):
    """AC#1/#2: Escape closes the dialog, on every one of them."""
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        results: list[object] = []
        app.push_screen(_make(dialog_cls), results.append)
        await pilot.pause()
        assert isinstance(app.screen, dialog_cls), "the dialog never opened"

        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()

        assert not isinstance(app.screen, dialog_cls), (
            f"{dialog_cls.__name__} is still open after Escape; a keyboard user "
            "cannot back out of it, and being modal they cannot do anything else"
        )


@pytest.mark.parametrize(
    "dialog_cls, cancel_value", DIALOGS, ids=lambda v: getattr(v, "__name__", str(v))
)
@pytest.mark.asyncio
async def test_escape_dismisses_with_the_same_value_as_cancel(dialog_cls, cancel_value):
    """AC#3: Escape must leave the watchlist untouched, exactly as Cancel does.

    `ConfirmDeleteDialog` is the one that matters: it cancels with `False`, and a
    caller that receives `None` instead is being handed a different answer to
    "should I delete this?".
    """
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        results: list[object] = []
        app.push_screen(_make(dialog_cls), results.append)
        await pilot.pause()

        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()

        assert results, f"{dialog_cls.__name__} dismissed without a result"
        assert results[-1] is cancel_value, (
            f"{dialog_cls.__name__} dismissed with {results[-1]!r} on Escape but "
            f"{cancel_value!r} on Cancel"
        )
