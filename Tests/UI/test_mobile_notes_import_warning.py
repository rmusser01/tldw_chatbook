"""The startup warning is separate from action/failure status and contains no evidence."""

from dataclasses import replace

import pytest
from textual.app import App
from textual.widgets import Static

from tldw_chatbook.app import TldwCli
from tldw_chatbook.Library.library_notes_lasting_sync_state import (
    initial_lasting_sync_snapshot,
)
from tldw_chatbook.Widgets.Library.library_notes_sync_roots_canvas import (
    LibraryNotesSyncRootsCanvas,
)


@pytest.mark.asyncio
async def test_startup_warning_does_not_replace_action_status():
    snapshot = replace(
        initial_lasting_sync_snapshot(),
        phase="roots",
        status_line="Existing action failure",
        import_review_required=True,
    )

    class Host(App):
        CSS_PATH = TldwCli.CSS_PATH

        def compose(self):
            yield LibraryNotesSyncRootsCanvas(snapshot)

    app = Host()
    async with app.run_test() as pilot:
        await pilot.pause()
        status = app.query_one("#notes-sync-roots-status", Static)
        warning = app.query_one("#notes-sync-import-review-warning", Static)
        assert str(status.content) == "Existing action failure"
        assert (
            str(warning.content)
            == "At startup, some file links could not be added. Notes remain available in Notes; review folder setup before syncing."
        )
        assert warning._render_markup is False
        assert warning.has_class("library-notes-lasting-status")
