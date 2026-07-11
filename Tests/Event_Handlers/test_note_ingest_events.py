# Tests/Event_Handlers/test_note_ingest_events.py
"""Tests for note ingestion event handlers.

T167: after a successful "Import Selected Notes Now" run, if the Library
screen is already mounted (the user was sitting on it while the import
ran), its local-source snapshot must be refreshed so the freshly-imported
notes show up without requiring the user to navigate away and back.
"""

from pathlib import Path
from unittest.mock import Mock

import pytest
from textual.css.query import QueryError
from textual.widgets import Button, Collapsible, RadioButton, TextArea

from tldw_chatbook.Event_Handlers.note_ingest_events import (
    handle_ingest_notes_import_now_button_pressed,
)
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen


def _make_mock_app(tmp_path: Path, *, screen: object) -> Mock:
    note_path = tmp_path / "my-note.md"
    note_path.write_text("# My Imported Note\n\nSome note content.\n", encoding="utf-8")

    app = Mock()
    app.selected_note_files_for_import = [note_path]
    app.notes_user_id = "user-1"
    app.notes_service = Mock()
    app.notes_service.add_note = Mock(return_value="note-id-1")
    app.notify = Mock()
    app.call_later = Mock()
    app.screen = screen

    widgets = {
        "#import-as-templates-radio": Mock(spec=RadioButton, value=False),
        "#ingest-notes-import-status-area": Mock(spec=TextArea, text=""),
        "#chat-notes-collapsible": Mock(spec=Collapsible),
    }

    def query_one_side_effect(selector, widget_type=None):
        try:
            return widgets[selector]
        except KeyError:
            raise QueryError(f"{selector} not found")

    app.query_one = Mock(side_effect=query_one_side_effect)

    captured_worker = {}

    def run_worker_side_effect(worker_callable, **kwargs):
        captured_worker["callable"] = worker_callable
        return Mock()

    app.run_worker = Mock(side_effect=run_worker_side_effect)
    app._captured_worker = captured_worker
    return app


@pytest.mark.asyncio
async def test_successful_note_import_refreshes_mounted_library_screen(tmp_path):
    """RED (T167): with Library mounted as the active screen, a successful
    notes import must trigger its local-source snapshot refresh so the
    newly-imported note appears without a manual re-visit."""
    library_screen = Mock(spec=LibraryScreen)
    app = _make_mock_app(tmp_path, screen=library_screen)

    await handle_ingest_notes_import_now_button_pressed(app, Button.Pressed(Mock(spec=Button)))

    worker_callable = app._captured_worker["callable"]
    await worker_callable()

    library_screen._refresh_local_source_snapshot.assert_called_once()


@pytest.mark.asyncio
async def test_successful_note_import_does_not_touch_non_library_screen(tmp_path):
    """A successful import while some other screen is active must not blow
    up (defensive isinstance guard) and obviously has nothing Library-shaped
    to call."""
    other_screen = Mock()
    app = _make_mock_app(tmp_path, screen=other_screen)

    await handle_ingest_notes_import_now_button_pressed(app, Button.Pressed(Mock(spec=Button)))

    worker_callable = app._captured_worker["callable"]
    # Must not raise even though `screen` has no _refresh_local_source_snapshot
    # of interest to us.
    await worker_callable()

    assert app.notes_service.add_note.called
