# Tests/Event_Handlers/test_character_ingest_events.py
"""Task 172: the character "Import Now" completion callbacks were dead code
(nothing dispatched the file_operations worker group). These assert that a
successful import invokes the success callback and a catastrophic worker
failure invokes the failure callback and re-raises."""
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from textual.css.query import QueryError
from textual.widgets import Button, TextArea

from tldw_chatbook.Event_Handlers.character_ingest_events import (
    handle_ingest_characters_import_now_button_pressed,
)


class _BoomList:
    """Truthy (passes the trigger's `if not …` guard) but raises on iteration,
    so `import_worker_char` fails at the loop -- the only way to exercise the
    catastrophic-failure path, since the worker swallows per-file errors."""
    def __bool__(self):
        return True
    def __iter__(self):
        raise RuntimeError("boom")


def _make_mock_app(*, selected) -> Mock:
    app = Mock()
    app.selected_character_files_for_import = selected
    app.notes_user_id = "user-1"
    app.notes_service = Mock()
    app.notes_service._get_db = Mock(return_value=Mock())
    app.notify = Mock()
    app.call_later = Mock()
    app._chat_character_filter_populated = False

    status_area = Mock(spec=TextArea, text="")
    widgets = {"#ingest-character-import-status-area": status_area}

    def query_one_side_effect(selector, widget_type=None):
        try:
            return widgets[selector]
        except KeyError:
            raise QueryError(f"{selector} not found")

    app.query_one = Mock(side_effect=query_one_side_effect)

    captured = {}
    def run_worker_side_effect(worker_callable, **kwargs):
        captured["callable"] = worker_callable
        return Mock()
    app.run_worker = Mock(side_effect=run_worker_side_effect)
    app._captured_worker = captured
    app._status_area = status_area
    return app


@pytest.mark.asyncio
async def test_successful_character_import_invokes_success_callback():
    app = _make_mock_app(selected=[Path("c.png")])
    with patch("tldw_chatbook.Event_Handlers.character_ingest_events.ccl.import_and_save_character_from_file",
               return_value=123), \
         patch("tldw_chatbook.Event_Handlers.character_ingest_events.ccl.load_character_card_from_file",
               return_value={"name": "TestChar"}):
        await handle_ingest_characters_import_now_button_pressed(app, Button.Pressed(Mock(spec=Button)))
        await app._captured_worker["callable"]()

    # success callback writes the summary (the LAST load_text call) and sets the flag
    assert "Summary:" in app._status_area.load_text.call_args.args[0]
    assert app._chat_character_filter_populated is True


@pytest.mark.asyncio
async def test_failed_character_import_invokes_failure_callback_and_reraises():
    app = _make_mock_app(selected=_BoomList())
    await handle_ingest_characters_import_now_button_pressed(app, Button.Pressed(Mock(spec=Button)))
    with pytest.raises(RuntimeError):
        await app._captured_worker["callable"]()

    assert any(c.kwargs.get("severity") == "error" for c in app.notify.call_args_list)
