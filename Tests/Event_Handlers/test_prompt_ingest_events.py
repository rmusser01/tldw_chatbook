# Tests/Event_Handlers/test_prompt_ingest_events.py
"""Task 172: the prompt "Import Now" completion callbacks were dead code
(nothing dispatched the file_operations worker group). These assert that a
successful import invokes the success callback and a catastrophic worker
failure invokes the failure callback and re-raises."""
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from textual.css.query import QueryError
from textual.widgets import Button, TextArea

from tldw_chatbook.Event_Handlers.prompt_ingest_events import (
    handle_ingest_prompts_import_now_button_pressed,
)


def _make_mock_app() -> Mock:
    app = Mock()
    app.selected_prompt_files_for_import = [Path("p.json")]
    app.notify = Mock()
    app.call_later = Mock()

    status_area = Mock(spec=TextArea, text="")
    widgets = {"#prompt-import-status-area": status_area}

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
async def test_successful_prompt_import_invokes_success_callback():
    """A successful prompt import dispatches process_prompt_import_success,
    which writes the results summary to the status area."""
    app = _make_mock_app()
    results = [{"status": "success", "file_path": "p.json", "prompt_name": "P", "message": "ok"}]
    with patch("tldw_chatbook.Event_Handlers.prompt_ingest_events.prompts_db_initialized", return_value=True), \
         patch("tldw_chatbook.Event_Handlers.prompt_ingest_events.import_prompts_from_files", return_value=results):
        await handle_ingest_prompts_import_now_button_pressed(app, Button.Pressed(Mock(spec=Button)))
        await app._captured_worker["callable"]()

    # The success callback is the ONLY code path that calls load_text on the
    # status area (the trigger only assigns `.text`).
    app._status_area.load_text.assert_called_once()
    assert "Summary:" in app._status_area.load_text.call_args.args[0]


@pytest.mark.asyncio
async def test_failed_prompt_import_invokes_failure_callback_and_reraises():
    """A catastrophic prompt import failure dispatches
    process_prompt_import_failure (an error-severity toast) and re-raises so
    Textual still records the worker error."""
    app = _make_mock_app()
    with patch("tldw_chatbook.Event_Handlers.prompt_ingest_events.prompts_db_initialized", return_value=True), \
         patch("tldw_chatbook.Event_Handlers.prompt_ingest_events.import_prompts_from_files",
               side_effect=RuntimeError("boom")):
        await handle_ingest_prompts_import_now_button_pressed(app, Button.Pressed(Mock(spec=Button)))
        with pytest.raises(RuntimeError):
            await app._captured_worker["callable"]()

    # failure callback surfaced an error-severity toast
    assert any(c.kwargs.get("severity") == "error" for c in app.notify.call_args_list)
