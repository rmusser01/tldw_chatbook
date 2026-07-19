import threading
from types import SimpleNamespace

import pytest

from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library.library_export_canvas import LibraryExportCanvas
from tldw_chatbook.Library.library_export_state import build_library_export_form_state
from tldw_chatbook.Library.library_export_scope import ExportScope


def test_cancel_apply_ignores_stale_run():
    calls = []
    fake = SimpleNamespace(
        _library_export_run_id=9, _library_export_running=True,
        _library_export_status="Packaging archive…  1/5", _library_export_error="x",
        _update_library_export_canvas_after_run=lambda: calls.append("update"),
    )
    LibraryScreen._apply_library_export_cancelled(fake, 4)  # 4 != 9
    assert fake._library_export_running is True
    assert calls == []


def test_cancel_apply_current_run_sets_cancelled_status():
    calls = []
    fake = SimpleNamespace(
        _library_export_run_id=9, _library_export_running=True,
        _library_export_status="Packaging archive…  1/5", _library_export_error="x",
        _update_library_export_canvas_after_run=lambda: calls.append("update"),
    )
    LibraryScreen._apply_library_export_cancelled(fake, 9)
    assert fake._library_export_running is False
    assert fake._library_export_status == "Export cancelled."
    assert fake._library_export_error == ""
    assert calls == ["update"]


def test_cancel_handler_sets_event():
    fake = SimpleNamespace(
        _library_export_cancel_event=threading.Event(),
        _library_export_running=True, _library_export_status="",
        _refresh_library_export_status_line=lambda: None,
    )
    LibraryScreen.handle_library_export_cancel(fake, None)
    assert fake._library_export_cancel_event.is_set()
    assert fake._library_export_status == "Cancelling…"


@pytest.mark.asyncio
async def test_cancel_button_visible_only_while_running():
    from textual.app import App

    def _state(running):
        return build_library_export_form_state(
            scope=ExportScope(kind="everything"), counts={"total": 3}, name="n",
            description="", media_quality="thumbnail", destination="/tmp/x.zip",
            running=running, status_line="Exporting…" if running else "",
        )

    class Host(App):
        def compose(self):
            yield LibraryExportCanvas(_state(True), id="library-export-canvas")

    app = Host()
    async with app.run_test() as pilot:
        cancel = pilot.app.query_one("#library-export-cancel")
        assert cancel.display is True
