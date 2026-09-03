import threading

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from types import SimpleNamespace

import pytest

from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library.library_export_canvas import LibraryExportCanvas
from tldw_chatbook.Library.library_export_state import build_library_export_form_state
from tldw_chatbook.Library.library_export_scope import ExportScope


def test_cancel_apply_ignores_stale_run():
    calls = []
    fake = SimpleNamespace(
        # Task 4 cleanup: the screen's flat `_library_export_<field>` shim
        # is gone -- `_apply_library_export_cancelled`'s body now reads
        # `self._export_state.<field>`, so this fake nests its export
        # fields under `_export_state` (recipe §11's "unbound fake-self"
        # retarget precedent).
        _export_state=SimpleNamespace(
            run_id=9,
            running=True,
            status="Packaging archive…  1/5",
            error="x",
        ),
        _update_library_export_canvas_after_run=lambda: calls.append("update"),
    )
    LibraryScreen._apply_library_export_cancelled(fake, 4)  # 4 != 9
    assert fake._export_state.running is True
    assert calls == []


def test_cancel_apply_current_run_sets_cancelled_status():
    calls = []
    fake = SimpleNamespace(
        _export_state=SimpleNamespace(
            run_id=9,
            running=True,
            status="Packaging archive…  1/5",
            error="x",
        ),
        _update_library_export_canvas_after_run=lambda: calls.append("update"),
    )
    LibraryScreen._apply_library_export_cancelled(fake, 9)
    assert fake._export_state.running is False
    assert fake._export_state.status == "Export cancelled."
    assert fake._export_state.error == ""
    assert calls == ["update"]


def test_cancel_handler_sets_event():
    fake = SimpleNamespace(
        _export_state=SimpleNamespace(
            cancel_event=threading.Event(),
            running=True,
            status="",
        ),
        _refresh_library_export_status_line=lambda: None,
    )
    LibraryScreen.handle_library_export_cancel(fake, None)
    assert fake._export_state.cancel_event.is_set()
    assert fake._export_state.status == "Cancelling…"


@pytest.mark.asyncio
async def test_cancel_button_visible_only_while_running():
    from textual.app import App

    def _state(running):
        return build_library_export_form_state(
            scope=ExportScope(kind="everything"),
            counts={"total": 3},
            name="n",
            description="",
            media_quality="thumbnail",
            destination="/tmp/x.zip",
            running=running,
            status_line="Exporting…" if running else "",
        )

    class Host(ConsolidatedCSSApp):
        def compose(self):
            yield LibraryExportCanvas(_state(True), id="library-export-canvas")

    app = Host()
    async with app.run_test() as pilot:
        cancel = pilot.app.query_one("#library-export-cancel")
        assert cancel.display is True
