from __future__ import annotations

import ast
import inspect
import textwrap
from unittest.mock import Mock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button

from tldw_chatbook.UI.MediaWindow_v2 import MediaWindow
from tldw_chatbook.UI.Screens.media_runtime_state import MediaRuntimeState
from tldw_chatbook.Widgets.Media.media_viewer_panel import MediaViewerPanel


class MediaViewerTestApp(App[None]):
    def __init__(self, panel: MediaViewerPanel):
        super().__init__()
        self.panel = panel

    def compose(self) -> ComposeResult:
        yield self.panel


def _media_app(runtime_backend: str = "local") -> Mock:
    app = Mock()
    app._media_types_for_ui = []
    app.media_runtime_state = MediaRuntimeState(runtime_backend=runtime_backend)
    app.notify = Mock()
    app.open_chat_with_handoff = Mock()
    return app


def test_media_viewer_builds_use_in_chat_event_for_loaded_media():
    panel = MediaViewerPanel(Mock())
    panel.media_data = {"id": "media-1", "title": "Lecture", "content": "Transcript"}

    event = panel._build_use_in_chat_event()

    assert event.media_data["id"] == "media-1"
    assert event.media_data["title"] == "Lecture"


def test_media_viewer_clear_display_does_not_use_bare_except():
    tree = ast.parse(textwrap.dedent(inspect.getsource(MediaViewerPanel.clear_display)))

    bare_handlers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler) and node.type is None
    ]
    assert bare_handlers == []


@pytest.mark.asyncio
async def test_media_viewer_use_in_chat_button_tracks_loaded_media():
    panel = MediaViewerPanel(Mock())
    app = MediaViewerTestApp(panel)

    async with app.run_test() as pilot:
        button = panel.query_one("#media-use-in-chat-button", Button)
        assert button.disabled is True

        panel.load_media({"id": "media-1", "title": "Lecture", "content": "Transcript"})
        await pilot.pause()

        assert button.disabled is False


def test_media_window_builds_handoff_from_hydrated_detail():
    app = _media_app(runtime_backend="server")
    app.media_runtime_state.selected_record_id = "record-1"
    app.media_runtime_state.detail_by_record_id["record-1"] = {
        "id": "record-1",
        "title": "Lecture",
        "content": "Transcript",
        "url": "https://example.com",
        "media_type": "video",
    }
    window = MediaWindow(app)
    window.runtime_state = app.media_runtime_state
    window.viewer_panel = Mock()
    window.viewer_panel.media_data = {"id": "record-1", "title": "Fallback"}

    payload = window._build_current_media_chat_handoff_payload()

    assert payload.source == "media"
    assert payload.item_type == "media"
    assert payload.runtime_backend == "server"
    assert payload.source_owner == "server"
    assert payload.source_selector_state == "server"
    assert payload.discovery_entity_id == "record-1"
    assert payload.source_id == "record-1"
    assert payload.title == "Lecture"
    assert payload.body == "Transcript"
    assert payload.metadata["url"] == "https://example.com"
    assert payload.metadata["media_type"] == "video"


def test_media_window_use_in_chat_handler_routes_event_to_app():
    app = _media_app()
    window = MediaWindow(app)
    window.runtime_state = app.media_runtime_state
    window.viewer_panel = Mock()
    window.viewer_panel.media_data = None
    event = MediaViewerPanel.UseInChatRequested(
        {"id": "media-1", "title": "Lecture", "content": "Transcript"}
    )

    window.handle_media_use_in_chat(event)

    payload = app.open_chat_with_handoff.call_args.args[0]
    assert payload.source == "media"
    assert payload.source_id == "media-1"
    assert payload.body == "Transcript"


def test_media_window_use_in_chat_unavailable_explains_recovery():
    app = _media_app()
    app.open_chat_with_handoff = None
    window = MediaWindow(app)
    window.runtime_state = app.media_runtime_state
    window.viewer_panel = Mock()
    window.viewer_panel.media_data = None
    event = MediaViewerPanel.UseInChatRequested(
        {"id": "media-1", "title": "Lecture", "content": "Transcript"}
    )

    window.handle_media_use_in_chat(event)

    message = app.notify.call_args.args[0]
    assert "Use in Chat is unavailable" in message
    assert "Open Chat" in message
    assert "try again" in message
    assert app.notify.call_args.kwargs["severity"] == "warning"
