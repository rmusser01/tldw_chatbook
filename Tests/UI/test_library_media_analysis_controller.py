"""Analysis behavior follows the current Library owners without a mounted app."""

from types import MethodType, SimpleNamespace

import pytest

from tldw_chatbook.UI.Library_Modules.library_media_controller import (
    LibraryMediaController,
)
from tldw_chatbook.UI.Library_Modules.library_media_state import LibraryMediaState
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen


@pytest.mark.asyncio
async def test_analysis_persistence_reads_replaced_service_and_selection_ports():
    calls = []

    async def service_call(method, **kwargs):
        return method(**kwargs)

    async def refreshed(media_id):
        calls.append(("refresh", media_id))

    async def reprojected(media_id):
        calls.append(("reproject", media_id))

    app = SimpleNamespace(media_reading_scope_service=None)
    screen = SimpleNamespace(
        app_instance=app,
        _media_state=LibraryMediaState(selected_media_id="old", editing_analysis=True),
    )
    save = MethodType(LibraryScreen._save_library_media_analysis, screen)
    screen._run_library_service_call = service_call
    screen._library_media_backing_id = lambda media_id: int(media_id)
    screen._refresh_library_media_detail = refreshed
    screen._reproject_library_media_analysis_row = reprojected
    screen._media_state.selected_media_id = "7"
    app.media_reading_scope_service = SimpleNamespace(
        save_analysis_version=lambda **kwargs: calls.append(("save", kwargs))
    )
    assert await save(
        "7", content="body", analysis_content="summary", viewer_owned=False
    )
    assert calls == [
        (
            "save",
            {
                "mode": "local",
                "media_id": 7,
                "content": "body",
                "analysis_content": "summary",
                "isolate_in_worker": True,
            },
        ),
        ("reproject", "7"),
        ("refresh", "7"),
    ]
    assert screen._media_state.editing_analysis is True


def test_media_framework_worker_property_reads_the_current_screen_callback():
    screen = SimpleNamespace(run_worker=lambda: "original worker")
    controller = object.__new__(LibraryMediaController)
    controller._screen = screen
    assert controller.run_worker is screen.run_worker
    assert controller.run_worker() == "original worker"
    screen.run_worker = lambda: "replacement worker"
    assert controller.run_worker is screen.run_worker
    assert controller.run_worker() == "replacement worker"
