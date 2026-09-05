"""Analysis ports work without a mounted Textual application."""

from types import SimpleNamespace

import pytest

from tldw_chatbook.UI.Library_Modules.media_analysis_wiring import (
    build_library_media_analysis_controller,
)


@pytest.mark.asyncio
async def test_analysis_persistence_reads_replaced_service_and_selection_ports():
    calls = []

    async def service_call(method, **kwargs):
        return method(**kwargs)

    async def refreshed(media_id):
        calls.append(("refresh", media_id))

    app = SimpleNamespace(media_reading_scope_service=None)
    screen = SimpleNamespace(app_instance=app, _selected_media_id="old")
    controller = build_library_media_analysis_controller(screen)
    screen._run_library_service_call = service_call
    screen._library_media_backing_id = lambda media_id: int(media_id)
    screen._refresh_library_media_detail = refreshed
    screen._selected_media_id = "7"
    app.media_reading_scope_service = SimpleNamespace(
        save_analysis_version=lambda **kwargs: calls.append(("save", kwargs))
    )
    controller._library_media_editing_analysis = True

    assert await controller._save_library_media_analysis(
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
        ("refresh", "7"),
    ]
    assert controller._library_media_editing_analysis is True


def test_framework_and_dispatch_ports_are_live_after_construction():
    screen = SimpleNamespace(app_instance=SimpleNamespace())
    controller = build_library_media_analysis_controller(screen)
    screen.run_worker = lambda: "replacement worker"
    screen._dispatch_library_media_analysis = lambda *_args: "replacement provider"
    assert controller.run_worker() == "replacement worker"
    assert (
        controller._dispatch_library_media_analysis("body", None)
        == "replacement provider"
    )
