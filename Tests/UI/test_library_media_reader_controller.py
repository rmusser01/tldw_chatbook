"""Reader progress and live state follow the current Library owners."""

from types import MethodType, SimpleNamespace

import pytest

from tldw_chatbook.UI.Library_Modules.library_media_controller import (
    LibraryMediaController,
)
from tldw_chatbook.UI.Library_Modules.library_media_state import LibraryMediaState
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen


@pytest.mark.asyncio
async def test_progress_drainer_coalesces_per_item_and_records_only_durable_offsets():
    writes = []

    async def run_service_call(method, **kwargs):
        return method(**kwargs)

    screen = SimpleNamespace(
        app_instance=SimpleNamespace(),
        is_attached=False,
        _media_state=LibraryMediaState(),
    )
    for name in (
        "_write_library_media_loaded_progress",
        "_queue_library_media_progress_write",
        "_drain_library_media_progress_writes",
        "_library_media_progress_write_is_current",
    ):
        setattr(screen, name, MethodType(getattr(LibraryScreen, name), screen))
    screen._run_library_service_call = run_service_call
    screen.app_instance.media_reading_scope_service = SimpleNamespace(
        update_reading_progress=lambda **kwargs: writes.append(kwargs)
    )
    screen._queue_library_media_progress_write("local:media:1", 1, (0, 2))
    screen._queue_library_media_progress_write("local:media:2", 2, (0, 4))
    screen._queue_library_media_progress_write("local:media:1", 1, (0, 8))
    await screen._drain_library_media_progress_writes()
    assert [entry["media_id"] for entry in writes] == [1, 2]
    assert [entry["progress_data"]["scroll_y"] for entry in writes] == [8, 4]
    assert screen._media_state.progress_persisted_offsets == {
        "local:media:1": (0, 8),
        "local:media:2": (0, 4),
    }
    assert screen._media_state.progress_pending_writes == {}
    assert screen._media_state.progress_inflight_write is None


def test_media_state_property_reads_and_writes_the_current_state():
    controller = object.__new__(LibraryMediaController)
    state = LibraryMediaState(content_query="first")
    controller._media_state_accessor = lambda: state
    assert controller._library_media_content_query == "first"
    controller._library_media_content_query = "updated"
    assert state.content_query == "updated"
    state = LibraryMediaState(content_query="replacement")
    assert controller._library_media_content_query == "replacement"
    with pytest.raises(AttributeError):
        _ = controller.unadvertised_field
