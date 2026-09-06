"""Reader state and service ports are independently testable."""

from types import SimpleNamespace

import pytest

from tldw_chatbook.UI.Library_Modules.controller_state import ControllerState
from tldw_chatbook.UI.Library_Modules.media_reader_wiring import (
    build_library_media_reader_controller,
)


@pytest.mark.asyncio
async def test_progress_drainer_coalesces_per_item_and_records_only_durable_offsets():
    writes = []

    async def run_service_call(method, **kwargs):
        return method(**kwargs)

    screen = SimpleNamespace(
        app_instance=SimpleNamespace(),
        is_attached=False,
    )
    controller = build_library_media_reader_controller(screen)
    screen._run_library_service_call = run_service_call
    screen.app_instance.media_reading_scope_service = SimpleNamespace(
        update_reading_progress=lambda **kwargs: writes.append(kwargs)
    )
    controller._queue_library_media_progress_write("local:media:1", 1, (0, 2))
    controller._queue_library_media_progress_write("local:media:2", 2, (0, 4))
    controller._queue_library_media_progress_write("local:media:1", 1, (0, 8))
    await controller._drain_library_media_progress_writes()
    assert [entry["media_id"] for entry in writes] == [1, 2]
    assert [entry["progress_data"]["scroll_y"] for entry in writes] == [8, 4]
    assert controller._library_media_progress_persisted_offsets == {
        "local:media:1": (0, 8),
        "local:media:2": (0, 4),
    }
    assert controller._library_media_progress_pending_writes == {}
    assert controller._library_media_progress_inflight_write is None


def test_exact_state_declaration_reads_and_writes_the_current_controller():
    class Host:
        query = ControllerState("reader", "query")

    host = Host()
    host.reader = SimpleNamespace(query="first")
    assert host.query == "first"
    host.query = "updated"
    assert host.reader.query == "updated"
    host.reader = SimpleNamespace(query="replacement")
    assert host.query == "replacement"
    with pytest.raises(AttributeError):
        _ = host.unadvertised_field
