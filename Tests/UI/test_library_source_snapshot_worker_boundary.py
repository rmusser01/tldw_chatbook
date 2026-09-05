"""Every potentially blocking source adapter stays off the Library UI loop."""

import threading
from types import SimpleNamespace

import pytest

from tldw_chatbook.UI.Screens.library_screen import LibraryScreen


@pytest.mark.asyncio
async def test_all_source_families_run_in_workers_and_preserve_results():
    threads = {}

    def list_source(source):
        async def list_records(**_kwargs):
            threads[source] = threading.get_ident()
            return {"items": [{"id": source, "title": source}], "total": 1}

        return list_records

    screen = LibraryScreen(
        SimpleNamespace(
            app_config={},
            notes_scope_service=SimpleNamespace(list_notes=list_source("notes")),
            media_reading_scope_service=SimpleNamespace(
                list_media_items=list_source("media")
            ),
            chat_conversation_scope_service=SimpleNamespace(
                list_conversations=list_source("conversations")
            ),
        )
    )
    (
        records,
        counts,
        total_known,
        error,
        recovery,
        _,
    ) = await screen._list_local_source_snapshot()
    assert set(threads) == {"notes", "media", "conversations"}
    assert all(worker != threading.get_ident() for worker in threads.values())
    for source in threads:
        assert records[source] == ({"id": source, "title": source},)
        assert counts[source] == 1
        assert total_known[source] is True
    assert error is None
    assert recovery is None
