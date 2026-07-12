"""Concurrency regression coverage for the local chatbook registry read-modify-write.

The registry file is a JSON document guarded (post-fix) by a single in-process
lock on ``LocalChatbookService``. Without the lock, two overlapping exports
running on separate OS threads (the real-world path: the F4 export worker
calls ``asyncio.run(service.export_chatbook(...))`` inside
``@work(thread=True)``) can interleave their
``_load_registry()`` -> mutate -> ``_save_registry()`` sequence and lose a
record or collide on ``next_id``.

This test drives the service the way production code does: one shared
``LocalChatbookService`` instance, many OS threads, each thread running
``asyncio.run(service.create_chatbook(...))`` against its own fresh event
loop (mirroring ``library_screen.py``'s worker-thread ``asyncio.run`` calls).
"""

from __future__ import annotations

import asyncio
import threading

from tldw_chatbook.Chatbooks import LocalChatbookService

THREAD_COUNT = 20


def test_concurrent_create_chatbook_preserves_all_records_and_unique_ids(tmp_path):
    registry_path = tmp_path / "chatbooks.json"
    service = LocalChatbookService(db_paths={}, registry_path=registry_path)

    errors: list[BaseException] = []
    barrier = threading.Barrier(THREAD_COUNT)

    def worker(index: int) -> None:
        try:
            barrier.wait()  # maximize overlap so the race actually fires
            asyncio.run(
                service.create_chatbook(
                    name=f"Concurrent Pack {index}",
                    description="created from a worker thread",
                )
            )
        except BaseException as exc:  # noqa: BLE001 - surface every failure to the main thread
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(THREAD_COUNT)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, f"worker thread(s) raised: {errors}"

    registry = service._load_registry()
    records = registry["records"]

    assert len(records) == THREAD_COUNT, (
        f"expected {THREAD_COUNT} records, found {len(records)} "
        "(a lost update dropped a record written by an overlapping thread)"
    )

    chatbook_ids = [record["chatbook_id"] for record in records]
    assert len(set(chatbook_ids)) == THREAD_COUNT, (
        f"expected {THREAD_COUNT} distinct chatbook_id values, found "
        f"{len(set(chatbook_ids))} (next_id collided across overlapping writers)"
    )
