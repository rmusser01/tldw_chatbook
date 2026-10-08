"""Finite citation-footer database work owned by its Console view worker."""

import asyncio
from collections.abc import Callable
from typing import Any


async def read_citation_counts(
    repository: Any,
    reader: Callable[..., dict[str, int]],
    eligible: tuple[tuple[str, str, str, str], ...],
) -> dict[str, int]:
    """Run the captured reader, retiring stock file-backed work before cancellation."""
    from ...Chat.citation_trace_repository import CitationTraceRepository
    from ...DB.ChaChaNotes_DB import CharactersRAGDB
    from ...DB.base_db import run_owned_db_call

    database = getattr(repository, "db", None)
    operation = run_owned_db_call(database, reader, repository, eligible)
    if (
        type(repository) is not CitationTraceRepository
        or type(database) is not CharactersRAGDB
        or database.is_memory_db
    ):
        return await operation
    worker = asyncio.Task(operation)
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:  # noqa: BLE001 - cancellation wins.
                break
        if not worker.cancelled():
            worker.exception()
        raise
