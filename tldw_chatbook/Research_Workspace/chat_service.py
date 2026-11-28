"""Track qualified Research Chat turns and stop their provider streams."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field

from loguru import logger

from .chat_models import (
    MAX_CHAT_SOURCES,
    ResearchChatTurnOutcome,
    ResearchChatTurnRequest,
    ResearchSourceVersionRef,
    ResearchTurnStatus,
)
from .contracts import QualifiedWorkspaceRef, ResearchWorkspacePort

MAX_ANSWER_BYTES = 256 * 1024
RunTurn = Callable[
    [ResearchChatTurnRequest, Callable[[str], None], asyncio.Event],
    Awaitable[ResearchChatTurnOutcome],
]


class ResearchChatService:
    """Coordinate active turns for a single canonical conversation owner."""

    def __init__(self, run_turn: RunTurn) -> None:
        self._run_turn = run_turn
        self._active: dict[
            tuple[QualifiedWorkspaceRef, str], tuple[str, asyncio.Event]
        ] = {}

    async def send(
        self,
        request: ResearchChatTurnRequest,
        on_chunk: Callable[[str], None],
        cancel: asyncio.Event,
    ) -> ResearchChatTurnOutcome:
        key = (request.workspace, request.conversation_id)
        if key in self._active:
            raise ValueError("Stop the active answer before sending another question")
        self._active[key] = (request.turn_id, cancel)
        try:
            return await self._run_turn(request, on_chunk, cancel)
        finally:
            del self._active[key]

    def stop(self, ref: QualifiedWorkspaceRef, turn_id: str) -> bool:
        for (owner, _), (active_id, cancel) in self._active.items():
            if owner == ref and active_id == turn_id:
                cancel.set()
                return True
        return False


@dataclass(frozen=True, slots=True)
class StreamAnswer:
    status: ResearchTurnStatus
    text: str = field(repr=False)


async def capture_selected_sources(
    port: ResearchWorkspacePort,
    ref: QualifiedWorkspaceRef,
) -> tuple[ResearchSourceVersionRef, ...]:
    """Read the selected association and content versions from the same owner."""
    page = await port.list_sources(ref, limit=100, offset=0)
    desired_ids = page.desired_source_ids
    if len(desired_ids) > MAX_CHAT_SOURCES:
        raise ValueError(f"Select at most {MAX_CHAT_SOURCES} sources for this answer")
    selected = [source for source in page.items if source.selected]
    while page.has_more and len(selected) < len(desired_ids):
        offset = page.offset + page.limit
        page = await port.list_sources(ref, limit=100, offset=offset)
        if page.desired_source_ids != desired_ids:
            raise ValueError("Source selection changed while reading the workspace")
        selected.extend(source for source in page.items if source.selected)
    if not selected:
        return ()
    readiness = await port.get_readiness(
        ref, source_ids=tuple(source.source_id for source in selected)
    )
    readiness_by_id = {row.source_id: row for row in readiness}
    return tuple(
        ResearchSourceVersionRef.capture(source, readiness_by_id.get(source.source_id))
        for source in selected
    )


async def collect_answer(
    stream: AsyncIterator[object],
    on_chunk: Callable[[str], None],
    cancel: asyncio.Event,
) -> StreamAnswer:
    """Collect visible text, racing the stop signal even during a stalled read."""
    pieces = []
    total_bytes = 0
    status: ResearchTurnStatus = "complete"
    stop_task = asyncio.create_task(cancel.wait())
    next_item: asyncio.Future | None = None
    try:
        while not cancel.is_set():
            next_item = asyncio.ensure_future(anext(stream))
            await asyncio.wait(
                (next_item, stop_task), return_when=asyncio.FIRST_COMPLETED
            )
            if cancel.is_set():
                break
            try:
                item = next_item.result()
            except StopAsyncIteration:
                break
            if not isinstance(item, str) or not item:
                continue
            total_bytes += len(item.encode("utf-8"))
            if total_bytes > MAX_ANSWER_BYTES:
                raise ValueError("Research answer exceeds the response limit")
            pieces.append(item)
            on_chunk(item)
        if cancel.is_set():
            status = "cancelled"
        elif not pieces:
            status = "failed"
    except Exception as exc:
        status = "failed"
        logger.warning(
            "Research answer stream failed (error_type={})", type(exc).__name__
        )
    finally:
        stop_task.cancel()
        if next_item is not None:
            if not next_item.done():
                next_item.cancel()
            await asyncio.gather(next_item, return_exceptions=True)
        await asyncio.gather(stop_task, return_exceptions=True)
        close = getattr(stream, "aclose", None)
        if callable(close):
            await close()
    return StreamAnswer(status, "".join(pieces))
