"""Lifecycle-owned served children with bounded, replaceable terminal views."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4


class StaleTerminalAttachment(RuntimeError):
    """The terminal no longer owns input or has not received a full render."""


class TerminalAttachment:
    """One generation of terminal input and output; never owns the child."""

    def __init__(self, host: ServedConsoleHost, browser_session_id: str) -> None:
        self.host = host
        self.browser_session_id = browser_session_id
        self.generation = uuid4().hex
        self.output: asyncio.Queue[tuple[str, bytes | str]] = asyncio.Queue(64)
        self.closed = asyncio.Event()
        self.begun = False
        self.ready = False
        self._queued_bytes = 0

    def _check_current(self, *, require_ready: bool = False) -> None:
        if self.host.current_attachment is not self or (
            require_ready and not self.ready
        ):
            raise StaleTerminalAttachment("terminal attachment unavailable")

    async def send_bytes(self, data: bytes) -> None:
        self._check_current(require_ready=True)
        await self.host.service.send_bytes(data)

    async def resize(self, width: int, height: int) -> None:
        self._check_current()
        await self.host.service.set_terminal_size(width, height)

    async def send_meta(self, data: dict[str, Any]) -> None:
        self._check_current(require_ready=True)
        await self.host.service.send_meta(data)

    async def detach(self) -> None:
        if self.host.current_attachment is self:
            self.host._attachment = None
        self.closed.set()
        self.ready = False
        while not self.output.empty():
            self.output.get_nowait()
        self._queued_bytes = 0

    async def receive(self) -> tuple[str, bytes | str]:
        kind, value = await self.output.get()
        self._queued_bytes -= len(value)
        self._check_current()
        return kind, value

    async def _publish(self, kind: str, value: bytes | str) -> None:
        if self.output.full() or self._queued_bytes + len(value) > 1024 * 1024:
            await self.detach()
            return
        self._queued_bytes += len(value)
        self.output.put_nowait((kind, value))


class ServedConsoleHost:
    """Retain an AppService and continuously drain it without browser waits."""

    def __init__(self, service_factory: Callable[..., Any]) -> None:
        self._attachment: TerminalAttachment | None = None
        self._attach_lock = asyncio.Lock()
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._saved_sessions: dict[str, str] = {}
        self._hydrations: dict[str, asyncio.Task[str]] = {}
        self.service = service_factory(
            write_bytes=self._write_bytes,
            write_str=self._write_str,
            close=self._child_closed,
        )

    @property
    def current_attachment(self) -> TerminalAttachment | None:
        """Return only the currently admitted terminal generation."""
        return self._attachment

    async def start(self) -> None:
        """Start the child once, before publishing this host to callers."""
        await self.service.start(80, 24)

    async def attach_terminal(
        self, browser_session_id: str, width: int, height: int
    ) -> TerminalAttachment:
        """Replace the terminal view and request a generation-tagged repaint."""
        async with self._attach_lock:
            if self._closed:
                raise RuntimeError("Console host is closed")
            if self._attachment is not None:
                await self._attachment.detach()
            attachment = TerminalAttachment(self, browser_session_id)
            self._attachment = attachment
            try:
                await self.service.send_meta(
                    {
                        "type": "console_reattach",
                        "generation": attachment.generation,
                        "width": width,
                        "height": height,
                    }
                )
            except BaseException:
                await attachment.detach()
                raise
            return attachment

    def register_saved_conversation(
        self, conversation_id: str, session_id: str
    ) -> None:
        """Register native save identity without replacing a competing reservation."""
        existing = self._saved_sessions.get(conversation_id)
        if self._closed or (existing is not None and existing != session_id):
            raise RuntimeError("saved conversation ownership unavailable")
        if conversation_id in self._hydrations and existing is None:
            raise RuntimeError("saved conversation hydration in progress")
        self._saved_sessions[conversation_id] = session_id

    async def reserve_saved_conversation(
        self, conversation_id: str, hydrate: Callable[[], Awaitable[str]]
    ) -> str:
        """Share one native hydration, surviving cancellation of a remote observer."""
        if self._closed:
            raise RuntimeError("Console host is closed")
        if conversation_id in self._saved_sessions:
            return self._saved_sessions[conversation_id]
        task = self._hydrations.get(conversation_id)
        if task is None:

            async def load() -> str:
                try:
                    session_id = await hydrate()
                    self._saved_sessions[conversation_id] = session_id
                    return session_id
                finally:
                    self._hydrations.pop(conversation_id, None)

            task = asyncio.create_task(load(), name="hydrate-served-conversation")
            self._hydrations[conversation_id] = task
        return await asyncio.shield(task)

    async def _write_bytes(self, data: bytes) -> None:
        attachment = self._attachment
        if attachment is not None and attachment.begun:
            await attachment._publish("bytes", data)

    async def _write_str(self, data: str) -> None:
        attachment = self._attachment
        if attachment is None:
            return
        envelope = json.loads(data)
        kind = envelope[0]
        if kind in {"console_attachment_begin", "console_attachment_ready"}:
            if envelope[1].get("generation") != attachment.generation:
                return
            if kind == "console_attachment_begin":
                attachment.begun = True
            elif attachment.begun:
                attachment.ready = True
            else:
                return
        elif not attachment.begun:
            return
        await attachment._publish("str", data)

    async def _child_closed(self) -> None:
        self._closed = True
        if self._attachment is not None:
            await self._attachment.detach()

    async def aclose(self) -> None:
        """Close admission and retire the service at server shutdown."""
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._retire())
        await asyncio.shield(self._close_task)

    async def _retire(self) -> None:
        await self._child_closed()
        if self._hydrations:
            await asyncio.gather(*self._hydrations.values(), return_exceptions=True)
        await self.service.stop()


class ServedConsoleHosts:
    """Serialize child creation by server-admitted storage identity."""

    def __init__(self, host_factory: Callable[[str], ServedConsoleHost]) -> None:
        self._host_factory = host_factory
        self._starts: dict[str, asyncio.Task[ServedConsoleHost]] = {}
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    def child_ids(self) -> tuple[str, ...]:
        """Return live retained children independently of attached browsers."""
        return tuple(
            task.result().service.app_service_id
            for task in self._starts.values()
            if task.done()
            and not task.cancelled()
            and task.exception() is None
            and not task.result()._closed
        )

    async def get_or_start(self, storage_identity: str) -> ServedConsoleHost:
        """Share one start task even when the initiating browser disappears."""
        if self._closed:
            raise RuntimeError("Console host registry is closed")
        task = self._starts.get(storage_identity)
        if (
            task is not None
            and task.done()
            and not task.cancelled()
            and task.exception() is None
        ):
            host = task.result()
            process = getattr(host.service, "_process", None)
            if host._closed and process is not None and process.returncode is not None:
                task = None
        if task is None:

            async def start() -> ServedConsoleHost:
                host = self._host_factory(storage_identity)
                await host.start()
                return host

            task = asyncio.create_task(start(), name="start-served-console")
            self._starts[storage_identity] = task
        return await asyncio.shield(task)

    async def aclose(self) -> None:
        """Drain all retained children, including an in-flight start."""
        self._closed = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._retire())
        await asyncio.shield(self._close_task)

    async def _retire(self) -> None:
        async def close(task: asyncio.Task[ServedConsoleHost]) -> None:
            host = await asyncio.shield(task)
            await host.aclose()

        results = await asyncio.gather(
            *(close(task) for task in self._starts.values()), return_exceptions=True
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result
