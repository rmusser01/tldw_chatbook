"""App-lifetime ownership for Console H3 image-edit operations.

This module deliberately has no Textual, attachment, or persistence imports.
It owns only cancellation/task identity and bounded, byte-free outcome records
so a fresh Console screen can reconcile durable success or failure guidance.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import logging
import threading
from typing import TypeAlias
from uuid import uuid4


ImageEditRunner: TypeAlias = Callable[[str], Awaitable[None]]
ImageEditSettled: TypeAlias = Callable[[str], None]

_LOGGER = logging.getLogger(__name__)


def _required_text(value: object, field_name: str) -> None:
    if type(value) is not str or not value:
        raise ValueError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True)
class ActiveImageEditOperation:
    """Immutable identity for one app-owned in-flight H3 edit."""

    session_id: str
    generation: str
    attachment_id: str
    captured_draft: str
    cancel_event: threading.Event
    task: asyncio.Task[None]

    def __post_init__(self) -> None:
        _required_text(self.session_id, "session_id")
        _required_text(self.generation, "generation")
        _required_text(self.attachment_id, "attachment_id")
        if type(self.captured_draft) is not str:
            raise TypeError("captured_draft must be a string")
        if not isinstance(self.cancel_event, threading.Event):
            raise TypeError("cancel_event must be a threading.Event")
        if not isinstance(self.task, asyncio.Task):
            raise TypeError("task must be an asyncio.Task")


@dataclass(frozen=True)
class ImageEditCompletion:
    """Byte-free durable-success cleanup awaiting Console acknowledgement."""

    session_id: str
    generation: str
    message_id: str
    attachment_id: str
    captured_draft: str

    def __post_init__(self) -> None:
        _required_text(self.session_id, "session_id")
        _required_text(self.generation, "generation")
        _required_text(self.message_id, "message_id")
        _required_text(self.attachment_id, "attachment_id")
        if type(self.captured_draft) is not str:
            raise TypeError("captured_draft must be a string")


@dataclass(frozen=True)
class ImageEditFailureNotice:
    """Byte-free durable failure guidance awaiting Console hydration."""

    session_id: str
    generation: str
    message_id: str

    def __post_init__(self) -> None:
        _required_text(self.session_id, "session_id")
        _required_text(self.generation, "generation")
        _required_text(self.message_id, "message_id")


class ImageEditOperationRegistry:
    """Single app-owned duplicate gate and durable-outcome ledger."""

    def __init__(self) -> None:
        self._active: dict[str, ActiveImageEditOperation] = {}
        self._completions: dict[str, ImageEditCompletion] = {}
        self._failure_notices: dict[str, ImageEditFailureNotice] = {}
        self._discarded_generations: set[str] = set()

    def start(
        self,
        *,
        session_id: str,
        attachment_id: str,
        captured_draft: str,
        cancel_event: threading.Event,
        runner: ImageEditRunner,
        on_settled: ImageEditSettled | None = None,
    ) -> ActiveImageEditOperation | None:
        """Start one owned child, or refuse when the session is already active."""
        if session_id in self._active or session_id in self._completions:
            return None
        _required_text(session_id, "session_id")
        _required_text(attachment_id, "attachment_id")
        if type(captured_draft) is not str:
            raise TypeError("captured_draft must be a string")
        if not isinstance(cancel_event, threading.Event):
            raise TypeError("cancel_event must be a threading.Event")
        if not callable(runner):
            raise TypeError("runner must be callable")
        if on_settled is not None and not callable(on_settled):
            raise TypeError("on_settled must be callable")
        generation = str(uuid4())

        async def _owned() -> None:
            runner_task = asyncio.create_task(
                runner(generation),
                name=f"console-h3-image-edit-runner-{generation}",
            )
            try:
                await asyncio.shield(runner_task)
            except asyncio.CancelledError as cancellation:
                cancel_event.set()
                while not runner_task.done():
                    try:
                        await asyncio.shield(runner_task)
                    except asyncio.CancelledError:
                        continue
                    except Exception:  # pragma: no cover - inspected below
                        break
                if runner_task.done():
                    try:
                        runner_task.result()
                    except asyncio.CancelledError:
                        pass
                    except Exception as exc:  # pragma: no cover - containment seam
                        _LOGGER.error(
                            "Console image edit runner failed during cancellation "
                            "settlement (error_type=%s)",
                            type(exc).__name__,
                        )
                raise cancellation
            except Exception as exc:  # pragma: no cover - final containment seam
                _LOGGER.error(
                    "Console image edit runner escaped containment (error_type=%s)",
                    type(exc).__name__,
                )
            finally:
                removed = self.remove_active(session_id, generation)
                if removed and on_settled is not None:
                    try:
                        on_settled(generation)
                    except Exception as exc:  # pragma: no cover - UI scheduling seam
                        _LOGGER.error(
                            "Console image edit settlement scheduling failed "
                            "(error_type=%s)",
                            type(exc).__name__,
                        )

        task = asyncio.create_task(_owned(), name=f"console-h3-image-edit-{generation}")
        operation = ActiveImageEditOperation(
            session_id=session_id,
            generation=generation,
            attachment_id=attachment_id,
            captured_draft=captured_draft,
            cancel_event=cancel_event,
            task=task,
        )
        self._active[session_id] = operation
        return operation

    async def shutdown(self) -> None:
        """Cancel and drain every app-owned operation to real settlement."""
        operations = self.active_operations()
        for operation in operations:
            operation.cancel_event.set()
            operation.task.cancel()
        if operations:
            await asyncio.gather(
                *(operation.task for operation in operations),
                return_exceptions=True,
            )

    def active(self, session_id: str) -> ActiveImageEditOperation | None:
        """Return the exact active operation for ``session_id``."""
        return self._active.get(session_id)

    def active_operations(self) -> tuple[ActiveImageEditOperation, ...]:
        """Return an immutable snapshot of all active operations."""
        return tuple(self._active.values())

    def request_cancel(self, session_id: str) -> ActiveImageEditOperation | None:
        """Set the exact operation event and return its immutable record."""
        operation = self._active.get(session_id)
        if operation is not None:
            operation.cancel_event.set()
        return operation

    def remove_active(self, session_id: str, generation: str) -> bool:
        """Remove only the matching generation, protecting a later operation."""
        operation = self._active.get(session_id)
        if operation is None or operation.generation != generation:
            self._discarded_generations.discard(generation)
            return False
        del self._active[session_id]
        self._discarded_generations.discard(generation)
        return True

    def publish_completion(self, completion: ImageEditCompletion) -> bool:
        """Retain one bounded cleanup record for the session."""
        if completion.generation in self._discarded_generations:
            return False
        active = self._active.get(completion.session_id)
        if active is not None and active.generation != completion.generation:
            return False
        self._completions[completion.session_id] = completion
        return True

    def completion(self, session_id: str) -> ImageEditCompletion | None:
        """Return the pending cleanup record for ``session_id``."""
        return self._completions.get(session_id)

    def completions(self) -> tuple[ImageEditCompletion, ...]:
        """Return an immutable snapshot of pending cleanup records."""
        return tuple(self._completions.values())

    def ack_completion(self, session_id: str, generation: str) -> bool:
        """Acknowledge only the matching generation's completed cleanup."""
        completion = self._completions.get(session_id)
        if completion is None or completion.generation != generation:
            return False
        del self._completions[session_id]
        return True

    def publish_failure_notice(self, notice: ImageEditFailureNotice) -> bool:
        """Retain one durable failure-message identity for the session."""
        if notice.generation in self._discarded_generations:
            return False
        active = self._active.get(notice.session_id)
        if active is not None and active.generation != notice.generation:
            return False
        self._failure_notices[notice.session_id] = notice
        return True

    def failure_notice(self, session_id: str) -> ImageEditFailureNotice | None:
        """Return pending durable failure guidance for ``session_id``."""
        return self._failure_notices.get(session_id)

    def failure_notices(self) -> tuple[ImageEditFailureNotice, ...]:
        """Return an immutable snapshot of pending failure guidance."""
        return tuple(self._failure_notices.values())

    def ack_failure_notice(self, session_id: str, generation: str) -> bool:
        """Acknowledge only the matching generation's hydrated guidance."""
        notice = self._failure_notices.get(session_id)
        if notice is None or notice.generation != generation:
            return False
        del self._failure_notices[session_id]
        return True

    def drop_session(self, session_id: str) -> None:
        """Cancel and forget all operation state for a deleted session."""
        operation = self._active.pop(session_id, None)
        if operation is not None:
            self._discarded_generations.add(operation.generation)
            operation.cancel_event.set()
        self._completions.pop(session_id, None)
        self._failure_notices.pop(session_id, None)
