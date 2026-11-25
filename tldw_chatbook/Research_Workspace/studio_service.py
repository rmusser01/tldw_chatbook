"""Coordinate Studio cancellation, retry, and canonical publication."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from .contracts import QualifiedWorkspaceRef
from .studio_generation import ResearchStudioGenerator, StudioGeneratedOutput
from .studio_models import ResearchStudioRequest, WorkspaceOutputRef


@dataclass(slots=True)
class _Attempt:
    request: ResearchStudioRequest
    replace_output: WorkspaceOutputRef | None
    supersedes: WorkspaceOutputRef | None
    generated: StudioGeneratedOutput | None = None
    saved: WorkspaceOutputRef | None = None


class ResearchStudioService:
    """Keep a bounded set of retries attached to their original owner."""

    def __init__(self, generator: ResearchStudioGenerator, owner) -> None:
        self.generator = generator
        self.owner = owner
        self._attempts: dict[tuple[QualifiedWorkspaceRef, str], _Attempt] = {}
        self._active: dict[tuple[QualifiedWorkspaceRef, str], asyncio.Event] = {}

    async def generate(
        self,
        request: ResearchStudioRequest,
        on_chunk: Callable[[str], None],
        cancel_event: asyncio.Event,
        *,
        replace_output: WorkspaceOutputRef | None = None,
        supersedes: WorkspaceOutputRef | None = None,
    ) -> WorkspaceOutputRef:
        key = (request.workspace, request.operation_id)
        if key in self._active:
            raise ValueError("This output is already being generated")
        attempt = self._attempts.get(key)
        if attempt is None:
            attempt = _Attempt(request, replace_output, supersedes)
            self._attempts[key] = attempt
        elif (attempt.request, attempt.replace_output, attempt.supersedes) != (
            request,
            replace_output,
            supersedes,
        ):
            raise ValueError("The generation operation already has a different request")
        if attempt.saved is not None:
            return attempt.saved

        self._active[key] = cancel_event
        try:
            if attempt.generated is None:
                native_generator = getattr(self.owner, "generate_native", None)
                generated = (
                    await native_generator(request, cancel_event=cancel_event)
                    if native_generator is not None else None
                )
                if generated is None:
                    generated = await self.generator.generate(
                        request, on_chunk=on_chunk, cancel_event=cancel_event
                    )
                if generated.status == "failed":
                    raise ValueError(generated.failure_reason)
                if generated.status == "complete":
                    attempt.generated = generated
            else:
                generated = attempt.generated
            if generated.status == "complete" and generated.native_owner is None and not cancel_event.is_set():
                prepare = getattr(self.owner, "prepare_output", None)
                prepared = await prepare(request, generated, cancel_event=cancel_event) if prepare else None
                if prepared is not None:
                    generated = prepared
                    if prepared.status == "complete":
                        attempt.generated = prepared
            if cancel_event.is_set():
                generated = StudioGeneratedOutput("cancelled")
            result = await self.owner.save(
                request, generated, replace_output=replace_output, supersedes=supersedes
            )
            if (
                result.workspace != request.workspace
                or result.output_kind is not request.output_kind
            ):
                raise ValueError(
                    "The owner returned an output from a different workspace or kind"
                )
            if result.status == "complete":
                attempt.saved = result
                attempt.generated = None
            return result
        finally:
            del self._active[key]
            # Only an active operation needs to retain its entry past the retry window.
            for stale in tuple(self._attempts)[:-32]:
                if stale not in self._active:
                    self._attempts.pop(stale, None)

    def cancel(self, ref: QualifiedWorkspaceRef, operation_id: str) -> bool:
        event = self._active.get((ref, operation_id))
        if event is None:
            return False
        event.set()
        return True

    def has_attempt(self, request: ResearchStudioRequest) -> bool:
        attempt = self._attempts.get((request.workspace, request.operation_id))
        return attempt is not None and attempt.request == request

    async def retry(
        self, ref: QualifiedWorkspaceRef, operation_id: str
    ) -> WorkspaceOutputRef:
        attempt = self._attempts.get((ref, operation_id))
        if attempt is None:
            raise ValueError("The output retry is unavailable. Generate a new output.")
        return await self.generate(
            attempt.request,
            lambda _chunk: None,
            asyncio.Event(),
            replace_output=attempt.replace_output,
            supersedes=attempt.supersedes,
        )
