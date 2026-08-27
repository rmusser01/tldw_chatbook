"""Pure round-owned capture of explicit Console provider thinking events."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Literal
from uuid import uuid4

from tldw_chatbook.Chat.console_provider_gateway import (
    ProviderProprietaryThinkingEvidence,
    ProviderStreamItem,
    ProviderThinkingCaptureError,
    ProviderThinkingDelta,
    ProviderToolCalls,
)
from tldw_chatbook.Chat.thinking_blocks import (
    DisplayableThinkingBlock,
    ProprietaryThinkingBlock,
    ThinkingBlock,
    ThinkingEnvelope,
    ThinkingEnvelopeValidationError,
    ThinkingStatus,
    dump_thinking_blocks_json,
)


@dataclass(frozen=True, slots=True)
class ThinkingCaptureUpdate:
    """One immutable live or terminal capture projection."""

    envelope: ThinkingEnvelope | None = field(default=None, repr=False)
    changed_block_id: str | None = None
    collapse_boundary_reached: bool = False
    terminal: bool = False


class ThinkingCapture:
    """Accumulate only typed provider evidence for one assistant generation."""

    def __init__(self, *, assistant_owner_id: str) -> None:
        if type(assistant_owner_id) is not str or not assistant_owner_id:
            raise ValueError("assistant_owner_id must be a non-empty string")
        self._owner_digest = hashlib.sha256(
            assistant_owner_id.encode("utf-8")
        ).hexdigest()[:20]
        self._capture_namespace = uuid4().hex
        self._blocks: tuple[ThinkingBlock, ...] = ()
        self._round_ordinal = 0
        self._sequence = 0
        self._boundary_reached = False
        self._collapsed_block_ids: set[str] = set()
        self._terminal_outcome: ThinkingStatus | None = None

    def snapshot(self) -> ThinkingCaptureUpdate:
        """Return the current process-local projection without changing it."""
        return ThinkingCaptureUpdate(
            envelope=self._envelope(),
            terminal=self._terminal_outcome is not None,
        )

    def observe(self, item: ProviderStreamItem) -> ThinkingCaptureUpdate:
        """Route one provider stream item through the explicit evidence seam."""
        self._require_live()
        if isinstance(item, ProviderThinkingDelta):
            return self.observe_thinking_delta(item)
        if isinstance(item, ProviderProprietaryThinkingEvidence):
            return self.observe_proprietary_evidence(item)
        if isinstance(item, ProviderToolCalls):
            return self.observe_tool()
        if type(item) is str:
            return self.observe_answer(item)
        raise TypeError("Unsupported provider stream item.")

    def observe_thinking_delta(
        self, event: ProviderThinkingDelta
    ) -> ThinkingCaptureUpdate:
        """Append one adapter-approved displayable delta to the current round."""
        self._require_live()
        current = self._current_block()
        if current is None:
            candidate: ThinkingBlock = DisplayableThinkingBlock(
                block_id=self._next_block_id(),
                round_ordinal=self._round_ordinal,
                provider=event.provider,
                model=event.model,
                protocol=event.protocol,
                source_format=event.source_format,
                status="complete",
                text=event.text,
            )
            blocks = (*self._blocks, candidate)
        elif isinstance(current, DisplayableThinkingBlock) and (
            current.provider,
            current.model,
            current.protocol,
            current.source_format,
        ) == (
            event.provider,
            event.model,
            event.protocol,
            event.source_format,
        ):
            try:
                candidate = replace(current, text=current.text + event.text)
            except ThinkingEnvelopeValidationError:
                self._capture_failed()
            blocks = (*self._blocks[:-1], candidate)
        else:
            self._capture_failed()
        self._install(blocks)
        return self._evidence_update(candidate.block_id)

    def observe_proprietary_evidence(
        self, event: ProviderProprietaryThinkingEvidence
    ) -> ThinkingCaptureUpdate:
        """Record one content-free proprietary occurrence per model round."""
        self._require_live()
        current = self._current_block()
        if isinstance(current, ProprietaryThinkingBlock):
            if (
                current.provider,
                current.model,
                current.protocol,
                current.source_format,
            ) != (
                event.provider,
                event.model,
                event.protocol,
                event.source_format,
            ):
                self._capture_failed()
            return self._evidence_update(current.block_id)
        if current is not None:
            self._capture_failed()
        candidate = ProprietaryThinkingBlock(
            block_id=self._next_block_id(),
            round_ordinal=self._round_ordinal,
            provider=event.provider,
            model=event.model,
            protocol=event.protocol,
            source_format=event.source_format,
            status="complete",
        )
        self._install((*self._blocks, candidate))
        return self._evidence_update(candidate.block_id)

    def observe_answer(self, content: str) -> ThinkingCaptureUpdate:
        """Mark the current round's first visible-answer boundary."""
        self._require_live()
        if type(content) is not str:
            raise TypeError("Provider answer chunks must be strings.")
        if content:
            return self._mark_boundary()
        return self._update()

    def observe_tool(self) -> ThinkingCaptureUpdate:
        """Close the current primary model round at its tool-call seam."""
        self._require_live()
        update = self._mark_boundary()
        self._round_ordinal += 1
        self._boundary_reached = False
        return update

    def settle(
        self, outcome: Literal["complete", "stopped", "failed"]
    ) -> ThinkingCaptureUpdate:
        """Freeze terminal block status without fabricating absent evidence."""
        if outcome not in {"complete", "stopped", "failed"}:
            raise ValueError("Invalid thinking capture outcome.")
        if self._terminal_outcome is not None:
            if outcome != self._terminal_outcome:
                raise RuntimeError("Thinking capture already settled.")
            return self.snapshot()
        current = self._current_block()
        if current is not None and current.status != outcome:
            self._install((*self._blocks[:-1], replace(current, status=outcome)))
            current = self._current_block()
        collapse_boundary_reached = bool(
            current is not None and current.block_id not in self._collapsed_block_ids
        )
        if collapse_boundary_reached:
            self._collapsed_block_ids.add(current.block_id)
        self._terminal_outcome = outcome
        return ThinkingCaptureUpdate(
            envelope=self._envelope(),
            collapse_boundary_reached=collapse_boundary_reached,
            terminal=True,
        )

    def _current_block(self) -> ThinkingBlock | None:
        if self._blocks and self._blocks[-1].round_ordinal == self._round_ordinal:
            return self._blocks[-1]
        return None

    def _next_block_id(self) -> str:
        block_id = (
            f"thinking-{self._owner_digest}-{self._capture_namespace}-"
            f"{self._round_ordinal}-{self._sequence}"
        )
        self._sequence += 1
        return block_id

    def _install(self, blocks: tuple[ThinkingBlock, ...]) -> None:
        try:
            envelope = ThinkingEnvelope(blocks=blocks)
            dump_thinking_blocks_json(envelope)
        except ThinkingEnvelopeValidationError:
            self._capture_failed()
        self._blocks = blocks

    def _envelope(self) -> ThinkingEnvelope | None:
        return ThinkingEnvelope(self._blocks) if self._blocks else None

    def _evidence_update(self, block_id: str) -> ThinkingCaptureUpdate:
        collapse_boundary_reached = (
            self._boundary_reached and block_id not in self._collapsed_block_ids
        )
        if collapse_boundary_reached:
            self._collapsed_block_ids.add(block_id)
        return self._update(
            changed_block_id=block_id,
            collapse_boundary_reached=collapse_boundary_reached,
        )

    def _mark_boundary(self) -> ThinkingCaptureUpdate:
        if self._boundary_reached:
            return self._update()
        self._boundary_reached = True
        current = self._current_block()
        if current is not None:
            self._collapsed_block_ids.add(current.block_id)
        return self._update(collapse_boundary_reached=True)

    def _update(
        self,
        *,
        changed_block_id: str | None = None,
        collapse_boundary_reached: bool = False,
    ) -> ThinkingCaptureUpdate:
        return ThinkingCaptureUpdate(
            envelope=self._envelope(),
            changed_block_id=changed_block_id,
            collapse_boundary_reached=collapse_boundary_reached,
        )

    def _require_live(self) -> None:
        if self._terminal_outcome is not None:
            raise RuntimeError("Thinking capture already settled.")

    @staticmethod
    def _capture_failed() -> None:
        raise ProviderThinkingCaptureError("Provider thinking capture failed.")
