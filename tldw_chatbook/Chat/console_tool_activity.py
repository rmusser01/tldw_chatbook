"""Session-only, call-correlated Console tool lifecycle projection (ADR-195)."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from typing import TYPE_CHECKING

from tldw_chatbook.Agents.agent_models import AgentStep
from tldw_chatbook.Chat.console_chat_models import (
    MAX_CONSOLE_TOOL_ARGUMENT_CHARS,
    ConsoleActivityPresentation,
    ConsoleActivityStatus,
    ConsoleMessageRole,
)

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore


_PENDING = frozenset({"queued", "awaiting_approval", "running"})
_ARGUMENT_TRUNCATION_SUFFIX = "\n… arguments truncated"


class ConsoleToolActivity:
    """Own display markers for one primary run; never owns execution authority."""

    def __init__(self, store: ConsoleChatStore, session_id: str) -> None:
        self.store = store
        self.session_id = session_id
        self._rows: dict[str, tuple[str, ConsoleActivityPresentation]] = {}
        self._proposal_indices: dict[str, int] = {}
        self._lock = threading.RLock()
        self._closed = False

    def observe(self, step: AgentStep, round_ordinal: int | None) -> None:
        """Project safe proposed-call arguments and execution start facts."""
        if not step.call_id:
            return
        with self._lock:
            if self._closed:
                return
            prior = self._rows.get(step.call_id)
            if step.kind == "tool_proposed" and (
                prior is None or prior[1].status not in _PENDING
            ):
                arguments = json.dumps(step.args or {}, ensure_ascii=False, indent=2)
                if len(arguments) > MAX_CONSOLE_TOOL_ARGUMENT_CHARS:
                    cutoff = MAX_CONSOLE_TOOL_ARGUMENT_CHARS - len(
                        _ARGUMENT_TRUNCATION_SUFFIX
                    )
                    arguments = arguments[:cutoff] + _ARGUMENT_TRUNCATION_SUFFIX
                label = " ".join(step.tool_name.split())[:200] or "Tool"
                presentation = ConsoleActivityPresentation(
                    "tool",
                    label,
                    "queued",
                    call_id=step.call_id,
                    arguments=arguments,
                )
                try:
                    marker = self.store.append_message(
                        self.session_id,
                        role=ConsoleMessageRole.TOOL,
                        content=f"⚙ {label}",
                        activity_presentation=presentation,
                        activity_round_ordinal=round_ordinal,
                        record_trajectory=False,
                    )
                except KeyError:
                    return
                self._rows[step.call_id] = (marker.id, presentation)
                self._proposal_indices[step.call_id] = step.index
            elif step.kind == "tool_output" and (
                prior is not None
                and prior[1].status == "running"
                and step.source_step_index == self._proposal_indices.get(step.call_id)
            ):
                from tldw_chatbook.Agents.tool_output import MAX_TOOL_OUTPUT_CHARS

                self._update(
                    step.call_id,
                    "running",
                    result_preview=step.result[:MAX_TOOL_OUTPUT_CHARS],
                )
            elif step.kind == "tool_execution_started":
                self._update(
                    step.call_id, "running", started_at_monotonic=time.monotonic()
                )

    def _update(self, call_id: str, status: ConsoleActivityStatus, **fields) -> None:
        row = self._rows.get(call_id)
        if row is None or row[1].status not in _PENDING or self._closed:
            return
        marker_id, old = row
        presentation = replace(old, status=status, **fields)
        try:
            self.store.update_tool_marker(
                self.session_id,
                marker_id,
                activity_presentation=presentation,
            )
        except KeyError:
            return
        self._rows[call_id] = (marker_id, presentation)

    def approval(self, call_ids: list[str], pending: bool) -> None:
        """Mark only calls belonging to a real pending approval round."""
        with self._lock:
            for call_id in call_ids:
                row = self._rows.get(call_id)
                if row is not None and row[1].status in {"queued", "awaiting_approval"}:
                    self._update(call_id, "awaiting_approval" if pending else "queued")

    def take(self, call_id: str) -> tuple[str, ConsoleActivityPresentation] | None:
        """Transfer a shell marker to its existing process-lifecycle owner."""
        with self._lock:
            self._proposal_indices.pop(call_id, None)
            return self._rows.pop(call_id, None)

    def complete(
        self,
        step: AgentStep,
        presentation: ConsoleActivityPresentation,
        content: str,
        tool_diff: tuple[str, str, str] | None,
        *,
        record_trajectory: bool,
    ) -> bool:
        """Settle one exact marker and retain its existing terminal trajectory."""
        with self._lock:
            row = self._rows.get(step.call_id)
            if row is None or self._closed:
                return False
            marker_id, old = row
            if old.status not in _PENDING:
                return True
            status = {"timeout": "timed_out", "cancelled": "stopped"}.get(
                step.tool_outcome,
                presentation.status,
            )
            result = str(step.result or "")
            prefix = f"⚙ {step.tool_name} → "
            preview = content[len(prefix) :] if content.startswith(prefix) else result
            final = replace(
                old,
                status=status,
                result_preview=preview,
                elapsed_seconds=(
                    max(0.0, time.monotonic() - old.started_at_monotonic)
                    if old.started_at_monotonic is not None
                    else None
                ),
                started_at_monotonic=None,
            )
            try:
                self.store.update_tool_marker(
                    self.session_id,
                    marker_id,
                    content=content,
                    tool_output_full=(
                        result if result and result not in content else None
                    ),
                    tool_diff=tool_diff,
                    activity_presentation=final,
                    record_trajectory=record_trajectory,
                )
                if status in {"timed_out", "stopped"} and old.result_preview:
                    # Capture above uses only the final result. Retained partial
                    # text is a second, explicitly session-only display update.
                    self.store.update_tool_marker(
                        self.session_id,
                        marker_id,
                        tool_output_full=f"{result}\n\nPartial output:\n{old.result_preview}",
                        record_trajectory=False,
                    )
            except KeyError:
                pass
            self._rows[step.call_id] = (marker_id, final)
            return True

    def finish(self, cancelled: bool) -> None:
        """End unresolved displays without asserting a worker was killed."""
        with self._lock:
            for call_id, (_, old) in tuple(self._rows.items()):
                if old.status in _PENDING:
                    self._update(
                        call_id,
                        "stopped" if cancelled else "failed",
                        result_preview=(
                            "Run ended before a result was received."
                            + (
                                "\nPartial output:\n" + old.result_preview
                                if old.result_preview
                                else ""
                            )
                        ),
                        elapsed_seconds=(
                            max(0.0, time.monotonic() - old.started_at_monotonic)
                            if old.started_at_monotonic is not None
                            else None
                        ),
                        started_at_monotonic=None,
                    )
            self._closed = True
            self._rows.clear()
            self._proposal_indices.clear()
