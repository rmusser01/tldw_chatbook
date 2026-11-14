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


_PENDING = frozenset({"queued", "awaiting_approval", "starting", "running"})
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
                and prior[1].status in {"starting", "running"}
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
                    step.call_id, "starting", started_at_monotonic=time.monotonic()
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

    def feedback(self, facts, *, call_keys=None) -> None:
        """Transfer reduced facts to existing session-only rows, including terminals."""
        with self._lock:
            for fact in facts:
                if fact.identity.session_id != self.session_id:
                    continue
                keys = (
                    call_keys(fact.identity)
                    if call_keys is not None
                    else (fact.identity.call_key,)
                )
                for call_key in keys:
                    row = self._rows.get(call_key)
                    if row is None:
                        continue
                    marker_id, old = row
                    previous = old.approval_feedback
                    if previous is not None and previous.sequence >= fact.sequence:
                        continue
                    presentation = replace(old, approval_feedback=fact)
                    try:
                        self.store.update_tool_marker(
                            self.session_id,
                            marker_id,
                            activity_presentation=presentation,
                            record_trajectory=False,
                        )
                    except KeyError:
                        continue
                    self._rows[call_key] = (marker_id, presentation)

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


def project_approval_feedback(
    bridge, session_id: str, run_id: str, *, replace_presentation
) -> None:
    """Keep the latest facts on the owning tool before its run maps retire."""
    try:
        feedback = getattr(bridge, "approval_feedback_store", None)
        if feedback is None:
            return
        facts = feedback.snapshot(session_id, run_id)
        activity = bridge._tool_activity_runs.get(run_id)
        if activity is not None and activity.session_id == session_id:
            try:
                activity.feedback(facts, call_keys=feedback.projection_call_keys)
            except Exception:  # noqa: BLE001, S110 - optional display projection
                pass  # Optional projection cannot change tool or teardown behavior.
        raw_updates = []
        with bridge._raw_shell_marker_lock:
            for fact in facts:
                for call_key in feedback.projection_call_keys(fact.identity):
                    state = bridge._raw_shell_markers.get((run_id, call_key))
                    if (
                        state is not None
                        and state.session_id == session_id
                        and state.tool_presentation is not None
                    ):
                        state.tool_presentation = replace_presentation(
                            state.tool_presentation, approval_feedback=fact
                        )
                        raw_updates.append(state)
        for state in raw_updates:
            try:
                bridge._update_raw_shell_marker(state)
            except Exception:  # noqa: BLE001, S110 - optional display projection
                pass  # A missing display consumer cannot affect execution.
    except Exception:  # noqa: BLE001, S110 - optional display observation
        pass  # Optional observations never interrupt execution or retirement.


def observe_approval_step(
    bridge, session_id: str, run_id: str, step: AgentStep
) -> None:
    try:
        from tldw_chatbook.Agents.approval_observation import ApprovalObservation

        feedback = getattr(bridge, "approval_feedback_store", None)
        if feedback is None:
            return
        identity = feedback.context_for_call(
            run_id, step.call_id, fallback_tool_name=step.tool_name
        )
        if identity is not None and identity.session_id == session_id:
            kind = {
                "tool_execution_started": "dispatch_started",
                "tool_output": "backend_started",
                "tool_result": "tool_completed",
            }.get(step.kind)
            if kind:
                outcome = {
                    "dispatch_started": "starting",
                    "backend_started": "running",
                }.get(
                    kind,
                    step.tool_outcome
                    if step.tool_outcome
                    in {"success", "failure", "blocked", "timeout", "cancelled"}
                    else "unknown",
                )
                feedback.publish(ApprovalObservation(identity, kind, outcome))
        elif step.kind == "model_request_started":
            for fact in feedback.snapshot(session_id, run_id):
                if fact.execution_state == "tool_completed":
                    feedback.publish(
                        ApprovalObservation(fact.identity, "model_wait", "waiting")
                    )
        bridge.project_approval_feedback(session_id, run_id)
    except Exception:  # noqa: BLE001, S110 - optional display observation
        pass  # Optional observations never interrupt execution or retirement.


def clear_raw_shell_progress(bridge, run_ids, *, cancelled: bool, log) -> None:
    """Detach ended runs before settling rows so late worker events are ignored."""
    if not run_ids:
        return
    with bridge._raw_shell_marker_lock:
        states = [
            bridge._raw_shell_markers.pop(key)
            for key in tuple(bridge._raw_shell_markers)
            if key[0] in run_ids
        ]
    for state in states:
        try:
            bridge._settle_raw_shell_marker(
                state,
                outcome="cancelled" if cancelled else "failure",
                message=(f"{state.stderr}\n\n" if state.stderr else "")
                + "Run ended before a result was received. Process cleanup is unknown.",
            )
        except Exception as exc:  # noqa: BLE001 — display cannot prevent run teardown
            log.warning(
                f"Console shell activity display could not be settled ({type(exc).__name__})"
            )


def retire_child_approval_feedback(bridge, run_id: str, observation_owners) -> None:
    # Primary rows transfer their final facts in run_reply's existing finally.
    # Children have no primary tool projection and retire at THEIR terminal.
    if run_id not in bridge._tool_activity_runs:
        try:
            from tldw_chatbook.Agents.approval_observation import (
                forget_approval_contexts,
            )

            if bridge.approval_feedback_store is not None:
                bridge.approval_feedback_store.retire_run(run_id)
            for owner in observation_owners:
                forget_approval_contexts(owner, run_id)
        except Exception:  # noqa: BLE001, S110 - optional display cleanup
            pass  # Display cleanup cannot suppress the terminal callback.


def finish_approval_activity(
    bridge,
    session_id: str,
    tool_activity,
    tool_run_ids,
    raw_run_ids,
    cancelled: bool,
    providers,
    *,
    log,
) -> None:
    for activity_run_id in tool_run_ids | raw_run_ids:
        bridge.project_approval_feedback(session_id, activity_run_id)
    bridge._clear_raw_shell_progress(raw_run_ids, cancelled=cancelled)
    try:
        tool_activity.finish(cancelled)
    except Exception as exc:  # noqa: BLE001 — display cannot prevent run teardown
        log.warning(
            f"Console tool activity display could not be settled ({type(exc).__name__})"
        )
    from tldw_chatbook.Agents.approval_observation import (
        forget_approval_contexts,
    )

    for activity_run_id in tool_run_ids | raw_run_ids:
        bridge._tool_activity_runs.pop(activity_run_id, None)
        if bridge.approval_feedback_store is not None:
            bridge.approval_feedback_store.retire_run(activity_run_id)
        for provider in providers:
            forget_approval_contexts(provider, activity_run_id)


def format_tool_activity_detail(
    activity, presentation: ConsoleActivityPresentation
) -> str:
    metadata = ""
    if activity.raw_cli_presentation is not None:
        from tldw_chatbook.Chat.console_raw_cli import format_raw_cli_metadata

        metadata = format_raw_cli_metadata(activity.raw_cli_presentation) + "\n\n"
    feedback_text = ""
    if presentation.approval_feedback is not None:
        from tldw_chatbook.Chat.console_approval_feedback import (
            format_approval_feedback,
        )

        feedback_text = (
            format_approval_feedback(
                presentation.approval_feedback, tool_status=presentation.status
            )
            + "\n\n"
        )
    detail = (
        feedback_text
        + f"Arguments\n{presentation.arguments or 'Not retained'}\n\n"
        + metadata
        + (
            "Live output\n"
            if presentation.status == "running"
            and presentation.result_preview is not None
            else "Result\n"
        )
        + (
            activity.tool_output_full
            or (
                (presentation.result_preview or "Completed — no output")
                if presentation.result_preview is not None
                else "Waiting for result…"
            )
        )
    )
    return detail


def apply_raw_shell_progress(
    bridge,
    run_id: str,
    call_id: str,
    event,
    *,
    result_type,
    stream_type,
    replace_presentation,
    step_factory,
) -> None:
    """Apply bounded progress to the exact app-owned raw-shell marker."""
    key = (run_id, call_id)
    with bridge._raw_shell_marker_lock:
        state = bridge._raw_shell_markers.get(key)
        if state is None:
            return
        if isinstance(event, result_type):
            state.result = event
            return
        if not isinstance(event, stream_type):
            return
        try:
            state.stdout, state.stderr, clipped = bridge._append_raw_shell_preview(
                state.stdout,
                state.stderr,
                event,
            )
        except UnicodeError:
            return
        state.truncated = state.truncated or event.truncated or clipped
        started_at = state.presentation.started_at_monotonic
        if started_at is None:
            started_at = bridge._clock()
        state.presentation = replace_presentation(
            state.presentation,
            lifecycle_state="running",
            started_at_monotonic=started_at,
            elapsed_seconds=max(0.0, bridge._clock() - started_at),
            truncated=state.truncated,
        )
        bridge._update_raw_shell_marker(state)
    # A real child stream is backend evidence; early dispatch is not.
    # This local projection step is never appended to a runtime trace.
    bridge._observe_approval_step(
        state.session_id,
        run_id,
        step_factory(
            index=0, kind="tool_output", tool_name="shell_exec", call_id=call_id
        ),
    )
