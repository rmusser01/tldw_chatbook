from __future__ import annotations
from typing import Any, Callable


class ConsoleSessionController:
    def __init__(
        self,
        *,
        read_trace_recovery_dispatch: Callable[[], Callable[..., Any]],
        read_trace_recovery_state: Callable[[], Callable[..., Any]],
        read_trace_recovery_started: Callable[[], Callable[..., Any]],
        read_trace_recovery_finished: Callable[[], Callable[..., Any]],
    ) -> None:
        """Additional parameters and assignments in the existing constructor."""
        self._read_trace_recovery_dispatch = read_trace_recovery_dispatch
        self._read_trace_recovery_state = read_trace_recovery_state
        self._read_trace_recovery_started = read_trace_recovery_started
        self._read_trace_recovery_finished = read_trace_recovery_finished

    async def _dispatch_console_trace_recovery(
        self, action: str, preparation_id: str
    ) -> object:
        """Route a pre-dispatch card action; a cancelled hold refills the composer."""

        controller = self._ensure_console_chat_controller()
        held = controller.trace_call_recovery_preparation()
        # TASK-34350: read the held text BEFORE the action: the UI sync that
        # follows it mirrors the (empty) composer back into the session draft.
        held_text = (
            held.executed_draft
            if held is not None
            and held.preparation_id == preparation_id
            and controller.context_compaction_hold(preparation_id) is not None
            else ""
        )
        result = await self._read_trace_recovery_dispatch()(
            controller,
            action,
            preparation_id,
            on_started=self._read_trace_recovery_started(),
            on_finished=self._read_trace_recovery_finished(),
        )
        composer = self._console_composer_or_none()
        if (
            action == "cancel"
            and held_text
            and controller.context_compaction_hold(preparation_id) is None
            and composer is not None
            and not composer.draft_text().strip()
        ):
            # Cancel puts the held message back where it came from.
            composer.load_draft(held_text)
        return result

    def _console_trace_recovery_state(self) -> Any:
        """Project the active pre-dispatch pause, with a context hold's numbers."""

        controller = self._ensure_console_chat_controller()
        preparation = controller.trace_call_recovery_preparation()
        return self._read_trace_recovery_state()(
            preparation,
            context_hold=(
                controller.context_compaction_hold(preparation.preparation_id)
                if preparation is not None
                else None
            ),
        )
