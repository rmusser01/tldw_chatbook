"""Disposable Next Send factories for the current Console Inspector."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...Chat.console_chat_models import ConsoleContextSnapshot
    from ...Widgets.Console.console_next_send_selection import (
        ConsoleNextSendSelectionResult,
    )


def build_next_send_factories(
    self, controller: Any, session_id: str
) -> tuple[
    Callable[[], Awaitable[ConsoleContextSnapshot | ConsoleNextSendSelectionResult]],
    Callable[[], int | None],
    int | None,
    bool,
]:
    """Build the Next Send tab's snapshot/estimate factories (task-18300).

    Shared by BOTH ``ConsoleConversationInspector`` entry points (the
    cost chip and Ctrl+Shift+P) -- the two push the SAME modal
    instance, and the user can switch to the Next Send tab regardless
    of which tab it opened on, so a caller that skipped this would
    leave the tab showing nothing. Building the closures themselves is
    cheap: the Context view calls ``snapshot_factory`` only when opened
    or explicitly refreshed. Usage and exchange views do not prepare
    hidden next-send content.

    ``session_id`` is threaded in rather than re-read from the store
    because the composer only reflects the ACTIVE session; see
    ``_captured_draft``.
    """
    from ...Widgets.Console.console_next_send_selection import (
        ConsoleNextSendSelectionResult,
    )
    from ..Screens import chat_screen as owner

    def _captured_draft() -> str:
        if controller.store.active_session_id == session_id:
            try:
                return self.query_one(
                    "#console-native-composer", owner.ConsoleComposerBar
                ).draft_text()
            except (owner.NoMatches, owner.QueryError):
                pass
        session = next(
            (item for item in controller.store.sessions() if item.id == session_id),
            None,
        )
        return session.draft if session is not None else ""

    def _captured_attachments() -> tuple[owner.MessageAttachment, ...]:
        pending = controller.store.pending_attachments(session_id)
        return tuple(
            owner.MessageAttachment(
                data=attachment.data,
                mime_type=attachment.mime_type or "image/png",
                display_name=attachment.display_name,
                position=index,
            )
            for index, attachment in enumerate(pending)
        )

    def _session_workspace() -> str | None:
        session = next(
            (item for item in controller.store.sessions() if item.id == session_id),
            None,
        )
        return session.workspace_id if session is not None else None

    async def _factory() -> ConsoleContextSnapshot | ConsoleNextSendSelectionResult:
        current_draft = _captured_draft()
        current_attachments = _captured_attachments()
        current_staged_sources = tuple(
            controller.store.workspace_context.allowed_sources
        )
        captured_turn_context = controller.resolve_turn_execution_context(session_id)
        canvas_current = controller.capture_personal_context_preview_guard(session_id)
        owner = (
            controller.store.active_session_id,
            _session_workspace(),
            controller.store.workspace_context.active_workspace_id,
            controller.store.conversation_context_epoch(session_id),
            controller._lifecycle_revision_for(session_id),
            captured_turn_context,
        )
        captured_selection: list[tuple[Any, Any, Any]] = []
        snapshot = await controller.build_context_snapshot(
            draft=current_draft,
            attachments=current_attachments,
            staged_sources=current_staged_sources,
            session_id=session_id,
            profile_selection_sink=lambda builder, request, explanation: (
                captured_selection.append((builder, request, explanation))
            ),
        )
        # The preview planner owns exactly one profile build. If that
        # contract changes, no callback can safely be paired with this
        # snapshot by guessing which one was final.
        if len(captured_selection) != 1:
            return snapshot

        def inputs_current() -> bool:
            try:
                return (
                    canvas_current()
                    and owner
                    == (
                        controller.store.active_session_id,
                        _session_workspace(),
                        controller.store.workspace_context.active_workspace_id,
                        controller.store.conversation_context_epoch(session_id),
                        controller._lifecycle_revision_for(session_id),
                        controller.resolve_turn_execution_context(session_id),
                    )
                    and current_draft == _captured_draft()
                    and current_attachments == _captured_attachments()
                    and current_staged_sources
                    == tuple(controller.store.workspace_context.allowed_sources)
                )
            except Exception:  # noqa: BLE001 - stale inspection fails closed.
                return False

        builder, request, explanation = captured_selection[0]

        async def selection_current() -> bool:
            if not inputs_current():
                return False
            if not await controller.personal_context_selection_current(
                builder,
                request,
                explanation,
                provider_selection=captured_turn_context.provider_selection,
            ):
                return False
            return inputs_current()

        return ConsoleNextSendSelectionResult(
            snapshot,
            explanation if inputs_current() else None,
            selection_current,
        )

    def _estimate_factory() -> int | None:
        return self._estimate_tokens({"draft": _captured_draft()})

    token_estimate = _estimate_factory()
    in_progress = controller.run_state.status in owner.CONSOLE_ACTIVE_RUN_STATUSES
    return _factory, _estimate_factory, token_estimate, in_progress
