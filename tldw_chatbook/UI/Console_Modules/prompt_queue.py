"""Visible Console prompt-queue presentation and draft dispatch authority.

The registry and coordinator in :mod:`tldw_chatbook.Chat` own queue state and
draining.  This module owns the UI boundary: content-free presentation,
optimistic queue admission, exact draft restoration on refusal, and the one
per-session Textual worker used to start a manual prompt chain.

``ConsolePromptQueueUIController`` deliberately owns no DOM.  Its dependencies
are named, late-bound callables supplied by ``wiring.py`` so tests and runtime
session switches are observed at call time rather than frozen at construction.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.css.query import NoMatches
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleControllerActivity,
    ConsoleDispatchRecoveryAction,
    ConsoleDispatchRecoveryState,
)
from tldw_chatbook.Chat.console_prompt_queue import (
    MAX_CONSOLE_QUEUE_ENTRIES,
    PromptQueueEntryPhase,
    PromptQueueMode,
    PromptQueueMutationResult,
    PromptQueuePauseReason,
    PromptQueueSnapshot,
    QueueMutationStatus,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleDraftStash


def commit_queued_draft_transaction(
    session_id: str,
    stash: "ConsoleDraftStash | None",
    *,
    composer: Any,
    visible_session_id: str | None,
    undo_histories: dict[str, Any],
    store: Any,
    sync_command_popup: Callable[[], None],
) -> None:
    """Clear only the admitted draft while keeping unsent text out of history."""

    if visible_session_id == session_id and composer is not None:
        if stash is None:
            composer.clear_draft()
        composer.clear_history()
        sync_command_popup()
    undo_histories.pop(session_id, None)
    try:
        store.set_session_draft(session_id, "")
    except KeyError:
        pass


class ConsolePromptDispatchStatus(str, Enum):
    """Typed outcome returned to every visible/programmatic send caller."""

    SENT = "sent"
    QUEUED = "queued"
    REFUSED = "refused"


@dataclass(frozen=True, slots=True)
class ConsolePromptDispatchResult:
    """Content-free result of one draft dispatch attempt."""

    status: ConsolePromptDispatchStatus
    session_id: str = ""
    detail: str = ""

    @property
    def accepted(self) -> bool:
        """Return whether the caller may truthfully say the draft was accepted."""

        return self.status is not ConsolePromptDispatchStatus.REFUSED


@dataclass(frozen=True, slots=True)
class ConsolePromptQueuePresentation:
    """Body-free, immutable projection consumed by queue widgets."""

    revision: int
    count: int
    send_label: str
    send_enabled: bool
    send_tooltip: str
    shelf_visible: bool
    state_label: str
    paused: bool
    next_preview: str
    pause_label: str
    primary_action: str
    pause_enabled: bool
    recovery_actions: tuple[ConsoleDispatchRecoveryAction, ...] = ()


def derive_prompt_queue_presentation(
    snapshot: PromptQueueSnapshot,
    activity: ConsoleControllerActivity,
    *,
    composer_collapsed: bool = False,
    dispatch_recovery: ConsoleDispatchRecoveryState | None = None,
) -> ConsolePromptQueuePresentation:
    """Derive exact visible queue vocabulary without reading a prompt body."""

    count = snapshot.total_count
    queue_owned = activity.accepted_live_turn or count > 0
    if activity.occupies_slot and not queue_owned:
        send_label = "Preparing..."
        send_enabled = False
        send_tooltip = "Wait for this turn to be accepted before queueing a message."
    elif queue_owned and count >= MAX_CONSOLE_QUEUE_ENTRIES:
        send_label = "Queue full"
        send_enabled = False
        send_tooltip = f"{count}/{MAX_CONSOLE_QUEUE_ENTRIES} · Manage to make room"
    elif queue_owned:
        send_label = "Queue"
        send_enabled = True
        send_tooltip = "Queue this draft after the current turn."
    else:
        send_label = "Send"
        send_enabled = True
        send_tooltip = "Send the active Console session draft."

    if snapshot.mode is PromptQueueMode.PAUSED:
        if snapshot.pause_reason is PromptQueuePauseReason.FAILED:
            state_label = "Turn failed"
            pause_label = "Retry"
            primary_action = "retry-failed"
        elif snapshot.pause_reason is PromptQueuePauseReason.STOPPED:
            state_label = "Turn stopped"
            pause_label = "Resume next"
            primary_action = "resume-next"
        elif snapshot.pause_reason is PromptQueuePauseReason.CONTEXT_CHANGED:
            state_label = "Context changed"
            pause_label = "Review"
            primary_action = "review"
        elif snapshot.pause_reason is PromptQueuePauseReason.DISPATCH_REFUSED:
            state_label = "Start refused"
            pause_label = "Try again"
            primary_action = "toggle-pause"
        else:
            state_label = "Paused"
            pause_label = "Resume"
            primary_action = "toggle-pause"
    elif snapshot.mode is PromptQueueMode.PAUSE_AFTER_TURN:
        state_label = "Pausing"
        pause_label = "Keep draining"
        primary_action = "toggle-pause"
    elif any(
        entry.phase is PromptQueueEntryPhase.STARTING for entry in snapshot.entries
    ):
        state_label = "Starting..."
        pause_label = "Pause"
        primary_action = "toggle-pause"
    else:
        state_label = "Draining"
        pause_label = "Pause"
        primary_action = "toggle-pause"

    next_preview = next(
        (
            entry.preview
            for entry in snapshot.entries
            if entry.phase is PromptQueueEntryPhase.WAITING
        ),
        "",
    )
    if dispatch_recovery is not None:
        state_label = dispatch_recovery.visible_copy
        pause_label = ""
        primary_action = "dispatch-recovery"
        pause_enabled = False
        recovery_actions = dispatch_recovery.actions
    else:
        pause_enabled = count > 0
        recovery_actions = ()
    return ConsolePromptQueuePresentation(
        revision=snapshot.revision,
        count=count,
        send_label=send_label,
        send_enabled=send_enabled,
        send_tooltip=send_tooltip,
        shelf_visible=count > 0 and not composer_collapsed,
        state_label=state_label,
        paused=snapshot.mode is PromptQueueMode.PAUSED,
        next_preview=next_preview,
        pause_label=pause_label,
        primary_action=primary_action,
        pause_enabled=pause_enabled,
        recovery_actions=recovery_actions,
    )


class ConsolePromptQueueRegion(Widget):
    """Always-mounted one-row queue shelf directly above the composer."""

    BUNDLED_CSS = """
    ConsolePromptQueueRegion {
        display: none;
        height: 1;
        min-height: 1;
        max-height: 1;
        width: 100%;
        background: $panel;
        color: $text;
    }

    ConsolePromptQueueRegion.-visible {
        display: block;
    }

    ConsolePromptQueueRegion > #console-prompt-queue-row {
        height: 1;
        width: 100%;
        layout: horizontal;
    }

    #console-prompt-queue-summary {
        width: auto;
        min-width: 20;
        height: 1;
        color: $warning;
    }

    #console-prompt-queue-preview {
        width: 1fr;
        height: 1;
        color: $text-muted;
        text-overflow: ellipsis;
    }

    #console-prompt-queue-manage {
        width: 8;
        min-width: 8;
        height: 1;
        min-height: 1;
        padding: 0 1;
    }

    #console-prompt-queue-pause {
        width: 15;
        min-width: 15;
        height: 1;
        min-height: 1;
        padding: 0 1;
    }

    ConsolePromptQueueRegion.-narrow #console-prompt-queue-preview {
        display: none;
    }
    """

    class ManageRequested(Message):
        """Request the focused manager for this shelf's owning session."""

        def __init__(self, session_id: str, revision: int) -> None:
            super().__init__()
            self.session_id = session_id
            self.revision = revision

    class PauseRequested(Message):
        """Request the presentation's pause/resume action."""

        def __init__(self, session_id: str, revision: int) -> None:
            super().__init__()
            self.session_id = session_id
            self.revision = revision

    def __init__(
        self,
        *args: Any,
        on_manage_requested: Callable[[str, int], None] | None = None,
        on_primary_requested: Callable[[str, int, str], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._session_id = ""
        self._presentation: ConsolePromptQueuePresentation | None = None
        self._last_render_key: tuple[Any, ...] | None = None
        self._on_manage_requested = on_manage_requested
        self._on_primary_requested = on_primary_requested

    def compose(self) -> ComposeResult:
        with Horizontal(id="console-prompt-queue-row"):
            yield Static("", id="console-prompt-queue-summary")
            yield Static("", id="console-prompt-queue-preview")
            yield Button("Manage", id="console-prompt-queue-manage")
            yield Button("Pause", id="console-prompt-queue-pause")

    def sync_presentation(
        self,
        session_id: str,
        presentation: ConsolePromptQueuePresentation,
    ) -> bool:
        """Apply a projection once; unchanged revision/presentation is a no-op."""

        key = (session_id, presentation)
        if key == self._last_render_key:
            return False
        self._last_render_key = key
        self._session_id = session_id
        self._presentation = presentation
        self.set_class(presentation.shelf_visible, "-visible")
        try:
            summary = self.query_one("#console-prompt-queue-summary", Static)
            preview = self.query_one("#console-prompt-queue-preview", Static)
            manage = self.query_one("#console-prompt-queue-manage", Button)
            pause = self.query_one("#console-prompt-queue-pause", Button)
        except NoMatches:
            return True
        summary.update(
            f"Queue {presentation.count}/{MAX_CONSOLE_QUEUE_ENTRIES} · "
            f"{presentation.state_label}"
        )
        preview.update(
            f' · Next: "{presentation.next_preview}"'
            if presentation.next_preview
            else ""
        )
        if presentation.primary_action == "dispatch-recovery":
            first = (
                presentation.recovery_actions[0]
                if presentation.recovery_actions
                else None
            )
            second = (
                presentation.recovery_actions[1]
                if len(presentation.recovery_actions) > 1
                else None
            )
            manage.label = first.label if first is not None else "Unavailable"
            manage.disabled = first is None or not first.enabled
            manage.tooltip = (
                first.disabled_reason or first.label
                if first is not None
                else presentation.state_label
            )
            pause.label = second.label if second is not None else "Unavailable"
            pause.disabled = second is None or not second.enabled
            pause.tooltip = (
                second.disabled_reason or second.label
                if second is not None
                else presentation.state_label
            )
        else:
            manage.label = "Manage"
            manage.disabled = presentation.count == 0
            manage.tooltip = "Open the prompt queue manager."
            pause.label = presentation.pause_label
            pause.disabled = not presentation.pause_enabled
            pause.tooltip = (
                f"{presentation.pause_label} this session's prompt queue."
            )
        self.refresh(layout=True)
        return True

    def on_resize(self) -> None:
        """Drop the optional preview before actions can collide."""

        self.set_class(self.size.width < 92, "-narrow")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        presentation = self._presentation
        if presentation is None:
            return
        if event.button.id == "console-prompt-queue-manage":
            event.stop()
            if (
                presentation.primary_action == "dispatch-recovery"
                and presentation.recovery_actions
                and self._on_primary_requested is not None
            ):
                self._on_primary_requested(
                    self._session_id,
                    presentation.revision,
                    presentation.recovery_actions[0].action_id.value,
                )
            elif self._on_manage_requested is not None:
                self._on_manage_requested(self._session_id, presentation.revision)
            else:
                self.post_message(
                    self.ManageRequested(self._session_id, presentation.revision)
                )
        elif event.button.id == "console-prompt-queue-pause":
            event.stop()
            if (
                presentation.primary_action == "dispatch-recovery"
                and len(presentation.recovery_actions) > 1
                and self._on_primary_requested is not None
            ):
                self._on_primary_requested(
                    self._session_id,
                    presentation.revision,
                    presentation.recovery_actions[1].action_id.value,
                )
            elif presentation.primary_action == "review":
                if self._on_manage_requested is not None:
                    self._on_manage_requested(
                        self._session_id, presentation.revision
                    )
                else:
                    self.post_message(
                        self.ManageRequested(
                            self._session_id, presentation.revision
                        )
                    )
            elif self._on_primary_requested is not None:
                self._on_primary_requested(
                    self._session_id,
                    presentation.revision,
                    presentation.primary_action,
                )
            else:
                self.post_message(
                    self.PauseRequested(self._session_id, presentation.revision)
                )


class ConsolePromptQueueUIController:
    """Join queue admission and normal chain launch behind one dispatcher."""

    def __init__(
        self,
        *,
        chat_controller_accessor: Callable[[], Any],
        ensure_active_session: Callable[[], None],
        blocked_reason_accessor: Callable[[], str],
        setup_blocked_reason_accessor: Callable[[], str],
        restore_stash: Callable[["ConsoleDraftStash | None"], None],
        append_system_message: Callable[[str], Awaitable[None]],
        notify: Callable[[str, str], None],
        focus_composer: Callable[[], None],
        inflight_stashes_accessor: Callable[[], dict[str, Any]],
        note_follow_intent: Callable[[], None],
        launch_chain: Callable[[str, str], None],
        commit_queued_draft: Callable[[str, "ConsoleDraftStash | None"], None],
        edit_refusal: Callable[[str], str],
        sync_ui: Callable[[], Awaitable[None]],
    ) -> None:
        self._chat_controller_accessor = chat_controller_accessor
        self._ensure_active_session = ensure_active_session
        self._blocked_reason_accessor = blocked_reason_accessor
        self._setup_blocked_reason_accessor = setup_blocked_reason_accessor
        self._restore_stash = restore_stash
        self._append_system_message = append_system_message
        self._notify = notify
        self._focus_composer = focus_composer
        self._inflight_stashes_accessor = inflight_stashes_accessor
        self._note_follow_intent = note_follow_intent
        self._launch_chain = launch_chain
        self._commit_queued_draft = commit_queued_draft
        self._edit_refusal = edit_refusal
        self._sync_ui = sync_ui

    async def handle_primary_intent(
        self, session_id: str, *, action: str, expected_revision: int
    ) -> None:
        """Apply the shelf's state-specific primary action and repaint."""

        if action in {"retry_response", "retry_anyway", "discard"}:
            controller = self._chat_controller_accessor()
            result = (
                await controller.discard_dispatch_recovery(session_id)
                if action == "discard"
                else await controller.retry_dispatch_recovery(session_id)
            )
            if not result.accepted:
                self._notify(
                    result.visible_copy or "That recovery action is unavailable.",
                    "warning",
                )
            await self._sync_ui()
            return
        if action == "toggle-pause":
            await self.handle_pause_intent(
                session_id, expected_revision=expected_revision
            )
            return
        result = await self.recover(
            session_id,
            action=action,
            expected_revision=expected_revision,
        )
        if not result.applied and result.status is not QueueMutationStatus.UNCHANGED:
            self._notify(
                result.detail or "That prompt queue action is unavailable.",
                "warning",
            )
        await self._sync_ui()

    async def handle_pause_intent(
        self, session_id: str, *, expected_revision: int
    ) -> None:
        """Apply a shelf pause intent, report refusal, and repaint."""

        result = await self.toggle_pause(
            session_id, expected_revision=expected_revision
        )
        if result.status is QueueMutationStatus.STALE_REVISION:
            self._notify("The prompt queue changed. Review it and try again.", "warning")
        elif not result.applied and result.status is not QueueMutationStatus.UNCHANGED:
            self._notify(
                result.detail or "That prompt queue action is unavailable.",
                "warning",
            )
        await self._sync_ui()

    def presentation_for(
        self, session_id: str, *, composer_collapsed: bool = False
    ) -> ConsolePromptQueuePresentation:
        """Return a body-free presentation for one session."""

        controller = self._chat_controller_accessor()
        snapshot = controller.prompt_queue_registry.snapshot(session_id)
        activity = controller.activity_for(session_id)
        return derive_prompt_queue_presentation(
            snapshot,
            activity,
            composer_collapsed=composer_collapsed,
            dispatch_recovery=controller.store.dispatch_recovery_for_session(
                session_id
            ),
        )

    def snapshot(self, session_id: str) -> PromptQueueSnapshot:
        """Return the immutable body-free snapshot for a pinned session."""

        return self._chat_controller_accessor().prompt_queue_registry.snapshot(
            session_id
        )

    def _latest_result(
        self, session_id: str, result: PromptQueueMutationResult
    ) -> PromptQueueMutationResult:
        """Pair an awaited recovery status with its final body-free revision."""

        latest = self.snapshot(session_id)
        if latest is result.snapshot:
            return result
        return PromptQueueMutationResult(
            result.status,
            latest,
            entry_id=result.entry_id,
            detail=result.detail,
        )

    def read_waiting_text(
        self, session_id: str, entry_id: str, *, expected_revision: int
    ) -> Any:
        """Materialize one selected edit target under a revision fence."""

        return self._chat_controller_accessor().prompt_queue_registry.read_waiting_text(
            session_id,
            entry_id=entry_id,
            expected_revision=expected_revision,
        )

    def edit_waiting(
        self,
        session_id: str,
        entry_id: str,
        *,
        text: str,
        expected_revision: int,
    ) -> PromptQueueMutationResult:
        """Edit one waiting entry in the pinned session."""

        if detail := self._edit_refusal(text):
            return PromptQueueMutationResult(
                QueueMutationStatus.INVALID,
                self.snapshot(session_id),
                detail=detail,
            )
        return self._chat_controller_accessor().edit_queued_prompt(
            session_id,
            entry_id=entry_id,
            text=text,
            expected_revision=expected_revision,
        )

    def move_waiting(
        self,
        session_id: str,
        entry_id: str,
        *,
        position: int,
        expected_revision: int,
    ) -> PromptQueueMutationResult:
        """Move one waiting entry to a zero-based waiting-list position."""

        return self._chat_controller_accessor().move_queued_prompt(
            session_id,
            entry_id=entry_id,
            new_index=position,
            expected_revision=expected_revision,
        )

    def remove_waiting(
        self, session_id: str, entry_id: str, *, expected_revision: int
    ) -> PromptQueueMutationResult:
        """Remove one waiting entry from the pinned session."""

        return self._chat_controller_accessor().remove_queued_prompt(
            session_id,
            entry_id=entry_id,
            expected_revision=expected_revision,
        )

    def clear_waiting(
        self, session_id: str, *, expected_revision: int
    ) -> PromptQueueMutationResult:
        """Clear waiting entries without touching a locked Starting entry."""

        return self._chat_controller_accessor().clear_queued_prompts(
            session_id, expected_revision=expected_revision
        )

    async def toggle_pause(
        self, session_id: str, *, expected_revision: int
    ) -> PromptQueueMutationResult:
        """Apply the shelf/manager's exact pause, keep-draining, or resume intent."""

        controller = self._chat_controller_accessor()
        snapshot = controller.prompt_queue_registry.snapshot(session_id)
        if snapshot.revision != expected_revision:
            return PromptQueueMutationResult(
                QueueMutationStatus.STALE_REVISION, snapshot
            )
        if snapshot.mode is PromptQueueMode.PAUSED:
            result = await controller.resume_prompt_queue(session_id)
            return self._latest_result(session_id, result)
        if snapshot.mode is PromptQueueMode.PAUSE_AFTER_TURN:
            return controller.keep_prompt_queue_draining(
                session_id, expected_revision=expected_revision
            )
        return controller.pause_prompt_queue_after_turn(
            session_id, expected_revision=expected_revision
        )

    def context_review(self, session_id: str) -> tuple[int | None, int]:
        """Return queue-baseline and current epochs for explicit review copy."""

        controller = self._chat_controller_accessor()
        snapshot = controller.prompt_queue_registry.snapshot(session_id)
        return (
            snapshot.expected_context_epoch,
            controller.store.conversation_context_epoch(session_id),
        )

    async def recover(
        self,
        session_id: str,
        *,
        action: str,
        expected_revision: int,
        reviewed_context_epoch: int | None = None,
    ) -> PromptQueueMutationResult:
        """Run one explicit paused-queue recovery action for a pinned session."""

        controller = self._chat_controller_accessor()
        snapshot = controller.prompt_queue_registry.snapshot(session_id)
        if snapshot.revision != expected_revision:
            return PromptQueueMutationResult(
                QueueMutationStatus.STALE_REVISION, snapshot
            )
        if action == "resume-next":
            result = await controller.skip_and_resume_prompt_queue(session_id)
            return self._latest_result(session_id, result)
        if action == "use-current-context":
            if reviewed_context_epoch is None:
                return PromptQueueMutationResult(
                    QueueMutationStatus.INVALID,
                    snapshot,
                    detail="Review the current context before using it.",
                )
            if (
                controller.store.conversation_context_epoch(session_id)
                != reviewed_context_epoch
            ):
                return PromptQueueMutationResult(
                    QueueMutationStatus.INVALID,
                    snapshot,
                    detail=(
                        "The conversation changed since review. Review the "
                        "current context again."
                    ),
                )
            result = await controller.use_current_context_and_resume_prompt_queue(
                session_id,
                expected_revision=expected_revision,
                reviewed_context_epoch=reviewed_context_epoch,
            )
            return self._latest_result(session_id, result)
        wanted_statuses = (
            {"failed"} if action == "retry-failed" else {"stopped", "interrupted"}
        )
        message = next(
            (
                item
                for item in reversed(controller.store.messages_for_session(session_id))
                if str(item.status) in wanted_statuses
            ),
            None,
        )
        if message is None:
            return PromptQueueMutationResult(
                QueueMutationStatus.INVALID,
                snapshot,
                detail="No matching stopped or failed turn is available.",
            )
        if action == "retry-failed":
            result = await controller.retry_failed_queue_turn(message.id)
            return self._latest_result(session_id, result)
        if action == "retry-stopped":
            result = await controller.retry_stopped_queue_turn(message.id)
            return self._latest_result(session_id, result)
        return PromptQueueMutationResult(
            QueueMutationStatus.INVALID,
            snapshot,
            detail="Unknown prompt queue recovery action.",
        )

    async def dispatch(
        self,
        draft: str,
        *,
        stash: "ConsoleDraftStash | None" = None,
    ) -> ConsolePromptDispatchResult:
        """Send now, queue behind accepted work, or refuse without draft loss."""

        blocked_reason = self._blocked_reason_accessor().strip()
        if blocked_reason:
            self._restore_stash(stash)
            setup_reason = self._setup_blocked_reason_accessor().strip()
            visible = (
                setup_reason
                if setup_reason
                and not blocked_reason.startswith(
                    "Console send blocked: Library Search/RAG"
                )
                else blocked_reason
            )
            await self._append_system_message(visible)
            if visible == setup_reason:
                self._notify(visible, "warning")
            self._focus_composer()
            return ConsolePromptDispatchResult(
                ConsolePromptDispatchStatus.REFUSED, detail=visible
            )

        self._ensure_active_session()
        controller = self._chat_controller_accessor()
        session_id = controller.store.active_session_id or ""
        snapshot = controller.prompt_queue_registry.snapshot(session_id)
        activity = controller.activity_for(session_id)

        if activity.preparing_before_acceptance and snapshot.total_count == 0:
            detail = "Preparing the current turn. Queueing becomes available once it is accepted."
            self._restore_stash(stash)
            self._notify(detail, "warning")
            return ConsolePromptDispatchResult(
                ConsolePromptDispatchStatus.REFUSED,
                session_id=session_id,
                detail=detail,
            )

        if activity.accepted_live_turn or snapshot.total_count > 0:
            queued = controller.queue_prompt(
                session_id,
                text=draft,
                expected_revision=snapshot.revision,
            )
            if queued.status is QueueMutationStatus.REROUTE_NORMAL_SEND:
                return await self._stage_normal_chain(
                    controller, session_id, draft, stash
                )
            if queued.applied:
                self._commit_queued_draft(session_id, stash)
                await self._sync_ui()
                return ConsolePromptDispatchResult(
                    ConsolePromptDispatchStatus.QUEUED, session_id=session_id
                )
            return self._refuse_queue_mutation(queued, session_id, stash)

        refusal = controller.send_refusal_copy(session_id)
        if refusal:
            self._restore_stash(stash)
            self._notify(refusal, "warning")
            return ConsolePromptDispatchResult(
                ConsolePromptDispatchStatus.REFUSED,
                session_id=session_id,
                detail=refusal,
            )
        return await self._stage_normal_chain(controller, session_id, draft, stash)

    async def _stage_normal_chain(
        self,
        controller: Any,
        session_id: str,
        draft: str,
        stash: "ConsoleDraftStash | None",
    ) -> ConsolePromptDispatchResult:
        # Re-check the controller gate at the exact manual/queue boundary.
        # An accepted-turn race is retried as queue admission once, while a
        # pre-acceptance run is refused and the draft is restored.
        activity = controller.activity_for(session_id)
        if activity.preparing_before_acceptance:
            detail = (
                "Preparing the current turn. Queueing becomes available once it "
                "is accepted."
            )
            self._restore_stash(stash)
            self._notify(detail, "warning")
            return ConsolePromptDispatchResult(
                ConsolePromptDispatchStatus.REFUSED,
                session_id=session_id,
                detail=detail,
            )
        if activity.accepted_live_turn:
            snapshot = controller.prompt_queue_registry.snapshot(session_id)
            queued = controller.queue_prompt(
                session_id, text=draft, expected_revision=snapshot.revision
            )
            if queued.applied:
                self._commit_queued_draft(session_id, stash)
                await self._sync_ui()
                return ConsolePromptDispatchResult(
                    ConsolePromptDispatchStatus.QUEUED, session_id=session_id
                )
            if queued.status is not QueueMutationStatus.REROUTE_NORMAL_SEND:
                return self._refuse_queue_mutation(queued, session_id, stash)
        inflight = self._inflight_stashes_accessor()
        if stash is not None:
            inflight[session_id] = stash
        else:
            inflight.pop(session_id, None)
        self._note_follow_intent()
        self._launch_chain(draft, session_id)
        return ConsolePromptDispatchResult(
            ConsolePromptDispatchStatus.SENT, session_id=session_id
        )

    def _refuse_queue_mutation(
        self,
        result: PromptQueueMutationResult,
        session_id: str,
        stash: "ConsoleDraftStash | None",
    ) -> ConsolePromptDispatchResult:
        self._restore_stash(stash)
        if result.status is QueueMutationStatus.FULL:
            detail = (
                "Queue full "
                f"({result.snapshot.total_count}/{MAX_CONSOLE_QUEUE_ENTRIES}). "
                "Manage or remove an item."
            )
        elif result.status is QueueMutationStatus.STALE_REVISION:
            detail = "The prompt queue changed. Review it and try again."
        else:
            detail = result.detail or "This draft could not be queued."
        self._notify(detail, "warning")
        return ConsolePromptDispatchResult(
            ConsolePromptDispatchStatus.REFUSED,
            session_id=session_id,
            detail=detail,
        )


__all__ = [
    "ConsolePromptDispatchResult",
    "ConsolePromptDispatchStatus",
    "ConsolePromptQueuePresentation",
    "ConsolePromptQueueRegion",
    "ConsolePromptQueueUIController",
    "commit_queued_draft_transaction",
    "derive_prompt_queue_presentation",
]
