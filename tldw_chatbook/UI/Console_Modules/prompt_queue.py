"""Visible Console prompt-queue presentation and draft dispatch authority.

The registry and coordinator in :mod:`tldw_chatbook.Chat` own queue state and
draining.  This module owns the UI boundary: content-free presentation,
optimistic queue admission, and synchronous transfer into app-owned runtime
custody before the accepted composer revision is committed.

``ConsolePromptQueueUIController`` deliberately owns no DOM.  Its dependencies
are named, late-bound callables supplied by ``wiring.py`` so tests and runtime
session switches are observed at call time rather than frozen at construction.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, TYPE_CHECKING

from loguru import logger
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.css.query import NoMatches
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleControllerActivity,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_display_state import (
    QUEUE_REASON_FULL,
    QUEUE_REASON_PREPARING,
    QUEUE_REASON_RUN_HOLD,
    SEND_LABEL_SENDING,
)
from tldw_chatbook.Chat.console_prompt_queue import (
    MAX_CONSOLE_QUEUE_ENTRIES,
    PromptQueueEntryPhase,
    PromptQueueMode,
    PromptQueueMutationResult,
    PromptQueuePauseReason,
    PromptQueueSnapshot,
    QueueMutationStatus,
    make_prompt_preview,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tldw_chatbook.Chat.console_turn_context import ConsoleTurnConfigurationSnapshot
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
    notify: Callable[[str], None] | None = None,
) -> None:
    """Clear only the admitted draft while keeping unsent text out of history.

    TASK-33620.15.2: the sent-draft commit takes it out, so a hidden chat
    keeps what was typed after it, and a draft that cannot leave is
    announced as queued instead of left there silently.
    """
    from .sent_draft import take_out_sent_draft

    undo_histories.pop(session_id, None)
    take_out_sent_draft(
        session_id,
        stash,
        composer=composer,
        visible_session_id=visible_session_id,
        store=store,
        undo_histories=undo_histories,
        notify=notify or (lambda _text: None),
        verb="queued",
    )
    if visible_session_id == session_id and composer is not None:
        sync_command_popup()


#: Assistant-message statuses each typed queue retry action can re-run.
_RECOVERY_TURN_STATUSES: dict[str, frozenset[str]] = {
    "retry-failed": frozenset({"failed"}),
    "retry-stopped": frozenset({"stopped", "interrupted"}),
}
#: Cells of the failed turn's prompt the one-row shelf names it by.
RECOVERY_TURN_PREVIEW_CELLS = 32


@dataclass(frozen=True, slots=True)
class ConsoleQueueRecoveryTurn:
    """The exact paused-queue turn a typed retry action re-runs."""

    message_id: str
    preview: str = field(repr=False)


class ConsolePromptDispatchStatus(str, Enum):
    """Typed outcome returned to every visible/programmatic send caller."""

    SENT = "sent"
    QUEUED = "queued"
    REFUSED = "refused"
    #: Parked behind a hook review that a worker now owns: neither sent nor
    #: refused yet, and still cancellable (TASK-33621.28).
    AWAITING_REVIEW = "awaiting_review"


@dataclass(frozen=True, slots=True)
class ConsolePromptDispatchResult:
    """Content-free result of one draft dispatch attempt."""

    status: ConsolePromptDispatchStatus
    session_id: str = ""
    detail: str = ""

    @property
    def accepted(self) -> bool:
        """Return whether the caller may truthfully say the draft was accepted."""

        return self.status in (
            ConsolePromptDispatchStatus.SENT,
            ConsolePromptDispatchStatus.QUEUED,
        )


@dataclass(frozen=True, slots=True)
class ConsolePromptQueuePresentation:
    """Immutable projection consumed by queue widgets.

    It never carries a full prompt body: ``next_preview`` and a failed turn's
    name inside ``state_label`` are bounded one-line ``make_prompt_preview``
    renderings only.
    """

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
    turn_recovery_id: str | None = field(default=None, repr=False)


#: TASK-33621.2: the widest unsent-turn summary that leaves whole Restore and
#: Discard buttons on an 80-column shelf (80 - 9 - 15, less a margin).
TURN_RECOVERY_SUMMARY_CELLS = 52


def turn_recovery_label(
    reason: str, *, budget: int = TURN_RECOVERY_SUMMARY_CELLS
) -> str:
    """Return the unsent-turn summary: the refusal reason, fitted to ``budget``.

    Args:
        reason: The controller's refusal copy, or "" when none was retained.
        budget: Cells the label may use on the shelf.

    Returns:
        ``Not sent: <reason>`` (ellipsized), or the generic label.
    """
    text = " ".join(str(reason or "").split()).rstrip(".")
    if not text:
        return "Unsent turn needs attention"
    label = f"Not sent: {text}"
    return label if len(label) <= budget else label[: budget - 1].rstrip() + "…"


def derive_prompt_queue_presentation(
    snapshot: PromptQueueSnapshot,
    activity: ConsoleControllerActivity,
    *,
    composer_collapsed: bool = False,
    dispatch_recovery_blocked: bool = False,
    turn_recovery_id: str | None = None,
    turn_recovery_reason: str = "",
    failed_turn_preview: str | None = None,
    sending: bool = False,
) -> ConsolePromptQueuePresentation:
    """Derive exact visible queue vocabulary from bounded previews only.

    No full prompt body is read here.

    Args:
        snapshot: Body-free queue snapshot for the session.
        activity: The session's controller activity; a live accepted turn or
            an occupied agent slot decides the Send/Queue/Preparing label.
        composer_collapsed: True hides the shelf whatever its state.
        dispatch_recovery_blocked: True when a response recovery blocks the
            queue; the shelf shows a disabled Resume. The recovery's own
            actions live only on the #console-dispatch-recovery card: the
            shelf never carries them (TASK-33625.5).
        turn_recovery_id: An unsent turn needing attention. It outranks every
            other state and keeps the shelf visible with no queued entries.
        turn_recovery_reason: Why that turn was refused; the shelf states it
            in place of the generic label (TASK-33621.2).
        failed_turn_preview: Bounded one-line preview of the prompt whose turn
            failed and paused the queue (the caller's ``make_prompt_preview``),
            or ``None`` when the newest assistant message on the active
            transcript is not failed (no turn failed, or the failed attempt
            is off-path, as after a failed regeneration). ``""`` is a failed
            turn whose preceding user prompt is missing or has no previewable
            text (only whitespace or control characters, or only
            attachments); the shelf then shows a bare "Turn failed" with
            Retry. A FAILED pause given ``None`` offers Resume, because a
            Retry there could only refuse (TASK-33621.19).
        sending: An Enter is acknowledged on screen but not yet admitted
            (TASK-33620.5): Send reads "Sending..." and refuses, and its
            reason is the queue's, never the empty-draft copy.

    Returns:
        The immutable shelf/composer presentation: Send label and gate, shelf
        visibility, state label, next waiting preview, and primary action and
        its label.
    """

    count = snapshot.total_count
    queue_owned = activity.accepted_live_turn or count > 0
    # TASK-33620.4: a refusing state's tooltip is also the composer's reason
    # strip copy (its own queue slot, never the provider-setup one), so it
    # names the actual wait and fits the strip's 52-cell budget. Only a
    # prompt-chain turn is ever queue-accepted: regenerate / continue / an
    # agent wake occupy the slot with no chain, so no queue opens behind them.
    if sending and not activity.occupies_slot and not queue_owned:
        send_label = SEND_LABEL_SENDING
        send_enabled = False
        send_tooltip = QUEUE_REASON_PREPARING
    elif activity.occupies_slot and not queue_owned:
        send_label = "Preparing..."
        send_enabled = False
        send_tooltip = (
            QUEUE_REASON_PREPARING
            if activity.preparing_before_acceptance
            else QUEUE_REASON_RUN_HOLD
        )
    elif queue_owned and count >= MAX_CONSOLE_QUEUE_ENTRIES:
        send_label = "Queue full"
        send_enabled = False
        send_tooltip = QUEUE_REASON_FULL
    elif queue_owned:
        send_label = "Queue"
        send_enabled = True
        send_tooltip = "Queue this draft after the current turn."
    else:
        send_label = "Send"
        send_enabled = True
        send_tooltip = "Send the active Console session draft."

    if snapshot.mode is PromptQueueMode.PAUSED:
        if (
            snapshot.pause_reason is PromptQueuePauseReason.FAILED
            and failed_turn_preview is not None
        ):
            state_label = (
                f'Turn failed: "{failed_turn_preview}"'
                if failed_turn_preview
                else "Turn failed"
            )
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
    if turn_recovery_id is not None:
        # An empty queue drops the "Queue 0/10 · " prefix (sync_presentation).
        prefix = len(f"Queue {count}/{MAX_CONSOLE_QUEUE_ENTRIES} · ") if count else 0
        state_label = turn_recovery_label(
            turn_recovery_reason, budget=TURN_RECOVERY_SUMMARY_CELLS - prefix
        )
        pause_label = ""
        primary_action = "turn-recovery"
        pause_enabled = False
    elif dispatch_recovery_blocked:
        state_label = "Paused for response recovery"
        pause_label = "Resume"
        primary_action = "toggle-pause"
        pause_enabled = False
    else:
        pause_enabled = count > 0
    return ConsolePromptQueuePresentation(
        revision=snapshot.revision,
        count=count,
        send_label=send_label,
        send_enabled=send_enabled,
        send_tooltip=send_tooltip,
        shelf_visible=(count > 0 or turn_recovery_id is not None)
        and not composer_collapsed,
        state_label=state_label,
        paused=snapshot.mode is PromptQueueMode.PAUSED,
        next_preview=next_preview,
        pause_label=pause_label,
        primary_action=primary_action,
        pause_enabled=pause_enabled,
        turn_recovery_id=turn_recovery_id,
    )


class ConsolePromptQueueRegion(Widget):
    """Always-mounted one-row queue shelf directly above the composer."""

    # Every byte below is parsed at boot (ADR-097's boot CSS ratchet), so
    # nothing here restates what already holds (TASK-33621.19). The
    # #console-prompt-queue-row Horizontal has no rule: its own defaults
    # (1fr x 1fr, horizontal layout) fill this region, which max-height
    # clamps to one row. No min-height sits beside a fixed `height: 1`:
    # it could never take effect. The two Buttons keep Button's own
    # `width: auto` (label + line-pad): fixed widths clipped 'Manage' to
    # 'Mana' and 'Keep draining' to 'Keep' (TASK-33625.4). min-width 8
    # undoes Button's 16; Pause's 15 keeps its short labels one size.
    BUNDLED_CSS = """
    ConsolePromptQueueRegion {
        display: none;
        height: 1;
        max-height: 1;
        width: 100%;
        background: $panel;
        color: $text;
    }

    ConsolePromptQueueRegion.-visible {
        display: block;
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
        min-width: 8;
        height: 1;
        padding: 0 1;
    }

    #console-prompt-queue-pause {
        min-width: 15;
        height: 1;
        padding: 0 1;
    }

    ConsolePromptQueueRegion.-narrow #console-prompt-queue-preview {
        display: none;
    }

    /* TASK-33621.19: a narrow shelf's summary (the named 'Turn failed: "..."'
       is its longest label) truncates instead of pushing Retry off-screen.
       TASK-33625.4: it also gives up its min-width 20. The shelf spans the
       Console shell (rails open or not), so this only binds in a terminal
       under 47 columns, where Manage + Keep draining (27 cells) now stay
       whole instead of painting 'Keep dra'. */
    ConsolePromptQueueRegion.-narrow #console-prompt-queue-summary {
        width: 1fr;
        min-width: 0;
        text-wrap: nowrap;
        text-overflow: ellipsis;
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
            yield Static("", id="console-prompt-queue-summary", markup=False)
            yield Static("", id="console-prompt-queue-preview", markup=False)
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
            presentation.state_label
            if presentation.primary_action == "turn-recovery" and not presentation.count
            else f"Queue {presentation.count}/{MAX_CONSOLE_QUEUE_ENTRIES} · "
            f"{presentation.state_label}"
        )
        preview.update(
            f' · Next: "{presentation.next_preview}"'
            if presentation.next_preview
            else ""
        )
        if presentation.primary_action == "turn-recovery":
            manage.label = "Restore"
            manage.disabled = False
            manage.tooltip = "Restore this unsent turn."
            pause.label = "Discard"
            pause.disabled = False
            pause.tooltip = "Discard this unsent turn."
        else:
            manage.label = "Manage"
            manage.disabled = presentation.count == 0
            manage.tooltip = "Open the prompt queue manager."
            pause.label = presentation.pause_label
            pause.disabled = not presentation.pause_enabled
            pause.tooltip = f"{presentation.pause_label} this session's prompt queue."
        self.refresh(layout=True)
        # A Button caches its box model per its OWN layout count, so a new
        # label keeps the old width unless the Button itself is re-laid
        # out: 'Pause' -> 'Keep draining' painted 'Keep' (TASK-33625.4).
        manage.refresh(layout=True)
        pause.refresh(layout=True)
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
                presentation.primary_action == "turn-recovery"
                and presentation.turn_recovery_id is not None
                and self._on_primary_requested is not None
            ):
                self._on_primary_requested(
                    self._session_id,
                    presentation.revision,
                    f"turn-recovery:restore:{presentation.turn_recovery_id}",
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
                presentation.primary_action == "turn-recovery"
                and presentation.turn_recovery_id is not None
                and self._on_primary_requested is not None
            ):
                self._on_primary_requested(
                    self._session_id,
                    presentation.revision,
                    f"turn-recovery:discard:{presentation.turn_recovery_id}",
                )
            elif presentation.primary_action == "review":
                if self._on_manage_requested is not None:
                    self._on_manage_requested(self._session_id, presentation.revision)
                else:
                    self.post_message(
                        self.ManageRequested(self._session_id, presentation.revision)
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


def _preparation_refusal_detail(error: RuntimeError | ValueError) -> str:
    """Keep the internal preparation reason out of the composer notice."""
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    if (
        isinstance(error, RecoveryRequired)
        and str(error) == "console_snapshot_owner_changed"
    ):
        return "Chat or settings changed while preparing Send. Your draft was kept; send again."
    return str(error) or "Console runtime refused this turn."


class ConsolePromptQueueUIController:
    """Join queue admission and normal chain launch behind one dispatcher."""

    def __init__(
        self,
        *,
        chat_controller_accessor: Callable[[], Any],
        capture_configuration: Callable[[str], "ConsoleTurnConfigurationSnapshot"],
        ensure_active_session: Callable[[], None],
        blocked_reason_accessor: Callable[[], str],
        setup_blocked_reason_accessor: Callable[[], str],
        append_system_message: Callable[[str], Awaitable[None]],
        notify: Callable[[str, str], None],
        focus_composer: Callable[[], None],
        note_follow_intent: Callable[[], None],
        launch_chain: Callable[[str, str], str],
        commit_captured_draft: Callable[[str, "ConsoleDraftStash | None"], None],
        commit_queued_draft: Callable[[str, "ConsoleDraftStash | None"], None],
        turn_recovery_ids: Callable[[str], tuple[str, ...]],
        restore_turn_recovery: Callable[[str], Any],
        discard_turn_recovery: Callable[[str], bool],
        load_recovered_turn: Callable[[str], None],
        edit_refusal: Callable[[str], str],
        sync_ui: Callable[[], Awaitable[None]],
        turn_recovery_reason: Callable[[str], str] = lambda _session_id: "",
        sending_accessor: Callable[[str], bool] | None = None,
        precapture: (
            Callable[[str], Awaitable[Callable[[], AbstractContextManager[None]]]]
            | None
        ) = None,
        capture_configuration_async: Callable[
            [str, Any], Awaitable["ConsoleTurnConfigurationSnapshot"]
        ]
        | None = None,
        launch_chain_async: Callable[
            [str, str, "ConsoleDraftStash | None", Any], Awaitable[str]
        ]
        | None = None,
    ) -> None:
        """Wire the dispatcher to its owners.

        Args:
            precapture: TASK-33620.15. Reads a send's service-owned turn
                authority off the UI pump and returns the context in which
                ``capture_configuration``/``launch_chain`` use it. ``None``
                captures everything inline, as before.
        """
        self._precapture = precapture
        self._chat_controller_accessor = chat_controller_accessor
        self._capture_configuration = capture_configuration
        self._capture_configuration_async = capture_configuration_async
        self._async_capture_sync_source = capture_configuration
        self._ensure_active_session = ensure_active_session
        self._blocked_reason_accessor = blocked_reason_accessor
        self._setup_blocked_reason_accessor = setup_blocked_reason_accessor
        self._append_system_message = append_system_message
        self._notify = notify
        self._focus_composer = focus_composer
        self._note_follow_intent = note_follow_intent
        self._launch_chain = launch_chain
        self._launch_chain_async = launch_chain_async
        self._async_launch_sync_source = launch_chain
        self._commit_captured_draft = commit_captured_draft
        self._commit_queued_draft = commit_queued_draft
        self._turn_recovery_ids = turn_recovery_ids
        self._turn_recovery_reason = turn_recovery_reason
        self._restore_turn_recovery = restore_turn_recovery
        self._discard_turn_recovery = discard_turn_recovery
        self._load_recovered_turn = load_recovered_turn
        self._edit_refusal = edit_refusal
        self._sync_ui = sync_ui
        self._sending_accessor = sending_accessor

    async def handle_primary_intent(
        self,
        session_id: str,
        *,
        action: str,
        expected_revision: int,
        on_recovery_complete: Callable[[], None] | None = None,
    ) -> None:
        """Apply the shelf's state-specific primary action and repaint.

        Response recovery actions (retry_response, retry_anyway, discard) come
        only from the #console-dispatch-recovery card; the shelf never offers
        them (TASK-33625.5).

        Args:
            session_id: Session whose recovery or queue action is requested.
            action: Recovery action or prompt-queue mutation to apply.
            expected_revision: Queue revision used to reject stale mutations.
            on_recovery_complete: Optional synchronous callback for response
                recovery actions (retry_response, retry_anyway, discard). Runs
                after the action exits, including refusal, error or cancellation,
                and before repainting. Other actions do not invoke it.

        Raises:
            Exception: Propagates action failures. Completion or repaint failures
                propagate if no action failure is already being preserved.
            asyncio.CancelledError: Propagates action cancellation. Cancellation
                during completion or repaint also propagates unless preserving
                an earlier action failure or cancellation.
        """

        if action.startswith("turn-recovery:"):
            await self._handle_turn_recovery_intent(session_id, action)
            return
        if action in {"retry_response", "retry_anyway", "discard"}:
            action_error: BaseException | None = None
            try:
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
                elif result.queue_notice:
                    # The response settled, but the prompts queued behind it
                    # stopped at a context review (TASK-33621.19).
                    self._notify(result.queue_notice, "warning")
            except (Exception, asyncio.CancelledError) as exc:
                action_error = exc
                raise
            finally:
                # Controller cleanup releases the model-owned action claim,
                # including cancellation. Release this click before repainting.
                try:
                    if on_recovery_complete is not None:
                        on_recovery_complete()
                    await self._sync_ui()
                except (Exception, asyncio.CancelledError) as exc:
                    if action_error is None:
                        raise
                    logger.warning(
                        "Recovery repaint failed during action unwind ({})",
                        type(exc).__name__,
                    )
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
        elif result.detail:
            # A Retry/Resume next stopped at a context review says why.
            self._notify(result.detail, "warning")
        await self._sync_ui()

    async def _handle_turn_recovery_intent(self, session_id: str, action: str) -> None:
        """Apply one exact, body-free runtime recovery action."""

        operation, separator, turn_id = action.removeprefix("turn-recovery:").partition(
            ":"
        )
        recovery_ids = self._turn_recovery_ids(session_id)
        oldest_id = recovery_ids[0] if recovery_ids else None
        if (
            not separator
            or operation not in {"restore", "discard"}
            or not turn_id
            or oldest_id != turn_id
        ):
            self._notify("That unsent turn is no longer available.", "warning")
            await self._sync_ui()
            return
        if operation == "restore":
            try:
                self._restore_turn_recovery(turn_id)
            except (KeyError, RuntimeError):
                self._notify(
                    "That unsent turn could not be restored safely.", "warning"
                )
                await self._sync_ui()
                return
            self._load_recovered_turn(session_id)
            await self._sync_ui()
            self._focus_composer()
            return
        if not self._discard_turn_recovery(turn_id):
            self._notify("That unsent turn is no longer available.", "warning")
        await self._sync_ui()

    async def handle_pause_intent(
        self, session_id: str, *, expected_revision: int
    ) -> None:
        """Apply a shelf pause intent, report refusal, and repaint.

        An accepted result can still carry ``detail``: a Resume that the
        coordinator stopped at a context review (TASK-33621.19). That notice
        is shown too, so the press does not look dead.
        """

        result = await self.toggle_pause(
            session_id, expected_revision=expected_revision
        )
        if result.status is QueueMutationStatus.STALE_REVISION:
            self._notify(
                "The prompt queue changed. Review it and try again.", "warning"
            )
        elif not result.applied and result.status is not QueueMutationStatus.UNCHANGED:
            self._notify(
                result.detail or "That prompt queue action is unavailable.",
                "warning",
            )
        elif result.detail:
            self._notify(result.detail, "warning")
        await self._sync_ui()

    def presentation_for(
        self, session_id: str, *, composer_collapsed: bool = False
    ) -> ConsolePromptQueuePresentation:
        """Return a body-free presentation for one session."""

        controller = self._chat_controller_accessor()
        snapshot = controller.prompt_queue_registry.snapshot(session_id)
        activity = controller.activity_for(session_id)
        recovery_ids = self._turn_recovery_ids(session_id)
        turn_recovery_id = recovery_ids[0] if recovery_ids else None
        failed_turn = (
            self.recovery_turn(session_id, action="retry-failed")
            if snapshot.mode is PromptQueueMode.PAUSED
            and snapshot.pause_reason is PromptQueuePauseReason.FAILED
            else None
        )
        presentation = derive_prompt_queue_presentation(
            snapshot,
            activity,
            composer_collapsed=composer_collapsed,
            dispatch_recovery_blocked=(
                controller.prompt_queue_coordinator.dispatch_recovery_blocks_queue(
                    session_id
                )
            ),
            turn_recovery_id=turn_recovery_id,
            turn_recovery_reason=(
                self._turn_recovery_reason(session_id) if turn_recovery_id else ""
            ),
            failed_turn_preview=(
                failed_turn.preview if failed_turn is not None else None
            ),
            sending=bool(
                self._sending_accessor and self._sending_accessor(session_id)
            ),
        )
        runtime = getattr(controller, "_hooks_v2_runtime", None)
        if runtime is not None and runtime.has_received_intents(
            session_id, unpromoted_only=True
        ):
            return replace(
                presentation,
                send_label="Preparing...",
                send_enabled=False,
                send_tooltip="Preparing this turn; draft kept until acceptance.",
            )
        if controller._chat_start.is_prepared(session_id):
            return replace(
                presentation,
                send_label="Send",
                send_enabled=True,
                send_tooltip="Send this draft and withdraw the prepared background start.",
            )
        if controller._chat_start.is_accepted(session_id):
            # Native starts own no FIFO chain; they have already been accepted.
            return replace(
                presentation, send_label="Running", send_enabled=False, send_tooltip=""
            )
        return presentation

    def recovery_turn(
        self, session_id: str, *, action: str
    ) -> ConsoleQueueRecoveryTurn | None:
        """Return the exact turn a paused queue's retry ``action`` re-runs.

        A paused queue gates every other generation in its session, so the
        turn that paused it is the transcript's newest assistant message.
        Only that message is a retry target: an older failure elsewhere in
        the conversation did not pause this queue. TASK-33621.19: a FAILED
        pause with no failed turn used to offer Retry, which then refused
        with "No matching stopped or failed turn is available."

        Args:
            session_id: Session owning the paused queue.
            action: ``"retry-failed"`` or ``"retry-stopped"``.

        Returns:
            The newest assistant message and a one-line preview of the
            prompt it answered, or ``None`` when that message is not in a
            status ``action`` can retry.
        """

        statuses = _RECOVERY_TURN_STATUSES.get(action)
        if statuses is None:
            return None
        target = None
        try:
            messages = (
                self._chat_controller_accessor().store.iter_messages_newest_first(
                    session_id
                )
            )
            for item in messages:
                if target is None:
                    if item.role is not ConsoleMessageRole.ASSISTANT:
                        continue
                    if str(item.status) not in statuses:
                        return None
                    target = item
                elif item.role is ConsoleMessageRole.USER:
                    return ConsoleQueueRecoveryTurn(
                        target.id,
                        make_prompt_preview(
                            item.content, cell_budget=RECOVERY_TURN_PREVIEW_CELLS
                        ),
                    )
        except KeyError:
            return None
        return None if target is None else ConsoleQueueRecoveryTurn(target.id, "")

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
            configuration=self._capture_configuration(session_id),
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
        # TASK-33621.19: retry exactly the turn the shelf/manager named.
        turn = self.recovery_turn(session_id, action=action)
        if action in _RECOVERY_TURN_STATUSES and turn is None:
            return PromptQueueMutationResult(
                QueueMutationStatus.INVALID,
                snapshot,
                detail="No matching stopped or failed turn is available.",
            )
        if action == "retry-failed" and turn is not None:
            result = await controller.retry_failed_queue_turn(turn.message_id)
            return self._latest_result(session_id, result)
        if action == "retry-stopped" and turn is not None:
            result = await controller.retry_stopped_queue_turn(turn.message_id)
            return self._latest_result(session_id, result)
        return PromptQueueMutationResult(
            QueueMutationStatus.INVALID,
            snapshot,
            detail="Unknown prompt queue recovery action.",
        )

    async def _capture_configuration_for_dispatch(
        self, session_id: str, expected_controller: Any = None
    ) -> "ConsoleTurnConfigurationSnapshot":
        """Prepare production snapshots while preserving injected sync callbacks."""
        if self._capture_configuration_async is None:
            return self._capture_configuration(session_id)
        from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

        controller = expected_controller or self._chat_controller_accessor()
        callback = self._capture_configuration_async
        context = await callback(session_id, controller)
        if (
            self._chat_controller_accessor() is not controller
            or self._capture_configuration_async is not callback
        ):
            raise RecoveryRequired("console_snapshot_owner_changed")
        return context

    async def dispatch(
        self,
        draft: str,
        *,
        session_id: str | None = None,
        stash: "ConsoleDraftStash | None" = None,
    ) -> ConsolePromptDispatchResult:
        """Send now, queue behind accepted work, or refuse without draft loss.

        ``session_id`` pins visible composer sends to the chat that owned the
        captured draft. Other callers retain the active-session fallback.

        TASK-33620.15: a pinned send that will build a turn reads its
        service-owned turn authority on a worker thread, after the send gate
        and before the session-pinned gates. Keys flow during that await, so
        the gate, which reads the VISIBLE chat, is read first: in the stretch
        where the hook gate checked that this send's chat is the visible one.
        """
        blocked_reason = self._blocked_reason_accessor().strip()
        if blocked_reason:
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

        prepared: Callable[[], AbstractContextManager[None]] = contextlib.nullcontext
        if (
            session_id is not None
            and self._precapture is not None
            and (
                self._capture_configuration_async is None
                or self._capture_configuration is not self._async_capture_sync_source
            )
            and self._dispatch_builds_turn(session_id)
        ):
            prepared = await self._precapture(session_id)
        if session_id is None:
            self._ensure_active_session()
        controller = self._chat_controller_accessor()
        if session_id is None:
            session_id = controller.store.active_session_id or ""
        chat_start = getattr(controller, "_chat_start", None)
        if chat_start is not None:
            await chat_start.withdraw_for_manual(session_id)
        snapshot = controller.prompt_queue_registry.snapshot(session_id)
        activity = controller.activity_for(session_id)

        if activity.preparing_before_acceptance and snapshot.total_count == 0:
            detail = "Preparing the current turn. Queueing becomes available once it is accepted."
            self._notify(detail, "warning")
            return ConsolePromptDispatchResult(
                ConsolePromptDispatchStatus.REFUSED,
                session_id=session_id,
                detail=detail,
            )

        if activity.accepted_live_turn or snapshot.total_count > 0:
            try:
                with prepared():
                    configuration = await self._capture_configuration_for_dispatch(
                        session_id, controller
                    )
            except (RuntimeError, ValueError) as exc:
                if self._capture_configuration_async is None:
                    raise
                detail = _preparation_refusal_detail(exc)
                self._notify(detail, "warning")
                return ConsolePromptDispatchResult(
                    ConsolePromptDispatchStatus.REFUSED,
                    session_id=session_id,
                    detail=detail,
                )
            queued = await controller.queue_prompt(
                session_id,
                text=draft,
                expected_revision=snapshot.revision,
                configuration=configuration,
            )
            if queued.status is QueueMutationStatus.REROUTE_NORMAL_SEND:
                return await self._stage_normal_chain(
                    controller, session_id, draft, stash, prepared
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
            self._notify(refusal, "warning")
            return ConsolePromptDispatchResult(
                ConsolePromptDispatchStatus.REFUSED,
                session_id=session_id,
                detail=refusal,
            )
        return await self._stage_normal_chain(
            controller, session_id, draft, stash, prepared
        )

    def _dispatch_builds_turn(self, session_id: str) -> bool:
        """Whether ``dispatch`` would now capture a turn rather than refuse it.

        TASK-33620.15: the preparing gate, read-only, so a send it refuses
        is refused at once. The rarer refusal copy (emergency stop, the
        parallel-run cap, recovery) is evaluated once, after the read, where
        ``dispatch`` always did.
        """
        controller = self._chat_controller_accessor()
        snapshot = controller.prompt_queue_registry.snapshot(session_id)
        activity = controller.activity_for(session_id)
        return not (activity.preparing_before_acceptance and snapshot.total_count == 0)

    async def _stage_normal_chain(
        self,
        controller: Any,
        session_id: str,
        draft: str,
        stash: "ConsoleDraftStash | None",
        prepared: Callable[[], AbstractContextManager[None]] = contextlib.nullcontext,
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
            self._notify(detail, "warning")
            return ConsolePromptDispatchResult(
                ConsolePromptDispatchStatus.REFUSED,
                session_id=session_id,
                detail=detail,
            )
        if activity.accepted_live_turn:
            snapshot = controller.prompt_queue_registry.snapshot(session_id)
            try:
                with prepared():
                    configuration = await self._capture_configuration_for_dispatch(
                        session_id, controller
                    )
            except (RuntimeError, ValueError) as exc:
                if self._capture_configuration_async is None:
                    raise
                detail = _preparation_refusal_detail(exc)
                self._notify(detail, "warning")
                return ConsolePromptDispatchResult(
                    ConsolePromptDispatchStatus.REFUSED,
                    session_id=session_id,
                    detail=detail,
                )
            queued = await controller.queue_prompt(
                session_id,
                text=draft,
                expected_revision=snapshot.revision,
                configuration=configuration,
            )
            if queued.applied:
                self._commit_queued_draft(session_id, stash)
                await self._sync_ui()
                return ConsolePromptDispatchResult(
                    ConsolePromptDispatchStatus.QUEUED, session_id=session_id
                )
            if queued.status is not QueueMutationStatus.REROUTE_NORMAL_SEND:
                return self._refuse_queue_mutation(queued, session_id, stash)
        self._note_follow_intent()
        try:
            if (
                self._launch_chain_async is not None
                and self._launch_chain is self._async_launch_sync_source
            ):
                await self._launch_chain_async(draft, session_id, stash, controller)
            else:
                with prepared():
                    self._launch_chain(draft, session_id)
        except (RuntimeError, ValueError) as exc:
            detail = _preparation_refusal_detail(exc)
            self._notify(detail, "warning")
            return ConsolePromptDispatchResult(
                ConsolePromptDispatchStatus.REFUSED,
                session_id=session_id,
                detail=detail,
            )
        self._commit_captured_draft(session_id, stash)
        return ConsolePromptDispatchResult(
            ConsolePromptDispatchStatus.SENT, session_id=session_id
        )

    def _refuse_queue_mutation(
        self,
        result: PromptQueueMutationResult,
        session_id: str,
        stash: "ConsoleDraftStash | None",
    ) -> ConsolePromptDispatchResult:
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
    "ConsoleQueueRecoveryTurn",
    "commit_queued_draft_transaction",
    "derive_prompt_queue_presentation",
]
