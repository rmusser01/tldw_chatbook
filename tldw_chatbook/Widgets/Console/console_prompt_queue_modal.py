"""Focused, session-pinned manager for the Console prompt queue."""

from __future__ import annotations

from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import Button, Static, TextArea

from tldw_chatbook.Chat.console_prompt_queue import (
    MAX_CONSOLE_QUEUE_ENTRIES,
    MAX_CONSOLE_QUEUED_PROMPT_LENGTH,
    PromptQueueEntryPhase,
    PromptQueueMode,
    PromptQueueMutationResult,
    PromptQueuePauseReason,
    PromptQueueSnapshot,
    QueueMutationStatus,
)
from tldw_chatbook.Utils.input_validation import validate_text_input
from tldw_chatbook.Widgets.cancel_confirmation_dialog import (
    CancelConfirmationDialog,
)
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin


class ConsolePromptQueueModal(SafeModalDismissMixin, ModalScreen[None]):
    """Manage one queue without ever retargeting to the viewed Console tab."""

    BINDINGS = [
        Binding("escape", "request_safe_cancel", "Close", show=False),
        Binding("j", "select_next", "Next", show=False),
        Binding("k", "select_previous", "Previous", show=False),
        Binding("e", "edit", "Edit", show=False),
        Binding("u", "move_up", "Move up", show=False),
        Binding("d", "move_down", "Move down", show=False),
        Binding("x", "remove", "Remove", show=False),
    ]
    SAFE_MODAL_CONTENT = "#console-prompt-queue-dialog"

    DEFAULT_CSS = """
    ConsolePromptQueueModal {
        align: center middle;
        background: $background 75%;
    }

    #console-prompt-queue-dialog {
        width: 76;
        max-width: 96%;
        height: 22;
        max-height: 92%;
        min-height: 18;
        background: $panel;
        border: solid $primary;
        padding: 1;
    }

    #console-prompt-queue-manager-title {
        height: 1;
        color: $accent;
        text-style: bold;
    }

    #console-prompt-queue-manager-state,
    #console-prompt-queue-manager-feedback {
        height: 1;
        color: $text-muted;
    }

    #console-prompt-queue-manager-feedback.-warning {
        color: $warning;
    }

    #console-prompt-queue-manager-list {
        height: 1fr;
        min-height: 6;
        border-top: solid $primary-background-lighten-2;
        border-bottom: solid $primary-background-lighten-2;
        padding: 0;
    }

    .console-prompt-queue-entry-row {
        height: 1;
        width: 100%;
    }

    .console-prompt-queue-entry-select {
        width: 1fr;
        height: 1;
        min-height: 1;
        content-align: left middle;
        padding: 0 1;
    }

    .console-prompt-queue-entry-phase {
        width: 12;
        height: 1;
        color: $text-muted;
        content-align: right middle;
    }

    .console-prompt-queue-entry-row.-selected {
        background: $primary-background;
    }

    #console-prompt-queue-edit-input {
        display: none;
        height: 3;
        border: solid $accent;
    }

    #console-prompt-queue-edit-input.-visible {
        display: block;
    }

    .console-prompt-queue-actions {
        height: 3;
        width: 100%;
        align-horizontal: left;
    }

    .console-prompt-queue-actions Button {
        height: 3;
        min-width: 8;
        padding: 0 1;
    }
    """

    def __init__(
        self,
        *,
        session_id: str,
        revision: int,
        queue_controller: Any,
    ) -> None:
        super().__init__()
        self.session_id = session_id
        self._revision = revision
        self._queue_controller = queue_controller
        self._snapshot = queue_controller.snapshot(session_id)
        self._selected_entry_id = (
            self._snapshot.entries[0].entry_id if self._snapshot.entries else None
        )
        self._editing_entry_id: str | None = None
        self._reviewed_context_epoch: int | None = None
        self._render_key: tuple[Any, ...] | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="console-prompt-queue-dialog"):
            yield Static("Prompt queue", id="console-prompt-queue-manager-title")
            yield Static("", id="console-prompt-queue-manager-state")
            yield Static("", id="console-prompt-queue-manager-feedback")
            yield VerticalScroll(id="console-prompt-queue-manager-list")
            yield TextArea(id="console-prompt-queue-edit-input")
            with Horizontal(classes="console-prompt-queue-actions"):
                yield Button("Edit", id="console-prompt-queue-edit")
                yield Button("Save", id="console-prompt-queue-save")
                yield Button("Up", id="console-prompt-queue-up")
                yield Button("Down", id="console-prompt-queue-down")
                yield Button("Remove", id="console-prompt-queue-remove")
                yield Button("Clear", id="console-prompt-queue-clear")
            with Horizontal(classes="console-prompt-queue-actions"):
                yield Button("Pause", id="console-prompt-queue-toggle-pause")
                yield Button("Resume next", id="console-prompt-queue-resume-next")
                yield Button("Review", id="console-prompt-queue-review-context")
                yield Button("Use current", id="console-prompt-queue-use-context")
                yield Button("Close", id="console-prompt-queue-close")
            with Horizontal(classes="console-prompt-queue-actions"):
                yield Button("Retry failed", id="console-prompt-queue-retry-failed")
                yield Button("Retry stopped", id="console-prompt-queue-retry-stopped")

    def on_mount(self) -> None:
        self._apply_snapshot(self._snapshot, force=True)
        self.set_interval(0.2, self._poll_snapshot)

    async def _poll_snapshot(self) -> None:
        snapshot = self._queue_controller.snapshot(self.session_id)
        if snapshot.revision != self._revision:
            self._apply_snapshot(snapshot)

    def _apply_snapshot(
        self, snapshot: PromptQueueSnapshot, *, force: bool = False
    ) -> None:
        """Render a new body-free revision while preserving entry identity."""

        key = (snapshot.revision, snapshot.entries, snapshot.mode, snapshot.pause_reason)
        if not force and key == self._render_key:
            return
        if self._render_key is not None and snapshot.revision != self._revision:
            self._reviewed_context_epoch = None
        self._render_key = key
        self._snapshot = snapshot
        self._revision = snapshot.revision
        entry_ids = tuple(entry.entry_id for entry in snapshot.entries)
        if self._selected_entry_id not in entry_ids:
            self._selected_entry_id = entry_ids[0] if entry_ids else None
        try:
            state = self.query_one("#console-prompt-queue-manager-state", Static)
            listing = self.query_one(
                "#console-prompt-queue-manager-list", VerticalScroll
            )
            pause = self.query_one("#console-prompt-queue-toggle-pause", Button)
            resume_next = self.query_one(
                "#console-prompt-queue-resume-next", Button
            )
            edit_button = self.query_one("#console-prompt-queue-edit", Button)
            save_button = self.query_one("#console-prompt-queue-save", Button)
            up_button = self.query_one("#console-prompt-queue-up", Button)
            down_button = self.query_one("#console-prompt-queue-down", Button)
            remove_button = self.query_one("#console-prompt-queue-remove", Button)
            clear_button = self.query_one("#console-prompt-queue-clear", Button)
            retry_failed = self.query_one(
                "#console-prompt-queue-retry-failed", Button
            )
            retry_stopped = self.query_one(
                "#console-prompt-queue-retry-stopped", Button
            )
            review = self.query_one(
                "#console-prompt-queue-review-context", Button
            )
            use_current = self.query_one(
                "#console-prompt-queue-use-context", Button
            )
        except NoMatches:
            return
        reason = snapshot.pause_reason.value.replace("_", " ") if snapshot.pause_reason else ""
        state.update(
            f"Queue {snapshot.total_count}/{MAX_CONSOLE_QUEUE_ENTRIES} · "
            f"{snapshot.mode.value.replace('_', ' ')}"
            + (f" · {reason}" if reason else "")
        )
        recovery_pause = (
            snapshot.mode is PromptQueueMode.PAUSED
            and snapshot.pause_reason
            not in {
                PromptQueuePauseReason.MANUAL,
                PromptQueuePauseReason.DISPATCH_REFUSED,
            }
        )
        pause.label = (
            "Paused"
            if recovery_pause
            else "Try again"
            if snapshot.pause_reason is PromptQueuePauseReason.DISPATCH_REFUSED
            else "Resume"
            if snapshot.mode is PromptQueueMode.PAUSED
            else "Keep draining"
            if snapshot.mode is PromptQueueMode.PAUSE_AFTER_TURN
            else "Pause"
        )
        pause.disabled = snapshot.total_count == 0 or recovery_pause
        resume_next.label = (
            "Skip & resume"
            if snapshot.pause_reason is PromptQueuePauseReason.FAILED
            else "Resume next"
        )
        selected = next(
            (
                entry
                for entry in snapshot.entries
                if entry.entry_id == self._selected_entry_id
            ),
            None,
        )
        selected_waiting = bool(
            selected is not None
            and selected.phase is PromptQueueEntryPhase.WAITING
        )
        selected_index = self._selected_index()
        edit_button.disabled = not selected_waiting
        save_button.disabled = self._editing_entry_id is None
        up_button.disabled = not selected_waiting or selected_index == 0
        down_button.disabled = (
            not selected_waiting
            or selected_index is None
            or selected_index >= snapshot.waiting_count - 1
        )
        remove_button.disabled = not selected_waiting
        clear_button.disabled = snapshot.waiting_count == 0
        retry_failed.disabled = snapshot.pause_reason is not PromptQueuePauseReason.FAILED
        retry_stopped.disabled = snapshot.pause_reason is not PromptQueuePauseReason.STOPPED
        resume_next.disabled = snapshot.pause_reason not in {
            PromptQueuePauseReason.FAILED,
            PromptQueuePauseReason.STOPPED,
        }
        review.disabled = (
            snapshot.pause_reason is not PromptQueuePauseReason.CONTEXT_CHANGED
        )
        use_current.disabled = review.disabled or self._reviewed_context_epoch is None
        listing.remove_children()
        for entry in snapshot.entries:
            row = Horizontal(
                Button(
                    f"{entry.position}. {entry.preview}",
                    id=f"console-prompt-queue-entry-{entry.entry_id}",
                    classes="console-prompt-queue-entry-select",
                ),
                Static(
                    "Starting..."
                    if entry.phase is PromptQueueEntryPhase.STARTING
                    else "Waiting",
                    classes="console-prompt-queue-entry-phase",
                ),
                classes="console-prompt-queue-entry-row",
            )
            row.set_class(
                entry.entry_id == self._selected_entry_id,
                "-selected",
            )
            listing.mount(row)
        self.call_after_refresh(self._restore_selection_focus)

    def _restore_selection_focus(self) -> None:
        if self._editing_entry_id is not None:
            try:
                edit = self.query_one("#console-prompt-queue-edit-input", TextArea)
            except NoMatches:
                pass
            else:
                if edit.has_class("-visible"):
                    edit.focus()
                    return
        if self._selected_entry_id is None:
            return
        try:
            self.query_one(
                f"#console-prompt-queue-entry-{self._selected_entry_id}", Button
            ).focus()
        except NoMatches:
            pass

    def _selected_index(self) -> int | None:
        waiting = [
            entry
            for entry in self._snapshot.entries
            if entry.phase is PromptQueueEntryPhase.WAITING
        ]
        for index, entry in enumerate(waiting):
            if entry.entry_id == self._selected_entry_id:
                return index
        return None

    def _show_feedback(self, text: str, *, warning: bool = False) -> None:
        feedback = self.query_one("#console-prompt-queue-manager-feedback", Static)
        feedback.update(text)
        feedback.set_class(warning, "-warning")

    def _accept_mutation(self, result: PromptQueueMutationResult) -> bool:
        if result.status in {QueueMutationStatus.APPLIED, QueueMutationStatus.UNCHANGED}:
            self._show_feedback("")
            self._apply_snapshot(result.snapshot, force=True)
            return True
        self._apply_snapshot(result.snapshot, force=True)
        copy = {
            QueueMutationStatus.STALE_REVISION: "Queue changed. Review it and try again.",
            QueueMutationStatus.LOCKED: "Starting prompts cannot be changed.",
            QueueMutationStatus.NOT_FOUND: "That prompt is no longer queued.",
            QueueMutationStatus.INVALID: result.detail or "That queue action is unavailable.",
        }.get(result.status, result.detail or "Queue action refused.")
        self._show_feedback(copy, warning=True)
        return False

    def action_select_next(self) -> None:
        """Select the next queue entry, wrapping at the end."""

        self._select_offset(1)

    def action_select_previous(self) -> None:
        """Select the previous queue entry, wrapping at the beginning."""

        self._select_offset(-1)

    def _select_offset(self, offset: int) -> None:
        entries = self._snapshot.entries
        if not entries:
            return
        ids = [entry.entry_id for entry in entries]
        try:
            index = ids.index(self._selected_entry_id)
        except ValueError:
            index = 0
        self._selected_entry_id = ids[(index + offset) % len(ids)]
        self._apply_snapshot(self._snapshot, force=True)

    def action_edit(self) -> None:
        """Begin editing the selected waiting prompt."""

        self._begin_edit()

    def action_move_up(self) -> None:
        """Move the selected waiting prompt one position earlier."""

        self._move_selected(-1)

    def action_move_down(self) -> None:
        """Move the selected waiting prompt one position later."""

        self._move_selected(1)

    def action_remove(self) -> None:
        """Request confirmation before removing the selected prompt."""

        self.run_worker(
            self._confirm_remove_selected(),
            exclusive=True,
            group="console-prompt-queue-confirm",
        )

    @on(Button.Pressed)
    async def handle_button(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id.startswith("console-prompt-queue-entry-"):
            event.stop()
            self._selected_entry_id = button_id.removeprefix(
                "console-prompt-queue-entry-"
            )
            self._apply_snapshot(self._snapshot, force=True)
            return
        handlers = {
            "console-prompt-queue-edit": self._begin_edit,
            "console-prompt-queue-save": self._save_edit,
            "console-prompt-queue-up": lambda: self._move_selected(-1),
            "console-prompt-queue-down": lambda: self._move_selected(1),
        }
        handler = handlers.get(button_id)
        if handler is not None:
            event.stop()
            handler()
        elif button_id == "console-prompt-queue-close":
            event.stop()
            await self.request_safe_cancel(source="visible")
        elif button_id == "console-prompt-queue-remove":
            event.stop()
            self.run_worker(
                self._confirm_remove_selected(),
                exclusive=True,
                group="console-prompt-queue-confirm",
            )
        elif button_id == "console-prompt-queue-clear":
            event.stop()
            self.run_worker(
                self._confirm_clear_waiting(),
                exclusive=True,
                group="console-prompt-queue-confirm",
            )
        elif button_id == "console-prompt-queue-toggle-pause":
            event.stop()
            self.run_worker(self._toggle_pause(), group="console-prompt-queue-modal")
        elif button_id in {
            "console-prompt-queue-resume-next",
            "console-prompt-queue-retry-failed",
            "console-prompt-queue-retry-stopped",
            "console-prompt-queue-use-context",
        }:
            event.stop()
            action = {
                "console-prompt-queue-resume-next": "resume-next",
                "console-prompt-queue-retry-failed": "retry-failed",
                "console-prompt-queue-retry-stopped": "retry-stopped",
                "console-prompt-queue-use-context": "use-current-context",
            }[button_id]
            self.run_worker(
                self._recover(action), group="console-prompt-queue-modal"
            )
        elif button_id == "console-prompt-queue-review-context":
            event.stop()
            baseline, current = self._queue_controller.context_review(
                self.session_id
            )
            self._reviewed_context_epoch = current
            self.query_one(
                "#console-prompt-queue-use-context", Button
            ).disabled = False
            self._show_feedback(
                f"Context review: queued baseline {baseline}; current {current}. "
                "Use current now adopts that reviewed version."
            )

    def has_unsaved_edit(self) -> bool:
        """Report whether the open edit view holds text the queue does not.

        TASK-31701: consumed by ``ChatScreen.flush_pending_work`` to veto
        navigation while a dirty edit is open -- the navigation seam
        dismisses pushed screens before switching, which would silently
        discard the typed text. Only a REAL divergence counts: an edit
        view showing the entry's current text loses nothing when
        dismissed, and an entry that changed or vanished under the edit
        has nothing recoverable to protect (the modal's own save path
        already refuses it with a feedback line).

        Returns:
            ``True`` when an edit is open and its text differs from the
            queued entry's current text.
        """
        if self._editing_entry_id is None:
            return False
        try:
            edit = self.query_one("#console-prompt-queue-edit-input", TextArea)
        except NoMatches:
            return False
        result = self._queue_controller.read_waiting_text(
            self.session_id,
            self._editing_entry_id,
            expected_revision=self._revision,
        )
        if result.status is not QueueMutationStatus.APPLIED or result.text is None:
            return False
        return edit.text != result.text

    def _begin_edit(self) -> None:
        entry_id = self._selected_entry_id
        if entry_id is None:
            return
        result = self._queue_controller.read_waiting_text(
            self.session_id,
            entry_id,
            expected_revision=self._revision,
        )
        if result.status is not QueueMutationStatus.APPLIED or result.text is None:
            self._apply_snapshot(self._queue_controller.snapshot(self.session_id), force=True)
            self._show_feedback(
                "That prompt changed before editing. Review the queue and try again.",
                warning=True,
            )
            return
        edit = self.query_one("#console-prompt-queue-edit-input", TextArea)
        self._editing_entry_id = entry_id
        edit.text = result.text
        edit.add_class("-visible")
        self.query_one("#console-prompt-queue-save", Button).disabled = False
        edit.focus()

    def _save_edit(self) -> None:
        if self._editing_entry_id is None:
            return
        edit = self.query_one("#console-prompt-queue-edit-input", TextArea)
        if not validate_text_input(
            edit.text,
            max_length=MAX_CONSOLE_QUEUED_PROMPT_LENGTH,
            allow_html=False,
        ):
            self._show_feedback(
                "Prompt blocked: remove unsafe markup or shorten it before saving.",
                warning=True,
            )
            return
        result = self._queue_controller.edit_waiting(
            self.session_id,
            self._editing_entry_id,
            text=edit.text,
            expected_revision=self._revision,
        )
        if result.status in {
            QueueMutationStatus.APPLIED,
            QueueMutationStatus.UNCHANGED,
        }:
            self._editing_entry_id = None
            edit.text = ""
            edit.remove_class("-visible")
        self._accept_mutation(result)

    def _move_selected(self, offset: int) -> None:
        index = self._selected_index()
        if index is None or self._selected_entry_id is None:
            return
        waiting_count = self._snapshot.waiting_count
        new_index = max(0, min(waiting_count - 1, index + offset))
        result = self._queue_controller.move_waiting(
            self.session_id,
            self._selected_entry_id,
            position=new_index,
            expected_revision=self._revision,
        )
        self._accept_mutation(result)

    async def _confirm_remove_selected(self) -> None:
        entry_id = self._selected_entry_id
        revision = self._revision
        if entry_id is None:
            return
        confirmed = await self.app.push_screen_wait(
            CancelConfirmationDialog(
                title="Remove queued prompt?",
                message="This unsent prompt will be discarded.",
                confirm_text="Remove",
                cancel_text="Keep",
            )
        )
        if not confirmed:
            self.call_after_refresh(self._restore_selection_focus)
            return
        result = self._queue_controller.remove_waiting(
            self.session_id,
            entry_id,
            expected_revision=revision,
        )
        self._accept_mutation(result)

    async def _confirm_clear_waiting(self) -> None:
        revision = self._revision
        if self._snapshot.waiting_count == 0:
            return
        confirmed = await self.app.push_screen_wait(
            CancelConfirmationDialog(
                title="Clear waiting prompts?",
                message=(
                    f"Discard {self._snapshot.waiting_count} unsent prompt(s)? "
                    "A Starting prompt is not affected."
                ),
                confirm_text="Clear waiting",
                cancel_text="Keep queue",
            )
        )
        if not confirmed:
            self.call_after_refresh(self._restore_selection_focus)
            return
        result = self._queue_controller.clear_waiting(
            self.session_id, expected_revision=revision
        )
        self._accept_mutation(result)

    async def _toggle_pause(self) -> None:
        result = await self._queue_controller.toggle_pause(
            self.session_id, expected_revision=self._revision
        )
        self._accept_mutation(result)

    async def _recover(self, action: str) -> None:
        result = await self._queue_controller.recover(
            self.session_id,
            action=action,
            expected_revision=self._revision,
            reviewed_context_epoch=(
                self._reviewed_context_epoch
                if action == "use-current-context"
                else None
            ),
        )
        self._accept_mutation(result)


__all__ = ["ConsolePromptQueueModal"]
