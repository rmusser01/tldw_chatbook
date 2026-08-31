"""Per-conversation Console Library authority editor."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, RadioButton, RadioSet, Static

from tldw_chatbook.Chat.console_display_state import (
    ConsoleLibraryPolicyDisplayState,
)
from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicyCandidate,
    ConsoleLibraryPolicySnapshot,
)
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin


@dataclass(frozen=True, slots=True)
class ConsoleLibraryPolicySaveOutcome:
    """Bounded result returned to the modal by its persistence controller."""

    status: Literal["saved", "conflict", "unavailable", "error"]
    snapshot: ConsoleLibraryPolicySnapshot
    copy: str


class ConsoleLibraryAccessModal(SafeModalDismissMixin, ModalScreen[None]):
    """Edit automatic retrieval and assistant access as independent axes."""

    SAFE_MODAL_CONTENT = "#console-library-access"
    BINDINGS = [("escape", "request_safe_cancel", "Cancel")]

    BUNDLED_CSS = """
    ConsoleLibraryAccessModal {
        align: center middle;
    }

    #console-library-access {
        width: 76;
        max-width: 96%;
        height: 34;
        max-height: 96%;
        border: tall $border;
        background: $surface;
        padding: 1 2;
    }

    #console-library-access-body {
        height: 1fr;
        min-height: 0;
        max-height: 24;
        scrollbar-gutter: stable;
    }

    .console-library-access-axis {
        height: auto;
        margin: 1 0;
    }

    .console-library-access-local,
    .console-library-access-axis-copy {
        height: auto;
        color: $text-muted;
    }

    .console-library-access-local {
        margin-bottom: 1;
    }

    .console-library-access-axis-title {
        height: auto;
        text-style: bold;
    }

    .console-library-access-actions {
        height: 1;
        min-height: 1;
        max-height: 1;
        margin-top: 1;
    }

    #console-library-access .console-library-access-actions Button {
        width: auto;
        min-width: 0;
        height: 1;
        min-height: 1;
        max-height: 1;
        border: none;
        padding: 0;
        margin-right: 1;
    }

    #library-access-feedback {
        color: $text-muted;
        margin-top: 1;
    }
    """

    def __init__(
        self,
        *,
        snapshot: ConsoleLibraryPolicySnapshot,
        state: ConsoleLibraryPolicyDisplayState,
        save_policy: Callable[
            [ConsoleLibraryPolicyCandidate],
            Awaitable[ConsoleLibraryPolicySaveOutcome],
        ],
        reload_policy: Callable[[], Awaitable[ConsoleLibraryPolicySnapshot]],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._snapshot = snapshot
        self._state = state
        self._save_policy = save_policy
        self._reload_policy = reload_policy
        self._operation_pending = False
        self._suppress_changes = False
        self._dirty = False

    def on_mount(self) -> None:
        """Capture opener focus, then focus the first usable policy choice."""
        super().on_mount()
        self.call_after_refresh(self._focus_initial_control)

    def _focus_initial_control(self) -> None:
        feedback = self.query_one("#library-access-feedback", Static)
        if feedback.can_focus:
            feedback.focus()
            return
        if self._state.editing_enabled:
            self.query_one("#library-auto-policy", RadioSet).focus()

    def compose(self) -> ComposeResult:
        with Vertical(id="console-library-access"):
            yield Static("Library access", classes="console-modal-header")
            yield Static(
                self._state.source_status,
                id="library-access-status",
                markup=False,
            )
            yield Static(
                "Stored only on this device. This conversation policy is not synced.",
                markup=False,
            )
            with VerticalScroll(id="console-library-access-body"):
                yield Static(
                    "Automatic retrieval",
                )
                yield Static(
                    "Choose whether the app searches your Library before each send.",
                    markup=False,
                )
                with RadioSet(
                    id="library-auto-policy",
                    classes="console-library-access-axis",
                ):
                    yield RadioButton(
                        "Never",
                        value=(
                            self._snapshot.auto_retrieve is ConsoleAutoRetrieve.NEVER
                        ),
                        id="library-auto-never",
                        disabled=not self._state.editing_enabled,
                    )
                    yield RadioButton(
                        "Automatic",
                        value=(
                            self._snapshot.auto_retrieve
                            is ConsoleAutoRetrieve.AUTOMATIC
                        ),
                        id="library-auto-automatic",
                        disabled=not self._state.editing_enabled,
                    )
                yield Static(
                    "Assistant Library access",
                )
                yield Static(
                    "Choose whether the assistant may use the built-in Library "
                    "capability during its work.",
                    markup=False,
                )
                with RadioSet(
                    id="library-agent-policy",
                    classes="console-library-access-axis",
                ):
                    yield RadioButton(
                        "Blocked",
                        value=(
                            self._snapshot.assistant_access
                            is ConsoleAssistantLibraryAccess.BLOCKED
                        ),
                        id="library-agent-blocked",
                        disabled=not self._state.editing_enabled,
                    )
                    yield RadioButton(
                        "Allowed",
                        value=(
                            self._snapshot.assistant_access
                            is ConsoleAssistantLibraryAccess.ALLOWED
                        ),
                        id="library-agent-allowed",
                        disabled=not self._state.editing_enabled,
                    )
                yield Static(
                    self._state.provider_intent_label,
                    id="library-access-provider",
                    markup=False,
                )
                yield Static(
                    self._state.resolved_destination_label,
                    id="library-access-destination",
                    markup=False,
                )
                feedback = Static(
                    self._state.feedback_copy,
                    id="library-access-feedback",
                    markup=False,
                )
                feedback.can_focus = self._state.feedback in {
                    "conflict",
                    "unavailable",
                    "error",
                }
                yield feedback
            with Horizontal(classes="console-library-access-actions"):
                yield Button(
                    "Save",
                    id="library-access-save",
                    variant="primary",
                    disabled=not self._state.save_enabled,
                )
                yield Button("Cancel", id="library-access-cancel")
                reload_button = Button("Reload", id="library-access-reload")
                reload_button.display = self._state.feedback == "conflict"
                yield reload_button
                retry_button = Button(
                    "Compare / Retry",
                    id="library-access-compare-retry",
                )
                retry_button.display = self._state.feedback == "conflict"
                yield retry_button
                discard = Button(
                    "Discard changes",
                    id="library-access-discard",
                    variant="warning",
                )
                discard.display = False
                yield discard

    def _candidate(self) -> ConsoleLibraryPolicyCandidate:
        automatic = self.query_one("#library-auto-automatic", RadioButton).value
        allowed = self.query_one("#library-agent-allowed", RadioButton).value
        return ConsoleLibraryPolicyCandidate(
            auto_retrieve=(
                ConsoleAutoRetrieve.AUTOMATIC
                if automatic
                else ConsoleAutoRetrieve.NEVER
            ),
            assistant_access=(
                ConsoleAssistantLibraryAccess.ALLOWED
                if allowed
                else ConsoleAssistantLibraryAccess.BLOCKED
            ),
        )

    def _refresh_dirty_state(self) -> None:
        if self._suppress_changes:
            return
        candidate = self._candidate()
        self._dirty = (
            candidate.auto_retrieve is not self._snapshot.auto_retrieve
            or candidate.assistant_access is not self._snapshot.assistant_access
        )
        self.query_one("#library-access-save", Button).disabled = (
            not self._dirty
            or self._operation_pending
            or not self._state.editing_enabled
        )

    @on(RadioSet.Changed)
    def _policy_changed(self, event: RadioSet.Changed) -> None:
        event.stop()
        self._refresh_dirty_state()

    def _set_feedback(
        self,
        copy: str,
        *,
        conflict: bool = False,
        focus: bool = False,
    ) -> None:
        feedback = self.query_one("#library-access-feedback", Static)
        feedback.update(copy)
        feedback.can_focus = focus
        self.query_one("#library-access-reload", Button).display = conflict
        self.query_one("#library-access-compare-retry", Button).display = conflict
        if focus:
            feedback.focus()

    def _set_snapshot_controls(
        self,
        snapshot: ConsoleLibraryPolicySnapshot,
    ) -> None:
        self._suppress_changes = True
        try:
            self.query_one("#library-auto-never", RadioButton).value = (
                snapshot.auto_retrieve is ConsoleAutoRetrieve.NEVER
            )
            self.query_one("#library-auto-automatic", RadioButton).value = (
                snapshot.auto_retrieve is ConsoleAutoRetrieve.AUTOMATIC
            )
            self.query_one("#library-agent-blocked", RadioButton).value = (
                snapshot.assistant_access is ConsoleAssistantLibraryAccess.BLOCKED
            )
            self.query_one("#library-agent-allowed", RadioButton).value = (
                snapshot.assistant_access is ConsoleAssistantLibraryAccess.ALLOWED
            )
        finally:
            self._suppress_changes = False

    async def _save(self) -> None:
        if self._operation_pending or not self._dirty:
            return
        self._operation_pending = True
        save = self.query_one("#library-access-save", Button)
        save.disabled = True
        self._set_feedback("Saving…")
        try:
            outcome = await self._save_policy(self._candidate())
        except Exception:
            self._operation_pending = False
            self._set_feedback(
                "Save failed. Your unsaved choices are still here.",
                focus=True,
            )
            self._refresh_dirty_state()
            return
        finally:
            self._operation_pending = False

        if outcome.status == "saved":
            self._snapshot = outcome.snapshot
            self._dirty = False
            self._set_feedback(outcome.copy)
        elif outcome.status == "conflict":
            self._set_feedback(outcome.copy, conflict=True, focus=True)
        else:
            self._set_feedback(outcome.copy, focus=True)
        self._refresh_dirty_state()

    async def _reload(self, *, preserve_candidate: bool) -> None:
        if self._operation_pending:
            return
        candidate = self._candidate()
        self._operation_pending = True
        self._set_feedback("Reloading…")
        try:
            snapshot = await self._reload_policy()
        except Exception:
            self._set_feedback(
                "Reload failed. Your unsaved choices are still here.",
                conflict=True,
                focus=True,
            )
            return
        finally:
            self._operation_pending = False
        self._snapshot = snapshot
        self._set_snapshot_controls(snapshot)
        if preserve_candidate:
            self._set_snapshot_controls(
                ConsoleLibraryPolicySnapshot(
                    auto_retrieve=candidate.auto_retrieve,
                    assistant_access=candidate.assistant_access,
                    policy_revision=snapshot.policy_revision,
                    source=snapshot.source,
                    error_code=snapshot.error_code,
                )
            )
        self._set_feedback(
            "Reloaded the latest saved policy."
            if not preserve_candidate
            else "Compared with the latest policy; retrying your choices."
        )
        self._refresh_dirty_state()
        if preserve_candidate:
            await self._save()

    async def _perform_safe_cancel(self, *, source: str) -> None:
        del source
        if self._dirty:
            self._set_feedback(
                "Unsaved changes. Save them or choose Discard changes.",
                focus=True,
            )
            self.query_one("#library-access-discard", Button).display = True
            # TASK-25722: with Discard on screen, "Cancel" stops meaning
            # "abandon my edits" and starts meaning "stay here" -- the same
            # word for two different outcomes. Name the outcome instead, so
            # the pair reads Keep editing / Discard changes.
            self.query_one("#library-access-cancel", Button).label = "Keep editing"
            return
        self.dismiss_safe_once(None)

    @on(Button.Pressed)
    async def _button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if not button_id or not button_id.startswith("library-access-"):
            return
        event.stop()
        if button_id == "library-access-save":
            await self._save()
        elif button_id == "library-access-cancel":
            await self.request_safe_cancel(source="cancel")
        elif button_id == "library-access-discard":
            self._dirty = False
            self.dismiss_safe_once(None)
        elif button_id == "library-access-reload":
            await self._reload(preserve_candidate=False)
        elif button_id == "library-access-compare-retry":
            await self._reload(preserve_candidate=True)


__all__ = [
    "ConsoleLibraryAccessModal",
    "ConsoleLibraryPolicySaveOutcome",
]
