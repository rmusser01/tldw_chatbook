"""Shared scoped controls for future Console exchange capture."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, RadioButton, RadioSet, Static

from tldw_chatbook.Chat.console_chat_controller import (
    CapturePolicyMutationResult,
    CapturePolicyMutationStatus,
    CapturePolicySnapshot,
    CapturePurgeAvailability,
    CapturePurgeResult,
    CapturePurgeStatus,
)
from tldw_chatbook.Chat.console_exchange_capture import (
    CaptureDetail,
    resolve_capture_policy,
)
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin


class CaptureScope(str, Enum):
    """The one future-policy scope changed by an Apply action."""

    NEXT_SEND = "next_send"
    CONVERSATION = "conversation"
    GLOBAL = "global"


@dataclass(frozen=True, slots=True)
class CapturePolicyBindings:
    """Callbacks frozen to the chat that opened a policy surface."""

    target_session_id: str
    target_conversation_id: str | None
    read: Callable[[], CapturePolicySnapshot]
    apply_next: Callable[
        [CaptureDetail | None, int], CapturePolicyMutationResult
    ]
    apply_conversation: Callable[
        [CaptureDetail | None, int], Awaitable[CapturePolicyMutationResult]
    ]
    apply_global: Callable[
        [bool, CaptureDetail, int, int], CapturePolicyMutationResult
    ]
    count_full: Callable[[], Awaitable[int]]
    purge_full: Callable[[int], Awaitable[CapturePurgeResult]]
    capture_revision: Callable[[], int]
    purge_availability: Callable[[], CapturePurgeAvailability]
    apply_next_privacy: Callable[
        [bool | None, bool | None, int], CapturePolicyMutationResult
    ] | None = None
    apply_conversation_privacy: Callable[
        [bool | None, bool | None, int], Awaitable[CapturePolicyMutationResult]
    ] | None = None
    apply_global_privacy: Callable[
        [bool, bool, int, int], CapturePolicyMutationResult
    ] | None = None


@dataclass(frozen=True, slots=True)
class CapturePolicyPreview:
    """Prospective effective state for exactly one scope edit."""

    scope: CaptureScope
    detail: CaptureDetail | None
    effective_detail: CaptureDetail
    enabled: bool
    requires_confirmation: bool


FULL_CAPTURE_WARNING = (
    "Full capture may retain prompts, injected instructions, tool arguments, "
    "outputs, and local paths. Credentials remain structurally blocked, but "
    "ordinary text may still contain secrets."
)
GLOBAL_FULL_CAPTURE_WARNING = (
    f"{FULL_CAPTURE_WARNING} Global Full affects all Console conversations "
    "and survives restart."
)
OFF_TO_ON_WARNING = (
    "Turning capture On resumes the stored detail. The dormant Full setting "
    "will become active for future exchanges."
)
PURGE_FULL_WARNING = "Delete stored Full captures"

_PURGE_REASON_COPY = {
    "target_missing": "Target conversation is no longer available",
    "purge_in_progress": "A Full-capture deletion is already in progress",
    "primary_writer_active": "Assistant response is still writing captures",
    "preparation_active": "A message is still being prepared for capture",
    "fleet_writer_active": "Fleet child is still able to write captures",
    "fleet_state_unavailable": "Fleet writer state is unavailable",
    "exchange_flush_active": "Capture persistence is still in progress",
    "retained_signals_active": "Provider signals can still attach captures",
    "stale_capture_revision": "Stored captures changed; review the count again",
    "persistence_unavailable": "Capture persistence is unavailable",
    "capture_count_unavailable": "Stored Full capture count is unavailable",
}


def _purge_reason(reason_code: str | None) -> str:
    return _PURGE_REASON_COPY.get(reason_code, "Full captures cannot be deleted right now")


def full_capture_confirmation(*, scope_label: str) -> ConfirmationDialog:
    """Build the one shared warning for a Full-effective policy edit."""
    return ConfirmationDialog(
        title=f"Enable Full capture for {scope_label}?",
        message=FULL_CAPTURE_WARNING,
        confirm_label="Enable Full",
        cancel_label="Keep current policy",
    )


class GlobalFullCaptureConfirmation(SafeModalDismissMixin, ModalScreen[bool]):
    """Require an explicit restart-aware acknowledgement for Global Full."""

    SAFE_MODAL_CONTENT = "#global-full-confirmation"
    message = GLOBAL_FULL_CAPTURE_WARNING
    BINDINGS = [Binding("escape", "request_safe_cancel", "Cancel", show=False)]
    #: TASK-22858 follow-on: BUNDLED_CSS, not DEFAULT_CSS — build_css.py
    #: lifts this into the app bundle. A class-level DEFAULT_CSS registers
    #: another stylesheet source against Textual's 64-entry parse cache
    #: (see Tests/UI/test_widget_css_consolidation.py); same fix and same
    #: ModalScreen shape as ProjectSkillsImportModal.
    BUNDLED_CSS = """
    GlobalFullCaptureConfirmation { align: center middle; }
    #global-full-confirmation {
        width: 72; max-width: 96%; height: 18; max-height: 96%;
        border: thick $error; background: $surface; padding: 1 2;
    }
    #global-full-title { text-style: bold; text-align: center; }
    #global-full-body { height: 1fr; scrollbar-gutter: stable; }
    #global-full-message, #global-full-ack { height: auto; margin-top: 1; }
    #global-full-actions { height: 3; min-height: 3; align-horizontal: right; }
    #global-full-actions Button { min-width: 14; margin-left: 1; }
    """

    def compose(self) -> ComposeResult:
        with Container(id="global-full-confirmation"):
            yield Static("Enable Global Full capture?", id="global-full-title")
            with VerticalScroll(id="global-full-body"):
                yield Static(
                    GLOBAL_FULL_CAPTURE_WARNING,
                    id="global-full-message",
                    markup=False,
                )
                yield Checkbox(
                    "I understand this affects every Console conversation and survives restart",
                    id="global-full-ack",
                )
            with Horizontal(id="global-full-actions"):
                yield Button("Keep current policy", id="global-full-cancel")
                yield Button(
                    "Enable Global Full",
                    id="global-full-confirm",
                    variant="error",
                    disabled=True,
                )

    @on(Checkbox.Changed, "#global-full-ack")
    def _acknowledgement_changed(self, event: Checkbox.Changed) -> None:
        self.query_one("#global-full-confirm", Button).disabled = not event.value

    @on(Button.Pressed)
    async def _button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "global-full-confirm" and not event.button.disabled:
            self.dismiss(True)
        elif event.button.id == "global-full-cancel":
            await self.request_safe_cancel(source="button")

    async def _perform_safe_cancel(self, *, source: str) -> None:
        del source
        self.dismiss_safe_once(False)


def global_full_capture_confirmation() -> GlobalFullCaptureConfirmation:
    """Build the shared stronger acknowledgement for Global Full."""
    return GlobalFullCaptureConfirmation()


def off_to_on_confirmation() -> ConfirmationDialog:
    """Build the shared dormant-Full resume warning."""
    return ConfirmationDialog(
        title="Resume Full capture?",
        message=OFF_TO_ON_WARNING,
        confirm_label="Turn capture On",
        cancel_label="Keep capture Off",
    )


__all__ = [
    "CapturePolicyBindings",
    "CapturePolicyPreview",
    "CaptureScope",
    "ConsoleCapturePolicyDialog",
    "ConsoleTracePrivacyDialog",
    "FULL_CAPTURE_WARNING",
    "GLOBAL_FULL_CAPTURE_WARNING",
    "GlobalFullCaptureConfirmation",
    "OFF_TO_ON_WARNING",
    "PURGE_FULL_WARNING",
    "full_capture_confirmation",
    "global_full_capture_confirmation",
    "off_to_on_confirmation",
]


class ConsoleCapturePolicyDialog(SafeModalDismissMixin, ModalScreen[None]):
    """Inspect and change one scoped future-capture policy at a time."""

    SAFE_MODAL_CONTENT = "#capture-policy-dialog"
    BINDINGS = [Binding("escape", "request_safe_cancel", "Cancel", show=False)]

    BUNDLED_SCREEN_CSS = """
    ConsoleCapturePolicyDialog { align: center middle; }
    #capture-policy-dialog {
        width: 76; max-width: 96%; height: 22; max-height: 96%;
        border: tall $primary; background: $surface; padding: 1 2;
    }
    #capture-policy-title { height: 1; text-style: bold; }
    #capture-policy-effective { height: auto; margin-bottom: 1; }
    #capture-policy-body { height: 1fr; scrollbar-gutter: stable; }
    #capture-policy-scopes, #capture-policy-details { height: auto; }
    #capture-policy-reason, #capture-policy-count, #capture-policy-status {
        height: auto;
    }
    #capture-policy-reason { color: $warning; }
    #capture-policy-actions { height: 3; min-height: 3; align-horizontal: right; }
    #capture-policy-actions Button { min-width: 10; margin-left: 1; }
    #capture-policy-status.-error { color: $error; }
    """

    def __init__(self, bindings: CapturePolicyBindings) -> None:
        super().__init__()
        self.bindings = bindings
        self.snapshot = bindings.read()
        self.selected_scope = CaptureScope.CONVERSATION
        self.selected_detail: CaptureDetail | None = self.snapshot.conversation_detail
        self.global_enabled = self.snapshot.enabled
        self.preview = self.preview_for(self.selected_scope, self.selected_detail)
        self.status_text = "Ready"
        self.full_capture_count: int | None = None
        try:
            self.purge_availability = bindings.purge_availability()
        except Exception:
            self.purge_availability = CapturePurgeAvailability(
                False,
                "capture_count_unavailable",
            )
        self._applying = False

    def compose(self) -> ComposeResult:
        with Vertical(id="capture-policy-dialog"):
            yield Static("Exchange Capture Policy", id="capture-policy-title", markup=False)
            yield Static(self._effective_text(), id="capture-policy-effective", markup=False)
            with VerticalScroll(id="capture-policy-body"):
                yield Static(
                    "Apply changes exactly one scope. Inherit removes that scope's override.",
                    id="capture-policy-guidance",
                    markup=False,
                )
                if self.bindings.target_conversation_id is None:
                    yield Static(
                        "Temporary chat: Capture On pauses each send until you choose "
                        "Save & Send or Send without capture.",
                        id="capture-policy-temporary-guidance",
                        markup=False,
                    )
                with RadioSet(id="capture-policy-scopes"):
                    yield RadioButton("Next send", id="capture-policy-scope-next")
                    yield RadioButton(
                        "This conversation",
                        id="capture-policy-scope-conversation",
                        value=True,
                    )
                    yield RadioButton("Global default", id="capture-policy-scope-global")
                with RadioSet(id="capture-policy-details"):
                    yield RadioButton("Inherit", id="capture-policy-detail-inherit")
                    yield RadioButton(
                        "Safe",
                        id="capture-policy-detail-safe",
                        value=self.selected_detail is CaptureDetail.SAFE,
                    )
                    yield RadioButton(
                        "Full",
                        id="capture-policy-detail-full",
                        value=self.selected_detail is CaptureDetail.FULL,
                        disabled=not self.snapshot.enabled,
                    )
                yield Button(
                    "Turn capture Off" if self.snapshot.enabled else "Turn capture On",
                    id="capture-policy-toggle",
                )
                yield Static(self._disabled_reason(), id="capture-policy-reason", markup=False)
                yield Static("Stored Full captures: counting…", id="capture-policy-count", markup=False)
                yield Button(
                    "Delete stored Full captures…",
                    id="capture-policy-purge",
                    disabled=True,
                )
            yield Static(self.status_text, id="capture-policy-status", markup=False)
            with Horizontal(id="capture-policy-actions"):
                yield Button("Cancel", id="capture-policy-cancel")
                yield Button("Apply", id="capture-policy-apply", variant="primary")

    async def on_mount(self) -> None:
        self._sync_scope_guidance()
        self._sync_detail_selection()
        self.query_one("#capture-policy-apply", Button).focus()
        await self.refresh_full_count()

    def preview_for(
        self, scope: CaptureScope, detail: CaptureDetail | None
    ) -> CapturePolicyPreview:
        """Resolve the result without creating a parallel policy owner."""
        snapshot = self.snapshot
        next_detail = snapshot.next_detail
        conversation_detail = snapshot.conversation_detail
        global_detail = snapshot.global_detail
        if scope is CaptureScope.NEXT_SEND:
            next_detail = detail
        elif scope is CaptureScope.CONVERSATION:
            conversation_detail = detail
        elif detail is not None:
            global_detail = detail
        resolution = resolve_capture_policy(
            enabled=self.global_enabled,
            next_send=next_detail,
            conversation=conversation_detail,
            global_default=global_detail,
        )
        revealing_full = (
            resolution.enabled
            and resolution.detail is CaptureDetail.FULL
            and (
                scope is not CaptureScope.NEXT_SEND
                or detail is None
            )
        )
        return CapturePolicyPreview(
            scope,
            detail,
            resolution.detail,
            resolution.enabled,
            revealing_full,
        )

    async def apply(
        self,
        scope: CaptureScope,
        detail: CaptureDetail | None,
        *,
        full_confirmation_done: bool = False,
    ) -> CapturePolicyMutationResult | None:
        """Apply one selected scope after preview and required confirmation."""
        if self._applying:
            return None
        try:
            fresh = self.bindings.read()
        except Exception:
            self._set_status("Failed — target chat is no longer available", error=True)
            return None
        if fresh.session_id != self.bindings.target_session_id:
            self._set_status("Failed — target chat is no longer available", error=True)
            return None
        self.snapshot = fresh
        if not fresh.enabled and detail is CaptureDetail.FULL:
            self._set_status("Failed — Capture Off disables Full edits", error=True)
            return None
        if scope is CaptureScope.GLOBAL and detail is None:
            self._set_status(
                "Failed — Global default requires explicit Safe or Full",
                error=True,
            )
            return None
        self.preview = self.preview_for(scope, detail)
        if (
            self.global_enabled
            and scope is CaptureScope.GLOBAL
            and detail is CaptureDetail.FULL
        ):
            confirmed = bool(
                await self.app.push_screen_wait(global_full_capture_confirmation())
            )
            if not confirmed:
                self._set_status("Full policy change cancelled")
                return None
        elif self.preview.requires_confirmation and not full_confirmation_done:
            confirmed = await self._confirm(
                FULL_CAPTURE_WARNING,
                title=f"Enable Full capture for {scope.value.replace('_', ' ')}?",
                confirm_label="Enable Full",
            )
            if not confirmed:
                self._set_status("Full policy change cancelled")
                return None
        self._applying = True
        self._set_controls_disabled(True)
        self._set_status("Applying")
        try:
            if scope is CaptureScope.NEXT_SEND:
                result = self.bindings.apply_next(detail, fresh.policy_revision)
            elif scope is CaptureScope.CONVERSATION:
                result = await self.bindings.apply_conversation(
                    detail, fresh.policy_revision
                )
            else:
                assert detail is not None
                result = self.bindings.apply_global(
                    self.global_enabled,
                    detail,
                    fresh.config_generation,
                    fresh.policy_revision,
                )
            self._consume_mutation(result)
            return result
        except Exception:
            self._set_status("Failed — capture policy could not be saved", error=True)
            return None
        finally:
            self._applying = False
            self._set_controls_disabled(False)

    def _consume_mutation(self, result: CapturePolicyMutationResult) -> None:
        self.snapshot = result.snapshot
        self.global_enabled = result.snapshot.enabled
        if result.reason_code == "cache_refresh_degraded":
            self._set_status("Saved and active — settings cache refresh degraded")
        elif result.status is CapturePolicyMutationStatus.APPLIED:
            self._set_status("Saved and active")
        elif result.status is CapturePolicyMutationStatus.SAFE_SESSION_ONLY:
            self._set_status("Failed — Safe remains active for this session", error=True)
        elif result.status is CapturePolicyMutationStatus.STALE:
            self._set_status("Failed — policy changed; reopen and try again", error=True)
        elif result.status is CapturePolicyMutationStatus.TARGET_MISSING:
            self._set_status("Failed — target chat is no longer available", error=True)
        else:
            self._set_status("Failed — capture policy could not be saved", error=True)
        if self.is_mounted:
            self.query_one("#capture-policy-effective", Static).update(
                self._effective_text()
            )

    async def set_capture_enabled(self, enabled: bool) -> CapturePolicyMutationResult | None:
        """Apply the global On/Off switch, warning before dormant Full resumes."""
        fresh = self.bindings.read()
        dormant = resolve_capture_policy(
            enabled=True,
            next_send=fresh.next_detail,
            conversation=fresh.conversation_detail,
            global_default=fresh.global_detail,
        )
        warned_for_full = (
            enabled
            and not fresh.enabled
            and dormant.detail is CaptureDetail.FULL
        )
        if warned_for_full:
            if not await self._confirm(
                OFF_TO_ON_WARNING,
                title="Resume Full capture?",
                confirm_label="Turn capture On",
            ):
                return None
        self.global_enabled = enabled
        return await self.apply(
            CaptureScope.GLOBAL,
            fresh.global_detail,
            full_confirmation_done=warned_for_full,
        )

    async def refresh_full_count(self) -> None:
        """Refresh the target chat's logical Full-capture count."""
        try:
            self.full_capture_count = await self.bindings.count_full()
            message = f"Stored Full captures: {self.full_capture_count}"
        except Exception:
            self.full_capture_count = None
            message = "Stored Full captures: unavailable"
        try:
            self.purge_availability = self.bindings.purge_availability()
        except Exception:
            self.purge_availability = CapturePurgeAvailability(
                False,
                "capture_count_unavailable",
            )
        if self.is_mounted:
            self.query_one("#capture-policy-count", Static).update(message)
            self._sync_purge_control()

    async def _fresh_purge_context(
        self,
    ) -> tuple[CapturePolicySnapshot, int, CapturePurgeAvailability, int] | None:
        try:
            snapshot = self.bindings.read()
            if snapshot.session_id != self.bindings.target_session_id:
                return None
            count = await self.bindings.count_full()
            availability = self.bindings.purge_availability()
            revision = self.bindings.capture_revision()
        except Exception:
            return None
        return snapshot, count, availability, revision

    @staticmethod
    def _purge_confirmation_message(
        snapshot: CapturePolicySnapshot,
        count: int,
    ) -> str:
        if snapshot.enabled:
            policy = snapshot.effective.detail.value.title()
        else:
            policy = "Off"
        return (
            f'Delete {count} stored Full captures from “{snapshot.conversation_title}”? '
            "This irreversible action performs logical record deletion only. "
            "SQLite WAL frames, free pages, filesystem snapshots, prior exports, "
            "and backups may retain older bytes; exports and backups are not deleted. "
            "Safe captures, messages, usage, and policy are unchanged. "
            f"The capture policy remains {policy}."
        )

    async def delete_full_captures(self) -> CapturePurgeResult | None:
        """Confirm the bounded logical purge and consume its structured result."""
        if self._applying:
            return None
        before = await self._fresh_purge_context()
        if before is None:
            self._set_status("Failed — stored Full capture state is unavailable", error=True)
            return None
        snapshot, count, availability, revision = before
        if not availability.can_purge:
            self._set_status(f"Failed — {_purge_reason(availability.reason_code)}", error=True)
            return None
        if count <= 0:
            self._set_status("No stored Full captures to delete")
            return None
        if not await self._confirm(
            self._purge_confirmation_message(snapshot, count),
            title="Delete stored Full captures?",
            confirm_label="Delete logical records",
        ):
            self._set_status("Deletion cancelled")
            return None
        after = await self._fresh_purge_context()
        if after is None:
            self._set_status("Failed — stored Full capture state is unavailable", error=True)
            return None
        fresh_snapshot, fresh_count, fresh_availability, fresh_revision = after
        if not fresh_availability.can_purge:
            self._set_status(
                f"Failed — {_purge_reason(fresh_availability.reason_code)}",
                error=True,
            )
            return None
        if (
            fresh_snapshot.conversation_title != snapshot.conversation_title
            or fresh_count != count
            or fresh_snapshot.enabled != snapshot.enabled
            or fresh_snapshot.effective != snapshot.effective
            or fresh_snapshot.policy_revision != snapshot.policy_revision
            or fresh_snapshot.config_generation != snapshot.config_generation
            or fresh_revision != revision
        ):
            self._set_status("Failed — capture state changed; review and confirm again", error=True)
            return None
        self._applying = True
        self._set_controls_disabled(True)
        self._set_status("Applying")
        try:
            result = await self.bindings.purge_full(fresh_revision)
            if result.status is CapturePurgeStatus.DELETED:
                self.full_capture_count = 0
                self._set_status(
                    f"Deleted {result.removed_count} Full capture logical records"
                )
                await self.refresh_full_count()
            elif result.status is CapturePurgeStatus.STALE:
                self._set_status("Failed — stored captures changed; try again", error=True)
            elif result.status is CapturePurgeStatus.BLOCKED:
                self._set_status(
                    f"Failed — {_purge_reason(result.reason_code)}",
                    error=True,
                )
            else:
                self._set_status("Failed — stored captures were not deleted", error=True)
            return result
        except Exception as exc:
            from tldw_chatbook.UI.Console_Modules.capture_policy_bindings import (
                CapturePurgeRefreshError,
            )

            if isinstance(exc, CapturePurgeRefreshError):
                self.full_capture_count = 0
                self._set_status(
                    f"Deleted {exc.result.removed_count} Full capture logical "
                    "records; refresh failed",
                    error=True,
                )
                return exc.result
            self._set_status("Failed — stored captures were not deleted", error=True)
            return None
        finally:
            self._applying = False
            self._set_controls_disabled(False)

    async def _confirm(
        self,
        message: str,
        *,
        title: str,
        confirm_label: str,
    ) -> bool:
        return bool(
            await self.app.push_screen_wait(
                ConfirmationDialog(
                    title=title,
                    message=message,
                    confirm_label=confirm_label,
                    cancel_label="Go back",
                )
            )
        )

    @on(RadioSet.Changed, "#capture-policy-scopes")
    def _scope_changed(self, event: RadioSet.Changed) -> None:
        self.selected_scope = {
            "capture-policy-scope-next": CaptureScope.NEXT_SEND,
            "capture-policy-scope-conversation": CaptureScope.CONVERSATION,
            "capture-policy-scope-global": CaptureScope.GLOBAL,
        }.get(event.pressed.id or "", self.selected_scope)
        if self.selected_scope is CaptureScope.NEXT_SEND:
            self.selected_detail = self.snapshot.next_detail
        elif self.selected_scope is CaptureScope.CONVERSATION:
            self.selected_detail = self.snapshot.conversation_detail
        else:
            self.selected_detail = self.snapshot.global_detail
        self._sync_scope_guidance()
        self._sync_detail_selection()
        self.preview = self.preview_for(self.selected_scope, self.selected_detail)
        self.query_one("#capture-policy-effective", Static).update(
            self._effective_text()
        )

    @on(RadioSet.Changed, "#capture-policy-details")
    def _detail_changed(self, event: RadioSet.Changed) -> None:
        if (
            self.selected_scope is CaptureScope.GLOBAL
            and event.pressed.id == "capture-policy-detail-inherit"
        ):
            self.selected_detail = self.snapshot.global_detail
            self._sync_detail_selection()
            return
        self.selected_detail = {
            "capture-policy-detail-inherit": None,
            "capture-policy-detail-safe": CaptureDetail.SAFE,
            "capture-policy-detail-full": CaptureDetail.FULL,
        }.get(event.pressed.id or "", self.selected_detail)
        self.preview = self.preview_for(self.selected_scope, self.selected_detail)
        self.query_one("#capture-policy-effective", Static).update(
            self._effective_text()
        )

    def _sync_detail_selection(self) -> None:
        inherit = self.query_one("#capture-policy-detail-inherit", RadioButton)
        inherit.disabled = self.selected_scope is CaptureScope.GLOBAL
        for detail, selector in (
            (None, "#capture-policy-detail-inherit"),
            (CaptureDetail.SAFE, "#capture-policy-detail-safe"),
            (CaptureDetail.FULL, "#capture-policy-detail-full"),
        ):
            self.query_one(selector, RadioButton).value = self.selected_detail is detail

    def _sync_scope_guidance(self) -> None:
        if not self.is_mounted:
            return
        text = (
            "Global default requires explicit Safe or Full."
            if self.selected_scope is CaptureScope.GLOBAL
            else "Inherit removes this scope's override."
        )
        self.query_one("#capture-policy-guidance", Static).update(text)

    @on(Button.Pressed, "#capture-policy-apply")
    def _apply_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.run_worker(
            self.apply(self.selected_scope, self.selected_detail),
            group="capture-policy",
            exclusive=True,
        )

    @on(Button.Pressed, "#capture-policy-toggle")
    def _toggle_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.run_worker(
            self.set_capture_enabled(not self.snapshot.enabled),
            group="capture-policy",
            exclusive=True,
        )

    @on(Button.Pressed, "#capture-policy-purge")
    def _purge_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.run_worker(
            self.delete_full_captures(), group="capture-purge", exclusive=True
        )

    @on(Button.Pressed, "#capture-policy-cancel")
    async def _cancel_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="button")

    def _effective_text(self) -> str:
        snapshot = self.snapshot
        if not snapshot.enabled:
            dormant = resolve_capture_policy(
                enabled=True,
                next_send=snapshot.next_detail,
                conversation=snapshot.conversation_detail,
                global_default=snapshot.global_detail,
            )
            text = (
                "Future exchange capture: Off · Dormant "
                f"{dormant.detail.value.title()} ({dormant.source.value.replace('_', ' ')})"
                " · Prospective dormant "
                f"{self.preview.effective_detail.value.title()}"
            )
        else:
            text = (
                f"Future exchange capture: {snapshot.effective.detail.value.title()} · "
                f"Prospective {self.preview.effective_detail.value.title()}"
            )
        if snapshot.active_run_detail is not None:
            text += f"\nActive run frozen at {snapshot.active_run_detail.value.title()}"
        elif snapshot.queued_consumer:
            text += "\nOne queued exchange will consume the current one-shot policy"
        else:
            text += "\nNo active run is frozen"
        return text

    def _disabled_reason(self) -> str:
        if not self.purge_availability.can_purge:
            return _purge_reason(self.purge_availability.reason_code) + "."
        if not self.snapshot.enabled:
            return "Capture Off: Full edits stay disabled until capture is On."
        if self.bindings.target_conversation_id is None:
            return "This chat has no durable conversation target; conversation changes are session-only."
        return ""

    def _set_controls_disabled(self, disabled: bool) -> None:
        if not self.is_mounted:
            return
        for selector in (
            "#capture-policy-apply",
            "#capture-policy-cancel",
            "#capture-policy-toggle",
            "#capture-policy-purge",
            "#capture-policy-scopes",
            "#capture-policy-details",
        ):
            self.query_one(selector).disabled = disabled
        if not disabled:
            self._sync_purge_control()

    def _sync_purge_control(self) -> None:
        if not self.is_mounted:
            return
        purge = self.query_one("#capture-policy-purge", Button)
        purge.disabled = (
            self._applying
            or not self.purge_availability.can_purge
            or not isinstance(self.full_capture_count, int)
            or self.full_capture_count <= 0
        )
        self.query_one("#capture-policy-reason", Static).update(
            self._disabled_reason()
        )

    def _set_status(self, message: str, *, error: bool = False) -> None:
        self.status_text = message
        if self.is_mounted:
            status = self.query_one("#capture-policy-status", Static)
            status.set_class(error, "-error")
            status.update(message)

    async def _perform_safe_cancel(self, *, source: str) -> None:
        del source
        if self._applying:
            self._set_status("Applying")
            return
        self.dismiss_safe_once(None)


class ConsoleTracePrivacyDialog(SafeModalDismissMixin, ModalScreen[None]):
    """Edit future Capture and PII policy without changing viewer disclosure."""

    SAFE_MODAL_CONTENT = "#trace-privacy-dialog"
    BINDINGS = [Binding("escape", "request_safe_cancel", "Cancel", show=False)]
    BUNDLED_CSS = """
    ConsoleTracePrivacyDialog { align: center middle; }
    #trace-privacy-dialog {
        width: 76; max-width: 96%; height: 24; max-height: 96%;
        border: tall $primary; background: $surface; padding: 1 2;
    }
    #trace-privacy-title { height: 1; text-style: bold; }
    #trace-privacy-help, #trace-privacy-effective, #trace-privacy-status {
        height: auto;
    }
    #trace-privacy-body { height: 1fr; scrollbar-gutter: stable; }
    #trace-privacy-actions { height: 3; min-height: 3; align-horizontal: right; }
    #trace-privacy-actions Button { min-width: 10; margin-left: 1; }
    #trace-privacy-status.-error { color: $error; }
    """

    def __init__(self, bindings: CapturePolicyBindings) -> None:
        super().__init__()
        self.bindings = bindings
        self.snapshot = bindings.read()
        self.selected_scope = CaptureScope.CONVERSATION
        self.selected_capture = self.snapshot.conversation_capture_enabled
        self.selected_pii = self.snapshot.conversation_pii_redaction_enabled
        self.status_text = "Ready"
        self._applying = False

    def compose(self) -> ComposeResult:
        with Vertical(id="trace-privacy-dialog"):
            yield Static("Trace Privacy", id="trace-privacy-title", markup=False)
            yield Static(self._effective_text(), id="trace-privacy-effective", markup=False)
            with VerticalScroll(id="trace-privacy-body"):
                yield Static(
                    "Capture and PII masking are independent. PII masking is "
                    "irreversible for provider traces and does not alter the saved "
                    "conversation. Safe/Full is chosen in the trace viewer.",
                    id="trace-privacy-help",
                    markup=False,
                )
                with RadioSet(id="trace-privacy-scopes"):
                    yield RadioButton("Next send", id="trace-privacy-scope-next")
                    yield RadioButton(
                        "This conversation",
                        id="trace-privacy-scope-conversation",
                        value=True,
                    )
                    yield RadioButton("Global default", id="trace-privacy-scope-global")
                yield Static("Capture", markup=False)
                with RadioSet(id="trace-privacy-capture"):
                    yield RadioButton("Inherit", id="trace-privacy-capture-inherit")
                    yield RadioButton("On", id="trace-privacy-capture-on")
                    yield RadioButton("Off", id="trace-privacy-capture-off")
                yield Static("PII masking", markup=False)
                with RadioSet(id="trace-privacy-pii"):
                    yield RadioButton("Inherit", id="trace-privacy-pii-inherit")
                    yield RadioButton("On", id="trace-privacy-pii-on")
                    yield RadioButton("Off", id="trace-privacy-pii-off")
            yield Static(self.status_text, id="trace-privacy-status", markup=False)
            with Horizontal(id="trace-privacy-actions"):
                yield Button("Cancel", id="trace-privacy-cancel")
                yield Button("Apply", id="trace-privacy-apply", variant="primary")

    def on_mount(self) -> None:
        self._sync_choices()
        self.query_one("#trace-privacy-apply", Button).focus()

    @on(RadioSet.Changed, "#trace-privacy-scopes")
    def _privacy_scope_changed(self, event: RadioSet.Changed) -> None:
        self.selected_scope = {
            "trace-privacy-scope-next": CaptureScope.NEXT_SEND,
            "trace-privacy-scope-conversation": CaptureScope.CONVERSATION,
            "trace-privacy-scope-global": CaptureScope.GLOBAL,
        }.get(event.pressed.id or "", self.selected_scope)
        if self.selected_scope is CaptureScope.NEXT_SEND:
            self.selected_capture = self.snapshot.next_capture_enabled
            self.selected_pii = self.snapshot.next_pii_redaction_enabled
        elif self.selected_scope is CaptureScope.CONVERSATION:
            self.selected_capture = self.snapshot.conversation_capture_enabled
            self.selected_pii = self.snapshot.conversation_pii_redaction_enabled
        else:
            self.selected_capture = self.snapshot.enabled
            self.selected_pii = self.snapshot.global_pii_redaction_enabled
        self._sync_choices()

    @on(RadioSet.Changed, "#trace-privacy-capture")
    def _capture_changed(self, event: RadioSet.Changed) -> None:
        selected = {
            "trace-privacy-capture-inherit": None,
            "trace-privacy-capture-on": True,
            "trace-privacy-capture-off": False,
        }
        self.selected_capture = selected.get(
            event.pressed.id or "", self.selected_capture
        )

    @on(RadioSet.Changed, "#trace-privacy-pii")
    def _pii_changed(self, event: RadioSet.Changed) -> None:
        selected = {
            "trace-privacy-pii-inherit": None,
            "trace-privacy-pii-on": True,
            "trace-privacy-pii-off": False,
        }
        self.selected_pii = selected.get(event.pressed.id or "", self.selected_pii)

    def _sync_choices(self) -> None:
        global_scope = self.selected_scope is CaptureScope.GLOBAL
        for prefix, value in (
            ("#trace-privacy-capture", self.selected_capture),
            ("#trace-privacy-pii", self.selected_pii),
        ):
            inherit = self.query_one(f"{prefix}-inherit", RadioButton)
            inherit.disabled = global_scope
            inherit.value = value is None and not global_scope
            self.query_one(f"{prefix}-on", RadioButton).value = value is True
            self.query_one(f"{prefix}-off", RadioButton).value = value is False

    async def apply_privacy(self) -> CapturePolicyMutationResult | None:
        """Apply exactly one sparse scope through frozen production bindings."""

        if self._applying:
            return None
        fresh = self.bindings.read()
        if fresh.session_id != self.bindings.target_session_id:
            self._set_privacy_status("Failed — target chat is unavailable", error=True)
            return None
        capture = self.selected_capture
        pii = self.selected_pii
        if self.selected_scope is CaptureScope.GLOBAL and (
            capture is None or pii is None
        ):
            self._set_privacy_status(
                "Failed — global defaults require explicit On or Off", error=True
            )
            return None
        self._applying = True
        self._set_privacy_disabled(True)
        try:
            if self.selected_scope is CaptureScope.NEXT_SEND:
                callback = self.bindings.apply_next_privacy
                if callback is None:
                    raise RuntimeError("next-send privacy unavailable")
                result = callback(capture, pii, fresh.policy_revision)
            elif self.selected_scope is CaptureScope.CONVERSATION:
                callback = self.bindings.apply_conversation_privacy
                if callback is None:
                    raise RuntimeError("conversation privacy unavailable")
                result = await callback(capture, pii, fresh.policy_revision)
            else:
                callback = self.bindings.apply_global_privacy
                if callback is None:
                    raise RuntimeError("global privacy unavailable")
                assert capture is not None and pii is not None
                result = callback(
                    capture,
                    pii,
                    fresh.config_generation,
                    fresh.policy_revision,
                )
            self.snapshot = result.snapshot
            if result.status is CapturePolicyMutationStatus.APPLIED:
                self._set_privacy_status("Saved and active")
            elif result.status is CapturePolicyMutationStatus.SAFE_SESSION_ONLY:
                self._set_privacy_status(
                    "Save failed — safer session policy remains active", error=True
                )
            else:
                self._set_privacy_status("Failed — policy changed", error=True)
            if self.is_mounted:
                self.query_one("#trace-privacy-effective", Static).update(
                    self._effective_text()
                )
            return result
        except Exception:
            self._set_privacy_status("Failed — trace privacy could not be saved", error=True)
            return None
        finally:
            self._applying = False
            self._set_privacy_disabled(False)

    @on(Button.Pressed, "#trace-privacy-apply")
    def _privacy_apply_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.run_worker(self.apply_privacy(), group="trace-privacy", exclusive=True)

    @on(Button.Pressed, "#trace-privacy-cancel")
    async def _privacy_cancel_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="button")

    def _effective_text(self) -> str:
        effective_capture = self.snapshot.effective_capture_enabled
        if effective_capture is None:
            effective_capture = self.snapshot.enabled
        capture = "On" if effective_capture else "Off"
        pii = "On" if self.snapshot.pii_redaction_enabled else "Off"
        return f"Future trace: Capture {capture} · PII masking {pii}"

    def _set_privacy_disabled(self, disabled: bool) -> None:
        if not self.is_mounted:
            return
        for selector in (
            "#trace-privacy-scopes",
            "#trace-privacy-capture",
            "#trace-privacy-pii",
            "#trace-privacy-apply",
            "#trace-privacy-cancel",
        ):
            self.query_one(selector).disabled = disabled

    def _set_privacy_status(self, message: str, *, error: bool = False) -> None:
        self.status_text = message
        if self.is_mounted:
            status = self.query_one("#trace-privacy-status", Static)
            status.set_class(error, "-error")
            status.update(message)

    async def _perform_safe_cancel(self, *, source: str) -> None:
        del source
        if not self._applying:
            self.dismiss_safe_once(None)
