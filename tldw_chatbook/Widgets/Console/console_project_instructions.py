"""Compact Console project-instruction status and decision surfaces."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import copy
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Literal, Sequence

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Button, Footer, Static
from textual.css.query import QueryError

from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin

from ...Chat.console_chat_controller import (
    ProjectInstructionBindingRecovery,
    ProjectInstructionDispatchNotice,
    commit_project_instruction_setup_decision,
    list_project_instruction_bindings,
)
from ...Chat.console_chat_models import ProjectInstructionPreview
from ...Chat.console_display_state import (
    ConsoleProjectInstructionSourceRow,
    ConsoleProjectInstructionState,
    merge_console_project_instruction_activations,
)


@dataclass(frozen=True, slots=True)
class ProjectInstructionBindingOption:
    """One content-free workspace binding choice."""

    binding_id: str
    label: str
    eligible: bool
    recovery: str = ""


@dataclass(frozen=True, slots=True)
class ProjectInstructionSetupResult:
    """One explicit setup decision."""

    action: Literal["select", "disable", "cancel"]
    binding_id: str | None = None


async def recover_console_project_instruction_session(
    session_id: str | None,
    action: str,
    *,
    store: Any,
    registry: Any,
    select_binding: Callable[
        [str, tuple[Any, ...], str], Awaitable[tuple[str, str | None]]
    ],
    refresh_state: Callable[[str], Awaitable[ConsoleProjectInstructionState]],
    clear_delivery: Callable[[str], None],
    activation_controller: Any = None,
) -> ConsoleProjectInstructionState | None:
    """Apply one race-safe recovery decision to the captured session.

    Args:
        session_id: Session captured when the recovery surface opened.
        action: Requested ``enable``, ``choose``, or ``disable`` action.
        store: Console session store mutated only after authority revalidation.
        registry: Workspace binding registry used to list and revalidate choices.
        select_binding: Async chooser for eligible binding options.
        refresh_state: Async content-free display-state refresher.
        clear_delivery: Callback that clears prior run activation metadata.
        activation_controller: Optional source of current activation events.

    Returns:
        Refreshed content-free UI state, or ``None`` when the captured session
        or requested action is no longer valid.
    """
    if session_id is None or action not in {"enable", "choose", "disable"}:
        return None
    session = next((item for item in store.sessions() if item.id == session_id), None)
    if session is None:
        return None
    expected_state = session.project_instruction_state
    options = ()
    decision = action
    binding_id = None
    if action != "disable":
        recovery_code = "no_eligible_binding"
        try:
            options = await asyncio.to_thread(
                list_project_instruction_bindings, session, registry
            )
        except ProjectInstructionBindingRecovery as exc:
            recovery_code = str(exc) or "binding_unavailable"
        decision, binding_id = await select_binding(session_id, options, recovery_code)

    def validate_decision():
        session_snapshot = copy.copy(session)
        pending_state = []

        class StoreSnapshot:
            def sessions(self):
                return (session_snapshot,)

            def set_session_project_instruction_state(self, _session_id, state):
                pending_state.append(state)
                session_snapshot.project_instruction_state = state

        outcome, selection = commit_project_instruction_setup_decision(
            store=StoreSnapshot(),
            session_id=session_id,
            registry=registry,
            expected_state=expected_state,
            expected_options=options,
            action=decision,
            binding_id=binding_id,
        )
        return outcome, selection, pending_state[0] if pending_state else None

    outcome, _selection, pending_state = await asyncio.to_thread(validate_decision)
    current = next((item for item in store.sessions() if item.id == session_id), None)
    if current is None or current.project_instruction_state != expected_state:
        outcome = "cancel"
    elif pending_state is not None:
        store.set_session_project_instruction_state(session_id, pending_state)
    if outcome != "cancel":
        clear_delivery(session_id)
    state = await refresh_state(session_id)
    return project_instruction_ui_state(
        state,
        store=store,
        controller=activation_controller,
        session_id=session_id,
    )


def project_instruction_ui_state(
    state: ConsoleProjectInstructionState,
    *,
    store: Any,
    controller: Any,
    session_id: str | None = None,
) -> ConsoleProjectInstructionState:
    """Add one captured session's content-free activation events to UI state.

    Args:
        state: Base authority and source display state.
        store: Console session store used to resolve the captured session.
        controller: Optional activation-event source.
        session_id: Explicit captured session, or the active session when absent.

    Returns:
        ``state`` enriched with matching content-free activation metadata.
    """
    target_id = session_id or store.active_session_id
    events = (
        controller.project_instruction_activation_events(target_id)
        if controller is not None and target_id is not None
        else ()
    )
    if not events or target_id is None:
        return state
    session = next((item for item in store.sessions() if item.id == target_id), None)
    if session is None:
        return state
    return merge_console_project_instruction_activations(
        state, session.project_instruction_state, events
    )


def sync_project_instruction_status_row(screen: Any, state: Any) -> None:
    """Refresh the mounted project-instruction rail row.

    Args:
        screen: Mounted Console screen containing the optional status row.
        state: Content-free state to publish into that row.
    """
    try:
        row = screen.query_one(
            "#console-project-instruction-status", ConsoleProjectInstructionStatusRow
        )
    except QueryError:
        return
    row.sync_state(state)


def project_instruction_context_kwargs(
    screen: Any, controller: Any, session_id: str
) -> dict[str, Any]:
    """Build captured-session project state and recovery arguments for Context.

    Args:
        screen: Console screen that owns the session UI controller.
        controller: Console chat controller and session store owner.
        session_id: Session captured when Context was opened.

    Returns:
        Keyword arguments for the shared Context modal.
    """
    session_controller = screen._session

    def project_state(state):
        return project_instruction_ui_state(
            state,
            store=controller.store,
            controller=controller,
            session_id=session_id,
        )

    async def state_factory():
        state = await session_controller._refresh_console_project_instruction_display_state(
            session_id
        )
        return project_state(state)

    return {
        "project_instruction_state": project_state(
            session_controller._build_console_project_instruction_display_state(session_id)
        ),
        "project_instruction_state_factory": state_factory,
        "project_instruction_session_id": session_id,
        "project_instruction_recovery": partial(
            recover_console_project_instruction_session,
            store=controller.store,
            registry=getattr(screen.app_instance, "workspace_registry_service", None),
            select_binding=session_controller._select_project_instruction_binding,
            refresh_state=session_controller._refresh_console_project_instruction_display_state,
            clear_delivery=controller._clear_project_instruction_delivery,
            activation_controller=controller,
        ),
    }


def project_instruction_ui_state_for_screen(
    screen: Any, state: ConsoleProjectInstructionState | None = None
) -> ConsoleProjectInstructionState:
    """Build active-session project-instruction UI state for a screen.

    Args:
        screen: Console screen owning the store and controller.
        state: Optional prebuilt base display state.

    Returns:
        Content-free state enriched with active-session activation metadata.
    """
    if state is None:
        state = screen._session._build_console_project_instruction_display_state()
    return project_instruction_ui_state(
        state,
        store=screen._ensure_console_chat_store(),
        controller=getattr(screen, "_console_chat_controller", None),
    )


def sync_project_instruction_status_for_screen(screen: Any) -> None:
    """Publish active-session activation state into the mounted rail row.

    Args:
        screen: Mounted Console screen whose project status should refresh.
    """
    sync_project_instruction_status_row(
        screen, project_instruction_ui_state_for_screen(screen)
    )


class ConsoleProjectInstructionStatusRow(Widget):
    """Compact Inspector authority row that opens the shared Context surface."""

    BUNDLED_CSS = """
    ConsoleProjectInstructionStatusRow {
        width: 100%;
        height: 1;
        min-height: 1;
    }
    #console-project-instruction-status-button {
        width: 100%;
        height: 1;
        min-height: 1;
        border: none;
        padding: 0 1;
        background: $surface;
        color: $text;
    }
    #console-project-instruction-status-button:focus {
        outline: heavy $accent;
    }
    """

    def __init__(self, state: ConsoleProjectInstructionState, **kwargs) -> None:
        super().__init__(id="console-project-instruction-status", **kwargs)
        self._state = state

    def compose(self) -> ComposeResult:
        yield Button(
            f"{self._state.status} · Project",
            id="console-project-instruction-status-button",
            # `console-rail-focus-carrier` keys the focus-edge rule
            # (TASK-31663); see `console_inspector_section.py`'s toggle.
            classes="console-rail-focus-carrier",
            compact=True,
        )

    def sync_state(self, state: ConsoleProjectInstructionState) -> None:
        """Refresh controlled status copy without recomposing the rail."""
        self._state = state
        self.query_one(
            "#console-project-instruction-status-button", Button
        ).label = f"{state.status} · Project"

    @on(Button.Pressed, "#console-project-instruction-status-button")
    def _open_context(self, event: Button.Pressed) -> None:
        event.stop()
        self.screen.action_view_chat_context()


class ConsoleProjectInstructionContextPanel(VerticalScroll):
    """Metadata-only Project Instructions section for Context."""

    BUNDLED_CSS = """
    ConsoleProjectInstructionContextPanel {
        width: 100%;
        height: auto;
        max-height: 9;
        padding: 0 1;
        background: $surface;
    }
    .console-project-instruction-meta { height: auto; color: $text-muted; }
    #console-project-instruction-recovery-actions { height: 1; margin-top: 1; }
    .console-project-instruction-recovery-action {
        width: auto;
        min-width: 10;
        height: 1;
        min-height: 1;
        border: none;
        padding: 0 1;
    }
    """

    class RecoveryRequested(Message):
        """Request one captured session's relevant recovery action."""

        def __init__(self, session_id: str | None, action: str) -> None:
            super().__init__()
            self.session_id = session_id
            self.action = action

    def __init__(
        self,
        state: ConsoleProjectInstructionState,
        *,
        session_id: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._state = state
        self._session_id = session_id

    def compose(self) -> ComposeResult:
        state = self._state
        yield Static("Project Instructions", markup=False)
        yield Static(
            f"State: {state.status} · Enabled: {'yes' if state.enabled else 'no'}",
            classes="console-project-instruction-meta",
            markup=False,
        )
        yield Static(
            f"Binding: {state.binding_label or 'not selected'} · Locator: {state.locator_match}",
            classes="console-project-instruction-meta",
            markup=False,
        )
        for source in state.sources:
            warning = f" · warning {source.warning_code}" if source.warning_code else ""
            byte_copy = (
                f" · {source.byte_count} bytes"
                if source.byte_count is not None
                else ""
            )
            basename = source.relative_source.rsplit("/", 1)[-1]
            precedence = (
                "override" if basename == "AGENTS.override.md" else "standard"
            )
            yield Static(
                f"{source.relative_source} · Precedence: {precedence} · "
                f"scope {source.scope}{byte_copy} · {source.outcome}{warning}",
                classes="console-project-instruction-meta",
                markup=False,
            )
        for code in state.warning_codes:
            yield Static(
                f"Warning: {code}",
                classes="console-project-instruction-meta",
                markup=False,
            )
        if state.recovery_actions:
            labels = {
                "enable": "Enable",
                "choose": "Choose folder",
                "disable": "Disable",
            }
            with Horizontal(id="console-project-instruction-recovery-actions"):
                for action in state.recovery_actions:
                    yield Button(
                        labels[action],
                        id=f"console-project-instruction-{action}",
                        classes="console-project-instruction-recovery-action",
                        compact=True,
                    )

    @on(Button.Pressed, ".console-project-instruction-recovery-action")
    def _request_recovery(self, event: Button.Pressed) -> None:
        event.stop()
        action = event.button.id.rsplit("-", 1)[-1]
        self.post_message(self.RecoveryRequested(self._session_id, action))

    def sync_preview(self, preview: ProjectInstructionPreview | None) -> None:
        """Refresh content-free source metadata from a disposable preview."""
        if preview is None:
            return
        sources_by_scope = {
            (source.relative_source, source.scope): source
            for source in self._state.sources
        }
        if preview.relative_source is not None:
            source = ConsoleProjectInstructionSourceRow(
                relative_source=preview.relative_source,
                scope=preview.scope,
                byte_count=preview.byte_count,
                outcome=preview.outcomes[0] if preview.outcomes else "active",
                warning_code=(
                    preview.warning_codes[0] if preview.warning_codes else ""
                ),
            )
            sources_by_scope[(source.relative_source, source.scope)] = source
        sources = tuple(sources_by_scope.values())
        warning_codes = tuple(
            dict.fromkeys((*self._state.warning_codes, *preview.warning_codes))
        )
        status = (
            "Warning"
            if warning_codes
            else f"{sum(source.outcome == 'active' for source in sources)} loaded"
        )
        if not sources and not preview.warning_codes:
            status = "None"
        self._state = replace(
            self._state,
            status=status,
            sources=sources,
            warning_codes=warning_codes,
            recovery_actions=("choose", "disable") if status == "Warning" else (),
        )
        self.refresh(recompose=True)

    def sync_state(self, state: ConsoleProjectInstructionState) -> None:
        """Replace the complete content-free state for a modal refresh."""
        if state == self._state:
            return
        self._state = state
        self.refresh(recompose=True)


class ProjectInstructionSetupModal(
    SafeModalDismissMixin, ModalScreen[ProjectInstructionSetupResult]
):
    """Choose one eligible binding, disable the feature, or cancel."""

    BUNDLED_CSS = """
    ProjectInstructionSetupModal { align: center middle; }
    #console-project-setup-modal {
        width: 90%;
        max-width: 76;
        height: auto;
        max-height: 80%;
        border: tall $accent;
        background: $panel;
        padding: 1 2;
    }
    #console-project-binding-list { height: auto; max-height: 14; }
    .console-project-binding-option { width: 100%; height: 1; min-height: 1; }
    .console-project-binding-recovery { height: auto; color: $text-muted; }
    #console-project-setup-actions { height: 1; margin-top: 1; }
    """

    SAFE_MODAL_CONTENT = "#console-project-setup-modal"
    BINDINGS = [
        ("escape", "request_safe_cancel", "Cancel"),
        ("d", "disable", "Disable"),
        ("c", "cancel", "Cancel"),
    ]
    AUTO_FOCUS = None

    def __init__(self, options: Sequence[ProjectInstructionBindingOption]) -> None:
        super().__init__()
        self._options = tuple(options)

    def compose(self) -> ComposeResult:
        with Vertical(id="console-project-setup-modal"):
            yield Static("Project instructions need a folder", markup=False)
            yield Static(
                "Select one authorized folder for this session. Stale folders cannot be selected.",
                markup=False,
            )
            with VerticalScroll(id="console-project-binding-list"):
                for index, option in enumerate(self._options):
                    yield Button(
                        Text(option.label),
                        id=f"console-project-binding-{index}",
                        classes="console-project-binding-option",
                        compact=True,
                        disabled=not option.eligible,
                    )
                    if option.recovery:
                        yield Static(
                            option.recovery,
                            classes="console-project-binding-recovery",
                            markup=False,
                        )
            with Horizontal(id="console-project-setup-actions"):
                yield Button(
                    "Disable", id="console-project-setup-disable", compact=True
                )
                yield Button("Cancel", id="console-project-setup-cancel", compact=True)
            yield Footer()

    def on_mount(self) -> None:
        super().on_mount()
        for button in self.query("Button.console-project-binding-option"):
            if not button.disabled:
                button.focus()
                return
        self.query_one("#console-project-setup-disable", Button).focus()

    @on(Button.Pressed, ".console-project-binding-option")
    def _select(self, event: Button.Pressed) -> None:
        event.stop()
        index = int(event.button.id.rsplit("-", 1)[-1])
        option = self._options[index]
        if option.eligible:
            self.dismiss_safe_once(
                ProjectInstructionSetupResult("select", option.binding_id)
            )

    def action_disable(self) -> None:
        self.dismiss_safe_once(ProjectInstructionSetupResult("disable"))

    def action_cancel(self) -> None:
        self.dismiss_safe_once(ProjectInstructionSetupResult("cancel"))

    async def _perform_safe_cancel(self, *, source: str) -> None:
        del source
        self.dismiss_safe_once(ProjectInstructionSetupResult("cancel"))

    @on(Button.Pressed, "#console-project-setup-disable")
    def _disable(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_disable()

    @on(Button.Pressed, "#console-project-setup-cancel")
    def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_cancel()


class ProjectInstructionNoticeModal(SafeModalDismissMixin, ModalScreen[str]):
    """First-use disclosure for one session and sanitized provider destination."""

    BUNDLED_CSS = """
    ProjectInstructionNoticeModal { align: center middle; }
    #console-project-notice-modal {
        width: 90%;
        max-width: 76;
        height: auto;
        max-height: 80%;
        border: tall $warning;
        background: $panel;
        padding: 1 2;
    }
    .console-project-notice-copy { height: auto; }
    #console-project-notice-actions { height: 1; margin-top: 1; }
    """

    SAFE_MODAL_CONTENT = "#console-project-notice-modal"
    BINDINGS = [
        ("escape", "request_safe_cancel", "Cancel"),
        ("p", "proceed", "Proceed"),
        ("c", "cancel", "Cancel"),
        ("d", "disable", "Disable"),
    ]
    AUTO_FOCUS = "#console-project-notice-proceed"

    def __init__(self, notice: ProjectInstructionDispatchNotice) -> None:
        super().__init__()
        self.notice = notice

    def compose(self) -> ComposeResult:
        source = self.notice.relative_source or "No root file found"
        with Vertical(id="console-project-notice-modal"):
            yield Static("Send project instructions?", markup=False)
            yield Static(
                f"Session: {self.notice.session_id}",
                classes="console-project-notice-copy",
                markup=False,
            )
            yield Static(
                f"Destination: {self.notice.destination_label}",
                classes="console-project-notice-copy",
                markup=False,
            )
            yield Static(
                f"Root source: {source} · scope {self.notice.scope} · "
                f"{self.notice.byte_count} bytes",
                classes="console-project-notice-copy",
                markup=False,
            )
            yield Static(
                "Repository text is untrusted project guidance. Its root content "
                "will be sent now; deeper AGENTS.md files may be sent later when "
                "local tools enter their scopes.",
                classes="console-project-notice-copy",
                markup=False,
            )
            if self.notice.outcomes:
                yield Static(
                    f"Outcomes: {', '.join(self.notice.outcomes)}",
                    classes="console-project-notice-copy",
                    markup=False,
                )
            if self.notice.warning_codes:
                yield Static(
                    f"Warnings: {', '.join(self.notice.warning_codes)}",
                    classes="console-project-notice-copy",
                    markup=False,
                )
            with Horizontal(id="console-project-notice-actions"):
                yield Button(
                    "Proceed", id="console-project-notice-proceed", compact=True
                )
                yield Button("Cancel", id="console-project-notice-cancel", compact=True)
                yield Button(
                    "Disable", id="console-project-notice-disable", compact=True
                )
            yield Footer()

    def action_proceed(self) -> None:
        self.dismiss_safe_once("proceed")

    def action_cancel(self) -> None:
        self.dismiss_safe_once("cancel")

    def action_disable(self) -> None:
        self.dismiss_safe_once("disable")

    async def _perform_safe_cancel(self, *, source: str) -> None:
        del source
        self.dismiss_safe_once("cancel")

    @on(Button.Pressed, "#console-project-notice-proceed")
    def _proceed(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_proceed()

    @on(Button.Pressed, "#console-project-notice-cancel")
    def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_cancel()

    @on(Button.Pressed, "#console-project-notice-disable")
    def _disable(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_disable()
