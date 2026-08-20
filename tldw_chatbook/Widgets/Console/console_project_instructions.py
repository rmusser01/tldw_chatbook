"""Compact Console project-instruction status and decision surfaces."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Sequence

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Button, Footer, Static

from ...Chat.console_chat_controller import ProjectInstructionDispatchNotice
from ...Chat.console_chat_models import ProjectInstructionPreview
from ...Chat.console_display_state import (
    ConsoleProjectInstructionSourceRow,
    ConsoleProjectInstructionState,
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


class ConsoleProjectInstructionStatusRow(Widget):
    """Compact Inspector authority row that opens the shared Context surface."""

    DEFAULT_CSS = """
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


class ConsoleProjectInstructionContextPanel(Widget):
    """Metadata-only Project Instructions section for Context."""

    DEFAULT_CSS = """
    ConsoleProjectInstructionContextPanel {
        width: 100%;
        height: auto;
        padding: 0 1;
        background: $surface;
    }
    .console-project-instruction-meta { height: auto; color: $text-muted; }
    """

    def __init__(self, state: ConsoleProjectInstructionState, **kwargs) -> None:
        super().__init__(**kwargs)
        self._state = state

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
            yield Static(
                f"{source.relative_source} · scope {source.scope} · "
                f"{source.byte_count} bytes · {source.outcome}{warning}",
                classes="console-project-instruction-meta",
                markup=False,
            )
        for code in state.warning_codes:
            yield Static(
                f"Warning: {code}",
                classes="console-project-instruction-meta",
                markup=False,
            )

    def sync_preview(self, preview: ProjectInstructionPreview | None) -> None:
        """Refresh content-free source metadata from a disposable preview."""
        if preview is None:
            return
        sources = ()
        if preview.relative_source is not None:
            sources = (
                ConsoleProjectInstructionSourceRow(
                    relative_source=preview.relative_source,
                    scope=preview.scope,
                    byte_count=preview.byte_count,
                    outcome=preview.outcomes[0] if preview.outcomes else "active",
                    warning_code=(
                        preview.warning_codes[0] if preview.warning_codes else ""
                    ),
                ),
            )
        status = "Warning" if preview.warning_codes else f"{len(sources)} loaded"
        if not sources and not preview.warning_codes:
            status = "None"
        self._state = replace(
            self._state,
            status=status,
            sources=sources,
            warning_codes=preview.warning_codes,
        )
        self.refresh(recompose=True)

    def sync_state(self, state: ConsoleProjectInstructionState) -> None:
        """Replace the complete content-free state for a modal refresh."""
        self._state = state
        self.refresh(recompose=True)


class ProjectInstructionSetupModal(ModalScreen[ProjectInstructionSetupResult]):
    """Choose one eligible binding, disable the feature, or cancel."""

    DEFAULT_CSS = """
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

    BINDINGS = [
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
        for button in self.query("Button.console-project-binding-option"):
            if not button.disabled:
                button.focus()
                break

    @on(Button.Pressed, ".console-project-binding-option")
    def _select(self, event: Button.Pressed) -> None:
        event.stop()
        index = int(event.button.id.rsplit("-", 1)[-1])
        option = self._options[index]
        if option.eligible:
            self.dismiss(ProjectInstructionSetupResult("select", option.binding_id))

    def action_disable(self) -> None:
        self.dismiss(ProjectInstructionSetupResult("disable"))

    def action_cancel(self) -> None:
        self.dismiss(ProjectInstructionSetupResult("cancel"))

    @on(Button.Pressed, "#console-project-setup-disable")
    def _disable(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_disable()

    @on(Button.Pressed, "#console-project-setup-cancel")
    def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_cancel()


class ProjectInstructionNoticeModal(ModalScreen[str]):
    """First-use disclosure for one session and sanitized provider destination."""

    DEFAULT_CSS = """
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

    BINDINGS = [
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
        self.dismiss("proceed")

    def action_cancel(self) -> None:
        self.dismiss("cancel")

    def action_disable(self) -> None:
        self.dismiss("disable")

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
