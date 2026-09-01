"""Validated create and rename forms for Console Terminal sessions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Select, Static

from tldw_chatbook.Terminal.launch import (
    ShellChoice,
    normalize_session_name,
    resolve_shell_choice,
    resolve_start_directory,
    session_name_key,
)
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin


@dataclass(frozen=True, slots=True)
class TerminalSessionFormResult:
    """Validated local form values returned to the controller."""

    name: str
    shell: str | None
    start_directory: Path | None


def build_default_terminal_name(existing_names: tuple[str, ...]) -> str:
    """Return the first case-insensitively unique ``Terminal N`` name."""
    used = {session_name_key(name) for name in existing_names}
    index = 1
    while session_name_key(f"Terminal {index}") in used:
        index += 1
    return f"Terminal {index}"


class ConsoleTerminalSessionModal(
    SafeModalDismissMixin, ModalScreen[TerminalSessionFormResult | None]
):
    """Collect one name and, for new sessions, allowlisted launch values."""

    SAFE_MODAL_CONTENT = "#console-terminal-session-modal"
    BINDINGS = [("escape", "request_safe_cancel", "Cancel")]

    DEFAULT_CSS = """
    ConsoleTerminalSessionModal {
        align: center middle;
    }

    #console-terminal-session-modal {
        width: 72;
        max-width: 94%;
        height: auto;
        border: round $primary;
        background: $surface;
        padding: 1 2;
    }

    #console-terminal-session-error {
        min-height: 1;
        color: $error;
    }

    #console-terminal-session-actions {
        height: auto;
        margin-top: 1;
        align-horizontal: right;
    }
    """

    def __init__(
        self,
        *,
        mode: Literal["new", "rename"],
        name: str,
        shell_choices: tuple[ShellChoice, ...],
        start_directory: Path,
        existing_names: tuple[str, ...],
    ) -> None:
        super().__init__()
        if mode not in {"new", "rename"}:
            raise ValueError("mode must be 'new' or 'rename'")
        self.mode = mode
        self._initial_name = name
        self._shell_choices = shell_choices
        self._start_directory = Path(start_directory)
        self._existing_names = existing_names
        self.shell_options = tuple(
            (choice.label, choice.key) for choice in shell_choices
        )

    def compose(self) -> ComposeResult:
        title = "New Terminal Session" if self.mode == "new" else "Rename Session"
        with Vertical(id="console-terminal-session-modal"):
            yield Static(title, classes="console-modal-header", markup=False)
            yield Static("Name", classes="console-terminal-session-label")
            yield Input(
                value=self._initial_name,
                max_length=1024,
                id="console-terminal-session-name",
            )
            if self.mode == "new":
                yield Static("Shell", classes="console-terminal-session-label")
                yield Select(
                    self.shell_options,
                    value="default",
                    allow_blank=False,
                    id="console-terminal-session-shell",
                )
                yield Static(
                    "Starting directory", classes="console-terminal-session-label"
                )
                yield Input(
                    value=str(self._start_directory),
                    id="console-terminal-session-directory",
                )
            yield Static("", id="console-terminal-session-error", markup=False)
            with Horizontal(id="console-terminal-session-actions"):
                yield Button("Cancel", id="console-terminal-session-cancel")
                yield Button(
                    "Create" if self.mode == "new" else "Rename",
                    id="console-terminal-session-save",
                    variant="primary",
                )

    @on(Button.Pressed, "#console-terminal-session-cancel")
    async def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="button")

    @on(Button.Pressed, "#console-terminal-session-save")
    def _save(self, event: Button.Pressed) -> None:
        event.stop()
        error = self.query_one("#console-terminal-session-error", Static)
        try:
            name = normalize_session_name(
                self.query_one("#console-terminal-session-name", Input).value,
                existing_names=self._existing_names,
            )
            if self.mode == "rename":
                self.dismiss(TerminalSessionFormResult(name, None, None))
                return

            selector = self.query_one("#console-terminal-session-shell", Select).value
            if not isinstance(selector, str):
                raise ValueError("choose an available shell")
            resolve_shell_choice(selector, self._shell_choices)
            directory = resolve_start_directory(
                None,
                requested_directory=Path(
                    self.query_one("#console-terminal-session-directory", Input).value
                ),
                account_home=self._start_directory,
            )
        except (TypeError, ValueError) as exc:
            error.update(str(exc))
            return
        error.update("")
        self.dismiss(TerminalSessionFormResult(name, selector, directory))
