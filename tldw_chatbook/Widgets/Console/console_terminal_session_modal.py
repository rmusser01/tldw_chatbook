"""Validated create and rename forms for Console Terminal sessions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    model_validator,
)
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


class TerminalSessionFormResult(BaseModel):
    """Strict validated local form values returned to the controller."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str = Field(min_length=1, max_length=1024)
    shell: str | None = Field(default=None, max_length=255)
    start_directory: Path | None = None

    @model_validator(mode="before")
    @classmethod
    def _validate_widget_values(
        cls,
        values: Any,
        info: ValidationInfo,
    ) -> Any:
        context = info.context
        if not context or "mode" not in context:
            return values
        if not isinstance(values, dict):
            raise ValueError("invalid terminal session values")

        name = normalize_session_name(
            values.get("name"),
            existing_names=context["existing_names"],
        )
        if context["mode"] == "rename":
            return {"name": name, "shell": None, "start_directory": None}

        selector = values.get("shell")
        if not isinstance(selector, str):
            raise ValueError("choose an available shell")
        resolve_shell_choice(selector, context["shell_choices"])

        raw_directory = values.get("start_directory")
        if not isinstance(raw_directory, str) or len(raw_directory) > 4096:
            raise ValueError("choose an absolute existing directory")
        directory = resolve_start_directory(
            None,
            requested_directory=Path(raw_directory),
            account_home=context["account_home"],
        )
        return {
            "name": name,
            "shell": selector,
            "start_directory": directory,
        }


def _validation_message(error: ValidationError) -> str:
    errors = error.errors(include_url=False, include_input=False)
    if not errors:
        return "invalid terminal session values"
    return str(errors[0].get("msg", "invalid terminal session values")).removeprefix(
        "Value error, "
    )


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
            result = TerminalSessionFormResult.model_validate(
                {
                    "name": self.query_one(
                        "#console-terminal-session-name", Input
                    ).value,
                    "shell": (
                        self.query_one("#console-terminal-session-shell", Select).value
                        if self.mode == "new"
                        else None
                    ),
                    "start_directory": (
                        self.query_one(
                            "#console-terminal-session-directory", Input
                        ).value
                        if self.mode == "new"
                        else None
                    ),
                },
                context={
                    "mode": self.mode,
                    "existing_names": self._existing_names,
                    "shell_choices": self._shell_choices,
                    "account_home": self._start_directory,
                },
            )
        except ValidationError as exc:
            error.update(_validation_message(exc))
            return
        except (TypeError, ValueError) as exc:
            error.update(str(exc))
            return
        error.update("")
        self.dismiss(result)
