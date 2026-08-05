"""Compose a ``/generate-image`` command from backend/style/prompt fields.

task-1681: the menu's "Generate Image" entry opens this modal, which
builds the command text and hands it back for the SCREEN to paste into the
composer. It deliberately does not generate anything: the user reviews and
edits the command in the composer, then sends it, so the existing
``/generate-image`` handler stays the single execution path.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Select, Static

from tldw_chatbook.Chat.console_command_grammar import (
    COMMAND_PREFIX,
    GENERATE_IMAGE_COMMAND_NAME,
)

#: Select sentinel for "use whatever the config default is".
DEFAULT_CHOICE = "__default__"


def build_generate_image_command(
    *, prompt: str, backend: str | None = None, style: str | None = None
) -> str:
    """Build the ``/generate-image`` command text.

    Token order follows the grammar in ``console_generate_image``: optional
    ``:backend`` and ``@style`` lead, then the prompt. Blank/default
    selections are omitted so the command stays as short as what the user
    actually chose.

    Args:
        prompt: The image prompt; surrounding whitespace is trimmed.
        backend: Optional backend id for a ``:backend`` token.
        style: Optional style/template id for an ``@style`` token.

    Returns:
        The full command line, e.g. ``/generate-image :swarmui @anime a fox``.
    """
    parts = [f"{COMMAND_PREFIX}{GENERATE_IMAGE_COMMAND_NAME}"]
    if backend and backend != DEFAULT_CHOICE:
        parts.append(f":{backend}")
    if style and style != DEFAULT_CHOICE:
        parts.append(f"@{style}")
    text = (prompt or "").strip()
    if text:
        parts.append(text)
    return " ".join(parts)


class ConsoleGenerateImageModal(ModalScreen["str | None"]):
    """Collect backend/style/prompt; dismiss with the command text."""

    DEFAULT_CSS = """
    ConsoleGenerateImageModal {
        align: center middle;
    }

    #console-generate-image-modal {
        width: 70;
        height: auto;
        border: tall gray;
        background: black;
        padding: 1 2;
    }

    .console-generate-image-label {
        color: $text-muted;
        margin: 1 0 0 0;
    }

    #console-generate-image-preview {
        color: $text-muted;
        margin: 1 0 0 0;
        /* cubic PR #1160: a long pasted prompt grew the preview until the
           Paste/Cancel buttons were pushed off-screen. */
        height: auto;
        max-height: 4;
        overflow-y: auto;
    }

    #console-generate-image-actions {
        height: 3;
        margin: 1 0 0 0;
    }
    """

    BINDINGS = [("escape", "dismiss_modal", "Cancel")]

    def __init__(
        self,
        *,
        backends: Sequence[str] = (),
        styles: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the modal.

        Args:
            backends: Selectable backend ids; the caller reads these from
                config so this widget stays I/O free.
            styles: Mapping of style id -> display name.
            **kwargs: Forwarded to ``ModalScreen``.
        """
        super().__init__(**kwargs)
        self._backends = tuple(backends)
        self._styles = dict(styles or {})

    def compose(self) -> ComposeResult:
        with Vertical(id="console-generate-image-modal"):
            yield Static("Generate image", classes="console-modal-header")
            yield Static("Prompt", classes="console-generate-image-label")
            yield Input(
                placeholder="what to draw…", id="console-generate-image-prompt"
            )
            yield Static("Backend", classes="console-generate-image-label")
            yield Select(
                [("Default (config)", DEFAULT_CHOICE)]
                + [(b, b) for b in self._backends],
                value=DEFAULT_CHOICE,
                id="console-generate-image-backend",
            )
            yield Static("Style", classes="console-generate-image-label")
            yield Select(
                [("None", DEFAULT_CHOICE)]
                + [(name, sid) for sid, name in sorted(self._styles.items())],
                value=DEFAULT_CHOICE,
                id="console-generate-image-style",
            )
            yield Static(
                self._preview_text(""),
                id="console-generate-image-preview",
                markup=False,
            )
            with Horizontal(id="console-generate-image-actions"):
                yield Button(
                    "Paste command",
                    id="console-generate-image-accept",
                    variant="primary",
                )
                yield Button("Cancel", id="console-generate-image-cancel")

    def on_mount(self) -> None:
        self.query_one("#console-generate-image-prompt", Input).focus()

    def _current_command(self) -> str:
        prompt = self.query_one("#console-generate-image-prompt", Input).value
        backend = self.query_one("#console-generate-image-backend", Select).value
        style = self.query_one("#console-generate-image-style", Select).value
        return build_generate_image_command(
            prompt=prompt,
            backend=backend if isinstance(backend, str) else None,
            style=style if isinstance(style, str) else None,
        )

    @staticmethod
    def _preview_text(command: str) -> str:
        return f"Command: {command}" if command else "Command: (enter a prompt)"

    def _refresh_preview(self) -> None:
        try:
            self.query_one("#console-generate-image-preview", Static).update(
                self._preview_text(self._current_command())
            )
        except Exception:
            pass

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "console-generate-image-prompt":
            event.stop()
            self._refresh_preview()

    def on_select_changed(self, event: Select.Changed) -> None:
        event.stop()
        self._refresh_preview()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "console-generate-image-prompt":
            event.stop()
            self._accept()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "console-generate-image-accept":
            event.stop()
            self._accept()
        elif event.button.id == "console-generate-image-cancel":
            event.stop()
            self.dismiss(None)

    def _accept(self) -> None:
        prompt = self.query_one("#console-generate-image-prompt", Input).value.strip()
        if not prompt:
            self.notify("Enter a prompt first.", severity="warning")
            return
        self.dismiss(self._current_command())

    def action_dismiss_modal(self) -> None:
        self.dismiss(None)
