"""Explicit runtime choice for a verified managed GGUF model."""

from __future__ import annotations

from typing import Literal

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from ..modal_dismissal import SafeModalDismissMixin


ManagedGGUFRuntimeChoice = Literal["llamacpp", "llamafile"]


class ManagedGGUFRuntimeChoiceModal(
    SafeModalDismissMixin,
    ModalScreen[ManagedGGUFRuntimeChoice | None],
):
    """Choose a compatible runtime without activating or starting it."""

    BUNDLED_CSS = """
    ManagedGGUFRuntimeChoiceModal {
        align: center middle;
    }

    ManagedGGUFRuntimeChoiceModal .managed-gguf-runtime-modal {
        width: 64;
        height: auto;
        border: tall $accent;
        background: $surface;
        padding: 1 2;
    }

    ManagedGGUFRuntimeChoiceModal .managed-gguf-runtime-title {
        text-style: bold;
        margin-bottom: 1;
    }

    ManagedGGUFRuntimeChoiceModal .managed-gguf-runtime-copy {
        height: auto;
        color: $text-muted;
        margin-bottom: 1;
    }

    ManagedGGUFRuntimeChoiceModal .managed-gguf-runtime-actions {
        height: 3;
        align-horizontal: right;
    }

    ManagedGGUFRuntimeChoiceModal .managed-gguf-runtime-actions Button {
        width: auto;
        margin-left: 1;
    }
    """

    BINDINGS = [("escape", "request_safe_cancel", "Cancel")]
    SAFE_MODAL_CONTENT = ".managed-gguf-runtime-modal"

    def compose(self) -> ComposeResult:
        """Compose one compact, keyboard-operable provider decision."""
        with Vertical(classes="managed-gguf-runtime-modal"):
            yield Static(
                "Configure managed GGUF",
                classes="managed-gguf-runtime-title",
                markup=False,
            )
            yield Static(
                "Choose a compatible local runtime. This preselects the exact "
                "managed model; it does not activate it or start a server.",
                classes="managed-gguf-runtime-copy",
                markup=False,
            )
            with Horizontal(classes="managed-gguf-runtime-actions"):
                yield Button(
                    "Cancel",
                    id="managed-gguf-runtime-cancel",
                    variant="default",
                )
                yield Button(
                    "Llamafile",
                    id="managed-gguf-runtime-llamafile",
                    variant="default",
                )
                yield Button(
                    "Llama.cpp",
                    id="managed-gguf-runtime-llamacpp",
                    variant="primary",
                )

    def on_mount(self) -> None:
        """Place keyboard users on the primary compatible runtime."""
        self.query_one("#managed-gguf-runtime-llamacpp", Button).focus()

    @on(Button.Pressed)
    async def _button_pressed(self, event: Button.Pressed) -> None:
        """Dismiss with only an explicit compatible provider identifier."""
        event.stop()
        if event.button.id == "managed-gguf-runtime-llamacpp":
            self.dismiss("llamacpp")
        elif event.button.id == "managed-gguf-runtime-llamafile":
            self.dismiss("llamafile")
        elif event.button.id == "managed-gguf-runtime-cancel":
            await self.request_safe_cancel(source="visible")

    async def _perform_safe_cancel(self, *, source: str) -> None:
        """Dismiss once without selecting a runtime."""
        del source
        self.dismiss_safe_once(None)
