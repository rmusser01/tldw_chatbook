"""Presentation-only confirmation dialog for Library Prompt deletion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static


ArtifactType = Literal["prompt", "recipe"]


@dataclass(frozen=True)
class PromptDeleteItem:
    """One immutable display target for a Prompt deletion confirmation."""

    name: str
    artifact_type: ArtifactType


@dataclass(frozen=True)
class PromptDeleteRequest:
    """Immutable data a host captures before it opens the confirmation modal."""

    items: tuple[PromptDeleteItem, ...]
    fingerprint: str | None = None
    dirty: bool = False
    preview_limit: int = 3

    def __post_init__(self) -> None:
        """Reject incomplete or mutable presentation requests at the boundary."""
        if not self.items:
            raise ValueError("Prompt deletion confirmation requires at least one item.")
        if not isinstance(self.items, tuple):
            raise TypeError("Prompt deletion confirmation items must be an immutable tuple.")
        if self.preview_limit < 1:
            raise ValueError("Prompt deletion confirmation preview_limit must be positive.")


@dataclass(frozen=True)
class PromptDeleteDecision:
    """Typed result returned to the host; the host validates the fingerprint."""

    confirmed: bool
    fingerprint: str | None


class PromptDeleteConfirmationModal(ModalScreen[PromptDeleteDecision]):
    """Render a safe, reusable Prompt/Recipe deletion confirmation.

    The modal deliberately owns no deletion or scope-service behavior. Hosts capture
    their identity or selection fingerprint in ``request`` and decide whether the
    returned result is still current before performing any mutation.
    """

    DEFAULT_CSS = """
    PromptDeleteConfirmationModal {
        align: center middle;
    }

    #prompt-delete-modal {
        width: 64;
        height: auto;
        border: tall $error;
        background: $surface;
        padding: 1 2;
    }

    #prompt-delete-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #prompt-delete-copy,
    #prompt-delete-preview {
        height: auto;
        margin-bottom: 1;
    }

    #prompt-delete-actions {
        height: 3;
        margin-top: 1;
        align-horizontal: right;
    }

    #prompt-delete-cancel,
    #prompt-delete-confirm {
        min-width: 12;
        margin-left: 1;
    }
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def __init__(self, request: PromptDeleteRequest) -> None:
        super().__init__()
        self.request = request

    @property
    def _is_single(self) -> bool:
        return len(self.request.items) == 1

    def compose(self) -> ComposeResult:
        """Compose literal-text copy plus cancellation and confirmation controls."""
        with Vertical(id="prompt-delete-modal"):
            yield Static(self._title_copy(), id="prompt-delete-title", markup=False)
            yield Static(self._body_copy(), id="prompt-delete-copy", markup=False)
            yield Static(self._preview_copy(), id="prompt-delete-preview", markup=False)
            with Horizontal(id="prompt-delete-actions"):
                yield Button("Cancel", id="prompt-delete-cancel")
                yield Button("Delete", id="prompt-delete-confirm", variant="error")

    def _title_copy(self) -> str:
        if not self._is_single:
            return f"Delete {len(self.request.items)} items?"
        return f"Delete {self.request.items[0].artifact_type.title()}?"

    def _body_copy(self) -> str:
        if self._is_single:
            item = self.request.items[0]
            artifact_label = item.artifact_type.title()
            if self.request.dirty:
                return (
                    f'The saved {artifact_label} "{item.name}" and this unsaved working copy '
                    "will be discarded."
                )
            return f'The saved {artifact_label} "{item.name}" will be discarded.'
        return f"This will discard {self._plural_count_copy()}."

    def _preview_copy(self) -> str:
        names = [item.name for item in self.request.items[: self.request.preview_limit]]
        hidden_count = len(self.request.items) - len(names)
        if hidden_count:
            names.append(f"and {hidden_count} more")
        return "\n".join(names)

    def _plural_count_copy(self) -> str:
        counts = {
            "prompt": sum(item.artifact_type == "prompt" for item in self.request.items),
            "recipe": sum(item.artifact_type == "recipe" for item in self.request.items),
        }
        parts = []
        for artifact_type in ("prompt", "recipe"):
            count = counts[artifact_type]
            if count:
                label = artifact_type.title() if count == 1 else artifact_type.title() + "s"
                parts.append(f"{count} {label}")
        return " and ".join(parts)

    def action_cancel(self) -> None:
        """Dismiss safely, preserving the captured stale-result fingerprint."""
        self.dismiss(PromptDeleteDecision(False, self.request.fingerprint))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Return a typed decision without invoking deletion infrastructure."""
        if event.button.id == "prompt-delete-cancel":
            event.stop()
            self.action_cancel()
        elif event.button.id == "prompt-delete-confirm":
            event.stop()
            self.dismiss(PromptDeleteDecision(True, self.request.fingerprint))
