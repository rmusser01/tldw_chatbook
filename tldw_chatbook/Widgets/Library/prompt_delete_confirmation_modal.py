"""Presentation-only confirmation dialog for Library Prompt deletion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from ..modal_dismissal import SafeModalDismissMixin


ArtifactType = Literal["prompt", "recipe"]
_ARTIFACT_TYPES = frozenset(("prompt", "recipe"))
_DISPLAY_NAME_LIMIT = 48


@dataclass(frozen=True)
class PromptDeleteItem:
    """One immutable display target for a Prompt deletion confirmation."""

    name: str
    artifact_type: ArtifactType

    def __post_init__(self) -> None:
        """Fail closed for data outside the supported Prompt/Recipe contract."""
        if type(self.name) is not str:
            raise TypeError("Prompt deletion item names must be strings.")
        if type(self.artifact_type) is not str:
            raise TypeError("Prompt deletion item artifact types must be strings.")
        if self.artifact_type not in _ARTIFACT_TYPES:
            raise ValueError("Prompt deletion item artifact type must be prompt or recipe.")


@dataclass(frozen=True)
class PromptDeleteRequest:
    """Immutable data a host captures before it opens the confirmation modal."""

    items: tuple[PromptDeleteItem, ...]
    fingerprint: str | None = None
    dirty: bool = False
    preview_limit: int = 3

    def __post_init__(self) -> None:
        """Reject incomplete or mutable presentation requests at the boundary."""
        if type(self.items) is not tuple:
            raise TypeError("Prompt deletion confirmation items must be an immutable tuple.")
        if not self.items:
            raise ValueError("Prompt deletion confirmation requires at least one item.")
        if not all(isinstance(item, PromptDeleteItem) for item in self.items):
            raise TypeError("Prompt deletion confirmation items must be PromptDeleteItem values.")
        if self.fingerprint is not None and type(self.fingerprint) is not str:
            raise TypeError("Prompt deletion confirmation fingerprints must be strings or None.")
        if type(self.dirty) is not bool:
            raise TypeError("Prompt deletion confirmation dirty state must be a bool.")
        if type(self.preview_limit) is not int:
            raise TypeError("Prompt deletion confirmation preview_limit must be an integer.")
        if self.preview_limit < 1:
            raise ValueError("Prompt deletion confirmation preview_limit must be positive.")


@dataclass(frozen=True)
class PromptDeleteDecision:
    """Typed result returned to the host; the host validates the fingerprint."""

    confirmed: bool
    fingerprint: str | None

    def __post_init__(self) -> None:
        """Keep the dismissal result as strictly typed as the opening request."""
        if type(self.confirmed) is not bool:
            raise TypeError("Prompt deletion decisions must have a bool confirmation.")
        if self.fingerprint is not None and type(self.fingerprint) is not str:
            raise TypeError("Prompt deletion decision fingerprints must be strings or None.")


class PromptDeleteConfirmationModal(
    SafeModalDismissMixin, ModalScreen[PromptDeleteDecision]
):
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

    BINDINGS = [Binding("escape", "request_safe_cancel", "Cancel", show=False)]
    SAFE_MODAL_CONTENT = "#prompt-delete-modal"

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
        recovery_copy = "You can Undo from the Prompts list after deletion."
        if self._is_single:
            item = self.request.items[0]
            artifact_label = item.artifact_type.title()
            display_name = _display_name(item.name)
            if self.request.dirty:
                return (
                    f'The saved {artifact_label} "{display_name}" and this unsaved working copy '
                    f"will be discarded. {recovery_copy}"
                )
            return (
                f'The saved {artifact_label} "{display_name}" will be discarded. '
                f"{recovery_copy}"
            )
        return (
            f"This will discard {self._plural_count_copy()}. "
            f"{recovery_copy}"
        )

    def _preview_copy(self) -> str:
        names = [
            _display_name(item.name)
            for item in self.request.items[: self.request.preview_limit]
        ]
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

    async def _perform_safe_cancel(self, *, source: str) -> None:
        """Dismiss safely, preserving the captured stale-result fingerprint."""
        del source
        self.dismiss_safe_once(PromptDeleteDecision(False, self.request.fingerprint))

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Return a typed decision without invoking deletion infrastructure.

        Args:
            event: Button press emitted by the modal's Cancel or Delete action.
        """
        if event.button.id == "prompt-delete-cancel":
            event.stop()
            await self.request_safe_cancel(source="visible")
        elif event.button.id == "prompt-delete-confirm":
            event.stop()
            self.dismiss(PromptDeleteDecision(True, self.request.fingerprint))


def _display_name(name: str) -> str:
    """Keep untrusted names literal and compact enough for a terminal modal."""
    normalized = " ".join(name.splitlines())
    if len(normalized) <= _DISPLAY_NAME_LIMIT:
        return normalized
    return normalized[: _DISPLAY_NAME_LIMIT - 1].rstrip() + "…"
