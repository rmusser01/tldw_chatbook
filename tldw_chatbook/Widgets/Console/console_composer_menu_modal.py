"""Composer overflow menu: the ``☰`` button's action list.

task-1680: the composer's action row is width-bounded, so new actions go
behind one menu button rather than growing the row. Each entry returns a
stable action id; the screen owns what each id does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

#: Action ids returned by the menu. Stable strings, not indexes: the
#: screen's dispatch table keys off these and tests pin them.
ACTION_GENERATE_IMAGE = "generate-image"
ACTION_GENERATE_CAPTION = "generate-caption"
ACTION_NARRATE_CONVERSATION = "narrate-conversation"
ACTION_IMPERSONATE = "impersonate"


@dataclass(frozen=True)
class ComposerMenuEntry:
    """One row in the composer menu."""

    action_id: str
    label: str
    description: str
    enabled: bool = True


def build_composer_menu_entries(
    *, has_attachment: bool = False
) -> tuple[ComposerMenuEntry, ...]:
    """Build the menu rows for the current composer state.

    Args:
        has_attachment: Whether the composer currently holds an
            attachment; Generate Caption needs one to caption.

    Returns:
        The menu entries in display order.
    """
    return (
        ComposerMenuEntry(
            ACTION_GENERATE_IMAGE,
            "Generate Image",
            "Build a /generate-image command",
        ),
        ComposerMenuEntry(
            ACTION_GENERATE_CAPTION,
            "Generate Caption",
            "Caption the attached image"
            if has_attachment
            else "Attach an image first",
            enabled=has_attachment,
        ),
        ComposerMenuEntry(
            ACTION_NARRATE_CONVERSATION,
            "Narrate Entire Conversation",
            "Per-speaker voices (not implemented yet)",
        ),
        ComposerMenuEntry(
            ACTION_IMPERSONATE,
            "Impersonate",
            "Draft your next reply with the current model",
        ),
    )


class ConsoleComposerMenuModal(ModalScreen["str | None"]):
    """Pick one composer action; dismisses with its action id."""

    DEFAULT_CSS = """
    ConsoleComposerMenuModal {
        align: center middle;
    }

    #console-composer-menu {
        width: 56;
        height: auto;
        border: tall gray;
        background: black;
        padding: 1 2;
    }

    .console-composer-menu-item {
        width: 100%;
        height: 3;
        margin: 0 0 1 0;
    }

    .console-composer-menu-hint {
        color: $text-muted;
    }
    """

    BINDINGS = [("escape", "dismiss_menu", "Cancel")]

    def __init__(self, *, has_attachment: bool = False, **kwargs: Any) -> None:
        """Initialize the menu.

        Args:
            has_attachment: Whether an attachment is pending, which decides
                if Generate Caption is actionable.
            **kwargs: Forwarded to ``ModalScreen``.
        """
        super().__init__(**kwargs)
        self._entries = build_composer_menu_entries(has_attachment=has_attachment)

    def compose(self) -> ComposeResult:
        with Vertical(id="console-composer-menu"):
            yield Static("Composer actions", classes="console-modal-header")
            for entry in self._entries:
                button = Button(
                    entry.label,
                    id=f"console-composer-menu-{entry.action_id}",
                    classes="console-composer-menu-item",
                )
                button.disabled = not entry.enabled
                button.tooltip = entry.description
                yield button
            yield Static(
                "Esc closes without changing your draft.",
                classes="console-composer-menu-hint",
                markup=False,
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        prefix = "console-composer-menu-"
        if not button_id.startswith(prefix):
            return
        event.stop()
        self.dismiss(button_id[len(prefix) :])

    def action_dismiss_menu(self) -> None:
        self.dismiss(None)
