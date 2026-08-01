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

from tldw_chatbook.Chat.console_ephemeral import ACTION_SAVE_CHAT, blocked_reason

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
    *, attachment_kind: str = "none", ephemeral: bool = False
) -> tuple[ComposerMenuEntry, ...]:
    """Build the menu rows for the current composer state.

    Generate Caption is disabled -- never hidden -- when it cannot act, and
    the row says which case applies: nothing staged, or a staged file that
    is not an image. Explicit unavailable states beat vanishing entries.

    "Save this chat" is the exception: it is ABSENT outside a temporary
    chat rather than disabled, because a disabled save on an already-saved
    conversation reads as a failure rather than as "already done".

    Args:
        attachment_kind: ``"image"``, ``"other"``, or ``"none"``.
        ephemeral: Whether the active session is temporary.

    Returns:
        The menu entries in display order.
    """
    caption_reason = {
        "image": "Caption the attached image",
        "other": "Attached file is not an image",
        "none": "Attach an image first",
    }.get(attachment_kind, "Attach an image first")
    image_blocked = blocked_reason(ACTION_GENERATE_IMAGE, ephemeral=ephemeral)
    entries = (
        ComposerMenuEntry(
            ACTION_GENERATE_IMAGE,
            "Generate Image",
            image_blocked or "Build a /generate-image command",
            enabled=image_blocked is None,
        ),
        ComposerMenuEntry(
            ACTION_GENERATE_CAPTION,
            "Generate Caption",
            caption_reason,
            enabled=attachment_kind == "image",
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
    if not ephemeral:
        return entries
    return (
        ComposerMenuEntry(
            ACTION_SAVE_CHAT,
            "Save this chat",
            "This chat is not saved locally — save it now",
        ),
        *entries,
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

    def __init__(
        self,
        *,
        attachment_kind: str = "none",
        ephemeral: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the menu.

        Args:
            attachment_kind: ``"image"``, ``"other"`` or ``"none"``, which
                decides whether Generate Caption is actionable and what its
                disabled row explains.
            ephemeral: Whether the active session is temporary, which
                decides whether "Save this chat" is offered at all.
            **kwargs: Forwarded to ``ModalScreen``.
        """
        super().__init__(**kwargs)
        self._entries = build_composer_menu_entries(
            attachment_kind=attachment_kind, ephemeral=ephemeral
        )

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
