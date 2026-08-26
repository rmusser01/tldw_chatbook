"""Composer overflow menu: the composer ``Menu`` button's action list.

task-1680: the composer's action row is width-bounded, so new actions go
behind one menu button rather than growing the row. Each entry returns a
stable action id; the screen owns what each id does. task-2154.14: the
button is labeled "Menu" in words (DS-01) -- a bare ☰ glyph had a
tooltip-only identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_ephemeral import ACTION_SAVE_CHAT, blocked_reason
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin

#: Action ids returned by the menu. Stable strings, not indexes: the
#: screen's dispatch table keys off these and tests pin them.
ACTION_GENERATE_IMAGE = "generate-image"
ACTION_GENERATE_CAPTION = "generate-caption"
#: task-2154.14 (DS-03): the "Narrate Entire Conversation" entry was removed
#: until per-speaker narration is actually implemented -- a visible menu row
#: whose only behavior is a "not implemented yet" toast erodes trust in every
#: other entry. Re-add the action id here when the feature lands.
ACTION_IMPERSONATE = "impersonate"
#: Moved here from the composer action row. These two strings are also the
#: ids of the buttons they replaced, so the screen's existing
#: `@on(Button.Pressed, "#console-<id>")` handlers stay the one
#: implementation and the menu just becomes a second way in.
ACTION_ATTACH_CONTEXT = "attach-context"
#: Matches the `EPHEMERAL_BLOCKED_ACTIONS` registry key, so the
#: temporary-chat block needs no translation layer.
ACTION_SAVE_CHATBOOK = "save-chatbook"
ACTION_PROMPTS = "prompts"
ACTION_IMPROVE_CURRENT_DRAFT = "improve-current-draft"
ACTION_UNDO_PROMPT_IMPROVEMENT = "undo-prompt-improvement"


@dataclass(frozen=True)
class ComposerMenuEntry:
    """One row in the composer menu."""

    action_id: str
    label: str
    description: str
    enabled: bool = True


def build_composer_menu_entries(
    *,
    attachment_kind: str = "none",
    ephemeral: bool = False,
    can_save_chatbook: bool = False,
    draft_available: bool = False,
    improvement_undo_available: bool = False,
) -> tuple[ComposerMenuEntry, ...]:
    """Build the menu rows for the current composer state.

    Generate Caption is disabled -- never hidden -- when it cannot act, and
    the row says which case applies: nothing staged, or a staged file that
    is not an image. Explicit unavailable states beat vanishing entries.

    "Save this chat" is the exception: it is ABSENT outside a temporary
    chat rather than disabled, because a disabled save on an already-saved
    conversation reads as a failure rather than as "already done".

    Attach and Save as Chatbook moved here from the composer's action row,
    which is width-bounded at a fixed cell count: every always-present
    button there is space the draft never gets back. The row keeps only
    ``Send``, ``Mic``, and the two CONDITIONAL controls (``Stop`` while a
    run is active, ``✕`` while an attachment is staged) -- those cost
    nothing at rest and are time-critical when they appear.

    Args:
        attachment_kind: ``"image"``, ``"other"``, or ``"none"``.
        ephemeral: Whether the active session is temporary.
        can_save_chatbook: Whether a Chatbook artifact is available to save.
        draft_available: Whether a nonblank unsent message can be improved.

    Returns:
        The menu entries in display order.
    """
    caption_reason = {
        "image": "Caption the attached image",
        "other": "Attached file is not an image",
        "none": "Attach an image first",
    }.get(attachment_kind, "Attach an image first")
    image_blocked = blocked_reason(ACTION_GENERATE_IMAGE, ephemeral=ephemeral)
    # Same disabled-with-a-reason contract the action-row button had, moved
    # verbatim: the temporary-chat block wins over artifact availability,
    # because the write itself is the problem and readiness is moot.
    chatbook_blocked = blocked_reason(ACTION_SAVE_CHATBOOK, ephemeral=ephemeral)
    chatbook_reason = chatbook_blocked or (
        "Open the available Chatbook artifact in Artifacts"
        if can_save_chatbook
        else "No Chatbook artifact is available to save yet"
    )
    entries = (
        *(
            (
                ComposerMenuEntry(
                    ACTION_IMPROVE_CURRENT_DRAFT,
                    "Improve current draft…",
                    "Improve the unsent message with the current provider and model",
                ),
            )
            if draft_available
            else ()
        ),
        ComposerMenuEntry(
            ACTION_PROMPTS,
            "Browse Prompt Library…",
            "Browse saved Prompts and Recipes",
        ),
        ComposerMenuEntry(
            # CN-03 (TASK-2154.13): this entry opens the file picker
            # (`_handle_console_attach_context`), while the control bar's
            # "Attach context" opens the Library/workspace rail -- two
            # actions, so two names, one word apart.
            ACTION_ATTACH_CONTEXT,
            "Attach file",
            "Attach a file to the draft",
        ),
        ComposerMenuEntry(
            ACTION_SAVE_CHATBOOK,
            "Save as Chatbook",
            chatbook_reason,
            enabled=chatbook_blocked is None and can_save_chatbook,
        ),
        *(
            (
                ComposerMenuEntry(
                    ACTION_UNDO_PROMPT_IMPROVEMENT,
                    "Undo Prompt change",
                    "Restore the draft captured before the latest Prompt change.",
                ),
            )
            if improvement_undo_available
            else ()
        ),
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


class ConsoleComposerMenuModal(SafeModalDismissMixin, ModalScreen["str | None"]):
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

    /* Sits directly under the disabled row it explains. `$warning` rather
       than `$text-muted`: the earlier live pass measured disabled-label
       contrast at ~1.1:1, so muting the explanation too would make the
       reason as unreadable as the state it explains. */
    .console-composer-menu-reason {
        color: $warning;
        margin: 0 0 1 2;
        width: 100%;
    }
    """

    SAFE_MODAL_CONTENT = "#console-composer-menu"
    BINDINGS = [("escape", "request_safe_cancel", "Cancel")]

    def __init__(
        self,
        *,
        attachment_kind: str = "none",
        ephemeral: bool = False,
        can_save_chatbook: bool = False,
        draft_available: bool = False,
        improvement_undo_available: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the menu.

        Args:
            attachment_kind: ``"image"``, ``"other"`` or ``"none"``, which
                decides whether Generate Caption is actionable and what its
                disabled row explains.
            ephemeral: Whether the active session is temporary, which
                decides whether "Save this chat" is offered at all.
            can_save_chatbook: Whether a Chatbook artifact is available,
                which decides whether the Save as Chatbook row is actionable.
            draft_available: Whether the active composer has a nonblank draft.
            **kwargs: Forwarded to ``ModalScreen``.
        """
        super().__init__(**kwargs)
        self._entries = build_composer_menu_entries(
            attachment_kind=attachment_kind,
            ephemeral=ephemeral,
            can_save_chatbook=can_save_chatbook,
            draft_available=draft_available,
            improvement_undo_available=improvement_undo_available,
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
                # A disabled row must SAY why on screen. The tooltip alone
                # made "disabled with a stated reason" a promise the TUI
                # never kept -- hovering is not a gesture keyboard users
                # make, and the dimming alone reads as "broken". Only
                # disabled rows carry the line, so the menu stays compact
                # in the common case where everything is available.
                if not entry.enabled:
                    yield Static(
                        entry.description,
                        classes="console-composer-menu-reason",
                        markup=False,
                    )
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
