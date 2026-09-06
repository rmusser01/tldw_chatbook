"""Read-only ds-native character card for the Personas workbench.

Replaces ``CCPCharacterCardWidget`` on the Personas screen only. It keeps the
legacy widget's external contract — the ``ccp-character-card-view`` default id
that ``CCPCharacterHandler._display_character_card`` queries, a
``load_character(data)`` entry point, and the legacy ``EditCharacterRequested``
message — while rendering with the workbench's flat ds vocabulary.
"""

from __future__ import annotations

from typing import Any, Dict

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import Button, Static

from ...UI.character_display_text import (
    sanitize_character_display_items,
    sanitize_character_display_label,
    sanitize_character_display_text,
)
from .personas_character_tts_widget import PersonasCharacterTTSWidget
from .personas_pane_messages import ConversationsRequested, EditCharacterRequested

_CARD_NAME_MAX_CHARACTERS = 200
_CARD_LONG_FIELD_MAX_CHARACTERS = 20_000
_CARD_METADATA_MAX_CHARACTERS = 1_000
_CARD_GREETING_PREVIEW_MAX_CHARACTERS = 5_000
_CARD_COLLECTION_MAX_ITEMS = 50


class PersonasCharacterCardWidget(Container):
    """Flat read-only character card with an Edit action."""

    # Structure only: colors come from the app stylesheet ($ds-* tokens do not
    # resolve in bare-App harnesses, so BUNDLED_CSS must not reference them).
    BUNDLED_CSS = """
    PersonasCharacterCardWidget {
        width: 100%;
        /* height: 100% fills the (scrollable) detail-stack viewport so the
           character info owns the center by default; min-height is the real
           floor under it (task-2231 AC#5) - attachment sections can never
           squeeze the card to a sliver, the stack scrolls instead. */
        height: 100%;
        min-height: 10;
    }

    PersonasCharacterCardWidget #personas-character-card-body {
        height: 1fr;
        display: none;
    }

    PersonasCharacterCardWidget .ds-field-row {
        height: auto;
    }

    PersonasCharacterCardWidget .ds-toolbar {
        height: 1;
        min-height: 1;
    }

    PersonasCharacterCardWidget .ds-toolbar Button {
        width: auto;
        min-width: 0;
        height: 1;
        min-height: 1;
        padding: 0 1;
        border: none;
        margin-right: 1;
    }
    """

    #: (label, value-Static id suffix) pairs, each rendered as ONE Static with
    #: the label inline ("Name: Detective Sam") for a clean read-only card.
    _FIELD_ROWS: tuple[tuple[str, str], ...] = (
        ("Name", "name"),
        ("Description", "description"),
        ("Personality", "personality"),
        ("Scenario", "scenario"),
        ("First message", "first-message"),
        ("System prompt", "system-prompt"),
        ("Post-history instructions", "post-history"),
        ("Creator", "creator"),
        ("Version", "version"),
    )

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("id", "ccp-character-card-view")
        super().__init__(**kwargs)
        self._character_id: str | None = None

    def compose(self) -> ComposeResult:
        yield Static("Character", classes="destination-section")
        yield Static(
            "No character loaded. Select one from the library.",
            id="personas-character-card-empty",
        )
        with VerticalScroll(id="personas-character-card-body"):
            # markup=False: these Statics render character-card content, which
            # must display literally (an unmatched [/tag] would raise
            # MarkupError at render time with markup enabled).
            # One Static per field ("Label: value") rather than Label+value
            # pairs: the read-only card reads as clean left-aligned lines.
            for label, suffix in self._FIELD_ROWS:
                yield Static(
                    f"{label}:",
                    id=f"personas-character-card-{suffix}",
                    classes="ds-field-row",
                    markup=False,
                )
            yield Static("Tags: none", id="personas-character-card-tags", markup=False)
            yield Static(
                "Alternate greetings: 0",
                id="personas-character-card-alt-greetings",
                markup=False,
            )
            yield Static(
                "", id="personas-character-card-greeting-preview", markup=False
            )
            yield PersonasCharacterTTSWidget(context="card")
            yield Static("Avatar: none", id="personas-card-avatar-status")
        with Horizontal(classes="ds-toolbar"):
            yield Button(
                "Edit",
                id="personas-card-edit-character",
                classes="console-action-secondary",
                disabled=True,
                # F-037: a disabled Edit explains itself.
                tooltip="Select a character to edit.",
            )
            yield Button(
                "Conversations (0)",
                id="personas-card-conversations",
                classes="console-action-secondary",
                disabled=True,
                tooltip="Select a character to browse its conversations.",
            )

    # ===== Public API =====

    def load_character(self, data: Dict[str, Any]) -> None:
        """Display ``data``; tolerant of ``first_mes``/``first_message`` aliases.

        ``CCPCharacterHandler._display_character_card`` calls this when it
        queries ``#ccp-character-card-view`` and finds a ``load_character``
        attribute, so the signature must stay handler-compatible.
        """
        record = dict(data or {})
        raw_id = record.get("id")
        self._character_id = str(raw_id) if raw_id is not None else None

        labels = {suffix: label for label, suffix in self._FIELD_ROWS}

        def _set(
            suffix: str,
            value: object,
            *,
            max_characters: int = _CARD_LONG_FIELD_MAX_CHARACTERS,
            single_line: bool = False,
        ) -> None:
            widget = self.query_one(f"#personas-character-card-{suffix}", Static)
            label = labels.get(suffix)
            sanitizer = (
                sanitize_character_display_label
                if single_line
                else sanitize_character_display_text
            )
            display_value = sanitizer(value, max_characters=max_characters)
            if label is not None:
                # Label inline with the value: "Name: Detective Sam". Rows
                # with no value are hidden outright - a bare "Label:" line is
                # noise, not information (display toggling keeps this
                # sync-safe and reversible on the next load).
                widget.display = bool(display_value)
                display_value = (
                    f"{label}: {display_value}" if display_value else f"{label}:"
                )
            widget.update(display_value)

        _set(
            "name",
            record.get("name") or "Unnamed Character",
            max_characters=_CARD_NAME_MAX_CHARACTERS,
            single_line=True,
        )
        _set("description", record.get("description") or "")
        _set("personality", record.get("personality") or "")
        _set("scenario", record.get("scenario") or "")
        _set(
            "first-message",
            record.get("first_mes", record.get("first_message", "")) or "",
        )
        _set(
            "system-prompt",
            record.get("system_prompt", record.get("system", "")) or "",
        )
        _set("post-history", record.get("post_history_instructions") or "")
        _set(
            "creator",
            record.get("creator") or "",
            max_characters=_CARD_METADATA_MAX_CHARACTERS,
        )
        _set(
            "version",
            record.get("character_version", record.get("version", "1.0")) or "",
            max_characters=_CARD_METADATA_MAX_CHARACTERS,
        )
        tags = sanitize_character_display_items(
            record.get("tags"),
            max_items=_CARD_COLLECTION_MAX_ITEMS,
            max_item_characters=_CARD_METADATA_MAX_CHARACTERS,
            max_total_characters=_CARD_LONG_FIELD_MAX_CHARACTERS,
            single_line=True,
        )
        tags_text = ", ".join(tags)
        _set(
            "tags",
            f"Tags: {tags_text}" if tags else "Tags: none",
            max_characters=_CARD_LONG_FIELD_MAX_CHARACTERS,
        )
        greetings = sanitize_character_display_items(
            record.get("alternate_greetings"),
            max_items=_CARD_COLLECTION_MAX_ITEMS,
            max_item_characters=_CARD_GREETING_PREVIEW_MAX_CHARACTERS,
            max_total_characters=_CARD_LONG_FIELD_MAX_CHARACTERS,
        )
        _set(
            "alt-greetings",
            f"Alternate greetings: {len(greetings)}",
            max_characters=_CARD_METADATA_MAX_CHARACTERS,
        )
        _set(
            "greeting-preview",
            greetings[0] if greetings else "",
            max_characters=_CARD_GREETING_PREVIEW_MAX_CHARACTERS,
        )
        # The preview row is unlabeled, so the labeled-row hiding above does
        # not cover it; an empty preview must not leave a blank line.
        # ("Tags: none", "Alternate greetings: 0", and "Avatar: none" stay
        # visible - they carry information even when empty.)
        self.query_one("#personas-character-card-greeting-preview").display = bool(
            greetings
        )
        avatar = "embedded" if (record.get("image") or record.get("avatar")) else "none"
        self.query_one("#personas-card-avatar-status", Static).update(
            f"Avatar: {avatar}"
        )

        # Display toggling (never remove/mount) keeps load_character sync-safe
        # for the handler's call_from_thread continuation.
        has_record = bool(record)
        self.query_one("#personas-character-card-empty").display = not has_record
        self.query_one("#personas-character-card-body").display = has_record
        edit_button = self.query_one("#personas-card-edit-character", Button)
        edit_button.disabled = self._character_id is None
        edit_button.tooltip = (
            "Select a character to edit."
            if not has_record
            else (
                None
                if self._character_id is not None
                else "This character has no saved record to edit."
            )
        )
        conversations = self.query_one("#personas-card-conversations", Button)
        conversations.disabled = self._character_id is None
        conversations.tooltip = (
            None
            if self._character_id is not None
            else "Select a character to browse its conversations."
        )

    def set_conversation_total(self, total: int) -> None:
        """Expose the selected character's authoritative conversation count."""

        self.query_one("#personas-card-conversations", Button).label = (
            f"Conversations ({max(0, int(total))})"
        )

    # ===== Events =====

    @on(Button.Pressed, "#personas-card-edit-character")
    def _edit_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self._character_id is not None:
            self.post_message(EditCharacterRequested(self._character_id))

    @on(Button.Pressed, "#personas-card-conversations")
    def _conversations_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self._character_id is not None:
            self.post_message(ConversationsRequested())


__all__ = ["PersonasCharacterCardWidget"]
