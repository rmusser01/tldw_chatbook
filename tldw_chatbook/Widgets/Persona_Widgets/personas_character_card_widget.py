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

from .personas_pane_messages import EditCharacterRequested


class PersonasCharacterCardWidget(Container):
    """Flat read-only character card with an Edit action."""

    # Structure only: colors come from the app stylesheet ($ds-* tokens do not
    # resolve in bare-App harnesses, so DEFAULT_CSS must not reference them).
    DEFAULT_CSS = """
    PersonasCharacterCardWidget {
        width: 100%;
        height: 100%;
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
            yield Static("Avatar: none", id="personas-card-avatar-status")
        with Horizontal(classes="ds-toolbar"):
            yield Button(
                "Edit",
                id="personas-card-edit-character",
                classes="console-action-secondary",
                disabled=True,
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

        def _set(suffix: str, value: str) -> None:
            widget = self.query_one(f"#personas-character-card-{suffix}", Static)
            label = labels.get(suffix)
            if label is not None:
                # Label inline with the value: "Name: Detective Sam". Rows
                # with no value are hidden outright - a bare "Label:" line is
                # noise, not information (display toggling keeps this
                # sync-safe and reversible on the next load).
                widget.display = bool(value)
                value = f"{label}: {value}" if value else f"{label}:"
            widget.update(value)

        _set("name", str(record.get("name") or "Unnamed Character"))
        _set("description", str(record.get("description") or ""))
        _set("personality", str(record.get("personality") or ""))
        _set("scenario", str(record.get("scenario") or ""))
        _set(
            "first-message",
            str(record.get("first_mes", record.get("first_message", "")) or ""),
        )
        _set(
            "system-prompt",
            str(record.get("system_prompt", record.get("system", "")) or ""),
        )
        _set("post-history", str(record.get("post_history_instructions") or ""))
        _set("creator", str(record.get("creator") or ""))
        _set(
            "version",
            str(record.get("character_version", record.get("version", "1.0")) or ""),
        )
        tags = [str(tag) for tag in (record.get("tags") or [])]
        _set("tags", f"Tags: {', '.join(tags)}" if tags else "Tags: none")
        greetings = [str(greeting) for greeting in (record.get("alternate_greetings") or [])]
        _set("alt-greetings", f"Alternate greetings: {len(greetings)}")
        _set("greeting-preview", greetings[0] if greetings else "")
        # The preview row is unlabeled, so the labeled-row hiding above does
        # not cover it; an empty preview must not leave a blank line.
        # ("Tags: none", "Alternate greetings: 0", and "Avatar: none" stay
        # visible - they carry information even when empty.)
        self.query_one("#personas-character-card-greeting-preview").display = bool(
            greetings
        )
        avatar = "embedded" if (record.get("image") or record.get("avatar")) else "none"
        self.query_one("#personas-card-avatar-status", Static).update(f"Avatar: {avatar}")

        # Display toggling (never remove/mount) keeps load_character sync-safe
        # for the handler's call_from_thread continuation.
        self.query_one("#personas-character-card-empty").display = False
        self.query_one("#personas-character-card-body").display = True
        self.query_one("#personas-card-edit-character", Button).disabled = (
            self._character_id is None
        )

    # ===== Events =====

    @on(Button.Pressed, "#personas-card-edit-character")
    def _edit_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self._character_id is not None:
            self.post_message(EditCharacterRequested(self._character_id))


__all__ = ["PersonasCharacterCardWidget"]
