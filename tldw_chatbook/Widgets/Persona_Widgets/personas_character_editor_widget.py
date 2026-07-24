"""ds-native character editor form for the Personas workbench.

Replaces ``CCPCharacterEditorWidget`` on the Personas screen only. It keeps
the legacy widget's external contract — the ``ccp-character-editor-view``
default id, the ``load_character``/``new_character``/``get_character_data``
API, and the legacy ``CharacterSaveRequested``/``CharacterEditorCancelled``
messages — while rendering with the workbench's flat ds vocabulary (primary
fields up top, an Advanced section for the long tail, an avatar upload/status
line, no image box).
"""

from __future__ import annotations

from typing import Any, Dict, List

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.timer import Timer
from textual.widgets import Button, DataTable, Input, Label, Static, TextArea

from ...Character_Chat.world_book_manager import CHARACTER_WORLD_BOOKS_KEY
from ...Chat.console_expression_state import EXPRESSION_IMAGE_STATES
from .personas_pane_messages import (
    CharacterEditorCancelled,
    CharacterExpressionClearRequested,
    CharacterExpressionSetExportRequested,
    CharacterExpressionSetImportRequested,
    CharacterExpressionUploadRequested,
    CharacterImageRemoveRequested,
    CharacterImageUploadRequested,
    CharacterSaveRequested,
    EditorContentChanged,
)


class PersonasCharacterEditorWidget(Container):
    """ds-field-row character form with an Advanced section and avatar status."""

    # Structure only: colors come from the app stylesheet ($ds-* tokens do not
    # resolve in bare-App harnesses, so DEFAULT_CSS must not reference them).
    DEFAULT_CSS = """
    PersonasCharacterEditorWidget {
        width: 100%;
        height: 100%;
    }

    PersonasCharacterEditorWidget #personas-char-editor-body {
        height: 1fr;
    }

    PersonasCharacterEditorWidget .ds-field-row {
        height: auto;
    }

    PersonasCharacterEditorWidget #personas-char-editor-first-message {
        height: 3;
    }

    PersonasCharacterEditorWidget #personas-char-editor-description {
        height: 4;
    }

    PersonasCharacterEditorWidget #personas-char-editor-personality {
        height: 3;
    }

    PersonasCharacterEditorWidget #personas-char-editor-system-prompt {
        height: 3;
    }

    PersonasCharacterEditorWidget #personas-char-editor-advanced {
        height: auto;
        display: none;
    }

    PersonasCharacterEditorWidget #personas-char-editor-scenario,
    PersonasCharacterEditorWidget #personas-char-editor-post-history,
    PersonasCharacterEditorWidget #personas-char-editor-creator-notes {
        height: 2;
    }

    PersonasCharacterEditorWidget #personas-char-editor-greetings-table {
        min-height: 4;
        max-height: 8;
    }

    PersonasCharacterEditorWidget #personas-char-editor-greeting-edit {
        height: 3;
    }

    PersonasCharacterEditorWidget #personas-char-editor-advanced-toggle {
        width: auto;
        min-width: 0;
        height: 1;
        min-height: 1;
        padding: 0 1;
        border: none;
    }

    PersonasCharacterEditorWidget #personas-char-editor-avatar-row {
        height: auto;
        min-height: 1;
        padding: 0 1;
    }

    PersonasCharacterEditorWidget #personas-char-editor-avatar-status {
        width: auto;
        margin-right: 2;
    }

    PersonasCharacterEditorWidget #personas-char-editor-avatar-upload,
    PersonasCharacterEditorWidget #personas-char-editor-avatar-remove {
        width: auto;
        min-width: 0;
        height: 1;
        min-height: 1;
        padding: 0 1;
        border: none;
    }

    /* A compact editor thumbnail box - smaller than the 80x40 chat
       transcript image box, since this is a single always-visible avatar
       preview rather than a scrolling message history. */
    PersonasCharacterEditorWidget #personas-char-editor-avatar-thumb {
        height: 10;
        max-width: 24;
        max-height: 10;
        padding: 0 1;
    }

    /* Import/export set buttons (Roleplay P3d-2 Task 4) - the "Expressions"
       section header plus its two whole-set actions on one line. */
    PersonasCharacterEditorWidget .personas-char-editor-expr-set-row {
        height: auto;
        min-height: 1;
    }

    PersonasCharacterEditorWidget #personas-char-editor-expr-import,
    PersonasCharacterEditorWidget #personas-char-editor-expr-export {
        width: auto;
        min-width: 0;
        height: 1;
        min-height: 1;
        padding: 0 1;
        margin-left: 1;
        border: none;
    }

    /* Expression authoring slots (Roleplay P3d-1 Task 4) - one row per
       state, structurally mirroring the avatar-row + avatar-thumb pair
       above (label/hint/buttons on one line, the thumbnail box below). */
    PersonasCharacterEditorWidget .personas-char-editor-expression-slot {
        height: auto;
        min-height: 1;
    }

    PersonasCharacterEditorWidget .personas-char-editor-expr-row {
        height: auto;
        min-height: 1;
        padding: 0 1;
    }

    PersonasCharacterEditorWidget .personas-char-editor-expr-label {
        width: auto;
        margin-right: 1;
    }

    PersonasCharacterEditorWidget .personas-char-editor-expr-hint {
        width: auto;
        margin-right: 2;
    }

    PersonasCharacterEditorWidget .personas-char-editor-expr-upload,
    PersonasCharacterEditorWidget .personas-char-editor-expr-clear {
        width: auto;
        min-width: 0;
        height: 1;
        min-height: 1;
        padding: 0 1;
        border: none;
    }

    /* Same box as #personas-char-editor-avatar-thumb above - Task 4 reuses
       PersonasScreen._fit_avatar_cell_size/_build_avatar_pixels unchanged
       for the expression thumbnails, so the box must match. */
    PersonasCharacterEditorWidget .personas-char-editor-expr-thumb {
        height: 10;
        max-width: 24;
        max-height: 10;
        padding: 0 1;
    }

    PersonasCharacterEditorWidget .ds-toolbar {
        height: 1;
        min-height: 1;
    }

    PersonasCharacterEditorWidget .ds-toolbar Button {
        width: auto;
        min-width: 0;
        height: 1;
        min-height: 1;
        padding: 0 1;
        border: none;
        margin-right: 1;
    }

    /* Live per-field validation (Roleplay P3b Task 4): a literal color, not
       a $ds-* token - DEFAULT_CSS must resolve in bare-App test harnesses
       that never load the app stylesheet. */
    PersonasCharacterEditorWidget .is-invalid {
        border: round red;
    }
    """

    #: CSS class toggled on an offending error-level field's enclosing row
    #: by ``_run_validation``.
    _FIELD_ERROR_CLASS = "is-invalid"
    #: Delay before a field-change-triggered validation pass runs, matching
    #: the library search debounce (``PERSONAS_SEARCH_DEBOUNCE_SECONDS`` in
    #: personas_screen.py).
    _VALIDATION_DEBOUNCE_SECONDS = 0.2

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("id", "ccp-character-editor-view")
        super().__init__(**kwargs)
        # Base copy of the loaded record: get_character_data starts from it so
        # id/version (and any keys the form does not edit, e.g.
        # character_book) survive a load -> save round trip.
        self._character_data: Dict[str, Any] = {}
        # Alternate greetings: a real list editor (DataTable + scratch edit
        # TextArea), not a newline-joined blob. Each greeting is a discrete
        # str mutated in place via _greetings_add/_update/_delete/_move, so a
        # greeting containing embedded newlines round-trips byte-identical —
        # there is no join/split step to corrupt it.
        self._greetings: List[str] = []
        self._selected_greeting_index: int | None = None
        # Dirty tracking (UX-E3): ``_loading`` suppresses Changed events that
        # are dispatched while a programmatic population is in progress;
        # ``_loaded_snapshot`` is the authoritative suppressor for the ones
        # Textual delivers AFTER ``load_character`` returns (programmatic
        # ``value``/``text`` sets post Changed asynchronously). ``None`` means
        # no editing session has started yet. ``_dirty_posted`` makes the
        # EditorContentChanged announcement once-per-session.
        self._loading: bool = False
        self._loaded_snapshot: tuple | None = None
        self._dirty_posted: bool = False
        # Live validation (Roleplay P3b Task 4): the pending debounce timer
        # scheduled by _field_changed, cancelled and re-armed on every real
        # field change so only the last edit in a typing burst validates.
        self._validation_timer: Timer | None = None
        # Fix-wave gate: a freshly-opened form (load_character/new_character)
        # must not display validation errors before the user has actually
        # interacted with it. Set True on a genuine field edit, an avatar
        # action, a greeting mutation, or a Save click; reset on every load.
        self._user_touched: bool = False

    def compose(self) -> ComposeResult:
        yield Static("Character Editor", classes="destination-section")
        with VerticalScroll(id="personas-char-editor-body"):
            with Vertical(classes="ds-field-row"):
                yield Label("Name")
                yield Input(
                    id="personas-char-editor-name", placeholder="Character name"
                )
            with Vertical(classes="ds-field-row"):
                yield Label("First message")
                yield TextArea(id="personas-char-editor-first-message")
            with Vertical(classes="ds-field-row"):
                yield Label("Description")
                yield TextArea(id="personas-char-editor-description")
            with Vertical(classes="ds-field-row"):
                yield Label("Personality")
                yield TextArea(id="personas-char-editor-personality")
            with Vertical(classes="ds-field-row"):
                yield Label("System prompt")
                yield TextArea(id="personas-char-editor-system-prompt")
            yield Button(
                "Advanced ▸",
                id="personas-char-editor-advanced-toggle",
                classes="console-action-subdued",
            )
            with Vertical(id="personas-char-editor-advanced"):
                # Long-form content fields first (grouped with the alternate-
                # greetings list editor, itself content), then the short
                # single-line bookkeeping inputs (Creator/Version/Tags) as a
                # trailing cluster - keeps similar-weight widgets together
                # rather than sandwiching the DataTable+toolbar list editor
                # between unrelated one-line Inputs (Roleplay P3b Task 5).
                with Vertical(classes="ds-field-row"):
                    yield Label("Scenario")
                    yield TextArea(id="personas-char-editor-scenario")
                with Vertical(classes="ds-field-row"):
                    yield Label("Post-history instructions")
                    yield TextArea(id="personas-char-editor-post-history")
                with Vertical(classes="ds-field-row"):
                    yield Label("Creator notes")
                    yield TextArea(id="personas-char-editor-creator-notes")
                with Vertical(classes="ds-field-row"):
                    yield Label("Alternate greetings")
                    yield DataTable(
                        id="personas-char-editor-greetings-table", cursor_type="row"
                    )
                    yield TextArea(id="personas-char-editor-greeting-edit")
                    with Horizontal(classes="ds-toolbar"):
                        yield Button(
                            "Add",
                            id="personas-char-editor-greeting-add",
                            classes="console-action-subdued",
                        )
                        yield Button(
                            "Update",
                            id="personas-char-editor-greeting-update",
                            classes="console-action-subdued",
                        )
                        yield Button(
                            "Delete",
                            id="personas-char-editor-greeting-delete",
                            classes="console-action-subdued",
                        )
                        yield Button(
                            "Move up",
                            id="personas-char-editor-greeting-move-up",
                            classes="console-action-subdued",
                        )
                        yield Button(
                            "Move down",
                            id="personas-char-editor-greeting-move-down",
                            classes="console-action-subdued",
                        )
                with Vertical(classes="ds-field-row"):
                    yield Label("Creator")
                    yield Input(
                        id="personas-char-editor-creator", placeholder="Creator name"
                    )
                with Vertical(classes="ds-field-row"):
                    yield Label("Version")
                    yield Input(id="personas-char-editor-version", value="1.0")
                with Vertical(classes="ds-field-row"):
                    yield Label("Tags (comma-separated)")
                    yield Input(
                        id="personas-char-editor-tags", placeholder="tag, another tag"
                    )
            with Horizontal(id="personas-char-editor-avatar-row"):
                yield Static("Avatar: none", id="personas-char-editor-avatar-status")
                yield Button(
                    "Upload",
                    id="personas-char-editor-avatar-upload",
                    classes="console-action-subdued",
                )
                yield Button(
                    "Remove",
                    id="personas-char-editor-avatar-remove",
                    classes="console-action-subdued",
                )
            yield Container(id="personas-char-editor-avatar-thumb")
            with Horizontal(classes="personas-char-editor-expr-set-row"):
                yield Static("Expressions", classes="destination-section")
                yield Button(
                    "Import set…",
                    id="personas-char-editor-expr-import",
                    classes="console-action-subdued",
                )
                yield Button(
                    "Export set…",
                    id="personas-char-editor-expr-export",
                    classes="console-action-subdued",
                )
            for state in EXPRESSION_IMAGE_STATES:
                with Vertical(
                    id=f"char-expression-slot-{state}",
                    classes="personas-char-editor-expression-slot",
                ):
                    with Horizontal(classes="personas-char-editor-expr-row"):
                        yield Static(
                            f"{state.capitalize()}:",
                            classes="personas-char-editor-expr-label",
                        )
                        yield Static(
                            "Save the character to add expressions.",
                            id=f"personas-char-editor-expr-{state}-hint",
                            classes="personas-char-editor-expr-hint",
                        )
                        yield Button(
                            "Upload",
                            id=f"personas-char-editor-expr-{state}-upload",
                            classes="console-action-subdued personas-char-editor-expr-upload",
                            disabled=True,
                        )
                        yield Button(
                            "Clear",
                            id=f"personas-char-editor-expr-{state}-clear",
                            classes="console-action-subdued personas-char-editor-expr-clear",
                            disabled=True,
                        )
                    yield Container(
                        id=f"personas-char-editor-expr-{state}-thumb",
                        classes="personas-char-editor-expr-thumb",
                    )
        yield Static("", id="personas-char-editor-validation")
        with Horizontal(classes="ds-toolbar"):
            yield Button(
                "Save", id="personas-char-editor-save", classes="console-action-primary"
            )
            yield Button(
                "Cancel",
                id="personas-char-editor-cancel",
                classes="console-action-secondary",
            )

    def on_mount(self) -> None:
        """Register the alternate-greetings table's single column."""
        self.query_one(
            "#personas-char-editor-greetings-table", DataTable
        ).add_column("Greeting", key="g")

    # ===== Field accessors =====

    def _input(self, suffix: str) -> Input:
        return self.query_one(f"#personas-char-editor-{suffix}", Input)

    def _area(self, suffix: str) -> TextArea:
        return self.query_one(f"#personas-char-editor-{suffix}", TextArea)

    # ===== Public API =====

    def load_character(self, data: Dict[str, Any]) -> None:
        """Fill the form from ``data`` (tolerant of legacy key aliases)."""
        self._loading = True
        try:
            self._populate_form(data)
        finally:
            self._loading = False
        # Re-arm dirty tracking for the new session. The Changed events fired
        # by the programmatic sets above are delivered after this method
        # returns, so the handler compares against this snapshot (taken from
        # the just-populated form) and ignores events that match it.
        self._loaded_snapshot = self._form_snapshot()
        self._dirty_posted = False
        self._user_touched = False

    def mark_saved(self, record: Dict[str, Any]) -> None:
        """Re-baseline dirty state to a just-persisted record (save-in-place).

        Adopts the saved record as the new base (so the next Save carries the
        new ``version`` and any DB-normalized keys), rebaselines the greetings
        list, resets the dirty snapshot from the CURRENT form (which already
        shows the saved values), and clears validation. Does NOT repopulate
        the form - the user's saved edits stay on screen.

        Args:
            record: The just-persisted character record (carries the
                incremented optimistic-lock ``version``).
        """
        self._character_data = dict(record or {})
        self._greetings = [
            str(g) for g in (self._character_data.get("alternate_greetings") or [])
        ]
        self._loaded_snapshot = self._form_snapshot()
        self._dirty_posted = False
        self.query_one("#personas-char-editor-validation", Static).update("")
        # A create-session's first Save assigns the brand-new character its
        # id here - the expression slots must flip from disabled to enabled
        # in that same moment (see _sync_expression_slots_enabled).
        self._sync_expression_slots_enabled()

    def _populate_form(self, data: Dict[str, Any]) -> None:
        self._character_data = dict(data or {})
        record = self._character_data
        self._input("name").value = str(record.get("name") or "")
        self._area("first-message").text = str(
            record.get("first_mes", record.get("first_message", "")) or ""
        )
        self._area("description").text = str(record.get("description") or "")
        self._area("personality").text = str(record.get("personality") or "")
        self._area("system-prompt").text = str(
            record.get("system_prompt", record.get("system", "")) or ""
        )
        self._area("scenario").text = str(record.get("scenario") or "")
        self._area("post-history").text = str(
            record.get("post_history_instructions") or ""
        )
        self._area("creator-notes").text = str(record.get("creator_notes") or "")
        self._input("creator").value = str(record.get("creator") or "")
        self._input("version").value = str(
            record.get("character_version", record.get("version", "1.0")) or "1.0"
        )
        self._input("tags").value = ", ".join(
            str(tag) for tag in (record.get("tags") or [])
        )
        self._greetings = [
            str(greeting) for greeting in (record.get("alternate_greetings") or [])
        ]
        self._selected_greeting_index = None
        self.query_one("#personas-char-editor-greeting-edit", TextArea).text = ""
        self._render_greetings_table()
        self._set_avatar_status_from_record()
        self._sync_expression_slots_enabled()
        self.query_one("#personas-char-editor-validation", Static).update("")
        # Clear any stale per-field invalid marks left by a prior session: if
        # the reopened record's values are byte-identical to what's already
        # displayed, no Changed event fires and _run_validation never runs to
        # self-heal a previously-marked row (Roleplay P3b review fix).
        for fid in self._validated_field_ids():
            self.query_one(f"#{fid}").parent.remove_class(self._FIELD_ERROR_CLASS)
        self._set_advanced_open(False)

    def new_character(self) -> None:
        """Clear the form for a new (unsaved) character; version defaults 1.0."""
        self.load_character({})

    def set_avatar_image(self, image_data: bytes) -> None:
        """Stage avatar image bytes for persistence on the next Save.

        Args:
            image_data: Non-empty avatar image bytes selected by the upload flow.

        Raises:
            ValueError: If ``image_data`` is not non-empty bytes.
        """
        if not isinstance(image_data, bytes) or not image_data:
            raise ValueError("Avatar image data must be non-empty bytes.")
        self._character_data["image"] = image_data
        self._set_avatar_status_from_record()
        self._mark_dirty()
        # A discrete user action (upload flow) - validate immediately, no
        # debounce needed, so an oversized avatar's error appears at once.
        self._user_touched = True
        self._run_validation()

    def current_avatar_bytes(self) -> bytes | None:
        """Return the loaded record's embedded avatar bytes, if any.

        Returns:
            The ``image`` key's bytes, or ``None`` when absent or not a
            bytes-like value (e.g. the legacy ``avatar`` URL/path string,
            which this editor does not decode as an image).
        """
        data = self._character_data.get("image")
        return data if isinstance(data, (bytes, bytearray)) else None

    def set_avatar_thumbnail(self, renderable: object | None) -> None:
        """Mount a prepared avatar renderable, or clear to the text status.

        The screen owns decoding (off-thread, via ``ConsoleImageRenderCache``)
        and passes the finished renderable here; this method only mounts it -
        a rich renderable (e.g. ``rich_pixels.Pixels``) mounts inside a
        ``Static``, while a Textual widget (e.g. a ``textual_image`` graphics
        ``Image``) mounts directly.

        Args:
            renderable: The prepared renderable to display, or ``None`` to
                clear the thumbnail (leaving the text status as the sole
                avatar indicator).
        """
        holder = self.query_one("#personas-char-editor-avatar-thumb", Container)
        holder.remove_children()
        if renderable is None:
            return
        from textual.widget import Widget as _W
        from textual.widgets import Static as _S

        holder.mount(renderable if isinstance(renderable, _W) else _S(renderable))

    def expression_character_id(self) -> int | None:
        """Return the loaded record's integer id, or ``None`` when unsaved.

        The expression-state images (``character_expression_images``, Task 1)
        are keyed on the character's row id, independent of the card's own
        optimistic-lock ``version`` - a brand-new, not-yet-saved character has
        no id to attach them to, which is what gates the upload/clear slots.
        """
        character_id = self._character_data.get("id")
        return character_id if isinstance(character_id, int) else None

    def _sync_expression_slots_enabled(self) -> None:
        """Enable the expression slots only for a saved character (has an id).

        Called from both ``_populate_form`` (a fresh load/new session) and
        ``mark_saved`` (a create-session's first Save, which is the moment an
        until-then-unsaved character gains its id).
        """
        enabled = self.expression_character_id() is not None
        hint_text = "" if enabled else "Save the character to add expressions."
        self.query_one("#personas-char-editor-expr-import", Button).disabled = not enabled
        self.query_one("#personas-char-editor-expr-export", Button).disabled = not enabled
        for state in EXPRESSION_IMAGE_STATES:
            self.query_one(
                f"#personas-char-editor-expr-{state}-upload", Button
            ).disabled = not enabled
            self.query_one(
                f"#personas-char-editor-expr-{state}-clear", Button
            ).disabled = not enabled
            self.query_one(
                f"#personas-char-editor-expr-{state}-hint", Static
            ).update(hint_text)

    def set_expression_thumbnail(self, state: str, renderable: object | None) -> None:
        """Mount a prepared expression-slot renderable, or clear it.

        Mirrors ``set_avatar_thumbnail`` exactly, scoped to one state's thumb
        container; the screen owns decoding and passes the finished
        renderable here.

        Args:
            state: One of ``EXPRESSION_IMAGE_STATES``.
            renderable: The prepared renderable to display, or ``None`` to
                clear the thumbnail.
        """
        holder = self.query_one(f"#personas-char-editor-expr-{state}-thumb", Container)
        holder.remove_children()
        if renderable is None:
            return
        from textual.widget import Widget as _W
        from textual.widgets import Static as _S

        holder.mount(renderable if isinstance(renderable, _W) else _S(renderable))

    def validate(self) -> list[tuple[str, str, str]]:
        """Live per-field checks: name required, oversized avatar, blank greetings.

        Returns:
            ``(field_id, message, level)`` tuples, ``level`` in
            ``{"error", "warning"}``. Errors block Save (see
            ``_save_pressed``); warnings are informational only and never
            mark a row invalid.
        """
        findings: list[tuple[str, str, str]] = []
        if not self._input("name").value.strip():
            findings.append(("personas-char-editor-name", "required", "error"))
        avatar_bytes = self.current_avatar_bytes()
        if avatar_bytes is not None:
            # Local import: this widget module is itself imported by
            # personas_screen at module-load time, so a top-level import of
            # its constants back here would deadlock as a circular import.
            from ...UI.Screens.personas_screen import (
                PERSONAS_AVATAR_MAX_BYTES,
                PERSONAS_AVATAR_MAX_SIZE_COPY,
            )

            if len(avatar_bytes) > PERSONAS_AVATAR_MAX_BYTES:
                findings.append(
                    (
                        "personas-char-editor-avatar-status",
                        f"image exceeds {PERSONAS_AVATAR_MAX_SIZE_COPY}",
                        "error",
                    )
                )
        for index, greeting in enumerate(self._greetings, start=1):
            if not greeting.strip():
                findings.append(
                    (
                        "personas-char-editor-greetings-table",
                        f"greeting {index} is blank",
                        "warning",
                    )
                )
        return findings

    def _validated_field_ids(self) -> set[str]:
        """Field ids ``validate()`` can flag at ``error`` level.

        Only these are reconciled (marked/un-marked) by ``_run_validation``;
        greetings are warning-only and never toggle ``.is-invalid``.
        """
        return {"personas-char-editor-name", "personas-char-editor-avatar-status"}

    def _run_validation(self) -> list[tuple[str, str, str]]:
        """Compute findings, mark/un-mark offending rows, render the footer.

        Runs debounced on field change (``_schedule_validation``, wired into
        ``_field_changed``) and directly off discrete actions (avatar
        upload/remove, greeting mutations) and authoritatively at Save
        (``_save_pressed``), which blocks when any finding is
        ``level == "error"``.

        Display is gated on ``_user_touched``: a freshly-opened form
        (``load_character``/``new_character``) must not show errors before
        the user has actually interacted with it, so while untouched no row
        is marked invalid and the footer stays clear.

        Returns:
            The findings actually rendered - empty when gated by
            ``_user_touched`` (not the raw ``validate()`` output, since
            nothing was displayed in that case).
        """
        if not self._user_touched:
            for fid in self._validated_field_ids():
                self.query_one(f"#{fid}").parent.remove_class(self._FIELD_ERROR_CLASS)
            self.show_validation(())
            return []
        findings = self.validate()
        invalid_ids = {fid for fid, _msg, level in findings if level == "error"}
        for fid in self._validated_field_ids():
            row = self.query_one(f"#{fid}").parent
            row.set_class(fid in invalid_ids, self._FIELD_ERROR_CLASS)
        self.show_validation(tuple(f"{fid}: {msg}" for fid, msg, _level in findings))
        return findings

    def _schedule_validation(self) -> None:
        """Debounce ``_run_validation`` so a burst of typing validates once."""
        if self._validation_timer is not None:
            self._validation_timer.stop()
        self._validation_timer = self.set_timer(
            self._VALIDATION_DEBOUNCE_SECONDS, self._run_validation
        )

    def show_validation(self, errors: tuple[str, ...]) -> None:
        """Render screen-side validation errors in the editor footer.

        The footer Static is the single in-editor validation surface: the
        editor's own name-required check and the screen's ``_validate_character``
        results (e.g. character_book errors) both land here, in the same
        format. An empty tuple clears it.
        """
        validation = self.query_one("#personas-char-editor-validation", Static)
        if errors:
            validation.update("Validation errors:\n" + "\n".join(errors))
        else:
            validation.update("")

    def get_character_data(self) -> Dict[str, Any]:
        """Current form values, in the legacy editor's key structure.

        Starts from the loaded record copy (preserving ``id``/``version`` and
        unedited keys) and overrides the editor-owned keys. ``first_mes`` is
        the legacy alias the save path normalizes; when the loaded record also
        carried ``first_message`` it is kept in sync so the stale loaded value
        cannot win in the DB save.
        """
        data = dict(self._character_data)
        data["name"] = self._input("name").value
        data["description"] = self._area("description").text
        data["personality"] = self._area("personality").text
        data["scenario"] = self._area("scenario").text
        first_message = self._area("first-message").text
        data["first_mes"] = first_message
        if "first_message" in data:
            data["first_message"] = first_message
        data["creator_notes"] = self._area("creator-notes").text
        data["system_prompt"] = self._area("system-prompt").text
        data["post_history_instructions"] = self._area("post-history").text
        data["creator"] = self._input("creator").value
        # Empty/whitespace Version falls back to the new_character default.
        version = self._input("version").value
        data["character_version"] = version if version.strip() else "1.0"
        # Each greeting is a discrete list entry (never blob-joined/split), so
        # this is always the exact, byte-identical list - including any
        # embedded newlines within a single greeting.
        data["alternate_greetings"] = list(self._greetings)
        data["tags"] = [
            tag.strip() for tag in self._input("tags").value.split(",") if tag.strip()
        ]
        return data

    def sync_attached_dictionaries(
        self, chat_dictionaries: list, new_version: int
    ) -> None:
        """Patch the loaded base after an out-of-band dictionary attach/detach.

        Updates only ``extensions['chat_dictionaries']`` and ``version`` on the
        base copy the Save path starts from, so an instant attach is neither
        clobbered by a later Save nor forces a version conflict — and the user's
        in-progress form edits are left untouched. No-op when no character is
        loaded (empty base).
        """
        if not self._character_data:
            return
        ext = self._character_data.get("extensions")
        if not isinstance(ext, dict):
            ext = {}
        ext["chat_dictionaries"] = list(chat_dictionaries)
        self._character_data["extensions"] = ext
        self._character_data["version"] = new_version

    def sync_attached_world_books(
        self, character_world_books: list, new_version: int
    ) -> None:
        """Patch the loaded base after an out-of-band world-book attach/detach.

        Updates only ``extensions['character_world_books']`` and ``version`` on
        the base copy the Save path starts from, so an instant attach is neither
        clobbered by a later Save nor forces a version conflict. No-op when no
        character is loaded (empty base).
        """
        if not self._character_data:
            return
        ext = self._character_data.get("extensions")
        if not isinstance(ext, dict):
            ext = {}
        ext[CHARACTER_WORLD_BOOKS_KEY] = list(character_world_books)
        self._character_data["extensions"] = ext
        self._character_data["version"] = new_version

    # ===== Internals =====

    def _form_snapshot(self) -> tuple:
        """Raw field values, for change detection (cheap, no parsing)."""
        return (
            self._input("name").value,
            self._area("first-message").text,
            self._area("description").text,
            self._area("personality").text,
            self._area("system-prompt").text,
            self._area("scenario").text,
            self._area("post-history").text,
            self._area("creator-notes").text,
            self._input("creator").value,
            self._input("version").value,
            self._input("tags").value,
            tuple(self._greetings),
        )

    def _set_advanced_open(self, open_: bool) -> None:
        """Show/hide the Advanced section and keep the toggle label in sync."""
        self.query_one("#personas-char-editor-advanced").display = open_
        self.query_one("#personas-char-editor-advanced-toggle", Button).label = (
            "Advanced ▾" if open_ else "Advanced ▸"
        )

    def _set_avatar_status_from_record(self) -> None:
        avatar = (
            "embedded"
            if (self._character_data.get("image") or self._character_data.get("avatar"))
            else "none"
        )
        self.query_one("#personas-char-editor-avatar-status", Static).update(
            f"Avatar: {avatar}"
        )

    def _mark_dirty(self) -> None:
        if self._loading or self._dirty_posted or self._loaded_snapshot is None:
            return
        self._dirty_posted = True
        self.post_message(EditorContentChanged())

    # ===== Alternate greetings (widget-local list editor) =====

    @staticmethod
    def _greeting_preview(text: str) -> str:
        """First-line, truncated preview for the greetings table row."""
        first = (text or "").splitlines()[0] if text else ""
        return (first[:60] + "…") if len(first) > 60 or "\n" in (text or "") else first

    def _render_greetings_table(self) -> None:
        table = self.query_one("#personas-char-editor-greetings-table", DataTable)
        table.clear()
        for i, greeting in enumerate(self._greetings):
            table.add_row(self._greeting_preview(greeting), key=str(i))

    def _load_greeting_into_edit(self, index: int) -> None:
        if 0 <= index < len(self._greetings):
            self.query_one(
                "#personas-char-editor-greeting-edit", TextArea
            ).text = self._greetings[index]

    def _select_greeting_row(self, index: int) -> None:
        """Move the table cursor to ``index`` after a mutation's re-render.

        ``_render_greetings_table`` clears and rebuilds the table, which
        resets DataTable's own cursor to row 0 and posts an async
        ``RowHighlighted(row=0)`` that would otherwise clobber the intended
        selection once the message queue drains. Explicitly re-issuing
        ``move_cursor`` here posts a second, later message that wins.
        """
        if 0 <= index < len(self._greetings):
            self.query_one(
                "#personas-char-editor-greetings-table", DataTable
            ).move_cursor(row=index)

    def _greetings_add(self, text: str = "") -> None:
        self._greetings.append(text)
        self._render_greetings_table()
        self._mark_dirty()
        # A discrete user action (Add button) - validate immediately, no
        # debounce, so a blank-greeting warning appears at once.
        self._user_touched = True
        self._run_validation()

    def _greetings_update(self, index: int, text: str) -> None:
        if 0 <= index < len(self._greetings):
            self._greetings[index] = text
            self._render_greetings_table()
            # Same race as Add/Move: re-render resets the DataTable cursor and
            # queues an async RowHighlighted(row=0). Re-select the row that was
            # actually updated so the edit box keeps showing it (not row 0's
            # text), and so a follow-up Update commits to the right entry.
            self._select_greeting_row(index)
            self._mark_dirty()
            self._user_touched = True
            self._run_validation()

    def _greetings_delete(self, index: int) -> None:
        if 0 <= index < len(self._greetings):
            del self._greetings[index]
            self._render_greetings_table()
            if self._greetings:
                # Select the surviving neighbor at the same position (or the
                # new last row if we deleted the tail) so the async
                # RowHighlighted(row=0) queued by the re-render above doesn't
                # silently revert the selection/edit box to row 0.
                new_index = min(index, len(self._greetings) - 1)
                self._select_greeting_row(new_index)
            else:
                # No rows left means no async RowHighlighted will ever fire to
                # clobber this, so it's safe to set the empty state directly.
                self._selected_greeting_index = None
                self.query_one(
                    "#personas-char-editor-greeting-edit", TextArea
                ).text = ""
            self._mark_dirty()
            self._user_touched = True
            self._run_validation()

    def _greetings_move(self, index: int, offset: int) -> None:
        j = index + offset
        if 0 <= index < len(self._greetings) and 0 <= j < len(self._greetings):
            self._greetings[index], self._greetings[j] = (
                self._greetings[j],
                self._greetings[index],
            )
            self._render_greetings_table()
            self._mark_dirty()
            self._user_touched = True
            self._run_validation()

    # ===== Events =====

    @on(Input.Changed)
    @on(TextArea.Changed)
    def _field_changed(self, event: Input.Changed | TextArea.Changed) -> None:
        """Announce the first real user modification of the session.

        All Inputs/TextAreas that bubble here are this editor's own fields,
        EXCEPT the alternate-greetings edit TextArea: that one is a scratch
        field for staging a single greeting's text (populated on row
        selection, committed via the Add/Update buttons which call
        ``_mark_dirty`` directly) and must not itself feed dirty detection -
        merely selecting a row would otherwise spuriously dirty the editor.

        Programmatic population also fires Changed; those events either land
        while ``_loading`` is set or (the usual case, since Textual posts them
        asynchronously) after ``load_character`` returned, where the snapshot
        comparison filters them out because the form still matches what was
        loaded. Paste and undo also fire Changed, so the comparison covers
        them too.
        """
        if (
            isinstance(event, TextArea.Changed)
            and event.text_area.id == "personas-char-editor-greeting-edit"
        ):
            return
        # Same condition _mark_dirty ultimately gates on (minus _dirty_posted,
        # which only suppresses the once-per-session announcement, not the
        # touched flag): a genuine edit, not the programmatic-population
        # Changed events load_character/new_character trigger.
        if (
            not self._loading
            and self._loaded_snapshot is not None
            and self._form_snapshot() != self._loaded_snapshot
        ):
            self._user_touched = True
        self._schedule_validation()
        if self._loading or self._dirty_posted or self._loaded_snapshot is None:
            return
        if self._form_snapshot() == self._loaded_snapshot:
            return
        self._mark_dirty()

    @on(DataTable.RowSelected, "#personas-char-editor-greetings-table")
    def _greeting_row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        if event.row_key is not None and event.row_key.value is not None:
            self._selected_greeting_index = int(event.row_key.value)
            self._load_greeting_into_edit(self._selected_greeting_index)

    @on(DataTable.RowHighlighted, "#personas-char-editor-greetings-table")
    def _greeting_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        # Arrow-key navigation only fires RowHighlighted (not RowSelected), so
        # without this the edit TextArea would silently keep stale content
        # from a prior row while _selected_greeting_index tracks the cursor.
        event.stop()
        if event.row_key is not None and event.row_key.value is not None:
            self._selected_greeting_index = int(event.row_key.value)
            self._load_greeting_into_edit(self._selected_greeting_index)

    @on(Button.Pressed, "#personas-char-editor-greeting-add")
    def _greeting_add_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        text = self.query_one("#personas-char-editor-greeting-edit", TextArea).text
        self._greetings_add(text)
        self._select_greeting_row(len(self._greetings) - 1)

    @on(Button.Pressed, "#personas-char-editor-greeting-update")
    def _greeting_update_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self._selected_greeting_index is None:
            return
        text = self.query_one("#personas-char-editor-greeting-edit", TextArea).text
        self._greetings_update(self._selected_greeting_index, text)

    @on(Button.Pressed, "#personas-char-editor-greeting-delete")
    def _greeting_delete_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self._selected_greeting_index is None:
            return
        # _greetings_delete now selects the surviving neighbor (or clears the
        # edit box if the list is empty) itself, so it must not be
        # unconditionally overwritten here afterward.
        self._greetings_delete(self._selected_greeting_index)

    @on(Button.Pressed, "#personas-char-editor-greeting-move-up")
    def _greeting_move_up_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self._selected_greeting_index is None:
            return
        target = self._selected_greeting_index - 1
        if 0 <= target < len(self._greetings):
            self._greetings_move(self._selected_greeting_index, -1)
            self._select_greeting_row(target)

    @on(Button.Pressed, "#personas-char-editor-greeting-move-down")
    def _greeting_move_down_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self._selected_greeting_index is None:
            return
        target = self._selected_greeting_index + 1
        if 0 <= target < len(self._greetings):
            self._greetings_move(self._selected_greeting_index, 1)
            self._select_greeting_row(target)

    @on(Button.Pressed, "#personas-char-editor-advanced-toggle")
    def _toggle_advanced(self, event: Button.Pressed) -> None:
        event.stop()
        self._set_advanced_open(
            not self.query_one("#personas-char-editor-advanced").display
        )

    @on(Button.Pressed, "#personas-char-editor-avatar-upload")
    def _upload_avatar_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(CharacterImageUploadRequested())

    @on(Button.Pressed, "#personas-char-editor-avatar-remove")
    def _remove_avatar_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(CharacterImageRemoveRequested())

    @staticmethod
    def _expression_state_from_button_id(button_id: str | None, *, suffix: str) -> str | None:
        """Recover the ``state`` a per-slot upload/clear button id encodes.

        Ids follow ``personas-char-editor-expr-{state}-{suffix}``; used by
        the two class-selector handlers below instead of six near-duplicate
        per-state handlers.
        """
        prefix = "personas-char-editor-expr-"
        if not button_id or not button_id.startswith(prefix) or not button_id.endswith(suffix):
            return None
        state = button_id[len(prefix) : -len(suffix)]
        return state if state in EXPRESSION_IMAGE_STATES else None

    @on(Button.Pressed, ".personas-char-editor-expr-upload")
    def _expression_upload_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        state = self._expression_state_from_button_id(event.button.id, suffix="-upload")
        if state is not None:
            self.post_message(CharacterExpressionUploadRequested(state))

    @on(Button.Pressed, ".personas-char-editor-expr-clear")
    def _expression_clear_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        state = self._expression_state_from_button_id(event.button.id, suffix="-clear")
        if state is not None:
            self.post_message(CharacterExpressionClearRequested(state))

    @on(Button.Pressed, "#personas-char-editor-expr-import")
    def _expression_set_import_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(CharacterExpressionSetImportRequested())

    @on(Button.Pressed, "#personas-char-editor-expr-export")
    def _expression_set_export_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(CharacterExpressionSetExportRequested())

    @on(Button.Pressed, "#personas-char-editor-save")
    def _save_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        # Save is itself a user action: authoritatively validate even an
        # untouched blank form (clicking Save with nothing else edited must
        # still block + mark the offending field).
        self._user_touched = True
        if any(level == "error" for _fid, _msg, level in self._run_validation()):
            return
        self.post_message(CharacterSaveRequested(self.get_character_data()))

    @on(Button.Pressed, "#personas-char-editor-cancel")
    def _cancel_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(CharacterEditorCancelled())


__all__ = ["PersonasCharacterEditorWidget"]
