"""Persona create/edit form for the Roleplay workbench."""

from __future__ import annotations

from typing import Any, Dict, Literal, get_args

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import Button, Input, Label, Select, Static, Switch, TextArea

from ...tldw_api.character_persona_schemas import PersonaMode
from .personas_pane_messages import (
    EditorContentChanged,
    PersonaProfileEditCancelled,
    PersonaProfileSaveRequested,
    VisualIdentityPackMetadata,
)
from .personas_persona_visual_pack_widget import PersonasPersonaVisualPackWidget
<<<<<<< HEAD
from .personas_visual_identity_pack_widget import PersonasVisualIdentityPackWidget
=======
from .personas_policy_rules_editor import PersonasPolicyRulesEditor
>>>>>>> d5daf64db (feat(personas): policy rules editor, switcher label, import review display)

#: The `PersonaMode` literal's values, for the editor's mode `Select` options.
PERSONA_MODES: tuple[str, ...] = get_args(PersonaMode)
#: Default mode for a persona with none set.
_DEFAULT_MODE = "session_scoped"
_LOCAL_FIELDS_NOTE = (
    "Description and personality traits are local-only and cannot be saved "
    "to the server."
)


class PersonaProfileEditorWidget(Container):
    """ds-field-row form: name, description, system prompt, personality
    traits, mode, and enabled toggle."""

    BUNDLED_CSS = """
    PersonaProfileEditorWidget {
        width: 100%;
        height: 100%;
    }

    PersonaProfileEditorWidget #personas-editor-body {
        height: 1fr;
    }

    PersonaProfileEditorWidget .ds-toolbar {
        height: 1;
        min-height: 1;
    }

    PersonaProfileEditorWidget .ds-toolbar Button {
        width: auto;
        min-width: 0;
        height: 1;
        min-height: 1;
        padding: 0 1;
        border: none;
        margin-right: 1;
    }

    /* Live per-field validation (Roleplay P3b Task 4): a literal color, not
       a $ds-* token - BUNDLED_CSS must resolve in bare-App test harnesses
       that never load the app stylesheet. */
    PersonaProfileEditorWidget .is-invalid {
        border: round red;
    }
    """

    #: CSS class toggled on an offending error-level field's enclosing row
    #: by ``_run_validation``.
    _FIELD_ERROR_CLASS = "is-invalid"
    #: Delay before a field-change-triggered validation pass runs, matching
    #: PersonasCharacterEditorWidget's debounce.
    _VALIDATION_DEBOUNCE_SECONDS = 0.2

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("id", "ccp-persona-editor-view")
        super().__init__(**kwargs)
        self._runtime_source: Literal["local", "server"] = "local"
        self._persona_id: str | None = None
        self._version: int | None = None
        # Dirty tracking (UX-E3): see PersonasCharacterEditorWidget for the
        # mechanism. The snapshot comparison (not the _loading flag) is the
        # reliable suppressor for programmatic-population Changed events,
        # which Textual delivers after load_persona returns.
        self._loading: bool = False
        self._loaded_snapshot: tuple | None = None
        self._dirty_posted: bool = False
        # Live validation (Roleplay P3b Task 4): see
        # PersonasCharacterEditorWidget._validation_timer for the mechanism.
        self._validation_timer: Timer | None = None
        # Fix-wave gate: a freshly-opened form (load_persona/new_persona)
        # must not display validation errors before the user has actually
        # interacted with it. Set True on a genuine field edit or a Save
        # click; reset on every load.
        self._user_touched: bool = False
        self._persona_visual_session_token = 0
        self._actor_pack_mode = False

    def compose(self) -> ComposeResult:
        yield Static(
            "Persona Editor",
            id="personas-editor-title",
            classes="destination-section",
        )
        with VerticalScroll(id="personas-editor-body"):
            with Vertical(id="personas-editor-pack-portrait") as pack_portrait:
                yield Label("Required portrait Character")
                yield Select(
                    [],
                    id="personas-editor-character-portrait",
                    allow_blank=True,
                    prompt="No eligible local portrait Character",
                )
                yield Static(
                    "Choose a local Character with an embedded portrait.",
                    id="personas-editor-pack-status",
                    classes="destination-purpose",
                    markup=False,
                )
            pack_portrait.display = False
            with Vertical(classes="ds-field-row"):
                yield Label("Name")
                yield Input(id="personas-editor-name", placeholder="Persona name")
            with Vertical(classes="ds-field-row"):
                yield Label("Description")
                yield TextArea(id="personas-editor-description")
            with Vertical(classes="ds-field-row"):
                yield Label("System prompt")
                yield TextArea(id="personas-editor-system-prompt")
            with Vertical(classes="ds-field-row"):
                yield Label("Personality traits")
                yield TextArea(id="personas-editor-personality-traits")
            local_fields_note = Static(
                _LOCAL_FIELDS_NOTE,
                id="personas-editor-local-fields-note",
            )
            local_fields_note.display = False
            yield local_fields_note
            with Vertical(classes="ds-field-row"):
                yield Label("Mode")
                yield Select(
                    [(m, m) for m in PERSONA_MODES],
                    id="personas-editor-mode",
                    allow_blank=False,
                    value=_DEFAULT_MODE,
                )
            with Horizontal(classes="ds-field-row"):
                yield Label("Enabled")
                yield Switch(id="personas-editor-enabled", value=True)
            yield Container(
                Static("Loading Shared Visual Identity reactions…", markup=False),
                id="personas-editor-shared-visual-identity-host",
            )
            yield Static(
                "Persona Visual operational states",
                id="personas-editor-persona-visual-title",
                classes="destination-section",
            )
            yield PersonasPersonaVisualPackWidget()
            # Task 11: narrowing-only tool policy rules live on the local
            # persona record; the section is shown/hidden by
            # _sync_runtime_source_controls (local + saved persona only).
            policy_editor = PersonasPolicyRulesEditor()
            policy_editor.display = False
            yield policy_editor
        # Validation stays outside the scroll body so it is always visible
        # next to Save (anchored-footer principle, same as the character editor).
        yield Static("", id="personas-editor-validation")
        with Horizontal(classes="ds-toolbar"):
            yield Button("Save", id="personas-editor-save")
            yield Button("Cancel", id="personas-editor-cancel")

    def on_mount(self) -> None:
        """Apply the initial local/server field availability after compose."""
        self._sync_runtime_source_controls()

    @property
    def runtime_source(self) -> Literal["local", "server"]:
        """The backend whose mutation contract this editor currently targets."""
        return self._runtime_source

    def set_runtime_source(self, runtime_source: str) -> None:
        """Set the mutation source and make local-only fields explicit.

        Args:
            runtime_source: Exactly ``"local"`` or ``"server"``.

        Raises:
            ValueError: If ``runtime_source`` is not supported.
        """
        normalized = str(runtime_source or "").strip().lower()
        if normalized not in {"local", "server"}:
            raise ValueError(f"Unsupported Persona runtime source: {runtime_source}")
        self._runtime_source = normalized
        if self.is_mounted:
            self._sync_runtime_source_controls()

    def _sync_runtime_source_controls(self) -> None:
        """Enable local extensions locally and explain their server exclusion."""
        is_server = self._runtime_source == "server"
        self.query_one("#personas-editor-description", TextArea).disabled = is_server
        self.query_one(
            "#personas-editor-personality-traits", TextArea
        ).disabled = is_server
        self.query_one("#personas-editor-local-fields-note", Static).display = is_server
        # Task 11: policy rules are a local-persona-record attribute.
        try:
            policy_editor = self.query_one(PersonasPolicyRulesEditor)
        except Exception:  # noqa: BLE001 - children not composed yet
            return
        policy_editor.display = not is_server and self._persona_id is not None
        if is_server or self._persona_id is None:
            policy_editor.clear_rules()

    def load_persona(
        self,
        data: Dict[str, Any],
        *,
        runtime_source: str | None = None,
    ) -> None:
        """Push persona data into the editor form fields.

        CCPPersonaHandler calls this method when it queries ``#ccp-persona-editor-view``
        and finds a ``load_persona`` attribute (see ccp_persona_handler._load_editor).
        """
        self._persona_visual_session_token += 1
        self._actor_pack_mode = False
        self._loading = True
        try:
            self.set_runtime_source(runtime_source or self._runtime_source)
            self._persona_id = str(data.get("id", "")) or None
            # Kept for optimistic locking: the save path passes it back as
            # ``expected_version`` (None for new personas).
            self._version = data.get("version")
            self.query_one("#personas-editor-name", Input).value = str(
                data.get("name", "")
            )
            self.query_one("#personas-editor-description", TextArea).text = str(
                data.get("description", "")
            )
            self.query_one("#personas-editor-system-prompt", TextArea).text = str(
                data.get("system_prompt", "")
            )
            self.query_one("#personas-editor-personality-traits", TextArea).text = str(
                data.get("personality_traits", "") or ""
            )
            mode = data.get("mode") or _DEFAULT_MODE
            self.query_one("#personas-editor-mode", Select).value = (
                mode if mode in PERSONA_MODES else _DEFAULT_MODE
            )
            self.query_one("#personas-editor-enabled", Switch).value = bool(
                data.get("is_active", True)
            )
            visual = self.query_one(PersonasPersonaVisualPackWidget)
            shared_host = self.query_one(
                "#personas-editor-shared-visual-identity-host", Container
            )
            if self._runtime_source == "server":
                visual.set_availability("server")
                self._set_shared_visual_identity_status(
                    shared_host, "Save a local copy first"
                )
            elif self._persona_id is None:
                visual.set_availability("unsaved")
                self._set_shared_visual_identity_status(
                    shared_host, "Save the Persona first"
                )
            else:
                visual.set_availability("loading")
<<<<<<< HEAD
                self._set_shared_visual_identity_status(
                    shared_host, "Loading Shared Visual Identity reactions…"
                )
=======
            # Task 11: policy rules ride the local persona record; the
            # record view always carries a normalized list (service views).
            policy_editor = self.query_one(PersonasPolicyRulesEditor)
            policy_local = (
                self._runtime_source == "local" and self._persona_id is not None
            )
            policy_editor.display = policy_local
            if policy_local:
                policy_editor.show_rules(data.get("policy_rules"))
            else:
                policy_editor.clear_rules()
>>>>>>> d5daf64db (feat(personas): policy rules editor, switcher label, import review display)
            self.query_one("#personas-editor-validation", Static).update("")
            # Clear any stale per-field invalid marks left by a prior session:
            # if the reopened record's values are byte-identical to what's
            # already displayed, no Changed event fires and _run_validation
            # never runs to self-heal a previously-marked row (Roleplay P3b
            # review fix).
            for fid in self._validated_field_ids():
                self.query_one(f"#{fid}").parent.remove_class(self._FIELD_ERROR_CLASS)
        finally:
            self._loading = False
        if self.is_mounted:
            selector = self.query_one("#personas-editor-character-portrait", Select)
            selector.set_options([])
        self._loaded_snapshot = self._form_snapshot()
        self._dirty_posted = False
        self._user_touched = False
        if self.is_mounted:
            self._sync_actor_pack_mode()

    def new_persona(self, *, runtime_source: str | None = None) -> None:
        """Clear the form for a new (unsaved) persona."""
        self.load_persona({}, runtime_source=runtime_source)

<<<<<<< HEAD
    def begin_actor_pack_creation(
        self, portrait_options: tuple[tuple[str, int], ...]
    ) -> None:
        """Reuse the local editor with one labelled required-portrait selector."""

        self.new_persona(runtime_source="local")
        self._actor_pack_mode = True
        selector = self.query_one("#personas-editor-character-portrait", Select)
        selector.set_options(list(portrait_options))
        selector.value = portrait_options[0][1] if portrait_options else Select.BLANK
        self._loaded_snapshot = self._form_snapshot()
        self._dirty_posted = False
        self._user_touched = False
        self._sync_actor_pack_mode()

    def mark_actor_pack_created(self, portable_uuid: str) -> None:
        """Show the committed portable UUID beside the canonical form."""

        self._actor_pack_mode = False
        self.query_one("#personas-editor-title", Static).update(
            "Persona Actor Pack created"
        )
        status = self.query_one("#personas-editor-pack-status", Static)
        status.update(f"Portable UUID: {portable_uuid}")
        self.query_one("#personas-editor-pack-portrait").display = True

    def _sync_actor_pack_mode(self) -> None:
        if not self.is_mounted:
            return
        self.query_one("#personas-editor-title", Static).update(
            "New Persona Actor Pack" if self._actor_pack_mode else "Persona Editor"
        )
        self.query_one("#personas-editor-pack-portrait").display = self._actor_pack_mode

    @staticmethod
    def _set_shared_visual_identity_status(host: Container, copy: str) -> None:
        """Replace or update the host's single path-free status line."""

        children = tuple(host.children)
        if len(children) == 1 and isinstance(children[0], Static):
            children[0].update(copy)
            return
        host.remove_children()
        host.mount(Static(copy, markup=False))

    async def show_shared_visual_identity_pack(
        self, pack: VisualIdentityPackMetadata
    ) -> PersonasVisualIdentityPackWidget:
        """Mount one local Persona Shared Visual Identity metadata browser."""

        host = self.query_one("#personas-editor-shared-visual-identity-host", Container)
        await host.remove_children()
        browser = PersonasVisualIdentityPackWidget(pack, actor_kind="persona")
        await host.mount(browser)
        return browser

    async def show_shared_visual_identity_unavailable(self) -> Static:
        """Show one path-free non-authoring Shared Visual Identity state."""

        host = self.query_one("#personas-editor-shared-visual-identity-host", Container)
        await host.remove_children()
        status = Static("Unavailable", markup=False)
        await host.mount(status)
        return status

    async def discard_shared_visual_identity_pack(self, content: Widget | None) -> None:
        """Remove only one stale Shared Visual Identity mount."""

        if content is None:
            return
        host = self.query_one("#personas-editor-shared-visual-identity-host", Container)
        if content.parent is host:
            await content.remove()
=======
    @property
    def persona_id(self) -> str | None:
        """The currently loaded persona's id, if any (Task 11 wiring)."""
        return self._persona_id

    def rebaseline_version(self, version: object) -> None:
        """Adopt a new optimistic-lock version after an out-of-band save.

        Task 11: policy-rule saves bump the record version without touching
        the form; without this the next main Save would fail the lock.
        """
        self._version = version  # type: ignore[assignment]
        self._loaded_snapshot = self._form_snapshot()
>>>>>>> d5daf64db (feat(personas): policy rules editor, switcher label, import review display)

    def mark_saved(self, record: Dict[str, Any]) -> None:
        """Re-baseline dirty state to a just-persisted persona (save-in-place).

        Adopts the saved ``id``/``version`` as the new base (so the next Save
        carries the incremented optimistic-lock version) and resets the dirty
        snapshot from the CURRENT form (which already shows the saved
        values). Does NOT repopulate the form - the user's saved edits stay
        on screen.

        Args:
            record: The just-persisted persona record (carries the
                incremented optimistic-lock ``version``).
        """
        self._persona_visual_session_token += 1
        self._persona_id = str(record.get("id", "")) or self._persona_id
        self._version = record.get("version", self._version)
        if self._runtime_source == "local" and self._persona_id is not None:
            self.query_one(PersonasPersonaVisualPackWidget).set_availability("loading")
        # Task 11: a create-save makes the record policy-editable; an
        # update-save keeps the persisted rule list authoritative on screen.
        policy_editor = self.query_one(PersonasPolicyRulesEditor)
        policy_editor.display = (
            self._runtime_source == "local" and self._persona_id is not None
        )
        if policy_editor.display:
            policy_editor.show_rules(record.get("policy_rules"))
        self._loaded_snapshot = self._form_snapshot()
        self._dirty_posted = False
        self.query_one("#personas-editor-validation", Static).update("")

    @property
    def persona_visual_session_token(self) -> int:
        """Return the identity of the currently loaded visual editor session."""

        return self._persona_visual_session_token

    def collect(self) -> Dict[str, Any]:
        """Return the current form values as a dict.

        The ``id`` and ``version`` keys are omitted when no persona has been
        loaded (new form).
        """
        data: Dict[str, Any] = {
            "name": self.query_one("#personas-editor-name", Input).value.strip(),
            "system_prompt": self.query_one(
                "#personas-editor-system-prompt", TextArea
            ).text,
            "mode": self.query_one("#personas-editor-mode", Select).value,
            "is_active": self.query_one("#personas-editor-enabled", Switch).value,
        }
        if self._runtime_source == "local":
            data["description"] = self.query_one(
                "#personas-editor-description", TextArea
            ).text
            data["personality_traits"] = self.query_one(
                "#personas-editor-personality-traits", TextArea
            ).text
        if self._actor_pack_mode:
            portrait = self.query_one(
                "#personas-editor-character-portrait", Select
            ).value
            if type(portrait) is int:
                data["character_card_id"] = portrait
        if self._persona_id is not None:
            data["id"] = self._persona_id
        if self._version is not None:
            data["version"] = self._version
        return data

    def _form_snapshot(self) -> tuple:
        """Raw field values, for change detection."""
        return (
            self.query_one("#personas-editor-name", Input).value,
            self.query_one("#personas-editor-description", TextArea).text,
            self.query_one("#personas-editor-system-prompt", TextArea).text,
            self.query_one("#personas-editor-personality-traits", TextArea).text,
            self.query_one("#personas-editor-mode", Select).value,
            self.query_one("#personas-editor-enabled", Switch).value,
            self.query_one("#personas-editor-character-portrait", Select).value,
        )

    @on(Input.Changed)
    @on(TextArea.Changed)
    @on(Select.Changed)
    @on(Switch.Changed)
    def _field_changed(
        self,
        event: Input.Changed | TextArea.Changed | Select.Changed | Switch.Changed,
    ) -> None:
        """Announce the first real user modification of the session.

        See PersonasCharacterEditorWidget._field_changed for the suppression
        mechanism (loading flag + loaded-snapshot comparison). ``Select`` and
        ``Switch`` (the mode/enabled fields) route through the same handler
        as ``Input``/``TextArea`` so a mode change or an Enabled toggle
        participates in dirty tracking identically to a text edit.
        """
        # Same condition the dirty-post below ultimately gates on (minus
        # _dirty_posted, which only suppresses the once-per-session
        # announcement, not the touched flag): a genuine edit, not the
        # programmatic-population Changed events load_persona/new_persona
        # trigger.
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
        self._dirty_posted = True
        self.post_message(EditorContentChanged())

    def validate(self) -> list[tuple[str, str, str]]:
        """Live per-field checks: name required.

        Returns:
            ``(field_id, message, level)`` tuples, ``level`` in
            ``{"error", "warning"}``.
        """
        findings: list[tuple[str, str, str]] = []
        if not self.query_one("#personas-editor-name", Input).value.strip():
            findings.append(("personas-editor-name", "required", "error"))
        if (
            self._actor_pack_mode
            and type(
                self.query_one("#personas-editor-character-portrait", Select).value
            )
            is not int
        ):
            findings.append(
                (
                    "personas-editor-character-portrait",
                    "portrait Character required",
                    "error",
                )
            )
        return findings

    def _validated_field_ids(self) -> set[str]:
        """Field ids ``validate()`` can flag at ``error`` level."""
        return {"personas-editor-name", "personas-editor-character-portrait"}

    def _run_validation(self) -> list[tuple[str, str, str]]:
        """Compute findings, mark/un-mark offending rows, render the footer.

        Runs debounced on field change (``_schedule_validation``, wired into
        ``_field_changed``) and authoritatively at Save (``_save_pressed``),
        which blocks when any finding is ``level == "error"``.

        Display is gated on ``_user_touched``: a freshly-opened form
        (``load_persona``/``new_persona``) must not show errors before the
        user has actually interacted with it, so while untouched no row is
        marked invalid and the footer stays clear.

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
        """Render validation errors in the editor footer (the single
        in-editor surface); an empty tuple clears it."""
        validation = self.query_one("#personas-editor-validation", Static)
        if errors:
            validation.update("Validation errors:\n" + "\n".join(errors))
        else:
            validation.update("")

    @on(Button.Pressed, "#personas-editor-save")
    def _save_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        # Save is itself a user action: authoritatively validate even an
        # untouched blank form (clicking Save with nothing else edited must
        # still block + mark the offending field).
        self._user_touched = True
        if any(level == "error" for _fid, _msg, level in self._run_validation()):
            return
        self.post_message(PersonaProfileSaveRequested(self.collect()))

    @on(Button.Pressed, "#personas-editor-cancel")
    def _cancel_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(PersonaProfileEditCancelled())


__all__ = ["PersonaProfileEditorWidget"]
