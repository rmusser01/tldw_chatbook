"""Briefing preset manager modal (spec #2 phase 2a, Task 3).

A preset is a named, reusable N-speaker roster (each speaker optionally bound
to a character card and a TTS voice profile), plus optional style notes and
a provider/model override -- `briefing_cast.generate_script`'s whole input
besides the briefing itself. This modal is the one surface that creates,
edits and deletes them: a compact list of existing presets down one side, an
editor for whichever preset (or blank draft) is selected down the other.

Modelled on this repo's own modal-editor idiom (`UI/stts_profile_library.py`
`TTSProfileEditorModal`/`TTSProfileDeleteModal`): compose, validate-with-
inline-error, `ModalScreen[bool]` dismiss protocol. Two differences from
that precedent, both load-bearing:

* This modal manages a *list* of records, not one immutable loaded token --
  so, unlike `TTSProfileEditorModal`, it owns its own DB writes (through
  `asyncio.to_thread`) rather than handing a draft back to a caller that
  writes it.
* Every text field is read PULL-style, at save/mutate time
  (`_sync_draft_from_form`), never push-style via `Input.Changed`/
  `Select.Changed` handlers. A `Select` fires `Changed` the moment it is
  mounted with a non-blank initial value (the "Library lesson" this plan's
  Task 4 names), and a mutable roster means the editor recomposes on every
  Add/Remove speaker and every preset switch -- reacting to that mount-time
  noise would corrupt the very state it is meant to protect. Reading values
  back only when something is about to change them sidesteps the whole
  class of bug.

**The modal never queries any other DB.** `character_options` and
`voice_options` are supplied by the screen (see
`WatchlistsCollectionsScreen._open_briefing_preset_manager`), which is the
only caller entitled to reach `chachanotes_db`/`_tts_profile_service` --
this keeps a missing service a degraded FIELD (the Select disables itself
with a tooltip explaining why) rather than a degraded modal.

`voice_profile_id` is recorded on the roster here but otherwise INERT in
phase 2a: nothing reads it yet. Phase 2b (audio synthesis) is what
consumes it.

**Write-completion owns its edit state** (review round 1). A Save/Delete's
`await` (`asyncio.to_thread`, or the confirmation dialog) is a window in
which nothing else may repoint `_editing_id` at a different preset --
`_refuse_if_write_in_flight` closes that window at the two places that
could (`_select_preset_for_edit`, `_start_new_preset`), and
`_handle_save`/`_handle_delete` additionally verify, AFTER their own
`await`, that they still own the edit target before mutating it. Both
exist: the gate is the user-facing fix (a switch mid-write is refused, not
silently lost), the post-await check is what makes a completion's own
effect correct even if some future caller reaches `_editing_id` a third
way this module does not yet anticipate.
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger
from rich.markup import escape as escape_markup
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Static

from ...Subscriptions.briefing_cast import (
    ScriptCastError,
    dump_roster,
    load_roster,
    validate_roster,
)
from ...Utils.input_validation import validate_text_input
from ...Widgets.confirmation_dialog import ConfirmationDialog

#: Shown as a speaker Select's tooltip when the screen could not supply any
#: options for it -- degrades the FIELD, never the modal (brief, Task 3).
_NO_CHARACTER_OPTIONS_COPY = (
    "No character cards available. Connect the character library to bind "
    "a speaker to one."
)
_NO_VOICE_OPTIONS_COPY = (
    "No voice profiles available. Connect TTS profiles to bind a speaker's "
    "voice."
)
#: Phase 2a records `voice_profile_id`; nothing consumes it yet -- phase 2b
#: (audio synthesis) is what reads it. Shown whenever a voice IS selectable,
#: so the choice is honestly labelled as inert rather than looking wired up.
_VOICE_INERT_COPY = (
    "Recorded, not yet used: audio synthesis (phase 2b) is what will read "
    "this voice profile."
)
#: Review round 1 (fix round 1): shown when Add/New-preset/switching the
#: selected preset is refused because a write is in flight -- see
#: `_refuse_if_write_in_flight`'s docstring for the race this closes.
_WRITE_IN_PROGRESS_COPY = "A save or delete is in progress. Try again once it finishes."

#: Max lengths for this modal's free-text preset fields (Qodo review):
#: `name`/`style_notes`/`provider`/`model` are persisted -- and
#: `style_notes`/`provider`/`model` later flow into the cast LLM prompt
#: (`briefing_cast.build_cast_prompt`) -- so a local `.strip()` alone is not
#: enough; each is run through `Utils.input_validation.validate_text_input`
#: before it reaches the DB. `name`/`provider`/`model` are short,
#: identifier-shaped fields -- matches this repo's own convention for
#: "name" inputs elsewhere (`sources_pane.py`, `opml_dialogs.py`: 255).
#: `style_notes` is free-text prose guidance, so it gets a much longer
#: bound.
_PRESET_NAME_MAX_LENGTH = 255
_PRESET_STYLE_NOTES_MAX_LENGTH = 5000
_PRESET_PROVIDER_MAX_LENGTH = 255
_PRESET_MODEL_MAX_LENGTH = 255


def _blank_speaker() -> dict[str, Any]:
    """One empty speaker row, in the exact shape `validate_roster` expects."""
    return {
        "name": "",
        "role_prompt": "",
        "character_card_id": None,
        "voice_profile_id": None,
    }


def _normalize_speaker(entry: Any) -> dict[str, Any]:
    """Coerce one stored roster entry into the editor's own speaker shape.

    Tolerant of a roster loaded from storage carrying only some of the
    fields (or, defensively, none) -- every entry ever written here came
    from `validate_roster`'s own output, but this guards a hand-edited row.
    """
    if not isinstance(entry, dict):
        return _blank_speaker()
    return {
        "name": str(entry.get("name") or ""),
        "role_prompt": str(entry.get("role_prompt") or ""),
        "character_card_id": entry.get("character_card_id"),
        "voice_profile_id": entry.get("voice_profile_id"),
    }


def _select_options_for(
    options: list[tuple[str, Any]], current: Any
) -> list[tuple[str, Any]]:
    """`options`, plus a synthetic entry so a `Select` can legally hold a
    value the option list no longer contains -- a character card deleted,
    or a voice profile removed, after this preset was last saved.

    Without this, mounting the `Select` with that stale value would either
    raise (the value is not among `_legal_values`) or silently fall back to
    blank -- and a blank read back at save time would DROP the reference an
    untouched field is supposed to preserve (the "editing preserves
    untouched fields" contract).
    """
    if current is None or any(value == current for _, value in options):
        return list(options)
    return [*options, (f"(unavailable) {current}", current)]


class BriefingPresetModal(ModalScreen[bool]):
    """List, create, edit and delete `briefing_presets` rows.

    Dismisses `True` if anything was actually persisted (an insert, an
    update, or a delete), `False` otherwise -- the caller (the screen)
    reloads its preset lists only when told something changed.

    Args:
        db: An open `SubscriptionsDB` (or test double exposing the same
            `*_briefing_preset` methods). All calls go through
            `asyncio.to_thread`.
        character_options: `[(name, character_card_id), ...]` for the
            per-speaker character Select, built by the screen. Empty when
            `chachanotes_db` is unbound.
        voice_options: `[(display_name, voice_profile_id), ...]` for the
            per-speaker voice Select, built by the screen. Empty when the
            TTS profile service is unbound.
    """

    # Escape closes, exactly as the Close button does -- the same
    # keyboard-parity rule every Watchlists modal in this module states
    # (TASK-1300).
    BINDINGS = [("escape", "close", "Close")]

    def __init__(
        self,
        db: Any,
        *,
        character_options: list[tuple[str, int]],
        voice_options: list[tuple[str, str]],
    ) -> None:
        super().__init__()
        self.db = db
        self.character_options = list(character_options)
        self.voice_options = list(voice_options)
        self._changed = False
        self._write_in_flight = False
        self._presets: list[dict[str, Any]] = []
        self._editing_id: int | None = None
        self._draft_name = ""
        self._draft_style_notes = ""
        self._draft_provider = ""
        self._draft_model = ""
        self._speakers: list[dict[str, Any]] = [_blank_speaker()]

    # --- Compose -------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Vertical(id="bpm-dialog"):
            yield Static("Briefing presets", id="bpm-title")
            with Horizontal(id="bpm-body"):
                with Vertical(id="bpm-list-column"):
                    yield Static("Presets", classes="bpm-column-heading")
                    yield Button(
                        "New preset",
                        id="bpm-new-preset",
                        compact=True,
                        variant="primary",
                    )
                    with VerticalScroll(id="bpm-preset-list"):
                        if self._presets:
                            for preset in self._presets:
                                preset_id = int(preset["id"])
                                selected = preset_id == self._editing_id
                                yield Button(
                                    Text(str(preset.get("name") or "")),
                                    id=f"bpm-preset-btn-{preset_id}",
                                    compact=True,
                                    variant="primary" if selected else "default",
                                )
                        else:
                            yield Static(
                                "No presets yet.", id="bpm-preset-list-empty"
                            )
                with VerticalScroll(id="bpm-editor-column"):
                    yield Static(
                        "Edit preset"
                        if self._editing_id is not None
                        else "New preset",
                        id="bpm-editor-heading",
                        classes="bpm-column-heading",
                    )
                    with Horizontal(classes="bpm-field"):
                        yield Label("Name", classes="bpm-field-label")
                        yield Input(
                            value=self._draft_name,
                            placeholder="Preset name",
                            id="bpm-name-input",
                        )
                    with Horizontal(classes="bpm-field"):
                        yield Label("Style notes", classes="bpm-field-label")
                        yield Input(
                            value=self._draft_style_notes,
                            placeholder="Optional style guidance",
                            id="bpm-style-input",
                        )
                    with Horizontal(classes="bpm-field"):
                        yield Label("Provider", classes="bpm-field-label")
                        yield Input(
                            value=self._draft_provider,
                            placeholder="App default",
                            id="bpm-provider-input",
                        )
                    with Horizontal(classes="bpm-field"):
                        yield Label("Model", classes="bpm-field-label")
                        yield Input(
                            value=self._draft_model,
                            placeholder="App default",
                            id="bpm-model-input",
                        )
                    yield Static("Speakers", classes="bpm-column-heading")
                    for index, speaker in enumerate(self._speakers):
                        with Horizontal(
                            classes="bpm-speaker-row",
                            id=f"bpm-speaker-row-{index}",
                        ):
                            yield Input(
                                value=speaker.get("name", ""),
                                placeholder="Speaker name",
                                id=f"bpm-speaker-name-{index}",
                                classes="bpm-speaker-name",
                            )
                            yield Input(
                                value=speaker.get("role_prompt", ""),
                                placeholder="Role prompt",
                                id=f"bpm-speaker-role-{index}",
                                classes="bpm-speaker-role",
                            )
                            character_id = speaker.get("character_card_id")
                            yield Select(
                                _select_options_for(
                                    self.character_options, character_id
                                ),
                                value=(
                                    character_id
                                    if character_id is not None
                                    else Select.NULL
                                ),
                                id=f"bpm-speaker-character-{index}",
                                classes="bpm-speaker-character",
                                disabled=not self.character_options,
                                tooltip=(
                                    _NO_CHARACTER_OPTIONS_COPY
                                    if not self.character_options
                                    else "Bind this speaker to a character card."
                                ),
                            )
                            voice_id = speaker.get("voice_profile_id")
                            yield Select(
                                _select_options_for(self.voice_options, voice_id),
                                value=(
                                    voice_id if voice_id is not None else Select.NULL
                                ),
                                id=f"bpm-speaker-voice-{index}",
                                classes="bpm-speaker-voice",
                                disabled=not self.voice_options,
                                tooltip=(
                                    _NO_VOICE_OPTIONS_COPY
                                    if not self.voice_options
                                    else _VOICE_INERT_COPY
                                ),
                            )
                            yield Button(
                                "Remove",
                                id=f"bpm-speaker-remove-{index}",
                                compact=True,
                                disabled=len(self._speakers) <= 1,
                            )
                    yield Button("Add speaker", id="bpm-add-speaker", compact=True)
                    yield Static("", id="bpm-error")
            with Horizontal(id="bpm-actions"):
                yield Button("Save", id="bpm-save", variant="primary")
                yield Button(
                    "Delete",
                    id="bpm-delete",
                    variant="error",
                    disabled=self._editing_id is None,
                )
                yield Button("Close", id="bpm-close")

    async def on_mount(self) -> None:
        await self._load_presets()
        self.query_one("#bpm-name-input", Input).focus()

    # --- Loading ---------------------------------------------------------

    async def _load_presets(self) -> None:
        """Re-read every stored preset, name-ASC (the DB's own ordering).

        Guarded on `is_attached`, not `is_mounted`: this is awaited directly
        from `on_mount`, and `is_mounted` only flips True once `on_mount`
        itself has returned (see `MessagePump._dispatch_message`'s
        `Mount`-then-`finally` ordering) -- an `is_mounted` guard here would
        skip the very first load's recompose every time. `is_attached`
        (already true once compose's children are mounted, which precedes
        `on_mount` firing) is what `Widget.recompose()` itself checks before
        doing anything, so it is both correct here and consistent with
        `refresh(recompose=True)`'s own contract.
        """
        try:
            rows = await asyncio.to_thread(self.db.list_briefing_presets)
        except Exception as exc:  # noqa: BLE001 - degrade the list, not the modal
            logger.warning(f"Failed to list briefing presets: {type(exc).__name__}")
            rows = []
        self._presets = [dict(row) for row in rows]
        if self.is_attached:
            self.refresh(recompose=True)

    # --- Draft <-> form sync ----------------------------------------------

    def _sync_draft_from_form(self) -> None:
        """Pull every current form value back into the draft attributes.

        Called immediately before any structural mutation (Add/Remove
        speaker, switching the selected preset, New preset) and before Save
        -- never in response to `Input.Changed`/`Select.Changed`. See the
        module docstring for why this is pull-, not push-, style.
        """
        if not self.is_mounted:
            return
        self._draft_name = self.query_one("#bpm-name-input", Input).value
        self._draft_style_notes = self.query_one("#bpm-style-input", Input).value
        self._draft_provider = self.query_one("#bpm-provider-input", Input).value
        self._draft_model = self.query_one("#bpm-model-input", Input).value
        for index, speaker in enumerate(self._speakers):
            name_input = self.query_one(f"#bpm-speaker-name-{index}", Input)
            role_input = self.query_one(f"#bpm-speaker-role-{index}", Input)
            character_select = self.query_one(
                f"#bpm-speaker-character-{index}", Select
            )
            voice_select = self.query_one(f"#bpm-speaker-voice-{index}", Select)
            speaker["name"] = name_input.value
            speaker["role_prompt"] = role_input.value
            character_value = character_select.value
            speaker["character_card_id"] = (
                None if character_value is Select.NULL else character_value
            )
            voice_value = voice_select.value
            speaker["voice_profile_id"] = (
                None if voice_value is Select.NULL else voice_value
            )

    # --- Preset switching / speaker rows -----------------------------------

    def _refuse_if_write_in_flight(self) -> bool:
        """Refuse to change WHICH preset is being edited mid-write.

        Review round 1, Important finding: `_handle_save`/`_handle_delete`
        capture their target id and their field values before their one
        `await`, but until this gate existed, nothing stopped
        `_select_preset_for_edit`/`_start_new_preset` from running DURING
        that `await` and repointing `_editing_id`/`_draft_*`/`_speakers` at
        a DIFFERENT preset. Concretely: save a brand-new preset ("A")
        in flight -> switch to an existing preset ("B") while the insert is
        still running -> the insert resolves and (pre-fix)
        unconditionally assigns `self._editing_id = new_id` (A's row),
        while the form still displays B's data (from the switch) -- so the
        editor now shows B's fields "over" A's id. The user's next Save
        would silently overwrite A's row with B's content.

        This refuses the switch outright rather than queueing it (the
        smaller of the brief's two options: no extra state to hold or
        replay, and the write itself finishes in well under a second on a
        local SQLite file, so "try again" costs nothing real). The
        post-await ownership check in `_handle_save`/`_handle_delete` is a
        second, independent guard for the same invariant -- see their own
        comments -- so this is belt-and-suspenders, not the only fix.
        """
        if not self._write_in_flight:
            return False
        self._show_error(_WRITE_IN_PROGRESS_COPY)
        return True

    def _start_new_preset(self) -> None:
        if self._refuse_if_write_in_flight():
            return
        self._sync_draft_from_form()
        self._editing_id = None
        self._draft_name = ""
        self._draft_style_notes = ""
        self._draft_provider = ""
        self._draft_model = ""
        self._speakers = [_blank_speaker()]
        self._clear_error()
        self.refresh(recompose=True)

    def _select_preset_for_edit(self, preset_id: int) -> None:
        if self._refuse_if_write_in_flight():
            return
        preset = next(
            (row for row in self._presets if int(row["id"]) == preset_id), None
        )
        if preset is None:
            return
        self._editing_id = preset_id
        self._draft_name = str(preset.get("name") or "")
        self._draft_style_notes = str(preset.get("style_notes") or "")
        self._draft_provider = str(preset.get("provider") or "")
        self._draft_model = str(preset.get("model") or "")
        try:
            roster = load_roster(preset["roster_json"])
        except ScriptCastError as exc:
            logger.warning(
                f"Failed to load roster for preset {preset_id}: {type(exc).__name__}"
            )
            roster = []
        self._speakers = [_normalize_speaker(entry) for entry in roster] or [
            _blank_speaker()
        ]
        self._clear_error()
        self.refresh(recompose=True)

    def _add_speaker_row(self) -> None:
        self._sync_draft_from_form()
        self._speakers.append(_blank_speaker())
        self.refresh(recompose=True)

    def _remove_speaker_row(self, index: int) -> None:
        if len(self._speakers) <= 1:
            # One-row minimum enforced -- the brief's explicit contract.
            return
        self._sync_draft_from_form()
        del self._speakers[index]
        self.refresh(recompose=True)

    # --- Error surface -----------------------------------------------------

    def _show_error(self, message: str) -> None:
        """Render `message` as plain text -- never markup-parsed.

        `validate_roster`'s `ScriptCastError` messages interpolate a
        speaker name the user typed into a plain `Input`, so this must
        render `markup=False`-safe: `Text(...)`, never
        `Text.from_markup(...)` or a bare `str` handed to `Static.update`
        (which parses Rich markup by default).
        """
        if self.is_mounted:
            self.query_one("#bpm-error", Static).update(Text(message))

    def _clear_error(self) -> None:
        if self.is_mounted:
            self.query_one("#bpm-error", Static).update("")

    # --- Save / delete -------------------------------------------------

    async def _handle_save(self) -> None:
        self._sync_draft_from_form()
        name = self._draft_name.strip()
        if not name:
            self._show_error("Enter a preset name.")
            return
        style_notes = self._draft_style_notes.strip() or None
        provider = self._draft_provider.strip() or None
        model = self._draft_model.strip() or None
        # Each persisted free-text field goes through the shared validator,
        # not just `.strip()` -- `style_notes`/`provider`/`model` later flow
        # into the cast LLM prompt (`briefing_cast.build_cast_prompt`), so
        # this is the same boundary-validation CLAUDE.md requires elsewhere.
        # Refused through the SAME inline-error surface a roster error uses
        # (`_show_error`/`#bpm-error`): markup-safe, no persistence, modal
        # stays open.
        for label, value, max_length in (
            ("Preset name", name, _PRESET_NAME_MAX_LENGTH),
            ("Style notes", style_notes, _PRESET_STYLE_NOTES_MAX_LENGTH),
            ("Provider", provider, _PRESET_PROVIDER_MAX_LENGTH),
            ("Model", model, _PRESET_MODEL_MAX_LENGTH),
        ):
            if not validate_text_input(value, max_length=max_length):
                self._show_error(
                    f"{label} must be {max_length} characters or fewer, with "
                    "no HTML/script content."
                )
                return
        try:
            roster = validate_roster(self._speakers)
        except ScriptCastError as exc:
            # The gate: a duplicate/empty speaker name never reaches the
            # DB. Rendered through `_show_error`, so the message -- which
            # names the offending speaker verbatim -- paints literally.
            self._show_error(str(exc))
            return

        roster_json = dump_roster(roster)
        # Snapshot WHICH preset this write targets, before the one `await`
        # below -- `None` means "insert a new row". `_refuse_if_write_in_flight`
        # already stops `_select_preset_for_edit`/`_start_new_preset` from
        # repointing `self._editing_id` while this is in flight, but this
        # completion still verifies it independently (review round 1: "the
        # post-await completions verify they still own the edit state
        # before mutating it") rather than trusting that gate alone -- two
        # call sites enforcing the same invariant is exactly the
        # belt-and-suspenders shape `_refuse_if_write_in_flight`'s own
        # docstring names.
        target_editing_id = self._editing_id

        if target_editing_id is None:
            new_id = await asyncio.to_thread(
                self.db.insert_briefing_preset,
                name,
                roster_json=roster_json,
                style_notes=style_notes,
                provider=provider,
                model=model,
            )
            # Only claim the new row as the active edit target if nothing
            # else has claimed a DIFFERENT one in the meantime -- otherwise
            # the next Save would silently overwrite this brand-new row
            # with whatever preset is now actually on screen.
            if self._editing_id == target_editing_id:
                self._editing_id = int(new_id)
        else:
            await asyncio.to_thread(
                self.db.update_briefing_preset,
                target_editing_id,
                name=name,
                roster_json=roster_json,
                style_notes=style_notes,
                provider=provider,
                model=model,
            )
        self._changed = True
        self._clear_error()
        await self._load_presets()

    async def _handle_delete(self) -> None:
        if self._editing_id is None:
            return
        # Snapshotted before the two `await`s below (the confirmation
        # dialog, then the delete itself) for the same reason `_handle_save`
        # snapshots `target_editing_id`: the confirmation dialog is itself a
        # NESTED modal, which already blocks clicks on this modal's own
        # preset-list buttons for that window, but `_refuse_if_write_in_flight`
        # (claimed for this whole coroutine's lifetime via `_write_in_flight`)
        # is what covers the second `await`. This local variable is the
        # completion's own independent verification, not a re-statement of
        # that gate.
        target_editing_id = self._editing_id
        preset_name = self._draft_name.strip() or "this preset"
        confirmed = await self.app.push_screen_wait(
            ConfirmationDialog(
                title="Delete preset",
                message=(
                    f'Delete the preset "{escape_markup(preset_name)}"? '
                    "This cannot be undone."
                ),
                confirm_label="Delete",
                cancel_label="Cancel",
            )
        )
        if not confirmed:
            return
        await asyncio.to_thread(self.db.delete_briefing_preset, target_editing_id)
        self._changed = True
        # Only clear the editor back to blank if it is still showing the
        # preset just deleted -- if something else is now the active edit
        # target, blanking it here would wipe out an in-progress edit of a
        # DIFFERENT preset that this delete never touched.
        if self._editing_id == target_editing_id:
            self._editing_id = None
            self._draft_name = ""
            self._draft_style_notes = ""
            self._draft_provider = ""
            self._draft_model = ""
            self._speakers = [_blank_speaker()]
        self._clear_error()
        await self._load_presets()

    # --- Dismiss protocol ----------------------------------------------

    def action_close(self) -> None:
        self.dismiss(self._changed)

    # --- Event routing ---------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        button_id = str(event.button.id or "")
        if button_id == "bpm-new-preset":
            self._start_new_preset()
        elif button_id == "bpm-add-speaker":
            self._add_speaker_row()
        elif button_id == "bpm-close":
            self.dismiss(self._changed)
        elif button_id == "bpm-save":
            self._dispatch_write(self._handle_save())
        elif button_id == "bpm-delete":
            self._dispatch_write(self._handle_delete())
        elif button_id.startswith("bpm-speaker-remove-"):
            index = int(button_id[len("bpm-speaker-remove-") :])
            self._remove_speaker_row(index)
        elif button_id.startswith("bpm-preset-btn-"):
            preset_id = int(button_id[len("bpm-preset-btn-") :])
            self._select_preset_for_edit(preset_id)

    def _dispatch_write(self, coro: Any) -> None:
        """Claim the one-write-at-a-time guard, then dispatch.

        Claimed here, before `run_worker` -- not inside the coroutine body
        -- for the same reason the screen's own `_briefing_in_flight` is
        claimed at dispatch time (see
        `handle_generate_briefing_requested`'s docstring): a check made
        inside the worker leaves a window where two presses both pass.
        """
        if self._write_in_flight:
            coro.close()
            return
        self._write_in_flight = True
        self.run_worker(
            self._run_write(coro), exclusive=True, group="bpm-write"
        )

    async def _run_write(self, coro: Any) -> None:
        try:
            await coro
        finally:
            self._write_in_flight = False
