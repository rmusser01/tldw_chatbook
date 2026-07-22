"""Roleplay P3b Task 4: live per-field validation (character editor).

``validate()`` returns ``(field_id, message, level)`` tuples (``level`` in
``{"error", "warning"}``). ``_run_validation()`` runs debounced off
``_field_changed`` (cancelling the prior timer) and authoritatively at Save,
toggling ``.is-invalid`` on each offending error-level field's enclosing row
and rendering the aggregated messages into the existing footer ``Static``.
Warnings never block Save nor mark a row invalid - only errors do.

Mirrors ``Tests/UI/test_personas_character_editor_avatar.py``'s bare-
``PersonasCharacterEditorWidget`` host harness and
``Tests/UI/test_personas_workbench.py``'s debounce-settle convention
(``pilot.pause(DEBOUNCE + margin)``).
"""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Static

from tldw_chatbook.UI.Screens.personas_screen import PERSONAS_AVATAR_MAX_BYTES
from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import (
    PersonasCharacterEditorWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
    CharacterSaveRequested,
)

pytestmark = pytest.mark.asyncio

_DEBOUNCE = PersonasCharacterEditorWidget._VALIDATION_DEBOUNCE_SECONDS


class _Host(App):
    def __init__(self):
        super().__init__()
        self.saves = []

    def compose(self) -> ComposeResult:
        yield PersonasCharacterEditorWidget()

    def on_character_save_requested(self, message: CharacterSaveRequested) -> None:
        self.saves.append(message.character_data)


async def _settle(pilot):
    """Wait past the validation debounce interval."""
    await pilot.pause(_DEBOUNCE + 0.05)


async def test_blank_name_marks_field_and_blocks_save():
    app = _Host()
    async with app.run_test() as pilot:
        ed = app.query_one(PersonasCharacterEditorWidget)
        ed.load_character({"name": "A"})
        await pilot.pause()

        ed.query_one("#personas-char-editor-name", Input).value = ""
        await pilot.pause()  # not yet - proves this is debounced, not synchronous
        row = ed.query_one("#personas-char-editor-name").parent
        assert not row.has_class("is-invalid")

        await _settle(pilot)
        assert row.has_class("is-invalid")

        # Save is blocked: no CharacterSaveRequested posted.
        await pilot.click("#personas-char-editor-save")
        await pilot.pause()
        assert app.saves == []

        # Fix it -> class clears live, and Save now posts.
        ed.query_one("#personas-char-editor-name", Input).value = "A"
        await _settle(pilot)
        assert not row.has_class("is-invalid")

        await pilot.click("#personas-char-editor-save")
        await pilot.pause()
        assert len(app.saves) == 1
        assert app.saves[0]["name"] == "A"


async def test_validate_returns_error_tuple_for_blank_name():
    app = _Host()
    async with app.run_test() as pilot:
        ed = app.query_one(PersonasCharacterEditorWidget)
        ed.load_character({"name": ""})
        await pilot.pause()
        findings = ed.validate()
        assert ("personas-char-editor-name", "required", "error") in findings


async def test_oversized_avatar_marks_field_and_blocks_save():
    """Fix A (review wave): set_avatar_image must trigger validation itself,
    directly and synchronously - not rely on a leftover debounce timer from
    the preceding load_character call. Proven by fully settling the
    load-triggered debounce FIRST (so no timer is left pending), then
    asserting the row is marked invalid after a single pilot.pause() with NO
    further debounce wait following set_avatar_image."""
    app = _Host()
    async with app.run_test() as pilot:
        ed = app.query_one(PersonasCharacterEditorWidget)
        ed.load_character({"name": "A"})
        await pilot.pause()
        await _settle(pilot)  # drain any load-triggered debounce fully

        row = ed.query_one("#personas-char-editor-avatar-status").parent
        assert not row.has_class("is-invalid")

        ed.set_avatar_image(b"x" * (PERSONAS_AVATAR_MAX_BYTES + 1))
        # No settle here: set_avatar_image is a discrete user action that
        # validates directly (no debounce) - a single pause lets the
        # class-toggle + footer render land.
        await pilot.pause()

        findings = ed.validate()
        assert any(
            fid == "personas-char-editor-avatar-status" and level == "error"
            for fid, _msg, level in findings
        )
        assert row.has_class("is-invalid")

        await pilot.click("#personas-char-editor-save")
        await pilot.pause()
        assert app.saves == []


async def test_avatar_within_limit_does_not_mark_field():
    app = _Host()
    async with app.run_test() as pilot:
        ed = app.query_one(PersonasCharacterEditorWidget)
        ed.load_character({"name": "A"})
        await pilot.pause()

        ed.set_avatar_image(b"x" * 16)
        await _settle(pilot)

        row = ed.query_one("#personas-char-editor-avatar-status").parent
        assert not row.has_class("is-invalid")

        await pilot.click("#personas-char-editor-save")
        await pilot.pause()
        assert len(app.saves) == 1


async def test_whitespace_greeting_is_warning_and_does_not_block_save():
    app = _Host()
    async with app.run_test() as pilot:
        ed = app.query_one(PersonasCharacterEditorWidget)
        ed.load_character({"name": "A"})
        await pilot.pause()

        ed._greetings_add("   ")
        await pilot.pause()

        findings = ed.validate()
        assert (
            "personas-char-editor-greetings-table",
            "greeting 1 is blank",
            "warning",
        ) in findings
        assert not any(level == "error" for _fid, _msg, level in findings)

        await pilot.click("#personas-char-editor-save")
        await pilot.pause()
        assert len(app.saves) == 1

        validation = ed.query_one("#personas-char-editor-validation", Static)
        assert "greeting 1 is blank" in str(validation.renderable)


async def test_validated_field_ids_covers_name_and_avatar():
    app = _Host()
    async with app.run_test() as pilot:
        ed = app.query_one(PersonasCharacterEditorWidget)
        ids = ed._validated_field_ids()
        assert "personas-char-editor-name" in ids
        assert "personas-char-editor-avatar-status" in ids


async def test_blank_new_form_does_not_mark_name_invalid_before_interaction():
    """Fix B (review wave): a freshly-opened blank form must not display
    validation errors before the user has touched anything, even after the
    debounce interval elapses (the async Changed events fired by
    new_character's programmatic population must not surface an error)."""
    app = _Host()
    async with app.run_test() as pilot:
        ed = app.query_one(PersonasCharacterEditorWidget)
        ed.new_character()
        await pilot.pause()
        await _settle(pilot)  # let any load-triggered debounce fire

        row = ed.query_one("#personas-char-editor-name").parent
        assert not row.has_class("is-invalid")
        validation = ed.query_one("#personas-char-editor-validation", Static)
        assert str(validation.renderable) == ""

        # Now a genuine interaction: type then clear -> touched, now invalid.
        name_input = ed.query_one("#personas-char-editor-name", Input)
        name_input.value = "X"
        await pilot.pause()
        name_input.value = ""
        await _settle(pilot)
        assert row.has_class("is-invalid")
        assert "personas-char-editor-name: required" in str(validation.renderable)


async def test_greeting_add_blank_warns_immediately_without_save():
    """Fix A (review wave): _greetings_add validates directly (no debounce,
    no Save needed) - the warning appears in the footer right after the Add
    action, proving the mutator triggers validation itself."""
    app = _Host()
    async with app.run_test() as pilot:
        ed = app.query_one(PersonasCharacterEditorWidget)
        ed.load_character({"name": "A"})
        await pilot.pause()
        await _settle(pilot)  # drain any load-triggered debounce fully

        ed._greetings_add("   ")
        # No debounce wait: the mutator must validate synchronously.
        await pilot.pause()

        validation = ed.query_one("#personas-char-editor-validation", Static)
        assert "greeting 1 is blank" in str(validation.renderable)

        findings = ed.validate()
        assert not any(level == "error" for _fid, _msg, level in findings)

        # Warning never blocks Save.
        await pilot.click("#personas-char-editor-save")
        await pilot.pause()
        assert len(app.saves) == 1
