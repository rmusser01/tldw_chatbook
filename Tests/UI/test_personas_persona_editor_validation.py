"""Roleplay P3b Task 4: live per-field validation (persona editor).

Mirrors ``Tests/UI/test_personas_character_editor_validation.py`` for the
persona profile editor: ``validate()`` now returns typed ``(field_id,
message, level)`` tuples instead of bare error strings, ``_run_validation()``
debounces off ``_field_changed`` and runs authoritatively at Save, and the
name row's ``.is-invalid`` class tracks live as the field is fixed.
"""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Input, Static

from tldw_chatbook.Widgets.Persona_Widgets.persona_profile_editor_widget import (
    PersonaProfileEditorWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
    PersonaProfileSaveRequested,
)

pytestmark = pytest.mark.asyncio

_DEBOUNCE = PersonaProfileEditorWidget._VALIDATION_DEBOUNCE_SECONDS


class _Host(App):
    def __init__(self):
        super().__init__()
        self.saves = []

    def compose(self) -> ComposeResult:
        yield PersonaProfileEditorWidget()

    def on_persona_profile_save_requested(
        self, message: PersonaProfileSaveRequested
    ) -> None:
        self.saves.append(message.data)


async def _settle(pilot):
    """Wait past the validation debounce interval."""
    await pilot.pause(_DEBOUNCE + 0.05)


async def test_blank_name_marks_field_and_blocks_save():
    app = _Host()
    async with app.run_test() as pilot:
        ed = app.query_one(PersonaProfileEditorWidget)
        ed.load_persona({"name": "A"})
        await pilot.pause()

        ed.query_one("#personas-editor-name", Input).value = ""
        await pilot.pause()  # not yet - proves this is debounced
        row = ed.query_one("#personas-editor-name").parent
        assert not row.has_class("is-invalid")

        await _settle(pilot)
        assert row.has_class("is-invalid")

        await pilot.click("#personas-editor-save")
        await pilot.pause()
        assert app.saves == []

        ed.query_one("#personas-editor-name", Input).value = "A"
        await _settle(pilot)
        assert not row.has_class("is-invalid")

        await pilot.click("#personas-editor-save")
        await pilot.pause()
        assert len(app.saves) == 1
        assert app.saves[0]["name"] == "A"


async def test_validate_returns_typed_findings():
    app = _Host()
    async with app.run_test() as pilot:
        ed = app.query_one(PersonaProfileEditorWidget)
        ed.load_persona({"name": ""})
        await pilot.pause()
        findings = ed.validate()
        assert findings == [("personas-editor-name", "required", "error")]


async def test_footer_still_shows_name_required_substring():
    """Backward-compat: the existing 'name: required' substring assertion in
    test_persona_profile_widgets.py must keep matching the new
    '<field_id>: <message>' rendering."""
    app = _Host()
    async with app.run_test() as pilot:
        ed = app.query_one(PersonaProfileEditorWidget)
        ed.new_persona()
        await pilot.pause()
        await pilot.click("#personas-editor-save")
        await pilot.pause()
        validation = ed.query_one("#personas-editor-validation", Static)
        assert "name: required" in str(validation.renderable)
    assert app.saves == []


async def test_validated_field_ids_covers_name():
    app = _Host()
    async with app.run_test() as pilot:
        ed = app.query_one(PersonaProfileEditorWidget)
        assert "personas-editor-name" in ed._validated_field_ids()


async def test_blank_new_form_does_not_mark_name_invalid_before_interaction():
    """Fix B (review wave): a freshly-opened blank form must not display
    validation errors before the user has touched anything, even after the
    debounce interval elapses (the async Changed events fired by
    new_persona's programmatic population must not surface an error)."""
    app = _Host()
    async with app.run_test() as pilot:
        ed = app.query_one(PersonaProfileEditorWidget)
        ed.new_persona()
        await pilot.pause()
        await _settle(pilot)  # let any load-triggered debounce fire

        row = ed.query_one("#personas-editor-name").parent
        assert not row.has_class("is-invalid")
        validation = ed.query_one("#personas-editor-validation", Static)
        assert str(validation.renderable) == ""

        # Now a genuine interaction: type then clear -> touched, now invalid.
        name_input = ed.query_one("#personas-editor-name", Input)
        name_input.value = "X"
        await pilot.pause()
        name_input.value = ""
        await _settle(pilot)
        assert row.has_class("is-invalid")
        assert "personas-editor-name: required" in str(validation.renderable)
