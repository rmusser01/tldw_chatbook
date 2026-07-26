"""Character editor generation affordances: buttons, context mode, preview.

The editor never calls a provider itself -- the screen owns that -- so these
tests drive the widget's own contract: which fields offer generation, which
context mode is active, and that a generated result is previewed rather than
written straight over the author's text.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, TextArea

from tldw_chatbook.Character_Chat.character_generation import GENERATABLE_FIELDS
from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import (
    PersonasCharacterEditorWidget,
)

pytestmark = pytest.mark.asyncio


class _EditorHost(App):
    def compose(self) -> ComposeResult:
        yield PersonasCharacterEditorWidget()


async def test_every_generatable_field_offers_a_generate_button():
    """A field the contract can generate must be generatable from the UI."""
    app = _EditorHost()
    async with app.run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        editor = app.query_one(PersonasCharacterEditorWidget)

        for field in GENERATABLE_FIELDS:
            assert editor.generate_button_id(field), f"no button id for {field}"
            app.query_one(f"#{editor.generate_button_id(field)}", Button)


async def test_context_mode_defaults_to_whole_character_and_toggles():
    """The active context mode must be visible and switchable, not hidden."""
    app = _EditorHost()
    async with app.run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        editor = app.query_one(PersonasCharacterEditorWidget)

        assert editor.generation_context_mode == "whole_character"
        toggle = app.query_one("#personas-char-editor-generate-context", Button)
        assert "whole character" in str(toggle.label).lower()

        await pilot.click("#personas-char-editor-generate-context")
        await pilot.pause()

        assert editor.generation_context_mode == "field_and_description"
        assert "description" in str(toggle.label).lower()


async def test_preview_does_not_overwrite_the_authors_text():
    """Generated text is previewed; the field keeps the author's value."""
    app = _EditorHost()
    async with app.run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        editor = app.query_one(PersonasCharacterEditorWidget)
        editor._area("description").text = "hand-written"

        editor.show_generation_preview("description", "generated text")
        await pilot.pause()

        assert editor._area("description").text == "hand-written"
        assert "generated text" in editor.generation_preview_text


async def test_accepting_a_preview_writes_the_field_and_clears_the_preview():
    app = _EditorHost()
    async with app.run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        editor = app.query_one(PersonasCharacterEditorWidget)
        editor._area("description").text = "hand-written"
        editor.show_generation_preview("description", "generated text")
        await pilot.pause()

        await pilot.click("#personas-char-editor-generate-accept")
        await pilot.pause()

        assert editor._area("description").text == "generated text"
        assert editor.pending_generation_field is None


async def test_discarding_a_preview_leaves_the_field_untouched():
    app = _EditorHost()
    async with app.run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        editor = app.query_one(PersonasCharacterEditorWidget)
        editor._area("description").text = "hand-written"
        editor.show_generation_preview("description", "generated text")
        await pilot.pause()

        await pilot.click("#personas-char-editor-generate-discard")
        await pilot.pause()

        assert editor._area("description").text == "hand-written"
        assert editor.pending_generation_field is None


async def test_accepting_marks_the_editor_dirty():
    """An accepted generation is an edit; Save must not be skippable."""
    app = _EditorHost()
    async with app.run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        editor = app.query_one(PersonasCharacterEditorWidget)
        editor._user_touched = False
        editor.show_generation_preview("description", "generated text")
        await pilot.pause()

        await pilot.click("#personas-char-editor-generate-accept")
        await pilot.pause()

        assert editor._user_touched is True


async def test_preview_is_hidden_until_a_generation_arrives():
    app = _EditorHost()
    async with app.run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        editor = app.query_one(PersonasCharacterEditorWidget)

        panel = app.query_one("#personas-char-editor-generate-preview")
        assert panel.display is False
        assert editor.pending_generation_field is None


async def test_name_field_is_not_generatable_per_field():
    """Name is a one-liner the author owns; it comes from whole-character gen."""
    assert "name" not in GENERATABLE_FIELDS


async def test_concept_input_never_schedules_validation():
    """The concept box is scratch and must not trigger re-validation.

    `_field_changed` calls `_schedule_validation()` for every Input in this
    editor. On a freshly-opened form that debounced run renders nothing (the
    `_user_touched` gate), so it CLEARED the validation footer a blocked save
    had just written -- which broke two save/validation regressions in
    test_personas_workbench.py that pass in isolation and only fail once
    enough editor churn accumulates. The greetings edit area is excluded from
    `_field_changed` for the same reason.

    Asserted on the handler directly: mounting the editor posts Changed events
    for its other Inputs, so a timing-based test would measure that churn
    rather than the concept box.
    """
    from textual.widgets import Input

    app = _EditorHost()
    async with app.run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        editor = app.query_one(PersonasCharacterEditorWidget)
        concept = editor.query_one("#personas-char-editor-concept", Input)

        scheduled: list[int] = []
        editor._schedule_validation = lambda: scheduled.append(1)

        editor._field_changed(Input.Changed(concept, "a concept"))

        assert scheduled == []


async def test_a_real_character_field_still_schedules_validation():
    """The exclusion must be surgical: real fields still validate."""
    from textual.widgets import Input

    app = _EditorHost()
    async with app.run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        editor = app.query_one(PersonasCharacterEditorWidget)
        name = editor.query_one("#personas-char-editor-name", Input)

        scheduled: list[int] = []
        editor._schedule_validation = lambda: scheduled.append(1)

        editor._field_changed(Input.Changed(name, "Seraphina"))

        assert scheduled == [1]
