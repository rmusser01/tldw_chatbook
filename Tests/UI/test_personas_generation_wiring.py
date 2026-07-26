"""Screen-level wiring for character generation.

The editor widget owns the buttons and the preview; the screen owns running a
generation against a provider. These tests substitute the controller so the
wiring (which field, which context mode, where the result lands, what happens
on failure) is exercised without a provider.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from textual.widgets import Input

import tldw_chatbook.config as config_module
from tldw_chatbook.Character_Chat.character_generation import (
    CharacterGenerationError,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import (
    PersonasCharacterEditorWidget,
)

from Tests.UI.test_personas_dictionaries import PersonasTestApp
from Tests.UI.test_personas_library_scale import scaled_db  # noqa: F401  (fixture)

pytestmark = pytest.mark.asyncio


class _FakeController:
    """Records generation calls and returns a canned result."""

    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.field_calls: list[tuple[str, str]] = []
        self.whole_calls: list[str] = []

    async def generate_field(
        self, field, record, *, context_mode, instruction=None, on_chunk=None
    ):
        self.field_calls.append((field, context_mode))
        if self.error is not None:
            raise self.error
        if on_chunk is not None:
            # Mirror the real controller: partial text is relayed as it arrives.
            on_chunk(self.result)
        return self.result

    async def generate_whole_character(self, concept):
        self.whole_calls.append(concept)
        if self.error is not None:
            raise self.error
        return self.result


@asynccontextmanager
async def _editor(mock_app_instance, db, monkeypatch, controller):
    monkeypatch.setattr(config_module, "get_chachanotes_db_lazy", lambda: db)
    mock_app_instance.chachanotes_db = db
    mock_app_instance.chat_dictionary_scope_service = None
    app = PersonasTestApp(mock_app_instance)
    async with app.run_test(size=(160, 60)) as pilot:
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        screen = pilot.app.screen
        screen._generation_controller_override = controller
        # Open the editor the way a user does, so the wiring under test is the
        # real one rather than a hand-built widget.
        await pilot.click("#personas-library-new")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        yield pilot, screen, screen.query_one(PersonasCharacterEditorWidget)


async def test_generate_button_previews_the_result(
    mock_app_instance, scaled_db, monkeypatch
):
    """Pressing Generate must put the result in the preview, not the field."""
    controller = _FakeController(result="A guarded archivist.")
    async with _editor(mock_app_instance, scaled_db, monkeypatch, controller) as (
        pilot,
        screen,
        editor,
    ):
        editor._area("description").text = "hand-written"

        await pilot.click("#personas-char-editor-generate-description")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert controller.field_calls == [("description", "whole_character")]
        assert editor.pending_generation_field == "description"
        assert "A guarded archivist." in editor.generation_preview_text
        assert editor._area("description").text == "hand-written"


async def test_generate_uses_the_editors_selected_context_mode(
    mock_app_instance, scaled_db, monkeypatch
):
    """The visible toggle must actually change what the screen requests."""
    controller = _FakeController(result="text")
    async with _editor(mock_app_instance, scaled_db, monkeypatch, controller) as (
        pilot,
        screen,
        editor,
    ):
        await pilot.click("#personas-char-editor-generate-context")
        await pilot.pause()
        # Scenario lives in the Advanced section; open it the way a user would.
        await pilot.click("#personas-char-editor-advanced-toggle")
        await pilot.pause()

        await pilot.click("#personas-char-editor-generate-scenario")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert controller.field_calls == [("scenario", "field_and_description")]


async def test_generate_sends_the_live_editor_values_not_the_saved_record(
    mock_app_instance, scaled_db, monkeypatch
):
    """Unsaved edits must inform generation; otherwise context is stale."""
    captured: dict = {}

    class _CapturingController(_FakeController):
        async def generate_field(
            self, field, record, *, context_mode, instruction=None, on_chunk=None
        ):
            captured.update(dict(record))
            return "text"

    controller = _CapturingController(result="text")
    async with _editor(mock_app_instance, scaled_db, monkeypatch, controller) as (
        pilot,
        screen,
        editor,
    ):
        editor._area("description").text = "a drowned library"

        await pilot.click("#personas-char-editor-generate-personality")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert captured.get("description") == "a drowned library"


async def test_generation_failure_notifies_and_leaves_no_preview(
    mock_app_instance, scaled_db, monkeypatch
):
    """A provider failure must be reported, not swallowed into a blank preview."""
    controller = _FakeController(error=CharacterGenerationError("connection refused"))
    notices: list[tuple[str, str]] = []

    async with _editor(mock_app_instance, scaled_db, monkeypatch, controller) as (
        pilot,
        screen,
        editor,
    ):
        screen._notify = lambda message, severity="information": notices.append(
            (str(message), severity)
        )

        await pilot.click("#personas-char-editor-generate-description")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert editor.pending_generation_field is None
        assert any("connection refused" in message for message, _ in notices)


async def test_regenerate_reruns_the_pending_field(
    mock_app_instance, scaled_db, monkeypatch
):
    """Regenerate must retry the field being previewed, not start over."""
    controller = _FakeController(result="take one")
    async with _editor(mock_app_instance, scaled_db, monkeypatch, controller) as (
        pilot,
        screen,
        editor,
    ):
        await pilot.click("#personas-char-editor-advanced-toggle")
        await pilot.pause()
        await pilot.click("#personas-char-editor-generate-scenario")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        controller.result = "take two"
        await pilot.click("#personas-char-editor-generate-regenerate")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert [field for field, _ in controller.field_calls] == [
            "scenario",
            "scenario",
        ]
        assert "take two" in editor.generation_preview_text


# --- Whole-character generation from a concept -------------------------------


async def test_whole_character_button_reveals_a_concept_input(
    mock_app_instance, scaled_db, monkeypatch
):
    """The concept prompt is asked for inline, not assumed."""
    controller = _FakeController(result={})
    async with _editor(mock_app_instance, scaled_db, monkeypatch, controller) as (
        pilot,
        screen,
        editor,
    ):
        row = editor.query_one("#personas-char-editor-concept-row")
        assert row.display is False

        await pilot.click("#personas-char-editor-generate-whole")
        await pilot.pause()

        assert row.display is True
        assert controller.whole_calls == []


async def test_whole_character_fills_empty_fields(
    mock_app_instance, scaled_db, monkeypatch
):
    controller = _FakeController(
        result={
            "name": "Brannock",
            "description": "A retired lighthouse keeper.",
            "personality": "Weathered and wry.",
            "first_message": "*He looks up from a salt-stained chart.* You're late.",
        }
    )
    async with _editor(mock_app_instance, scaled_db, monkeypatch, controller) as (
        pilot,
        screen,
        editor,
    ):
        await pilot.click("#personas-char-editor-generate-whole")
        await pilot.pause()
        editor.query_one("#personas-char-editor-concept", Input).value = "a map trader"

        await pilot.click("#personas-char-editor-concept-run")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert controller.whole_calls == ["a map trader"]
        assert editor.query_one("#personas-char-editor-name", Input).value == "Brannock"
        assert editor._area("description").text == "A retired lighthouse keeper."
        assert "You're late." in editor._area("first-message").text


async def test_whole_character_does_not_clobber_fields_the_author_wrote(
    mock_app_instance, scaled_db, monkeypatch
):
    """Filling a blank card must not overwrite work already done."""
    controller = _FakeController(
        result={"name": "Generated", "description": "generated description"}
    )
    async with _editor(mock_app_instance, scaled_db, monkeypatch, controller) as (
        pilot,
        screen,
        editor,
    ):
        editor.query_one("#personas-char-editor-name", Input).value = "Mine"
        editor._area("description").text = "my own description"

        await pilot.click("#personas-char-editor-generate-whole")
        await pilot.pause()
        editor.query_one("#personas-char-editor-concept", Input).value = "anything"
        await pilot.click("#personas-char-editor-concept-run")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert editor.query_one("#personas-char-editor-name", Input).value == "Mine"
        assert editor._area("description").text == "my own description"


async def test_whole_character_requires_a_concept(
    mock_app_instance, scaled_db, monkeypatch
):
    """Running with an empty concept must not call the provider."""
    controller = _FakeController(result={})
    notices: list[str] = []
    async with _editor(mock_app_instance, scaled_db, monkeypatch, controller) as (
        pilot,
        screen,
        editor,
    ):
        screen._notify = lambda message, severity="information": notices.append(
            str(message)
        )
        await pilot.click("#personas-char-editor-generate-whole")
        await pilot.pause()

        await pilot.click("#personas-char-editor-concept-run")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert controller.whole_calls == []
        assert notices
