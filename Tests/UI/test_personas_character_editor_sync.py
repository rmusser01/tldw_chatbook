"""P1f: an out-of-band attach patches the editor base without a clobber."""

import pytest
from textual.app import App, ComposeResult

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp

from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import (
    PersonasCharacterEditorWidget,
)

pytestmark = pytest.mark.asyncio


class _Host(ConsolidatedCSSApp):
    def compose(self) -> ComposeResult:
        yield PersonasCharacterEditorWidget()


async def test_sync_patches_base_and_survives_get_character_data():
    async with _Host().run_test(size=(120, 40)) as pilot:
        editor = pilot.app.query_one(PersonasCharacterEditorWidget)
        editor.load_character({"id": 5, "name": "Noir", "version": 1, "extensions": {}})
        await pilot.pause()

        editor.sync_attached_dictionaries(
            [{"name": "Slang", "enabled": True, "entries": []}], new_version=2
        )
        data = editor.get_character_data()
        assert data["version"] == 2
        assert data["extensions"]["chat_dictionaries"][0]["name"] == "Slang"


async def test_sync_is_noop_without_a_loaded_character():
    async with _Host().run_test(size=(120, 40)) as pilot:
        editor = pilot.app.query_one(PersonasCharacterEditorWidget)
        # no load_character
        editor.sync_attached_dictionaries([{"name": "X", "entries": []}], new_version=9)
        assert editor._character_data == {}


async def test_discard_restores_raw_baseline_without_reverting_synced_metadata():
    async with _Host().run_test(size=(120, 40)) as pilot:
        editor = pilot.app.query_one(PersonasCharacterEditorWidget)
        record = {
            "id": 5,
            "name": "Saved name",
            "first_message": "Saved greeting",
            "description": "Saved description",
            "personality": "Saved personality",
            "system_prompt": "Saved system",
            "scenario": "Saved scenario",
            "post_history_instructions": "Saved history",
            "creator_notes": "Saved notes",
            "creator": "Saved creator",
            "version": 1,
            "tags": ["one", "two"],
            "alternate_greetings": ["First\nline", "Second"],
            "image": b"saved avatar",
        }
        editor.load_character(record)
        # Save-in-place snapshots the raw form, including unnormalized tags.
        editor._input("tags").value = "one,  two ,"
        editor.mark_saved(record)
        raw_before = editor._form_snapshot()
        editor.sync_attached_dictionaries(
            [{"name": "Slang", "enabled": True, "entries": []}], new_version=2
        )
        before = editor.get_character_data()
        for name in ("name", "creator", "version", "tags"):
            editor._input(name).value = "Discarded"
        for name in (
            "first-message",
            "description",
            "personality",
            "system-prompt",
            "scenario",
            "post-history",
            "creator-notes",
        ):
            editor._area(name).text = "Discarded"
        editor._greetings_add("Discarded greeting")
        editor.set_avatar_image(b"discarded avatar")
        await pilot.pause()

        editor.discard_unsaved_form()
        await pilot.pause()

        assert editor._form_snapshot() == raw_before
        assert editor.get_character_data() == before
        assert editor.get_character_data()["version"] == 2
        assert not editor.has_unsaved_attachment()
        assert not editor._dirty_posted
