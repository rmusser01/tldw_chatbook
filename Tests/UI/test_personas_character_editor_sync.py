"""P1f: an out-of-band attach patches the editor base without a clobber."""

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import (
    PersonasCharacterEditorWidget,
)

pytestmark = pytest.mark.asyncio


class _Host(App):
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
