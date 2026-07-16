"""Roleplay P1f Task 8: screen wiring for character-dictionary attach/detach.

Mirrors the P1e conversation-attach flow (``TestDictionaryAttachFlow`` in
``test_personas_dictionaries.py``) but targets a *character* selection and
the Task 6 ``PersonasCharacterDictionariesWidget`` panel instead of the
dictionary detail's Attachments tab.

Two harness pieces from ``test_personas_dictionaries.py`` are deliberately
NOT reused verbatim:

- ``stub_characters`` there stubs an EMPTY character library (it exists only
  so Characters-mode code paths don't explode while dictionary tests select
  the Dictionaries mode). This suite needs a real, selectable character, so
  it defines its own ``stub_characters`` mirroring the one already proven in
  ``test_personas_workbench.py``.
- ``fake_dict_service`` there builds a plain ``FakeDictScopeService`` with no
  character-attach surface. This suite subclasses it to add the in-memory
  character store plus ``list_character_dictionaries`` /
  ``attach_to_character`` / ``detach_from_character``, matching the real
  ``ChatDictionaryScopeService`` shapes.
"""

import pytest

import tldw_chatbook.UI.CCP_Modules.ccp_character_handler as character_handler_module
from Tests.UI.test_personas_dictionaries import (
    FakeDictScopeService,
    PersonasTestApp,
    make_dict_record,
)

pytestmark = pytest.mark.asyncio

CHARACTERS = [
    {
        "id": 1,
        "name": "Test Character",
        "description": "A stub character for the dictionary-attach flow",
        "first_message": "Hello.",
        "version": 1,
    },
]


class CharAttachFakeDictScopeService(FakeDictScopeService):
    """Extends the P1a fake with the P1f character-dictionary attach surface.

    ``self.characters`` mirrors the shape a real character card's
    ``extensions['chat_dictionaries']`` takes: ``{char_id: {"extensions":
    {"chat_dictionaries": [...]}, "version": N}}``. ``self._names`` maps a
    dictionary id to its name, standing in for a real dictionary-id -> name
    lookup (``attach_to_character`` takes an id but the embedded block is
    keyed by name).
    """

    def __init__(self, records: list[dict] | None = None) -> None:
        super().__init__(records)
        self.characters: dict[int, dict] = {}
        self._names: dict[int, str] = {}

    async def list_character_dictionaries(self, character_id, mode: str = "local") -> dict:
        blocks = (
            self.characters.get(int(character_id), {})
            .get("extensions", {})
            .get("chat_dictionaries", [])
        )
        return {
            "dictionaries": [
                {
                    "name": b["name"],
                    "entry_count": len(b.get("entries") or []),
                    "enabled": bool(b.get("enabled", True)),
                }
                for b in blocks
            ],
            "source": "local",
        }

    async def attach_to_character(self, dictionary_id, character_id, mode: str = "local") -> dict:
        char = self.characters.setdefault(
            int(character_id), {"extensions": {"chat_dictionaries": []}, "version": 1}
        )
        blocks = char["extensions"].setdefault("chat_dictionaries", [])
        name = self._names.get(int(dictionary_id), f"dict-{dictionary_id}")
        if not any(b["name"] == name for b in blocks):
            blocks.append({"name": name, "enabled": True, "entries": []})
            char["version"] += 1
        return {
            "dictionary_id": int(dictionary_id),
            "character_id": int(character_id),
            "dictionary_name": name,
            "character_dictionaries": [b["name"] for b in blocks],
            "source": "local",
        }

    async def detach_from_character(self, character_id, dictionary_name, mode: str = "local") -> dict:
        char = self.characters.get(
            int(character_id), {"extensions": {"chat_dictionaries": []}, "version": 1}
        )
        blocks = char["extensions"].get("chat_dictionaries", [])
        char["extensions"]["chat_dictionaries"] = [b for b in blocks if b["name"] != dictionary_name]
        char["version"] += 1
        return {
            "character_id": int(character_id),
            "dictionary_name": dictionary_name,
            "character_dictionaries": [b["name"] for b in char["extensions"]["chat_dictionaries"]],
            "source": "local",
        }


@pytest.fixture
def fake_dict_service(mock_app_instance):
    service = CharAttachFakeDictScopeService([make_dict_record(1, "Slang", entries=[])])
    mock_app_instance.chat_dictionary_scope_service = service
    return service


@pytest.fixture
def stub_characters(monkeypatch):
    """A real, selectable character (id 1) - the shared dictionaries-suite
    fixture of the same name stubs an EMPTY library, which this suite can't
    use since it needs to select a character row."""
    monkeypatch.setattr(
        character_handler_module, "fetch_all_characters", lambda: [dict(c) for c in CHARACTERS]
    )
    monkeypatch.setattr(
        character_handler_module,
        "fetch_character_by_id",
        lambda character_id: next(
            dict(c) for c in CHARACTERS if str(c["id"]) == str(character_id)
        ),
    )


async def _mounted(pilot):
    await pilot.pause()
    return pilot.app.screen


async def _enter_characters(pilot):
    """Characters is the default active mode; select the first row (id "1")."""
    screen = await _mounted(pilot)
    await pilot.click("#personas-library-row-character-1")
    await pilot.pause()
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()
    return screen


class TestCharacterDictionaryAttach:
    async def test_character_attach_via_picker_then_detach(
        self, mock_app_instance, stub_characters, fake_dict_service, monkeypatch
    ):
        from textual.widgets import DataTable
        from tldw_chatbook.Widgets.Persona_Widgets.dictionary_picker import DictionaryPicker

        fake_dict_service._names = {1: "Slang"}  # dict id 1 -> "Slang"

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_characters(pilot)
            assert screen.state.selected_entity_kind == "character"
            assert screen.state.selected_entity_id == "1"

            # Auto-pick dictionary id 1 instead of showing the modal.
            async def _fake_push(screen_obj):
                return 1 if isinstance(screen_obj, DictionaryPicker) else None

            monkeypatch.setattr(screen.app, "push_screen_wait", _fake_push, raising=False)
            monkeypatch.setattr(
                screen,
                "_list_attachable_dictionaries",
                lambda cid: [{"dictionary_id": 1, "name": "Slang"}],
            )
            await pilot.click("#personas-char-dicts-add")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert any(
                b["name"] == "Slang"
                for b in fake_dict_service.characters[1]["extensions"]["chat_dictionaries"]
            )
            table = screen.query_one("#personas-char-dicts-table", DataTable)
            assert table.row_count == 1

            table.move_cursor(row=0)
            await pilot.click("#personas-char-dicts-detach")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert fake_dict_service.characters[1]["extensions"]["chat_dictionaries"] == []
