"""TASK-32954 Task 5: ``CharacterCardChanged`` -> Personas refresh.

Unit-level: the handler is exercised by calling the unbound method
``PersonasScreen._on_character_card_changed`` against a lightweight
stand-in object (rather than ``PersonasScreen.__new__``, which pulls in
Textual widget machinery the handler never touches) -- see the task-5
controller ruling R3.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_chatbook.Character_Chat.character_events import CharacterCardChanged
from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen

pytestmark = pytest.mark.asyncio


class _Editor:
    def __init__(self, dirty: bool) -> None:
        self._dirty_posted = dirty


def _screen(selected_id: int, dirty: bool):
    calls = {"reload": 0, "notice": []}

    async def _reload(entity_id, entity_name, **_):
        calls["reload"] += 1

    screen = SimpleNamespace(
        state=SimpleNamespace(
            selected_entity_kind="character",
            selected_entity_id=str(selected_id),
            selected_entity_name="Aria",
            runtime_source="local",
        ),
        _select_character=_reload,
        _character_editor_is_active=lambda: True,
        query_one=lambda *_a, **_k: _Editor(dirty),
        _notify=lambda msg, sev="information": calls["notice"].append(msg),
    )
    return screen, calls


async def test_clean_editor_reloads_changed_character():
    screen, calls = _screen(7, dirty=False)
    await PersonasScreen._on_character_card_changed(screen, CharacterCardChanged(7))
    assert calls["reload"] == 1


async def test_dirty_editor_keeps_edits_and_warns():
    screen, calls = _screen(7, dirty=True)
    await PersonasScreen._on_character_card_changed(screen, CharacterCardChanged(7))
    assert calls["reload"] == 0 and "changed elsewhere" in calls["notice"][0]


async def test_other_character_ignored():
    screen, calls = _screen(7, dirty=False)
    await PersonasScreen._on_character_card_changed(screen, CharacterCardChanged(8))
    assert calls["reload"] == 0 and not calls["notice"]
