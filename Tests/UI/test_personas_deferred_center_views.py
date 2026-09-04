"""TASK-31215: heavy Personas center views mount only on first use.

TASK-2725 moved four hidden bodies past first paint, but still mounted all of
them during the initial load. TASK-31215 keeps stable slots in document order
and mounts only the body an editor/detail workflow first requests.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from textual.widgets import Input

from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen
from tldw_chatbook.Widgets.Persona_Widgets.persona_profile_editor_widget import (
    PersonaProfileEditorWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import (
    PersonasCharacterEditorWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_dictionary_detail import (
    PersonasDictionaryDetailWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_lore_detail import (
    PersonasLoreDetailWidget,
)

pytestmark = pytest.mark.asyncio

_DEFERRED_TYPES = (
    PersonasCharacterEditorWidget,
    PersonasDictionaryDetailWidget,
    PersonasLoreDetailWidget,
    PersonaProfileEditorWidget,
)

#: The stack's document order — scroll flow depends on it, so the lightweight
#: slots must occupy exactly the positions their bodies historically used.
_EXPECTED_STACK_ORDER = [
    "ccp-character-card-view",
    "personas-character-editor-slot",
    "personas-character-attachments",
    "ccp-persona-card-view",
    "personas-persona-editor-slot",
    "personas-conversation-actions",
    "personas-dictionary-detail-slot",
    "personas-lore-detail-slot",
    "personas-conversation-transcript-view",
    "personas-mode-placeholder",
    "personas-characters-empty",
]


async def test_first_paint_excludes_the_four_heavy_center_views(monkeypatch):
    """Compose alone must not mount the deferred views (the perf mechanism)."""
    from Tests.UI.app_factory import _build_test_app

    monkeypatch.setattr(
        PersonasScreen, "_load_after_mount", AsyncMock(), raising=True
    )

    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        screen = PersonasScreen(app)
        await app.push_screen(screen)
        await pilot.pause()

        for deferred_type in _DEFERRED_TYPES:
            assert not list(screen.query(deferred_type)), (
                f"{deferred_type.__name__} mounted during compose — "
                "back on the click→paint critical path"
            )
        total = len(list(screen.query("*")))
        assert total < 300, (
            f"first-paint widget count regressed to {total} (pre-fix: 494)"
        )


async def test_settled_initial_load_keeps_heavy_views_unmounted():
    """The real load must settle without constructing inactive heavy bodies."""
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        screen = PersonasScreen(app)
        await app.push_screen(screen)
        for _ in range(6):
            await pilot.pause()

        for deferred_type in _DEFERRED_TYPES:
            assert not list(screen.query(deferred_type)), (
                f"{deferred_type.__name__} mounted during initial load — "
                "inactive bodies must wait for first use"
            )

        stack = screen.query_one("#personas-detail-stack")
        assert [child.id for child in stack.children] == _EXPECTED_STACK_ORDER


async def test_first_use_mounts_only_requested_view_and_caches_it():
    """One requested body mounts once and retains its in-screen form state."""
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        screen = PersonasScreen(app)
        await app.push_screen(screen)
        for _ in range(6):
            await pilot.pause()

        first = await screen._ensure_center_view("character-editor")
        assert isinstance(first, PersonasCharacterEditorWidget)
        name = first.query_one("#personas-char-editor-name", Input)
        name.value = "Keep this draft"

        second = await screen._ensure_center_view("character-editor")

        assert second is first
        assert second.query_one("#personas-char-editor-name", Input).value == (
            "Keep this draft"
        )
        assert not list(screen.query(PersonaProfileEditorWidget))
        assert not list(screen.query(PersonasDictionaryDetailWidget))
        assert not list(screen.query(PersonasLoreDetailWidget))
