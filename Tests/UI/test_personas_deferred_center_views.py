"""task-2725: the four heavy hidden center views mount after first paint.

Profiling (task file) showed the Roleplay switch cost is widget-mount CSS
application, with 290 of the detail stack's 358 widgets in four views that
arrive hidden: the character editor (132), dictionary detail (67), lore
detail (60), and persona profile editor (31). They now mount as the first
step of `_load_after_mount` — off the click→paint critical path — instead
of inside compose.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

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

#: The stack's document order — scroll flow depends on it, so the deferred
#: mounts must land exactly where compose used to put them.
_EXPECTED_STACK_ORDER = [
    "ccp-character-card-view",
    "ccp-character-editor-view",
    "personas-character-attachments",
    "ccp-persona-card-view",
    "ccp-persona-editor-view",
    "personas-conversation-actions",
    "personas-dictionary-detail",
    "personas-lore-detail",
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


async def test_load_mounts_all_four_hidden_in_document_order():
    """After the real load, the full DOM exists exactly as compose built it
    pre-change: every deferred view present, hidden, in historical order."""
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        screen = PersonasScreen(app)
        await app.push_screen(screen)
        for _ in range(6):
            await pilot.pause()

        for deferred_type in _DEFERRED_TYPES:
            found = list(screen.query(deferred_type))
            assert len(found) == 1, f"{deferred_type.__name__} missing after load"
            assert found[0].display is False, (
                f"{deferred_type.__name__} arrived visible — it must stay "
                "hidden until _show_center reveals it"
            )

        stack = screen.query_one("#personas-detail-stack")
        assert [child.id for child in stack.children] == _EXPECTED_STACK_ORDER
