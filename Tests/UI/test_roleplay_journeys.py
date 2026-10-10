"""Roleplay journeys, one test per journey a frame slice touches (spec 5.7.3).

From frame slice B1: Edit -> type -> the header's unsaved chip appears ->
Ctrl+S -> the chip goes (B1 AC#5). Under both styled tiers, driven like a
user: a click on the card's Edit, keystrokes, Ctrl+S. Both tiers stub the
character seams (the full tier is not DB-seeded until B2a), so B1 checks the
payload that reached ``update_character``, not persistence. Later slices add
the spec's J1-J4 journeys here, each as a passing prefix test plus a strict
``xfail(raises=NotYetDelivered)`` full-journey test (RC-7).
"""

from __future__ import annotations

import pytest
from textual.widgets import TextArea

import tldw_chatbook.app  # noqa: F401  -- collection-time import (lessons-testing-evidence: Tests/UI RecoveryRequired at setup)
import tldw_chatbook.UI.CCP_Modules.ccp_character_handler as character_handler_module
from Tests.UI.roleplay_frame_harness import (
    open_styled_roleplay,
    seed_mock_characters,
    settle,
    styled_tiers,
    wait_until,
)
from Tests.UI.test_personas_workbench import (
    _conversation_record,
    _install_conversation_db,
)
from tldw_chatbook.UI.Workbench.workbench_widgets import FittedText

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]


@styled_tiers
async def test_editing_shows_the_unsaved_chip_and_saving_clears_it(
    styled_tier, mock_app_instance, monkeypatch
):
    seed_mock_characters(
        monkeypatch,
        [{"id": 1, "name": "Detective Sam", "description": "Noir", "version": 1}],
    )
    monkeypatch.setattr(
        character_handler_module, "_default_character_db", lambda: object()
    )
    saved: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        character_handler_module,
        "update_character",
        lambda cid, data: saved.append((str(cid), dict(data))) or True,
    )
    _install_conversation_db(monkeypatch, [_conversation_record(1)])
    async with open_styled_roleplay(
        styled_tier, mock_app_instance, size=(120, 36)
    ) as pilot:
        screen = pilot.app.screen
        chip = screen.query_one("#personas-header-unsaved", FittedText)
        item = screen.query_one("#personas-header-item", FittedText)
        assert not chip.display

        assert await pilot.click("#personas-card-edit-character")
        await wait_until(pilot, lambda: screen._edit_mode == "edit", what="Edit")
        assert item.value == ("Detective Sam", True)
        assert not chip.display  # opening the editor is not an edit

        screen.query_one("#personas-char-editor-description", TextArea).focus()
        await pilot.pause()
        await pilot.press("x")
        await wait_until(pilot, lambda: chip.display, what="the unsaved chip")
        assert chip.value == "Unsaved changes"

        await pilot.press("ctrl+s")
        await settle(pilot)
        await wait_until(pilot, lambda: not chip.display, what="the chip to go")
        assert screen._edit_mode == "edit"  # save-in-place keeps the editor
        assert item.value == ("Detective Sam", True)
        # The save reached the character seam with the typed text.
        assert saved and saved[-1][0] == "1" and "x" in saved[-1][1]["description"]
