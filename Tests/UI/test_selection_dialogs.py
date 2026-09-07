# test_selection_dialogs.py
# Description: End-to-end drives of the STTS selection dialogs (TASK-15992).
#
# These dialogs shipped broken for so long (invalid CSS in TASK-15450, then a
# nonexistent `Vertical.clear()` in `on_mount`) that a mount-only pin could
# hide further defects. These tests drive each dialog the way a user would:
# push it, pick an item, press Generate, and assert the dismissal result the
# STTS window's callback actually receives.

from __future__ import annotations

from typing import Any

import pytest
from textual.widgets import Button, Checkbox, RadioButton, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Widgets.conversation_selection_dialog import (
    ConversationSelectionDialog,
)
from tldw_chatbook.Widgets.Note_Widgets.note_selection_dialog import (
    NoteSelectionDialog,
)

NOTES = [
    {
        "note_id": 1,
        "title": "First note",
        "content": "alpha",
        "created_at": "2026-01-01",
    },
    {
        "note_id": 2,
        "title": "Second note",
        "content": "beta",
        "created_at": "2026-01-02",
    },
]

CONVERSATIONS = [
    {
        "conversation_id": 11,
        "title": "Conv A",
        "model_name": "model-a",
        "message_count": 3,
        "created_at": "2026-01-01",
        "updated_at": "2026-01-02",
    },
    {
        "conversation_id": 22,
        "title": "Conv B",
        "model_name": "model-b",
        "message_count": 5,
        "created_at": "2026-01-03",
        "updated_at": "2026-01-04",
    },
]


@pytest.mark.asyncio
async def test_note_selection_dialog_returns_checked_note_ids():
    """Checking a note enables Generate, and Generate dismisses with its id."""
    app = ConsolidatedCSSApp()
    results: list[Any] = []
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(NoteSelectionDialog(NOTES), results.append)
        await pilot.pause()
        dialog = app.screen
        assert isinstance(dialog, NoteSelectionDialog)
        assert len(dialog.note_items) == 2

        generate = dialog.query_one("#generate-btn", Button)
        assert generate.disabled, "Generate must start disabled with nothing checked"

        dialog.query_one("#note-checkbox-2", Checkbox).value = True
        await pilot.pause()
        assert not generate.disabled
        info = dialog.query_one("#selection-info", Static)
        assert "1 note selected" in str(info.renderable)

        generate.press()
        await pilot.pause()

    assert results == [[2]]


@pytest.mark.asyncio
async def test_conversation_selection_dialog_returns_selected_conversation():
    """Picking a conversation enables Generate, and Generate dismisses with it."""
    app = ConsolidatedCSSApp()
    results: list[Any] = []
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(ConversationSelectionDialog(CONVERSATIONS), results.append)
        await pilot.pause()
        dialog = app.screen
        assert isinstance(dialog, ConversationSelectionDialog)
        assert len(dialog.conversation_items) == 2

        generate = dialog.query_one("#generate-btn", Button)
        assert generate.disabled, "Generate must start disabled with no selection"

        dialog.query_one("#conv-radio-22", RadioButton).value = True
        await pilot.pause()
        assert dialog.selected_conversation_id == 22
        assert not generate.disabled

        generate.press()
        await pilot.pause()

    assert len(results) == 1
    result = results[0]
    assert result["conversation_id"] == 22
    # The default export options must survive the round trip.
    assert result["include_all"] is True
    assert result["include_user"] is False
    assert result["include_assistant"] is False
    assert result["include_speakers"] is True


@pytest.mark.asyncio
async def test_conversation_radios_stay_mutually_exclusive():
    """Selecting a second conversation unselects the first; untoggling clears."""
    app = ConsolidatedCSSApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(ConversationSelectionDialog(CONVERSATIONS), lambda _: None)
        await pilot.pause()
        dialog = app.screen
        radio_a = dialog.query_one("#conv-radio-11", RadioButton)
        radio_b = dialog.query_one("#conv-radio-22", RadioButton)
        generate = dialog.query_one("#generate-btn", Button)

        radio_a.value = True
        await pilot.pause()
        assert dialog.selected_conversation_id == 11

        radio_b.value = True
        await pilot.pause()
        assert dialog.selected_conversation_id == 22
        assert radio_a.value is False, "previous selection must be turned off"
        assert radio_b.value is True

        radio_b.value = False
        await pilot.pause()
        assert dialog.selected_conversation_id is None
        assert generate.disabled, "clearing the selection must disable Generate"

        dialog.query_one("#cancel-btn", Button).press()
        await pilot.pause()
