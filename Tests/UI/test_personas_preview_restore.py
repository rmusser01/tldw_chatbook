"""Tests for PersonasPreviewController.restore_conversation (task-434)."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from tldw_chatbook.UI.Persona_Modules.personas_preview_controller import (
    PersonasPreviewController,
)


def _controller_with_mock_pane():
    ctrl = PersonasPreviewController.__new__(PersonasPreviewController)
    ctrl.history = [{"role": "assistant", "content": "old"}]
    ctrl.seeded_for = None
    ctrl.generation = 0
    ctrl.gateway = None
    ctrl._greetings = []
    ctrl._current_greeting_index = 0
    events = []
    pane = MagicMock()
    pane.append_user = lambda t: events.append(("user", t))
    pane.append_reply = lambda t: events.append(("reply", t))

    async def _seed(text):
        # Ordering guard: seeded_for MUST already be set when the first await runs.
        events.append(("seed", text, ctrl.seeded_for))
    pane.seed_greeting = AsyncMock(side_effect=_seed)

    screen = MagicMock()
    screen.query_one.return_value = pane
    screen.workers.cancel_group = MagicMock()
    ctrl.screen = screen
    return ctrl, events


@pytest.mark.asyncio
async def test_restore_conversation_seeds_greeting_then_turns_and_sets_seeded_for_first():
    ctrl, events = _controller_with_mock_pane()
    history = [
        {"role": "assistant", "content": "Greetings."},   # note: greeting is NOT in history
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    await ctrl.restore_conversation(
        greeting="Greetings.", history=history, seeded_for="7"
    )
    # seeded_for was already "7" at the first await (the seed call)
    seed_events = [e for e in events if e[0] == "seed"]
    assert seed_events and seed_events[0][2] == "7"
    # greeting seeded, then each history turn rendered in order
    assert events[0] == ("seed", "Greetings.", "7")
    assert ("user", "hi") in events and ("reply", "hello") in events
    # controller state updated
    assert ctrl.seeded_for == "7"
    assert ctrl.history == history


@pytest.mark.asyncio
async def test_handle_character_loaded_does_not_relabel_on_preserve():
    # task-437 review: a same-character reload that PRESERVES an in-progress
    # transcript must not call set_speakers — that only relabels FUTURE lines,
    # leaving the already-rendered lines under the old prefix (mixed/stale).
    ctrl, _ = _controller_with_mock_pane()
    ctrl.seeded_for = "7"
    pane = ctrl.screen.query_one.return_value
    pane.transcript_text = MagicMock(return_value="Sam: hi")
    pane.refresh_greeting_seed = MagicMock()
    ctrl.refresh_provider_readout = MagicMock()
    ctrl.screen.state = MagicMock(
        active_mode="characters",
        selected_entity_kind="character",
        selected_entity_id="7",
        selected_entity_name="Sam",
    )
    await ctrl.handle_character_loaded(
        character_id="7",
        card_data={"name": "Detective Sam", "first_message": "Hi from {{char}}."},
    )
    # Reload refreshed the reset seed but did NOT relabel (no mixed prefixes).
    pane.refresh_greeting_seed.assert_called_once()
    pane.set_speakers.assert_not_called()
