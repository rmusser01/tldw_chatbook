"""Character rail publication reuses only its issued synchronous display proof."""

import asyncio
from types import MethodType, SimpleNamespace

import pytest

from Tests.UI.test_console_checked_display_scope import _actual_calls, _screen, _warm
from tldw_chatbook.UI.Console_Modules import wiring
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.asyncio
@pytest.mark.parametrize("reads", [1, 4])
async def test_actual_character_rail_uses_warm_checked_display_without_native_scope(
    tmp_path, reads
):
    database, _store, _controller, screen, tasks = _screen(tmp_path)
    character_states, rail_states, visible_states, mappings = [], [], [], []
    state = object()
    rail_state = object()
    screen.is_attached = True
    screen.app = screen.app_instance
    screen.app.screen_stack = [screen]
    screen._run_console_config_sync = MethodType(
        ChatScreen._run_console_config_sync, screen
    )
    widget = SimpleNamespace(sync_state=character_states.append)
    rail = SimpleNamespace(sync_character_context=rail_states.append)
    screen.query_one = lambda selector, *_: (
        widget if selector == "#console-character-context" else rail
    )

    def current_rail():
        for _ in range(reads):
            mappings.append(screen._provider_readiness_app_config())
        return rail_state

    screen._current_console_rail_state = current_rail
    screen._sync_console_rail_visibility_if_changed = visible_states.append
    try:
        projection = await _warm(screen, tasks)
        with _actual_calls() as calls:
            wiring._sync_character_context_presentation(screen, state)
        assert character_states == rail_states == [state]
        assert visible_states == [rail_state]
        assert len(mappings) == reads
        assert all(
            value is projection.value for value in mappings
        ), "Character rail bypasses the exact issued display mapping"
        assert (
            calls["main_scopes"] == 0
        ), "Character rail opens fresh config on the UI loop"
        assert (
            calls["main_opens"] == 0
        ), "Character rail repeats native proof on the UI loop"
    finally:
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        database.close()
