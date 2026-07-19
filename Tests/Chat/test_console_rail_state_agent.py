"""agent_open threads through rail preferences + effective state."""
from tldw_chatbook.Chat.console_rail_state import (
    ConsoleRailPreferences, serialize_console_rail_preferences,
    coerce_console_rail_preferences as deserialize_console_rail_preferences,
    build_console_rail_preference_key, build_console_rail_state,
)


def test_agent_open_defaults_false_and_round_trips():
    prefs = ConsoleRailPreferences()
    assert prefs.agent_open is False
    blob = serialize_console_rail_preferences(prefs)
    assert blob["agent_open"] is False
    restored = deserialize_console_rail_preferences({**blob, "agent_open": True})
    assert restored.agent_open is True


def test_build_console_rail_state_carries_agent_open():
    key = build_console_rail_preference_key(
        workspace_id="workspace-1",
        session_id="session-1",
    )
    state = build_console_rail_state(
        preference_key=key,
        stored_preferences={"agent_open": True},
    )
    assert state.agent_open is True
