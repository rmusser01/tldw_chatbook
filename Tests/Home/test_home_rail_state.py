"""Home rail preference contracts."""

from tldw_chatbook.Home.home_rail_state import (
    HOME_RAIL_SECTION_IDS,
    HomeRailPreferences,
    coerce_home_rail_preferences,
    serialize_home_rail_preferences,
)


def test_home_rail_defaults():
    prefs = HomeRailPreferences()
    assert HOME_RAIL_SECTION_IDS == ("attention", "running", "recent", "details")
    assert prefs.attention_open is True
    assert prefs.running_open is True
    assert prefs.recent_open is True
    assert prefs.details_open is False


def test_coerce_reads_fields_and_defaults_missing():
    coerced = coerce_home_rail_preferences({"details_open": "true", "recent_open": "off"})
    assert coerced.details_open is True
    assert coerced.recent_open is False
    assert coerced.attention_open is True


def test_serialize_round_trips():
    prefs = HomeRailPreferences(details_open=True, running_open=False)
    serialized = serialize_home_rail_preferences(prefs)
    assert serialized["details_open"] is True
    assert serialized["running_open"] is False
    assert coerce_home_rail_preferences(serialized) == prefs


def test_coerce_unknown_input_returns_defaults():
    assert coerce_home_rail_preferences(None) == HomeRailPreferences()
    assert coerce_home_rail_preferences("junk") == HomeRailPreferences()
    assert coerce_home_rail_preferences(42) == HomeRailPreferences()
