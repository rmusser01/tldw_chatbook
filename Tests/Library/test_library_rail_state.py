"""Library rail preference contracts."""

from tldw_chatbook.Library.library_rail_state import (
    LIBRARY_RAIL_SECTION_IDS,
    LibraryRailPreferences,
    coerce_library_rail_preferences,
    serialize_library_rail_preferences,
)


def test_library_rail_defaults():
    prefs = LibraryRailPreferences()
    assert LIBRARY_RAIL_SECTION_IDS == ("browse", "create", "ingest", "details")
    assert prefs.browse_open is True
    assert prefs.create_open is True
    assert prefs.ingest_open is True
    assert prefs.details_open is False


def test_coerce_reads_fields_and_defaults_missing():
    coerced = coerce_library_rail_preferences({"details_open": "true", "ingest_open": "off"})
    assert coerced.details_open is True
    assert coerced.ingest_open is False
    assert coerced.browse_open is True


def test_serialize_round_trips():
    prefs = LibraryRailPreferences(details_open=True, create_open=False)
    serialized = serialize_library_rail_preferences(prefs)
    assert serialized["details_open"] is True
    assert serialized["create_open"] is False
    assert coerce_library_rail_preferences(serialized) == prefs


def test_coerce_unknown_input_returns_defaults():
    assert coerce_library_rail_preferences(None) == LibraryRailPreferences()
    assert coerce_library_rail_preferences("junk") == LibraryRailPreferences()
    assert coerce_library_rail_preferences(42) == LibraryRailPreferences()
