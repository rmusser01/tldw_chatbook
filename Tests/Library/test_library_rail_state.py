"""Library rail preference contracts."""

from tldw_chatbook.Library.library_rail_state import (
    LIBRARY_RAIL_SECTION_IDS,
    LibraryLifecycle,
    LibraryRailPreferences,
    coerce_library_lifecycle,
    coerce_library_rail_preferences,
    serialize_library_lifecycle,
    serialize_library_rail_preferences,
)


def test_library_rail_defaults():
    prefs = LibraryRailPreferences()
    assert LIBRARY_RAIL_SECTION_IDS == (
        "browse",
        "create",
        "study",
        "ingest",
        "details",
    )
    assert prefs.browse_open is True
    assert prefs.create_open is True
    assert prefs.study_open is True
    assert prefs.ingest_open is True
    assert prefs.details_open is False


def test_coerce_reads_fields_and_defaults_missing():
    coerced = coerce_library_rail_preferences(
        {"details_open": "true", "ingest_open": "off"}
    )
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


def test_missing_lifecycle_uses_unknown_only_for_new_profile():
    assert (
        coerce_library_lifecycle(None, is_new_profile=True) is LibraryLifecycle.UNKNOWN
    )
    assert (
        coerce_library_lifecycle(None, is_new_profile=False)
        is LibraryLifecycle.EXPANDED
    )


def test_corrupt_lifecycle_fails_safe_to_expanded_without_resetting_sections():
    sections = {"browse_open": "off", "details_open": "true"}
    expected_sections = LibraryRailPreferences(browse_open=False, details_open=True)

    for corrupt in ("", "starter-ish", 0, {}, []):
        assert (
            coerce_library_lifecycle(corrupt, is_new_profile=True)
            is LibraryLifecycle.EXPANDED
        )
        assert coerce_library_rail_preferences(sections) == expected_sections


def test_lifecycle_round_trips_beside_section_preferences():
    preferences = LibraryRailPreferences(create_open=False, details_open=True)
    stored = {
        "sections": serialize_library_rail_preferences(preferences),
        "lifecycle": serialize_library_lifecycle(LibraryLifecycle.STARTER),
    }

    assert stored["lifecycle"] == "starter"
    assert (
        coerce_library_lifecycle(stored["lifecycle"], is_new_profile=False)
        is LibraryLifecycle.STARTER
    )
    assert coerce_library_rail_preferences(stored["sections"]) == preferences
