from tldw_chatbook.Scheduling.recurring_question_scope import (
    DEFAULT_SEARCHABLE_SOURCES,
    SUPPORTED_SCOPE_FIELDS,
    library_source_types,
    normalize_recurring_question_scope,
)


def test_default_mode_resolves_all_three_sources():
    normalized, errors, warnings = normalize_recurring_question_scope({})
    assert normalized == {
        "mode": "all_searchable_library",
        "resolved_sources": list(DEFAULT_SEARCHABLE_SOURCES),
    }
    assert errors == []
    assert warnings == []


def test_unknown_scope_field_reports_unsupported_error():
    normalized, errors, warnings = normalize_recurring_question_scope({"bogus": 1})
    assert errors == [
        {
            "field": "config.scope.bogus",
            "code": "unsupported",
            "message": "Unsupported scope field: bogus",
        }
    ]
    # Still resolves the default mode since "mode" itself was not the bad field.
    assert normalized["mode"] == "all_searchable_library"


def test_unknown_mode_returns_single_unsupported_error():
    normalized, errors, warnings = normalize_recurring_question_scope({"mode": "bogus_mode"})
    assert normalized == {"mode": "bogus_mode"}
    assert errors == [
        {
            "field": "config.scope.mode",
            "code": "unsupported",
            "message": "Unsupported scope mode: bogus_mode",
        }
    ]
    assert warnings == []


def test_sources_mode_drops_unavailable_sources_into_warnings():
    normalized, errors, warnings = normalize_recurring_question_scope(
        {"mode": "sources", "sources": ["notes", "not_a_real_source"]},
        available_sources=["media_db", "notes", "chats"],
    )
    assert normalized == {"mode": "sources", "sources": ["notes"]}
    assert errors == []
    assert warnings == [{"code": "source_unavailable", "source": "not_a_real_source"}]


def test_empty_resolution_reports_scope_empty_error():
    normalized, errors, warnings = normalize_recurring_question_scope(
        {"mode": "sources", "sources": ["nope"]},
        available_sources=["media_db"],
    )
    assert normalized == {"mode": "sources", "sources": []}
    assert errors == [
        {
            "field": "config.scope",
            "code": "scope_empty",
            "message": "Scope must include at least one readable searchable source.",
        }
    ]


def test_supported_scope_fields_matches_server_set():
    assert SUPPORTED_SCOPE_FIELDS == {
        "mode",
        "sources",
        "collection_ids",
        "tag_ids",
        "saved_search_ids",
        "source_types",
        "date_window",
        "workspace_id",
        "advanced_filters",
    }


def test_library_source_types_maps_all_searchable_library_mode():
    result = library_source_types(
        {"mode": "all_searchable_library", "resolved_sources": ["media_db", "notes", "chats"]}
    )
    assert result == ("media", "notes", "conversations")


def test_library_source_types_maps_sources_mode():
    result = library_source_types({"mode": "sources", "sources": ["chats", "media_db"]})
    assert result == ("conversations", "media")


def test_library_source_types_skips_unknown_names():
    result = library_source_types({"mode": "sources", "sources": ["notes", "mystery_source"]})
    assert result == ("notes",)


def test_library_source_types_map_is_a_subset_of_the_library_service_fts_servable_types():
    """Drift guard (Finding A): `library_source_types` maps into the
    LIBRARY service's vocabulary, not the retrieval engine's. If the
    Library service's servable source-type set ever changes, this test
    must break instead of silently unmapping a source name.
    """
    from tldw_chatbook.Library.library_local_rag_search_service import (
        _FTS_SERVABLE_SOURCE_TYPES,
    )
    from tldw_chatbook.Scheduling.recurring_question_scope import _LIBRARY_SOURCE_TYPE_MAP

    for library_source_type in _LIBRARY_SOURCE_TYPE_MAP.values():
        assert library_source_type in _FTS_SERVABLE_SOURCE_TYPES
