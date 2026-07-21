from tldw_chatbook.Chat.rag_scope import (
    RagScope, ScopeItem, SCOPE_VERSION, parse_scope, serialize_scope,
    SOURCE_TYPE_MEDIA, SOURCE_TYPE_NOTE,
)

def test_round_trip():
    """serialize→parse preserves items and stamps."""
    scope = RagScope(items=(ScopeItem(SOURCE_TYPE_MEDIA, "42"), ScopeItem(SOURCE_TYPE_NOTE, "n-1")), updated_at="2026-07-21T00:00:00+00:00")
    raw = serialize_scope(scope)
    assert raw["version"] == SCOPE_VERSION
    parsed = parse_scope(raw)
    assert parsed == scope

def test_parse_guards():
    """Missing/None/malformed/newer-version → None (unscoped), never raises."""
    for bad in (None, "", 7, [], {}, {"version": 1}, {"version": 99, "items": []},
                {"version": 1, "items": [{"source_type": "media"}]},
                {"version": 1, "items": "nope", "updated_at": "x"}):
        assert parse_scope(bad) is None

def test_source_id_coerced_to_str():
    """Integer ids from raw payloads coerce to str at the boundary."""
    parsed = parse_scope({"version": 1, "updated_at": "t",
                          "items": [{"source_type": "media", "source_id": 42}]})
    assert parsed.items == (ScopeItem(SOURCE_TYPE_MEDIA, "42"),)

def test_unknown_source_type_dropped():
    """Forward-compat: unknown source types (e.g. 'conversation') are dropped, not fatal."""
    parsed = parse_scope({"version": 1, "updated_at": "t", "items": [
        {"source_type": "conversation", "source_id": "c1"},
        {"source_type": "note", "source_id": "n1"}]})
    assert parsed.items == (ScopeItem(SOURCE_TYPE_NOTE, "n1"),)

def test_malformed_payloads_log_warning():
    """Each malformed payload (except None) emits exactly one warning.

    Malformed cases: raw not a dict (string, int, list), missing/invalid version,
    items not a list, updated_at not a string, non-dict item entry, missing source_id.
    None (absent scope) emits NO warning — it's normal, not an error.
    """
    from loguru import logger as loguru_logger

    malformed_cases = [
        ("", "raw not a dict"),
        (7, "raw not a dict"),
        ([], "raw not a dict"),
        ({"version": "1"}, "invalid version type"),
        ({"version": 99}, "version too new"),
        ({"version": 1}, "missing items or updated_at"),
        ({"version": 1, "updated_at": "t"}, "items missing"),
        ({"version": 1, "items": "not a list", "updated_at": "t"}, "items not a list"),
        ({"version": 1, "items": [], "updated_at": 123}, "updated_at not a string"),
        ({"version": 1, "items": ["not a dict"], "updated_at": "t"}, "item not a dict"),
        ({"version": 1, "items": [{"source_type": "media"}], "updated_at": "t"}, "missing source_id"),
    ]

    for raw, description in malformed_cases:
        messages: list[str] = []
        sink_id = loguru_logger.add(
            messages.append, level="WARNING", format="{message}"
        )
        try:
            result = parse_scope(raw)
        finally:
            loguru_logger.remove(sink_id)

        # Assert the payload returns None (unscoped)
        assert result is None, f"Case '{description}': expected None, got {result}"

        # Assert exactly one warning was logged mentioning "rag_scope"
        warning_count = sum(1 for m in messages if "rag_scope" in m)
        assert warning_count == 1, f"Case '{description}': expected 1 warning, got {warning_count}. Messages: {messages}"

    # Test that None (absent scope) does NOT log a warning
    messages: list[str] = []
    sink_id = loguru_logger.add(
        messages.append, level="WARNING", format="{message}"
    )
    try:
        result = parse_scope(None)
    finally:
        loguru_logger.remove(sink_id)

    assert result is None
    warning_count = sum(1 for m in messages if "rag_scope" in m)
    assert warning_count == 0, f"None should not log warning. Messages: {messages}"
