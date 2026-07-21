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
