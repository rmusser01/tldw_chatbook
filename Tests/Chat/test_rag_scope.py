from tldw_chatbook.Chat.rag_scope import (
    RagScope, ScopeItem, SCOPE_VERSION, parse_scope, serialize_scope,
    SOURCE_TYPE_MEDIA, SOURCE_TYPE_NOTE,
    EffectiveScope, resolve_effective_scope, ScopeCache,
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


def test_intersection_and_causes():
    conv = RagScope(items=(ScopeItem("media", "1"), ScopeItem("media", "2")), updated_at="t1")
    ws = RagScope(items=(ScopeItem("media", "2"), ScopeItem("media", "3")), updated_at="t2")
    keep_all = lambda st, ids: ids
    eff = resolve_effective_scope(conv, ws, keep_all)
    assert eff.state == "scoped" and eff.allowlist["media"] == frozenset({"2"})
    disjoint = RagScope(items=(ScopeItem("media", "9"),), updated_at="t3")
    assert resolve_effective_scope(disjoint, ws, keep_all).cause == "no-workspace-overlap"
    gone = lambda st, ids: frozenset()
    assert resolve_effective_scope(conv, None, gone).cause == "deleted-items"

def test_cache_key_includes_ids_not_just_stamps():
    cache = ScopeCache()
    eff = EffectiveScope(state="unscoped", allowlist={}, cause=None)
    cache.put("c1", "w1", "s1", "s2", eff)
    assert cache.get("c1", "w1", "s1", "s2") is eff
    assert cache.get("c1", "w2", "s1", "s2") is None  # re-linked workspace: same stamps, different key

def test_both_scopes_none_is_unscoped_with_canonical_shape():
    """No conv scope and no workspace scope → unscoped, with the canonical empty-allowlist/no-cause shape."""
    eff = resolve_effective_scope(None, None, lambda st, ids: ids)
    assert eff.state == "unscoped"
    assert eff.allowlist == {}
    assert eff.cause is None

def test_single_level_resolution_conv_only():
    """A conversation-only scope (no workspace scope) resolves against that level alone."""
    conv = RagScope(items=(ScopeItem(SOURCE_TYPE_MEDIA, "1"), ScopeItem(SOURCE_TYPE_NOTE, "n1")), updated_at="t1")
    eff = resolve_effective_scope(conv, None, lambda st, ids: ids)
    assert eff.state == "scoped"
    assert eff.allowlist == {SOURCE_TYPE_MEDIA: frozenset({"1"}), SOURCE_TYPE_NOTE: frozenset({"n1"})}
    assert eff.cause is None

def test_single_level_resolution_workspace_only():
    """A workspace-only scope (no conversation scope) resolves against that level alone."""
    ws = RagScope(items=(ScopeItem(SOURCE_TYPE_MEDIA, "5"),), updated_at="t2")
    eff = resolve_effective_scope(None, ws, lambda st, ids: ids)
    assert eff.state == "scoped"
    assert eff.allowlist == {SOURCE_TYPE_MEDIA: frozenset({"5"})}
    assert eff.cause is None

def test_partial_dangling_drops_only_missing_ids():
    """existing_ids removing some (not all) ids leaves the survivors, still 'scoped' with no cause."""
    conv = RagScope(items=(ScopeItem(SOURCE_TYPE_MEDIA, "1"), ScopeItem(SOURCE_TYPE_MEDIA, "2")), updated_at="t1")
    drop_only_1 = lambda st, ids: ids - {"1"}
    eff = resolve_effective_scope(conv, None, drop_only_1)
    assert eff.state == "scoped"
    assert eff.allowlist == {SOURCE_TYPE_MEDIA: frozenset({"2"})}
    assert eff.cause is None

def test_allowlist_omits_source_types_with_no_survivors():
    """The allowlist contains only non-empty entries; a source_type fully dropped is absent, not {}."""
    conv = RagScope(items=(ScopeItem(SOURCE_TYPE_MEDIA, "1"), ScopeItem(SOURCE_TYPE_NOTE, "n1")), updated_at="t1")
    drop_notes = lambda st, ids: frozenset() if st == SOURCE_TYPE_NOTE else ids
    eff = resolve_effective_scope(conv, None, drop_notes)
    assert eff.state == "scoped"
    assert eff.allowlist == {SOURCE_TYPE_MEDIA: frozenset({"1"})}
    assert SOURCE_TYPE_NOTE not in eff.allowlist

def test_cache_clear_removes_all_entries():
    """clear() empties the cache; a subsequent get() on a previously-stored key returns None."""
    cache = ScopeCache()
    eff = EffectiveScope(state="unscoped", allowlist={}, cause=None)
    cache.put("c1", "w1", "s1", "s2", eff)
    cache.clear()
    assert cache.get("c1", "w1", "s1", "s2") is None
