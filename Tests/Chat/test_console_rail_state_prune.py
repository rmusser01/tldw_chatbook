"""Pure tests for collect_prunable_console_rail_keys (Console rail-state pruning)."""

from tldw_chatbook.Chat.console_rail_state import collect_prunable_console_rail_keys


def test_prunes_dead_scope_key():
    stored_keys = ["console_rail_state:ws:dead-conv"]

    prunable = collect_prunable_console_rail_keys(
        stored_keys, live_scope_ids=["live-conv"]
    )

    assert prunable == ["console_rail_state:ws:dead-conv"]


def test_keeps_live_scope_key():
    stored_keys = ["console_rail_state:ws:live-conv"]

    prunable = collect_prunable_console_rail_keys(
        stored_keys, live_scope_ids=["live-conv"]
    )

    assert prunable == []


def test_keeps_global_scope_key_even_with_no_live_scopes():
    stored_keys = ["console_rail_state:ws:global"]

    prunable = collect_prunable_console_rail_keys(stored_keys, live_scope_ids=[])

    assert prunable == []


def test_keeps_malformed_and_foreign_keys():
    stored_keys = [
        "other_prefix:ws:scope",  # wrong prefix
        "console_rail_state:ws",  # too few parts
        "console_rail_state:ws:scope:extra",  # too many parts
        123,  # non-str
        None,  # non-str
    ]

    prunable = collect_prunable_console_rail_keys(stored_keys, live_scope_ids=[])

    assert prunable == []


def test_live_ids_are_matched_after_sanitization():
    # The module sanitizes raw scope ids (replacing characters outside
    # [A-Za-z0-9_.-] with "_") before building persistence keys, so a raw id
    # containing a colon is stored under its sanitized form.
    stored_keys = ["console_rail_state:ws:conv_1"]

    prunable = collect_prunable_console_rail_keys(
        stored_keys, live_scope_ids=["conv:1"]
    )

    assert prunable == []


def test_mixed_stored_keys_only_prunes_dead_scopes():
    stored_keys = [
        "console_rail_state:ws:global",
        "console_rail_state:ws:live-conv",
        "console_rail_state:ws:orphan-1",
        "console_rail_state:ws:orphan-2",
        "other_prefix:ws:scope",
    ]

    prunable = collect_prunable_console_rail_keys(
        stored_keys, live_scope_ids=["live-conv"]
    )

    assert sorted(prunable) == [
        "console_rail_state:ws:orphan-1",
        "console_rail_state:ws:orphan-2",
    ]
