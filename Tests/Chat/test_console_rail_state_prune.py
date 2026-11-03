"""Console rail-state pruning: the key filter and the launch-time dispatch."""

from tldw_chatbook.Chat.console_rail_state import collect_prunable_console_rail_keys


def test_prunes_conversation_keys():
    """TASK-718: preferences are per-workspace now; every per-conversation key
    is stale, whether or not its conversation still exists."""
    stored_keys = ["console_rail_state:ws:live-conv"]

    prunable = collect_prunable_console_rail_keys(stored_keys)

    assert prunable == ["console_rail_state:ws:live-conv"]


def test_keeps_layout_scope_key():
    stored_keys = ["console_rail_state:ws:layout"]

    prunable = collect_prunable_console_rail_keys(stored_keys)

    assert prunable == []


def test_keeps_global_scope_key_even_with_no_live_scopes():
    stored_keys = ["console_rail_state:ws:global"]

    prunable = collect_prunable_console_rail_keys(stored_keys)

    assert prunable == []


def test_keeps_reserved_shared_layout_scope_even_with_no_live_scopes():
    stored_keys = ["console_rail_state:global:shared-layout-v1"]

    prunable = collect_prunable_console_rail_keys(stored_keys)

    assert prunable == []


def test_keeps_malformed_and_foreign_keys():
    stored_keys = [
        "other_prefix:ws:scope",  # wrong prefix
        "console_rail_state:ws",  # too few parts
        "console_rail_state:ws:scope:extra",  # too many parts
        123,  # non-str
        None,  # non-str
    ]

    prunable = collect_prunable_console_rail_keys(stored_keys)

    assert prunable == []


def test_sanitized_conversation_keys_are_prunable():
    # Under the per-workspace scheme (TASK-718) sanitized conversation keys
    # are stale like any other conversation-scoped entry.
    stored_keys = ["console_rail_state:ws:conv_1"]

    prunable = collect_prunable_console_rail_keys(stored_keys)

    assert prunable == ["console_rail_state:ws:conv_1"]


def test_mixed_stored_keys_keeps_only_layout_legacy_and_shared_scopes():
    """TASK-718: per-workspace scheme - every conversation/session-scoped key
    is prunable; ``:layout`` (current) and ``:global`` (migration source) and
    foreign shapes are kept."""
    stored_keys = [
        "console_rail_state:ws:global",
        "console_rail_state:ws:layout",
        "console_rail_state:global:shared-layout-v1",
        "console_rail_state:ws:live-conv",
        "console_rail_state:ws:orphan-1",
        "console_rail_state:ws:orphan-2",
        "other_prefix:ws:scope",
    ]

    prunable = collect_prunable_console_rail_keys(stored_keys)

    assert sorted(prunable) == [
        "console_rail_state:ws:live-conv",
        "console_rail_state:ws:orphan-1",
        "console_rail_state:ws:orphan-2",
    ]


_CURRENT_KEYS = (
    "console_rail_state:global:shared-layout-v1",
    "console_rail_state:ws:layout",
    "console_rail_state:ws:global",
)
_LEGACY_KEYS = ("console_rail_state:ws:conv-1", "console_rail_state:ws:session-2")


def _launch(monkeypatch, rail_state):
    """Run Console's prune dispatch, its worker body inline, against a fake app."""
    from types import SimpleNamespace

    from tldw_chatbook.UI.Screens import chat_screen
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    seen = SimpleNamespace(started=[], reads=[], deleted=[])

    def list_all_active_conversations(**kwargs):
        seen.reads.append(kwargs)
        return []

    def delete_settings(section, keys):
        seen.deleted.append((section, list(keys)))
        return True

    monkeypatch.setattr(chat_screen, "delete_settings_from_cli_config", delete_settings)
    screen = SimpleNamespace(
        _console_rail_prune_dispatched=False,
        _console_chat_store=SimpleNamespace(sessions=lambda: ()),
        app_instance=SimpleNamespace(
            app_config={"console": {"rail_state": rail_state}},
            chachanotes_db=SimpleNamespace(
                list_all_active_conversations=list_all_active_conversations
            ),
        ),
        app=SimpleNamespace(call_from_thread=lambda callback, *args: callback(*args)),
    )
    screen._drop_console_rail_preference_keys_in_memory = lambda keys: (
        ChatScreen._drop_console_rail_preference_keys_in_memory(screen, keys)
    )

    def worker(*args):
        seen.started.append(args)
        ChatScreen._prune_console_rail_preferences.__wrapped__(screen, *args)

    screen._prune_console_rail_preferences = worker
    ChatScreen._dispatch_console_rail_preference_prune(screen)
    ChatScreen._dispatch_console_rail_preference_prune(screen)  # a later sync
    return seen


def test_launch_without_legacy_keys_reads_nothing_and_starts_no_worker(monkeypatch):
    """TASK-36518 AC#1: no legacy key, no conversation scan, no prune worker."""
    rail_state = {key: {} for key in _CURRENT_KEYS}

    seen = _launch(monkeypatch, rail_state)

    assert (seen.started, seen.reads, seen.deleted) == ([], [], [])
    assert list(rail_state) == list(_CURRENT_KEYS)


def test_launch_with_legacy_keys_deletes_exactly_them_once(monkeypatch):
    """TASK-36518 AC#2: the legacy keys go, once; current layout keys stay."""
    rail_state = {key: {} for key in (*_CURRENT_KEYS, *_LEGACY_KEYS)}

    seen = _launch(monkeypatch, rail_state)

    assert seen.reads == []
    assert seen.deleted == [("console.rail_state", list(_LEGACY_KEYS))]
    assert list(rail_state) == list(_CURRENT_KEYS)
    # The next launch finds nothing left to prune.
    assert _launch(monkeypatch, rail_state).started == []
