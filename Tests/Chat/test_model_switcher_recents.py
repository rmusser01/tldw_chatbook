"""Switch model recents and PREVIOUS from existing data only (TASK-33004.3).

Spec §4 rule 6: RECENT comes from open Console sessions plus the ADR-095
snapshots of the 50 newest global-scope chats; PREVIOUS lives in process
memory. Every DB test runs against a real ChaChaNotes database.
"""

from __future__ import annotations

import asyncio
import json
import threading
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession, ConsoleChatStore
from tldw_chatbook.Chat.console_conversation_hydration import hydrate_console_session
from tldw_chatbook.Chat.console_generation_settings_metadata import (
    merge_console_generation_settings,
    snapshot_from_session_settings,
)
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.LLM_Provider_Catalog.llm_provider_catalog_scope_service import (
    LLMProviderCatalogScopeService,
)
from tldw_chatbook.LLM_Provider_Catalog.local_llm_provider_catalog_service import (
    LocalLLMProviderCatalogService,
)
from tldw_chatbook.UI.Console_Modules.model_switcher import (
    RECENT_CONVERSATION_LIMIT,
    ModelPairUse,
    PreviousPairMemory,
    load_provider_catalog,
    persisted_model_pair_uses,
    read_recent_model_pairs,
)

NOW = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)


def _iso(when: datetime) -> str:
    return when.isoformat().replace("+00:00", "Z")


def _owned(provider: str, model: str | None) -> dict:
    settings = ConsoleSessionSettings(provider=provider, model=model)
    return merge_console_generation_settings(
        None, snapshot_from_session_settings(settings)
    )


def _add(db, metadata: object, *, age: timedelta, **extra) -> str:
    raw = None if metadata is None else json.dumps(metadata)
    cid = db.add_conversation({"title": "t", "metadata": raw, **extra})
    with db.transaction() as conn:
        conn.execute(
            "UPDATE conversations SET last_modified = ? WHERE id = ?",
            (_iso(NOW - age), cid),
        )
    return cid


def _session(provider, model, *, age, conversation_id=None, ephemeral=False):
    return ConsoleChatSession(
        settings=ConsoleSessionSettings(provider=provider, model=model),
        persisted_conversation_id=conversation_id,
        updated_at=(NOW - age).isoformat(),
        ephemeral=ephemeral,
    )


def _pairs(uses):
    return [use.pair for use in uses]


@pytest.fixture
def memory_db():
    db = CharactersRAGDB(":memory:", "t3-recents")
    yield db
    db.close_connection()


@pytest.fixture
def file_db(tmp_path):
    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "t3-recents")
    yield db
    db.close_connection()


def test_persisted_pairs_fail_closed_on_bad_snapshots(memory_db) -> None:
    """AC#4: malformed, missing, future-version and model-less rows are skipped."""
    db = memory_db
    _add(db, _owned("openai", "gpt-5.1"), age=timedelta(days=3))
    future = _owned("anthropic", "claude-future")
    future["console_generation_settings"]["version"] = 2
    _add(db, future, age=timedelta(minutes=1))
    invalid = _owned("anthropic", "claude-hot")
    invalid["console_generation_settings"]["temperature"] = 9.0
    _add(db, invalid, age=timedelta(minutes=2))
    extra_key = _owned("anthropic", "claude-extra")
    extra_key["console_generation_settings"]["surprise"] = 1
    _add(db, extra_key, age=timedelta(minutes=3))
    _add(db, {"other": 1}, age=timedelta(minutes=4))
    _add(db, None, age=timedelta(minutes=5))
    _add(db, _owned("openai", None), age=timedelta(minutes=6))
    blank_model = _owned("openai", "gpt-5.1")
    blank_model["console_generation_settings"]["model"] = ""
    _add(db, blank_model, age=timedelta(minutes=6, seconds=30))
    broken = _add(db, {"x": 1}, age=timedelta(minutes=7))
    with db.transaction() as conn:
        conn.execute(
            "UPDATE conversations SET metadata = ? WHERE id = ?", ("{not json", broken)
        )
    # Workspace chats are read only while open (live sessions), never here.
    _add(
        db,
        _owned("deepseek", "deepseek-reasoner"),
        age=timedelta(minutes=8),
        scope_type="workspace",
        workspace_id="ws-1",
    )

    uses = persisted_model_pair_uses(db)

    assert _pairs(uses) == [("openai", "gpt-5.1")]
    assert uses[0].last_used == NOW - timedelta(days=3)


def test_persisted_read_is_bounded_to_fifty_with_one_metadata_batch(memory_db) -> None:
    """AC#3: one listing capped at 50 and exactly one batched metadata read."""
    db = memory_db
    for index in range(60):
        _add(db, _owned("openai", f"m{index}"), age=timedelta(minutes=60 - index))
    calls: dict[str, list] = {"list": [], "meta": []}
    listing = db.list_all_active_conversations
    batch = db.get_conversations_metadata_by_ids

    def spy_list(*args, **kwargs):
        calls["list"].append((args, kwargs))
        return listing(*args, **kwargs)

    def spy_meta(ids):
        ids = list(ids)
        calls["meta"].append(ids)
        return batch(ids)

    db.list_all_active_conversations = spy_list
    db.get_conversations_metadata_by_ids = spy_meta

    uses = persisted_model_pair_uses(db)

    assert RECENT_CONVERSATION_LIMIT == 50
    assert calls["list"] == [((), {"limit": 50})]
    assert len(calls["meta"]) == 1 and len(calls["meta"][0]) == 50
    assert _pairs(uses) == [("openai", f"m{index}") for index in range(59, 9, -1)]


async def test_recents_merge_open_sessions_newest_first_and_distinct(file_db) -> None:
    """AC#1: open sessions (temporary included) + global chats, newest first."""
    db = file_db
    _add(db, _owned("openai", "gpt-5.1"), age=timedelta(days=3))
    _add(db, _owned("deepseek", "deepseek-reasoner"), age=timedelta(days=5))
    open_id = _add(db, _owned("openai", "stale-model"), age=timedelta(hours=1))
    sessions = [
        # An open persisted chat contributes only through its live session.
        _session(
            "ollama", "qwen3:32b", age=timedelta(minutes=1), conversation_id=open_id
        ),
        _session(
            "anthropic", "claude-sonnet-4-5", age=timedelta(hours=2), ephemeral=True
        ),
        _session("openai", "gpt-5.1", age=timedelta(days=1)),
        ConsoleChatSession(settings=None),
        _session("openai", None, age=timedelta(seconds=5)),
    ]

    uses = await read_recent_model_pairs(db, sessions)

    assert _pairs(uses) == [
        ("ollama", "qwen3:32b"),
        ("anthropic", "claude-sonnet-4-5"),
        ("openai", "gpt-5.1"),
        ("deepseek", "deepseek-reasoner"),
    ]
    assert uses[2].last_used == NOW - timedelta(days=1)


async def test_an_open_chat_with_an_unchanged_pair_counts_at_its_live_use(
    file_db,
) -> None:
    """AC#1/#2: a message sent in an open saved chat is a use of its pair.

    Sending moves the session's ``updated_at`` but never the row's
    ``last_modified`` (``add_message`` does not touch ``conversations``), so
    while a chat is open its live session is the only truth about its last
    use, even when its pair still equals the stored snapshot.
    """
    db = file_db
    open_id = _add(db, _owned("openai", "gpt-5.1"), age=timedelta(days=1))
    _add(db, _owned("anthropic", "claude-sonnet-4-5"), age=timedelta(hours=3))
    live = [
        _session("openai", "gpt-5.1", age=timedelta(minutes=1), conversation_id=open_id)
    ]

    uses = await read_recent_model_pairs(db, live)

    assert _pairs(uses) == [("openai", "gpt-5.1"), ("anthropic", "claude-sonnet-4-5")]
    assert uses[0].used_label(NOW) == "used 1m ago"
    chat_c = ("ollama", "qwen3:32b")
    assert PreviousPairMemory().previous("chat-c", chat_c, uses).pair == (
        "openai",
        "gpt-5.1",
    )


async def test_a_chat_reopened_from_sessions_keeps_its_stored_last_use(file_db) -> None:
    """AC#1: opening an old chat is not a use of its pair.

    The real reopen path (Sessions, launch wake) stamps the new session's
    ``updated_at`` from the row's ``last_modified``, as a chat restored at
    startup keeps its saved ``updated_at``. The reopened chat therefore
    neither jumps to the top of RECENT nor becomes another chat's PREVIOUS
    fallback.
    """
    db = file_db
    old_id = _add(db, _owned("openai", "gpt-5.1"), age=timedelta(days=3))
    _add(db, _owned("deepseek", "deepseek-reasoner"), age=timedelta(days=1))
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))

    reopened = await hydrate_console_session(
        app=SimpleNamespace(chachanotes_db=db),
        store=store,
        conversation_id=old_id,
        tree=ChatConversationService(db).get_conversation_tree(old_id),
        settings=ConsoleSessionSettings(provider="openai", model="gpt-5.1"),
    )
    uses = await read_recent_model_pairs(db, [reopened])

    assert datetime.fromisoformat(reopened.updated_at) == NOW - timedelta(days=3)
    assert _pairs(uses) == [("deepseek", "deepseek-reasoner"), ("openai", "gpt-5.1")]
    assert uses[1].last_used == NOW - timedelta(days=3)
    other_chat = ("anthropic", "claude-sonnet-4-5")
    assert PreviousPairMemory().previous("other", other_chat, uses).pair == (
        "deepseek",
        "deepseek-reasoner",
    )


async def test_a_pair_switched_in_an_open_chat_is_used_now(file_db) -> None:
    """Qodo #2947: switching an open saved chat from A to B is a use of B.

    The switch moves the row's ``last_modified``, but RECENT reads an open
    chat only through its live session, so the session's last use must move
    too: B reads "used just now" and sorts first, ahead of a newer chat.
    """
    from tldw_chatbook.Chat.console_context_policy import (
        ConsoleContextPolicyOverrides,
    )
    from tldw_chatbook.Chat.console_settings_apply import (
        ConsoleSettingsAction,
        ConsoleSettingsDraftState,
        ConsoleSettingsSubmission,
        ConsoleSettingsSurface,
    )

    db = file_db
    old_id = _add(db, _owned("openai", "gpt-5.1"), age=timedelta(days=3))
    _add(db, _owned("deepseek", "deepseek-reasoner"), age=timedelta(days=1))
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    session = await hydrate_console_session(
        app=SimpleNamespace(chachanotes_db=db),
        store=store,
        conversation_id=old_id,
        tree=ChatConversationService(db).get_conversation_tree(old_id),
        settings=ConsoleSessionSettings(provider="openai", model="gpt-5.1"),
    )
    switch = ConsoleSessionSettings(provider="anthropic", model="claude-sonnet-4-5")

    store.commit_console_settings_live(
        ConsoleSettingsSubmission(
            submission_id="switch-a-to-b",
            action=ConsoleSettingsAction.APPLY_TO_CHAT,
            surface=ConsoleSettingsSurface.QUICK_POPOVER,
            origin=store.capture_console_settings_origin(session.id),
            draft=ConsoleSettingsDraftState(
                settings=switch,
                context_policy_overrides=ConsoleContextPolicyOverrides(),
                field_drafts=(),
                model_drafts=(),
                endpoint_draft=None,
            ),
            user_display_name_override=None,
            default_field_mask=frozenset(),
        )
    )
    uses = await read_recent_model_pairs(db, store.sessions())

    assert _pairs(uses) == [
        ("anthropic", "claude-sonnet-4-5"),
        ("deepseek", "deepseek-reasoner"),
    ]
    assert uses[0].used_label(datetime.now(UTC)) == "used just now"


async def test_read_runs_off_the_ui_thread_without_blocking_the_loop(file_db) -> None:
    """AC#3: the DB read runs in a worker thread while the loop keeps running."""
    db = file_db
    _add(db, _owned("openai", "gpt-5.1"), age=timedelta(days=3))
    loop_thread = threading.get_ident()
    release = threading.Event()
    read_threads: list[int] = []
    listing = db.list_all_active_conversations

    def slow_list(*args, **kwargs):
        read_threads.append(threading.get_ident())
        release.wait(5)
        return listing(*args, **kwargs)

    db.list_all_active_conversations = slow_list
    live = [_session("ollama", "qwen3:32b", age=timedelta(minutes=1))]

    task = asyncio.create_task(read_recent_model_pairs(db, live))
    ticks = 0
    while not read_threads and ticks < 500:
        ticks += 1
        await asyncio.sleep(0.001)
    for _ in range(5):
        await asyncio.sleep(0.001)
        ticks += 1
    assert read_threads and not task.done()
    release.set()
    uses = await asyncio.wait_for(task, 5)

    assert read_threads[0] != loop_thread
    assert _pairs(uses) == [("ollama", "qwen3:32b"), ("openai", "gpt-5.1")]


async def test_a_failed_read_keeps_the_open_session_pairs(memory_db) -> None:
    """A DB failure never empties RECENT: open sessions still contribute."""

    def broken(*_args, **_kwargs):
        raise RuntimeError("database_maintenance_in_progress")

    memory_db.list_all_active_conversations = broken
    live = [_session("ollama", "qwen3:32b", age=timedelta(minutes=1))]

    assert _pairs(await read_recent_model_pairs(memory_db, live)) == [
        ("ollama", "qwen3:32b")
    ]
    assert _pairs(await read_recent_model_pairs(None, live)) == [
        ("ollama", "qwen3:32b")
    ]


def test_previous_pair_toggles_between_the_last_two_pairs_of_a_chat() -> None:
    """AC#2: A->B makes PREVIOUS A; switching back makes it B."""
    memory = PreviousPairMemory()
    a, b = ("openai", "gpt-5.1"), ("anthropic", "claude-sonnet-4-5")
    recent = [ModelPairUse("ollama", "qwen3:32b", NOW)]

    memory.record_switch("chat-1", a, b, now=NOW)
    first = memory.previous("chat-1", b, recent)
    memory.record_switch("chat-1", b, a, now=NOW + timedelta(minutes=1))
    second = memory.previous("chat-1", a, recent)

    assert first.pair == a and first.in_this_chat
    assert second.pair == b and second.last_used == NOW + timedelta(minutes=1)
    # Another chat has its own memory and falls back to RECENT.
    assert memory.previous("chat-2", a, recent).pair == ("ollama", "qwen3:32b")


def test_previous_pair_falls_back_to_the_newest_other_recent_pair() -> None:
    """AC#2: with no switch in this chat, PREVIOUS is the newest non-current pair."""
    memory = PreviousPairMemory()
    current = ("ollama", "qwen3:32b")
    recent = [
        ModelPairUse("ollama", "qwen3:32b", NOW),
        ModelPairUse("openai", "gpt-5.1", NOW - timedelta(hours=1)),
        ModelPairUse("deepseek", "deepseek-reasoner", NOW - timedelta(days=1)),
    ]

    assert memory.previous("chat-1", current, recent).pair == ("openai", "gpt-5.1")
    # A no-op or model-less switch is not a pair and is not remembered.
    memory.record_switch("chat-1", current, current, now=NOW)
    memory.record_switch("chat-1", ("openai", None), current, now=NOW)
    assert memory.previous("chat-1", current, recent).pair == ("openai", "gpt-5.1")
    # A remembered pair equal to the current one falls back as well.
    memory.record_switch("chat-1", current, ("openai", "gpt-5.1"), now=NOW)
    assert memory.previous("chat-1", current, recent[:1]) is None


def test_each_pair_says_when_it_was_last_used() -> None:
    """AC#1: every row can print its last use."""
    assert (
        ModelPairUse("o", "m", NOW - timedelta(hours=2)).used_label(NOW)
        == "used 2h ago"
    )
    assert (
        ModelPairUse("o", "m", NOW - timedelta(days=3)).used_label(NOW) == "used 3d ago"
    )
    assert (
        ModelPairUse("o", "m", NOW - timedelta(seconds=20)).used_label(NOW)
        == "used just now"
    )
    assert (
        ModelPairUse("o", "m", NOW - timedelta(hours=2), in_this_chat=True).used_label(
            NOW
        )
        == "used 2h ago in this chat"
    )


async def test_a_catalog_resolves_off_the_ui_thread_and_is_remembered_on_it() -> None:
    """TASK-33004.4 first-open cost: the real catalog merge is synchronous under
    its async wrapper (13 ready providers held the loop 110 ms in one block),
    so it runs in a worker thread; the options are remembered on the UI thread."""
    loop_thread = threading.get_ident()
    providers = {"Ollama": ["qwen3:32b", "llama3.1:8b"]}
    merge_threads: list[int] = []
    remembered: list[tuple[int, str, list[str]]] = []

    def saved_catalog() -> dict[str, list[str]]:
        merge_threads.append(threading.get_ident())
        return dict(providers)

    local = LocalLLMProviderCatalogService(
        provider_catalog_loader=saved_catalog, settings_loader=dict, environ={}
    )
    screen = SimpleNamespace(
        app_instance=SimpleNamespace(
            llm_provider_catalog_scope_service=LLMProviderCatalogScopeService(
                local_service=local, server_service=None
            )
        ),
        _remember_console_model_options=lambda provider, options: remembered.append(
            (threading.get_ident(), provider, [option.model_id for option in options])
        ),
    )

    models = await load_provider_catalog(screen, providers, "ollama")

    assert models == ["qwen3:32b", "llama3.1:8b"]
    assert merge_threads and loop_thread not in merge_threads
    assert remembered == [(loop_thread, "ollama", models)]
