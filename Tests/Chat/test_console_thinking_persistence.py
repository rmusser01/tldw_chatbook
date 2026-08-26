"""Selected Console generations retain bounded thinking as one owner."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from Tests.Chat.test_console_dispatch_recovery import _database, _insert, _acceptance
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
    ConsoleVariant,
    ConsoleVariantSet,
)
from tldw_chatbook.Chat.console_chat_store import (
    ConsoleChatStore,
    ConsoleThinkingCompatibilityError,
)
from tldw_chatbook.Chat.thinking_blocks import (
    DisplayableThinkingBlock,
    ThinkingEnvelope,
    dump_thinking_blocks_json,
    parse_thinking_blocks_json,
)
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationRound,
    ProviderContinuationCheckpoint,
    dump_provider_continuation_json,
)


def _thinking(text: str, *, status: str = "complete") -> ThinkingEnvelope:
    return ThinkingEnvelope(
        (
            DisplayableThinkingBlock(
                block_id="reasoning-1",
                round_ordinal=0,
                provider="llama_cpp",
                model="test-model",
                protocol="chat_completions",
                source_format="think_tag",
                status=status,
                text=text,
            ),
        )
    )


def _restored_store(db, conversation_id: str) -> tuple[ConsoleChatStore, str]:
    rows = db.get_messages_for_conversation(conversation_id, limit=100)
    nodes = [
        ConsoleChatMessage(
            id=str(row["id"]),
            role=ConsoleMessageRole(str(row["role"])),
            content=str(row.get("content") or ""),
            persisted_message_id=str(row["id"]),
            parent_message_id=row.get("parent_message_id"),
        )
        for row in rows
    ]
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    session = store.restore_persisted_session(
        title="thinking",
        workspace_id=None,
        persisted_conversation_id=conversation_id,
        all_nodes=nodes,
        active_leaf_persisted_id="assistant-1",
    )
    return store, session.id


def _continuation(content: str) -> ProviderContinuationCheckpoint:
    return ProviderContinuationCheckpoint(
        schema_version=1,
        checkpoint_revision=1,
        provider="moonshot",
        protocol="chat_completions",
        model="kimi-k3",
        api_base_url="https://api.moonshot.ai/v1",
        state="complete",
        rounds=(
            ContinuationRound(
                assistant_content=content,
                reasoning_blocks=("private reasoning",),
                calls=(),
            ),
        ),
    )


def test_restore_hydrates_supported_thinking(tmp_path: Path) -> None:
    db, conversation_id, repository = _database(tmp_path / "supported.sqlite")
    _insert(db, repository, _acceptance(conversation_id))
    canonical = dump_thinking_blocks_json(_thinking("restored reasoning"))
    db.get_connection().execute(
        "UPDATE messages SET thinking_blocks_json = ? WHERE id = 'assistant-1'",
        (canonical,),
    )

    store, _session_id = _restored_store(db, conversation_id)
    restored = store.get_message("assistant-1")

    assert restored.thinking == _thinking("restored reasoning")
    assert restored.opaque_thinking_json is None
    assert restored.thinking_actions_enabled is True


def test_restore_preserves_unknown_opaque_and_blocks_generation_mutations(
    tmp_path: Path,
) -> None:
    db, conversation_id, repository = _database(tmp_path / "unknown.sqlite")
    _insert(db, repository, _acceptance(conversation_id))
    raw = '{ "version" : 99, "future" : {"secret":"value"} }'
    db.get_connection().execute(
        "UPDATE messages SET thinking_blocks_json = ? WHERE id = 'assistant-1'",
        (raw,),
    )
    store, _session_id = _restored_store(db, conversation_id)
    restored = store.get_message("assistant-1")

    assert restored.opaque_thinking_json == raw
    assert restored.thinking_warning == "Thinking data version is unsupported."
    assert restored.thinking_actions_enabled is False
    assert "secret" not in repr(restored)
    with pytest.raises(
        ConsoleThinkingCompatibilityError, match="newer thinking format"
    ):
        store.begin_variant_stream("assistant-1")
    with pytest.raises(
        ConsoleThinkingCompatibilityError, match="upgrade before editing"
    ):
        store.update_message_content("assistant-1", "must not commit")
    assert store.get_message("assistant-1").content == ""

    store.set_message_feedback("assistant-1", "up")
    durable = (
        db.get_connection()
        .execute("SELECT thinking_blocks_json FROM messages WHERE id = 'assistant-1'")
        .fetchone()
    )
    assert durable["thinking_blocks_json"] == raw


def test_persist_selected_generation_replaces_projection_and_refreshes_version(
    tmp_path: Path,
) -> None:
    db, conversation_id, repository = _database(tmp_path / "projection.sqlite")
    _insert(db, repository, _acceptance(conversation_id))
    db.get_connection().execute(
        "UPDATE messages SET content = 'old', assistant_generation_state = 'complete' "
        "WHERE id = 'assistant-1'"
    )
    store, _session_id = _restored_store(db, conversation_id)
    message = store._message_or_raise("assistant-1")
    message.content = "new answer"
    message.thinking = _thinking("new reasoning")
    message.assistant_generation_state = "complete"

    committed = store.persist_selected_generation("assistant-1")
    row = db.get_message_by_id("assistant-1")

    assert committed is True
    assert row["content"] == "new answer"
    assert row["thinking_blocks_json"] == dump_thinking_blocks_json(message.thinking)
    assert message.provider_continuation_message_version == row["version"]


def test_persisted_variant_swipe_uses_current_row_version_for_every_selection(
    tmp_path: Path,
) -> None:
    db, conversation_id, repository = _database(tmp_path / "swipe-version.sqlite")
    _insert(db, repository, _acceptance(conversation_id))
    original_thinking = dump_thinking_blocks_json(_thinking("original reasoning"))
    db.get_connection().execute(
        "UPDATE messages SET content = 'original answer', thinking_blocks_json = ?, "
        "assistant_generation_state = 'complete' WHERE id = 'assistant-1'",
        (original_thinking,),
    )
    store, _session_id = _restored_store(db, conversation_id)

    store.begin_variant_stream("assistant-1")
    store.replace_message_thinking("assistant-1", _thinking("new reasoning"))
    store.append_stream_chunk("assistant-1", "new answer")
    store.finalize_variant_stream("assistant-1")
    after_new = db.get_message_by_id("assistant-1")

    restored = store.select_variant("assistant-1", 0)
    after_original = db.get_message_by_id("assistant-1")

    assert restored.content == "original answer"
    assert restored.thinking == _thinking("original reasoning")
    assert after_original["content"] == "original answer"
    assert after_original["thinking_blocks_json"] == original_thinking
    assert after_original["version"] == after_new["version"] + 1
    assert restored.provider_continuation_message_version == after_original["version"]


def test_failed_variant_projection_preserves_live_and_durable_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db, conversation_id, repository = _database(tmp_path / "swipe-failure.sqlite")
    _insert(db, repository, _acceptance(conversation_id))
    original_thinking = dump_thinking_blocks_json(_thinking("original reasoning"))
    db.get_connection().execute(
        "UPDATE messages SET content = 'original answer', thinking_blocks_json = ?, "
        "assistant_generation_state = 'complete' WHERE id = 'assistant-1'",
        (original_thinking,),
    )
    store, _session_id = _restored_store(db, conversation_id)
    live = store._message_or_raise("assistant-1")
    live.variants = ConsoleVariantSet.from_generations(
        turn_id="assistant-1",
        generations=[
            ConsoleVariant(
                content="original answer",
                thinking=_thinking("original reasoning"),
                assistant_generation_state="complete",
            ),
            ConsoleVariant(
                content="alternative answer",
                thinking=_thinking("alternative reasoning"),
                assistant_generation_state="complete",
            ),
        ],
        selected_index=0,
    )

    def fail_projection(**_kwargs: object) -> int:
        raise RuntimeError("injected projection failure")

    monkeypatch.setattr(
        store.persistence,
        "replace_assistant_generation_projection",
        fail_projection,
    )

    with pytest.raises(RuntimeError, match="injected projection failure"):
        store.select_variant("assistant-1", 1)

    unchanged = store.get_message("assistant-1")
    durable = db.get_message_by_id("assistant-1")
    assert unchanged.variants is not None
    assert unchanged.variants.selected_index == 0
    assert unchanged.content == "original answer"
    assert unchanged.thinking == _thinking("original reasoning")
    assert durable["content"] == "original answer"
    assert durable["thinking_blocks_json"] == original_thinking


@pytest.mark.parametrize("failure_mode", ["incompatible", "writer"])
def test_select_variant_failure_preserves_pending_stream_and_full_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    db, conversation_id, repository = _database(
        tmp_path / f"pending-swipe-{failure_mode}.sqlite"
    )
    _insert(db, repository, _acceptance(conversation_id))
    usage = ProviderUsage(uncached_input=2, output=3)
    continuation = _continuation("original answer")
    db.get_connection().execute(
        "UPDATE messages SET content = 'original answer', thinking_blocks_json = ?, "
        "usage_json = ?, provider_continuation_json = ?, "
        "assistant_generation_state = 'complete' WHERE id = 'assistant-1'",
        (
            dump_thinking_blocks_json(_thinking("original reasoning")),
            usage.to_json(),
            dump_provider_continuation_json(continuation),
        ),
    )
    store, _session_id = _restored_store(db, conversation_id)
    live = store._message_or_raise("assistant-1")
    target = ConsoleVariant(
        content="alternative answer",
        thinking=_thinking("alternative reasoning"),
        assistant_generation_state="complete",
    )
    if failure_mode == "incompatible":
        target = ConsoleVariant(
            content="future answer",
            opaque_thinking_json='{"version":99,"secret":"future"}',
            thinking_actions_enabled=False,
        )
    live.variants = ConsoleVariantSet.from_generations(
        turn_id="assistant-1",
        generations=[store._generation_variant(live), target],
        selected_index=0,
    )
    live.status = "streaming"
    store.append_stream_chunk("assistant-1", " pending chunk")
    before_generation = store._generation_variant(live)
    before_buffer = tuple(store._stream_chunks_by_message["assistant-1"])
    before_count = store._stream_materialized_counts.get("assistant-1")
    before_row = dict(db.get_message_by_id("assistant-1"))

    if failure_mode == "writer":

        def fail_projection(**_kwargs: object) -> int:
            raise RuntimeError("injected projection failure")

        monkeypatch.setattr(
            store.persistence,
            "replace_assistant_generation_projection",
            fail_projection,
        )
        expected_error = RuntimeError
    else:
        expected_error = ConsoleThinkingCompatibilityError

    with pytest.raises(expected_error):
        store.select_variant("assistant-1", 1)

    unchanged = store._message_or_raise("assistant-1")
    after_generation = store._generation_variant(unchanged)
    assert replace(after_generation, id=before_generation.id) == before_generation
    assert unchanged.status == "streaming"
    assert unchanged.variants is not None
    assert unchanged.variants.selected_index == 0
    assert tuple(store._stream_chunks_by_message["assistant-1"]) == before_buffer
    assert store._stream_materialized_counts.get("assistant-1") == before_count
    assert dict(db.get_message_by_id("assistant-1")) == before_row


def test_successful_variant_projection_discards_superseded_pending_stream(
    tmp_path: Path,
) -> None:
    db, conversation_id, repository = _database(tmp_path / "pending-swipe-ok.sqlite")
    _insert(db, repository, _acceptance(conversation_id))
    db.get_connection().execute(
        "UPDATE messages SET content = 'original answer', thinking_blocks_json = ?, "
        "assistant_generation_state = 'complete' WHERE id = 'assistant-1'",
        (dump_thinking_blocks_json(_thinking("original reasoning")),),
    )
    store, _session_id = _restored_store(db, conversation_id)
    live = store._message_or_raise("assistant-1")
    target = ConsoleVariant(
        content="alternative answer",
        thinking=_thinking("alternative reasoning"),
        assistant_generation_state="complete",
    )
    live.variants = ConsoleVariantSet.from_generations(
        turn_id="assistant-1",
        generations=[store._generation_variant(live), target],
        selected_index=0,
    )
    live.status = "streaming"
    store.append_stream_chunk("assistant-1", " superseded chunk")

    selected = store.select_variant("assistant-1", 1)
    row = db.get_message_by_id("assistant-1")

    assert selected.content == "alternative answer"
    assert selected.thinking == _thinking("alternative reasoning")
    assert selected.status == "complete"
    assert selected.variants is not None
    assert selected.variants.selected_index == 1
    assert "assistant-1" not in store._stream_chunks_by_message
    assert "assistant-1" not in store._stream_materialized_counts
    assert row["content"] == "alternative answer"
    assert row["thinking_blocks_json"] == dump_thinking_blocks_json(
        _thinking("alternative reasoning")
    )


@pytest.mark.parametrize("action", ["add", "select"])
def test_generation_outbox_candidate_owns_exact_target_variant_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
) -> None:
    db, conversation_id, repository = _database(
        tmp_path / f"candidate-variants-{action}.sqlite"
    )
    _insert(db, repository, _acceptance(conversation_id))
    db.get_connection().execute(
        "UPDATE messages SET content = 'original answer', thinking_blocks_json = ?, "
        "assistant_generation_state = 'complete' WHERE id = 'assistant-1'",
        (dump_thinking_blocks_json(_thinking("original reasoning")),),
    )
    store, _session_id = _restored_store(db, conversation_id)
    live = store._message_or_raise("assistant-1")
    if action == "select":
        live.variants = ConsoleVariantSet.from_generations(
            turn_id="assistant-1",
            generations=[
                store._generation_variant(live),
                ConsoleVariant(
                    content="alternative answer",
                    thinking=_thinking("alternative reasoning"),
                    assistant_generation_state="complete",
                ),
            ],
            selected_index=0,
        )
    captured: list[tuple[str, int, tuple[ConsoleVariant, ...]]] = []

    def capture_candidate(candidate: ConsoleChatMessage, **_kwargs: object) -> None:
        owner = store._message_or_raise("assistant-1")
        assert owner.content == "original answer"
        assert owner.thinking == _thinking("original reasoning")
        if action == "add":
            assert owner.variants is None
        else:
            assert owner.variants is not None
            assert owner.variants.selected_index == 0
        assert candidate.variants is not None
        captured.append(
            (
                candidate.variants.turn_id,
                candidate.variants.selected_index,
                tuple(candidate.variants.variants),
            )
        )

    monkeypatch.setattr(store, "_enqueue_sync_v2_message_if_ready", capture_candidate)

    if action == "add":
        result = store.add_variant("assistant-1", "manual alternative")
    else:
        result = store.select_variant("assistant-1", 1)

    assert len(captured) == 1
    assert result.variants is not None
    assert captured[0] == (
        result.variants.turn_id,
        result.variants.selected_index,
        tuple(result.variants.variants),
    )


def test_add_variant_writer_failure_keeps_live_variant_owner_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db, conversation_id, repository = _database(tmp_path / "candidate-add-fail.sqlite")
    _insert(db, repository, _acceptance(conversation_id))
    db.get_connection().execute(
        "UPDATE messages SET content = 'original answer', thinking_blocks_json = ?, "
        "assistant_generation_state = 'complete' WHERE id = 'assistant-1'",
        (dump_thinking_blocks_json(_thinking("original reasoning")),),
    )
    store, _session_id = _restored_store(db, conversation_id)
    live = store._message_or_raise("assistant-1")
    before_generation = store._generation_variant(live)
    before_row = dict(db.get_message_by_id("assistant-1"))

    def fail_projection(**_kwargs: object) -> int:
        raise RuntimeError("injected projection failure")

    monkeypatch.setattr(
        store.persistence,
        "replace_assistant_generation_projection",
        fail_projection,
    )

    with pytest.raises(RuntimeError, match="injected projection failure"):
        store.add_variant("assistant-1", "manual alternative")

    unchanged = store._message_or_raise("assistant-1")
    after_generation = store._generation_variant(unchanged)
    assert replace(after_generation, id=before_generation.id) == before_generation
    assert unchanged.variants is None
    assert dict(db.get_message_by_id("assistant-1")) == before_row


@pytest.mark.parametrize("action", ["add", "select"])
def test_content_only_fallback_rejects_clearing_current_generation_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
) -> None:
    db, conversation_id, repository = _database(
        tmp_path / f"content-only-{action}.sqlite"
    )
    _insert(db, repository, _acceptance(conversation_id))
    original_thinking = dump_thinking_blocks_json(_thinking("original reasoning"))
    db.get_connection().execute(
        "UPDATE messages SET content = 'original answer', thinking_blocks_json = ?, "
        "assistant_generation_state = 'complete' WHERE id = 'assistant-1'",
        (original_thinking,),
    )
    store, _session_id = _restored_store(db, conversation_id)
    live = store._message_or_raise("assistant-1")
    target = ConsoleVariant(
        content="manual alternative",
        assistant_generation_state="complete",
    )
    if action == "select":
        live.variants = ConsoleVariantSet.from_generations(
            turn_id="assistant-1",
            generations=[store._generation_variant(live), target],
            selected_index=0,
        )
    before_generation = store._generation_variant(live)
    before_row = dict(db.get_message_by_id("assistant-1"))
    monkeypatch.setattr(
        store.persistence,
        "replace_assistant_generation_projection",
        None,
    )

    with pytest.raises(RuntimeError, match="projection persistence is unavailable"):
        if action == "add":
            store.add_variant("assistant-1", "manual alternative")
        else:
            store.select_variant("assistant-1", 1)

    unchanged = store._message_or_raise("assistant-1")
    after_generation = store._generation_variant(unchanged)
    assert replace(after_generation, id=before_generation.id) == before_generation
    assert unchanged.variants is None or unchanged.variants.selected_index == 0
    assert dict(db.get_message_by_id("assistant-1")) == before_row


def test_add_variant_persists_one_evidence_free_complete_generation(
    tmp_path: Path,
) -> None:
    db, conversation_id, repository = _database(tmp_path / "manual-variant.sqlite")
    _insert(db, repository, _acceptance(conversation_id))
    old_thinking = dump_thinking_blocks_json(_thinking("old reasoning"))
    db.get_connection().execute(
        "UPDATE messages SET content = 'old answer', thinking_blocks_json = ?, "
        "usage_json = ?, assistant_generation_state = 'complete' "
        "WHERE id = 'assistant-1'",
        (old_thinking, ProviderUsage(uncached_input=2, output=3).to_json()),
    )
    store, _session_id = _restored_store(db, conversation_id)

    added = store.add_variant("assistant-1", "manual alternative")
    row = db.get_message_by_id("assistant-1")

    assert added.content == "manual alternative"
    assert added.thinking is None
    assert added.usage is None
    assert added.provider_continuation is None
    assert added.assistant_generation_state == "complete"
    assert row["content"] == "manual alternative"
    assert row["thinking_blocks_json"] is None
    assert row["usage_json"] is None
    assert row["provider_continuation_json"] is None
    assert row["assistant_generation_state"] == "complete"


def test_add_variant_rejects_unknown_thinking_before_live_mutation(
    tmp_path: Path,
) -> None:
    db, conversation_id, repository = _database(tmp_path / "manual-unknown.sqlite")
    _insert(db, repository, _acceptance(conversation_id))
    raw = '{ "version" : 99, "future" : {"secret":"value"} }'
    db.get_connection().execute(
        "UPDATE messages SET content = 'durable answer', thinking_blocks_json = ? "
        "WHERE id = 'assistant-1'",
        (raw,),
    )
    store, _session_id = _restored_store(db, conversation_id)

    with pytest.raises(ConsoleThinkingCompatibilityError):
        store.add_variant("assistant-1", "must not apply")

    unchanged = store.get_message("assistant-1")
    assert unchanged.content == "durable answer"
    assert unchanged.variants is None
    assert db.get_message_by_id("assistant-1")["thinking_blocks_json"] == raw


@pytest.mark.parametrize(
    "raw",
    [
        '{ "version" : 99, "future" : {"secret":"value"} }',
        json.dumps({"version": 1, "blocks": [{"text": "do-not-leak"}]}),
    ],
)
def test_prepare_retry_rejects_unreadable_thinking_before_provider_visible_mutation(
    tmp_path: Path, raw: str
) -> None:
    db, conversation_id, repository = _database(tmp_path / "retry-blocked.sqlite")
    _insert(db, repository, _acceptance(conversation_id))
    db.get_connection().execute(
        "UPDATE messages SET content = 'failed answer', thinking_blocks_json = ?, "
        "assistant_generation_state = 'failed' WHERE id = 'assistant-1'",
        (raw,),
    )
    store, _session_id = _restored_store(db, conversation_id)
    live = store._message_or_raise("assistant-1")
    live.status = "failed"

    with pytest.raises(ConsoleThinkingCompatibilityError):
        store.prepare_message_retry("assistant-1")

    unchanged = store.get_message("assistant-1")
    assert unchanged.content == "failed answer"
    assert unchanged.status == "failed"
    assert "assistant-1" not in store._failed_retry_message_ids


def test_first_terminal_create_persists_complete_generation_in_initial_row(
    tmp_path: Path,
) -> None:
    db, conversation_id, repository = _database(tmp_path / "atomic-create.sqlite")
    _insert(db, repository, _acceptance(conversation_id))
    store, session_id = _restored_store(db, conversation_id)
    message = store.append_message(
        session_id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        persist=True,
        defer_terminal_persistence=True,
    )
    live = store._message_or_raise(message.id)
    live.thinking = _thinking("atomic reasoning")
    live.provider_continuation = _continuation("atomic answer")
    live.assistant_generation_state = "streaming"
    usage = ProviderUsage(uncached_input=7, output=11)
    store.set_message_usage(message.id, usage)
    store.append_stream_chunk(message.id, "atomic answer")

    completed = store.mark_message_complete(message.id)
    row = db.get_message_by_id(message.id)

    assert completed.persisted_message_id == message.id
    assert row["version"] == 1
    assert row["content"] == "atomic answer"
    assert row["thinking_blocks_json"] == dump_thinking_blocks_json(completed.thinking)
    assert row["usage_json"] == usage.to_json()
    assert row["provider_continuation_json"] == dump_provider_continuation_json(
        completed.provider_continuation
    )
    assert row["assistant_generation_state"] == "complete"
    assert completed.provider_continuation_message_version == 1


def test_failed_atomic_terminal_create_remains_unsaved_retryable_and_unrecorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db, conversation_id, repository = _database(tmp_path / "atomic-failure.sqlite")
    _insert(db, repository, _acceptance(conversation_id))
    service = ChatPersistenceService(db)
    store, session_id = _restored_store(db, conversation_id)
    store.persistence = service
    message = store.append_message(
        session_id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        persist=True,
        defer_terminal_persistence=True,
    )
    live = store._message_or_raise(message.id)
    live.thinking = _thinking("retryable reasoning")
    live.assistant_generation_state = "streaming"
    store.append_stream_chunk(message.id, "retryable answer")
    calls: list[dict[str, object]] = []

    def fail_create(**kwargs: object) -> str:
        calls.append(dict(kwargs))
        raise RuntimeError("injected create failure")

    monkeypatch.setattr(service, "create_message", fail_create)

    result = store.mark_message_complete(message.id)

    assert len(calls) == 1
    assert calls[0]["content"] == "retryable answer"
    assert calls[0]["thinking_blocks_json"] == dump_thinking_blocks_json(
        _thinking("retryable reasoning")
    )
    assert calls[0]["assistant_generation_state"] == "complete"
    assert db.get_message_by_id(message.id) is None
    assert result.persisted_message_id is None
    assert result.status == "streaming"
    assert result.assistant_generation_state == "streaming"
    assert message.id in store._pending_persistence_message_ids
    assert store.message_completion_generation(message.id) == 0


def test_malformed_known_thinking_is_content_free_and_blocks_generation(
    tmp_path: Path,
) -> None:
    db, conversation_id, repository = _database(tmp_path / "malformed.sqlite")
    _insert(db, repository, _acceptance(conversation_id))
    raw = json.dumps({"version": 1, "blocks": [{"text": "do-not-leak"}]})
    db.get_connection().execute(
        "UPDATE messages SET thinking_blocks_json = ? WHERE id = 'assistant-1'",
        (raw,),
    )

    store, _session_id = _restored_store(db, conversation_id)
    restored = store.get_message("assistant-1")

    assert restored.thinking is None
    assert restored.opaque_thinking_json is None
    assert restored.thinking_warning is not None
    assert "do-not-leak" not in restored.thinking_warning
    assert restored.thinking_actions_enabled is False


def test_explicit_assistant_edit_clears_generation_provenance(tmp_path: Path) -> None:
    db, conversation_id, repository = _database(tmp_path / "edit.sqlite")
    _insert(db, repository, _acceptance(conversation_id))
    canonical = dump_thinking_blocks_json(_thinking("old reasoning"))
    db.get_connection().execute(
        "UPDATE messages SET content = 'old answer', thinking_blocks_json = ?, "
        "assistant_generation_state = 'complete' WHERE id = 'assistant-1'",
        (canonical,),
    )
    store, _session_id = _restored_store(db, conversation_id)

    edited = store.update_message_content("assistant-1", "human correction")
    row = db.get_message_by_id("assistant-1")

    assert edited.thinking is None
    assert edited.assistant_generation_state == "complete"
    assert row["content"] == "human correction"
    assert row["thinking_blocks_json"] is None
    assert row["provider_continuation_json"] is None
    assert row["assistant_generation_state"] == "complete"


@pytest.mark.parametrize(
    ("terminal", "expected_status"),
    [
        ("mark_message_complete", "complete"),
        ("mark_message_stopped", "stopped"),
        ("mark_message_failed", "failed"),
    ],
)
def test_normal_terminal_projects_paired_thinking_status(
    tmp_path: Path, terminal: str, expected_status: str
) -> None:
    db, conversation_id, repository = _database(
        tmp_path / f"terminal-{expected_status}.sqlite"
    )
    _insert(db, repository, _acceptance(conversation_id))
    db.get_connection().execute(
        "DELETE FROM console_dispatch_checkpoints WHERE assistant_message_id = "
        "'assistant-1'"
    )
    store, session_id = _restored_store(db, conversation_id)
    store._dispatch_recoveries_by_session.pop(session_id, None)
    live = store._message_or_raise("assistant-1")
    live.status = "streaming"
    live.content = "terminal answer"
    live.thinking = _thinking("terminal reasoning")
    live.assistant_generation_state = "streaming"

    getattr(store, terminal)("assistant-1")
    row = db.get_message_by_id("assistant-1")
    durable = parse_thinking_blocks_json(row["thinking_blocks_json"])

    assert row["assistant_generation_state"] == expected_status
    assert {block.status for block in durable.blocks} == {expected_status}
