"""Task 14 review fix round 3: pre-fingerprint commit reservation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
from typing import Any

import pytest

from Tests.Chat.test_console_durable_turn_acceptance import (
    _database_snapshot,
    _ready_store,
    _resume_persistence_retry,
)
from tldw_chatbook.Chat.console_turn_preparation import (
    ConsolePreparationPauseKind,
    ConsoleTurnPreparationState,
    preparation_actions,
)


@pytest.mark.parametrize(
    ("owner_body", "contender_body"),
    (
        ("exact captured draft", "forged competing body"),
        ("first reservation owns changed body", "exact captured draft"),
    ),
)
def test_pre_fingerprint_reservation_first_caller_owns_body(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    owner_body: str,
    contender_body: str,
) -> None:
    db, _service, store, preparation, acceptance = _ready_store(tmp_path)
    store.stage_durable_turn_owner_ids(
        preparation.session_id,
        preparation.preparation_id,
        user_message_id=acceptance.user_message_id,
        assistant_message_id=acceptance.assistant_message_id,
    )
    owner_acceptance = replace(acceptance, user_content=owner_body)
    contender_acceptance = replace(acceptance, user_content=contender_body)
    before = _database_snapshot(db)
    original_fingerprint = store._durable_acceptance_fingerprint
    entered = Event()
    release = Event()
    fingerprint_calls = 0

    def blocked_fingerprint(*args: Any, **kwargs: Any):
        nonlocal fingerprint_calls
        fingerprint_calls += 1
        if not entered.is_set():
            entered.set()
            assert release.wait(timeout=5)
        return original_fingerprint(*args, **kwargs)

    monkeypatch.setattr(store, "_durable_acceptance_fingerprint", blocked_fingerprint)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(store.commit_durable_turn, owner_acceptance)
        assert entered.wait(timeout=5)
        try:
            with pytest.raises(RuntimeError, match="already in flight"):
                store.commit_durable_turn(contender_acceptance)
            assert fingerprint_calls == 1
            assert _database_snapshot(db) == before
            current = store.preparation_by_id(preparation.preparation_id)
            assert current is not None
            assert current.state is ConsoleTurnPreparationState.COMMITTING
        finally:
            release.set()
        committed = future.result(timeout=5)

    assert fingerprint_calls == 1
    assert committed.user_message_id == acceptance.user_message_id
    row = (
        db.get_connection()
        .execute(
            "SELECT content FROM messages WHERE id = ?",
            (acceptance.user_message_id,),
        )
        .fetchone()
    )
    assert row is not None and row["content"] == owner_body
    assert (
        db.get_connection()
        .execute(
            "SELECT COUNT(*) FROM console_dispatch_checkpoints "
            "WHERE preparation_id = ?",
            (preparation.preparation_id,),
        )
        .fetchone()[0]
        == 1
    )
    assert store.commit_durable_turn(owner_acceptance) == committed


def test_fingerprint_canonicalization_owner_clears_reservation_for_clean_retry(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db, _service, store, preparation, acceptance = _ready_store(tmp_path)
    owners = store.stage_durable_turn_owner_ids(
        preparation.session_id,
        preparation.preparation_id,
        user_message_id=acceptance.user_message_id,
        assistant_message_id=acceptance.assistant_message_id,
    )
    before = _database_snapshot(db)
    original_fingerprint = store._durable_acceptance_fingerprint
    calls = 0

    def fail_once(*args: Any, **kwargs: Any):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("canonicalizer exploded")
        return original_fingerprint(*args, **kwargs)

    monkeypatch.setattr(store, "_durable_acceptance_fingerprint", fail_once)

    with pytest.raises(RuntimeError, match="canonicalizer exploded"):
        store.commit_durable_turn(acceptance)

    assert _database_snapshot(db) == before
    paused = store.preparation_by_id(preparation.preparation_id)
    assert paused is not None
    assert paused.state is ConsoleTurnPreparationState.PAUSED
    assert paused.pause_kind is ConsolePreparationPauseKind.PERSISTENCE
    assert preparation_actions(paused) == ("retry", "cancel")

    _resume_persistence_retry(store, paused)
    committed = store.commit_durable_turn(acceptance)

    assert committed.user_message_id == owners.user_message_id
    assert committed.assistant_message_id == owners.assistant_message_id
    assert calls == 2
    assert (
        db.get_connection()
        .execute(
            "SELECT COUNT(*) FROM console_dispatch_checkpoints "
            "WHERE preparation_id = ?",
            (preparation.preparation_id,),
        )
        .fetchone()[0]
        == 1
    )
