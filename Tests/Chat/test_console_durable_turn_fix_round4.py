"""Task 14 review fix round 4: atomic failed-commit ownership release."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event, get_ident
from typing import Any

import pytest

from Tests.Chat.test_console_durable_turn_acceptance import (
    _database_snapshot,
    _install_failure,
    _ready_store,
)
from tldw_chatbook.Chat.console_turn_preparation import (
    ConsolePreparationPauseKind,
    ConsolePreparationTransition,
    ConsoleTurnPreparationState,
)


def _result_or_exception(future: Future[Any]) -> object:
    try:
        return future.result(timeout=5)
    except Exception as exc:  # noqa: BLE001 - the public boundary is fail-closed
        return exc


@pytest.mark.parametrize("failure_kind", ("canonicalizer", "sqlite_commit"))
def test_failed_owner_cannot_leave_a_retry_commit_paused(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    """A failed owner must release and pause before Retry can reserve."""

    db, service, store, preparation, acceptance = _ready_store(tmp_path)
    before = _database_snapshot(db)

    def cleanup_failure() -> None:
        return None

    if failure_kind == "canonicalizer":
        original_fingerprint = store._durable_acceptance_fingerprint
        calls = 0

        def fail_once(*args: Any, **kwargs: Any):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("canonicalizer exploded")
            return original_fingerprint(*args, **kwargs)

        monkeypatch.setattr(store, "_durable_acceptance_fingerprint", fail_once)
    else:
        cleanup_failure = _install_failure(db, service, "commit", monkeypatch)

    owner_thread: list[int] = []
    owner_between_release_and_pause = Event()
    release_owner = Event()
    original_cas = store.compare_and_set_preparation

    def block_legacy_split_cleanup(session_id, transition):
        if (
            owner_thread
            and get_ident() == owner_thread[0]
            and transition.preparation_id == preparation.preparation_id
            and transition.expected_state is ConsoleTurnPreparationState.COMMITTING
            and transition.new_state is ConsoleTurnPreparationState.PAUSED
            and transition.pause_kind is ConsolePreparationPauseKind.PERSISTENCE
        ):
            owner_between_release_and_pause.set()
            assert release_owner.wait(timeout=5)
        return original_cas(session_id, transition)

    monkeypatch.setattr(
        store, "compare_and_set_preparation", block_legacy_split_cleanup
    )

    def fail_owner() -> object:
        owner_thread.append(get_ident())
        return store.commit_durable_turn(acceptance)

    with ThreadPoolExecutor(max_workers=2) as executor:
        owner = executor.submit(fail_owner)
        # The broken implementation reaches this hook after removing its
        # reservation and unlocking. The fixed composite never calls the public
        # CAS, so the owner simply finishes in PAUSED.
        owner_between_release_and_pause.wait(timeout=1)
        cleanup_failure()
        contender = executor.submit(store.commit_durable_turn, acceptance)
        contender_outcome = _result_or_exception(contender)
        release_owner.set()
        owner_outcome = _result_or_exception(owner)

    assert isinstance(owner_outcome, Exception)
    paused = store.preparation_by_id(preparation.preparation_id)
    assert paused is not None
    assert paused.state is ConsoleTurnPreparationState.PAUSED
    assert paused.pause_kind is ConsolePreparationPauseKind.PERSISTENCE
    checkpoint_count = (
        db.get_connection()
        .execute(
            "SELECT COUNT(*) FROM console_dispatch_checkpoints "
            "WHERE preparation_id = ?",
            (preparation.preparation_id,),
        )
        .fetchone()[0]
    )
    assert isinstance(contender_outcome, RuntimeError), (
        "retry committed while owner failure transition was unresolved: "
        f"checkpoint_count={checkpoint_count}, preparation_state={paused.state.value}"
    )
    assert "not committing" in str(contender_outcome).lower()
    assert _database_snapshot(db) == before

    committing = original_cas(
        preparation.session_id,
        ConsolePreparationTransition(
            preparation_id=preparation.preparation_id,
            expected_state=ConsoleTurnPreparationState.PAUSED,
            new_state=ConsoleTurnPreparationState.COMMITTING,
            pause_kind=None,
            new_attempt_id=None,
        ),
    )
    assert committing is not None
    committed = store.commit_durable_turn(acceptance)
    assert store.commit_durable_turn(acceptance) == committed
    accepted = original_cas(
        preparation.session_id,
        ConsolePreparationTransition(
            preparation_id=preparation.preparation_id,
            expected_state=ConsoleTurnPreparationState.COMMITTING,
            new_state=ConsoleTurnPreparationState.ACCEPTED,
            pause_kind=None,
            new_attempt_id=None,
        ),
    )
    assert accepted is not None
    assert accepted.state is ConsoleTurnPreparationState.ACCEPTED
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


def test_failed_owner_does_not_overwrite_legitimate_session_close(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Owner cleanup must not recreate or pause state removed by close."""

    _db, service, store, preparation, acceptance = _ready_store(tmp_path)
    entered = Event()
    release = Event()

    def blocked_failure(**_kwargs: Any):
        entered.set()
        assert release.wait(timeout=5)
        raise RuntimeError("persistence failed after close")

    monkeypatch.setattr(service, "commit_durable_turn", blocked_failure)
    with ThreadPoolExecutor(max_workers=1) as executor:
        owner = executor.submit(store.commit_durable_turn, acceptance)
        assert entered.wait(timeout=5)
        store.close_session(preparation.session_id)
        release.set()
        outcome = _result_or_exception(owner)

    assert isinstance(outcome, RuntimeError)
    assert store.preparation_by_id(preparation.preparation_id) is None
    assert store.preparation_for_session(preparation.session_id) is None
