from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicyCandidate,
    ConsoleLibraryPolicyHolder,
    ConsoleLibraryPolicySnapshot,
    ConsoleLibraryPolicyWriteStatus,
)
from tldw_chatbook.Chat.console_library_policy_coordinator import (
    ConsoleLibraryPolicyCoordinator,
)
from tldw_chatbook.Chat.console_library_policy_repository import (
    ConsoleLibraryPolicyRepository,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def _candidate(*, allowed: bool) -> ConsoleLibraryPolicyCandidate:
    return ConsoleLibraryPolicyCandidate(
        auto_retrieve=(
            ConsoleAutoRetrieve.AUTOMATIC if allowed else ConsoleAutoRetrieve.NEVER
        ),
        assistant_access=(
            ConsoleAssistantLibraryAccess.ALLOWED
            if allowed
            else ConsoleAssistantLibraryAccess.BLOCKED
        ),
    )


def _holder(*, allowed: bool) -> ConsoleLibraryPolicyHolder:
    return ConsoleLibraryPolicyHolder(
        ConsoleLibraryPolicySnapshot(
            auto_retrieve=(
                ConsoleAutoRetrieve.AUTOMATIC if allowed else ConsoleAutoRetrieve.NEVER
            ),
            assistant_access=(
                ConsoleAssistantLibraryAccess.ALLOWED
                if allowed
                else ConsoleAssistantLibraryAccess.BLOCKED
            ),
            policy_revision=None,
            source="new_session",
        )
    )


@pytest.mark.asyncio
async def test_load_and_save_run_repository_work_off_event_loop(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "off-loop.sqlite", client_id="coordinator")
    conversation_id = db.add_conversation({"title": "off loop"})
    assert conversation_id is not None
    repository = ConsoleLibraryPolicyRepository(db)
    coordinator = ConsoleLibraryPolicyCoordinator(repository)
    holder = _holder(allowed=False)
    coordinator.register_holder("session", conversation_id, holder)
    event_loop_thread = threading.get_ident()
    observed: list[int] = []
    original_read = repository.read
    original_insert = repository.insert

    def recording_read(target: str):
        observed.append(threading.get_ident())
        return original_read(target)

    def recording_insert(target: str, candidate: ConsoleLibraryPolicyCandidate):
        observed.append(threading.get_ident())
        return original_insert(target, candidate)

    repository.read = recording_read  # type: ignore[method-assign]
    repository.insert = recording_insert  # type: ignore[method-assign]

    await coordinator.load("session", conversation_id)
    saved = await coordinator.save("session", _candidate(allowed=True))

    assert saved.status is ConsoleLibraryPolicyWriteStatus.COMMITTED
    assert len(observed) == 2
    assert all(thread_id != event_loop_thread for thread_id in observed)


@pytest.mark.asyncio
async def test_committed_save_publishes_to_all_same_process_holders_only_after_commit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "publication.sqlite"
    db = CharactersRAGDB(path, client_id="coordinator")
    conversation_id = db.add_conversation({"title": "publication"})
    assert conversation_id is not None
    repository = ConsoleLibraryPolicyRepository(db)
    coordinator = ConsoleLibraryPolicyCoordinator(repository)
    first = _holder(allowed=False)
    second = _holder(allowed=False)
    unrelated = _holder(allowed=False)
    coordinator.register_holder("first", conversation_id, first)
    coordinator.register_holder("second", conversation_id, second)
    coordinator.register_holder("unrelated", None, unrelated)

    entered = threading.Event()
    release = threading.Event()
    original_insert = repository.insert

    def blocked_insert(target: str, candidate: ConsoleLibraryPolicyCandidate):
        entered.set()
        assert release.wait(timeout=5)
        return original_insert(target, candidate)

    repository.insert = blocked_insert  # type: ignore[method-assign]
    save_task = asyncio.create_task(
        coordinator.save("first", _candidate(allowed=True))
    )
    assert await asyncio.to_thread(entered.wait, 5)
    assert first.snapshot.source == "new_session"
    assert second.snapshot.source == "new_session"
    release.set()
    result = await save_task

    assert result.status is ConsoleLibraryPolicyWriteStatus.COMMITTED
    assert first.snapshot == result.snapshot
    assert second.snapshot == result.snapshot
    assert unrelated.snapshot.source == "new_session"
    assert first.snapshot.policy_revision == 1


@pytest.mark.asyncio
async def test_fresh_execution_read_defeats_stale_allowed_holder(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fresh.sqlite"
    first_db = CharactersRAGDB(path, client_id="first-process")
    conversation_id = first_db.add_conversation({"title": "fresh"})
    assert conversation_id is not None
    first_repository = ConsoleLibraryPolicyRepository(first_db)
    assert first_repository.insert(conversation_id, _candidate(allowed=True)).status is (
        ConsoleLibraryPolicyWriteStatus.COMMITTED
    )
    second_db = CharactersRAGDB(path, client_id="second-process")
    second_repository = ConsoleLibraryPolicyRepository(second_db)
    coordinator = ConsoleLibraryPolicyCoordinator(first_repository)
    holder = _holder(allowed=True)
    holder.snapshot = first_repository.read(conversation_id).snapshot
    coordinator.register_holder("session", conversation_id, holder)

    assert second_repository.compare_and_swap(
        conversation_id, 1, _candidate(allowed=False)
    ).status is ConsoleLibraryPolicyWriteStatus.COMMITTED
    captured = await coordinator.capture_for_execution("session")

    assert captured.policy_revision == 2
    assert captured.auto_retrieve is ConsoleAutoRetrieve.NEVER
    assert captured.assistant_access is ConsoleAssistantLibraryAccess.BLOCKED
    assert holder.snapshot == captured


@pytest.mark.asyncio
async def test_unavailable_execution_read_returns_and_publishes_fail_closed_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = CharactersRAGDB(tmp_path / "unavailable.sqlite", client_id="coordinator")
    conversation_id = db.add_conversation({"title": "unavailable"})
    assert conversation_id is not None
    repository = ConsoleLibraryPolicyRepository(db)
    coordinator = ConsoleLibraryPolicyCoordinator(repository)
    holder = _holder(allowed=True)
    coordinator.register_holder("session", conversation_id, holder)

    def unavailable() -> object:
        raise RuntimeError("private database detail")

    monkeypatch.setattr(db, "get_connection", unavailable)
    captured = await coordinator.capture_for_execution("session")

    assert (
        captured.auto_retrieve,
        captured.assistant_access,
        captured.source,
        captured.error_code,
    ) == (
        ConsoleAutoRetrieve.NEVER,
        ConsoleAssistantLibraryAccess.BLOCKED,
        "unavailable",
        "policy_read_error",
    )
    assert holder.snapshot == captured


@pytest.mark.asyncio
async def test_commit_after_capture_changes_only_the_next_capture(tmp_path: Path) -> None:
    path = tmp_path / "linearization.sqlite"
    first_db = CharactersRAGDB(path, client_id="capture")
    conversation_id = first_db.add_conversation({"title": "linearization"})
    assert conversation_id is not None
    first_repository = ConsoleLibraryPolicyRepository(first_db)
    first_repository.insert(conversation_id, _candidate(allowed=True))
    coordinator = ConsoleLibraryPolicyCoordinator(first_repository)
    coordinator.register_holder("session", conversation_id, _holder(allowed=True))

    captured = await coordinator.capture_for_execution("session")
    second_repository = ConsoleLibraryPolicyRepository(
        CharactersRAGDB(path, client_id="later-writer")
    )
    second_repository.compare_and_swap(conversation_id, 1, _candidate(allowed=False))
    next_capture = await coordinator.capture_for_execution("session")

    assert captured.policy_revision == 1
    assert captured.assistant_access is ConsoleAssistantLibraryAccess.ALLOWED
    assert next_capture.policy_revision == 2
    assert next_capture.assistant_access is ConsoleAssistantLibraryAccess.BLOCKED


@pytest.mark.asyncio
async def test_unregister_removes_holder_from_later_publication(tmp_path: Path) -> None:
    db = CharactersRAGDB(tmp_path / "unregister.sqlite", client_id="coordinator")
    conversation_id = db.add_conversation({"title": "unregister"})
    assert conversation_id is not None
    coordinator = ConsoleLibraryPolicyCoordinator(ConsoleLibraryPolicyRepository(db))
    first = _holder(allowed=False)
    closed = _holder(allowed=False)
    coordinator.register_holder("first", conversation_id, first)
    coordinator.register_holder("closed", conversation_id, closed)
    coordinator.unregister_holder("closed")

    result = await coordinator.save("first", _candidate(allowed=True))

    assert result.status is ConsoleLibraryPolicyWriteStatus.COMMITTED
    assert first.snapshot == result.snapshot
    assert closed.snapshot.source == "new_session"
