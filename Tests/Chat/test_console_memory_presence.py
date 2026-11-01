"""When can the Console skip proving which memory applies? (TASK-33628.5.2)

``context_control_inputs`` skips the per-message lineage capture only when no
memory can apply: no ``/rewind`` summary and no active selection event. These
tests pin the repository probe against real SQLite and the decision against
every input that can make memory possible.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_context_repository import (
    ConsoleContextRepository,
    ConsoleMemorySelectionRecord,
    MemorySelectionKind,
    may_hold_branch_memory,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def _database() -> tuple[CharactersRAGDB, ConsoleContextRepository, str, str]:
    db = CharactersRAGDB(":memory:", client_id="memory-presence")
    conversation_id = db.add_conversation({"title": "memory presence"})
    message_id = db.add_message(
        {
            "id": "m1",
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "hi",
        }
    )
    assert message_id is not None
    return db, ConsoleContextRepository(db), conversation_id, message_id


def _reset(conversation_id: str, message_id: str, *, active: bool = True):
    return ConsoleMemorySelectionRecord(
        sequence=1,
        selection_id=f"reset-{active}",
        conversation_id=conversation_id,
        activation_message_id=message_id,
        selected_memory_id=None,
        event_kind=MemorySelectionKind.RESET,
        suppresses_legacy=True,
        created_at="2026-10-05T00:00:00Z",
        active=active,
    )


def test_the_probe_sees_active_selection_rows_only() -> None:
    _db, repository, conversation_id, message_id = _database()
    assert repository.has_active_memory_selection(conversation_id) is False

    repository.insert_memory_selection(_reset(conversation_id, message_id, active=False))
    assert repository.has_active_memory_selection(conversation_id) is False
    assert repository.list_active_memory_selections(conversation_id) == ()

    repository.insert_memory_selection(_reset(conversation_id, message_id))
    assert repository.has_active_memory_selection(conversation_id) is True
    # The probe agrees with the reader the branch-head lookup starts from.
    assert len(repository.list_active_memory_selections(conversation_id)) == 1


def _controller(repository, *, summary=(None, None)):
    store = SimpleNamespace(session_context_summary=lambda _session_id: summary)
    return SimpleNamespace(store=store, _context_repository=repository)


def test_no_summary_and_no_active_selection_means_no_memory() -> None:
    _db, repository, conversation_id, message_id = _database()
    assert not may_hold_branch_memory(_controller(repository), "s", conversation_id)

    repository.insert_memory_selection(_reset(conversation_id, message_id))
    assert may_hold_branch_memory(_controller(repository), "s", conversation_id)


@pytest.mark.parametrize(
    ("summary", "expected"),
    [
        (("Earlier turns, summarized.", "native-boundary"), True),
        (("   ", "native-boundary"), False),
        (("Earlier turns, summarized.", None), False),
        ((None, None), False),
    ],
)
def test_only_a_usable_rewind_summary_makes_legacy_memory_possible(
    summary, expected
) -> None:
    """Mirrors ``_validated_legacy_memory``'s own early rejections."""
    _db, repository, conversation_id, _message_id = _database()
    controller = _controller(repository, summary=summary)
    assert may_hold_branch_memory(controller, "s", conversation_id) is expected


def test_no_conversation_or_repository_means_no_memory() -> None:
    _db, repository, _conversation_id, _message_id = _database()
    assert not may_hold_branch_memory(_controller(repository), "s", None)
    assert not may_hold_branch_memory(_controller(None), "s", "conversation")


def test_a_repository_without_the_real_api_keeps_the_full_path() -> None:
    """Doubles may derive a branch head from memories alone, so never skip."""
    probe_only = SimpleNamespace(has_active_memory_selection=lambda _id: False)
    reader_only = SimpleNamespace(load_applicable_branch_memory=lambda *_a: None)
    for repository in (probe_only, reader_only, SimpleNamespace()):
        assert may_hold_branch_memory(_controller(repository), "s", "conversation")
