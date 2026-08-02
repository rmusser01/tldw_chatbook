"""Portable Database Note session and serialized-save contracts."""

from __future__ import annotations

import asyncio
import builtins
from collections import deque
from datetime import datetime, timezone
from typing import Callable

import pytest

from tldw_chatbook.Library.library_notes_session import (
    DatabaseNotePortLoadReply,
    DatabaseNotePortSaveReply,
    DatabaseNoteSessionCoordinator,
    NoteLoadOutcomeKind,
    NoteSaveOutcomeKind,
    PortLoadKind,
)
from tldw_chatbook.Library.library_notes_state import NormalizedDatabaseNote


NOW = datetime(2026, 7, 31, 12, 34, tzinfo=timezone.utc)


def _detail(
    note_id: str = "n-1",
    *,
    title: str = "Original",
    body: str = "Original body",
    keywords: tuple[str, ...] = ("one",),
    version: int = 1,
) -> NormalizedDatabaseNote:
    return NormalizedDatabaseNote(
        note_id=note_id,
        title=title,
        body=body,
        keywords=keywords,
        version=version,
        created_at="2026-07-01T00:00:00+00:00",
        modified_at="2026-07-02T00:00:00+00:00",
    )


class FakeDatabaseNotePort:
    """Controllable async session port with bounded save-concurrency evidence."""

    def __init__(self, detail: NormalizedDatabaseNote | None = None) -> None:
        self.load_calls: list[str] = []
        self.load_replies: deque[DatabaseNotePortLoadReply] = deque(
            [DatabaseNotePortLoadReply.loaded(detail or _detail())]
        )
        self.load_gates: deque[asyncio.Event | None] = deque()
        self.save_calls = []
        self.save_replies: deque[DatabaseNotePortSaveReply] = deque()
        self.save_gates: deque[asyncio.Event | None] = deque()
        self.active_saves = 0
        self.max_active_saves = 0

    async def load_note(self, note_id: str) -> DatabaseNotePortLoadReply:
        self.load_calls.append(note_id)
        reply = self.load_replies.popleft()
        gate = self.load_gates.popleft() if self.load_gates else None
        if gate is not None:
            await gate.wait()
        return reply

    async def save_note(self, note_id, expected_version, payload):
        self.save_calls.append((note_id, expected_version, payload))
        reply = self.save_replies.popleft()
        gate = self.save_gates.popleft() if self.save_gates else None
        self.active_saves += 1
        self.max_active_saves = max(self.max_active_saves, self.active_saves)
        try:
            if gate is not None:
                await gate.wait()
            return reply
        finally:
            self.active_saves -= 1


def _coordinator(
    port: FakeDatabaseNotePort,
    *,
    clock: Callable[[], datetime] = lambda: NOW,
) -> DatabaseNoteSessionCoordinator:
    return DatabaseNoteSessionCoordinator(port, clock=clock)


async def _wait_for_call_count(calls: list, expected: int) -> None:
    async def wait() -> None:
        while len(calls) < expected:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), timeout=1)


@pytest.mark.asyncio
async def test_open_session_seeds_one_coherent_baseline_and_exact_draft():
    detail = _detail(
        title="[draft] <plan>",
        body="line 1\nline 2",
        keywords=("Alpha", "βeta"),
        version=7,
    )
    port = FakeDatabaseNotePort(detail)
    coordinator = _coordinator(port)

    outcome = await coordinator.open_session("n-1")

    assert outcome.kind is NoteLoadOutcomeKind.LOADED
    assert port.load_calls == ["n-1"]
    snapshot = coordinator.snapshot
    assert snapshot is not None
    assert snapshot.baseline == detail
    assert snapshot.note_id == "n-1"
    assert snapshot.title == "[draft] <plan>"
    assert snapshot.body == "line 1\nline 2"
    assert snapshot.keywords_text == "Alpha, βeta"
    assert snapshot.version == 7
    assert snapshot.draft_revision == 0
    assert snapshot.saved_revision == 0
    assert snapshot.dirty is False


@pytest.mark.asyncio
async def test_stale_open_session_cannot_replace_a_newer_loaded_session():
    first_gate = asyncio.Event()
    port = FakeDatabaseNotePort()
    port.load_replies = deque(
        [
            DatabaseNotePortLoadReply.loaded(_detail("n-1", title="First")),
            DatabaseNotePortLoadReply.loaded(_detail("n-2", title="Second")),
        ]
    )
    port.load_gates = deque([first_gate, None])
    coordinator = _coordinator(port)

    first = asyncio.create_task(coordinator.open_session("n-1"))
    await _wait_for_call_count(port.load_calls, 1)
    second = await coordinator.open_session("n-2")
    first_gate.set()
    stale = await first

    assert second.kind is NoteLoadOutcomeKind.LOADED
    assert stale.kind is NoteLoadOutcomeKind.STALE
    assert coordinator.snapshot is not None
    assert coordinator.snapshot.note_id == "n-2"
    assert coordinator.snapshot.title == "Second"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reply", "expected_kind"),
    (
        (DatabaseNotePortLoadReply.missing("Gone"), NoteLoadOutcomeKind.MISSING),
        (DatabaseNotePortLoadReply.failed("Offline"), NoteLoadOutcomeKind.FAILED),
    ),
)
async def test_missing_and_failed_loads_do_not_seed_a_false_session(
    reply, expected_kind
):
    port = FakeDatabaseNotePort()
    port.load_replies = deque([reply])
    coordinator = _coordinator(port)

    outcome = await coordinator.open_session("missing")

    assert outcome.kind is expected_kind
    assert outcome.message
    assert coordinator.snapshot is None


def test_load_reply_requires_detail_only_for_loaded_kind():
    with pytest.raises(ValueError, match="requires detail"):
        DatabaseNotePortLoadReply(kind=PortLoadKind.LOADED)
    with pytest.raises(ValueError, match="cannot carry detail"):
        DatabaseNotePortLoadReply(
            kind=PortLoadKind.MISSING,
            detail=_detail(),
        )


@pytest.mark.asyncio
async def test_loaded_detail_with_wrong_identity_is_a_failure_not_a_session():
    port = FakeDatabaseNotePort(_detail("n-2"))
    coordinator = _coordinator(port)

    outcome = await coordinator.open_session("n-1")

    assert outcome.kind is NoteLoadOutcomeKind.FAILED
    assert "identity" in outcome.message.lower()
    assert coordinator.snapshot is None


@pytest.mark.asyncio
async def test_invalidating_a_pending_open_makes_its_completion_stale():
    gate = asyncio.Event()
    port = FakeDatabaseNotePort()
    port.load_gates = deque([gate])
    coordinator = _coordinator(port)

    pending = asyncio.create_task(coordinator.open_session("n-1"))
    await _wait_for_call_count(port.load_calls, 1)
    coordinator.invalidate_session_request()
    gate.set()

    assert (await pending).kind is NoteLoadOutcomeKind.STALE
    assert coordinator.snapshot is None


@pytest.mark.asyncio
async def test_genuine_mutation_increments_revision_once_and_marks_dirty():
    port = FakeDatabaseNotePort()
    coordinator = _coordinator(port)
    await coordinator.open_session("n-1")

    assert coordinator.mutate(body="Changed") is True
    snapshot = coordinator.snapshot
    assert snapshot is not None
    assert snapshot.body == "Changed"
    assert snapshot.draft_revision == 1
    assert snapshot.saved_revision == 0
    assert snapshot.dirty is True
    assert coordinator.mutate(body="Changed") is False
    assert coordinator.snapshot is snapshot


@pytest.mark.asyncio
async def test_snapshot_reads_are_programmatic_and_do_not_mutate_revision():
    port = FakeDatabaseNotePort()
    coordinator = _coordinator(port)
    await coordinator.open_session("n-1")

    first = coordinator.snapshot
    second = coordinator.snapshot

    assert first is second
    assert first is not None and first.draft_revision == 0


@pytest.mark.asyncio
async def test_construction_load_and_save_do_not_import_textual(monkeypatch):
    real_import = builtins.__import__

    def reject_textual(name, *args, **kwargs):
        if name == "textual" or name.startswith("textual."):
            raise AssertionError(f"Portable coordinator imported {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_textual)
    port = FakeDatabaseNotePort()
    port.save_replies.append(DatabaseNotePortSaveReply.saved(version=2))
    coordinator = _coordinator(port)

    await coordinator.open_session("n-1")
    coordinator.mutate(body="Changed")
    outcome = await coordinator.request_save(explicit=True)

    assert outcome.kind is NoteSaveOutcomeKind.SAVED


@pytest.mark.asyncio
async def test_invalid_payload_vetoes_without_calling_the_port():
    port = FakeDatabaseNotePort()
    coordinator = _coordinator(port)
    await coordinator.open_session("n-1")
    coordinator.mutate(title="x" * 301)

    outcome = await coordinator.request_save(explicit=True)

    assert outcome.kind is NoteSaveOutcomeKind.VALIDATION_VETO
    assert outcome.veto is not None and outcome.veto.field == "title"
    assert port.save_calls == []
    assert coordinator.snapshot is not None
    assert coordinator.snapshot.dirty is True
    assert coordinator.snapshot.saving is False
    assert coordinator.snapshot.title == "x" * 301


@pytest.mark.asyncio
async def test_explicit_noop_save_is_acknowledged_without_a_port_call():
    port = FakeDatabaseNotePort()
    coordinator = _coordinator(port)
    await coordinator.open_session("n-1", untouched_create_token="create-7")

    outcome = await coordinator.request_save(explicit=True)

    assert outcome.kind is NoteSaveOutcomeKind.ACKNOWLEDGED
    assert port.save_calls == []
    assert coordinator.untouched_create_token is None


@pytest.mark.asyncio
async def test_three_edits_while_first_save_runs_coalesce_to_latest_revision():
    first_gate = asyncio.Event()
    second_gate = asyncio.Event()
    port = FakeDatabaseNotePort()
    port.save_gates = deque([first_gate, second_gate])
    port.save_replies = deque(
        [
            DatabaseNotePortSaveReply.saved(version=2),
            DatabaseNotePortSaveReply.saved(version=3),
        ]
    )
    coordinator = _coordinator(port)
    await coordinator.open_session("n-1")
    coordinator.mutate(body="revision 1")

    save = asyncio.create_task(coordinator.request_save(explicit=False))
    await _wait_for_call_count(port.save_calls, 1)
    coordinator.mutate(body="revision 2")
    coordinator.mutate(body="revision 3")
    coordinator.mutate(body="revision 4")
    assert coordinator.snapshot is not None
    assert coordinator.snapshot.status_message == "Saving…"
    first_gate.set()
    await _wait_for_call_count(port.save_calls, 2)

    assert [call[2].revision for call in port.save_calls] == [1, 4]
    assert port.save_calls[1][1] == 2
    assert coordinator.snapshot is not None
    assert coordinator.snapshot.baseline.body == "revision 1"
    assert coordinator.snapshot.body == "revision 4"
    assert coordinator.snapshot.saved_revision == 1
    assert coordinator.snapshot.dirty is True
    assert coordinator.snapshot.saving is True
    second_gate.set()
    outcome = await save

    assert outcome.kind is NoteSaveOutcomeKind.SAVED
    assert coordinator.snapshot is not None
    assert coordinator.snapshot.saved_revision == 4
    assert coordinator.snapshot.draft_revision == 4
    assert coordinator.snapshot.dirty is False
    assert coordinator.snapshot.version == 3
    assert port.max_active_saves == 1


@pytest.mark.asyncio
async def test_edit_during_followup_attempt_continues_without_a_two_save_limit():
    gates = [asyncio.Event() for _ in range(3)]
    port = FakeDatabaseNotePort()
    port.save_gates = deque(gates)
    port.save_replies = deque(
        DatabaseNotePortSaveReply.saved(version=version) for version in (2, 3, 4)
    )
    coordinator = _coordinator(port)
    await coordinator.open_session("n-1")
    coordinator.mutate(body="revision 1")

    save = asyncio.create_task(coordinator.request_save(explicit=False))
    await _wait_for_call_count(port.save_calls, 1)
    coordinator.mutate(body="revision 2")
    gates[0].set()
    await _wait_for_call_count(port.save_calls, 2)
    coordinator.mutate(body="revision 3")
    gates[1].set()
    await _wait_for_call_count(port.save_calls, 3)
    gates[2].set()
    await save

    assert [call[2].revision for call in port.save_calls] == [1, 2, 3]
    assert port.max_active_saves == 1
    assert coordinator.snapshot is not None
    assert coordinator.snapshot.saved_revision == 3
    assert coordinator.snapshot.dirty is False


@pytest.mark.asyncio
async def test_same_revision_save_request_joins_the_active_attempt():
    gate = asyncio.Event()
    port = FakeDatabaseNotePort()
    port.save_gates = deque([gate])
    port.save_replies = deque([DatabaseNotePortSaveReply.saved(version=2)])
    coordinator = _coordinator(port)
    await coordinator.open_session("n-1")
    coordinator.mutate(body="revision 1")

    first = asyncio.create_task(coordinator.request_save(explicit=False))
    await _wait_for_call_count(port.save_calls, 1)
    second = asyncio.create_task(coordinator.request_save(explicit=True))
    await asyncio.sleep(0)
    gate.set()
    outcomes = await asyncio.gather(first, second)

    assert [outcome.kind for outcome in outcomes] == [
        NoteSaveOutcomeKind.SAVED,
        NoteSaveOutcomeKind.SAVED,
    ]
    assert len(port.save_calls) == 1


@pytest.mark.asyncio
async def test_failure_stops_chaining_and_explicit_retry_saves_latest_revision():
    failure_gate = asyncio.Event()
    port = FakeDatabaseNotePort()
    port.save_gates = deque([failure_gate, None])
    port.save_replies = deque(
        [
            DatabaseNotePortSaveReply.failed("Offline"),
            DatabaseNotePortSaveReply.saved(version=2),
        ]
    )
    coordinator = _coordinator(port)
    await coordinator.open_session("n-1")
    coordinator.mutate(body="revision 1")

    first = asyncio.create_task(coordinator.request_save(explicit=False))
    await _wait_for_call_count(port.save_calls, 1)
    coordinator.mutate(body="revision 2")
    failure_gate.set()
    failed = await first

    assert failed.kind is NoteSaveOutcomeKind.FAILED
    assert len(port.save_calls) == 1
    assert coordinator.snapshot is not None
    assert coordinator.snapshot.body == "revision 2"
    assert coordinator.snapshot.saved_revision == 0
    assert coordinator.snapshot.dirty is True
    assert coordinator.snapshot.saving is False
    assert "Press Save to retry" in coordinator.snapshot.status_message

    retried = await coordinator.request_save(explicit=True)

    assert retried.kind is NoteSaveOutcomeKind.SAVED
    assert [call[2].revision for call in port.save_calls] == [1, 2]
    assert coordinator.snapshot.saved_revision == 2
    assert coordinator.snapshot.dirty is False


@pytest.mark.asyncio
async def test_conflict_stops_chaining_and_preserves_the_newest_draft():
    gate = asyncio.Event()
    port = FakeDatabaseNotePort()
    port.save_gates = deque([gate])
    port.save_replies = deque([DatabaseNotePortSaveReply.conflict()])
    coordinator = _coordinator(port)
    await coordinator.open_session("n-1")
    coordinator.mutate(body="revision 1")

    save = asyncio.create_task(coordinator.request_save(explicit=False))
    await _wait_for_call_count(port.save_calls, 1)
    coordinator.mutate(body="revision 2")
    gate.set()
    outcome = await save

    assert outcome.kind is NoteSaveOutcomeKind.CONFLICTED
    assert len(port.save_calls) == 1
    assert coordinator.snapshot is not None
    assert coordinator.snapshot.body == "revision 2"
    assert coordinator.snapshot.saved_revision == 0
    assert coordinator.snapshot.dirty is True
    assert coordinator.snapshot.saving is False
    assert coordinator.snapshot.in_conflict is True

    coordinator.mutate(body="revision 3")

    assert coordinator.snapshot.in_conflict is True
    assert coordinator.snapshot.status_message == "Conflict — review the choices below."
