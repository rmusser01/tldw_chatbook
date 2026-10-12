"""Portable Database Note session and serialized-save contracts."""

from __future__ import annotations

import asyncio
import builtins
import importlib.util
import time
from collections import deque
from dataclasses import replace
from datetime import datetime, timezone
from typing import Callable

import pytest

from tldw_chatbook.Library.library_notes_session import (
    ConflictAction,
    ConflictOutcomeKind,
    DatabaseNotePortLoadReply,
    DatabaseNotePortSaveReply,
    DatabaseNoteSessionCoordinator,
    DestructiveAdmissionOutcomeKind,
    DestructiveKind,
    NoteFlushOutcomeKind,
    NoteLoadOutcomeKind,
    NoteSaveOutcomeKind,
    PortLoadKind,
)
from tldw_chatbook.Library.library_notes_state import (
    DatabaseNoteSavePayload,
    NormalizedDatabaseNote,
)

NOW = datetime(2026, 7, 31, 12, 34, tzinfo=timezone.utc)


def test_notes_reader_has_no_parallel_state_authority() -> None:
    """The session coordinator remains the only Database Notes reader model."""
    assert (
        importlib.util.find_spec("tldw_chatbook.Library.library_notes_reader_state")
        is None
    )


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
        self.save_calls: list[tuple[str, int, DatabaseNoteSavePayload]] = []
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

    async def save_note(
        self,
        note_id: str,
        expected_version: int,
        payload: DatabaseNoteSavePayload,
    ) -> DatabaseNotePortSaveReply:
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
async def test_close_session_clears_loaded_state_and_create_token():
    port = FakeDatabaseNotePort()
    coordinator = _coordinator(port)
    await coordinator.open_session("n-1", untouched_create_token="create-1")

    assert coordinator.untouched_create_token == "create-1"

    coordinator.close_session()

    assert coordinator.snapshot is None
    assert coordinator.untouched_create_token is None


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


async def _seed_conflict(
    coordinator: DatabaseNoteSessionCoordinator,
    port: FakeDatabaseNotePort,
    *,
    body: str = "local revision 1",
) -> None:
    port.save_replies.append(DatabaseNotePortSaveReply.conflict())
    coordinator.mutate(body=body)
    outcome = await coordinator.request_save(explicit=True)
    assert outcome.kind is NoteSaveOutcomeKind.CONFLICTED


@pytest.mark.asyncio
async def test_reload_applies_only_when_captured_revision_is_still_current():
    gate = asyncio.Event()
    port = FakeDatabaseNotePort()
    coordinator = _coordinator(port)
    await coordinator.open_session("n-1")
    await _seed_conflict(coordinator, port)
    port.load_replies.append(
        DatabaseNotePortLoadReply.loaded(_detail(body="remote revision", version=8))
    )
    port.load_gates.append(gate)

    reload_task = asyncio.create_task(
        coordinator.resolve_conflict(ConflictAction.RELOAD)
    )
    await _wait_for_call_count(port.load_calls, 2)
    coordinator.mutate(body="local revision 2")
    gate.set()
    outcome = await reload_task

    assert outcome.kind is ConflictOutcomeKind.DRAFT_CHANGED
    assert coordinator.snapshot is not None
    assert coordinator.snapshot.body == "local revision 2"
    assert coordinator.snapshot.in_conflict is True
    assert coordinator.conflict_resolution_running is False


@pytest.mark.asyncio
async def test_reload_replaces_an_unchanged_conflicted_draft_atomically():
    port = FakeDatabaseNotePort()
    coordinator = _coordinator(port)
    await coordinator.open_session("n-1")
    await _seed_conflict(coordinator, port)
    port.load_replies.append(
        DatabaseNotePortLoadReply.loaded(
            _detail(body="remote revision", keywords=("remote",), version=8)
        )
    )

    outcome = await coordinator.resolve_conflict(ConflictAction.RELOAD)

    assert outcome.kind is ConflictOutcomeKind.RELOADED
    assert coordinator.snapshot is not None
    assert coordinator.snapshot.body == "remote revision"
    assert coordinator.snapshot.keywords_text == "remote"
    assert coordinator.snapshot.version == 8
    assert coordinator.snapshot.dirty is False
    assert coordinator.snapshot.in_conflict is False
    assert coordinator.snapshot.saved_revision == coordinator.snapshot.draft_revision


@pytest.mark.asyncio
async def test_overwrite_rebases_and_keeps_edits_during_fetch_and_save():
    load_gate = asyncio.Event()
    first_save_gate = asyncio.Event()
    second_save_gate = asyncio.Event()
    port = FakeDatabaseNotePort()
    coordinator = _coordinator(port)
    await coordinator.open_session("n-1")
    await _seed_conflict(coordinator, port)
    port.load_replies.append(
        DatabaseNotePortLoadReply.loaded(_detail(body="remote", version=8))
    )
    port.load_gates.append(load_gate)
    port.save_replies.extend(
        (
            DatabaseNotePortSaveReply.saved(version=9),
            DatabaseNotePortSaveReply.saved(version=10),
        )
    )
    port.save_gates.extend((first_save_gate, second_save_gate))

    overwrite = asyncio.create_task(
        coordinator.resolve_conflict(ConflictAction.OVERWRITE)
    )
    await _wait_for_call_count(port.load_calls, 2)
    coordinator.mutate(body="edit during fetch")
    load_gate.set()
    await _wait_for_call_count(port.save_calls, 2)
    assert port.save_calls[1][1] == 8
    assert port.save_calls[1][2].body == "edit during fetch"
    coordinator.mutate(body="edit during save")
    first_save_gate.set()
    await _wait_for_call_count(port.save_calls, 3)
    assert port.save_calls[2][1] == 9
    assert port.save_calls[2][2].body == "edit during save"
    second_save_gate.set()
    outcome = await overwrite

    assert outcome.kind is ConflictOutcomeKind.OVERWRITTEN
    assert coordinator.snapshot is not None
    assert coordinator.snapshot.body == "edit during save"
    assert coordinator.snapshot.version == 10
    assert coordinator.snapshot.dirty is False
    assert port.max_active_saves == 1


@pytest.mark.asyncio
async def test_duplicate_or_opposite_conflict_action_is_ignored():
    gate = asyncio.Event()
    port = FakeDatabaseNotePort()
    coordinator = _coordinator(port)
    await coordinator.open_session("n-1")
    await _seed_conflict(coordinator, port)
    port.load_replies.append(
        DatabaseNotePortLoadReply.loaded(_detail(body="remote", version=8))
    )
    port.load_gates.append(gate)

    first = asyncio.create_task(coordinator.resolve_conflict(ConflictAction.RELOAD))
    await _wait_for_call_count(port.load_calls, 2)
    duplicate = await coordinator.resolve_conflict(ConflictAction.OVERWRITE)

    assert duplicate.kind is ConflictOutcomeKind.ALREADY_RUNNING
    assert port.load_calls == ["n-1", "n-1"]
    gate.set()
    assert (await first).kind is ConflictOutcomeKind.RELOADED


@pytest.mark.asyncio
async def test_cancelled_conflict_refresh_releases_the_operation_gate():
    gate = asyncio.Event()
    port = FakeDatabaseNotePort()
    coordinator = _coordinator(port)
    await coordinator.open_session("n-1")
    await _seed_conflict(coordinator, port)
    port.load_replies.append(
        DatabaseNotePortLoadReply.loaded(_detail(body="remote", version=8))
    )
    port.load_gates.append(gate)

    operation = asyncio.create_task(coordinator.resolve_conflict(ConflictAction.RELOAD))
    await _wait_for_call_count(port.load_calls, 2)
    operation.cancel()

    with pytest.raises(asyncio.CancelledError):
        await operation

    assert coordinator.conflict_resolution_running is False
    assert (await coordinator.flush()).kind is NoteFlushOutcomeKind.CONFLICTED


@pytest.mark.asyncio
async def test_flush_is_vetoed_until_overwrite_publishes_its_terminal_outcome():
    load_gate = asyncio.Event()
    save_gate = asyncio.Event()
    port = FakeDatabaseNotePort()
    coordinator = _coordinator(port)
    await coordinator.open_session("n-1")
    await _seed_conflict(coordinator, port)
    port.load_replies.append(
        DatabaseNotePortLoadReply.loaded(_detail(body="remote", version=8))
    )
    port.load_gates.append(load_gate)
    port.save_replies.append(DatabaseNotePortSaveReply.saved(version=9))
    port.save_gates.append(save_gate)
    overwrite = asyncio.create_task(
        coordinator.resolve_conflict(ConflictAction.OVERWRITE)
    )
    await _wait_for_call_count(port.load_calls, 2)
    load_gate.set()
    await _wait_for_call_count(port.save_calls, 2)

    blocked = await coordinator.flush()

    assert blocked.kind is NoteFlushOutcomeKind.BLOCKED
    assert coordinator.conflict_resolution_running is True
    save_gate.set()
    assert (await overwrite).kind is ConflictOutcomeKind.OVERWRITTEN
    assert (await coordinator.flush()).kind is NoteFlushOutcomeKind.PERMITTED


@pytest.mark.asyncio
async def test_overwrite_that_conflicts_again_returns_renewed_conflict():
    port = FakeDatabaseNotePort()
    coordinator = _coordinator(port)
    await coordinator.open_session("n-1")
    await _seed_conflict(coordinator, port)
    port.load_replies.append(
        DatabaseNotePortLoadReply.loaded(_detail(body="remote", version=8))
    )
    port.save_replies.append(DatabaseNotePortSaveReply.conflict())

    outcome = await coordinator.resolve_conflict(ConflictAction.OVERWRITE)

    assert outcome.kind is ConflictOutcomeKind.RENEWED_CONFLICT
    assert coordinator.snapshot is not None
    assert coordinator.snapshot.in_conflict is True
    assert coordinator.snapshot.conflict_generation == 2
    assert coordinator.snapshot.body == "local revision 1"


@pytest.mark.asyncio
async def test_missing_overwrite_keeps_draft_but_missing_reload_discards_it():
    port = FakeDatabaseNotePort()
    coordinator = _coordinator(port)
    await coordinator.open_session("n-1")
    await _seed_conflict(coordinator, port)
    port.load_replies.append(DatabaseNotePortLoadReply.missing())

    overwrite = await coordinator.resolve_conflict(ConflictAction.OVERWRITE)

    assert overwrite.kind is ConflictOutcomeKind.MISSING
    assert coordinator.snapshot is not None
    assert coordinator.snapshot.body == "local revision 1"
    assert coordinator.snapshot.in_conflict is True

    port.load_replies.append(DatabaseNotePortLoadReply.missing())
    reload_outcome = await coordinator.resolve_conflict(ConflictAction.RELOAD)

    assert reload_outcome.kind is ConflictOutcomeKind.MISSING
    assert coordinator.snapshot is None


@pytest.mark.asyncio
async def test_missing_reload_does_not_discard_a_draft_changed_during_fetch():
    gate = asyncio.Event()
    port = FakeDatabaseNotePort()
    coordinator = _coordinator(port)
    await coordinator.open_session("n-1")
    await _seed_conflict(coordinator, port)
    port.load_replies.append(DatabaseNotePortLoadReply.missing())
    port.load_gates.append(gate)

    reload_task = asyncio.create_task(
        coordinator.resolve_conflict(ConflictAction.RELOAD)
    )
    await _wait_for_call_count(port.load_calls, 2)
    coordinator.mutate(body="newer local draft")
    gate.set()
    outcome = await reload_task

    assert outcome.kind is ConflictOutcomeKind.DRAFT_CHANGED
    assert coordinator.snapshot is not None
    assert coordinator.snapshot.body == "newer local draft"
    assert coordinator.snapshot.in_conflict is True


@pytest.mark.asyncio
async def test_conflict_fetch_failure_retains_conflict_and_draft():
    port = FakeDatabaseNotePort()
    coordinator = _coordinator(port)
    await coordinator.open_session("n-1")
    await _seed_conflict(coordinator, port)
    port.load_replies.append(DatabaseNotePortLoadReply.failed("Offline"))

    outcome = await coordinator.resolve_conflict(ConflictAction.RELOAD)

    assert outcome.kind is ConflictOutcomeKind.FAILED
    assert coordinator.snapshot is not None
    assert coordinator.snapshot.body == "local revision 1"
    assert coordinator.snapshot.in_conflict is True
    assert "again" in coordinator.snapshot.status_message.lower()


@pytest.mark.asyncio
async def test_conflict_completion_cannot_apply_to_a_newer_session():
    gate = asyncio.Event()
    port = FakeDatabaseNotePort()
    coordinator = _coordinator(port)
    await coordinator.open_session("n-1")
    await _seed_conflict(coordinator, port)
    port.load_replies.extend(
        (
            DatabaseNotePortLoadReply.loaded(_detail("n-1", version=8)),
            DatabaseNotePortLoadReply.loaded(_detail("n-2", title="Second")),
        )
    )
    port.load_gates.extend((gate, None))

    reload_task = asyncio.create_task(
        coordinator.resolve_conflict(ConflictAction.RELOAD)
    )
    await _wait_for_call_count(port.load_calls, 2)
    assert (await coordinator.open_session("n-2")).kind is NoteLoadOutcomeKind.LOADED
    gate.set()
    outcome = await reload_task

    assert outcome.kind is ConflictOutcomeKind.STALE
    assert coordinator.snapshot is not None
    assert coordinator.snapshot.note_id == "n-2"
    assert coordinator.snapshot.title == "Second"


@pytest.mark.asyncio
async def test_conflict_fetch_exception_is_stale_after_session_replacement():
    gate = asyncio.Event()
    port = FakeDatabaseNotePort()
    coordinator = _coordinator(port)
    await coordinator.open_session("n-1")
    await _seed_conflict(coordinator, port)

    async def load_with_stale_failure(note_id: str):
        port.load_calls.append(note_id)
        if note_id == "n-1":
            await gate.wait()
            raise RuntimeError("late failure")
        return DatabaseNotePortLoadReply.loaded(_detail("n-2", title="Second"))

    port.load_note = load_with_stale_failure
    reload_task = asyncio.create_task(
        coordinator.resolve_conflict(ConflictAction.RELOAD)
    )
    await _wait_for_call_count(port.load_calls, 2)
    assert (await coordinator.open_session("n-2")).kind is NoteLoadOutcomeKind.LOADED
    gate.set()
    outcome = await reload_task

    assert outcome.kind is ConflictOutcomeKind.STALE
    assert coordinator.snapshot is not None
    assert coordinator.snapshot.note_id == "n-2"


@pytest.mark.asyncio
async def test_flush_waits_for_the_complete_coalesced_save_chain():
    gates = (asyncio.Event(), asyncio.Event())
    port = FakeDatabaseNotePort()
    port.save_gates.extend(gates)
    port.save_replies.extend(
        (
            DatabaseNotePortSaveReply.saved(version=2),
            DatabaseNotePortSaveReply.saved(version=3),
        )
    )
    coordinator = _coordinator(port)
    await coordinator.open_session("n-1")
    coordinator.mutate(body="revision 1")
    save = asyncio.create_task(coordinator.request_save(explicit=False))
    await _wait_for_call_count(port.save_calls, 1)
    coordinator.mutate(body="revision 2")

    flush = asyncio.create_task(coordinator.flush())
    await asyncio.sleep(0)
    assert flush.done() is False
    gates[0].set()
    await _wait_for_call_count(port.save_calls, 2)
    assert flush.done() is False
    gates[1].set()
    await save
    outcome = await flush

    assert outcome.kind is NoteFlushOutcomeKind.PERMITTED
    assert coordinator.snapshot is not None
    assert coordinator.snapshot.saved_revision == 2
    assert coordinator.snapshot.dirty is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reply", "expected_kind"),
    (
        (
            DatabaseNotePortSaveReply.failed("Offline"),
            NoteFlushOutcomeKind.FAILED,
        ),
        (
            DatabaseNotePortSaveReply.conflict(),
            NoteFlushOutcomeKind.CONFLICTED,
        ),
    ),
)
async def test_flush_failure_and_conflict_veto_navigation(reply, expected_kind):
    port = FakeDatabaseNotePort()
    port.save_replies.append(reply)
    coordinator = _coordinator(port)
    await coordinator.open_session("n-1")
    coordinator.mutate(body="dirty")

    outcome = await coordinator.flush()

    assert outcome.kind is expected_kind
    assert coordinator.snapshot is not None
    assert coordinator.snapshot.dirty is True


@pytest.mark.asyncio
async def test_flush_validation_veto_retains_invalid_dirty_draft():
    port = FakeDatabaseNotePort()
    coordinator = _coordinator(port)
    await coordinator.open_session("n-1")
    coordinator.mutate(title="x" * 301)

    outcome = await coordinator.flush()

    assert outcome.kind is NoteFlushOutcomeKind.VALIDATION_VETO
    assert coordinator.snapshot is not None
    assert coordinator.snapshot.title == "x" * 301
    assert coordinator.snapshot.dirty is True
    assert port.save_calls == []


def _session_tokens(coordinator: DatabaseNoteSessionCoordinator):
    snapshot = coordinator.snapshot
    assert snapshot is not None
    return {
        "note_id": snapshot.note_id,
        "session_generation": snapshot.session_generation,
        "expected_version": snapshot.version,
    }


@pytest.mark.asyncio
async def test_discard_admission_blocks_mutation_save_duplicate_and_preserves_token():
    port = FakeDatabaseNotePort()
    coordinator = _coordinator(port)
    await coordinator.open_session("n-1", untouched_create_token="create-7")

    outcome = await coordinator.request_destructive_admission(
        DestructiveKind.DISCARD_NEW_NOTE,
        create_token="create-7",
        **_session_tokens(coordinator),
    )

    assert outcome.kind is DestructiveAdmissionOutcomeKind.ADMITTED
    assert outcome.admission is not None
    assert coordinator.mutate(body="blocked") is False
    blocked_save = await coordinator.request_save(explicit=True)
    assert blocked_save.kind is NoteSaveOutcomeKind.BLOCKED
    assert coordinator.untouched_create_token == "create-7"
    duplicate = await coordinator.request_destructive_admission(
        DestructiveKind.DISCARD_NEW_NOTE,
        create_token="create-7",
        **_session_tokens(coordinator),
    )
    assert duplicate.kind is DestructiveAdmissionOutcomeKind.ALREADY_RUNNING
    assert coordinator.cancel_destructive(outcome.admission) is True
    assert coordinator.mutate(body="now editable") is True
    assert coordinator.untouched_create_token is None


@pytest.mark.asyncio
async def test_stale_or_ineligible_destructive_tokens_never_enter_the_gate():
    port = FakeDatabaseNotePort()
    coordinator = _coordinator(port)
    await coordinator.open_session("n-1", untouched_create_token="create-7")
    tokens = _session_tokens(coordinator)

    stale = await coordinator.request_destructive_admission(
        DestructiveKind.DELETE,
        **{**tokens, "expected_version": tokens["expected_version"] + 1},
    )
    ineligible = await coordinator.request_destructive_admission(
        DestructiveKind.DISCARD_NEW_NOTE,
        create_token="wrong-token",
        **tokens,
    )

    assert stale.kind is DestructiveAdmissionOutcomeKind.STALE
    assert ineligible.kind is DestructiveAdmissionOutcomeKind.NOT_ELIGIBLE
    assert coordinator.destructive_admission is None
    assert coordinator.mutate(body="still editable") is True


@pytest.mark.asyncio
async def test_running_destructive_action_cannot_cancel_and_failure_unlocks_draft():
    port = FakeDatabaseNotePort()
    coordinator = _coordinator(port)
    await coordinator.open_session("n-1")
    admitted = await coordinator.request_destructive_admission(
        DestructiveKind.DELETE,
        **_session_tokens(coordinator),
    )
    assert admitted.admission is not None

    assert coordinator.mark_destructive_running(admitted.admission) is True
    assert coordinator.cancel_destructive(admitted.admission) is False
    assert coordinator.mutate(body="blocked") is False
    assert (
        await coordinator.request_save(explicit=True)
    ).kind is NoteSaveOutcomeKind.BLOCKED
    assert coordinator.finish_destructive(admitted.admission, success=False) is True
    assert coordinator.snapshot is not None
    assert coordinator.mutate(body="editable after failure") is True


@pytest.mark.asyncio
async def test_successful_destructive_action_closes_session_and_stales_old_admission():
    port = FakeDatabaseNotePort()
    coordinator = _coordinator(port)
    await coordinator.open_session("n-1")
    admitted = await coordinator.request_destructive_admission(
        DestructiveKind.DELETE,
        **_session_tokens(coordinator),
    )
    assert admitted.admission is not None
    assert coordinator.mark_destructive_running(admitted.admission) is True

    stale_copy = replace(admitted.admission, operation_token=999)
    assert coordinator.mark_destructive_running(stale_copy) is False
    assert coordinator.finish_destructive(admitted.admission, success=True) is True
    assert coordinator.snapshot is None
    assert coordinator.destructive_admission is None
    assert coordinator.finish_destructive(admitted.admission, success=True) is False


@pytest.mark.asyncio
async def test_failed_flush_never_opens_a_destructive_gate():
    port = FakeDatabaseNotePort()
    port.save_replies.append(DatabaseNotePortSaveReply.failed("Offline"))
    coordinator = _coordinator(port)
    await coordinator.open_session("n-1")
    coordinator.mutate(body="dirty")

    outcome = await coordinator.request_destructive_admission(
        DestructiveKind.DELETE,
        **_session_tokens(coordinator),
    )

    assert outcome.kind is DestructiveAdmissionOutcomeKind.FLUSH_VETOED
    assert coordinator.destructive_admission is None
    assert coordinator.mutate(body="still editable") is True


@pytest.mark.asyncio
async def test_destructive_admission_uses_version_advanced_by_its_own_flush():
    port = FakeDatabaseNotePort()
    port.save_replies.append(DatabaseNotePortSaveReply.saved(version=2))
    coordinator = _coordinator(port)
    await coordinator.open_session("n-1")
    tokens_before_edit = _session_tokens(coordinator)
    coordinator.mutate(body="dirty before delete")

    outcome = await coordinator.request_destructive_admission(
        DestructiveKind.DELETE,
        **tokens_before_edit,
    )

    assert outcome.kind is DestructiveAdmissionOutcomeKind.ADMITTED
    assert outcome.admission is not None
    assert outcome.admission.expected_version == 2
    assert coordinator.snapshot is not None
    assert coordinator.snapshot.dirty is False
    assert coordinator.cancel_destructive(outcome.admission) is True


@pytest.mark.asyncio
async def test_explicit_noop_save_removes_discard_eligibility():
    port = FakeDatabaseNotePort()
    coordinator = _coordinator(port)
    await coordinator.open_session("n-1", untouched_create_token="create-7")
    await coordinator.request_save(explicit=True)

    outcome = await coordinator.request_destructive_admission(
        DestructiveKind.DISCARD_NEW_NOTE,
        create_token="create-7",
        **_session_tokens(coordinator),
    )

    assert outcome.kind is DestructiveAdmissionOutcomeKind.NOT_ELIGIBLE
    assert coordinator.destructive_admission is None


# --- task-32542: the saved-status clock is the reader's, not UTC -------------


@pytest.fixture
def los_angeles_zone(monkeypatch):
    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset is unavailable on this platform")
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()
    yield
    monkeypatch.delenv("TZ", raising=False)
    time.tzset()


@pytest.mark.asyncio
async def test_saved_time_is_rendered_in_local_time(los_angeles_zone):
    """Critique #3 (A 07/18, B 09): "Saved 05:48" at 22:48 PDT. The clock is
    UTC by design (the persisted ``modified_at`` stays ISO/UTC); only the
    status line's rendering changes zone."""
    port = FakeDatabaseNotePort()
    port.save_replies.append(DatabaseNotePortSaveReply.saved(version=2))
    coordinator = _coordinator(
        port, clock=lambda: datetime(2026, 9, 13, 5, 48, tzinfo=timezone.utc)
    )
    await coordinator.open_session("n-1")
    coordinator.mutate(body="edited")

    outcome = await coordinator.request_save(explicit=False)

    assert outcome.kind is NoteSaveOutcomeKind.SAVED
    assert coordinator.snapshot.status_message == "Saved 22:48"
    assert coordinator.snapshot.baseline.modified_at == "2026-09-13T05:48:00+00:00"


@pytest.mark.asyncio
async def test_language_only_edit_is_a_real_save():
    port = FakeDatabaseNotePort(_detail())
    port.save_replies.append(DatabaseNotePortSaveReply.saved(version=2))
    owner = DatabaseNoteSessionCoordinator(port=port)
    await owner.open_session("n-1")
    owner.mutate(content_language="yaml")
    await owner.request_save(explicit=True)
    assert port.save_calls[-1][2].content_language == "yaml"
    assert port.save_calls[-1][2].body == "Original body"
    assert len(port.save_calls) == 1
    assert owner.snapshot.draft_revision == owner.snapshot.saved_revision == 1
    assert owner.snapshot.version == 2


@pytest.mark.asyncio
async def test_language_omission_preserves_and_null_resets_with_source():
    port = FakeDatabaseNotePort(replace(_detail(), content_language="future-mode"))
    owner = DatabaseNoteSessionCoordinator(port)
    await owner.open_session("n-1")
    owner.mutate(body="<>&\n雪")
    assert owner.snapshot.draft.content_language == "future-mode"
    owner.mutate(content_language=None)
    port.save_replies.append(DatabaseNotePortSaveReply.saved(version=2))
    await owner.request_save(explicit=True)
    assert port.save_calls[-1][2].content_language is None
    assert port.save_calls[-1][2].body == "<>&\n雪"
    assert owner.snapshot.draft_revision == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("action", [ConflictAction.RELOAD, ConflictAction.OVERWRITE])
async def test_conflict_resolves_language_with_its_source_revision(action):
    port = FakeDatabaseNotePort(replace(_detail(), content_language="json"))
    owner = DatabaseNoteSessionCoordinator(port)
    await owner.open_session("n-1")
    owner.mutate(body="local: true", content_language="yaml")
    port.save_replies.append(DatabaseNotePortSaveReply.conflict())
    await owner.request_save(explicit=True)
    port.load_replies.append(
        DatabaseNotePortLoadReply.loaded(
            replace(_detail(body="remote", version=3), content_language="future-mode")
        )
    )
    port.save_replies.append(DatabaseNotePortSaveReply.saved(version=4))
    await owner.resolve_conflict(action)
    expected = (
        ("remote", "future-mode")
        if action is ConflictAction.RELOAD
        else ("local: true", "yaml")
    )
    assert (
        owner.snapshot.draft.body,
        owner.snapshot.draft.content_language,
    ) == expected
    assert (
        owner.snapshot.baseline.body,
        owner.snapshot.baseline.content_language,
    ) == expected


@pytest.mark.asyncio
async def test_rejected_empty_draft_recovers_after_navigation_and_restart(tmp_path):
    from tldw_chatbook.Notes.note_draft_recovery import (
        NoteDraftRecovery,
        NoteDraftScope,
    )
    from tldw_chatbook.Notes.notes_device_state_store import NotesDeviceStateStore

    path = tmp_path / "recovery.db"
    scope = NoteDraftScope("profile", "alice", None)
    store = NotesDeviceStateStore(path)
    port = FakeDatabaseNotePort(_detail())
    port.save_replies.extend(
        [
            DatabaseNotePortSaveReply.failed("empty content"),
            DatabaseNotePortSaveReply.failed("empty content"),
        ]
    )
    owner = DatabaseNoteSessionCoordinator(
        port, recovery=NoteDraftRecovery(store), recovery_scope=scope
    )
    await owner.open_session("n-1")
    owner.mutate(body="", content_language="json")
    assert (await owner.request_save(explicit=True)).kind is NoteSaveOutcomeKind.FAILED
    assert (await owner.flush()).kind is NoteFlushOutcomeKind.PERMITTED
    assert owner.close_session()
    store.close()
    reopened = NotesDeviceStateStore(path)
    owner = DatabaseNoteSessionCoordinator(
        FakeDatabaseNotePort(_detail()),
        recovery=NoteDraftRecovery(reopened),
        recovery_scope=scope,
    )
    await owner.open_session("n-1")
    assert owner.snapshot.draft.body == ""
    assert owner.snapshot.draft.content_language == "json"
    assert owner.snapshot.baseline.body == "Original body"
    assert owner.snapshot.dirty
    reopened.close()


@pytest.mark.asyncio
async def test_recovery_storage_failure_fences_navigation(tmp_path, monkeypatch):
    from tldw_chatbook.Notes.note_draft_recovery import (
        NoteDraftRecovery,
        NoteDraftScope,
    )
    from tldw_chatbook.Notes.notes_device_state_store import NotesDeviceStateStore

    store = NotesDeviceStateStore(tmp_path / "recovery.db")
    recovery = NoteDraftRecovery(store)
    owner = DatabaseNoteSessionCoordinator(
        FakeDatabaseNotePort(_detail()),
        recovery=recovery,
        recovery_scope=NoteDraftScope("profile", "alice", None),
    )
    await owner.open_session("n-1")

    async def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(recovery, "save", fail)
    owner.mutate(body="", content_language="yaml")
    assert (await owner.flush()).kind is NoteFlushOutcomeKind.FAILED
    assert not owner.close_session()
    assert owner.snapshot.draft.body == ""
    assert "storage failed" in owner.snapshot.status_message
    store.close()


@pytest.mark.asyncio
async def test_recovered_conflict_keeps_original_base_after_another_edit(tmp_path):
    from tldw_chatbook.Library.library_notes_state import DatabaseNoteDraft
    from tldw_chatbook.Notes.note_draft_recovery import (
        NoteDraftRecovery,
        NoteDraftScope,
    )
    from tldw_chatbook.Notes.notes_device_state_store import NotesDeviceStateStore

    store = NotesDeviceStateStore(tmp_path / "recovery.db")
    recovery = NoteDraftRecovery(store)
    scope = NoteDraftScope("profile", "alice", None)
    await recovery.save(scope, DatabaseNoteDraft("n-1", "Local", "", "", 1, "yaml"), 1)
    owner = DatabaseNoteSessionCoordinator(
        FakeDatabaseNotePort(_detail(version=2)),
        recovery=recovery,
        recovery_scope=scope,
    )
    await owner.open_session("n-1")
    assert owner.snapshot.in_conflict
    owner.mutate(body="still local")
    await owner._retain_current_draft()
    reopened = DatabaseNoteSessionCoordinator(
        FakeDatabaseNotePort(_detail(version=2)),
        recovery=recovery,
        recovery_scope=scope,
    )
    await reopened.open_session("n-1")
    assert reopened.snapshot.in_conflict
    assert (await recovery.load(scope, "n-1"))[1] == 1
    store.close()


class DeferredDraftRecovery:
    """Pause only the storage boundary under test, without replacing the owner."""

    def __init__(self):
        self.pause_save = False
        self.pause_ack = False
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def load(self, scope, note_id):
        return None

    async def save(self, scope, draft, base_version):
        if self.pause_save:
            self.entered.set()
            await self.release.wait()

    async def acknowledge(self, scope, note_id, revision):
        if self.pause_ack:
            self.entered.set()
            await self.release.wait()


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_body", ["pending", ""])
@pytest.mark.parametrize("switch_session", [False, True])
async def test_save_revalidates_after_deferred_retention(initial_body, switch_session):
    recovery = DeferredDraftRecovery()
    port = FakeDatabaseNotePort()
    owner = DatabaseNoteSessionCoordinator(
        port, recovery=recovery, recovery_scope="scope"
    )
    await owner.open_session("n-1")
    owner.mutate(body=initial_body)
    await owner._retain_current_draft()
    recovery.pause_save = True
    port.save_replies.append(DatabaseNotePortSaveReply.saved(version=2))
    saving = asyncio.create_task(owner.request_save(explicit=True))
    await asyncio.wait_for(recovery.entered.wait(), 1)
    if switch_session:
        assert owner.close_session()
        port.load_replies.append(DatabaseNotePortLoadReply.loaded(_detail("n-2")))
        await owner.open_session("n-2")
    else:
        owner.mutate(body="newer <>&", content_language="yaml")
    recovery.release.set()
    outcome = await saving
    if switch_session:
        assert outcome.kind is NoteSaveOutcomeKind.STALE
        assert owner.snapshot.note_id == "n-2"
        assert port.save_calls == []
    else:
        assert outcome.kind is NoteSaveOutcomeKind.SAVED
        assert owner.snapshot.body == "newer <>&"
        assert port.save_calls[0][2].content_language == "yaml"
        assert port.save_calls[0][2].body == "newer <>&"


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("switch_session", [False, True])
async def test_reload_revalidates_after_deferred_acknowledgement(
    missing, switch_session
):
    recovery = DeferredDraftRecovery()
    port = FakeDatabaseNotePort()
    owner = DatabaseNoteSessionCoordinator(
        port, recovery=recovery, recovery_scope="scope"
    )
    await owner.open_session("n-1")
    await _seed_conflict(owner, port)
    port.load_replies.append(
        DatabaseNotePortLoadReply.missing()
        if missing
        else DatabaseNotePortLoadReply.loaded(_detail(version=2))
    )
    recovery.pause_ack = True
    reloading = asyncio.create_task(owner.resolve_conflict(ConflictAction.RELOAD))
    await asyncio.wait_for(recovery.entered.wait(), 1)
    if switch_session:
        assert owner.close_session()
        port.load_replies.append(DatabaseNotePortLoadReply.loaded(_detail("n-2")))
        await owner.open_session("n-2")
    owner.mutate(body="newer <>&", content_language="yaml")
    recovery.release.set()
    outcome = await reloading
    assert outcome.kind is (
        ConflictOutcomeKind.STALE
        if switch_session
        else ConflictOutcomeKind.DRAFT_CHANGED
    )
    assert owner.snapshot.note_id == ("n-2" if switch_session else "n-1")
    assert owner.snapshot.body == "newer <>&"
    await owner._retain_current_draft()


@pytest.mark.asyncio
async def test_recovery_burst_coalesces_and_navigation_drains_latest(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Notes.note_draft_recovery import (
        NoteDraftRecovery,
        NoteDraftScope,
    )
    from tldw_chatbook.Notes.notes_device_state_store import NotesDeviceStateStore

    store = NotesDeviceStateStore(tmp_path / "bounded-recovery.db")
    recovery = NoteDraftRecovery(store)
    scope = NoteDraftScope("profile", "alice", None)
    owner = DatabaseNoteSessionCoordinator(
        FakeDatabaseNotePort(_detail()), recovery=recovery, recovery_scope=scope
    )
    await owner.open_session("n-1")
    owner._port.load_replies.append(DatabaseNotePortLoadReply.loaded(_detail("n-2")))
    original = recovery.save
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def held(scope, draft, base):
        calls.append((scope, draft, base))
        entered.set()
        await release.wait()
        await original(scope, draft, base)

    monkeypatch.setattr(recovery, "save", held)
    try:
        owner.mutate(body="first", content_language="json")
        await entered.wait()
        task = owner._recovery_task
        for revision in range(150):
            owner.mutate(body=f"\ufeffvalue: {revision}\r\n", content_language="yaml")
            assert owner._recovery_task is task
        latest = owner.snapshot.draft
        assert owner._pending_recovery[1].draft == latest
        assert len(calls) == 1
        opening = asyncio.create_task(owner.open_session("n-2"))
        await asyncio.sleep(0)
        assert not opening.done()
        assert not owner.close_session()
        release.set()
        assert (await opening).kind is NoteLoadOutcomeKind.LOADED
        assert owner.snapshot.note_id == "n-2"
        assert len(calls) == 2
        assert calls[-1] == (scope, latest, 1)
        assert await recovery.load(scope, "n-1") == (latest, 1)
        assert owner._pending_recovery is None
        assert task.done()
    finally:
        release.set()
        await asyncio.shield(owner._recovery_task)
        store.close()


@pytest.mark.asyncio
async def test_coalesced_recovery_failure_retries_latest_before_navigation(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Notes.note_draft_recovery import (
        NoteDraftRecovery,
        NoteDraftScope,
    )
    from tldw_chatbook.Notes.notes_device_state_store import NotesDeviceStateStore

    store = NotesDeviceStateStore(tmp_path / "retry-recovery.db")
    recovery = NoteDraftRecovery(store)
    scope = NoteDraftScope("profile", "alice", None)
    owner = DatabaseNoteSessionCoordinator(
        FakeDatabaseNotePort(_detail()), recovery=recovery, recovery_scope=scope
    )
    await owner.open_session("n-1")
    original = recovery.save

    async def fail(*args):
        raise OSError("owned fixture failure")

    monkeypatch.setattr(recovery, "save", fail)
    for revision in range(30):
        owner.mutate(body=f"{revision}\r\n", content_language="jsonl")
    latest = owner.snapshot.draft
    assert (await owner.open_session("n-2")).kind is NoteLoadOutcomeKind.FAILED
    assert not owner.close_session()
    monkeypatch.setattr(recovery, "save", original)
    assert await owner._retain_current_draft()
    assert await recovery.load(scope, "n-1") == (latest, 1)
    assert owner.close_session()
    store.close()


@pytest.mark.asyncio
async def test_recovery_preserves_old_session_edits_during_navigation(tmp_path):
    from tldw_chatbook.Notes.note_draft_recovery import (
        NoteDraftRecovery,
        NoteDraftScope,
    )
    from tldw_chatbook.Notes.notes_device_state_store import NotesDeviceStateStore

    store = NotesDeviceStateStore(tmp_path / "navigation-recovery.db")
    recovery = NoteDraftRecovery(store)
    scope = NoteDraftScope("profile", "alice", None)
    port = FakeDatabaseNotePort(_detail())
    owner = DatabaseNoteSessionCoordinator(
        port, recovery=recovery, recovery_scope=scope
    )
    await owner.open_session("n-1")
    gate = asyncio.Event()
    port.load_gates.append(gate)
    port.load_replies.append(DatabaseNotePortLoadReply.loaded(_detail("n-2")))
    opening = asyncio.create_task(owner.open_session("n-2"))
    await asyncio.sleep(0)
    owner.mutate(body="old session still typing\r\n", content_language="yaml")
    latest = owner.snapshot.draft
    gate.set()
    assert (await opening).kind is NoteLoadOutcomeKind.STALE
    assert owner.snapshot.note_id == "n-1"
    assert await owner._retain_current_draft()
    port.load_replies.append(DatabaseNotePortLoadReply.loaded(_detail("n-2")))
    assert (await owner.open_session("n-2")).kind is NoteLoadOutcomeKind.LOADED
    owner.mutate(body="second session", content_language="text")
    assert await owner._retain_current_draft()
    assert await recovery.load(scope, "n-1") == (latest, 1)
    assert (await recovery.load(scope, "n-2"))[0].body == "second session"
    store.close()


@pytest.mark.asyncio
async def test_late_ack_keeps_newer_coalesced_source_and_mode(tmp_path, monkeypatch):
    from tldw_chatbook.Notes.note_draft_recovery import (
        NoteDraftRecovery,
        NoteDraftScope,
    )
    from tldw_chatbook.Notes.notes_device_state_store import NotesDeviceStateStore

    store = NotesDeviceStateStore(tmp_path / "ack-recovery.db")
    recovery = NoteDraftRecovery(store)
    scope = NoteDraftScope("profile", "alice", None)
    port = FakeDatabaseNotePort(_detail())
    owner = DatabaseNoteSessionCoordinator(
        port, recovery=recovery, recovery_scope=scope
    )
    await owner.open_session("n-1")
    owner.mutate(body="first", content_language="text")
    port.save_replies.extend(
        [
            DatabaseNotePortSaveReply.saved(version=2),
            DatabaseNotePortSaveReply.failed("newer save refused"),
        ]
    )
    entered, release = asyncio.Event(), asyncio.Event()
    original = recovery.acknowledge

    async def held_ack(*args):
        entered.set()
        await release.wait()
        await original(*args)

    monkeypatch.setattr(recovery, "acknowledge", held_ack)
    saving = asyncio.create_task(owner.request_save(explicit=True))
    await entered.wait()
    for revision in range(30):
        owner.mutate(body=f"\ufeffkey: {revision}\r\n", content_language="yaml")
    latest = owner.snapshot.draft
    assert await owner._retain_current_draft()
    assert await recovery.load(scope, "n-1") == (latest, 1)
    release.set()
    assert (await saving).kind is NoteSaveOutcomeKind.FAILED
    assert owner.snapshot.draft == latest
    assert await recovery.load(scope, "n-1") == (latest, 2)
    store.close()
