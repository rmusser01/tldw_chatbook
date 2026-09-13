"""TASK-23027: per-item observation reuse in the production runtime adapter.

The adapter used to re-read every file and re-select every note on every
``observe_root`` pass. Reuse is only allowed after a per-item freshness check
(file stat identity; bulk note versions), so the tests here prove, per
mutation class, that reuse DOES NOT happen when it must not -- and that a
fully warm no-change pass performs zero per-file reads and zero per-note
selects while producing observations equal to a cold pass.

Every world is real: a real temp sync root, a real ``CharactersRAGDB``, the
real device-state store, the real Posix filesystem adapter, and the real
authority/service seams. No hand-made observation doubles.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

import tldw_chatbook.Notes.notes_sync_runtime as runtime_module
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from tldw_chatbook.Notes.notes_device_state_store import (
    NotesDeviceStateStore,
    NotesSyncBindingRecord,
    NotesSyncRootRecord,
)
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService, ScopeType
from tldw_chatbook.Notes.notes_sync_authority import (
    NotesScopeSyncAuthority,
    NotesSyncAuthorityError,
)
from tldw_chatbook.Notes.notes_sync_executor import NotesSyncExecutor
from tldw_chatbook.Notes.notes_sync_filesystem import PosixNotesSyncFilesystem
from tldw_chatbook.Notes.notes_sync_models import (
    NotesSyncBindingState,
    NotesSyncDirection,
    NotesSyncRootState,
)
from tldw_chatbook.Notes.notes_sync_reconciler import plan_reconciliation
from tldw_chatbook.Notes.notes_sync_runtime import _ProductionRuntimeAdapter

pytestmark = pytest.mark.unit

_USER = "test-user"
_N = 4


class _CountingLocalNotes:
    """The production call surface over a real DB, with read counters."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self._db = db
        self.get_note_calls = 0
        self.version_state_calls = 0

    def get_note_by_id(self, _user_id: str, note_id: str):
        self.get_note_calls += 1
        return self._db.get_note_by_id(note_id)

    def get_note_version_states(self, _user_id: str, note_ids):
        self.version_state_calls += 1
        return self._db.get_note_version_states(note_ids)

    def update_note(self, _user_id: str, note_id: str, data, expected_version: int):
        return self._db.update_note(note_id, data, expected_version)

    def soft_delete_note(self, _user_id: str, note_id: str, version: int):
        return self._db.soft_delete_note(note_id, version)

    def add_note(self, _user_id: str, title: str, content: str, note_id=None):
        return self._db.add_note(title, content, note_id=note_id)


@dataclass
class _World:
    root_dir: Path
    db: CharactersRAGDB
    local_notes: _CountingLocalNotes
    scope_service: NotesScopeService
    authority: NotesScopeSyncAuthority
    store: NotesDeviceStateStore
    root: NotesSyncRootRecord
    adapter: _ProductionRuntimeAdapter
    fs_observe_calls: list[str]

    def fresh_adapter(self) -> _ProductionRuntimeAdapter:
        """A cold adapter over the same durable state: the ground truth."""

        return _ProductionRuntimeAdapter(
            self.store,
            self.scope_service,
            local_user_id=_USER,
            recovery_capacity_bytes=64 * 1024 * 1024,
        )

    def reset_counters(self) -> None:
        self.local_notes.get_note_calls = 0
        self.local_notes.version_state_calls = 0
        self.fs_observe_calls.clear()


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _content(index: int) -> str:
    return f"note body {index}\n"


def _rel(index: int) -> str:
    return f"note-{index:04d}.md"


def _note_id(index: int) -> str:
    return f"note-{index:04d}"


@pytest.fixture()
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _World:
    root_dir = (tmp_path / "sync-root").resolve()
    root_dir.mkdir(mode=0o700)
    db = CharactersRAGDB(tmp_path / "chachanotes.sqlite3", client_id=_USER)
    local_notes = _CountingLocalNotes(db)
    scope_service = NotesScopeService(
        local_notes_service=local_notes,
        server_service=None,
        folder_repository=LocalNoteFolderRepository(db),
    )
    authority = NotesScopeSyncAuthority(
        scope_service,
        scope=ScopeType.LOCAL_NOTE,
        user_id=_USER,
        note_scope_id="local_note",
    )
    store = NotesDeviceStateStore(tmp_path / "device-state.sqlite3")
    store.initialize()
    root = NotesSyncRootRecord(
        root_id="root-1",
        note_scope_id="local_note",
        logical_folder_id="folder-1",
        canonical_path=str(root_dir),
        direction=NotesSyncDirection.BIDIRECTIONAL,
        state=NotesSyncRootState.ACTIVE,
    )
    store.create_root(root)

    fs = PosixNotesSyncFilesystem(root_dir)
    with fs:
        for index in range(_N):
            (root_dir / _rel(index)).write_text(_content(index), encoding="utf-8")
            db.add_note(f"Note {index}", _content(index), note_id=_note_id(index))
            snapshot = fs.observe(_rel(index))
            store.create_binding(
                NotesSyncBindingRecord(
                    binding_id=f"binding-{index:04d}",
                    root_id="root-1",
                    note_scope_id="local_note",
                    note_id=_note_id(index),
                    normalized_relative_path=_rel(index),
                    stable_identity_digest=NotesSyncExecutor.stable_identity_digest(
                        snapshot
                    ),
                    state=NotesSyncBindingState.ACTIVE,
                    serialization=snapshot.observation.serialization,
                    content_digest=_digest(_content(index)),
                    note_version=1,
                )
            )

    fs_observe_calls: list[str] = []
    real_observe = PosixNotesSyncFilesystem.observe

    def counting_observe(self, relative_path, **kwargs):
        fs_observe_calls.append(str(relative_path))
        return real_observe(self, relative_path, **kwargs)

    monkeypatch.setattr(PosixNotesSyncFilesystem, "observe", counting_observe)

    adapter = _ProductionRuntimeAdapter(
        store,
        scope_service,
        local_user_id=_USER,
        recovery_capacity_bytes=64 * 1024 * 1024,
    )
    built = _World(
        root_dir=root_dir,
        db=db,
        local_notes=local_notes,
        scope_service=scope_service,
        authority=authority,
        store=store,
        root=root,
        adapter=adapter,
        fs_observe_calls=fs_observe_calls,
    )
    yield built
    adapter.close()
    db.close_connection()


async def _observe(adapter: _ProductionRuntimeAdapter, root: NotesSyncRootRecord):
    request = await adapter.observe_root(root)
    token = plan_reconciliation(request).observation_token
    adapter.release_observation(token)
    return request


def _by_binding(request) -> dict[str, object]:
    return {item.binding_id: item for item in request.bindings}


async def _assert_matches_cold(world: _World, warm_request) -> None:
    """The reused pass must equal what a cold adapter reads from scratch."""

    cold = world.fresh_adapter()
    cold_request = await _observe(cold, world.root)
    assert warm_request == cold_request
    cold.close()


# ---------------------------------------------------------------------------
# The warm no-change pass: zero re-reads, zero re-selects, equal output.
# ---------------------------------------------------------------------------


async def test_warm_unchanged_pass_reads_no_files_and_selects_no_notes(
    world: _World,
) -> None:
    first = await _observe(world.adapter, world.root)
    world.reset_counters()

    second = await _observe(world.adapter, world.root)

    assert world.fs_observe_calls == []
    assert world.local_notes.get_note_calls == 0
    assert world.local_notes.version_state_calls == 1
    assert second == first
    await _assert_matches_cold(world, second)


async def test_cold_pass_still_reads_everything(world: _World) -> None:
    world.reset_counters()

    request = await _observe(world.adapter, world.root)

    assert sorted(world.fs_observe_calls) == [_rel(i) for i in range(_N)]
    assert world.local_notes.get_note_calls == _N
    assert len(request.bindings) == _N


# ---------------------------------------------------------------------------
# Mutation classes: each must break reuse for exactly the changed item.
# ---------------------------------------------------------------------------


async def test_file_edit_is_observed_after_a_warm_pass(world: _World) -> None:
    await _observe(world.adapter, world.root)
    (world.root_dir / _rel(0)).write_text("edited body 0\n", encoding="utf-8")
    world.reset_counters()

    request = await _observe(world.adapter, world.root)

    assert _rel(0) in world.fs_observe_calls
    observed = _by_binding(request)["binding-0000"]
    assert observed.file_digest == _digest("edited body 0\n")
    await _assert_matches_cold(world, request)


async def test_same_length_file_edit_is_observed_after_a_warm_pass(
    world: _World,
) -> None:
    """Size stays identical; only mtime/ctime can catch this class."""

    original = _content(1)
    flipped = original.replace("body", "ydob")
    assert len(flipped) == len(original) and flipped != original
    await _observe(world.adapter, world.root)
    (world.root_dir / _rel(1)).write_text(flipped, encoding="utf-8")
    world.reset_counters()

    request = await _observe(world.adapter, world.root)

    assert _rel(1) in world.fs_observe_calls
    observed = _by_binding(request)["binding-0001"]
    assert observed.file_digest == _digest(flipped)
    await _assert_matches_cold(world, request)


async def test_added_file_is_observed_after_a_warm_pass(world: _World) -> None:
    await _observe(world.adapter, world.root)
    (world.root_dir / "brand-new.md").write_text("fresh\n", encoding="utf-8")
    world.reset_counters()

    request = await _observe(world.adapter, world.root)

    assert "brand-new.md" in world.fs_observe_calls
    added = [
        item for item in request.bindings if item.relative_path == "brand-new.md"
    ]
    assert len(added) == 1 and added[0].file_digest == _digest("fresh\n")
    await _assert_matches_cold(world, request)


async def test_deleted_file_disappears_after_a_warm_pass(world: _World) -> None:
    await _observe(world.adapter, world.root)
    (world.root_dir / _rel(2)).unlink()
    world.reset_counters()

    request = await _observe(world.adapter, world.root)

    observed = _by_binding(request)["binding-0002"]
    assert observed.file_digest is None
    await _assert_matches_cold(world, request)


async def test_renamed_file_is_observed_under_its_new_path(world: _World) -> None:
    await _observe(world.adapter, world.root)
    (world.root_dir / _rel(3)).rename(world.root_dir / "moved.md")
    world.reset_counters()

    request = await _observe(world.adapter, world.root)

    assert "moved.md" in world.fs_observe_calls
    observed = _by_binding(request)["binding-0003"]
    # Same inode → the identity match binds the moved file to its binding.
    assert observed.relative_path == "moved.md"
    assert observed.file_digest == _digest(_content(3))
    await _assert_matches_cold(world, request)


async def test_mtime_only_touch_forces_a_re_read(world: _World) -> None:
    """Same bytes, same size — a touched stat must still invalidate reuse."""

    await _observe(world.adapter, world.root)
    target = world.root_dir / _rel(0)
    stat = target.stat()
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
    world.reset_counters()

    request = await _observe(world.adapter, world.root)

    assert world.fs_observe_calls == [_rel(0)]
    observed = _by_binding(request)["binding-0000"]
    assert observed.file_digest == _digest(_content(0))
    await _assert_matches_cold(world, request)


async def test_db_side_note_edit_is_selected_after_a_warm_pass(
    world: _World,
) -> None:
    await _observe(world.adapter, world.root)
    snapshot = await world.authority.observe(_note_id(0))
    await world.authority.replace(
        snapshot, title="Note 0", content="db edited body\n"
    )
    world.reset_counters()

    request = await _observe(world.adapter, world.root)

    assert world.local_notes.get_note_calls == 1
    observed = _by_binding(request)["binding-0000"]
    assert observed.note_digest == _digest("db edited body\n")
    assert observed.note_version == snapshot.version + 1
    await _assert_matches_cold(world, request)


async def test_db_side_note_delete_is_selected_after_a_warm_pass(
    world: _World,
) -> None:
    await _observe(world.adapter, world.root)
    snapshot = await world.authority.observe(_note_id(1))
    await world.authority.delete(snapshot)
    world.reset_counters()

    request = await _observe(world.adapter, world.root)

    assert world.local_notes.get_note_calls == 1
    observed = _by_binding(request)["binding-0001"]
    assert observed.note_digest is None
    await _assert_matches_cold(world, request)


async def test_new_binding_between_passes_is_observed(world: _World) -> None:
    """Bindings are never cached — a fresh row must appear immediately."""

    await _observe(world.adapter, world.root)
    (world.root_dir / "late.md").write_text("late\n", encoding="utf-8")
    with PosixNotesSyncFilesystem(world.root_dir) as fs:
        snapshot = fs.observe("late.md")
    world.db.add_note("Late", "late\n", note_id="note-late")
    world.store.create_binding(
        NotesSyncBindingRecord(
            binding_id="binding-late",
            root_id="root-1",
            note_scope_id="local_note",
            note_id="note-late",
            normalized_relative_path="late.md",
            stable_identity_digest=NotesSyncExecutor.stable_identity_digest(snapshot),
            state=NotesSyncBindingState.ACTIVE,
            serialization=snapshot.observation.serialization,
            content_digest=_digest("late\n"),
            note_version=1,
        )
    )
    world.reset_counters()

    request = await _observe(world.adapter, world.root)

    observed = _by_binding(request)["binding-late"]
    assert observed.note_digest == _digest("late\n")
    assert observed.bound is True
    await _assert_matches_cold(world, request)


# ---------------------------------------------------------------------------
# Walks: skip-then-change, failure mid-pass, cancellation mid-pass.
# ---------------------------------------------------------------------------


async def test_skip_then_real_change_never_stays_skipped(world: _World) -> None:
    await _observe(world.adapter, world.root)
    warm = await _observe(world.adapter, world.root)
    assert world.fs_observe_calls.count(_rel(0)) == 1  # fixture pass only

    (world.root_dir / _rel(0)).write_text("changed now\n", encoding="utf-8")
    changed = await _observe(world.adapter, world.root)
    observed = _by_binding(changed)["binding-0000"]
    assert observed.file_digest == _digest("changed now\n")
    assert changed != warm

    # And the pass after the change is warm again — over the NEW content.
    world.reset_counters()
    settled = await _observe(world.adapter, world.root)
    assert world.fs_observe_calls == []
    assert _by_binding(settled)["binding-0000"].file_digest == _digest(
        "changed now\n"
    )


async def test_failed_pass_leaves_reuse_and_store_usable(world: _World) -> None:
    await _observe(world.adapter, world.root)

    real = NotesScopeService.get_note_for_sync
    calls = {"count": 0}

    async def failing(self, **kwargs):
        calls["count"] += 1
        raise RuntimeError("private backend outage")

    NotesScopeService.get_note_for_sync = failing
    try:
        (world.root_dir / _rel(0)).write_text("outage edit\n", encoding="utf-8")
        # Sanity: authority observe is service-backed, so it must fail too.
        with pytest.raises(NotesSyncAuthorityError):
            await world.authority.observe(_note_id(1))
    finally:
        NotesScopeService.get_note_for_sync = real

    async def flip_note(content: str) -> None:
        snapshot = await world.authority.observe(_note_id(1))
        await world.authority.replace(snapshot, title="Note 1", content=content)

    await flip_note("post outage\n")
    NotesScopeService.get_note_for_sync = failing
    with pytest.raises(NotesSyncAuthorityError):
        await _observe(world.adapter, world.root)
    NotesScopeService.get_note_for_sync = real

    request = await _observe(world.adapter, world.root)
    observed = _by_binding(request)["binding-0001"]
    assert observed.note_digest == _digest("post outage\n")
    await _assert_matches_cold(world, request)


async def test_bulk_version_read_failure_propagates(world: _World) -> None:
    await _observe(world.adapter, world.root)

    real = NotesScopeService.get_note_version_states_for_sync

    async def failing(self, **kwargs):
        raise RuntimeError("private backend outage")

    NotesScopeService.get_note_version_states_for_sync = failing
    try:
        with pytest.raises(NotesSyncAuthorityError):
            await _observe(world.adapter, world.root)
    finally:
        NotesScopeService.get_note_version_states_for_sync = real

    request = await _observe(world.adapter, world.root)
    await _assert_matches_cold(world, request)


async def test_cancel_mid_pass_leaves_adapter_usable(world: _World) -> None:
    """Quit with a sync in flight: cancel inside the pass, then observe again."""

    await _observe(world.adapter, world.root)
    (world.root_dir / _rel(0)).write_text("mid flight\n", encoding="utf-8")

    entered = asyncio.Event()
    release = threading.Event()
    real_observe = PosixNotesSyncFilesystem.observe
    loop = asyncio.get_running_loop()

    def blocking_observe(self, relative_path, **kwargs):
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5.0)
        return real_observe(self, relative_path, **kwargs)

    PosixNotesSyncFilesystem.observe = blocking_observe
    task = asyncio.create_task(world.adapter.observe_root(world.root))
    try:
        await asyncio.wait_for(entered.wait(), 5.0)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        PosixNotesSyncFilesystem.observe = real_observe
        release.set()

    request = await _observe(world.adapter, world.root)
    observed = _by_binding(request)["binding-0000"]
    assert observed.file_digest == _digest("mid flight\n")
    await _assert_matches_cold(world, request)


# ---------------------------------------------------------------------------
# Interleaving: concurrent DB writes never leave the cache permanently stale
# and every observed (version, content) pair is a really committed pair.
# ---------------------------------------------------------------------------


async def test_concurrent_note_writer_yields_only_committed_pairs(
    world: _World,
) -> None:
    await _observe(world.adapter, world.root)
    committed: dict[int, str] = {}
    snapshot = await world.authority.observe(_note_id(0))
    committed[snapshot.version] = snapshot.content
    stop = threading.Event()
    failures: list[BaseException] = []

    def writer() -> None:
        try:
            expected_version = snapshot.version
            flip = 0
            while not stop.is_set():
                flip += 1
                content = f"flip {flip}\n"
                world.db.update_note(
                    _note_id(0), {"content": content}, expected_version
                )
                expected_version += 1
                committed[expected_version] = content
                time.sleep(0.001)
        except BaseException as error:  # pragma: no cover - failure reporting
            failures.append(error)

    thread = threading.Thread(target=writer)
    thread.start()
    try:
        for _ in range(10):
            request = await _observe(world.adapter, world.root)
            observed = _by_binding(request)["binding-0000"]
            assert observed.note_version in committed
            assert observed.note_digest == _digest(committed[observed.note_version])
    finally:
        stop.set()
        thread.join(5.0)
    assert not failures

    # Quiescent: the final pass equals a cold read of the final state.
    request = await _observe(world.adapter, world.root)
    await _assert_matches_cold(world, request)


# ---------------------------------------------------------------------------
# Budget: an over-budget item is simply not cached; correctness unchanged.
# ---------------------------------------------------------------------------


async def test_over_budget_items_are_re_read_not_mis_served(
    world: _World, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime_module, "_OBSERVATION_REUSE_BUDGET_BYTES", 40)
    await _observe(world.adapter, world.root)
    world.reset_counters()

    request = await _observe(world.adapter, world.root)

    # Under a 40-byte budget only the first file fits; the rest re-read.
    assert len(world.fs_observe_calls) == _N - 1
    await _assert_matches_cold(world, request)


async def test_close_drops_the_reuse_cache(world: _World) -> None:
    await _observe(world.adapter, world.root)
    assert world.adapter._observation_reuse
    world.adapter.close()
    assert not world.adapter._observation_reuse
