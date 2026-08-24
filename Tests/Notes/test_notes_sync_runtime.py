"""Application-owned lasting-sync runtime contracts."""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from tldw_chatbook.Notes.notes_device_state_store import (
    NotesDeviceStateStore,
    NotesSyncBindingRecord,
    NotesSyncOperationRecord,
    NotesSyncRootRecord,
    NotesSyncStoreSetting,
)
from tldw_chatbook.Notes.notes_sync_coordinator import RootAdmissionState
from tldw_chatbook.Notes.notes_sync_models import (
    NotesSyncActionKind,
    NotesSyncBindingState,
    NotesSyncDirection,
    NotesSyncOperationState,
    NotesSyncRootState,
    NotesSyncSerializationProfile,
)
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
from tldw_chatbook.Notes.notes_sync_executor import (
    NotesSyncExecutionResult,
    NotesSyncExecutor,
)
from tldw_chatbook.Notes.notes_sync_filesystem import PosixNotesSyncFilesystem
from tldw_chatbook.Notes.notes_sync_reconciler import (
    BindingObservation,
    ReconciliationInput,
    plan_reconciliation,
)


pytestmark = pytest.mark.unit
_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_EMPTY_LEGACY_SCHEMA = """
CREATE TABLE notes (
    id TEXT, version INTEGER, file_path_on_disk TEXT,
    relative_file_path_on_disk TEXT, sync_root_folder TEXT,
    last_synced_disk_file_hash TEXT, last_synced_disk_file_mtime REAL,
    is_externally_synced INTEGER, sync_strategy TEXT,
    sync_excluded INTEGER, file_extension TEXT, deleted INTEGER
);
CREATE TABLE sync_sessions (
    session_id TEXT, sync_root_folder TEXT, sync_direction TEXT,
    conflict_resolution TEXT, started_at TEXT, completed_at TEXT,
    status TEXT, total_files INTEGER, processed_files INTEGER,
    conflicts_found INTEGER, errors_count INTEGER, client_id TEXT, summary TEXT
);
"""


def _input(
    *,
    generation: int = 1,
    file_digest: str | None = _B,
    note_digest: str | None = _A,
    root_available: bool = True,
    root_overlap: bool = False,
    write_capable: bool = True,
    direction: NotesSyncDirection = NotesSyncDirection.BIDIRECTIONAL,
    relative_path: str = "note.md",
) -> ReconciliationInput:
    return ReconciliationInput(
        root_id="root-1",
        direction=direction,
        bindings=(
            BindingObservation(
                binding_id="binding-1",
                baseline_file_digest=_A,
                baseline_note_digest=_A,
                baseline_identity_digest=_C,
                baseline_relative_path="note.md",
                file_digest=file_digest,
                note_digest=note_digest,
                file_identity_digest=_C if file_digest is not None else None,
                relative_path=relative_path,
                note_scope_id="local_note",
                note_id="note-1",
                note_version=generation,
            ),
        ),
        observation_generation=generation,
        expected_generation=generation,
        root_available=root_available,
        root_overlap=root_overlap,
        write_capable=write_capable,
    )


def _two_action_input() -> ReconciliationInput:
    first_input = _input()
    first = first_input.bindings[0]
    second = replace(
        first,
        binding_id="binding-2",
        relative_path="second.md",
        baseline_relative_path="second.md",
        note_id="note-2",
        file_digest=_A,
        note_digest=_B,
    )
    return replace(first_input, bindings=(first, second))


def _store(tmp_path: Path, *, marker: bool = True) -> NotesDeviceStateStore:
    store = NotesDeviceStateStore(tmp_path / "sync.sqlite3")
    store.initialize()
    store.create_root(
        NotesSyncRootRecord(
            root_id="root-1",
            note_scope_id="local_note",
            logical_folder_id="folder-1",
            canonical_path=str(tmp_path / "root"),
            direction=NotesSyncDirection.BIDIRECTIONAL,
            state=NotesSyncRootState.ACTIVE,
        )
    )
    if marker:
        store.set_setting(
            NotesSyncStoreSetting("cutover_marker", "notes-sync-cutover-v1")
        )
    (tmp_path / "root").mkdir()
    return store


@pytest.mark.asyncio
async def test_abandon_setup_ignores_persisted_root_review_authority(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    owner, coordinator, _watcher = _owner(
        store=store,
        admitted=True,
        adapter=_Adapter([_input()]),
    )
    await owner.start()
    try:
        await owner.check_root("root-1")
        path_before = owner._root_paths["root-1"]
        status_before = owner.snapshot().roots
        events_before = tuple(coordinator.events)
        assert "root-1" not in owner._setup_reviews

        await owner.abandon_setup("root-1")

        assert owner._root_paths["root-1"] == path_before
        assert owner.snapshot().roots == status_before
        assert tuple(coordinator.events) == events_before
        assert store.get_root("root-1").canonical_path == path_before
    finally:
        await owner.shutdown()


@dataclass
class _Lease:
    authoritative: bool = True


class _Admission:
    def __init__(self, state: RootAdmissionState) -> None:
        self.state = state
        self.lease = _Lease() if state is RootAdmissionState.OWNER else None
        self.reason_code = None if state is RootAdmissionState.OWNER else state.value

    @property
    def can_plan(self) -> bool:
        return self.state is RootAdmissionState.OWNER

    @property
    def can_write(self) -> bool:
        return self.can_plan

    def require_authority(self, _operation: str) -> _Lease:
        if self.lease is None or not self.lease.authoritative:
            raise RuntimeError("admission_closed")
        return self.lease


class _Coordinator:
    def __init__(self, state: RootAdmissionState = RootAdmissionState.OWNER) -> None:
        self.state = state
        self.acquire_calls = 0
        self.events: list[str] = []
        self.validations: list[dict[str, object]] = []
        self.last_admission: _Admission | None = None

    def try_acquire(self, _path: str, **_kwargs: object) -> _Admission:
        self.acquire_calls += 1
        self.validations.append(dict(_kwargs))
        self.last_admission = _Admission(self.state)
        return self.last_admission

    def close_admission(self, _lease: _Lease, settle) -> None:
        self.events.append("lease-admission-closed")
        settle()
        self.events.append("lease-settled")
        self.events.append("lease-released")


class _RetryingCloseCoordinator(_Coordinator):
    def __init__(self, root_paths: dict[str, str]) -> None:
        super().__init__()
        self._root_paths = {path: root_id for root_id, path in root_paths.items()}
        self._lease_roots: dict[int, str] = {}
        self.close_attempts: list[str] = []
        self._fail_root_one = True

    def try_acquire(self, path: str, **kwargs: object) -> _Admission:
        admission = super().try_acquire(path, **kwargs)
        assert admission.lease is not None
        self._lease_roots[id(admission.lease)] = self._root_paths[path]
        return admission

    def close_admission(self, lease: _Lease, settle) -> None:
        root_id = self._lease_roots[id(lease)]
        self.close_attempts.append(root_id)
        if root_id == "root-1" and self._fail_root_one:
            self._fail_root_one = False
            raise RuntimeError("private path must not escape")
        super().close_admission(lease, settle)


class _BlockingRootListStore(NotesDeviceStateStore):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.list_started = threading.Event()
        self.list_release = threading.Event()

    def list_root_summaries(self):
        self.list_started.set()
        self.list_release.wait()
        return super().list_root_summaries()


class _PostCommitReadFailStore(NotesDeviceStateStore):
    """Inject the former read-after-commit failure window."""

    fail_committed_read = False

    def activate_migration_candidate(
        self,
        root_id: str,
        logical_folder_id: str,
        binding_ids: tuple[str, ...],
    ) -> NotesSyncRootRecord:
        self.fail_committed_read = True
        try:
            return super().activate_migration_candidate(
                root_id, logical_folder_id, binding_ids
            )
        finally:
            self.fail_committed_read = False

    def get_root(self, root_id: str) -> NotesSyncRootRecord:
        if self.fail_committed_read:
            raise RuntimeError("injected post-commit read failure")
        return super().get_root(root_id)


class _RecoveryPostCommitReadFailStore(NotesDeviceStateStore):
    """Reject a read after a committed activation-recovery write."""

    fail_committed_read = False

    def record_root_activation_recovery(
        self, root_id: str, logical_folder_id: str
    ) -> NotesSyncRootRecord:
        self.fail_committed_read = True
        try:
            return super().record_root_activation_recovery(root_id, logical_folder_id)
        finally:
            self.fail_committed_read = False

    def get_root(self, root_id: str) -> NotesSyncRootRecord:
        if self.fail_committed_read:
            raise RuntimeError("injected post-commit recovery read failure")
        return super().get_root(root_id)


class _StartupInventoryFailStore(NotesDeviceStateStore):
    def __init__(self, path: Path, stage: str) -> None:
        super().__init__(path)
        self.stage = stage

    def list_root_summaries(self):
        if self.stage == "roots":
            raise RuntimeError("private inventory failure")
        return super().list_root_summaries()

    def list_incomplete_operations(self):
        if self.stage == "recovery":
            raise RuntimeError("private recovery failure")
        return super().list_incomplete_operations()


class _Executor:
    def __init__(self) -> None:
        self.executed: list[object] = []
        self.reconstructed: list[str] = []
        self.resumed: list[object] = []

    async def execute(self, request: object):
        self.executed.append(request)
        return NotesSyncExecutionResult(
            operation_id=getattr(
                request,
                "operation_id",
                getattr(request, "action_id", "operation-1"),
            ),
            state=NotesSyncOperationState.COMPLETED,
            recovery_required=False,
        )

    async def reconstruct_request(self, operation_id: str) -> object:
        self.reconstructed.append(operation_id)
        return operation_id

    async def resume(self, request: object):
        self.resumed.append(request)
        return type(
            "Result",
            (),
            {"state": NotesSyncOperationState.COMPLETED, "reason_code": None},
        )()


class _Adapter:
    def __init__(self, observations: list[ReconciliationInput]) -> None:
        self.observations = observations
        self.observe_calls = 0
        self.executor = _Executor()
        self.created_folders: list[str] = []
        self.rolled_back_folders: list[object] = []

    async def observe_root(self, _root: NotesSyncRootRecord) -> ReconciliationInput:
        result = self.observations[min(self.observe_calls, len(self.observations) - 1)]
        self.observe_calls += 1
        return result

    async def build_execution_request(self, _root, _observations, _plan, action):
        return action

    def executor_for(
        self, _root: NotesSyncRootRecord, *, after_stage=None
    ) -> _Executor:
        self.executor.after_stage = after_stage
        return self.executor

    async def create_root_folder(self, display_name: str) -> tuple[str, object]:
        self.created_folders.append(display_name)
        return "folder-real", "folder-receipt"

    async def rollback_root_folder(self, receipt: object) -> None:
        self.rolled_back_folders.append(receipt)


class _MultiRootAdapter(_Adapter):
    async def observe_root(self, root: NotesSyncRootRecord) -> ReconciliationInput:
        self.observe_calls += 1
        return replace(_input(file_digest=_A, note_digest=_A), root_id=root.root_id)


class _ConcurrentBundleAdapter(_Adapter):
    def __init__(self, observations: list[ReconciliationInput]) -> None:
        super().__init__(observations)
        self.private_bundles: dict[str, object] = {}
        self.build_started: list[str] = []
        self.release_build: dict[str, asyncio.Event] = {}
        self.both_started = asyncio.Event()

    async def observe_root(self, root: NotesSyncRootRecord) -> ReconciliationInput:
        result = await super().observe_root(root)
        token = plan_reconciliation(result).observation_token
        self.private_bundles[token] = object()
        return result

    async def build_execution_request(self, _root, _observations, plan, action):
        token = plan.observation_token
        assert token in self.private_bundles
        self.build_started.append(token)
        release = self.release_build.setdefault(token, asyncio.Event())
        if len(self.build_started) == 2:
            self.both_started.set()
        await release.wait()
        assert token in self.private_bundles
        return action

    def release_observation(self, observation_token: str) -> None:
        self.private_bundles.pop(observation_token, None)


class _LocalNotes:
    def __init__(self, content: str) -> None:
        self.note = {
            "id": "note-1",
            "title": "Note",
            "content": content,
            "version": 1,
            "deleted": False,
        }

    def get_note_by_id(self, _user_id: str, _note_id: str):
        return dict(self.note)

    def update_note(
        self,
        _user_id: str,
        _note_id: str,
        values: dict[str, str],
        expected_version: int,
    ) -> bool:
        if self.note["version"] != expected_version:
            return False
        self.note.update(values)
        self.note["version"] = expected_version + 1
        return True


class _BlockingHeartbeatLocalNotes(_LocalNotes):
    def __init__(self, content: str) -> None:
        super().__init__(content)
        self.started = threading.Event()
        self.release = threading.Event()

    def get_note_by_id(self, _user_id: str, _note_id: str):
        self.started.set()
        self.release.wait(timeout=2)
        return dict(self.note)


class _Folders:
    def __init__(self) -> None:
        self.created: list[str] = []
        self.deleted: list[str] = []

    def create_folder(self, *, name: str, parent_id: str | None):
        from tldw_chatbook.Notes.note_folder_models import NoteFolder

        self.created.append(name)
        return NoteFolder(
            folder_id="folder-real",
            parent_id=parent_id,
            name=name,
            path=name,
            normalized_path=name.casefold(),
            version=1,
            deleted=False,
        )

    def soft_delete_folder(self, folder_id: str, *, expected_version: int):
        self.deleted.append(folder_id)
        return object()

    def reconcile_managed(self, **_kwargs: object) -> tuple[object, ...]:
        return ()


class _CreatingLocalNotes(_LocalNotes):
    def get_note_by_id(self, _user_id: str, note_id: str):
        return dict(self.note) if self.note["id"] == note_id else None

    def add_note(
        self,
        _user_id: str,
        title: str,
        content: str,
        *,
        note_id: str,
    ) -> str:
        self.note = {
            "id": note_id,
            "title": title,
            "content": content,
            "version": 1,
            "deleted": False,
        }
        return note_id


class _FailingExecutor(_Executor):
    async def execute(self, request: object):
        self.executed.append(request)
        raise RuntimeError("forced executor failure")


class _PartialCleanupExecutor(_Executor):
    def __init__(self) -> None:
        super().__init__()
        self.cleanup: list[str] = []

    async def execute(self, request: object):
        self.executed.append(request)
        return type(
            "Result",
            (),
            {
                "operation_id": "operation-1",
                "state": NotesSyncOperationState.FIRST_AUTHORITY_APPLIED,
                "reason_code": "replacement_cleanup_pending",
            },
        )()

    async def resolve_filesystem_cleanup(self, operation_id: str):
        self.cleanup.append(operation_id)
        return type(
            "Result",
            (),
            {"state": NotesSyncOperationState.COMPLETED, "reason_code": None},
        )()


class _AttentionCleanupExecutor(_PartialCleanupExecutor):
    async def resolve_filesystem_cleanup(self, operation_id: str):
        self.cleanup.append(operation_id)
        return type(
            "Result",
            (),
            {
                "operation_id": operation_id,
                "state": NotesSyncOperationState.NEEDS_ATTENTION,
                "reason_code": "replacement_cleanup_pending",
            },
        )()


class _FailingObservationAdapter(_Adapter):
    async def observe_root(self, root: NotesSyncRootRecord) -> ReconciliationInput:
        if self.observe_calls:
            raise RuntimeError("forced observation failure")
        return await super().observe_root(root)


class _OverrideAdapter(_Adapter):
    async def build_execution_request(self, _root, _observations, _plan, action):
        return type("Request", (), {"action": action, "direction_override": object()})()


class _BlockingExecutor(_Executor):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def execute(self, request: object):
        self.executed.append(request)
        self.started.set()
        await self.release.wait()
        return NotesSyncExecutionResult(
            operation_id=getattr(
                request,
                "operation_id",
                getattr(request, "action_id", "operation-1"),
            ),
            state=NotesSyncOperationState.COMPLETED,
            recovery_required=False,
        )


class _InvalidatingExecutor(_Executor):
    def __init__(self, invalidate) -> None:
        super().__init__()
        self.invalidate = invalidate

    async def execute(self, request: object):
        self.executed.append(request)
        self.invalidate()
        return NotesSyncExecutionResult(
            operation_id=getattr(
                request,
                "operation_id",
                getattr(request, "action_id", "operation-1"),
            ),
            state=NotesSyncOperationState.COMPLETED,
            recovery_required=False,
        )


class _InvalidatingReconstructExecutor(_Executor):
    def __init__(self, invalidate) -> None:
        super().__init__()
        self.invalidate = invalidate

    async def reconstruct_request(self, operation_id: str) -> object:
        self.reconstructed.append(operation_id)
        self.invalidate()
        return operation_id


class _Watcher:
    def __init__(self, events: list[str] | None = None) -> None:
        self.events = events if events is not None else []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def run(self) -> None:
        self.events.append("watcher-started")
        self.started.set()
        await self.release.wait()

    async def stop(self) -> None:
        self.events.append("watcher-stopped")
        self.release.set()


class _FailedWatcher(_Watcher):
    async def run(self) -> None:
        self.events.append("watcher-started")
        self.started.set()
        raise RuntimeError("forced watcher failure")


def _owner(
    *,
    store: NotesDeviceStateStore,
    admitted: bool,
    adapter: _Adapter,
    coordinator: _Coordinator | None = None,
    watcher: _Watcher | None = None,
    migrated: list[str] | None = None,
):
    from tldw_chatbook.Notes.notes_sync_runtime import NotesSyncRuntimeOwner

    selected_coordinator = coordinator or _Coordinator()
    selected_watcher = watcher or _Watcher()
    migrations = migrated if migrated is not None else []
    owner = NotesSyncRuntimeOwner(
        store=store,
        migrate_legacy=lambda: migrations.append("migrated"),
        coordinator=selected_coordinator,
        adapter=adapter,
        watcher_factory=lambda _schedule_hint: selected_watcher,
        cutover_admitted=admitted,
        profile_process_is_sole=True,
    )
    return owner, selected_coordinator, selected_watcher


@pytest.mark.asyncio
@pytest.mark.parametrize("admitted,marker", [(False, True)])
async def test_startup_is_inert_until_both_private_cutover_gates_exist(
    tmp_path: Path,
    admitted: bool,
    marker: bool,
) -> None:
    migrations: list[str] = []
    adapter = _Adapter([_input()])
    owner, coordinator, watcher = _owner(
        store=_store(tmp_path, marker=marker),
        admitted=admitted,
        adapter=adapter,
        migrated=migrations,
    )

    await owner.start()

    assert migrations == []
    assert owner.snapshot().status == "awaiting_cutover"
    assert owner.snapshot().next_action == "finish_upgrade"
    assert coordinator.acquire_calls == 0
    assert adapter.observe_calls == 0
    assert not watcher.started.is_set()
    with pytest.raises(RuntimeError, match="cutover"):
        await owner.activate_root("root-1", authorization=None)
    await owner.shutdown()


@pytest.mark.asyncio
async def test_startup_rejects_a_noncanonical_cutover_marker(tmp_path: Path) -> None:
    store = _store(tmp_path, marker=False)
    store.set_setting(NotesSyncStoreSetting("cutover_marker", "wrong-marker"))
    adapter = _Adapter([_input()])
    migrations: list[str] = []
    before = (tmp_path / "sync.sqlite3").read_bytes()
    owner, coordinator, watcher = _owner(
        store=store,
        admitted=True,
        adapter=adapter,
        migrated=migrations,
    )

    await owner.start()

    assert owner.snapshot().status == "awaiting_cutover"
    assert coordinator.acquire_calls == 0
    assert adapter.observe_calls == 0
    assert not watcher.started.is_set()
    assert migrations == []
    assert (tmp_path / "sync.sqlite3").read_bytes() == before
    await owner.shutdown()


@pytest.mark.asyncio
async def test_migration_failure_is_bounded_and_never_starts_runtime_work(
    tmp_path: Path,
) -> None:
    from tldw_chatbook.Notes.notes_sync_runtime import NotesSyncRuntimeOwner

    coordinator = _Coordinator()
    watcher = _Watcher()
    owner = NotesSyncRuntimeOwner(
        store=_store(tmp_path, marker=False),
        migrate_legacy=lambda: (_ for _ in ()).throw(RuntimeError("private")),
        coordinator=coordinator,
        adapter=_Adapter([_input()]),
        watcher_factory=lambda _schedule: watcher,
        cutover_admitted=True,
        profile_process_is_sole=True,
    )

    await owner.start()

    assert (owner.snapshot().status, owner.snapshot().next_action) == (
        "failed",
        "review_settings",
    )
    assert coordinator.acquire_calls == 0
    assert not watcher.started.is_set()
    assert "private" not in repr(owner.snapshot())
    await owner.shutdown()


@pytest.mark.asyncio
async def test_first_cutover_does_not_reread_the_marker_after_durable_write(
    tmp_path: Path,
) -> None:
    class _SingleReadMarkerStore(NotesDeviceStateStore):
        marker_reads = 0

        def get_setting(self, key: str) -> NotesSyncStoreSetting | None:
            if key == "cutover_marker":
                self.marker_reads += 1
                if self.marker_reads > 1:
                    raise RuntimeError("private second marker read")
            return super().get_setting(key)

    store = _SingleReadMarkerStore(tmp_path / "sync.sqlite3")
    store.initialize()
    owner, _, _ = _owner(
        store=store,
        admitted=True,
        adapter=_Adapter([]),
    )

    await owner.start()

    assert store.marker_reads == 1
    assert owner.snapshot().status == "active"
    assert NotesDeviceStateStore.get_setting(store, "cutover_marker") is not None
    await owner.shutdown()


@pytest.mark.asyncio
async def test_inert_builder_performs_no_legacy_read_or_runtime_construction(
    tmp_path: Path,
) -> None:
    from tldw_chatbook.Notes.notes_sync_runtime import (
        build_notes_sync_runtime_owner,
    )

    calls: list[str] = []

    def forbidden() -> object:
        calls.append("forbidden")
        raise AssertionError("inert builder crossed a post-cutover boundary")

    owner = build_notes_sync_runtime_owner(
        notes_scope_service=object(),
        cutover_admitted=False,
        profile_process_is_sole=True,
        database_path=tmp_path / "sync.sqlite3",
        migrate_legacy=lambda: None,
        adapter=_Adapter([_input()]),
        coordinator=forbidden,
    )

    await owner.start()

    assert calls == []
    assert owner.snapshot().status == "awaiting_cutover"
    await owner.shutdown()


def test_builder_requires_the_concrete_notes_scope_service_for_production(
    tmp_path: Path,
) -> None:
    from tldw_chatbook.Notes.notes_sync_runtime import (
        build_notes_sync_runtime_owner,
    )

    with pytest.raises(TypeError, match="NotesScopeService"):
        build_notes_sync_runtime_owner(
            notes_scope_service=object(),
            cutover_admitted=True,
            profile_process_is_sole=True,
            database_path=tmp_path / "sync.sqlite3",
            migrate_legacy=lambda: None,
        )


def test_builder_cannot_admit_cutover_without_the_idempotent_migrator(
    tmp_path: Path,
) -> None:
    from tldw_chatbook.Notes.notes_sync_runtime import (
        build_notes_sync_runtime_owner,
    )

    with pytest.raises(ValueError, match="migrate_legacy"):
        build_notes_sync_runtime_owner(
            notes_scope_service=object(),
            cutover_admitted=True,
            profile_process_is_sole=True,
            database_path=tmp_path / "sync.sqlite3",
            adapter=_Adapter([_input()]),
            watcher_factory=lambda _schedule: _Watcher(),
        )


@pytest.mark.asyncio
async def test_production_migration_wrapper_is_idempotent_on_the_real_stores(
    tmp_path: Path,
) -> None:
    from tldw_chatbook.Notes.notes_sync_runtime import (
        build_notes_sync_legacy_migrator,
    )

    legacy_path = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(legacy_path)
    connection.executescript(_EMPTY_LEGACY_SCHEMA)
    connection.close()
    database_path = tmp_path / "sync.sqlite3"
    store = NotesDeviceStateStore(database_path)
    store.initialize()
    migrate = build_notes_sync_legacy_migrator(
        database_path=database_path,
        legacy_connection=lambda: sqlite3.connect(legacy_path),
        settings={},
        note_scope_id="local_note",
        file_notes_binding=lambda: None,
        private_paths=(),
    )

    first = await asyncio.to_thread(migrate)
    second = await asyncio.to_thread(migrate)

    assert not first.already_migrated
    assert second.already_migrated
    with store.transaction() as db:
        assert (
            db.execute("SELECT COUNT(*) FROM notes_sync_legacy_migrations").fetchone()[
                0
            ]
            == 1
        )


@pytest.mark.asyncio
async def test_production_builder_executes_one_safe_action_durably(
    tmp_path: Path,
) -> None:
    from tldw_chatbook.Notes.notes_sync_runtime import (
        build_notes_sync_runtime_owner,
    )

    store = _store(tmp_path)
    file_path = tmp_path / "root" / "note.md"
    file_path.write_text("new", encoding="utf-8")
    with PosixNotesSyncFilesystem(tmp_path / "root") as filesystem:
        file = filesystem.observe("note.md")
    store.create_binding(
        NotesSyncBindingRecord(
            binding_id="binding-1",
            root_id="root-1",
            note_scope_id="local_note",
            note_id="note-1",
            normalized_relative_path="note.md",
            stable_identity_digest=NotesSyncExecutor.stable_identity_digest(file),
            state=NotesSyncBindingState.ACTIVE,
            serialization=file.observation.serialization,
            content_digest=hashlib.sha256(b"old").hexdigest(),
            note_version=1,
        )
    )
    local_notes = _LocalNotes("old")
    service = NotesScopeService(local_notes, None, folder_repository=_Folders())
    owner = build_notes_sync_runtime_owner(
        notes_scope_service=service,
        cutover_admitted=True,
        profile_process_is_sole=True,
        database_path=tmp_path / "sync.sqlite3",
        migrate_legacy=lambda: None,
        local_user_id="user-1",
        recovery_capacity_bytes=1024 * 1024,
    )

    await owner.start()

    assert local_notes.note["content"] == "new"
    operations = store.list_incomplete_operations()
    assert [(item.state, item.reason_code) for item in operations] == []
    with store.transaction() as connection:
        assert (
            connection.execute("SELECT state FROM notes_sync_operations").fetchone()[0]
            == NotesSyncOperationState.COMPLETED.value
        )
    assert owner._adapter._bundles == {}
    review_tokens: list[str] = []
    for content in ("next", "latest"):
        file_path.write_text(content, encoding="utf-8")
        review = await owner.check_root("root-1")
        review_tokens.append(review.observation_token)
        assert owner._adapter._bundles == {}
    assert len(set(review_tokens)) == 2
    await owner.shutdown()


@pytest.mark.asyncio
async def test_fresh_authority_releases_only_its_private_observation_bundle(
    tmp_path: Path,
) -> None:
    adapter = _ConcurrentBundleAdapter(
        [
            _input(file_digest=_A, note_digest=_A),
            _input(generation=2),
            _input(generation=3, file_digest=_C),
        ]
    )
    owner, _, _ = _owner(store=_store(tmp_path), admitted=True, adapter=adapter)
    await owner.start()
    assert adapter.private_bundles == {}
    root = owner._store.get_root("root-1")

    first = asyncio.create_task(owner._fresh_authority(root))
    second = asyncio.create_task(owner._fresh_authority(root))
    await adapter.both_started.wait()
    first_token, second_token = adapter.build_started
    adapter.release_build[first_token].set()
    await first

    assert first_token not in adapter.private_bundles
    assert second_token in adapter.private_bundles
    adapter.release_build[second_token].set()
    await second

    assert adapter.private_bundles == {}
    await owner.shutdown()


@pytest.mark.asyncio
async def test_production_note_observation_does_not_block_the_event_loop(
    tmp_path: Path,
) -> None:
    from tldw_chatbook.Notes.notes_sync_runtime import (
        build_notes_sync_runtime_owner,
    )

    store = _store(tmp_path)
    file_path = tmp_path / "root" / "note.md"
    file_path.write_text("same", encoding="utf-8")
    with PosixNotesSyncFilesystem(tmp_path / "root") as filesystem:
        file = filesystem.observe("note.md")
    store.create_binding(
        NotesSyncBindingRecord(
            binding_id="binding-1",
            root_id="root-1",
            note_scope_id="local_note",
            note_id="note-1",
            normalized_relative_path="note.md",
            stable_identity_digest=NotesSyncExecutor.stable_identity_digest(file),
            state=NotesSyncBindingState.ACTIVE,
            serialization=file.observation.serialization,
            content_digest=file.observation.content_digest,
            note_version=1,
        )
    )
    local_notes = _BlockingHeartbeatLocalNotes("same")
    service = NotesScopeService(local_notes, None, folder_repository=_Folders())
    owner = build_notes_sync_runtime_owner(
        notes_scope_service=service,
        cutover_admitted=True,
        profile_process_is_sole=True,
        database_path=tmp_path / "sync.sqlite3",
        migrate_legacy=lambda: None,
        local_user_id="user-1",
        recovery_capacity_bytes=1024 * 1024,
    )
    loop = asyncio.get_running_loop()
    heartbeat = asyncio.Event()
    heartbeat_seen_before_release: list[bool] = []

    def coordinate_release() -> None:
        assert local_notes.started.wait(timeout=1)
        loop.call_soon_threadsafe(heartbeat.set)
        time.sleep(0.05)
        heartbeat_seen_before_release.append(heartbeat.is_set())
        local_notes.release.set()

    coordinator = threading.Thread(target=coordinate_release)
    coordinator.start()
    await owner.start()
    await asyncio.to_thread(coordinator.join)

    assert heartbeat_seen_before_release == [True]
    await owner.shutdown()


@pytest.mark.asyncio
async def test_production_builder_discovers_an_unbound_text_file_once(
    tmp_path: Path,
) -> None:
    from tldw_chatbook.Notes.notes_sync_runtime import (
        build_notes_sync_runtime_owner,
    )

    store = _store(tmp_path)
    sync_file = tmp_path / "root" / "new.markdown"
    sync_file.write_text("new body", encoding="utf-8")
    local_notes = _CreatingLocalNotes("old")
    service = NotesScopeService(local_notes, None, folder_repository=_Folders())
    owner = build_notes_sync_runtime_owner(
        notes_scope_service=service,
        cutover_admitted=True,
        profile_process_is_sole=True,
        database_path=tmp_path / "sync.sqlite3",
        migrate_legacy=lambda: None,
        local_user_id="user-1",
        recovery_capacity_bytes=1024 * 1024,
    )

    await owner.start()

    assert local_notes.note["content"] == "new body"
    binding = store.list_bindings("root-1")
    assert len(binding) == 1
    assert binding[0].normalized_relative_path == "new.markdown"
    assert binding[0].state is NotesSyncBindingState.ACTIVE
    assert store.list_incomplete_operations() == ()
    assert owner._changed_root_ids() == ()
    (tmp_path / "root" / "ignored.json").write_text("{}", encoding="utf-8")
    assert owner._changed_root_ids() == ()
    sync_file.write_text("changed body", encoding="utf-8")
    assert owner._changed_root_ids() == ("root-1",)
    await owner.shutdown()


@pytest.mark.asyncio
async def test_clean_setup_activation_creates_note_binding_and_completed_operation(
    tmp_path: Path,
) -> None:
    from tldw_chatbook.Notes.notes_sync_runtime import (
        NotesSyncRootSetup,
        build_notes_sync_runtime_owner,
    )

    root_path = tmp_path / "new-root"
    root_path.mkdir()
    (root_path / "new.md").write_text("new body", encoding="utf-8")
    folders = _Folders()
    local_notes = _CreatingLocalNotes("old")
    owner = build_notes_sync_runtime_owner(
        notes_scope_service=NotesScopeService(
            local_notes, None, folder_repository=folders
        ),
        cutover_admitted=True,
        profile_process_is_sole=True,
        database_path=tmp_path / "sync.sqlite3",
        migrate_legacy=lambda: None,
        local_user_id="user-1",
        recovery_capacity_bytes=1024 * 1024,
    )
    await owner.start()
    review = await owner.review_setup(
        NotesSyncRootSetup(
            display_name="Research",
            canonical_path=str(root_path),
            note_scope_id="local_note",
            direction=NotesSyncDirection.BIDIRECTIONAL,
        )
    )

    result = await owner.activate_root(review.root_id, review.observation_token)

    assert result.accepted is True
    assert result.applied_count == 1
    assert local_notes.note["content"] == "new body"
    bindings = owner._store.list_bindings(review.root_id)
    assert len(bindings) == 1
    assert bindings[0].state is NotesSyncBindingState.ACTIVE
    with owner._store.transaction() as connection:
        assert (
            connection.execute("SELECT state FROM notes_sync_operations").fetchone()[0]
            == NotesSyncOperationState.COMPLETED.value
        )
    assert owner.snapshot().roots[0].status == "up_to_date"
    assert folders.created == ["Research"]
    await owner.shutdown()


@pytest.mark.asyncio
async def test_post_persist_activation_execution_failure_returns_root_recovery(
    tmp_path: Path,
) -> None:
    from tldw_chatbook.Notes.notes_sync_runtime import (
        NotesSyncRootSetup,
        build_notes_sync_runtime_owner,
    )

    root_path = tmp_path / "failing-root"
    root_path.mkdir()
    (root_path / "new.md").write_text("new body", encoding="utf-8")
    folders = _Folders()
    owner = build_notes_sync_runtime_owner(
        notes_scope_service=NotesScopeService(
            _CreatingLocalNotes("old"), None, folder_repository=folders
        ),
        cutover_admitted=True,
        profile_process_is_sole=True,
        database_path=tmp_path / "sync.sqlite3",
        migrate_legacy=lambda: None,
        local_user_id="user-1",
        recovery_capacity_bytes=1024 * 1024,
    )
    await owner.start()
    review = await owner.review_setup(
        NotesSyncRootSetup(
            display_name="Research",
            canonical_path=str(root_path),
            note_scope_id="local_note",
            direction=NotesSyncDirection.BIDIRECTIONAL,
        )
    )
    failing = _FailingExecutor()
    owner._adapter.executor_for = lambda _root, **_kwargs: failing

    result = await owner.activate_root(review.root_id, review.observation_token)

    assert (result.accepted, result.status, result.next_action) == (
        False,
        "failed",
        "review_changes",
    )
    assert owner._store.get_root(review.root_id).state is NotesSyncRootState.ACTIVE
    assert folders.created == ["Research"]
    assert folders.deleted == []
    assert owner.snapshot().roots[0].status == "failed"
    assert review.root_id not in owner._setup_reviews
    await owner.shutdown()


@pytest.mark.asyncio
async def test_failed_setup_observation_releases_provisional_lease_before_retry(
    tmp_path: Path,
) -> None:
    from tldw_chatbook.Notes.notes_sync_runtime import (
        NotesSyncRootSetup,
        NotesSyncRuntimeOwner,
    )

    store = NotesDeviceStateStore(tmp_path / "sync.sqlite3")
    store.initialize()
    store.set_setting(NotesSyncStoreSetting("cutover_marker", "notes-sync-cutover-v1"))
    root_path = tmp_path / "setup-root"
    root_path.mkdir()
    coordinator = _Coordinator()
    adapter = _MultiRootAdapter([_input(file_digest=_A, note_digest=_A)])
    original_observe = adapter.observe_root
    attempts = 0

    async def fail_then_observe(root: NotesSyncRootRecord) -> ReconciliationInput:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("private observation failure")
        return await original_observe(root)

    adapter.observe_root = fail_then_observe
    owner = NotesSyncRuntimeOwner(
        store=store,
        migrate_legacy=lambda: None,
        coordinator=coordinator,
        adapter=adapter,
        watcher_factory=lambda _schedule: _Watcher(),
        cutover_admitted=True,
        profile_process_is_sole=True,
    )
    await owner.start()
    setup = NotesSyncRootSetup(
        display_name="Research",
        canonical_path=str(root_path),
        note_scope_id="local_note",
        direction=NotesSyncDirection.BIDIRECTIONAL,
    )

    with pytest.raises(RuntimeError, match="private observation failure"):
        await owner.review_setup(setup)

    assert coordinator.events[-3:] == [
        "lease-admission-closed",
        "lease-settled",
        "lease-released",
    ]
    assert owner._leases == {}
    assert owner._admissions == {}
    assert owner._root_paths == {}

    review = await owner.review_setup(setup)
    assert review.root_id
    assert coordinator.acquire_calls == 2
    await owner.abandon_setup(review.root_id)
    await owner.shutdown()


@pytest.mark.asyncio
async def test_migrated_candidate_requires_current_review_then_activates_exact_pair(
    tmp_path: Path,
) -> None:
    from tldw_chatbook.Notes.notes_sync_runtime import build_notes_sync_runtime_owner

    store = NotesDeviceStateStore(tmp_path / "sync.sqlite3")
    store.initialize()
    store.set_setting(NotesSyncStoreSetting("cutover_marker", "notes-sync-cutover-v1"))
    root_path = tmp_path / "legacy"
    root_path.mkdir()
    file_path = root_path / "note.md"
    file_path.write_text("same", encoding="utf-8")
    with PosixNotesSyncFilesystem(root_path) as filesystem:
        file = filesystem.observe("note.md")
    root_id = "legacy-root-" + "a" * 40
    binding_id = "legacy-binding-" + "b" * 40
    store.create_root(
        NotesSyncRootRecord(
            root_id=root_id,
            note_scope_id="local_note",
            logical_folder_id=None,
            canonical_path=str(root_path),
            direction=NotesSyncDirection.BIDIRECTIONAL,
            state=NotesSyncRootState.PAUSED,
            last_status_code="migration_review_required",
        )
    )
    store.create_binding(
        NotesSyncBindingRecord(
            binding_id=binding_id,
            root_id=root_id,
            note_scope_id="local_note",
            note_id="note-1",
            normalized_relative_path="note.md",
            stable_identity_digest=NotesSyncExecutor.stable_identity_digest(file),
            state=NotesSyncBindingState.CANDIDATE,
            serialization=file.observation.serialization,
            content_digest=file.observation.content_digest,
            note_version=1,
        )
    )
    folders = _Folders()
    owner = build_notes_sync_runtime_owner(
        notes_scope_service=NotesScopeService(
            _LocalNotes("same"), None, folder_repository=folders
        ),
        cutover_admitted=True,
        profile_process_is_sole=True,
        database_path=tmp_path / "sync.sqlite3",
        migrate_legacy=lambda: None,
        local_user_id="user-1",
        recovery_capacity_bytes=1024 * 1024,
    )
    await owner.start()
    assert owner.snapshot().roots[0].next_action == "review_migration"

    review = await owner.check_root(root_id)
    assert owner._adapter._bundles == {}
    assert review.attention == ()
    assert [action.kind for action in review.safe_actions] == [
        NotesSyncActionKind.NO_CHANGE
    ]
    result = await owner.activate_root(root_id, review.observation_token)

    assert result.accepted is True
    assert store.get_root(root_id).state is NotesSyncRootState.ACTIVE
    assert store.get_root(root_id).logical_folder_id == "folder-real"
    assert store.get_binding(binding_id).state is NotesSyncBindingState.ACTIVE
    assert len(folders.created) == 1
    assert folders.created[0].startswith("Migrated notes ")
    assert owner._adapter._bundles == {}
    await owner.shutdown()


@pytest.mark.asyncio
async def test_migration_activation_has_no_read_after_commit_folder_rollback_window(
    tmp_path: Path,
) -> None:
    store = _PostCommitReadFailStore(tmp_path / "sync.sqlite3")
    store.initialize()
    store.set_setting(NotesSyncStoreSetting("cutover_marker", "notes-sync-cutover-v1"))
    root_path = tmp_path / "legacy-post-commit"
    root_path.mkdir()
    root_id = "legacy-root-" + "c" * 40
    store.create_root(
        NotesSyncRootRecord(
            root_id=root_id,
            note_scope_id="local_note",
            logical_folder_id=None,
            canonical_path=str(root_path),
            direction=NotesSyncDirection.BIDIRECTIONAL,
            state=NotesSyncRootState.PAUSED,
            last_status_code="migration_review_required",
        )
    )
    store.create_binding(
        NotesSyncBindingRecord(
            binding_id="binding-1",
            root_id=root_id,
            note_scope_id="local_note",
            note_id="note-1",
            normalized_relative_path="note.md",
            stable_identity_digest=_C,
            state=NotesSyncBindingState.CANDIDATE,
            serialization=NotesSyncSerializationProfile(False, "lf", False, 0o644),
            content_digest=_A,
            note_version=1,
        )
    )
    adapter = _MultiRootAdapter([_input(file_digest=_A, note_digest=_A)])
    owner, _, _ = _owner(store=store, admitted=True, adapter=adapter)
    await owner.start()
    review = await owner.check_root(root_id)

    result = await owner.activate_root(root_id, review.observation_token)

    assert result.accepted is True
    assert (
        NotesDeviceStateStore.get_root(store, root_id).state
        is NotesSyncRootState.ACTIVE
    )
    assert adapter.rolled_back_folders == []
    await owner.shutdown()


@pytest.mark.asyncio
async def test_activation_recovery_record_has_no_postcommit_read_and_reopens_blocked(
    tmp_path: Path,
) -> None:
    store = _RecoveryPostCommitReadFailStore(tmp_path / "sync.sqlite3")
    store.initialize()
    store.set_setting(NotesSyncStoreSetting("cutover_marker", "notes-sync-cutover-v1"))
    root_path = tmp_path / "recovery-root"
    root_path.mkdir()
    store.create_root(
        NotesSyncRootRecord(
            root_id="root-recovery",
            note_scope_id="local_note",
            logical_folder_id=None,
            canonical_path=str(root_path),
            direction=NotesSyncDirection.BIDIRECTIONAL,
            state=NotesSyncRootState.PENDING,
        )
    )

    committed = store.record_root_activation_recovery(
        "root-recovery", "folder-recovery"
    )

    assert committed.state is NotesSyncRootState.PAUSED
    assert committed.last_status_code == "activation_recovery_required"
    reopened = NotesDeviceStateStore(tmp_path / "sync.sqlite3")
    owner, _, _ = _owner(
        store=reopened,
        admitted=True,
        adapter=_Adapter([_input(file_digest=_A, note_digest=_A)]),
    )
    await owner.start()
    root = owner.snapshot().roots[0]
    assert (root.status, root.next_action) == (
        "needs_attention",
        "review_settings",
    )
    assert reopened.get_root("root-recovery").last_status_code == (
        "activation_recovery_required"
    )
    await owner.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ("coordinator", "roots", "recovery", "resume"))
async def test_startup_inventory_failure_never_publishes_active_or_opens_admission(
    tmp_path: Path, stage: str
) -> None:
    store = _StartupInventoryFailStore(tmp_path / "sync.sqlite3", stage)
    store.initialize()
    store.set_setting(NotesSyncStoreSetting("cutover_marker", "notes-sync-cutover-v1"))

    def coordinator_source():
        if stage == "coordinator":
            raise RuntimeError("private coordinator failure")
        return _Coordinator()

    owner, _, _ = _owner(
        store=store,
        admitted=True,
        adapter=_Adapter([_input(file_digest=_A, note_digest=_A)]),
        coordinator=coordinator_source,
    )
    if stage == "resume":

        async def fail_resume(*_args: object) -> None:
            raise RuntimeError("private resume failure")

        owner._resume_incomplete = fail_resume

    await owner.start()

    assert (owner.snapshot().status, owner.snapshot().next_action) == (
        "failed",
        "review_settings",
    )
    assert owner._admission_open is False
    assert owner.schedule_hint("root-1") is None


@pytest.mark.asyncio
async def test_shutdown_settles_activation_and_partial_cardinality_is_not_success(
    tmp_path: Path,
) -> None:
    from tldw_chatbook.Notes.notes_sync_runtime import NotesSyncRootSetup

    store = NotesDeviceStateStore(tmp_path / "sync.sqlite3")
    store.initialize()
    store.set_setting(NotesSyncStoreSetting("cutover_marker", "notes-sync-cutover-v1"))
    root_path = tmp_path / "activation-shutdown"
    root_path.mkdir()

    class _SetupTwoActionAdapter(_Adapter):
        async def observe_root(self, root: NotesSyncRootRecord) -> ReconciliationInput:
            self.observe_calls += 1
            return replace(_two_action_input(), root_id=root.root_id)

    adapter = _SetupTwoActionAdapter([_two_action_input()])
    executor = _BlockingExecutor()
    adapter.executor = executor
    owner, coordinator, _ = _owner(
        store=store,
        admitted=True,
        adapter=adapter,
        coordinator=_Coordinator(),
    )
    await owner.start()
    review = await owner.review_setup(
        NotesSyncRootSetup(
            display_name="Research",
            canonical_path=str(root_path),
            note_scope_id="local_note",
            direction=NotesSyncDirection.BIDIRECTIONAL,
        )
    )
    activation = asyncio.create_task(
        owner.activate_root(review.root_id, review.observation_token)
    )
    await executor.started.wait()

    shutdown = asyncio.create_task(owner.shutdown())
    await asyncio.sleep(0)
    assert not shutdown.done()
    assert coordinator.events == []
    executor.release.set()
    result = await activation
    await shutdown

    assert (result.accepted, result.status, result.next_action) == (
        False,
        "partial",
        "review_changes",
    )
    assert result.applied_count == 1
    assert len(executor.executed) == 1
    assert coordinator.events[-1] == "lease-released"


@pytest.mark.asyncio
async def test_migration_activation_adopts_candidates_and_executes_extra_file(
    tmp_path: Path,
) -> None:
    store = NotesDeviceStateStore(tmp_path / "sync.sqlite3")
    store.initialize()
    store.set_setting(NotesSyncStoreSetting("cutover_marker", "notes-sync-cutover-v1"))
    root_path = tmp_path / "legacy-extra"
    root_path.mkdir()
    root_id = "legacy-root-" + "d" * 40
    store.create_root(
        NotesSyncRootRecord(
            root_id=root_id,
            note_scope_id="local_note",
            logical_folder_id=None,
            canonical_path=str(root_path),
            direction=NotesSyncDirection.BIDIRECTIONAL,
            state=NotesSyncRootState.PAUSED,
            last_status_code="migration_review_required",
        )
    )
    store.create_binding(
        NotesSyncBindingRecord(
            binding_id="binding-1",
            root_id=root_id,
            note_scope_id="local_note",
            note_id="note-1",
            normalized_relative_path="note.md",
            stable_identity_digest=_C,
            state=NotesSyncBindingState.CANDIDATE,
            serialization=NotesSyncSerializationProfile(False, "lf", False, 0o644),
            content_digest=_A,
            note_version=1,
        )
    )
    base = _input(file_digest=_A, note_digest=_A)
    extra = replace(
        base.bindings[0],
        binding_id="binding-extra",
        baseline_file_digest=_A,
        baseline_note_digest=_A,
        baseline_relative_path="extra.md",
        baseline_identity_digest=_C,
        relative_path="extra.md",
        file_digest=_B,
        note_digest=None,
        file_identity_digest=_C,
        note_id="note-extra",
        bound=False,
    )
    adapter = _Adapter(
        [replace(base, root_id=root_id, bindings=(base.bindings[0], extra))]
    )
    owner, _, _ = _owner(store=store, admitted=True, adapter=adapter)
    await owner.start()
    review = await owner.check_root(root_id)

    result = await owner.activate_root(root_id, review.observation_token)

    assert result.accepted is True
    assert store.get_binding("binding-1").state is NotesSyncBindingState.ACTIVE
    assert [action.binding_id for action in adapter.executor.executed] == [
        "binding-extra"
    ]
    await owner.shutdown()


@pytest.mark.asyncio
async def test_migrated_candidate_refuses_stale_two_sided_drift(
    tmp_path: Path,
) -> None:
    from tldw_chatbook.Notes.notes_sync_runtime import build_notes_sync_runtime_owner

    store = NotesDeviceStateStore(tmp_path / "sync.sqlite3")
    store.initialize()
    store.set_setting(NotesSyncStoreSetting("cutover_marker", "notes-sync-cutover-v1"))
    root_path = tmp_path / "legacy"
    root_path.mkdir()
    file_path = root_path / "note.md"
    file_path.write_text("same", encoding="utf-8")
    with PosixNotesSyncFilesystem(root_path) as filesystem:
        file = filesystem.observe("note.md")
    root_id = "legacy-root-" + "a" * 40
    binding_id = "legacy-binding-" + "b" * 40
    store.create_root(
        NotesSyncRootRecord(
            root_id=root_id,
            note_scope_id="local_note",
            logical_folder_id=None,
            canonical_path=str(root_path),
            direction=NotesSyncDirection.BIDIRECTIONAL,
            state=NotesSyncRootState.PAUSED,
            last_status_code="migration_review_required",
        )
    )
    store.create_binding(
        NotesSyncBindingRecord(
            binding_id=binding_id,
            root_id=root_id,
            note_scope_id="local_note",
            note_id="note-1",
            normalized_relative_path="note.md",
            stable_identity_digest=NotesSyncExecutor.stable_identity_digest(file),
            state=NotesSyncBindingState.CANDIDATE,
            serialization=file.observation.serialization,
            content_digest=file.observation.content_digest,
            note_version=1,
        )
    )
    local_notes = _LocalNotes("same")
    folders = _Folders()
    owner = build_notes_sync_runtime_owner(
        notes_scope_service=NotesScopeService(
            local_notes, None, folder_repository=folders
        ),
        cutover_admitted=True,
        profile_process_is_sole=True,
        database_path=tmp_path / "sync.sqlite3",
        migrate_legacy=lambda: None,
        local_user_id="user-1",
        recovery_capacity_bytes=1024 * 1024,
    )
    await owner.start()
    review = await owner.check_root(root_id)
    file_path.write_text("file changed", encoding="utf-8")
    local_notes.note["content"] = "note changed"
    local_notes.note["version"] = 2

    with pytest.raises(ValueError, match="stale_review"):
        await owner.activate_root(root_id, review.observation_token)

    assert store.get_root(root_id).state is NotesSyncRootState.PAUSED
    assert store.get_binding(binding_id).state is NotesSyncBindingState.CANDIDATE
    assert folders.created == []
    await owner.shutdown()


@pytest.mark.asyncio
async def test_two_migrated_candidates_activate_with_distinct_safe_folder_names(
    tmp_path: Path,
) -> None:
    from tldw_chatbook.Notes.note_folder_models import NoteFolder
    from tldw_chatbook.Notes.notes_sync_runtime import build_notes_sync_runtime_owner

    class _DistinctFolders(_Folders):
        def create_folder(self, *, name: str, parent_id: str | None):
            if name in self.created:
                raise ValueError("folder name already exists")
            self.created.append(name)
            ordinal = len(self.created)
            return NoteFolder(
                folder_id=f"folder-real-{ordinal}",
                parent_id=parent_id,
                name=name,
                path=name,
                normalized_path=name.casefold(),
                version=1,
                deleted=False,
            )

    class _TwoNotes(_LocalNotes):
        def get_note_by_id(self, _user_id: str, note_id: str):
            if note_id not in {"note-1", "note-2"}:
                return None
            return {
                "id": note_id,
                "title": "Note",
                "content": "same",
                "version": 1,
                "deleted": False,
            }

    store = NotesDeviceStateStore(tmp_path / "sync.sqlite3")
    store.initialize()
    store.set_setting(NotesSyncStoreSetting("cutover_marker", "notes-sync-cutover-v1"))
    root_ids: list[str] = []
    for ordinal, character in enumerate(("a", "b"), start=1):
        root_path = tmp_path / f"legacy-{ordinal}"
        root_path.mkdir()
        (root_path / "note.md").write_text("same", encoding="utf-8")
        with PosixNotesSyncFilesystem(root_path) as filesystem:
            file = filesystem.observe("note.md")
        root_id = "legacy-root-" + character * 40
        root_ids.append(root_id)
        store.create_root(
            NotesSyncRootRecord(
                root_id=root_id,
                note_scope_id="local_note",
                logical_folder_id=None,
                canonical_path=str(root_path),
                direction=NotesSyncDirection.BIDIRECTIONAL,
                state=NotesSyncRootState.PAUSED,
                last_status_code="migration_review_required",
            )
        )
        store.create_binding(
            NotesSyncBindingRecord(
                binding_id=f"legacy-binding-{character * 40}",
                root_id=root_id,
                note_scope_id="local_note",
                note_id=f"note-{ordinal}",
                normalized_relative_path="note.md",
                stable_identity_digest=NotesSyncExecutor.stable_identity_digest(file),
                state=NotesSyncBindingState.CANDIDATE,
                serialization=file.observation.serialization,
                content_digest=file.observation.content_digest,
                note_version=1,
            )
        )
    folders = _DistinctFolders()
    owner = build_notes_sync_runtime_owner(
        notes_scope_service=NotesScopeService(
            _TwoNotes("same"), None, folder_repository=folders
        ),
        cutover_admitted=True,
        profile_process_is_sole=True,
        database_path=tmp_path / "sync.sqlite3",
        migrate_legacy=lambda: None,
        local_user_id="user-1",
        recovery_capacity_bytes=1024 * 1024,
    )
    await owner.start()

    for root_id in root_ids:
        review = await owner.check_root(root_id)
        result = await owner.activate_root(root_id, review.observation_token)
        assert result.accepted is True

    assert len(folders.created) == 2
    assert len(set(folders.created)) == 2
    assert all(name.startswith("Migrated notes ") for name in folders.created)
    await owner.shutdown()


@pytest.mark.asyncio
async def test_post_cutover_startup_claims_reconciles_and_starts_watcher(
    tmp_path: Path,
) -> None:
    adapter = _Adapter([_input(file_digest=_A, note_digest=_A)])
    owner, coordinator, watcher = _owner(
        store=_store(tmp_path), admitted=True, adapter=adapter
    )

    await owner.start()
    await watcher.started.wait()

    root = owner.snapshot().roots[0]
    assert coordinator.acquire_calls == 1
    assert adapter.observe_calls == 1
    assert root.status == "up_to_date"
    assert root.next_action == "sync_now"
    await owner.shutdown()


@pytest.mark.asyncio
async def test_manual_admission_stays_closed_until_startup_inventory_finishes(
    tmp_path: Path,
) -> None:
    _store(tmp_path)
    store = _BlockingRootListStore(tmp_path / "sync.sqlite3")
    owner, coordinator, _ = _owner(
        store=store,
        admitted=True,
        adapter=_Adapter([_input(file_digest=_A, note_digest=_A)]),
    )
    start = asyncio.create_task(owner.start())
    await asyncio.to_thread(store.list_started.wait)

    with pytest.raises(RuntimeError, match="cutover"):
        await owner.check_root("root-1")
    with pytest.raises(RuntimeError, match="cutover"):
        await owner.apply_reviewed("root-1", _A, ())
    assert coordinator.acquire_calls == 0

    store.list_release.set()
    await start
    assert coordinator.acquire_calls == 1
    await owner.shutdown()


@pytest.mark.asyncio
async def test_each_root_lease_revalidates_every_other_registered_root(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    other = tmp_path / "other-root"
    other.mkdir()
    store.create_root(
        NotesSyncRootRecord(
            root_id="root-2",
            note_scope_id="local_note",
            logical_folder_id=None,
            canonical_path=str(other),
            direction=NotesSyncDirection.BIDIRECTIONAL,
            state=NotesSyncRootState.PAUSED,
        )
    )
    coordinator = _Coordinator()
    owner, _, _ = _owner(
        store=store,
        admitted=True,
        adapter=_Adapter([_input(file_digest=_A, note_digest=_A)]),
        coordinator=coordinator,
    )

    await owner.start()

    assert coordinator.validations == [
        {"lasting_roots": (str(other),), "file_notes_binding": None}
    ]
    await owner.shutdown()


def _pending_operation(store: NotesDeviceStateStore) -> None:
    store.create_operation(
        NotesSyncOperationRecord(
            operation_id="operation-1",
            root_id="root-1",
            binding_id=None,
            kind="create_file",
            state=NotesSyncOperationState.PENDING,
            reason_code=None,
            observation_token="review-1",
            expected_note_version=None,
            expected_file_digest=None,
        )
    )


@pytest.mark.asyncio
async def test_startup_resumes_incomplete_work_under_the_claimed_root_lease(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    _pending_operation(store)
    adapter = _Adapter([_input(file_digest=_A, note_digest=_A)])
    coordinator = _Coordinator()
    owner, _, _ = _owner(
        store=store,
        admitted=True,
        adapter=adapter,
        coordinator=coordinator,
    )

    await owner.start()

    assert coordinator.acquire_calls == 1
    assert adapter.executor.reconstructed == ["operation-1"]
    assert adapter.executor.resumed == ["operation-1"]
    await owner.shutdown()


@pytest.mark.asyncio
async def test_startup_attention_journal_blocks_reconciliation_and_mutation(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    _pending_operation(store)
    store.mark_operation_attention("operation-1", "recovery_review_required")
    adapter = _Adapter([_input()])
    owner, _, _ = _owner(store=store, admitted=True, adapter=adapter)

    await owner.start()

    assert adapter.observe_calls == 0
    assert adapter.executor.executed == []
    assert adapter.executor.reconstructed == []
    root = owner.snapshot().roots[0]
    assert (root.status, root.next_action) == (
        "needs_attention",
        "review_changes",
    )
    await owner.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reason_code", "expected_action"),
    [
        ("recovery_review_required", "review_changes"),
        ("replacement_cleanup_pending", "resolve_cleanup"),
    ],
)
async def test_incomplete_journal_overrides_durable_attention_after_owner_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reason_code: str,
    expected_action: str,
) -> None:
    store = _store(tmp_path)
    _pending_operation(store)
    store.mark_operation_attention("operation-1", reason_code)
    store.update_root_status("root-1", "needs_attention")
    coordinator = _Coordinator()
    events: list[str] = []
    update_root_status = store.update_root_status
    try_acquire = coordinator.try_acquire

    def recording_update_root_status(
        root_id: str, status_code: str
    ) -> NotesSyncRootRecord:
        events.append("status")
        return update_root_status(root_id, status_code)

    def recording_try_acquire(path: str, **kwargs: object) -> _Admission:
        events.append("acquire")
        return try_acquire(path, **kwargs)

    monkeypatch.setattr(store, "update_root_status", recording_update_root_status)
    monkeypatch.setattr(coordinator, "try_acquire", recording_try_acquire)
    adapter = _Adapter([_input()])
    owner, _, _ = _owner(
        store=store,
        admitted=True,
        adapter=adapter,
        coordinator=coordinator,
    )

    await owner.start()

    root = owner.snapshot().roots[0]
    assert events[0] == "acquire"
    assert (root.status, root.next_action, root.action_id) == (
        "needs_attention",
        expected_action,
        "operation-1",
    )
    assert adapter.observe_calls == 0
    assert adapter.executor.reconstructed == []
    await owner.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "partial"])
async def test_pending_journal_does_not_resume_past_a_durable_failure(
    tmp_path: Path,
    status: str,
) -> None:
    store = _store(tmp_path)
    _pending_operation(store)
    store.update_root_status("root-1", status)
    coordinator = _Coordinator()
    adapter = _Adapter([_input()])
    owner, _, _ = _owner(
        store=store,
        admitted=True,
        adapter=adapter,
        coordinator=coordinator,
    )

    await owner.start()

    root = owner.snapshot().roots[0]
    assert coordinator.acquire_calls == 1
    assert (root.status, root.next_action, root.action_id) == (
        status,
        "review_changes",
        "operation-1",
    )
    assert adapter.observe_calls == 0
    assert adapter.executor.reconstructed == []
    assert adapter.executor.resumed == []
    await owner.shutdown()


@pytest.mark.asyncio
async def test_passive_root_does_not_classify_its_incomplete_journal(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    _pending_operation(store)
    store.mark_operation_attention("operation-1", "recovery_review_required")
    adapter = _Adapter([_input()])
    owner, _, _ = _owner(
        store=store,
        admitted=True,
        adapter=adapter,
        coordinator=_Coordinator(RootAdmissionState.PASSIVE),
    )

    await owner.start()

    root = owner.snapshot().roots[0]
    assert (root.status, root.next_action, root.action_id) == (
        "passive",
        "open_active_process",
        None,
    )
    assert adapter.observe_calls == 0
    await owner.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "partial"])
async def test_durable_blocked_status_requires_an_explicit_fresh_check(
    tmp_path: Path,
    status: str,
) -> None:
    store = _store(tmp_path)
    store.update_root_status("root-1", status)
    adapter = _Adapter([_input()])
    owner, _, _ = _owner(store=store, admitted=True, adapter=adapter)

    await owner.start()

    root = owner.snapshot().roots[0]
    assert (root.status, root.next_action) == (status, "review_changes")
    assert adapter.observe_calls == 0
    assert adapter.executor.executed == []
    assert owner.schedule_hint("root-1") is None

    await owner.check_root("root-1")

    assert adapter.observe_calls == 1
    assert adapter.executor.executed == []
    assert store.get_root("root-1").last_status_code == "changes_available"
    await owner.shutdown()


@pytest.mark.asyncio
async def test_failed_automatic_action_blocks_retry_and_publishes_next_action(
    tmp_path: Path,
) -> None:
    adapter = _Adapter([_input()])
    adapter.executor = _FailingExecutor()
    owner, _, _ = _owner(store=_store(tmp_path), admitted=True, adapter=adapter)

    await owner.start()

    root = owner.snapshot().roots[0]
    assert (root.status, root.next_action) == ("failed", "review_changes")
    assert owner.schedule_hint("root-1") is None
    assert len(adapter.executor.executed) == 1
    await owner.shutdown()


@pytest.mark.asyncio
async def test_lost_lease_after_first_action_prevents_the_second_action(
    tmp_path: Path,
) -> None:
    coordinator = _Coordinator()

    def invalidate() -> None:
        assert coordinator.last_admission is not None
        assert coordinator.last_admission.lease is not None
        coordinator.last_admission.lease.authoritative = False

    adapter = _Adapter([_two_action_input()])
    adapter.executor = _InvalidatingExecutor(invalidate)
    owner, _, _ = _owner(
        store=_store(tmp_path),
        admitted=True,
        adapter=adapter,
        coordinator=coordinator,
    )

    await owner.start()

    assert len(adapter.executor.executed) == 1
    root = owner.snapshot().roots[0]
    assert (root.status, root.next_action) == ("failed", "review_changes")
    await owner.shutdown()


@pytest.mark.asyncio
async def test_lost_lease_after_reconstruct_prevents_incomplete_resume(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    _pending_operation(store)
    coordinator = _Coordinator()

    def invalidate() -> None:
        assert coordinator.last_admission is not None
        assert coordinator.last_admission.lease is not None
        coordinator.last_admission.lease.authoritative = False

    adapter = _Adapter([_input(file_digest=_A, note_digest=_A)])
    adapter.executor = _InvalidatingReconstructExecutor(invalidate)
    owner, _, _ = _owner(
        store=store,
        admitted=True,
        adapter=adapter,
        coordinator=coordinator,
    )

    await owner.start()

    assert adapter.executor.reconstructed == ["operation-1"]
    assert adapter.executor.resumed == []
    root = owner.snapshot().roots[0]
    assert (root.status, root.next_action) == ("failed", "review_changes")
    await owner.shutdown()


@pytest.mark.asyncio
async def test_failed_hint_observation_blocks_automatic_retry(tmp_path: Path) -> None:
    adapter = _FailingObservationAdapter([_input(file_digest=_A, note_digest=_A)])
    owner, _, _ = _owner(store=_store(tmp_path), admitted=True, adapter=adapter)
    await owner.start()

    owner.schedule_hint("root-1")
    await owner.settle()

    root = owner.snapshot().roots[0]
    assert (root.status, root.next_action) == ("failed", "review_changes")
    assert owner.schedule_hint("root-1") is None
    await owner.shutdown()


@pytest.mark.asyncio
async def test_hint_with_lost_root_authority_publishes_offline(tmp_path: Path) -> None:
    coordinator = _Coordinator()
    owner, _, _ = _owner(
        store=_store(tmp_path),
        admitted=True,
        adapter=_Adapter([_input(file_digest=_A, note_digest=_A)]),
        coordinator=coordinator,
    )
    await owner.start()
    assert coordinator.last_admission is not None
    assert coordinator.last_admission.lease is not None
    coordinator.last_admission.lease.authoritative = False

    owner.schedule_hint("root-1")
    await owner.settle()

    root = owner.snapshot().roots[0]
    assert (root.status, root.next_action) == ("offline", "reconnect_folder")
    await owner.shutdown()


@pytest.mark.asyncio
async def test_observed_direction_must_match_the_durable_root_direction(
    tmp_path: Path,
) -> None:
    adapter = _Adapter([_input(direction=NotesSyncDirection.FOLDER_TO_NOTES)])
    owner, _, _ = _owner(store=_store(tmp_path), admitted=True, adapter=adapter)

    await owner.start()

    assert adapter.executor.executed == []
    root = owner.snapshot().roots[0]
    assert (root.status, root.next_action) == ("failed", "review_changes")
    await owner.shutdown()


@pytest.mark.asyncio
async def test_automatic_sync_never_applies_a_direction_override(
    tmp_path: Path,
) -> None:
    adapter = _OverrideAdapter([_input()])
    owner, _, _ = _owner(store=_store(tmp_path), admitted=True, adapter=adapter)

    await owner.start()

    assert adapter.executor.executed == []
    root = owner.snapshot().roots[0]
    assert (root.status, root.next_action) == ("needs_attention", "review_changes")
    await owner.shutdown()


@pytest.mark.asyncio
async def test_cleanup_pending_publishes_an_explicit_cleanup_action(
    tmp_path: Path,
) -> None:
    adapter = _Adapter([_input()])
    adapter.executor = _PartialCleanupExecutor()
    owner, _, _ = _owner(store=_store(tmp_path), admitted=True, adapter=adapter)

    await owner.start()

    root = owner.snapshot().roots[0]
    assert (root.status, root.next_action) == ("partial", "resolve_cleanup")
    assert owner.schedule_hint("root-1") is None
    await owner.shutdown()


@pytest.mark.asyncio
async def test_explicit_cleanup_action_runs_under_the_root_lease(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    _pending_operation(store)
    store.mark_operation_attention("operation-1", "replacement_cleanup_pending")
    adapter = _Adapter([_input(file_digest=_A, note_digest=_A)])
    adapter.executor = _PartialCleanupExecutor()
    owner, _, _ = _owner(store=store, admitted=True, adapter=adapter)
    await owner.start()

    root = owner.snapshot().roots[0]
    assert (root.next_action, root.action_id) == (
        "resolve_cleanup",
        "operation-1",
    )
    await owner.resolve_cleanup("root-1", root.action_id)

    assert adapter.executor.cleanup == ["operation-1"]
    assert owner.snapshot().roots[0].status == "up_to_date"
    await owner.shutdown()


@pytest.mark.asyncio
async def test_cleanup_attention_retains_the_operation_for_review(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    _pending_operation(store)
    store.mark_operation_attention("operation-1", "replacement_cleanup_pending")
    adapter = _Adapter([_input(file_digest=_A, note_digest=_A)])
    adapter.executor = _AttentionCleanupExecutor()
    owner, _, _ = _owner(store=store, admitted=True, adapter=adapter)
    await owner.start()

    await owner.resolve_cleanup("root-1", "operation-1")

    root = owner.snapshot().roots[0]
    assert (root.status, root.next_action, root.action_id) == (
        "needs_attention",
        "review_changes",
        "operation-1",
    )
    await owner.shutdown()


@pytest.mark.asyncio
async def test_manual_apply_rejects_a_stale_review_before_executor_call(
    tmp_path: Path,
) -> None:
    adapter = _Adapter(
        [_input(generation=1), _input(generation=1), _input(generation=2)]
    )
    owner, _, _ = _owner(store=_store(tmp_path), admitted=True, adapter=adapter)
    await owner.start()
    adapter.executor.executed.clear()
    reviewed = await owner.check_root("root-1")

    with pytest.raises(ValueError, match="stale_review"):
        await owner.apply_reviewed(
            "root-1",
            reviewed.observation_token,
            tuple(action.action_id for action in reviewed.safe_actions),
        )

    assert adapter.executor.executed == []
    await owner.shutdown()


@pytest.mark.asyncio
async def test_manual_apply_rechecks_that_the_root_is_still_active(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    adapter = _Adapter(
        [
            _input(file_digest=_A, note_digest=_A),
            _input(),
            _input(),
        ]
    )
    owner, _, _ = _owner(store=store, admitted=True, adapter=adapter)
    await owner.start()
    reviewed = await owner.check_root("root-1")
    store.transition_root("root-1", NotesSyncRootState.PAUSED)

    with pytest.raises(RuntimeError, match="not_active"):
        await owner.apply_reviewed(
            "root-1",
            reviewed.observation_token,
            tuple(action.action_id for action in reviewed.safe_actions),
        )

    assert adapter.executor.executed == []
    await owner.shutdown()


@pytest.mark.asyncio
async def test_manual_empty_apply_keeps_content_conflict_attention(
    tmp_path: Path,
) -> None:
    adapter = _Adapter([_input(file_digest=_B, note_digest=_C)])
    owner, _, _ = _owner(store=_store(tmp_path), admitted=True, adapter=adapter)
    await owner.start()
    reviewed = await owner.check_root("root-1")

    result = await owner.apply_reviewed("root-1", reviewed.observation_token, ())

    root = owner.snapshot().roots[0]
    assert result.unresolved_conflicts == 1
    assert result.attention_remains is True
    assert result.fresh_plan == reviewed
    assert (root.status, root.next_action) == ("needs_attention", "review_changes")
    assert adapter.executor.executed == []
    await owner.shutdown()


@pytest.mark.asyncio
async def test_manual_apply_accepts_reviewed_no_change_rows_without_executing_them(
    tmp_path: Path,
) -> None:
    unchanged = _input(file_digest=_A, note_digest=_A)
    changed = replace(
        unchanged.bindings[0],
        binding_id="binding-2",
        note_id="note-2",
        relative_path="second.md",
        baseline_relative_path="second.md",
        file_digest=_B,
    )
    mixed = replace(unchanged, bindings=(*unchanged.bindings, changed))
    adapter = _Adapter([mixed, mixed, mixed, mixed])
    owner, _, _ = _owner(store=_store(tmp_path), admitted=True, adapter=adapter)
    await owner.start()
    adapter.executor.executed.clear()
    reviewed = await owner.check_root("root-1")
    assert [action.kind for action in reviewed.safe_actions] == [
        NotesSyncActionKind.NO_CHANGE,
        NotesSyncActionKind.UPDATE_NOTE,
    ]

    result = await owner.apply_reviewed(
        "root-1",
        reviewed.observation_token,
        tuple(action.action_id for action in reviewed.safe_actions),
    )

    assert result.safe_completed == 1
    assert result.attention_remains is False
    assert [action.kind for action in adapter.executor.executed] == [
        NotesSyncActionKind.UPDATE_NOTE
    ]
    await owner.shutdown()


@pytest.mark.asyncio
async def test_manual_apply_rejects_unknown_safe_action_ids(tmp_path: Path) -> None:
    observed = _input()
    adapter = _Adapter([observed] * 3)
    owner, _, _ = _owner(store=_store(tmp_path), admitted=True, adapter=adapter)
    await owner.start()
    adapter.executor.executed.clear()
    reviewed = await owner.check_root("root-1")

    with pytest.raises(ValueError, match="reviewed_action_mismatch"):
        await owner.apply_reviewed(
            "root-1",
            reviewed.observation_token,
            ("unknown-action",),
        )

    assert adapter.executor.executed == []
    await owner.shutdown()


@pytest.mark.asyncio
async def test_automatic_hint_applies_only_a_safe_direction_authorized_action(
    tmp_path: Path,
) -> None:
    adapter = _Adapter([_input(), _input()])
    store = _store(tmp_path)
    owner, _, _ = _owner(store=store, admitted=True, adapter=adapter)
    await owner.start()
    adapter.executor.executed.clear()

    owner.schedule_hint("root-1")
    owner.schedule_hint("root-1")
    await owner.settle()

    assert len(adapter.executor.executed) == 1
    assert owner.snapshot().roots[0].status == "up_to_date"
    await owner.shutdown()


@pytest.mark.asyncio
async def test_hint_during_reconcile_queues_exactly_one_trailing_check(
    tmp_path: Path,
) -> None:
    adapter = _Adapter(
        [
            _input(file_digest=_A, note_digest=_A),
            _input(),
            _input(file_digest=_A, note_digest=_A),
        ]
    )
    executor = _BlockingExecutor()
    adapter.executor = executor
    owner, _, _ = _owner(store=_store(tmp_path), admitted=True, adapter=adapter)
    await owner.start()

    owner.schedule_hint("root-1")
    await executor.started.wait()
    owner.schedule_hint("root-1")
    owner.schedule_hint("root-1")
    executor.release.set()
    await owner.settle()

    assert adapter.observe_calls == 3
    assert len(executor.executed) == 1
    await owner.shutdown()


@pytest.mark.asyncio
async def test_automatic_sync_blocks_a_managed_placement_effect(
    tmp_path: Path,
) -> None:
    adapter = _Adapter(
        [_input(file_digest=_A, note_digest=_A, relative_path="moved.md")]
    )
    owner, _, _ = _owner(store=_store(tmp_path), admitted=True, adapter=adapter)

    await owner.start()

    assert adapter.executor.executed == []
    root = owner.snapshot().roots[0]
    assert (root.status, root.next_action) == (
        "needs_attention",
        "review_changes",
    )
    assert owner.schedule_hint("root-1") is None
    await owner.shutdown()


@pytest.mark.asyncio
async def test_pause_closes_root_admission_and_releases_its_lease(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    coordinator = _Coordinator()
    adapter = _Adapter([_input(file_digest=_A, note_digest=_A)])
    owner, _, _ = _owner(
        store=store,
        admitted=True,
        adapter=adapter,
        coordinator=coordinator,
    )
    await owner.start()

    result = await owner.pause_root("root-1")

    assert result.accepted
    assert store.get_root("root-1").state is NotesSyncRootState.PAUSED
    assert owner.schedule_hint("root-1") is None
    assert coordinator.events == [
        "lease-admission-closed",
        "lease-settled",
        "lease-released",
    ]
    await owner.shutdown()


class _GatedTransactionStore(NotesDeviceStateStore):
    """Hold one armed transaction open with its connection checked out.

    task-21101 review round: models a pool thread parked inside
    ``transaction()`` while shutdown runs, so the store's held connection is
    exactly mid-transaction when ``close()`` could fire.
    """

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.mid_transaction = threading.Event()
        self.release = threading.Event()
        self._armed = False

    def arm(self) -> None:
        self._armed = True

    @contextmanager
    def transaction(self, *, immediate: bool = False):
        with super().transaction(immediate=immediate) as connection:
            if self._armed:
                self._armed = False
                self.mid_transaction.set()
                assert self.release.wait(timeout=5)
            yield connection


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["pause_root", "resume_root"])
async def test_shutdown_settles_in_flight_pause_and_resume_before_store_close(
    tmp_path: Path,
    operation: str,
) -> None:
    """Reviewer probe for task-21101: pause/resume mid-transaction at shutdown.

    pause_root/resume_root touch the held-connection store but were invisible
    to settle(), so _shutdown_once could close the store while their pool
    thread held a checked-out connection (ProgrammingError on a closed
    database). Shutdown must now wait for them.
    """

    store = _GatedTransactionStore(tmp_path / "sync.sqlite3")
    store.initialize()
    store.create_root(
        NotesSyncRootRecord(
            root_id="root-1",
            note_scope_id="local_note",
            logical_folder_id="folder-1",
            canonical_path=str(tmp_path / "root"),
            direction=NotesSyncDirection.BIDIRECTIONAL,
            state=NotesSyncRootState.ACTIVE,
        )
    )
    store.set_setting(NotesSyncStoreSetting("cutover_marker", "notes-sync-cutover-v1"))
    (tmp_path / "root").mkdir()
    adapter = _Adapter(
        [
            _input(file_digest=_A, note_digest=_A),
            _input(file_digest=_A, note_digest=_A),
        ]
    )
    owner, _, _ = _owner(store=store, admitted=True, adapter=adapter)
    await owner.start()
    if operation == "resume_root":
        await owner.pause_root("root-1")
        expected_state = NotesSyncRootState.ACTIVE
    else:
        expected_state = NotesSyncRootState.PAUSED

    store.arm()
    in_flight = asyncio.create_task(getattr(owner, operation)("root-1"))
    assert await asyncio.to_thread(store.mid_transaction.wait, 5)

    shutdown = asyncio.create_task(owner.shutdown())
    await asyncio.sleep(0.05)
    assert not shutdown.done()

    store.release.set()
    result = await in_flight
    await shutdown

    assert result.accepted is True
    assert store.get_root("root-1").state is expected_state


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("resume_input", "accepted", "status", "next_action", "root_state"),
    (
        (
            _input(file_digest=_A, note_digest=_A),
            True,
            "up_to_date",
            "sync_now",
            NotesSyncRootState.ACTIVE,
        ),
        (
            _input(),
            False,
            "changes_available",
            "review_changes",
            NotesSyncRootState.PAUSED,
        ),
    ),
)
async def test_resume_checks_fresh_state_before_reopening_a_paused_root(
    tmp_path: Path,
    resume_input: ReconciliationInput,
    accepted: bool,
    status: str,
    next_action: str,
    root_state: NotesSyncRootState,
) -> None:
    store = _store(tmp_path)
    adapter = _Adapter([_input(file_digest=_A, note_digest=_A), resume_input])
    owner, _, _ = _owner(store=store, admitted=True, adapter=adapter)
    await owner.start()
    await owner.pause_root("root-1")

    result = await owner.resume_root("root-1")

    assert (result.accepted, result.status, result.next_action) == (
        accepted,
        status,
        next_action,
    )
    assert store.get_root("root-1").state is root_state
    assert adapter.observe_calls == 2
    assert adapter.executor.executed == []
    await owner.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "admission,input_value,expected_status,expected_action",
    [
        (RootAdmissionState.PASSIVE, _input(), "passive", "open_active_process"),
        (RootAdmissionState.OFFLINE, _input(), "offline", "reconnect_folder"),
        (
            RootAdmissionState.OWNER,
            _input(file_digest=_B, note_digest=_C),
            "needs_attention",
            "review_changes",
        ),
        (
            RootAdmissionState.OWNER,
            _input(root_overlap=True),
            "unsupported",
            "review_settings",
        ),
    ],
)
async def test_non_mutating_states_publish_one_safe_next_action(
    tmp_path: Path,
    admission: RootAdmissionState,
    input_value: ReconciliationInput,
    expected_status: str,
    expected_action: str,
) -> None:
    adapter = _Adapter([input_value])
    owner, _, _ = _owner(
        store=_store(tmp_path),
        admitted=True,
        adapter=adapter,
        coordinator=_Coordinator(admission),
    )

    await owner.start()

    assert adapter.executor.executed == []
    root = owner.snapshot().roots[0]
    assert (root.status, root.next_action) == (expected_status, expected_action)
    assert owner.schedule_hint("root-1") is None
    await owner.shutdown()


@pytest.mark.asyncio
async def test_shutdown_closes_runtime_admission_then_hints_then_releases_lease(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    watcher = _Watcher(events)
    coordinator = _Coordinator()
    coordinator.events = events
    owner, _, _ = _owner(
        store=_store(tmp_path),
        admitted=True,
        adapter=_Adapter([_input(file_digest=_A, note_digest=_A)]),
        coordinator=coordinator,
        watcher=watcher,
    )
    await owner.start()
    await watcher.started.wait()

    await owner.shutdown()

    assert events == [
        "watcher-started",
        "watcher-stopped",
        "lease-admission-closed",
        "lease-settled",
        "lease-released",
    ]


@pytest.mark.asyncio
async def test_shutdown_exhausts_leases_and_retries_only_failed_closes(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    second_path = tmp_path / "root-2"
    second_path.mkdir()
    store.create_root(
        NotesSyncRootRecord(
            root_id="root-2",
            note_scope_id="local_note",
            logical_folder_id="folder-2",
            canonical_path=str(second_path),
            direction=NotesSyncDirection.BIDIRECTIONAL,
            state=NotesSyncRootState.ACTIVE,
        )
    )
    coordinator = _RetryingCloseCoordinator(
        {
            "root-1": str(tmp_path / "root"),
            "root-2": str(second_path),
        }
    )
    owner, _, _ = _owner(
        store=store,
        admitted=True,
        adapter=_MultiRootAdapter([_input()]),
        coordinator=coordinator,
    )
    await owner.start()

    await asyncio.gather(owner.shutdown(), owner.shutdown())

    assert coordinator.close_attempts.count("root-1") == 1
    assert coordinator.close_attempts.count("root-2") == 1
    assert set(owner._leases) == {"root-1"}
    assert (owner.snapshot().status, owner.snapshot().next_action) == (
        "failed",
        "review_settings",
    )
    assert "private path" not in repr(owner.snapshot())

    await owner.shutdown()

    assert coordinator.close_attempts.count("root-1") == 2
    assert coordinator.close_attempts.count("root-2") == 1
    assert owner._leases == {}
    assert (owner.snapshot().status, owner.snapshot().next_action) == (
        "stopped",
        "none",
    )


@pytest.mark.asyncio
async def test_shutdown_during_startup_settles_current_action_without_reopening(
    tmp_path: Path,
) -> None:
    adapter = _Adapter([_two_action_input()])
    executor = _BlockingExecutor()
    adapter.executor = executor
    watcher = _Watcher()
    coordinator = _Coordinator()
    owner, _, _ = _owner(
        store=_store(tmp_path),
        admitted=True,
        adapter=adapter,
        coordinator=coordinator,
        watcher=watcher,
    )
    start = asyncio.create_task(owner.start())
    await executor.started.wait()

    shutdown = asyncio.create_task(owner.shutdown())
    await asyncio.sleep(0)
    assert not shutdown.done()
    assert not watcher.started.is_set()

    executor.release.set()
    await start
    await shutdown

    assert len(executor.executed) == 1
    assert not watcher.started.is_set()
    assert owner._store.get_root("root-1").last_status_code == "partial"
    assert coordinator.events[-1] == "lease-released"


@pytest.mark.asyncio
async def test_shutdown_joins_an_admitted_manual_apply_before_releasing_lease(
    tmp_path: Path,
) -> None:
    adapter = _Adapter(
        [
            _input(file_digest=_A, note_digest=_A),
            _input(),
            _input(),
        ]
    )
    executor = _BlockingExecutor()
    adapter.executor = executor
    coordinator = _Coordinator()
    owner, _, _ = _owner(
        store=_store(tmp_path),
        admitted=True,
        adapter=adapter,
        coordinator=coordinator,
    )
    await owner.start()
    reviewed = await owner.check_root("root-1")
    apply = asyncio.create_task(
        owner.apply_reviewed(
            "root-1",
            reviewed.observation_token,
            tuple(action.action_id for action in reviewed.safe_actions),
        )
    )
    await executor.started.wait()

    shutdown = asyncio.create_task(owner.shutdown())
    await asyncio.sleep(0)

    assert not shutdown.done()
    assert coordinator.events == []
    executor.release.set()
    await apply
    await shutdown
    assert coordinator.events[-1] == "lease-released"


@pytest.mark.asyncio
async def test_shutdown_releases_the_lease_after_a_watcher_failure(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    watcher = _FailedWatcher(events)
    coordinator = _Coordinator()
    coordinator.events = events
    owner, _, _ = _owner(
        store=_store(tmp_path),
        admitted=True,
        adapter=_Adapter([_input(file_digest=_A, note_digest=_A)]),
        coordinator=coordinator,
        watcher=watcher,
    )
    await owner.start()
    await watcher.started.wait()

    await asyncio.sleep(0)
    snapshot = owner.snapshot()
    assert (snapshot.status, snapshot.next_action) == (
        "failed",
        "sync_now",
    )

    await owner.shutdown()

    assert events[-1] == "lease-released"


@pytest.mark.asyncio
async def test_sync_now_restarts_one_failed_watcher_after_a_fresh_check(
    tmp_path: Path,
) -> None:
    from tldw_chatbook.Notes.notes_sync_runtime import NotesSyncRuntimeOwner

    failed = _FailedWatcher()
    replacement = _Watcher()
    watchers = iter((failed, replacement))
    adapter = _Adapter(
        [
            _input(file_digest=_A, note_digest=_A),
            _input(file_digest=_A, note_digest=_A),
        ]
    )
    owner = NotesSyncRuntimeOwner(
        store=_store(tmp_path),
        migrate_legacy=lambda: None,
        coordinator=_Coordinator(),
        adapter=adapter,
        watcher_factory=lambda _schedule: next(watchers),
        cutover_admitted=True,
        profile_process_is_sole=True,
    )
    await owner.start()
    await failed.started.wait()
    await asyncio.sleep(0)
    assert owner.snapshot().status == "failed"
    assert owner.schedule_hint("root-1") is None
    assert adapter.observe_calls == 1
    assert adapter.executor.executed == []

    await owner.request_sync_now("root-1")
    await replacement.started.wait()

    assert owner.snapshot().status == "active"
    assert adapter.observe_calls == 2
    hint = owner.schedule_hint("root-1")
    assert hint is not None
    await hint
    assert adapter.observe_calls == 3
    await owner.shutdown()


@pytest.mark.asyncio
async def test_failed_sync_now_check_does_not_restart_the_watcher(
    tmp_path: Path,
) -> None:
    from tldw_chatbook.Notes.notes_sync_runtime import NotesSyncRuntimeOwner

    failed = _FailedWatcher()
    factory_calls = 0

    def watcher_factory(_schedule):
        nonlocal factory_calls
        factory_calls += 1
        return failed

    owner = NotesSyncRuntimeOwner(
        store=_store(tmp_path),
        migrate_legacy=lambda: None,
        coordinator=_Coordinator(),
        adapter=_FailingObservationAdapter([_input(file_digest=_A, note_digest=_A)]),
        watcher_factory=watcher_factory,
        cutover_admitted=True,
        profile_process_is_sole=True,
    )
    await owner.start()
    await failed.started.wait()
    await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="observation"):
        await owner.request_sync_now("root-1")

    assert owner.snapshot().status == "failed"
    assert factory_calls == 1
    await owner.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation_state", "root_status", "expected_action"),
    [
        ("attention", "needs_attention", "resolve_cleanup"),
        ("pending", "failed", "review_changes"),
    ],
)
async def test_sync_now_cannot_clear_an_unresolved_recovery_journal(
    tmp_path: Path,
    operation_state: str,
    root_status: str,
    expected_action: str,
) -> None:
    from tldw_chatbook.Notes.notes_sync_runtime import NotesSyncRuntimeOwner

    store = _store(tmp_path)
    _pending_operation(store)
    if operation_state == "attention":
        store.mark_operation_attention("operation-1", "replacement_cleanup_pending")
    store.update_root_status("root-1", root_status)
    failed = _FailedWatcher()
    replacement = _Watcher()
    watchers = iter((failed, replacement))
    factory_calls = 0

    def watcher_factory(_schedule):
        nonlocal factory_calls
        factory_calls += 1
        return next(watchers)

    adapter = _Adapter([_input(file_digest=_A, note_digest=_A)])
    owner = NotesSyncRuntimeOwner(
        store=store,
        migrate_legacy=lambda: None,
        coordinator=_Coordinator(),
        adapter=adapter,
        watcher_factory=watcher_factory,
        cutover_admitted=True,
        profile_process_is_sole=True,
    )
    await owner.start()
    await failed.started.wait()
    await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="recovery_unresolved"):
        await owner.request_sync_now("root-1")

    root = owner.snapshot().roots[0]
    assert (root.status, root.next_action, root.action_id) == (
        root_status,
        expected_action,
        "operation-1",
    )
    assert factory_calls == 1
    assert adapter.observe_calls == 0
    assert adapter.executor.executed == []
    await owner.shutdown()


@pytest.mark.asyncio
async def test_repeated_shutdown_cancellation_waits_for_admitted_stage_and_lease(
    tmp_path: Path,
) -> None:
    adapter = _Adapter([_input(file_digest=_A, note_digest=_A), _input()])
    executor = _BlockingExecutor()
    adapter.executor = executor
    coordinator = _Coordinator()
    owner, _, _ = _owner(
        store=_store(tmp_path),
        admitted=True,
        adapter=adapter,
        coordinator=coordinator,
    )
    await owner.start()
    owner.schedule_hint("root-1")
    await executor.started.wait()

    shutdown = asyncio.create_task(owner.shutdown())
    await asyncio.sleep(0)
    shutdown.cancel("first cancellation")
    await asyncio.sleep(0)
    shutdown.cancel("second cancellation")
    await asyncio.sleep(0)

    assert not shutdown.done()
    assert coordinator.events == []

    executor.release.set()
    with pytest.raises(asyncio.CancelledError) as cancellation:
        await shutdown

    assert cancellation.value.args == ("first cancellation",)
    assert coordinator.events[-1] == "lease-released"


def test_builder_forwards_watcher_intervals_to_the_default_polling_watcher(
    tmp_path: Path,
) -> None:
    """TASK-21112: [notes] watcher intervals must reach the built watcher."""

    from tldw_chatbook.Notes.notes_sync_runtime import build_notes_sync_runtime_owner
    from tldw_chatbook.Notes.notes_sync_watcher import PollingNotesSyncWatcher

    owner = build_notes_sync_runtime_owner(
        notes_scope_service=object(),
        cutover_admitted=True,
        profile_process_is_sole=True,
        database_path=tmp_path / "sync.sqlite3",
        migrate_legacy=lambda: None,
        adapter=_Adapter([_input()]),
        watcher_interval_seconds=2.5,
        watcher_max_interval_seconds=30.0,
    )

    watcher = owner._watcher_factory(lambda _root_id: None)

    assert type(watcher) is PollingNotesSyncWatcher
    assert watcher._interval == 2.5
    assert watcher._max_interval == 30.0


def test_builder_defaults_keep_the_watcher_base_and_cap(tmp_path: Path) -> None:
    from tldw_chatbook.Notes.notes_sync_runtime import build_notes_sync_runtime_owner

    owner = build_notes_sync_runtime_owner(
        notes_scope_service=object(),
        cutover_admitted=True,
        profile_process_is_sole=True,
        database_path=tmp_path / "sync.sqlite3",
        migrate_legacy=lambda: None,
        adapter=_Adapter([_input()]),
    )

    watcher = owner._watcher_factory(lambda _root_id: None)

    assert watcher._interval == 1.0
    assert watcher._max_interval == 10.0
