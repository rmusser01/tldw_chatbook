"""Application-owned, cutover-gated runtime for lasting Database Notes sync."""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Protocol
from uuid import uuid4

from tldw_chatbook.Notes.note_import_discovery import (
    ImportSelectionError,
    discover_import_sources,
)
from tldw_chatbook.Notes.note_import_plan_models import ImportBounds
from tldw_chatbook.Notes.notes_device_state_store import (
    NotesDeviceStateStore,
    NotesSyncBindingRecord,
    NotesSyncOperationRecord,
    NotesSyncRootRecord,
    NotesSyncStoreSetting,
)
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService, ScopeType
from tldw_chatbook.Notes.notes_sync_authority import (
    NotesScopeSyncAuthority,
    NotesSyncAuthorityError,
    NotesSyncNoteSnapshot,
)
from tldw_chatbook.Notes.notes_sync_coordinator import (
    NotesSyncRootCoordinator,
    RootAdmissionState,
)
from tldw_chatbook.Notes.notes_sync_models import (
    NotesSyncAction,
    NotesSyncActionKind,
    NotesSyncBindingState,
    NotesSyncDirection,
    NotesSyncOperationState,
    NotesSyncRootState,
    validate_notes_sync_opaque_id,
)
from tldw_chatbook.Notes.notes_sync_executor import (
    NotesSyncExecutionRequest,
    NotesSyncExecutor,
)
from tldw_chatbook.Notes.notes_sync_filesystem import (
    NotesSyncFileSnapshot,
    NotesSyncFilesystemError,
    PosixNotesSyncFilesystem,
)
from tldw_chatbook.Notes.notes_sync_legacy import (
    persist_legacy_notes_sync_migration,
    plan_legacy_notes_sync_migration,
    snapshot_legacy_notes_sync,
)
from tldw_chatbook.Notes.notes_sync_reconciler import (
    BindingObservation,
    ReconciliationInput,
    ReconciliationPlan,
    assert_review_current,
    plan_reconciliation,
)
from tldw_chatbook.Notes.notes_sync_watcher import PollingNotesSyncWatcher


CUTOVER_MARKER = "notes-sync-cutover-v1"
_AUTOMATIC_ACTIONS = frozenset(
    {
        NotesSyncActionKind.CREATE_NOTE,
        NotesSyncActionKind.UPDATE_NOTE,
        NotesSyncActionKind.CREATE_FILE,
        NotesSyncActionKind.UPDATE_FILE,
    }
)
_EXECUTABLE_ACTIONS = _AUTOMATIC_ACTIONS
_SYNC_FILE_EXTENSIONS = frozenset({".md", ".markdown", ".txt"})
_OBSERVATION_BUNDLE_LIMIT = 8
_DURABLE_BLOCKED_STATUS = MappingProxyType(
    {
        "activation_recovery_required": ("needs_attention", "review_settings"),
        "failed": ("failed", "review_changes"),
        "needs_attention": ("needs_attention", "review_changes"),
        "partial": ("partial", "review_changes"),
        "unsupported": ("unsupported", "review_settings"),
    }
)

_RUNTIME_STATUSES = frozenset(
    {
        "active",
        "awaiting_cutover",
        "changes_available",
        "failed",
        "needs_attention",
        "offline",
        "partial",
        "passive",
        "paused",
        "starting",
        "stopped",
        "stopping",
        "unsupported",
        "up_to_date",
    }
)
_NEXT_ACTIONS = frozenset(
    {
        "apply_reviewed",
        "finish_upgrade",
        "none",
        "open_active_process",
        "reconnect_folder",
        "close_other_process_and_restart",
        "resolve_cleanup",
        "resume_sync",
        "review_changes",
        "review_settings",
        "review_migration",
        "sync_now",
        "wait",
    }
)


def _validate_projection(status: str, next_action: str) -> None:
    if status not in _RUNTIME_STATUSES:
        raise ValueError("unknown runtime status")
    if next_action not in _NEXT_ACTIONS:
        raise ValueError("unknown runtime next action")


@dataclass(frozen=True, slots=True)
class NotesSyncRootRuntimeSnapshot:
    """Path-free status projection for one lasting-sync root."""

    root_id: str
    status: str
    next_action: str
    action_id: str | None = None

    def __post_init__(self) -> None:
        validate_notes_sync_opaque_id(self.root_id, field_name="root_id")
        _validate_projection(self.status, self.next_action)
        if self.action_id is not None:
            validate_notes_sync_opaque_id(self.action_id, field_name="action_id")


@dataclass(frozen=True, slots=True)
class NotesSyncRuntimeSnapshot:
    """Path-free application runtime projection."""

    status: str
    next_action: str
    roots: tuple[NotesSyncRootRuntimeSnapshot, ...] = ()

    def __post_init__(self) -> None:
        _validate_projection(self.status, self.next_action)
        if type(self.roots) is not tuple or any(
            type(root) is not NotesSyncRootRuntimeSnapshot for root in self.roots
        ):
            raise TypeError("roots must be runtime root snapshots")


@dataclass(frozen=True, slots=True)
class NotesSyncControlResult:
    """Bounded outcome for a root lifecycle control."""

    accepted: bool
    status: str
    next_action: str
    applied_count: int = 0

    def __post_init__(self) -> None:
        if type(self.accepted) is not bool:
            raise TypeError("accepted must be a boolean")
        _validate_projection(self.status, self.next_action)
        if type(self.applied_count) is not int or not 0 <= self.applied_count <= 1_000:
            raise ValueError("applied_count must be a bounded non-negative integer")


@dataclass(frozen=True, slots=True, repr=False)
class NotesSyncRootSetup:
    """Private local-root configuration used only for review and activation."""

    display_name: str
    canonical_path: str
    note_scope_id: str
    direction: NotesSyncDirection

    def __post_init__(self) -> None:
        if (
            type(self.display_name) is not str
            or not self.display_name.strip()
            or len(self.display_name) > 160
            or "\n" in self.display_name
            or "/" in self.display_name
            or "\\" in self.display_name
        ):
            raise ValueError("display_name must be a bounded non-path label")
        path = Path(self.canonical_path)
        if (
            type(self.canonical_path) is not str
            or not self.canonical_path
            or len(self.canonical_path) > 4096
            or "\x00" in self.canonical_path
            or not path.is_absolute()
        ):
            raise ValueError("canonical_path must be a bounded absolute path")
        validate_notes_sync_opaque_id(self.note_scope_id, field_name="note_scope_id")
        if type(self.direction) is not NotesSyncDirection:
            raise TypeError("direction must be a NotesSyncDirection")

    def __repr__(self) -> str:
        return "NotesSyncRootSetup(<private>)"


@dataclass(frozen=True, slots=True, repr=False)
class _SetupReview:
    setup: NotesSyncRootSetup
    plan: ReconciliationPlan

    def __repr__(self) -> str:
        return "_SetupReview(<private>)"


@dataclass(frozen=True, slots=True, repr=False)
class _FreshAuthority:
    observations: ReconciliationInput
    plan: ReconciliationPlan
    requests: Mapping[str, object]

    def __repr__(self) -> str:
        return "_FreshAuthority(<private>)"


class _RuntimeAdapter(Protocol):
    async def observe_root(self, root: NotesSyncRootRecord) -> ReconciliationInput: ...

    async def build_execution_request(
        self,
        root: NotesSyncRootRecord,
        observations: ReconciliationInput,
        plan: ReconciliationPlan,
        action: NotesSyncAction,
    ) -> object: ...

    def executor_for(
        self,
        root: NotesSyncRootRecord,
        *,
        after_stage: Callable[[NotesSyncOperationState], None],
    ) -> object: ...

    async def create_root_folder(self, display_name: str) -> tuple[str, object]: ...

    async def rollback_root_folder(self, receipt: object) -> None: ...


@dataclass(frozen=True, slots=True, repr=False)
class _ObservedBinding:
    record: NotesSyncBindingRecord
    note: NotesSyncNoteSnapshot | None
    file: NotesSyncFileSnapshot | None


class _ProductionRuntimeAdapter:
    """Small concrete bridge over the completed TASK-19005/19007 boundaries."""

    def __init__(
        self,
        store: NotesDeviceStateStore,
        notes_scope_service: NotesScopeService,
        *,
        local_user_id: str,
        recovery_capacity_bytes: int,
    ) -> None:
        self._store = store
        self._service = notes_scope_service
        self._user_id = local_user_id
        self._capacity = recovery_capacity_bytes
        self._filesystems: dict[str, PosixNotesSyncFilesystem] = {}
        self._bundles: dict[str, Mapping[str, _ObservedBinding]] = {}
        self._root_signatures: dict[str, tuple[object, ...]] = {}
        file_limit = min(recovery_capacity_bytes, 10 * 1024 * 1024)
        self._discovery_bounds = ImportBounds(
            max_files=1_000,
            max_file_bytes=file_limit,
            max_total_bytes=min(
                max(file_limit, recovery_capacity_bytes), 512 * 1024 * 1024
            ),
            max_depth=32,
            max_entries=10_000,
        )

    async def create_root_folder(self, display_name: str) -> tuple[str, object]:
        folder = await self._service.create_note_folder(
            scope=ScopeType.LOCAL_NOTE,
            name=display_name,
            parent_id=None,
            user_id=self._user_id,
        )
        return folder.folder_id, (folder.folder_id, folder.version)

    async def rollback_root_folder(self, receipt: object) -> None:
        if (
            type(receipt) is not tuple
            or len(receipt) != 2
            or type(receipt[0]) is not str
            or type(receipt[1]) is not int
        ):
            raise TypeError("folder rollback receipt is invalid")
        await self._service.delete_note_folder(
            scope=ScopeType.LOCAL_NOTE,
            folder_id=receipt[0],
            expected_version=receipt[1],
            user_id=self._user_id,
        )

    def _notes(self, root: NotesSyncRootRecord) -> NotesScopeSyncAuthority:
        return NotesScopeSyncAuthority(
            self._service,
            scope=ScopeType.LOCAL_NOTE,
            user_id=self._user_id,
            note_scope_id=root.note_scope_id,
        )

    def _filesystem(self, root: NotesSyncRootRecord) -> PosixNotesSyncFilesystem:
        filesystem = self._filesystems.get(root.root_id)
        if filesystem is None:
            filesystem = PosixNotesSyncFilesystem(root.canonical_path)
            filesystem.__enter__()
            self._filesystems[root.root_id] = filesystem
        return filesystem

    async def observe_root(self, root: NotesSyncRootRecord) -> ReconciliationInput:
        notes = self._notes(root)
        filesystem = self._filesystem(root)
        observed: list[BindingObservation] = []
        bundle: dict[str, _ObservedBinding] = {}
        discovery = await asyncio.to_thread(
            discover_import_sources,
            (Path(root.canonical_path),),
            self._discovery_bounds,
        )
        if discovery.failures:
            raise RuntimeError("root_discovery_incomplete")
        discovered: dict[str, NotesSyncFileSnapshot] = {}
        for candidate in discovery.candidates:
            relative_path = candidate.source.source_path.relative_to(
                Path(root.canonical_path)
            ).as_posix()
            if Path(relative_path).suffix.casefold() not in _SYNC_FILE_EXTENSIONS:
                continue
            try:
                discovered[relative_path] = await asyncio.to_thread(
                    filesystem.observe, relative_path
                )
            except NotesSyncFilesystemError as error:
                if error.reason_code != "missing_target":
                    raise RuntimeError("root_discovery_incomplete") from None

        bindings = await asyncio.to_thread(self._store.list_bindings, root.root_id)
        if any(
            binding.state
            not in {NotesSyncBindingState.ACTIVE, NotesSyncBindingState.CANDIDATE}
            for binding in bindings
        ):
            raise RuntimeError("binding_review_required")

        def observe_notes() -> dict[str, NotesSyncNoteSnapshot | None]:
            async def observe_all() -> dict[str, NotesSyncNoteSnapshot | None]:
                observed_notes: dict[str, NotesSyncNoteSnapshot | None] = {}
                for binding in bindings:
                    try:
                        observed_notes[binding.binding_id] = await notes.observe(
                            binding.note_id
                        )
                    except NotesSyncAuthorityError as error:
                        if error.reason_code != "note_missing":
                            raise
                        observed_notes[binding.binding_id] = None
                return observed_notes

            return asyncio.run(observe_all())

        note_observations = await asyncio.to_thread(observe_notes)
        claimed_paths: set[str] = set()
        for binding in bindings:
            note = note_observations[binding.binding_id]
            file = discovered.get(binding.normalized_relative_path)
            if file is None:
                identity_matches = tuple(
                    item
                    for item in discovered.values()
                    if NotesSyncExecutor.stable_identity_digest(item)
                    == binding.stable_identity_digest
                )
                file = identity_matches[0] if len(identity_matches) == 1 else None
            if file is not None:
                claimed_paths.add(file.observation.relative_path)
            file_digest = file.observation.content_digest if file else None
            note_digest = note.content_digest if note else None
            file_identity = (
                NotesSyncExecutor.stable_identity_digest(file) if file else None
            )
            observed.append(
                BindingObservation(
                    binding_id=binding.binding_id,
                    baseline_file_digest=binding.content_digest,
                    baseline_note_digest=binding.content_digest,
                    baseline_identity_digest=binding.stable_identity_digest,
                    baseline_relative_path=binding.normalized_relative_path,
                    file_digest=file_digest,
                    note_digest=note_digest,
                    file_identity_digest=file_identity,
                    relative_path=(
                        file.observation.relative_path
                        if file
                        else binding.normalized_relative_path
                    ),
                    note_scope_id=binding.note_scope_id,
                    note_id=binding.note_id,
                    note_version=note.version if note else binding.note_version,
                    bound=(
                        binding.state is NotesSyncBindingState.ACTIVE
                        or (
                            binding.state is NotesSyncBindingState.CANDIDATE
                            and root.state is NotesSyncRootState.PAUSED
                            and root.last_status_code == "migration_review_required"
                        )
                    ),
                    baseline_serialization=binding.serialization,
                    serialization=(file.observation.serialization if file else None),
                )
            )
            bundle[binding.binding_id] = _ObservedBinding(binding, note, file)

        for relative_path, file in discovered.items():
            if relative_path in claimed_paths:
                continue
            binding_id = hashlib.sha256(
                f"binding\0{root.root_id}\0{relative_path}".encode("utf-8")
            ).hexdigest()
            note_id = hashlib.sha256(
                f"note\0{root.root_id}\0{relative_path}".encode("utf-8")
            ).hexdigest()
            identity_digest = NotesSyncExecutor.stable_identity_digest(file)
            candidate = NotesSyncBindingRecord(
                binding_id=binding_id,
                root_id=root.root_id,
                note_scope_id=root.note_scope_id,
                note_id=note_id,
                normalized_relative_path=relative_path,
                stable_identity_digest=identity_digest,
                state=NotesSyncBindingState.CANDIDATE,
                serialization=file.observation.serialization,
                content_digest=file.observation.content_digest,
                note_version=0,
            )
            observed.append(
                BindingObservation(
                    binding_id=binding_id,
                    baseline_file_digest=file.observation.content_digest,
                    baseline_note_digest=file.observation.content_digest,
                    baseline_identity_digest=identity_digest,
                    baseline_relative_path=relative_path,
                    file_digest=file.observation.content_digest,
                    note_digest=None,
                    file_identity_digest=identity_digest,
                    relative_path=relative_path,
                    note_scope_id=root.note_scope_id,
                    note_id=note_id,
                    note_version=0,
                    bound=False,
                    baseline_serialization=file.observation.serialization,
                    serialization=file.observation.serialization,
                )
            )
            bundle[binding_id] = _ObservedBinding(candidate, None, file)
        request = ReconciliationInput(
            root_id=root.root_id,
            direction=root.direction,
            bindings=tuple(observed),
            observation_generation=max(
                (item.note_version for item in observed), default=0
            ),
            expected_generation=max(
                (item.note_version for item in observed), default=0
            ),
        )
        token = plan_reconciliation(request).observation_token
        if len(self._bundles) >= _OBSERVATION_BUNDLE_LIMIT:
            raise RuntimeError("observation_capacity_exceeded")
        self._bundles[token] = MappingProxyType(bundle)
        self._root_signatures[root.root_id] = self._discovery_signature(discovery)
        return request

    @staticmethod
    def _discovery_signature(discovery: object) -> tuple[object, ...]:
        return tuple(
            (
                candidate.source.display_path,
                candidate.identity.device,
                candidate.identity.inode,
                candidate.identity.size,
                candidate.identity.modified_ns,
                candidate.identity.changed_ns,
            )
            for candidate in discovery.candidates
            if Path(candidate.source.display_path).suffix.casefold()
            in _SYNC_FILE_EXTENSIONS
        )

    def changed_root_ids(self, roots: Mapping[str, str]) -> tuple[str, ...]:
        """Return only roots whose bounded metadata inventory changed."""

        changed: list[str] = []
        for root_id, path in roots.items():
            try:
                discovery = discover_import_sources(
                    (Path(path),), self._discovery_bounds
                )
                signature: tuple[object, ...] = self._discovery_signature(discovery)
            except (ImportSelectionError, OSError):
                signature = ("unavailable",)
            previous = self._root_signatures.get(root_id)
            self._root_signatures[root_id] = signature
            if previous is not None and previous != signature:
                changed.append(root_id)
        return tuple(changed)

    async def build_execution_request(
        self,
        root: NotesSyncRootRecord,
        observations: ReconciliationInput,
        plan: ReconciliationPlan,
        action: NotesSyncAction,
    ) -> NotesSyncExecutionRequest:
        if action.binding_id is None or root.logical_folder_id is None:
            raise RuntimeError("execution_binding_required")
        binding = self._bundles.get(plan.observation_token, {}).get(action.binding_id)
        if binding is None:
            raise RuntimeError("private_execution_authority_missing")
        operation_id = hashlib.sha256(
            f"operation\0{action.action_id}".encode("ascii")
        ).hexdigest()
        note = binding.note
        file = binding.file
        return NotesSyncExecutionRequest(
            operation_id=operation_id,
            root_id=root.root_id,
            logical_folder_id=root.logical_folder_id,
            direction=root.direction,
            binding_id=binding.record.binding_id,
            observation_token=plan.observation_token,
            action_kind=action.kind,
            note=note,
            file=file,
            desired_title=(
                note.title
                if note is not None
                else Path(binding.record.normalized_relative_path).stem
            ),
            recovery_id=f"recovery-{operation_id}",
            recovery_expires_at=time.time_ns() + 86_400_000_000_000,
            candidate_note_scope_id=(
                binding.record.note_scope_id
                if action.kind is NotesSyncActionKind.CREATE_NOTE
                else None
            ),
            candidate_note_id=(
                binding.record.note_id
                if action.kind is NotesSyncActionKind.CREATE_NOTE
                else None
            ),
            candidate_relative_path=(
                binding.record.normalized_relative_path
                if action.kind is NotesSyncActionKind.CREATE_FILE
                else None
            ),
            candidate_serialization=(
                binding.record.serialization
                if action.kind is NotesSyncActionKind.CREATE_FILE
                else None
            ),
        )

    def release_observation(self, observation_token: str) -> None:
        self._bundles.pop(observation_token, None)

    def executor_for(
        self,
        root: NotesSyncRootRecord,
        *,
        after_stage: Callable[[NotesSyncOperationState], None],
    ) -> NotesSyncExecutor:
        return NotesSyncExecutor(
            self._store,
            self._notes(root),
            self._filesystem(root),
            recovery_capacity_bytes=self._capacity,
            after_stage=after_stage,
        )

    def close(self) -> None:
        for filesystem in self._filesystems.values():
            filesystem.__exit__(None, None, None)
        self._filesystems.clear()


class NotesSyncRuntimeOwner:
    """Own root leases, reviewed reconciliation, watcher hints, and shutdown."""

    def __init__(
        self,
        *,
        store: NotesDeviceStateStore,
        migrate_legacy: Callable[[], object],
        coordinator: object | Callable[[], object],
        adapter: _RuntimeAdapter,
        watcher_factory: Callable[[Callable[[str], object]], object],
        file_notes_binding: Callable[[], object | None] = lambda: None,
        cutover_admitted: bool,
        profile_process_is_sole: bool,
    ) -> None:
        if type(cutover_admitted) is not bool:
            raise TypeError("cutover_admitted must be a private boolean.")
        if type(profile_process_is_sole) is not bool:
            raise TypeError("profile_process_is_sole must be a private boolean.")
        if not callable(migrate_legacy) or not callable(watcher_factory):
            raise TypeError("runtime factories must be callable.")
        self._store = store
        self._migrate_legacy = migrate_legacy
        self._coordinator_source = coordinator
        self._adapter = adapter
        self._watcher_factory = watcher_factory
        self._file_notes_binding = file_notes_binding
        self._cutover_admitted = cutover_admitted
        self._profile_process_is_sole = profile_process_is_sole
        self._coordinator: object | None = None
        self._watcher: object | None = None
        self._watcher_task: asyncio.Task[None] | None = None
        self._start_task: asyncio.Task[None] | None = None
        self._shutdown_task: asyncio.Task[None] | None = None
        self._hint_tasks: dict[str, asyncio.Task[None]] = {}
        self._dirty_hints: set[str] = set()
        self._active_tasks: dict[str, set[asyncio.Task[object]]] = {}
        self._leases: dict[str, object] = {}
        self._admissions: dict[str, object] = {}
        self._blocked_roots: set[str] = set()
        self._durably_blocked_roots: set[str] = set()
        self._closed_roots: set[str] = set()
        self._reviews: dict[str, ReconciliationPlan] = {}
        self._setup_reviews: dict[str, _SetupReview] = {}
        self._root_status: dict[str, NotesSyncRootRuntimeSnapshot] = {}
        self._root_paths: dict[str, str] = {}
        self._status = "starting"
        self._next_action = "wait"
        self._admission_open = False
        self._closing = False

    def snapshot(self) -> NotesSyncRuntimeSnapshot:
        """Return the current path-free UI projection."""

        return NotesSyncRuntimeSnapshot(
            self._status,
            self._next_action,
            tuple(self._root_status[key] for key in sorted(self._root_status)),
        )

    async def start(self) -> None:
        """Initialize once and remain inert unless both cutover gates match."""

        if self._start_task is None:
            self._start_task = asyncio.create_task(
                self._start_once(), name="notes_sync_runtime_start"
            )
        await asyncio.shield(self._start_task)

    async def _start_once(self) -> None:
        try:
            await asyncio.to_thread(self._store.initialize)
            marker = await asyncio.to_thread(self._store.get_setting, "cutover_marker")
        except Exception:
            self._status = "failed"
            self._next_action = "review_settings"
            return
        if self._closing:
            return
        if marker is not None and marker.value != CUTOVER_MARKER:
            self._status = "awaiting_cutover"
            self._next_action = "finish_upgrade"
            return
        if marker is None:
            try:
                await asyncio.to_thread(self._migrate_legacy)
                if not self._cutover_admitted:
                    self._status = "awaiting_cutover"
                    self._next_action = "finish_upgrade"
                    return
                marker = NotesSyncStoreSetting("cutover_marker", CUTOVER_MARKER)
                await asyncio.to_thread(
                    self._store.set_setting,
                    marker,
                )
            except Exception:
                self._status = "failed"
                self._next_action = "review_settings"
                return
        if not self._cutover_admitted or marker is None:
            self._status = "awaiting_cutover"
            self._next_action = "finish_upgrade"
            return
        if not self._profile_process_is_sole:
            self._status = "awaiting_cutover"
            self._next_action = "close_other_process_and_restart"
            return

        try:
            self._coordinator = (
                self._coordinator_source()
                if callable(self._coordinator_source)
                else self._coordinator_source
            )
            roots = await self._load_roots()
            self._root_paths = {
                root_id: root.canonical_path for root_id, root in roots.items()
            }
            incomplete_operations = await asyncio.to_thread(
                self._store.list_incomplete_operations
            )
        except Exception:
            self._status = "failed"
            self._next_action = "review_settings"
            return
        try:
            incomplete_root_ids = {
                operation.root_id for operation in incomplete_operations
            }
            for root in roots.values():
                blocked = _DURABLE_BLOCKED_STATUS.get(root.last_status_code or "")
                if root.state is not NotesSyncRootState.ACTIVE or blocked is None:
                    continue
                self._durably_blocked_roots.add(root.root_id)
                if root.root_id in incomplete_root_ids:
                    continue
                self._blocked_roots.add(root.root_id)
                await self._publish(root.root_id, *blocked)
            await self._resume_incomplete(roots, incomplete_operations)
        except Exception:
            self._status = "failed"
            self._next_action = "review_settings"
            return
        for root in roots.values():
            if self._closing:
                break
            if root.state is NotesSyncRootState.PAUSED:
                durable = _DURABLE_BLOCKED_STATUS.get(root.last_status_code or "")
                if durable is not None:
                    self._blocked_roots.add(root.root_id)
                    self._durably_blocked_roots.add(root.root_id)
                    await self._publish(root.root_id, *durable, persist=False)
                elif root.last_status_code == "migration_review_required":
                    await self._publish(
                        root.root_id,
                        "needs_attention",
                        "review_migration",
                        persist=False,
                    )
                else:
                    await self._publish(root.root_id, "paused", "resume_sync")
                continue
            if root.state is not NotesSyncRootState.ACTIVE:
                continue
            if root.root_id in self._blocked_roots:
                continue
            if not await self._ensure_lease(root):
                continue
            try:
                await self._reconcile(root, automatic=True)
            except Exception:
                self._blocked_roots.add(root.root_id)
                await self._publish(root.root_id, "failed", "review_changes")

        self._status = "active"
        self._next_action = "sync_now"
        self._admission_open = True
        if self._closing:
            self._admission_open = False
            return
        self._start_watcher()

    def _start_watcher(self) -> None:
        if self._closing or not self._admission_open or not self._leases:
            return
        if self._watcher_task is not None and not self._watcher_task.done():
            return
        self._watcher = self._watcher_factory(self.schedule_hint)
        self._watcher_task = asyncio.create_task(
            self._watcher.run(), name="notes_sync_watcher"
        )
        self._watcher_task.add_done_callback(self._watcher_finished)

    def _watcher_finished(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        task.exception()
        if self._admission_open:
            self._status = "failed"
            self._next_action = "sync_now"

    async def _load_roots(self) -> dict[str, NotesSyncRootRecord]:
        summaries = await asyncio.to_thread(self._store.list_root_summaries)
        roots: dict[str, NotesSyncRootRecord] = {}
        for summary in summaries:
            if summary.state is NotesSyncRootState.DISCONNECTED:
                continue
            roots[summary.root_id] = await asyncio.to_thread(
                self._store.get_root, summary.root_id
            )
        return roots

    async def _ensure_lease(self, root: NotesSyncRootRecord) -> bool:
        existing = self._leases.get(root.root_id)
        if existing is not None and getattr(existing, "authoritative", False):
            return True
        assert self._coordinator is not None
        admission = await asyncio.to_thread(
            self._coordinator.try_acquire,
            root.canonical_path,
            lasting_roots=tuple(
                path
                for root_id, path in self._root_paths.items()
                if root_id != root.root_id
            ),
            file_notes_binding=self._file_notes_binding(),
        )
        if admission.state is RootAdmissionState.OWNER:
            self._leases[root.root_id] = admission.require_authority("plan")
            self._admissions[root.root_id] = admission
            if root.root_id not in self._durably_blocked_roots:
                self._blocked_roots.discard(root.root_id)
            return True
        status, action = {
            RootAdmissionState.PASSIVE: ("passive", "open_active_process"),
            RootAdmissionState.OFFLINE: ("offline", "reconnect_folder"),
            RootAdmissionState.REJECTED: ("unsupported", "review_settings"),
        }[admission.state]
        await self._publish(root.root_id, status, action)
        return False

    async def _fresh_authority(self, root: NotesSyncRootRecord) -> _FreshAuthority:
        self._require_authority(root.root_id, "plan")
        observations = await self._adapter.observe_root(root)
        plan: ReconciliationPlan | None = None
        try:
            plan = plan_reconciliation(observations)
            self._require_authority(root.root_id, "plan")
            if observations.root_id != root.root_id:
                raise RuntimeError("root_observation_mismatch")
            if observations.direction is not root.direction:
                raise RuntimeError("root_direction_changed")
            requests: dict[str, object] = {}
            if not plan.attention and not plan.skips and not plan.deletion_groups:
                for action in plan.safe_actions:
                    if action.kind in _EXECUTABLE_ACTIONS:
                        self._require_authority(root.root_id, "write")
                        requests[
                            action.action_id
                        ] = await self._adapter.build_execution_request(
                            root, observations, plan, action
                        )
                        self._require_authority(root.root_id, "write")
            return _FreshAuthority(observations, plan, MappingProxyType(requests))
        finally:
            release = getattr(self._adapter, "release_observation", None)
            if plan is not None and callable(release):
                release(plan.observation_token)

    async def _reconcile(
        self,
        root: NotesSyncRootRecord,
        *,
        automatic: bool,
    ) -> ReconciliationPlan:
        lease = self._leases.get(root.root_id)
        if lease is None or not getattr(lease, "authoritative", False):
            raise RuntimeError("root_lease_required")
        authority = await self._fresh_authority(root)
        plan = authority.plan
        self._reviews[root.root_id] = plan
        blocked = self._blocked_plan_status(plan)
        if blocked is not None:
            self._blocked_roots.add(root.root_id)
            await self._publish(root.root_id, *blocked)
            return plan
        selected = tuple(
            action
            for action in plan.safe_actions
            if action.kind in (_AUTOMATIC_ACTIONS if automatic else _EXECUTABLE_ACTIONS)
        )
        if automatic and any(
            getattr(authority.requests[action.action_id], "direction_override", None)
            is not None
            for action in selected
        ):
            self._blocked_roots.add(root.root_id)
            await self._publish(root.root_id, "needs_attention", "review_changes")
            return plan
        if automatic and selected:
            await self._execute(root, authority, selected)
        elif selected:
            await self._publish(root.root_id, "changes_available", "review_changes")
        else:
            await self._publish(root.root_id, "up_to_date", "sync_now")
        return plan

    @staticmethod
    def _blocked_plan_status(
        plan: ReconciliationPlan,
    ) -> tuple[str, str] | None:
        if plan.attention or plan.deletion_groups or plan.managed_placement_effects:
            return "needs_attention", "review_changes"
        if plan.skips:
            reason = plan.skips[0].reason_code
            if reason == "root_offline":
                return "offline", "reconnect_folder"
            return "unsupported", "review_settings"
        return None

    async def _review_candidate(self, root: NotesSyncRootRecord) -> ReconciliationPlan:
        """Build a complete mutation-free plan for a pending or migrated root."""

        if root.note_scope_id != ScopeType.LOCAL_NOTE.value:
            await self._publish(root.root_id, "unsupported", "review_settings")
            raise RuntimeError("noncanonical_note_scope")
        self._require_authority(root.root_id, "plan")
        observations = await self._adapter.observe_root(root)
        plan: ReconciliationPlan | None = None
        try:
            plan = plan_reconciliation(observations)
            self._require_authority(root.root_id, "plan")
            if observations.root_id != root.root_id:
                raise RuntimeError("root_observation_mismatch")
            if observations.direction is not root.direction:
                raise RuntimeError("root_direction_changed")
            return plan
        finally:
            release = getattr(self._adapter, "release_observation", None)
            if plan is not None and callable(release):
                release(plan.observation_token)

    async def review_setup(self, setup: NotesSyncRootSetup) -> ReconciliationPlan:
        """Review a new local root without persisting root or note/file changes."""

        if type(setup) is not NotesSyncRootSetup:
            raise TypeError("setup must be a NotesSyncRootSetup")
        if setup.note_scope_id != ScopeType.LOCAL_NOTE.value:
            raise ValueError("setup note scope must be local_note")
        self._require_cutover("setup-review")
        matching_root_id = next(
            (
                root_id
                for root_id, review in self._setup_reviews.items()
                if review.setup == setup
            ),
            None,
        )
        for root_id in tuple(self._setup_reviews):
            if root_id != matching_root_id:
                await self.abandon_setup(root_id)
        root_id = matching_root_id or str(uuid4())
        task = self._register_task(root_id)
        try:
            root = NotesSyncRootRecord(
                root_id=root_id,
                note_scope_id=setup.note_scope_id,
                logical_folder_id=None,
                canonical_path=setup.canonical_path,
                direction=setup.direction,
                state=NotesSyncRootState.PENDING,
            )
            self._root_paths[root_id] = setup.canonical_path
            if matching_root_id is None and not await self._ensure_lease(root):
                self._root_paths.pop(root_id, None)
                raise RuntimeError("root_lease_unavailable")
            try:
                plan = await self._review_candidate(root)
            except Exception:
                await self._release_setup_authority(root_id)
                raise
            self._setup_reviews[root_id] = _SetupReview(setup, plan)
            return plan
        finally:
            self._finish_task(root_id, task)

    async def abandon_setup(self, root_id: str) -> None:
        """Release one unpersisted setup review and its provisional lease."""

        validate_notes_sync_opaque_id(root_id, field_name="root_id")
        if not any(
            root_id in owner
            for owner in (
                self._setup_reviews,
                self._leases,
                self._admissions,
                self._root_paths,
            )
        ):
            return
        await self._release_setup_authority(root_id)

    async def _release_setup_authority(self, root_id: str) -> None:
        """Forget every provisional setup owner, even before a review exists."""

        lease = self._leases.get(root_id)
        if lease is not None and self._coordinator is not None:
            await asyncio.to_thread(
                self._coordinator.close_admission, lease, lambda: None
            )
        self._leases.pop(root_id, None)
        self._admissions.pop(root_id, None)
        self._setup_reviews.pop(root_id, None)
        self._root_paths.pop(root_id, None)
        self._root_status.pop(root_id, None)

    async def _retire_failed_setup(self, root_id: str) -> None:
        """Retire a persisted setup after proving no folder authority remains."""

        await asyncio.to_thread(
            self._store.transition_root,
            root_id,
            NotesSyncRootState.DISCONNECTED,
        )
        await self._release_setup_authority(root_id)
        self._blocked_roots.discard(root_id)
        self._durably_blocked_roots.discard(root_id)

    async def check_root(self, root_id: str) -> ReconciliationPlan:
        """Perform one fresh, mutation-free complete reconciliation."""

        task = self._admit_task(root_id)
        try:
            root = await asyncio.to_thread(self._store.get_root, root_id)
            migration_candidate = (
                root.state is NotesSyncRootState.PAUSED
                and root.last_status_code == "migration_review_required"
            )
            if root.state is not NotesSyncRootState.ACTIVE and not migration_candidate:
                await self._publish(root_id, "paused", "resume_sync")
                raise RuntimeError("sync_root_not_active")
            if not await self._ensure_lease(root):
                raise RuntimeError("root_lease_unavailable")
            incomplete = await asyncio.to_thread(self._store.list_incomplete_operations)
            root_operations = tuple(
                operation for operation in incomplete if operation.root_id == root_id
            )
            if root_operations:
                operation = next(
                    (
                        item
                        for item in root_operations
                        if item.state is NotesSyncOperationState.NEEDS_ATTENTION
                    ),
                    root_operations[0],
                )
                await self._classify_incomplete_block(
                    root, operation, block_pending=True
                )
                raise RuntimeError("sync_recovery_unresolved")
            self._blocked_roots.discard(root_id)
            self._durably_blocked_roots.discard(root_id)
            try:
                plan = await (
                    self._review_candidate(root)
                    if migration_candidate
                    else self._reconcile(root, automatic=False)
                )
                if migration_candidate:
                    self._reviews[root_id] = plan
                    await self._publish(
                        root_id,
                        "changes_available",
                        "apply_reviewed",
                        persist=False,
                    )
                return plan
            except Exception:
                self._blocked_roots.add(root_id)
                self._durably_blocked_roots.add(root_id)
                await self._publish(root_id, "failed", "review_changes")
                raise
        finally:
            self._finish_task(root_id, task)

    async def request_sync_now(self, root_id: str) -> ReconciliationPlan:
        """Alias for a fresh mutation-free manual review check."""

        plan = await self.check_root(root_id)
        if root_id not in self._blocked_roots:
            if self._status == "failed":
                self._status = "active"
                self._next_action = "sync_now"
            self._start_watcher()
        return plan

    async def apply_reviewed(
        self,
        root_id: str,
        observation_token: str,
        action_ids: tuple[str, ...],
    ) -> tuple[object, ...]:
        """Apply only selected actions after rebuilding fresh private authority."""

        task = self._admit_task(root_id)
        try:
            reviewed = self._reviews.get(root_id)
            if reviewed is None or reviewed.observation_token != observation_token:
                raise ValueError("stale_review")
            root = await asyncio.to_thread(self._store.get_root, root_id)
            if root.state is not NotesSyncRootState.ACTIVE:
                await self._publish(root_id, "paused", "resume_sync")
                raise RuntimeError("sync_root_not_active")
            authority = await self._fresh_authority(root)
            assert_review_current(reviewed, authority.observations)
            if authority.plan != reviewed:
                raise ValueError("stale_review")
            blocked = self._blocked_plan_status(authority.plan)
            if blocked is not None:
                self._blocked_roots.add(root_id)
                await self._publish(root_id, *blocked)
                raise ValueError("review_not_executable")
            selected_ids = set(action_ids)
            selected = tuple(
                action
                for action in authority.plan.safe_actions
                if action.action_id in selected_ids
                and action.kind in _EXECUTABLE_ACTIONS
            )
            if len(selected) != len(selected_ids):
                raise ValueError("reviewed_action_mismatch")
            if not selected:
                return ()
            return await self._execute(root, authority, selected)
        finally:
            self._finish_task(root_id, task)

    async def _execute(
        self,
        root: NotesSyncRootRecord,
        authority: _FreshAuthority,
        actions: tuple[NotesSyncAction, ...],
    ) -> tuple[object, ...]:
        self._require_authority(root.root_id, "write")
        executor = self._adapter.executor_for(
            root,
            after_stage=lambda _state: self._require_authority(root.root_id, "write"),
        )
        self._require_authority(root.root_id, "write")
        results: list[object] = []
        for index, action in enumerate(actions):
            if self._closing:
                return tuple(results)
            self._require_authority(root.root_id, "write")
            request = authority.requests.get(action.action_id)
            if request is None:
                raise RuntimeError("private_execution_authority_missing")
            try:
                result = await executor.execute(request)
            except Exception:
                self._blocked_roots.add(root.root_id)
                await self._publish(root.root_id, "failed", "review_changes")
                raise
            self._require_authority(root.root_id, "write")
            results.append(result)
            state = getattr(result, "state", None)
            if state is NotesSyncOperationState.NEEDS_ATTENTION:
                self._blocked_roots.add(root.root_id)
                await self._publish(
                    root.root_id,
                    "needs_attention",
                    "review_changes",
                    action_id=getattr(result, "operation_id", None),
                )
                return tuple(results)
            if state is not NotesSyncOperationState.COMPLETED:
                self._blocked_roots.add(root.root_id)
                next_action = (
                    "resolve_cleanup"
                    if getattr(result, "reason_code", None)
                    == "replacement_cleanup_pending"
                    else "review_changes"
                )
                await self._publish(
                    root.root_id,
                    "partial",
                    next_action,
                    action_id=(
                        getattr(result, "operation_id", None)
                        if next_action == "resolve_cleanup"
                        else None
                    ),
                )
                return tuple(results)
            if self._closing:
                if index == len(actions) - 1:
                    await self._publish(root.root_id, "up_to_date", "sync_now")
                else:
                    await self._publish(root.root_id, "partial", "review_changes")
                return tuple(results)
        await self._publish(root.root_id, "up_to_date", "sync_now")
        return tuple(results)

    async def _classify_incomplete_block(
        self,
        root: NotesSyncRootRecord,
        operation: NotesSyncOperationRecord,
        *,
        block_pending: bool = False,
    ) -> bool:
        durable = _DURABLE_BLOCKED_STATUS.get(root.last_status_code or "")
        if operation.state is NotesSyncOperationState.NEEDS_ATTENTION:
            status = "needs_attention"
            action = (
                "resolve_cleanup"
                if operation.reason_code == "replacement_cleanup_pending"
                else "review_changes"
            )
        elif durable is not None:
            status, action = durable
            self._durably_blocked_roots.add(root.root_id)
        elif block_pending:
            status, action = "partial", "review_changes"
        else:
            return False
        self._blocked_roots.add(root.root_id)
        await self._publish(
            root.root_id,
            status,
            action,
            action_id=operation.operation_id,
        )
        return True

    async def _resume_incomplete(
        self,
        roots: Mapping[str, NotesSyncRootRecord],
        operations: tuple[NotesSyncOperationRecord, ...],
    ) -> None:
        recovery_blocked: set[str] = set()
        for operation in operations:
            if self._closing:
                break
            root = roots.get(operation.root_id)
            if (
                root is None
                or root.state is not NotesSyncRootState.ACTIVE
                or root.root_id in recovery_blocked
            ):
                continue
            if not await self._ensure_lease(root):
                continue
            if await self._classify_incomplete_block(root, operation):
                recovery_blocked.add(root.root_id)
                continue
            try:
                self._require_authority(root.root_id, "write")
                executor = self._adapter.executor_for(
                    root,
                    after_stage=lambda _state, root_id=root.root_id: (
                        self._require_authority(root_id, "write")
                    ),
                )
                request = await executor.reconstruct_request(operation.operation_id)
                self._require_authority(root.root_id, "write")
                result = await executor.resume(request)
                self._require_authority(root.root_id, "write")
            except Exception:
                self._blocked_roots.add(root.root_id)
                await self._publish(
                    root.root_id,
                    "failed",
                    "review_changes",
                    action_id=operation.operation_id,
                )
                recovery_blocked.add(root.root_id)
                continue
            if result.state is NotesSyncOperationState.COMPLETED:
                await self._publish(root.root_id, "up_to_date", "sync_now")
            elif result.state is NotesSyncOperationState.NEEDS_ATTENTION:
                self._blocked_roots.add(root.root_id)
                await self._publish(
                    root.root_id,
                    "needs_attention",
                    "review_changes",
                    action_id=operation.operation_id,
                )
                recovery_blocked.add(root.root_id)
            else:
                self._blocked_roots.add(root.root_id)
                next_action = (
                    "resolve_cleanup"
                    if getattr(result, "reason_code", None)
                    == "replacement_cleanup_pending"
                    else "review_changes"
                )
                await self._publish(
                    root.root_id,
                    "partial",
                    next_action,
                    action_id=operation.operation_id,
                )
                recovery_blocked.add(root.root_id)

    def schedule_hint(self, root_id: str) -> object | None:
        """Coalesce one watcher hint without granting it mutation authority."""

        validate_notes_sync_opaque_id(root_id, field_name="root_id")
        if (
            not self._admission_open
            or self._status != "active"
            or self._watcher_task is None
            or self._watcher_task.done()
            or root_id in self._blocked_roots
            or root_id not in self._leases
        ):
            return None
        existing = self._hint_tasks.get(root_id)
        if existing is not None and not existing.done():
            self._dirty_hints.add(root_id)
            return existing
        task = asyncio.create_task(self._run_hint(root_id), name="notes_sync_hint")
        self._hint_tasks[root_id] = task
        return task

    def _watchable_root_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                root_id
                for root_id in self._leases
                if root_id not in self._blocked_roots
            )
        )

    def _changed_root_ids(self) -> tuple[str, ...]:
        source = getattr(self._adapter, "changed_root_ids", None)
        if not callable(source):
            return ()
        watchable = set(self._watchable_root_ids())
        return source(
            {
                root_id: path
                for root_id, path in self._root_paths.items()
                if root_id in watchable
            }
        )

    async def _run_hint(self, root_id: str) -> None:
        try:
            while self._admission_open and root_id not in self._blocked_roots:
                self._dirty_hints.discard(root_id)
                root = await asyncio.to_thread(self._store.get_root, root_id)
                if root.state is not NotesSyncRootState.ACTIVE:
                    break
                await self._reconcile(root, automatic=True)
                if root_id not in self._dirty_hints:
                    break
        except RuntimeError as error:
            self._blocked_roots.add(root_id)
            if str(error) == "root_lease_required":
                await self._publish(root_id, "offline", "reconnect_folder")
            else:
                await self._publish(root_id, "failed", "review_changes")
        except Exception:
            self._blocked_roots.add(root_id)
            await self._publish(root_id, "failed", "review_changes")
        finally:
            self._dirty_hints.discard(root_id)
            current = asyncio.current_task()
            if self._hint_tasks.get(root_id) is current:
                self._hint_tasks.pop(root_id, None)

    async def settle(self) -> None:
        """Join all currently admitted hint work."""

        tasks = tuple(self._hint_tasks.values()) + tuple(
            task for values in self._active_tasks.values() for task in values
        )
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def pause_root(self, root_id: str) -> NotesSyncControlResult:
        self._require_cutover(root_id)
        task = self._register_task(root_id)
        try:
            self._closed_roots.add(root_id)
            self._blocked_roots.add(root_id)
            await self._settle_root(root_id)
            await asyncio.to_thread(
                self._store.transition_root, root_id, NotesSyncRootState.PAUSED
            )
            lease = self._leases.pop(root_id, None)
            self._admissions.pop(root_id, None)
            if lease is not None and self._coordinator is not None:
                await asyncio.to_thread(
                    self._coordinator.close_admission, lease, lambda: None
                )
            await self._publish(root_id, "paused", "resume_sync")
            return NotesSyncControlResult(True, "paused", "resume_sync")
        finally:
            self._finish_task(root_id, task)

    async def resolve_cleanup(self, root_id: str, operation_id: str) -> object:
        """Resolve one explicit durable cleanup action under root authority."""

        task = self._admit_task(root_id)
        try:
            root = await asyncio.to_thread(self._store.get_root, root_id)
            operation = await asyncio.to_thread(self._store.get_operation, operation_id)
            if operation.root_id != root_id:
                raise ValueError("operation_root_mismatch")
            if root.state is not NotesSyncRootState.ACTIVE:
                raise RuntimeError("sync_root_not_active")
            if not await self._ensure_lease(root):
                raise RuntimeError("root_lease_unavailable")
            self._require_authority(root_id, "write")
            executor = self._adapter.executor_for(
                root,
                after_stage=lambda _state: self._require_authority(root_id, "write"),
            )
            result = await executor.resolve_filesystem_cleanup(operation_id)
            self._require_authority(root_id, "write")
            if getattr(result, "state", None) is NotesSyncOperationState.COMPLETED:
                self._blocked_roots.discard(root_id)
                await self._publish(root_id, "up_to_date", "sync_now")
            else:
                self._blocked_roots.add(root_id)
                await self._publish(
                    root_id,
                    "needs_attention",
                    "review_changes",
                    action_id=operation_id,
                )
            return result
        finally:
            self._finish_task(root_id, task)

    async def resume_root(self, root_id: str) -> NotesSyncControlResult:
        self._require_cutover(root_id)
        task = self._register_task(root_id)
        try:
            return await self._resume_root(root_id)
        finally:
            self._finish_task(root_id, task)

    async def _resume_root(self, root_id: str) -> NotesSyncControlResult:
        root = await asyncio.to_thread(self._store.get_root, root_id)
        if root.state is not NotesSyncRootState.PAUSED:
            return NotesSyncControlResult(False, "needs_attention", "review_settings")
        self._closed_roots.discard(root_id)
        if not await self._ensure_lease(root):
            current = self._root_status.get(
                root_id, ("passive", "open_active_process", None)
            )
            return NotesSyncControlResult(False, current[0], current[1])
        try:
            plan = await self._review_candidate(root)
        except Exception:
            self._blocked_roots.add(root_id)
            await self._publish(root_id, "failed", "review_changes")
            return NotesSyncControlResult(False, "failed", "review_changes")
        blocked = self._blocked_plan_status(plan)
        executable = tuple(
            action for action in plan.safe_actions if action.kind in _EXECUTABLE_ACTIONS
        )
        if blocked is not None or executable:
            self._reviews[root_id] = plan
            self._blocked_roots.add(root_id)
            await self._publish(root_id, "changes_available", "review_changes")
            return NotesSyncControlResult(False, "changes_available", "review_changes")
        active = await asyncio.to_thread(
            self._store.transition_root, root_id, NotesSyncRootState.ACTIVE
        )
        self._blocked_roots.discard(root_id)
        await self._publish(active.root_id, "up_to_date", "sync_now")
        self._start_watcher()
        return NotesSyncControlResult(True, "up_to_date", "sync_now")

    async def activate_root(
        self, root_id: str, authorization: object
    ) -> NotesSyncControlResult:
        task = self._admit_task(root_id)
        try:
            return await self._activate_root(root_id, authorization)
        finally:
            self._finish_task(root_id, task)

    async def _activate_root(
        self, root_id: str, authorization: object
    ) -> NotesSyncControlResult:
        self._require_cutover(root_id)
        token = (
            authorization
            if type(authorization) is str
            else getattr(authorization, "observation_token", None)
        )
        if type(token) is not str:
            raise ValueError("current_review_required")
        setup_review = self._setup_reviews.get(root_id)
        if setup_review is not None:
            setup = setup_review.setup
            reviewed = setup_review.plan
            root = NotesSyncRootRecord(
                root_id=root_id,
                note_scope_id=setup.note_scope_id,
                logical_folder_id=None,
                canonical_path=setup.canonical_path,
                direction=setup.direction,
                state=NotesSyncRootState.PENDING,
            )
        else:
            setup = None
            reviewed = self._reviews.get(root_id)
            root = await asyncio.to_thread(self._store.get_root, root_id)
            if not (
                root.state is NotesSyncRootState.PAUSED
                and root.last_status_code == "migration_review_required"
            ):
                raise ValueError("activation_review_required")
        if reviewed is None or reviewed.observation_token != token:
            raise ValueError("stale_review")
        if not await self._ensure_lease(root):
            return NotesSyncControlResult(False, "passive", "open_active_process")
        fresh = await self._review_candidate(root)
        if fresh != reviewed:
            raise ValueError("stale_review")
        if self._blocked_plan_status(fresh) is not None:
            return NotesSyncControlResult(False, "needs_attention", "review_changes")
        display_name = (
            setup.display_name
            if setup is not None
            else f"Migrated notes {hashlib.sha256(root_id.encode()).hexdigest()[:8]}"
        )
        persisted = setup is None
        attached = root.logical_folder_id is not None
        folder_receipt: object | None = None
        logical_folder_id: str | None = root.logical_folder_id
        try:
            if setup is not None:
                root = await asyncio.to_thread(self._store.create_root, root)
                persisted = True
            logical_folder_id, folder_receipt = await self._adapter.create_root_folder(
                display_name
            )
            validate_notes_sync_opaque_id(
                logical_folder_id, field_name="logical_folder_id"
            )
            await asyncio.to_thread(
                self._store.assign_root_folder, root_id, logical_folder_id
            )
            attached = True
            planned_root = replace(root, logical_folder_id=logical_folder_id)
            authority = await self._fresh_authority(planned_root)
            if authority.plan != reviewed:
                raise ValueError("stale_review")
            selected = tuple(
                action
                for action in authority.plan.safe_actions
                if action.kind in _EXECUTABLE_ACTIONS
            )
            if setup is not None:
                active_root = await asyncio.to_thread(
                    self._store.transition_root,
                    root_id,
                    NotesSyncRootState.ACTIVE,
                )
            else:
                candidate_ids = tuple(
                    sorted(
                        binding.binding_id
                        for binding in await asyncio.to_thread(
                            self._store.list_bindings, root_id
                        )
                        if binding.state is NotesSyncBindingState.CANDIDATE
                    )
                )
                reviewed_binding_ids = {
                    action.binding_id
                    for action in authority.plan.safe_actions
                    if action.binding_id is not None
                }
                if not set(candidate_ids) <= reviewed_binding_ids:
                    raise ValueError("stale_review")
                active_root = await asyncio.to_thread(
                    self._store.activate_migration_candidate,
                    root_id,
                    logical_folder_id,
                    candidate_ids,
                )
            persisted = True
        except Exception as error:
            if (
                setup is not None
                and persisted
                and folder_receipt is None
                and not attached
            ):
                await self._retire_failed_setup(root_id)
                return NotesSyncControlResult(False, "failed", "review_settings")
            if folder_receipt is not None and not attached:
                try:
                    await self._adapter.rollback_root_folder(folder_receipt)
                except Exception:
                    if persisted and logical_folder_id is not None:
                        await asyncio.to_thread(
                            self._store.record_root_activation_recovery,
                            root_id,
                            logical_folder_id,
                        )
                    await self._publish(
                        root_id,
                        "needs_attention",
                        "review_settings",
                        persist=False,
                    )
                    return NotesSyncControlResult(
                        False, "needs_attention", "review_settings"
                    )
                if setup is not None and persisted:
                    await self._retire_failed_setup(root_id)
                    return NotesSyncControlResult(False, "failed", "review_settings")
            if attached and persisted and logical_folder_id is not None:
                await asyncio.to_thread(
                    self._store.record_root_activation_recovery,
                    root_id,
                    logical_folder_id,
                )
                self._setup_reviews.pop(root_id, None)
                self._reviews.pop(root_id, None)
                self._blocked_roots.add(root_id)
                self._durably_blocked_roots.add(root_id)
                await self._publish(
                    root_id,
                    "needs_attention",
                    "review_settings",
                    persist=False,
                )
                return NotesSyncControlResult(
                    False, "needs_attention", "review_settings"
                )
            if isinstance(error, ValueError):
                raise
            if persisted:
                await self._publish(root_id, "failed", "review_changes")
                return NotesSyncControlResult(False, "failed", "review_changes")
            raise RuntimeError("root_activation_persistence_failed") from None
        self._setup_reviews.pop(root_id, None)
        self._reviews.pop(root_id, None)
        if selected:
            try:
                results = await self._execute(active_root, authority, selected)
            except Exception:
                return NotesSyncControlResult(False, "failed", "review_changes")
            applied_count = sum(
                getattr(result, "state", None) is NotesSyncOperationState.COMPLETED
                for result in results
            )
            if len(results) != len(selected):
                await self._publish(root_id, "partial", "review_changes")
                return NotesSyncControlResult(
                    False, "partial", "review_changes", applied_count
                )
            if any(
                getattr(result, "state", None) is not NotesSyncOperationState.COMPLETED
                for result in results
            ):
                current = self._root_status.get(
                    root_id, ("needs_attention", "review_changes", None)
                )
                return NotesSyncControlResult(
                    False, current[0], current[1], applied_count
                )
        else:
            applied_count = 0
            await self._publish(root_id, "up_to_date", "sync_now")
        self._start_watcher()
        return NotesSyncControlResult(True, "up_to_date", "sync_now", applied_count)

    async def retarget_root(
        self, root_id: str, *_args: object
    ) -> NotesSyncControlResult:
        return await self._blocked_control(root_id)

    async def disconnect_root(
        self, root_id: str, *_args: object
    ) -> NotesSyncControlResult:
        return await self._blocked_control(root_id)

    async def _blocked_control(self, root_id: str) -> NotesSyncControlResult:
        self._require_cutover(root_id)
        await self._publish(root_id, "needs_attention", "review_settings")
        return NotesSyncControlResult(False, "needs_attention", "review_settings")

    def _require_cutover(self, root_id: str) -> None:
        validate_notes_sync_opaque_id(root_id, field_name="root_id")
        if not self._admission_open:
            raise RuntimeError("notes_sync_cutover_not_admitted")

    def _require_authority(self, root_id: str, operation: str) -> object:
        admission = self._admissions.get(root_id)
        if admission is None:
            raise RuntimeError("root_lease_required")
        try:
            return admission.require_authority(operation)
        except Exception:
            raise RuntimeError("root_lease_required") from None

    def _admit_task(self, root_id: str) -> asyncio.Task[object]:
        self._require_cutover(root_id)
        if root_id in self._closed_roots:
            raise RuntimeError("root_admission_closed")
        return self._register_task(root_id)

    def _register_task(self, root_id: str) -> asyncio.Task[object]:
        """Make this coroutine's store work visible to settle()/shutdown.

        task-21101 review round: pause_root, resume_root, and review_setup
        touch the held-connection store but were invisible to settle(), so
        _shutdown_once could close the store while their pool thread held a
        checked-out connection. Unlike _admit_task, registration does not
        gate on _closed_roots -- resume_root must run on a closed root, and
        pause_root closes its own root as it starts.
        """

        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("runtime_task_required")
        self._active_tasks.setdefault(root_id, set()).add(task)
        return task

    def _finish_task(self, root_id: str, task: asyncio.Task[object]) -> None:
        tasks = self._active_tasks.get(root_id)
        if tasks is None:
            return
        tasks.discard(task)
        if not tasks:
            self._active_tasks.pop(root_id, None)

    async def _settle_root(self, root_id: str) -> None:
        current = asyncio.current_task()
        tasks = tuple(
            task
            for task in (
                *self._active_tasks.get(root_id, ()),
                self._hint_tasks.get(root_id),
            )
            if task is not None and task is not current
        )
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _publish(
        self,
        root_id: str,
        status: str,
        next_action: str,
        *,
        action_id: str | None = None,
        persist: bool = True,
    ) -> None:
        if persist:
            await asyncio.to_thread(self._store.update_root_status, root_id, status)
        self._root_status[root_id] = NotesSyncRootRuntimeSnapshot(
            root_id, status, next_action, action_id
        )

    async def shutdown(self) -> None:
        """Close admission, stop hints, settle work, then release leases once."""

        self._closing = True
        self._admission_open = False
        if self._shutdown_task is None:
            self._shutdown_task = asyncio.create_task(
                self._shutdown_once(), name="notes_sync_runtime_shutdown"
            )
        shutdown_task = self._shutdown_task
        cancellation: asyncio.CancelledError | None = None
        while True:
            try:
                await asyncio.shield(shutdown_task)
            except asyncio.CancelledError as error:
                if shutdown_task.done():
                    await shutdown_task
                cancellation = cancellation or error
                continue
            break
        if self._status == "failed" and self._shutdown_task is shutdown_task:
            self._shutdown_task = None
        if cancellation is not None:
            raise cancellation

    async def _shutdown_once(self) -> None:
        self._admission_open = False
        self._status = "stopping"
        self._next_action = "wait"
        if self._watcher is not None:
            await self._watcher.stop()
        if self._watcher_task is not None:
            await asyncio.gather(self._watcher_task, return_exceptions=True)
        if (
            self._start_task is not None
            and self._start_task is not asyncio.current_task()
        ):
            await asyncio.gather(self._start_task, return_exceptions=True)
        await self.settle()
        close_adapter = getattr(self._adapter, "close", None)
        if callable(close_adapter):
            try:
                close_adapter()
            except Exception:
                pass
        close_failed = False
        if self._coordinator is not None:
            for root_id, lease in tuple(self._leases.items()):
                try:
                    await asyncio.to_thread(
                        self._coordinator.close_admission, lease, lambda: None
                    )
                except Exception:
                    close_failed = True
                    continue
                self._leases.pop(root_id, None)
                self._admissions.pop(root_id, None)
        # Release the store's held per-thread connections (task-21101); the
        # store transparently re-opens if a later start() needs it again.
        # getattr-guarded like close_adapter above: tests supply fake stores.
        close_store = getattr(self._store, "close", None)
        if callable(close_store):
            try:
                await asyncio.to_thread(close_store)
            except Exception:
                pass
        if close_failed:
            self._status = "failed"
            self._next_action = "review_settings"
            return
        self._status = "stopped"
        self._next_action = "none"


def build_notes_sync_runtime_owner(
    *,
    notes_scope_service: object,
    cutover_admitted: bool,
    profile_process_is_sole: bool,
    database_path: Path | str,
    migrate_legacy: Callable[[], object] | None = None,
    adapter: _RuntimeAdapter | None = None,
    coordinator: object | Callable[[], object] | None = None,
    watcher_factory: Callable[[Callable[[str], object]], object] | None = None,
    file_notes_binding: Callable[[], object | None] | None = None,
    local_user_id: str | None = None,
    recovery_capacity_bytes: int | None = None,
) -> NotesSyncRuntimeOwner:
    """Build the application-owned lasting-sync runtime."""

    if notes_scope_service is None:
        raise ValueError("notes_scope_service is required.")
    if migrate_legacy is None:
        raise ValueError("the idempotent migrate_legacy callable is required.")
    path = Path(database_path)
    store = NotesDeviceStateStore(path)
    if adapter is None:
        if not isinstance(notes_scope_service, NotesScopeService):
            raise TypeError("notes_scope_service must be a NotesScopeService.")
        if type(local_user_id) is not str or not local_user_id:
            raise ValueError("local_user_id is required for lasting sync.")
        if type(recovery_capacity_bytes) is not int or recovery_capacity_bytes <= 0:
            raise ValueError("recovery_capacity_bytes must be positive.")
        adapter = _ProductionRuntimeAdapter(
            store,
            notes_scope_service,
            local_user_id=local_user_id,
            recovery_capacity_bytes=recovery_capacity_bytes,
        )
    selected_coordinator = coordinator or (
        lambda: NotesSyncRootCoordinator(path.parent / "notes_sync_locks")
    )
    owner_holder: dict[str, NotesSyncRuntimeOwner] = {}
    selected_watcher_factory = watcher_factory or (
        lambda schedule: PollingNotesSyncWatcher(
            lambda: owner_holder["owner"]._changed_root_ids(), schedule
        )
    )
    owner = NotesSyncRuntimeOwner(
        store=store,
        migrate_legacy=migrate_legacy,
        coordinator=selected_coordinator,
        adapter=adapter,
        watcher_factory=selected_watcher_factory,
        file_notes_binding=file_notes_binding or (lambda: None),
        cutover_admitted=cutover_admitted,
        profile_process_is_sole=profile_process_is_sole,
    )
    owner_holder["owner"] = owner
    return owner


def build_notes_sync_legacy_migrator(
    *,
    database_path: Path | str,
    legacy_connection: Callable[[], sqlite3.Connection],
    settings: Mapping[str, object],
    note_scope_id: str,
    file_notes_binding: Callable[[], object | None],
    private_paths: Iterable[Path | str],
) -> Callable[[], object]:
    """Compose TASK-19008's exact idempotent migration for app startup."""

    store = NotesDeviceStateStore(database_path)
    selected_private_paths = tuple(private_paths)

    def migrate() -> object:
        binding = file_notes_binding()
        root_key = getattr(binding, "root_key", None)
        file_notes_roots = (root_key,) if isinstance(root_key, str) else ()
        snapshot = snapshot_legacy_notes_sync(
            legacy_connection(),
            settings,
            note_scope_id=note_scope_id,
            file_notes_roots=file_notes_roots,
            private_paths=selected_private_paths,
        )
        plan = plan_legacy_notes_sync_migration(snapshot)
        return persist_legacy_notes_sync_migration(store, plan)

    return migrate


__all__ = [
    "CUTOVER_MARKER",
    "NotesSyncControlResult",
    "NotesSyncRootRuntimeSnapshot",
    "NotesSyncRootSetup",
    "NotesSyncRuntimeOwner",
    "NotesSyncRuntimeSnapshot",
    "build_notes_sync_legacy_migrator",
    "build_notes_sync_runtime_owner",
]
