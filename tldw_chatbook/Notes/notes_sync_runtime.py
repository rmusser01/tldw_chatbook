"""Application-owned, cutover-gated runtime for lasting Database Notes sync."""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
import time
import weakref
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, cast
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
    NotesSyncResolutionHistoryRecord,
    NotesSyncRootRecord,
    NotesSyncStoreSetting,
)
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService, ScopeType
from tldw_chatbook.Notes.notes_sync_conflicts import (
    ConflictApplyResult,
    ConflictComparison,
    ConflictHistoryRow,
    ConflictReceipt,
    ConflictSelection,
    NotesSyncConflictChoice,
    build_conflict_comparison,
    conflict_copies_folder_id,
    conflict_copy_note_id,
    conflict_resolution_operation_id,
    conflict_root_folder_id,
    eligible_conflict_reason,
)
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
    normalize_notes_sync_relative_path,
    validate_notes_sync_opaque_id,
)
from tldw_chatbook.Notes.notes_sync_executor import (
    CONFLICT_RECOVERY_RETENTION_NS,
    NotesSyncDirectionOverride,
    NotesSyncExecutionRequest,
    NotesSyncExecutionResult,
    NotesSyncExecutor,
    NotesSyncKeepBothAuthority,
    NotesSyncUndoProjection,
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
    ReconciliationAttention,
    ReconciliationAttentionKind,
    ReconciliationInput,
    ReconciliationPlan,
    _observation_token,
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
_DISPLAY_LABEL_MAX_CHARS = 160
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
        "not_configured",
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
class RuntimeConflictReceipt:
    """Fresh, bounded display projection for one active receipt."""

    operation_id: str
    item_label: str
    choice: NotesSyncConflictChoice
    state: str
    undo_available: bool
    undo_reason: str | None = None

    def __post_init__(self) -> None:
        ConflictReceipt(
            self.operation_id,
            self.choice,
            self.state,
            self.undo_available,
            self.undo_reason,
        )
        _validate_item_label(self.item_label)

    def __repr__(self) -> str:
        return "RuntimeConflictReceipt(<private>)"


@dataclass(frozen=True, slots=True, repr=False)
class RuntimeConflictHistoryRow:
    """Fresh, bounded display projection for one durable history row."""

    operation_id: str
    item_label: str
    choice: NotesSyncConflictChoice
    state: str
    completed_at: str | None
    updated_at: str
    undo_available: bool
    undo_reason: str | None = None

    def __post_init__(self) -> None:
        ConflictHistoryRow(
            self.operation_id,
            self.choice,
            self.state,
            self.completed_at,
            self.updated_at,
            self.undo_available,
            self.undo_reason,
        )
        _validate_item_label(self.item_label)

    def __repr__(self) -> str:
        return "RuntimeConflictHistoryRow(<private>)"


def _validate_item_label(value: str) -> None:
    if (
        type(value) is not str
        or not value
        or len(value) > _DISPLAY_LABEL_MAX_CHARS
        or "\n" in value
        or "\r" in value
    ):
        raise ValueError("item_label must be bounded single-line text")


def _ignore_operation_stage(_state: NotesSyncOperationState) -> None:
    return None


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

    async def build_conflict_execution_request(
        self,
        root: NotesSyncRootRecord,
        observations: ReconciliationInput,
        plan: ReconciliationPlan,
        selection: ConflictSelection,
    ) -> object: ...

    def executor_for(
        self,
        root: NotesSyncRootRecord,
        *,
        after_stage: Callable[[NotesSyncOperationState], None],
    ) -> object: ...

    async def create_root_folder(self, display_name: str) -> tuple[str, object]: ...

    async def rollback_root_folder(self, receipt: object) -> None: ...

    async def build_conflict_comparison(
        self,
        root: NotesSyncRootRecord,
        plan: ReconciliationPlan,
        binding_id: str,
    ) -> ConflictComparison: ...


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

    async def build_conflict_execution_request(
        self,
        root: NotesSyncRootRecord,
        observations: ReconciliationInput,
        plan: ReconciliationPlan,
        selection: ConflictSelection,
    ) -> NotesSyncExecutionRequest:
        """Build one occurrence-only reviewed conflict request."""

        action_kind = {
            NotesSyncConflictChoice.KEEP_FILE: NotesSyncActionKind.UPDATE_NOTE,
            NotesSyncConflictChoice.KEEP_NOTE: NotesSyncActionKind.UPDATE_FILE,
            NotesSyncConflictChoice.KEEP_BOTH: NotesSyncActionKind.UPDATE_NOTE,
        }.get(selection.choice)
        if action_kind is None:
            raise ValueError("conflict_choice_not_executable")
        operation_id = conflict_resolution_operation_id(
            root.root_id,
            selection.binding_id,
            plan.observation_token,
            selection.choice,
        )
        action = NotesSyncAction(
            action_id=operation_id,
            kind=action_kind,
            binding_id=selection.binding_id,
            reason_code="reviewed_conflict_resolution",
        )
        request = await self.build_execution_request(root, observations, plan, action)
        keep_both: NotesSyncKeepBothAuthority | None = None
        if selection.choice is NotesSyncConflictChoice.KEEP_BOTH:
            binding = self._bundles.get(plan.observation_token, {}).get(
                selection.binding_id
            )
            if binding is None or binding.note is None:
                raise RuntimeError("private_conflict_authority_missing")
            if root.logical_folder_id is None:
                raise RuntimeError("folder_owner_missing")
            logical_folder = await self._service.get_note_folder_by_id_for_sync(
                scope=ScopeType.LOCAL_NOTE,
                folder_id=root.logical_folder_id,
                include_deleted=True,
                user_id=self._user_id,
            )
            if logical_folder is None or logical_folder.deleted:
                raise RuntimeError("folder_owner_missing")
            keep_both = NotesSyncKeepBothAuthority(
                parent_folder_id=conflict_copies_folder_id(root.note_scope_id),
                parent_folder_name="Conflict copies",
                root_folder_id=conflict_root_folder_id(
                    root.note_scope_id, root.root_id
                ),
                root_folder_name=logical_folder.name,
                copy_note_id=conflict_copy_note_id(
                    root.root_id,
                    selection.binding_id,
                    plan.observation_token,
                ),
                copy_title=binding.note.title,
            )
        override_needed = (
            action_kind is NotesSyncActionKind.UPDATE_NOTE
            and root.direction is NotesSyncDirection.NOTES_TO_FOLDER
        ) or (
            action_kind is NotesSyncActionKind.UPDATE_FILE
            and root.direction is NotesSyncDirection.FOLDER_TO_NOTES
        )
        return replace(
            request,
            operation_id=operation_id,
            recovery_id=f"recovery-{operation_id}",
            recovery_expires_at=time.time_ns() + CONFLICT_RECOVERY_RETENTION_NS,
            journal_kind=f"resolve_{selection.choice.value}",
            keep_both=keep_both,
            direction_override=(
                NotesSyncDirectionOverride(
                    review_id=operation_id,
                    action_kind=action_kind,
                    observation_token=plan.observation_token,
                )
                if override_needed
                else None
            ),
        )

    def release_observation(self, observation_token: str) -> None:
        self._bundles.pop(observation_token, None)

    async def build_conflict_comparison(
        self,
        root: NotesSyncRootRecord,
        plan: ReconciliationPlan,
        binding_id: str,
    ) -> ConflictComparison:
        """Project one live private binding into a bounded comparison."""

        binding = self._bundles.get(plan.observation_token, {}).get(binding_id)
        if binding is None or binding.note is None or binding.file is None:
            raise RuntimeError("private_comparison_authority_missing")
        if binding.record.root_id != root.root_id:
            raise RuntimeError("comparison_root_mismatch")
        return build_conflict_comparison(
            binding_id=binding_id,
            title=binding.note.title,
            relative_path=binding.file.observation.relative_path,
            note_text=binding.note.content,
            file_text=binding.file.text,
            note_version=binding.note.version,
            note_updated_at=binding.note.updated_at,
            file_modified_ns=binding.file.reviewed_state.mtime_ns,
        )

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
        start_evidence: Callable[[], bool] | None = None,
    ) -> None:
        if type(cutover_admitted) is not bool:
            raise TypeError("cutover_admitted must be a private boolean.")
        if type(profile_process_is_sole) is not bool:
            raise TypeError("profile_process_is_sole must be a private boolean.")
        if not callable(migrate_legacy) or not callable(watcher_factory):
            raise TypeError("runtime factories must be callable.")
        if start_evidence is not None and not callable(start_evidence):
            raise TypeError("start_evidence must be callable when provided.")
        self._store = store
        self._migrate_legacy = migrate_legacy
        self._coordinator_source = coordinator
        self._adapter = adapter
        self._watcher_factory = watcher_factory
        self._file_notes_binding = file_notes_binding
        self._cutover_admitted = cutover_admitted
        self._profile_process_is_sole = profile_process_is_sole
        self._start_evidence = start_evidence
        self._start_deferred = False
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
        self._mutation_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )
        self._active_receipts: dict[
            str,
            OrderedDict[
                str,
                tuple[str, NotesSyncConflictChoice],
            ],
        ] = {}
        self._status = "starting"
        self._next_action = "wait"
        self._admission_open = False
        self._closing = False

    def _mutation_lock(self, root_id: str) -> asyncio.Lock:
        """Return the one live in-process mutation lock for a root."""

        lock = self._mutation_locks.get(root_id)
        if lock is None:
            lock = asyncio.Lock()
            self._mutation_locks[root_id] = lock
        return lock

    def snapshot(self) -> NotesSyncRuntimeSnapshot:
        """Return the current path-free UI projection."""

        return NotesSyncRuntimeSnapshot(
            self._status,
            self._next_action,
            tuple(self._root_status[key] for key in sorted(self._root_status)),
        )

    def _remember_conflict_receipt(
        self,
        root_id: str,
        binding_id: str,
        operation_id: str,
        choice: NotesSyncConflictChoice,
    ) -> None:
        """Retain one current-runtime receipt, superseding the same item."""

        validate_notes_sync_opaque_id(root_id, field_name="root_id")
        validate_notes_sync_opaque_id(binding_id, field_name="binding_id")
        validate_notes_sync_opaque_id(operation_id, field_name="operation_id")
        if (
            type(choice) is not NotesSyncConflictChoice
            or choice is NotesSyncConflictChoice.SKIP
        ):
            raise TypeError("choice must be a mutating NotesSyncConflictChoice")
        receipts = self._active_receipts.setdefault(root_id, OrderedDict())
        for prior_id, (prior_binding, _prior_choice) in tuple(receipts.items()):
            if prior_binding == binding_id and prior_id != operation_id:
                receipts.pop(prior_id)
        receipts[operation_id] = (binding_id, choice)
        receipts.move_to_end(operation_id)
        while len(receipts) > 100:
            receipts.popitem(last=False)

    async def active_conflict_receipts(
        self, root_id: str
    ) -> tuple[RuntimeConflictReceipt, ...]:
        """Return fresh bounded receipts retained by this process."""

        validate_notes_sync_opaque_id(root_id, field_name="root_id")
        retained = tuple(self._active_receipts.get(root_id, OrderedDict()).items())
        if not retained:
            return ()
        try:
            root = await asyncio.to_thread(self._store.get_root, root_id)
            executor = cast(
                NotesSyncExecutor,
                self._adapter.executor_for(root, after_stage=_ignore_operation_stage),
            )
        except Exception:
            executor = None
        projected: list[RuntimeConflictReceipt] = []
        for operation_id, retained_value in retained:
            projection = await self._inspect_resolution_undo(
                executor, root_id, operation_id
            )
            current = self._active_receipts.get(root_id)
            if current is None or current.get(operation_id) != retained_value:
                continue
            if projection.state == "undone" or projection.undo_reason == "Undone":
                current.pop(operation_id, None)
                continue
            _binding_id, choice = retained_value
            projected.append(
                RuntimeConflictReceipt(
                    operation_id=operation_id,
                    item_label=self._resolution_item_label(operation_id, projection),
                    choice=choice,
                    state=projection.state,
                    undo_available=projection.undo_available,
                    undo_reason=projection.undo_reason,
                )
            )
        return tuple(projected)

    def dismiss_conflict_receipt(self, root_id: str, operation_id: str) -> None:
        """Dismiss one process-local receipt without changing durable history."""

        validate_notes_sync_opaque_id(root_id, field_name="root_id")
        validate_notes_sync_opaque_id(operation_id, field_name="operation_id")
        receipts = self._active_receipts.get(root_id)
        if receipts is not None:
            receipts.pop(operation_id, None)

    async def undo_resolution(
        self,
        root_id: str,
        source_operation_id: str,
    ) -> NotesSyncExecutionResult:
        """Run one durable linked Undo under the root mutation authority."""

        validate_notes_sync_opaque_id(root_id, field_name="root_id")
        validate_notes_sync_opaque_id(
            source_operation_id, field_name="source_operation_id"
        )
        task = self._admit_task(root_id)
        try:
            async with self._mutation_lock(root_id):
                root = await asyncio.to_thread(self._store.get_root, root_id)
                if root.root_id != root_id:
                    raise RuntimeError("root_authority_mismatch")
                if root.state is not NotesSyncRootState.ACTIVE:
                    raise RuntimeError("sync_root_not_active")
                self._require_authority(root_id, "write")

                def require_write(_state: object) -> None:
                    self._require_authority(root_id, "write")

                executor = cast(
                    NotesSyncExecutor,
                    self._adapter.executor_for(
                        root,
                        after_stage=require_write,
                    ),
                )
                result = await executor.undo_resolution(root_id, source_operation_id)
                self._require_authority(root_id, "write")
                if type(result) is not NotesSyncExecutionResult:
                    raise RuntimeError("invalid_execution_result")
                if result.state is NotesSyncOperationState.COMPLETED:
                    self.dismiss_conflict_receipt(root_id, source_operation_id)
                    fresh = await self._fresh_authority(root)
                    self._reviews[root_id] = fresh.plan
                return result
        finally:
            self._finish_task(root_id, task)

    async def resolution_history(
        self,
        root_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        now: int | None = None,
    ) -> tuple[RuntimeConflictHistoryRow, ...]:
        """Return one durable page decorated from fresh private authority."""

        current_time = time.time_ns() if now is None else now
        records = await asyncio.to_thread(
            self._store.list_resolution_history,
            root_id,
            limit=limit,
            offset=offset,
            now=current_time,
        )
        choices = {
            "resolve_keep_file": NotesSyncConflictChoice.KEEP_FILE,
            "resolve_keep_note": NotesSyncConflictChoice.KEEP_NOTE,
            "resolve_keep_both": NotesSyncConflictChoice.KEEP_BOTH,
        }
        try:
            root = await asyncio.to_thread(self._store.get_root, root_id)
            executor = cast(
                NotesSyncExecutor,
                self._adapter.executor_for(root, after_stage=_ignore_operation_stage),
            )
        except Exception:
            executor = None
        projected: list[RuntimeConflictHistoryRow] = []
        for record in records:
            fallback = self._history_fallback_projection(record, current_time)
            projection = await self._inspect_resolution_undo(
                executor,
                root_id,
                record.operation_id,
                now=current_time,
                fallback=fallback,
            )
            projected.append(
                RuntimeConflictHistoryRow(
                    operation_id=record.operation_id,
                    item_label=self._resolution_item_label(
                        record.operation_id, projection
                    ),
                    choice=choices[record.kind],
                    state=projection.state,
                    completed_at=(
                        self._history_timestamp(record.completed_at)
                        if record.completed_at is not None
                        else None
                    ),
                    updated_at=self._history_timestamp(record.updated_at),
                    undo_available=projection.undo_available,
                    undo_reason=projection.undo_reason,
                )
            )
        return tuple(projected)

    async def _inspect_resolution_undo(
        self,
        executor: NotesSyncExecutor | None,
        root_id: str,
        operation_id: str,
        *,
        now: int | None = None,
        fallback: NotesSyncUndoProjection | None = None,
    ) -> NotesSyncUndoProjection:
        if executor is not None:
            try:
                projection = await executor.inspect_resolution_undo(
                    root_id, operation_id, now=now
                )
                if isinstance(projection, NotesSyncUndoProjection):
                    return projection
                return NotesSyncUndoProjection(
                    projection.undo_available,
                    projection.undo_reason,
                    projection.state,
                    projection.note_title,
                    projection.relative_path,
                )
            except Exception:
                pass
        return fallback or NotesSyncUndoProjection(
            False,
            "Unavailable",
            "unavailable",
            None,
            None,
        )

    @staticmethod
    def _history_fallback_projection(
        record: NotesSyncResolutionHistoryRecord, current_time: int
    ) -> NotesSyncUndoProjection:
        if record.undone:
            return NotesSyncUndoProjection(False, "Undone", "undone", None, None)
        if record.undo_state is not None:
            return NotesSyncUndoProjection(
                False,
                "Changed since resolution",
                record.undo_state.value,
                None,
                None,
            )
        if (
            record.recovery_expires_at is None
            or record.recovery_expires_at <= current_time
        ):
            return NotesSyncUndoProjection(
                False, "Undo expired", record.state.value, None, None
            )
        return NotesSyncUndoProjection(
            False,
            "Unavailable",
            record.state.value,
            None,
            None,
        )

    @staticmethod
    def _resolution_item_label(
        operation_id: str, projection: NotesSyncUndoProjection
    ) -> str:
        title = (
            " ".join(projection.note_title.split())
            if type(projection.note_title) is str
            else ""
        )
        try:
            relative_path = (
                normalize_notes_sync_relative_path(projection.relative_path)
                if projection.relative_path is not None
                else ""
            )
        except (TypeError, ValueError):
            relative_path = ""
        label = " — ".join(value for value in (title, relative_path) if value)
        return label[:_DISPLAY_LABEL_MAX_CHARS] or operation_id[:8]

    @staticmethod
    def _history_timestamp(value: int) -> str:
        return datetime.fromtimestamp(value / 1_000_000_000, UTC).isoformat()

    async def start(self, *, force: bool = False) -> None:
        """Initialize once and remain inert unless both cutover gates match.

        TASK-21112: when a ``start_evidence`` probe was supplied and reports
        no configured lasting sync, the start defers inert (status
        ``not_configured``) without opening — or creating — the state
        database. ``force=True`` re-arms exactly one full start after such a
        deferral; it is how first-time setup brings the machinery up at
        runtime. A completed non-deferred start is never re-run.
        """

        while True:
            if self._start_task is None:
                self._start_task = asyncio.create_task(
                    self._start_once(force=force), name="notes_sync_runtime_start"
                )
            task = self._start_task
            await asyncio.shield(task)
            if not force or self._closing:
                return
            if self._start_deferred:
                # Deferred boot start: re-arm one full start.
                if self._start_task is task:
                    self._start_task = None
                continue
            if self._start_task is not task:
                # A concurrent forced caller re-armed; await its start too.
                continue
            return

    async def _start_once(self, *, force: bool = False) -> None:
        self._start_deferred = False
        if not force and self._start_evidence is not None:
            try:
                configured = bool(await asyncio.to_thread(self._start_evidence))
            except Exception:
                # Fail open: a broken probe must never silently disable a
                # configured user's sync. Starting is the safe direction.
                configured = True
            if not configured:
                self._start_deferred = True
                self._status = "not_configured"
                self._next_action = "none"
                return
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
        if automatic:
            lock = self._mutation_lock(root.root_id)
            async with lock:
                return await self._reconcile_locked(root, automatic=True)
        return await self._reconcile_locked(root, automatic=False)

    async def _reconcile_locked(
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
            await self._execute_locked(root, authority, selected)
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
        # TASK-21112: setting up the first root is explicit feature use, so a
        # boot-deferred (unconfigured) runtime is brought up here on demand.
        # For an already-started runtime this awaits the memoized start task
        # and has no side effects.
        await self.start(force=True)
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

    async def compare_conflict(
        self,
        root_id: str,
        observation_token: str,
        binding_id: str,
    ) -> ConflictComparison:
        """Return one bounded comparison under exact fresh planning authority."""

        task = self._admit_task(root_id)
        plan: ReconciliationPlan | None = None
        observed_token: str | None = None
        try:
            reviewed = self._reviews.get(root_id)
            if reviewed is None or reviewed.observation_token != observation_token:
                raise ValueError("stale_review")
            root = await asyncio.to_thread(self._store.get_root, root_id)
            if root.root_id != root_id:
                raise RuntimeError("root_authority_mismatch")
            if root.state is not NotesSyncRootState.ACTIVE:
                raise RuntimeError("sync_root_not_active")
            self._require_authority(root_id, "plan")
            observations = await self._adapter.observe_root(root)
            observed_token = _observation_token(observations)
            plan = plan_reconciliation(observations)
            self._require_authority(root_id, "plan")
            if observations.root_id != root_id or plan.root_id != root_id:
                raise RuntimeError("root_observation_mismatch")
            if observations.direction is not root.direction:
                raise RuntimeError("root_direction_changed")
            if plan.observation_token != observation_token:
                raise ValueError("stale_review")
            if plan != reviewed:
                raise ValueError("stale_review")
            managed = {effect.binding_id for effect in plan.managed_placement_effects}
            matches = tuple(
                attention
                for attention in plan.attention
                if attention.kind is ReconciliationAttentionKind.CONFLICT
                and attention.binding_id == binding_id
                and eligible_conflict_reason(
                    attention.reason_code,
                    managed=binding_id in managed,
                )
            )
            if len(matches) != 1:
                raise ValueError("comparison_binding_mismatch")
            return await self._adapter.build_conflict_comparison(
                root,
                plan,
                binding_id,
            )
        finally:
            if observed_token is not None:
                release = getattr(self._adapter, "release_observation", None)
                if callable(release):
                    release(observed_token)
            self._finish_task(root_id, task)

    async def apply_reviewed(
        self,
        root_id: str,
        observation_token: str,
        safe_action_ids: tuple[str, ...],
        selections: tuple[ConflictSelection, ...] = (),
    ) -> ConflictApplyResult:
        """Apply safe work plus an exact reviewed conflict subset."""

        task = self._admit_task(root_id)
        try:
            reviewed = self._reviews.get(root_id)
            if reviewed is None or reviewed.observation_token != observation_token:
                raise ValueError("stale_review")
            if type(safe_action_ids) is not tuple or any(
                type(action_id) is not str for action_id in safe_action_ids
            ):
                raise TypeError("safe_action_ids must be a tuple of strings")
            if len(set(safe_action_ids)) != len(safe_action_ids):
                raise ValueError("reviewed_action_selection_duplicate")
            if type(selections) is not tuple or any(
                type(selection) is not ConflictSelection for selection in selections
            ):
                raise TypeError("selections must be a tuple of ConflictSelection")
            selection_ids = tuple(selection.binding_id for selection in selections)
            if len(set(selection_ids)) != len(selection_ids):
                raise ValueError("conflict_selection_duplicate")
            lock = self._mutation_lock(root_id)
            async with lock:
                root = await asyncio.to_thread(self._store.get_root, root_id)
                if root.root_id != root_id:
                    raise RuntimeError("root_authority_mismatch")
                if root.state is not NotesSyncRootState.ACTIVE:
                    await self._publish(root_id, "paused", "resume_sync")
                    raise RuntimeError("sync_root_not_active")
                if (
                    root.last_status_code
                    in {
                        "activation_recovery_required",
                        "migration_review_required",
                    }
                    or root_id in self._setup_reviews
                ):
                    raise ValueError("review_not_executable")
                self._require_authority(root_id, "plan")
                observations = await self._adapter.observe_root(root)
                observed_token = _observation_token(observations)
                plan: ReconciliationPlan | None = None
                try:
                    plan = plan_reconciliation(observations)
                    self._require_authority(root_id, "plan")
                    if observations.root_id != root_id or plan.root_id != root_id:
                        raise RuntimeError("root_observation_mismatch")
                    if observations.direction is not root.direction:
                        raise RuntimeError("root_direction_changed")
                    if plan.observation_token != observation_token:
                        raise ValueError("stale_review")
                    assert_review_current(reviewed, observations)
                    if plan != reviewed:
                        raise ValueError("stale_review")
                    eligible = self._eligible_conflicts(plan)
                    if self._reviewed_plan_has_non_content_blocker(plan):
                        self._blocked_roots.add(root_id)
                        await self._publish(
                            root_id, "needs_attention", "review_changes"
                        )
                        raise ValueError("review_not_executable")
                    if any(binding_id not in eligible for binding_id in selection_ids):
                        raise ValueError("conflict_selection_mismatch")
                    selected_safe_ids = set(safe_action_ids)
                    safe_actions = tuple(
                        action
                        for action in plan.safe_actions
                        if action.action_id in selected_safe_ids
                        and action.kind in _EXECUTABLE_ACTIONS
                    )
                    if len(safe_actions) != len(selected_safe_ids):
                        raise ValueError("reviewed_action_mismatch")
                    requests: dict[str, object] = {}
                    for action in safe_actions:
                        self._require_authority(root_id, "write")
                        requests[
                            action.action_id
                        ] = await self._adapter.build_execution_request(
                            root, observations, plan, action
                        )
                    conflict_actions: list[NotesSyncAction] = []
                    mutating = sorted(
                        (
                            selection
                            for selection in selections
                            if selection.choice is not NotesSyncConflictChoice.SKIP
                        ),
                        key=lambda selection: selection.binding_id,
                    )
                    for selection in mutating:
                        self._require_authority(root_id, "write")
                        request = await self._adapter.build_conflict_execution_request(
                            root, observations, plan, selection
                        )
                        action = NotesSyncAction(
                            action_id=getattr(request, "operation_id"),
                            kind=getattr(request, "action_kind"),
                            binding_id=selection.binding_id,
                            reason_code=eligible[selection.binding_id].reason_code,
                        )
                        requests[action.action_id] = request
                        conflict_actions.append(action)
                    authority = _FreshAuthority(
                        observations,
                        plan,
                        MappingProxyType(requests),
                    )
                finally:
                    release = getattr(self._adapter, "release_observation", None)
                    if callable(release):
                        release(observed_token)

                actions = (*safe_actions, *conflict_actions)
                raw_results = (
                    await self._execute_locked(
                        root,
                        authority,
                        actions,
                        publish_terminal=False,
                    )
                    if actions
                    else ()
                )
                if any(
                    type(result) is not NotesSyncExecutionResult
                    for result in raw_results
                ):
                    raise RuntimeError("invalid_execution_result")
                results = cast(tuple[NotesSyncExecutionResult, ...], tuple(raw_results))
                completed = tuple(
                    result.state is NotesSyncOperationState.COMPLETED
                    for result in results
                )
                safe_result_count = min(len(results), len(safe_actions))
                safe_completed = sum(completed[:safe_result_count])
                conflict_completed = sum(completed[len(safe_actions) :])
                for selection, result in zip(
                    mutating,
                    results[len(safe_actions) :],
                    strict=False,
                ):
                    if result.state is NotesSyncOperationState.COMPLETED:
                        self._remember_conflict_receipt(
                            root_id,
                            selection.binding_id,
                            result.operation_id,
                            selection.choice,
                        )
                all_completed = len(results) == len(actions) and all(completed)
                needs_recovery = any(result.recovery_required for result in results)
                partial = not all_completed and (
                    any(completed) or len(results) < len(actions)
                )
                fresh_plan: ReconciliationPlan | None = None
                unresolved = len(eligible) - conflict_completed
                attention_remains = unresolved > 0 or not all_completed
                if all_completed:
                    fresh = await self._fresh_authority(root)
                    fresh_plan = fresh.plan
                    self._reviews[root_id] = fresh_plan
                    unresolved = len(self._eligible_conflicts(fresh_plan))
                    blocked = self._blocked_plan_status(fresh_plan)
                    fresh_safe_actions = tuple(
                        action
                        for action in fresh_plan.safe_actions
                        if action.kind in _EXECUTABLE_ACTIONS
                    )
                    attention_remains = blocked is not None
                    if blocked is not None:
                        self._blocked_roots.add(root_id)
                        await self._publish(root_id, *blocked)
                    elif fresh_safe_actions:
                        self._blocked_roots.discard(root_id)
                        await self._publish(
                            root_id, "changes_available", "review_changes"
                        )
                    else:
                        self._blocked_roots.discard(root_id)
                        await self._publish(root_id, "up_to_date", "sync_now")
                return ConflictApplyResult(
                    results=results,
                    safe_completed=safe_completed,
                    conflicts_resolved=conflict_completed,
                    unresolved_conflicts=unresolved,
                    attention_remains=attention_remains,
                    partial=partial,
                    needs_recovery=needs_recovery,
                    fresh_plan=fresh_plan,
                )
        finally:
            self._finish_task(root_id, task)

    @staticmethod
    def _eligible_conflicts(
        plan: ReconciliationPlan,
    ) -> dict[str, ReconciliationAttention]:
        managed = {effect.binding_id for effect in plan.managed_placement_effects}
        return {
            attention.binding_id: attention
            for attention in plan.attention
            if attention.kind is ReconciliationAttentionKind.CONFLICT
            and attention.binding_id is not None
            and eligible_conflict_reason(
                attention.reason_code,
                managed=attention.binding_id in managed,
            )
        }

    @classmethod
    def _reviewed_plan_has_non_content_blocker(
        cls,
        plan: ReconciliationPlan,
    ) -> bool:
        eligible = cls._eligible_conflicts(plan)
        return bool(
            plan.deletion_groups
            or plan.managed_placement_effects
            or plan.skips
            or any(attention.binding_id not in eligible for attention in plan.attention)
        )

    async def _execute(
        self,
        root: NotesSyncRootRecord,
        authority: _FreshAuthority,
        actions: tuple[NotesSyncAction, ...],
    ) -> tuple[object, ...]:
        lock = self._mutation_lock(root.root_id)
        async with lock:
            return await self._execute_locked(root, authority, actions)

    async def _execute_locked(
        self,
        root: NotesSyncRootRecord,
        authority: _FreshAuthority,
        actions: tuple[NotesSyncAction, ...],
        *,
        publish_terminal: bool = True,
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
                    action_id=(
                        getattr(result, "operation_id", None)
                        if getattr(result, "recovery_required", False)
                        else None
                    ),
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
        if publish_terminal:
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
            if operation.kind == "undo_resolution" and not block_pending:
                return False
            status, action = (
                "needs_attention",
                (
                    "resolve_cleanup"
                    if operation.reason_code == "replacement_cleanup_pending"
                    else "review_changes"
                ),
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
            lock = self._mutation_lock(root.root_id)
            async with lock:
                try:
                    current_root = await asyncio.to_thread(
                        self._store.get_root,
                        operation.root_id,
                    )
                except Exception:
                    recovery_blocked.add(operation.root_id)
                    continue
                if (
                    current_root.root_id != operation.root_id
                    or current_root.state is not NotesSyncRootState.ACTIVE
                ):
                    recovery_blocked.add(operation.root_id)
                    continue
                if await self._classify_incomplete_block(current_root, operation):
                    recovery_blocked.add(current_root.root_id)
                    continue
                try:
                    self._require_authority(current_root.root_id, "write")
                    executor = self._adapter.executor_for(
                        current_root,
                        after_stage=lambda _state, root_id=current_root.root_id: (
                            self._require_authority(root_id, "write")
                        ),
                    )
                    request = await executor.reconstruct_request(operation.operation_id)
                    self._require_authority(current_root.root_id, "write")
                    result = await executor.resume(request)
                    self._require_authority(current_root.root_id, "write")
                except Exception:
                    self._blocked_roots.add(current_root.root_id)
                    await self._publish(
                        current_root.root_id,
                        (
                            "needs_attention"
                            if operation.kind == "undo_resolution"
                            else "failed"
                        ),
                        "review_changes",
                        action_id=operation.operation_id,
                    )
                    recovery_blocked.add(current_root.root_id)
                    continue
                if result.state is NotesSyncOperationState.COMPLETED:
                    await self._publish(current_root.root_id, "up_to_date", "sync_now")
                elif result.state is NotesSyncOperationState.NEEDS_ATTENTION:
                    self._blocked_roots.add(current_root.root_id)
                    await self._publish(
                        current_root.root_id,
                        "needs_attention",
                        "review_changes",
                        action_id=operation.operation_id,
                    )
                    recovery_blocked.add(current_root.root_id)
                else:
                    self._blocked_roots.add(current_root.root_id)
                    next_action = (
                        "resolve_cleanup"
                        if getattr(result, "reason_code", None)
                        == "replacement_cleanup_pending"
                        else "review_changes"
                    )
                    await self._publish(
                        current_root.root_id,
                        "partial",
                        next_action,
                        action_id=operation.operation_id,
                    )
                    recovery_blocked.add(current_root.root_id)

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
    start_evidence: Callable[[], bool] | None = None,
    watcher_interval_seconds: float | None = None,
    watcher_max_interval_seconds: float | None = None,
) -> NotesSyncRuntimeOwner:
    """Build the application-owned lasting-sync runtime.

    ``start_evidence`` (TASK-21112) gates the boot-time start: when it
    reports False, :meth:`NotesSyncRuntimeOwner.start` defers inert without
    creating the state database; first-time setup forces the start later.
    ``watcher_interval_seconds`` / ``watcher_max_interval_seconds`` shape the
    default polling watcher's idle backoff; ``None`` keeps its defaults.
    """

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
    watcher_intervals: dict[str, float] = {}
    if watcher_interval_seconds is not None:
        watcher_intervals["interval_seconds"] = float(watcher_interval_seconds)
    if watcher_max_interval_seconds is not None:
        watcher_intervals["max_interval_seconds"] = float(
            watcher_max_interval_seconds
        )
    selected_watcher_factory = watcher_factory or (
        lambda schedule: PollingNotesSyncWatcher(
            lambda: owner_holder["owner"]._changed_root_ids(),
            schedule,
            **watcher_intervals,
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
        start_evidence=start_evidence,
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
    "RuntimeConflictHistoryRow",
    "RuntimeConflictReceipt",
    "NotesSyncRuntimeOwner",
    "NotesSyncRuntimeSnapshot",
    "build_notes_sync_legacy_migrator",
    "build_notes_sync_runtime_owner",
]
