"""Fresh-authority and per-root serialization for conflict review."""

from __future__ import annotations

import asyncio
import gc
import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.Notes.notes_device_state_store import NotesSyncRootRecord
from tldw_chatbook.Notes.notes_sync_conflicts import (
    ConflictApplyResult,
    ConflictComparison,
    ConflictSelection,
    NotesSyncConflictChoice,
    build_conflict_comparison,
    conflict_resolution_operation_id,
)
from tldw_chatbook.Notes.notes_sync_authority import NotesSyncNoteSnapshot
from tldw_chatbook.Notes.notes_sync_filesystem import NotesSyncFileSnapshot
from tldw_chatbook.Notes.notes_sync_models import (
    NotesSyncActionKind,
    NotesSyncDirection,
    NotesSyncFileIdentity,
    NotesSyncFileObservation,
    NotesSyncOperationState,
    NotesSyncRootState,
    NotesSyncSerializationProfile,
)
from tldw_chatbook.Notes.notes_sync_reconciler import (
    BindingObservation,
    DeletionGroup,
    ManagedPlacementEffect,
    ManagedPlacementEffectKind,
    ReconciliationAttention,
    ReconciliationAttentionKind,
    ReconciliationInput,
    ReconciliationSkip,
    ReconciliationSkipKind,
    plan_reconciliation,
)
from tldw_chatbook.Notes.notes_sync_runtime import (
    NotesSyncRuntimeOwner,
    _ProductionRuntimeAdapter,
)
from tldw_chatbook.Notes.notes_sync_executor import (
    NotesSyncDirectionOverride,
    NotesSyncExecutionRequest,
    NotesSyncExecutionResult,
)
from tldw_chatbook.Notes.sync_paths import SafeSyncBytes, SafeSyncFileIdentity


pytestmark = pytest.mark.unit
_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_PHASE_TIMEOUT = 1.0
_CONFLICT_RETENTION_NS = 30 * 24 * 60 * 60 * 1_000_000_000


async def _wait(event: asyncio.Event) -> None:
    await asyncio.wait_for(event.wait(), _PHASE_TIMEOUT)


async def _finish_tasks(*tasks: asyncio.Task[object]) -> tuple[object, ...]:
    for task in tasks:
        if not task.done():
            task.cancel()
    return tuple(await asyncio.gather(*tasks, return_exceptions=True))


def _input(
    root_id: str = "root-1",
    *,
    generation: int = 1,
    file_digest: str = _B,
    note_digest: str = _A,
    direction: NotesSyncDirection = NotesSyncDirection.BIDIRECTIONAL,
) -> ReconciliationInput:
    return ReconciliationInput(
        root_id=root_id,
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
                file_identity_digest=_C,
                relative_path="note.md",
                note_scope_id="local_note",
                note_id="note-1",
                note_version=generation,
            ),
        ),
        observation_generation=generation,
        expected_generation=generation,
    )


def _root(
    root_id: str = "root-1",
    *,
    direction: NotesSyncDirection = NotesSyncDirection.BIDIRECTIONAL,
) -> NotesSyncRootRecord:
    return NotesSyncRootRecord(
        root_id=root_id,
        note_scope_id="local_note",
        logical_folder_id="folder-1",
        canonical_path=f"/private/{root_id}",
        direction=direction,
        state=NotesSyncRootState.ACTIVE,
    )


class _Store:
    def __init__(self, *roots: NotesSyncRootRecord) -> None:
        self.roots = {root.root_id: root for root in roots}

    def get_root(self, root_id: str) -> NotesSyncRootRecord:
        return self.roots[root_id]

    def update_root_status(self, _root_id: str, _status: str) -> None:
        return None


class _Lease:
    def __init__(self, authoritative: bool = True) -> None:
        self.authoritative = authoritative


class _Admission:
    def __init__(self, lease: _Lease) -> None:
        self.lease = lease

    def require_authority(self, _operation: str) -> _Lease:
        if not self.lease.authoritative:
            raise RuntimeError("admission_closed")
        return self.lease


class _ObservedLock(asyncio.Lock):
    def __init__(self) -> None:
        super().__init__()
        self.waiting = asyncio.Event()
        self._attempts = 0

    async def acquire(self) -> bool:
        self._attempts += 1
        if self._attempts > 1:
            self.waiting.set()
        return await super().acquire()


class _Executor:
    def __init__(self, adapter: "_Adapter") -> None:
        self.adapter = adapter
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.resume_entered = asyncio.Event()
        self.resume_release = asyncio.Event()
        self.mutations = 0

    async def execute(self, _request: object) -> object:
        self.entered.set()
        await self.release.wait()
        self.mutations += 1
        self.adapter.inputs["root-1"] = _input(
            generation=2, file_digest=_A, note_digest=_A
        )
        return NotesSyncExecutionResult(
            operation_id=getattr(_request, "operation_id", "operation-1"),
            state=NotesSyncOperationState.COMPLETED,
            reason_code=None,
        )

    async def reconstruct_request(self, operation_id: str) -> str:
        return operation_id

    async def resume(self, _request: object) -> object:
        self.resume_entered.set()
        await self.resume_release.wait()
        self.mutations += 1
        self.adapter.inputs["root-1"] = _input(
            generation=2, file_digest=_A, note_digest=_A
        )
        return SimpleNamespace(
            state=NotesSyncOperationState.COMPLETED,
            reason_code=None,
        )


class _Adapter:
    def __init__(self, *inputs: ReconciliationInput) -> None:
        self.inputs = {item.root_id: item for item in inputs}
        self.observe_calls: list[str] = []
        self.executor = _Executor(self)
        self.live_bundles: set[str] = set()
        self.released: list[str] = []
        self.comparison_started = asyncio.Event()
        self.comparison_release: asyncio.Event | None = None
        self.close_lease_on_observe: _Lease | None = None
        self.comparison_builds = 0

    async def observe_root(self, root: NotesSyncRootRecord) -> ReconciliationInput:
        observed = self.inputs[root.root_id]
        self.observe_calls.append(root.root_id)
        token = plan_reconciliation(observed).observation_token
        self.live_bundles.add(token)
        if self.close_lease_on_observe is not None:
            self.close_lease_on_observe.authoritative = False
        return observed

    async def build_execution_request(
        self, _root: object, _observations: object, _plan: object, action: object
    ) -> object:
        return action

    def executor_for(self, _root: object, **_kwargs: object) -> _Executor:
        return self.executor

    async def build_conflict_comparison(
        self, _root: object, plan: object, binding_id: str
    ) -> ConflictComparison:
        token = plan.observation_token
        assert token in self.live_bundles
        self.comparison_builds += 1
        self.comparison_started.set()
        if self.comparison_release is not None:
            await self.comparison_release.wait()
        assert token in self.live_bundles
        return build_conflict_comparison(
            binding_id=binding_id,
            title="Private title",
            relative_path="note.md",
            note_text="note side\n",
            file_text="file side\n",
            note_version=1,
            note_updated_at="2026-08-22T12:30:00+00:00",
            file_modified_ns=7,
        )

    def release_observation(self, observation_token: str) -> None:
        self.released.append(observation_token)
        self.live_bundles.discard(observation_token)


class _SubsetExecutor:
    def __init__(
        self,
        adapter: "_SubsetAdapter",
        final_input: ReconciliationInput,
        *,
        stop_binding_id: str | None = None,
    ) -> None:
        self.adapter = adapter
        self.final_input = final_input
        self.stop_binding_id = stop_binding_id
        self.requests: list[object] = []

    async def execute(self, request: object) -> NotesSyncExecutionResult:
        self.requests.append(request)
        binding_id = getattr(request, "binding_id")
        operation_id = getattr(request, "operation_id")
        if binding_id == self.stop_binding_id:
            return NotesSyncExecutionResult(
                operation_id=operation_id,
                state=NotesSyncOperationState.FIRST_AUTHORITY_APPLIED,
            )
        self.adapter.inputs[self.final_input.root_id] = self.final_input
        return NotesSyncExecutionResult(
            operation_id=operation_id,
            state=NotesSyncOperationState.COMPLETED,
        )


class _SubsetAdapter(_Adapter):
    def __init__(
        self,
        initial: ReconciliationInput,
        final: ReconciliationInput,
        *,
        stop_binding_id: str | None = None,
    ) -> None:
        super().__init__(initial)
        self.executor = _SubsetExecutor(
            self,
            final,
            stop_binding_id=stop_binding_id,
        )

    async def build_execution_request(
        self, root: object, _observations: object, _plan: object, action: object
    ) -> object:
        return SimpleNamespace(
            operation_id=f"safe-{action.action_id}",
            root_id=root.root_id,
            binding_id=action.binding_id,
            action_kind=action.kind,
            direction_override=None,
        )

    async def build_conflict_execution_request(
        self,
        root: object,
        _observations: object,
        plan: object,
        selection: ConflictSelection,
    ) -> object:
        action_kind = (
            NotesSyncActionKind.UPDATE_NOTE
            if selection.choice is NotesSyncConflictChoice.KEEP_FILE
            else NotesSyncActionKind.UPDATE_FILE
        )
        return SimpleNamespace(
            operation_id=f"conflict-{selection.binding_id}",
            root_id=root.root_id,
            binding_id=selection.binding_id,
            action_kind=action_kind,
            journal_kind=f"resolve_{selection.choice.value}",
            direction_override=None,
            observation_token=plan.observation_token,
        )


def _reviewed_subset_input(
    *,
    direction: NotesSyncDirection = NotesSyncDirection.BIDIRECTIONAL,
    resolved: frozenset[str] = frozenset(),
    conflict_count: int = 2,
) -> ReconciliationInput:
    bindings: list[BindingObservation] = [
        BindingObservation(
            binding_id="binding-safe",
            baseline_file_digest=(_B if resolved else _A),
            baseline_note_digest=(_B if resolved else _A),
            baseline_identity_digest=_C,
            baseline_relative_path="safe.md",
            file_digest=_B,
            note_digest=(_B if resolved else _A),
            file_identity_digest=_C,
            relative_path="safe.md",
            note_scope_id="local_note",
            note_id="note-safe",
            note_version=2,
        )
    ]
    for index, (file_digest, note_digest) in enumerate(
        ((_B, _C), (_C, _B), (_B, _C))[:conflict_count],
        start=1,
    ):
        binding_id = f"binding-{index}"
        if binding_id in resolved:
            baseline_file = file_digest
            baseline_note = file_digest
            note_digest = file_digest
        else:
            baseline_file = _A
            baseline_note = _A
        bindings.append(
            BindingObservation(
                binding_id=binding_id,
                baseline_file_digest=baseline_file,
                baseline_note_digest=baseline_note,
                baseline_identity_digest=_C,
                baseline_relative_path=f"note-{index}.md",
                file_digest=file_digest,
                note_digest=note_digest,
                file_identity_digest=_C,
                relative_path=f"note-{index}.md",
                note_scope_id="local_note",
                note_id=f"note-{index}",
                note_version=2,
            )
        )
    return ReconciliationInput(
        root_id="root-1",
        direction=direction,
        bindings=tuple(bindings),
        observation_generation=1,
        expected_generation=1,
    )


def _owner(
    adapter: _Adapter,
    *roots: NotesSyncRootRecord,
) -> NotesSyncRuntimeOwner:
    owner = NotesSyncRuntimeOwner(
        store=_Store(*roots),
        migrate_legacy=lambda: None,
        coordinator=object(),
        adapter=adapter,
        watcher_factory=lambda _schedule: object(),
        cutover_admitted=True,
        profile_process_is_sole=True,
    )
    owner._admission_open = True
    owner._status = "active"
    for root in roots:
        lease = _Lease()
        owner._leases[root.root_id] = lease
        owner._admissions[root.root_id] = _Admission(lease)
    return owner


def _install_review(owner: NotesSyncRuntimeOwner, observed: ReconciliationInput) -> str:
    plan = plan_reconciliation(observed)
    owner._reviews[observed.root_id] = plan
    return plan.observation_token


@pytest.mark.asyncio
async def test_two_reviewed_apply_paths_share_one_root_lock_and_reobserve() -> None:
    adapter = _Adapter(_input())
    owner = _owner(adapter, _root())
    token = _install_review(owner, adapter.inputs["root-1"])
    action_id = owner._reviews["root-1"].safe_actions[0].action_id
    lock = _ObservedLock()
    owner._mutation_locks["root-1"] = lock

    first = asyncio.create_task(owner.apply_reviewed("root-1", token, (action_id,)))
    second: asyncio.Task[object] | None = None
    try:
        await _wait(adapter.executor.entered)
        second = asyncio.create_task(
            owner.apply_reviewed("root-1", token, (action_id,))
        )
        await _wait(lock.waiting)
        assert adapter.observe_calls == ["root-1"]
        assert adapter.executor.mutations == 0
        adapter.executor.release.set()
        results = tuple(
            await asyncio.wait_for(
                asyncio.gather(first, second, return_exceptions=True),
                _PHASE_TIMEOUT,
            )
        )
    finally:
        adapter.executor.release.set()
        await _finish_tasks(
            first,
            *(() if second is None else (second,)),
        )

    assert adapter.observe_calls == ["root-1", "root-1", "root-1"]
    assert adapter.executor.mutations == 1
    assert any(isinstance(result, ValueError) for result in results)


@pytest.mark.asyncio
async def test_automatic_and_reviewed_apply_share_root_lock() -> None:
    adapter = _Adapter(_input())
    owner = _owner(adapter, _root())
    token = _install_review(owner, adapter.inputs["root-1"])
    action_id = owner._reviews["root-1"].safe_actions[0].action_id
    lock = _ObservedLock()
    owner._mutation_locks["root-1"] = lock

    automatic = asyncio.create_task(owner._reconcile(_root(), automatic=True))
    reviewed: asyncio.Task[object] | None = None
    try:
        await _wait(adapter.executor.entered)
        reviewed = asyncio.create_task(
            owner.apply_reviewed("root-1", token, (action_id,))
        )
        await _wait(lock.waiting)
        assert adapter.observe_calls == ["root-1"]
        assert adapter.executor.mutations == 0
        adapter.executor.release.set()
        results = tuple(
            await asyncio.wait_for(
                asyncio.gather(automatic, reviewed, return_exceptions=True),
                _PHASE_TIMEOUT,
            )
        )
    finally:
        adapter.executor.release.set()
        await _finish_tasks(
            automatic,
            *(() if reviewed is None else (reviewed,)),
        )

    assert adapter.observe_calls == ["root-1", "root-1"]
    assert adapter.executor.mutations == 1
    assert isinstance(results[1], ValueError)


@pytest.mark.asyncio
async def test_startup_recovery_and_reviewed_apply_share_root_lock() -> None:
    adapter = _Adapter(_input())
    root = _root()
    owner = _owner(adapter, root)
    token = _install_review(owner, adapter.inputs["root-1"])
    action_id = owner._reviews["root-1"].safe_actions[0].action_id
    lock = _ObservedLock()
    owner._mutation_locks["root-1"] = lock
    operation = SimpleNamespace(
        root_id="root-1",
        operation_id="operation-1",
        state=NotesSyncOperationState.RECOVERY_ADMITTED,
    )

    recovery = asyncio.create_task(
        owner._resume_incomplete({"root-1": root}, (operation,))
    )
    reviewed: asyncio.Task[object] | None = None
    try:
        await _wait(adapter.executor.resume_entered)
        reviewed = asyncio.create_task(
            owner.apply_reviewed("root-1", token, (action_id,))
        )
        await _wait(lock.waiting)
        assert adapter.observe_calls == []
        assert adapter.executor.mutations == 0
        adapter.executor.resume_release.set()
        results = tuple(
            await asyncio.wait_for(
                asyncio.gather(recovery, reviewed, return_exceptions=True),
                _PHASE_TIMEOUT,
            )
        )
    finally:
        adapter.executor.resume_release.set()
        await _finish_tasks(
            recovery,
            *(() if reviewed is None else (reviewed,)),
        )

    assert adapter.observe_calls == ["root-1"]
    assert adapter.executor.mutations == 1
    assert isinstance(results[1], ValueError)


@pytest.mark.asyncio
async def test_root_lock_waiter_retains_lock_across_registry_gc() -> None:
    owner = _owner(_Adapter(_input()), _root())
    lock_factory = owner._mutation_lock
    holder_entered = asyncio.Event()
    holder_release = asyncio.Event()
    waiter_resolved = asyncio.Event()
    waiter_entered = asyncio.Event()
    identities: list[int] = []

    async def holder() -> None:
        lock = lock_factory("root-1")
        identities.append(id(lock))
        async with lock:
            holder_entered.set()
            await holder_release.wait()

    async def waiter() -> None:
        lock = lock_factory("root-1")
        identities.append(id(lock))
        waiter_resolved.set()
        async with lock:
            waiter_entered.set()

    holding = asyncio.create_task(holder())
    waiting: asyncio.Task[object] | None = None
    try:
        await _wait(holder_entered)
        waiting = asyncio.create_task(waiter())
        await _wait(waiter_resolved)
        gc.collect()
        identities.append(id(lock_factory("root-1")))
        assert len(set(identities)) == 1
        assert not waiter_entered.is_set()
        holder_release.set()
        await asyncio.wait_for(
            asyncio.gather(holding, waiting),
            _PHASE_TIMEOUT,
        )
    finally:
        holder_release.set()
        await _finish_tasks(
            holding,
            *(() if waiting is None else (waiting,)),
        )


@pytest.mark.asyncio
async def test_different_root_locks_proceed_independently() -> None:
    owner = _owner(_Adapter(_input(), _input("root-2")), _root(), _root("root-2"))
    lock_factory = owner._mutation_lock
    first_lock = lock_factory("root-1")
    second_entered = asyncio.Event()

    async with first_lock:

        async def enter_second() -> None:
            async with lock_factory("root-2"):
                second_entered.set()

        second = asyncio.create_task(enter_second())
        try:
            await _wait(second_entered)
        finally:
            await _finish_tasks(second)


@pytest.mark.asyncio
async def test_locked_execution_helper_does_not_reacquire_root_lock() -> None:
    adapter = _Adapter(_input())
    owner = _owner(adapter, _root())
    token = _install_review(owner, adapter.inputs["root-1"])
    action_id = owner._reviews["root-1"].safe_actions[0].action_id
    adapter.executor.release.set()
    acquisitions = 0
    lock = asyncio.Lock()

    def guarded_lock(_root_id: str) -> asyncio.Lock:
        nonlocal acquisitions
        acquisitions += 1
        if acquisitions > 1:
            raise AssertionError("nested root-lock acquisition")
        return lock

    owner._mutation_lock = guarded_lock
    result = await owner.apply_reviewed("root-1", token, (action_id,))

    assert type(result) is ConflictApplyResult
    assert len(result.results) == 1
    assert acquisitions == 1


@pytest.mark.asyncio
async def test_comparison_reobserves_exact_authority_and_returns_projection() -> None:
    observed = _input(file_digest=_B, note_digest=_C)
    adapter = _Adapter(observed)
    owner = _owner(adapter, _root())
    token = _install_review(owner, observed)

    comparison = await owner.compare_conflict("root-1", token, "binding-1")

    assert type(comparison) is ConflictComparison
    assert comparison.binding_id == "binding-1"
    assert comparison.note_updated_at == "2026-08-22T12:30:00+00:00"
    assert comparison.diff.startswith("--- Note\n+++ File\n")
    assert adapter.observe_calls == ["root-1"]
    assert adapter.comparison_builds == 1
    assert adapter.live_bundles == set()
    assert not hasattr(comparison, "note")
    assert not hasattr(comparison, "file")


@pytest.mark.asyncio
async def test_production_comparison_adapter_projects_only_bounded_content() -> None:
    note_text = "private note\n"
    file_text = "private file\n"
    profile = NotesSyncSerializationProfile(False, "lf", True, 0o600)
    note = NotesSyncNoteSnapshot(
        note_scope_id="local_note",
        note_id="note-1",
        title="Private title",
        content=note_text,
        version=3,
        content_digest=hashlib.sha256(note_text.encode()).hexdigest(),
        updated_at=None,
    )
    reviewed_state = SafeSyncBytes(
        relative_path=Path("note.md"),
        content=file_text.encode(),
        identity=SafeSyncFileIdentity(1, 2, 1),
        mode=0o600,
        size=len(file_text),
        mtime_ns=9,
        ctime_ns=8,
        owner_user=1,
        owner_group=1,
        flags=0,
        extended_attributes=(),
        has_extended_acl=False,
    )
    file = NotesSyncFileSnapshot(
        observation=NotesSyncFileObservation(
            relative_path="note.md",
            identity=NotesSyncFileIdentity(1, 2, 1),
            content_digest=hashlib.sha256(file_text.encode()).hexdigest(),
            size_bytes=len(file_text),
            serialization=profile,
        ),
        text=file_text,
        raw_bytes=file_text.encode(),
        reviewed_state=reviewed_state,
        representation_digest=_A,
    )
    plan = plan_reconciliation(_input(file_digest=_B, note_digest=_C))
    adapter = object.__new__(_ProductionRuntimeAdapter)
    adapter._bundles = {
        plan.observation_token: {
            "binding-1": SimpleNamespace(
                record=SimpleNamespace(root_id="root-1"),
                note=note,
                file=file,
            )
        }
    }

    comparison = await adapter.build_conflict_comparison(_root(), plan, "binding-1")

    assert type(comparison) is ConflictComparison
    assert comparison.note_updated_at is None
    assert comparison.file_modified_ns == 9
    assert comparison.diff.startswith("--- Note\n+++ File\n")
    assert not hasattr(comparison, "raw_bytes")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    ["missing_root", "missing_lease", "stale_lease", "inactive", "wrong"],
)
async def test_comparison_refuses_invalid_root_authority_before_content(
    case: str,
) -> None:
    observed = _input(file_digest=_B, note_digest=_C)
    adapter = _Adapter(observed)
    root = _root()
    owner = _owner(adapter, root)
    token = _install_review(owner, observed)
    if case == "missing_root":
        owner._store.roots.clear()
    elif case == "missing_lease":
        owner._admissions.clear()
    elif case == "stale_lease":
        owner._admissions["root-1"].lease.authoritative = False
    elif case == "inactive":
        owner._store.roots["root-1"] = replace(root, state=NotesSyncRootState.PAUSED)
    else:
        owner._store.roots["root-1"] = _root("root-wrong")

    with pytest.raises((RuntimeError, ValueError, KeyError)):
        await owner.compare_conflict("root-1", token, "binding-1")

    assert adapter.observe_calls == []
    assert adapter.comparison_builds == 0


@pytest.mark.asyncio
async def test_comparison_requires_exact_plan_even_with_reviewed_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_chatbook.Notes.notes_sync_runtime as runtime_module

    observed = _input(file_digest=_B, note_digest=_C)
    adapter = _Adapter(observed)
    owner = _owner(adapter, _root())
    reviewed = plan_reconciliation(observed)
    owner._reviews["root-1"] = reviewed
    monkeypatch.setattr(
        runtime_module,
        "plan_reconciliation",
        lambda _value: replace(reviewed, attention=()),
    )

    with pytest.raises(ValueError, match="stale_review"):
        await owner.compare_conflict("root-1", reviewed.observation_token, "binding-1")

    assert adapter.comparison_builds == 0
    assert adapter.released == [reviewed.observation_token]


@pytest.mark.asyncio
async def test_comparison_releases_bundle_on_stale_post_observation_authority() -> None:
    observed = _input(file_digest=_B, note_digest=_C)
    adapter = _Adapter(observed)
    owner = _owner(adapter, _root())
    token = _install_review(owner, observed)
    lease = owner._admissions["root-1"].lease
    adapter.close_lease_on_observe = lease

    with pytest.raises(RuntimeError, match="lease"):
        await owner.compare_conflict("root-1", token, "binding-1")

    assert adapter.comparison_builds == 0
    assert adapter.released == [token]
    assert adapter.live_bundles == set()


@pytest.mark.asyncio
async def test_comparison_releases_observation_when_planner_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_chatbook.Notes.notes_sync_runtime as runtime_module

    observed = _input(file_digest=_B, note_digest=_C)
    token = plan_reconciliation(observed).observation_token
    adapter = _Adapter(observed)
    owner = _owner(adapter, _root())
    _install_review(owner, observed)
    monkeypatch.setattr(
        runtime_module,
        "plan_reconciliation",
        lambda _value: (_ for _ in ()).throw(RuntimeError("planner_failed")),
    )

    with pytest.raises(RuntimeError, match="planner_failed"):
        await owner.compare_conflict("root-1", token, "binding-1")

    assert adapter.released == [token]
    assert adapter.live_bundles == set()


@pytest.mark.asyncio
async def test_comparison_releases_bundle_when_cancelled() -> None:
    observed = _input(file_digest=_B, note_digest=_C)
    adapter = _Adapter(observed)
    adapter.comparison_release = asyncio.Event()
    owner = _owner(adapter, _root())
    token = _install_review(owner, observed)
    task = asyncio.create_task(owner.compare_conflict("root-1", token, "binding-1"))
    try:
        await _wait(adapter.comparison_started)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, _PHASE_TIMEOUT)
    finally:
        if adapter.comparison_release is not None:
            adapter.comparison_release.set()
        await _finish_tasks(task)

    assert adapter.released == [token]
    assert adapter.live_bundles == set()


@pytest.mark.asyncio
async def test_comparison_refuses_stale_plan_and_wrong_binding_then_releases() -> None:
    reviewed_input = _input(file_digest=_B, note_digest=_C)
    fresh_input = replace(
        reviewed_input,
        observation_generation=2,
        expected_generation=2,
    )
    adapter = _Adapter(fresh_input)
    owner = _owner(adapter, _root())
    reviewed_token = _install_review(owner, reviewed_input)

    with pytest.raises(ValueError, match="stale_review"):
        await owner.compare_conflict("root-1", reviewed_token, "binding-1")
    fresh_token = _install_review(owner, fresh_input)
    with pytest.raises(ValueError, match="binding"):
        await owner.compare_conflict("root-1", fresh_token, "binding-wrong")

    assert adapter.comparison_builds == 0
    assert adapter.live_bundles == set()
    assert adapter.released == [
        plan_reconciliation(fresh_input).observation_token,
        plan_reconciliation(fresh_input).observation_token,
    ]


@pytest.mark.asyncio
async def test_comparison_requires_final_token_even_if_plan_claims_equality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_chatbook.Notes.notes_sync_runtime as runtime_module

    observed = _input(file_digest=_B, note_digest=_C)
    adapter = _Adapter(observed)
    owner = _owner(adapter, _root())
    reviewed = plan_reconciliation(observed)
    owner._reviews["root-1"] = reviewed
    stale_token = hashlib.sha256(b"stale").hexdigest()

    class _EqualPlan:
        root_id = reviewed.root_id
        observation_token = stale_token
        attention = reviewed.attention
        managed_placement_effects = reviewed.managed_placement_effects

        def __eq__(self, _other: object) -> bool:
            return True

    async def observe_with_stale_bundle(_root: object) -> ReconciliationInput:
        adapter.observe_calls.append("root-1")
        adapter.live_bundles.add(reviewed.observation_token)
        return observed

    async def build_equal_plan_comparison(
        _root: object, _plan: object, binding_id: str
    ) -> ConflictComparison:
        return build_conflict_comparison(
            binding_id=binding_id,
            title="Private title",
            relative_path="note.md",
            note_text="note side\n",
            file_text="file side\n",
            note_version=1,
            note_updated_at=None,
            file_modified_ns=7,
        )

    adapter.observe_root = observe_with_stale_bundle
    adapter.build_conflict_comparison = build_equal_plan_comparison
    monkeypatch.setattr(
        runtime_module, "plan_reconciliation", lambda _value: _EqualPlan()
    )

    with pytest.raises(ValueError, match="stale_review"):
        await owner.compare_conflict("root-1", reviewed.observation_token, "binding-1")

    assert adapter.comparison_builds == 0
    assert adapter.live_bundles == set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "direction",
        "choice",
        "action",
        "override_expected",
        "file_digest",
        "note_digest",
    ),
    (
        (
            NotesSyncDirection.BIDIRECTIONAL,
            NotesSyncConflictChoice.KEEP_FILE,
            NotesSyncActionKind.UPDATE_NOTE,
            False,
            _B,
            _C,
        ),
        (
            NotesSyncDirection.BIDIRECTIONAL,
            NotesSyncConflictChoice.KEEP_NOTE,
            NotesSyncActionKind.UPDATE_FILE,
            False,
            _B,
            _C,
        ),
        (
            NotesSyncDirection.FOLDER_TO_NOTES,
            NotesSyncConflictChoice.KEEP_FILE,
            NotesSyncActionKind.UPDATE_NOTE,
            False,
            _B,
            _C,
        ),
        (
            NotesSyncDirection.FOLDER_TO_NOTES,
            NotesSyncConflictChoice.KEEP_NOTE,
            NotesSyncActionKind.UPDATE_FILE,
            True,
            _B,
            _C,
        ),
        (
            NotesSyncDirection.NOTES_TO_FOLDER,
            NotesSyncConflictChoice.KEEP_FILE,
            NotesSyncActionKind.UPDATE_NOTE,
            True,
            _B,
            _C,
        ),
        (
            NotesSyncDirection.NOTES_TO_FOLDER,
            NotesSyncConflictChoice.KEEP_NOTE,
            NotesSyncActionKind.UPDATE_FILE,
            False,
            _B,
            _C,
        ),
        (
            NotesSyncDirection.FOLDER_TO_NOTES,
            NotesSyncConflictChoice.KEEP_FILE,
            NotesSyncActionKind.UPDATE_NOTE,
            False,
            _A,
            _C,
        ),
        (
            NotesSyncDirection.FOLDER_TO_NOTES,
            NotesSyncConflictChoice.KEEP_NOTE,
            NotesSyncActionKind.UPDATE_FILE,
            True,
            _A,
            _C,
        ),
        (
            NotesSyncDirection.NOTES_TO_FOLDER,
            NotesSyncConflictChoice.KEEP_FILE,
            NotesSyncActionKind.UPDATE_NOTE,
            True,
            _B,
            _A,
        ),
        (
            NotesSyncDirection.NOTES_TO_FOLDER,
            NotesSyncConflictChoice.KEEP_NOTE,
            NotesSyncActionKind.UPDATE_FILE,
            False,
            _B,
            _A,
        ),
    ),
)
async def test_production_keep_file_and_keep_note_requests_preserve_direction(
    direction: NotesSyncDirection,
    choice: NotesSyncConflictChoice,
    action: NotesSyncActionKind,
    override_expected: bool,
    file_digest: str,
    note_digest: str,
) -> None:
    note_text = "note side"
    file_text = "file side"
    note = NotesSyncNoteSnapshot(
        note_scope_id="local_note",
        note_id="note-1",
        title="Title",
        content=note_text,
        version=5,
        content_digest=hashlib.sha256(note_text.encode()).hexdigest(),
    )
    file = NotesSyncFileSnapshot(
        observation=NotesSyncFileObservation(
            relative_path="note.md",
            identity=NotesSyncFileIdentity(1, 2, 1),
            content_digest=hashlib.sha256(file_text.encode()).hexdigest(),
            size_bytes=len(file_text),
            serialization=NotesSyncSerializationProfile(False, "lf", False, 0o600),
        ),
        text=file_text,
        raw_bytes=file_text.encode(),
        reviewed_state=SafeSyncBytes(
            relative_path=Path("note.md"),
            content=file_text.encode(),
            identity=SafeSyncFileIdentity(1, 2, 1),
            mode=0o600,
            size=len(file_text),
            mtime_ns=9,
            ctime_ns=8,
            owner_user=1,
            owner_group=1,
            flags=0,
            extended_attributes=(),
            has_extended_acl=False,
        ),
        representation_digest=_A,
    )
    observed = _input(
        file_digest=file_digest,
        note_digest=note_digest,
        direction=direction,
    )
    plan = plan_reconciliation(observed)
    assert plan.attention[0].reason_code in {
        "both_sides_changed",
        "out_of_direction_change",
    }
    root = _root(direction=direction)
    adapter = object.__new__(_ProductionRuntimeAdapter)
    adapter._bundles = {
        plan.observation_token: {
            "binding-1": SimpleNamespace(
                record=SimpleNamespace(
                    binding_id="binding-1",
                    root_id="root-1",
                    normalized_relative_path="note.md",
                    note_scope_id="local_note",
                    note_id="note-1",
                    serialization=file.observation.serialization,
                ),
                note=note,
                file=file,
            )
        }
    }
    before = __import__("time").time_ns()

    request = await adapter.build_conflict_execution_request(
        root,
        observed,
        plan,
        ConflictSelection("binding-1", choice),
    )

    assert type(request) is NotesSyncExecutionRequest
    assert request.action_kind is action
    assert request.journal_kind == f"resolve_{choice.value}"
    assert request.direction is direction
    assert request.operation_id == conflict_resolution_operation_id(
        "root-1", "binding-1", plan.observation_token, choice
    )
    assert before + _CONFLICT_RETENTION_NS <= request.recovery_expires_at
    assert isinstance(request.direction_override, NotesSyncDirectionOverride) is (
        override_expected
    )
    if request.direction_override is not None:
        assert request.direction_override.action_kind is action
        assert request.direction_override.observation_token == plan.observation_token


@pytest.mark.asyncio
async def test_selected_keep_file_executes_while_skip_remains_attention() -> None:
    initial = _reviewed_subset_input()
    final = _reviewed_subset_input(resolved=frozenset({"binding-1"}))
    adapter = _SubsetAdapter(initial, final)
    owner = _owner(adapter, _root())
    token = _install_review(owner, initial)
    safe_id = owner._reviews["root-1"].safe_actions[0].action_id

    result = await owner.apply_reviewed(
        "root-1",
        token,
        (safe_id,),
        (
            ConflictSelection("binding-1", NotesSyncConflictChoice.KEEP_FILE),
            ConflictSelection("binding-2", NotesSyncConflictChoice.SKIP),
        ),
    )

    assert type(result) is ConflictApplyResult
    assert result.safe_completed == 1
    assert result.conflicts_resolved == 1
    assert result.unresolved_conflicts == 1
    assert result.attention_remains is True
    assert result.partial is False
    assert result.needs_recovery is False
    assert result.fresh_plan is not None
    assert (
        owner.snapshot().roots[0].status,
        owner.snapshot().roots[0].next_action,
    ) == (
        "needs_attention",
        "review_changes",
    )
    assert [request.binding_id for request in adapter.executor.requests] == [
        "binding-safe",
        "binding-1",
    ]


@pytest.mark.asyncio
async def test_terminal_refresh_with_fresh_safe_actions_remains_reviewable() -> None:
    initial = _reviewed_subset_input(conflict_count=1)
    resolved = _reviewed_subset_input(
        resolved=frozenset({"binding-1"}),
        conflict_count=1,
    )
    safe = replace(
        resolved.bindings[0],
        baseline_file_digest=_A,
        baseline_note_digest=_A,
        note_digest=_A,
    )
    final = replace(resolved, bindings=(safe, *resolved.bindings[1:]))
    assert any(
        action.kind
        in {NotesSyncActionKind.UPDATE_NOTE, NotesSyncActionKind.UPDATE_FILE}
        for action in plan_reconciliation(final).safe_actions
    )
    adapter = _SubsetAdapter(initial, final)
    owner = _owner(adapter, _root())
    token = _install_review(owner, initial)

    result = await owner.apply_reviewed(
        "root-1",
        token,
        (),
        (ConflictSelection("binding-1", NotesSyncConflictChoice.KEEP_FILE),),
    )

    assert result.conflicts_resolved == 1
    assert result.unresolved_conflicts == 0
    assert result.attention_remains is False
    assert result.partial is False
    assert result.needs_recovery is False
    assert result.fresh_plan is not None and any(
        action.kind
        in {NotesSyncActionKind.UPDATE_NOTE, NotesSyncActionKind.UPDATE_FILE}
        for action in result.fresh_plan.safe_actions
    )
    root_status = owner.snapshot().roots[0]
    assert (root_status.status, root_status.next_action) == (
        "changes_available",
        "review_changes",
    )


@pytest.mark.asyncio
async def test_terminal_refresh_with_no_remaining_work_is_up_to_date() -> None:
    initial = _reviewed_subset_input(conflict_count=1)
    final = _reviewed_subset_input(
        resolved=frozenset({"binding-1"}),
        conflict_count=1,
    )
    assert not any(
        action.kind
        in {NotesSyncActionKind.UPDATE_NOTE, NotesSyncActionKind.UPDATE_FILE}
        for action in plan_reconciliation(final).safe_actions
    )
    assert plan_reconciliation(final).attention == ()
    adapter = _SubsetAdapter(initial, final)
    owner = _owner(adapter, _root())
    token = _install_review(owner, initial)

    result = await owner.apply_reviewed(
        "root-1",
        token,
        (),
        (ConflictSelection("binding-1", NotesSyncConflictChoice.KEEP_FILE),),
    )

    assert result.attention_remains is False
    assert result.fresh_plan == plan_reconciliation(final)
    root_status = owner.snapshot().roots[0]
    assert (root_status.status, root_status.next_action) == (
        "up_to_date",
        "sync_now",
    )


@pytest.mark.asyncio
async def test_safe_actions_run_in_plan_order_before_conflicts_in_binding_order() -> (
    None
):
    initial = _reviewed_subset_input(conflict_count=3)
    final = _reviewed_subset_input(
        resolved=frozenset({"binding-1", "binding-2"}),
        conflict_count=3,
    )
    adapter = _SubsetAdapter(initial, final)
    owner = _owner(adapter, _root())
    token = _install_review(owner, initial)
    safe_id = owner._reviews["root-1"].safe_actions[0].action_id

    result = await owner.apply_reviewed(
        "root-1",
        token,
        (safe_id,),
        (
            ConflictSelection("binding-2", NotesSyncConflictChoice.KEEP_NOTE),
            ConflictSelection("binding-3", NotesSyncConflictChoice.SKIP),
            ConflictSelection("binding-1", NotesSyncConflictChoice.KEEP_FILE),
        ),
    )

    assert [request.binding_id for request in adapter.executor.requests] == [
        "binding-safe",
        "binding-1",
        "binding-2",
    ]
    assert result.safe_completed == 1
    assert result.conflicts_resolved == 2
    assert result.unresolved_conflicts == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "direction",
    (
        NotesSyncDirection.BIDIRECTIONAL,
        NotesSyncDirection.FOLDER_TO_NOTES,
        NotesSyncDirection.NOTES_TO_FOLDER,
    ),
)
async def test_reviewed_conflict_apply_never_changes_root_direction(
    direction: NotesSyncDirection,
) -> None:
    initial = _reviewed_subset_input(direction=direction)
    final = _reviewed_subset_input(
        direction=direction,
        resolved=frozenset({"binding-1"}),
    )
    adapter = _SubsetAdapter(initial, final)
    root = _root(direction=direction)
    owner = _owner(adapter, root)
    token = _install_review(owner, initial)

    await owner.apply_reviewed(
        "root-1",
        token,
        (),
        (ConflictSelection("binding-1", NotesSyncConflictChoice.KEEP_NOTE),),
    )

    assert owner._store.get_root("root-1").direction is direction


@pytest.mark.asyncio
async def test_nonterminal_conflict_stops_later_work_and_reports_honest_partial() -> (
    None
):
    initial = _reviewed_subset_input(conflict_count=3)
    final = _reviewed_subset_input(conflict_count=3)
    adapter = _SubsetAdapter(initial, final, stop_binding_id="binding-1")
    owner = _owner(adapter, _root())
    token = _install_review(owner, initial)
    safe_id = owner._reviews["root-1"].safe_actions[0].action_id

    result = await owner.apply_reviewed(
        "root-1",
        token,
        (safe_id,),
        (
            ConflictSelection("binding-2", NotesSyncConflictChoice.KEEP_NOTE),
            ConflictSelection("binding-1", NotesSyncConflictChoice.KEEP_FILE),
        ),
    )

    assert [request.binding_id for request in adapter.executor.requests] == [
        "binding-safe",
        "binding-1",
    ]
    assert result.safe_completed == 1
    assert result.conflicts_resolved == 0
    assert result.partial is True
    assert result.needs_recovery is True
    assert result.fresh_plan is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    (
        "deletion_group",
        "deletion_attention",
        "pause",
        "managed_placement",
        "root_skip",
        "capability_skip",
        "ineligible_reason",
    ),
)
async def test_non_content_blockers_refuse_before_recovery_admission(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    import tldw_chatbook.Notes.notes_sync_runtime as runtime_module

    observed = _input(file_digest=_B, note_digest=_C)
    plan = plan_reconciliation(observed)
    deletion = ReconciliationAttention(
        ReconciliationAttentionKind.DELETION_REVIEW,
        "file_missing",
        "binding-1",
    )
    if case == "deletion_group":
        plan = replace(
            plan, attention=(), deletion_groups=(DeletionGroup((deletion,)),)
        )
    elif case == "deletion_attention":
        plan = replace(plan, attention=(deletion,))
    elif case == "pause":
        plan = replace(
            plan,
            attention=(
                ReconciliationAttention(
                    ReconciliationAttentionKind.PAUSE,
                    "duplicate_authority",
                    "binding-1",
                ),
            ),
        )
    elif case == "managed_placement":
        plan = replace(
            plan,
            managed_placement_effects=(
                ManagedPlacementEffect(
                    ManagedPlacementEffectKind.FILE_MOVE,
                    "binding-1",
                ),
            ),
        )
    elif case in {"root_skip", "capability_skip"}:
        plan = replace(
            plan,
            attention=(),
            skips=(
                ReconciliationSkip(
                    (
                        ReconciliationSkipKind.OFFLINE
                        if case == "root_skip"
                        else ReconciliationSkipKind.CAPABILITY
                    ),
                    "root_offline" if case == "root_skip" else "write_unsupported",
                ),
            ),
        )
    else:
        plan = replace(
            plan,
            attention=(
                ReconciliationAttention(
                    ReconciliationAttentionKind.CONFLICT,
                    "duplicate_authority",
                    "binding-1",
                ),
            ),
        )
    adapter = _SubsetAdapter(observed, observed)
    owner = _owner(adapter, _root())
    owner._reviews["root-1"] = plan
    monkeypatch.setattr(runtime_module, "plan_reconciliation", lambda _value: plan)

    with pytest.raises(ValueError, match="not_executable"):
        await owner.apply_reviewed(
            "root-1",
            plan.observation_token,
            (),
            (ConflictSelection("binding-1", NotesSyncConflictChoice.KEEP_FILE),),
        )

    assert adapter.executor.requests == []
    assert adapter.live_bundles == set()


@pytest.mark.asyncio
async def test_activation_review_refuses_before_recovery_admission() -> None:
    observed = _input(file_digest=_B, note_digest=_C)
    root = replace(_root(), last_status_code="migration_review_required")
    adapter = _SubsetAdapter(observed, observed)
    owner = _owner(adapter, root)
    token = _install_review(owner, observed)

    with pytest.raises(ValueError, match="not_executable"):
        await owner.apply_reviewed(
            "root-1",
            token,
            (),
            (ConflictSelection("binding-1", NotesSyncConflictChoice.KEEP_FILE),),
        )

    assert adapter.executor.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ("unknown", "duplicate"))
async def test_invalid_conflict_selection_refuses_before_recovery_admission(
    case: str,
) -> None:
    observed = _input(file_digest=_B, note_digest=_C)
    adapter = _SubsetAdapter(observed, observed)
    owner = _owner(adapter, _root())
    token = _install_review(owner, observed)
    selection = ConflictSelection(
        "binding-other" if case == "unknown" else "binding-1",
        NotesSyncConflictChoice.KEEP_FILE,
    )
    selections = (selection, selection) if case == "duplicate" else (selection,)

    with pytest.raises(ValueError, match="selection"):
        await owner.apply_reviewed("root-1", token, (), selections)

    assert adapter.executor.requests == []
    assert adapter.live_bundles == set()


@pytest.mark.asyncio
async def test_apply_requires_final_observation_token_even_if_plan_claims_equality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_chatbook.Notes.notes_sync_runtime as runtime_module

    observed = _input(file_digest=_B, note_digest=_C)
    adapter = _SubsetAdapter(observed, observed)
    owner = _owner(adapter, _root())
    reviewed = plan_reconciliation(observed)
    owner._reviews["root-1"] = reviewed
    stale_token = hashlib.sha256(b"stale-apply").hexdigest()

    class _EqualPlan:
        root_id = reviewed.root_id
        observation_token = stale_token
        safe_actions = reviewed.safe_actions
        attention = reviewed.attention
        skips = reviewed.skips
        managed_placement_effects = reviewed.managed_placement_effects
        deletion_groups = reviewed.deletion_groups

        def __eq__(self, _other: object) -> bool:
            return True

    monkeypatch.setattr(
        runtime_module, "plan_reconciliation", lambda _value: _EqualPlan()
    )

    with pytest.raises(ValueError, match="stale_review"):
        await owner.apply_reviewed("root-1", reviewed.observation_token, (), ())

    assert adapter.executor.requests == []
    assert adapter.live_bundles == set()
