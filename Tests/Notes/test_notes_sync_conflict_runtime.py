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
    ConflictComparison,
    build_conflict_comparison,
)
from tldw_chatbook.Notes.notes_sync_authority import NotesSyncNoteSnapshot
from tldw_chatbook.Notes.notes_sync_filesystem import NotesSyncFileSnapshot
from tldw_chatbook.Notes.notes_sync_models import (
    NotesSyncDirection,
    NotesSyncFileIdentity,
    NotesSyncFileObservation,
    NotesSyncOperationState,
    NotesSyncRootState,
    NotesSyncSerializationProfile,
)
from tldw_chatbook.Notes.notes_sync_reconciler import (
    BindingObservation,
    ReconciliationInput,
    plan_reconciliation,
)
from tldw_chatbook.Notes.notes_sync_runtime import (
    NotesSyncRuntimeOwner,
    _ProductionRuntimeAdapter,
)
from tldw_chatbook.Notes.sync_paths import SafeSyncBytes, SafeSyncFileIdentity


pytestmark = pytest.mark.unit
_A = "a" * 64
_B = "b" * 64
_C = "c" * 64


def _input(
    root_id: str = "root-1",
    *,
    generation: int = 1,
    file_digest: str = _B,
    note_digest: str = _A,
) -> ReconciliationInput:
    return ReconciliationInput(
        root_id=root_id,
        direction=NotesSyncDirection.BIDIRECTIONAL,
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


def _root(root_id: str = "root-1") -> NotesSyncRootRecord:
    return NotesSyncRootRecord(
        root_id=root_id,
        note_scope_id="local_note",
        logical_folder_id="folder-1",
        canonical_path=f"/private/{root_id}",
        direction=NotesSyncDirection.BIDIRECTIONAL,
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
        return SimpleNamespace(
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
    await adapter.executor.entered.wait()
    second = asyncio.create_task(owner.apply_reviewed("root-1", token, (action_id,)))
    await lock.waiting.wait()

    assert adapter.observe_calls == ["root-1"]
    assert adapter.executor.mutations == 0
    adapter.executor.release.set()
    results = await asyncio.gather(first, second, return_exceptions=True)

    assert adapter.observe_calls == ["root-1", "root-1"]
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
    await adapter.executor.entered.wait()
    reviewed = asyncio.create_task(owner.apply_reviewed("root-1", token, (action_id,)))
    await lock.waiting.wait()

    assert adapter.observe_calls == ["root-1"]
    assert adapter.executor.mutations == 0
    adapter.executor.release.set()
    results = await asyncio.gather(automatic, reviewed, return_exceptions=True)

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
    await adapter.executor.resume_entered.wait()
    reviewed = asyncio.create_task(owner.apply_reviewed("root-1", token, (action_id,)))
    await lock.waiting.wait()

    assert adapter.observe_calls == []
    assert adapter.executor.mutations == 0
    adapter.executor.resume_release.set()
    results = await asyncio.gather(recovery, reviewed, return_exceptions=True)

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
    await holder_entered.wait()
    waiting = asyncio.create_task(waiter())
    await waiter_resolved.wait()
    gc.collect()

    identities.append(id(lock_factory("root-1")))
    assert len(set(identities)) == 1
    assert not waiter_entered.is_set()
    holder_release.set()
    await asyncio.gather(holding, waiting)


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
        await second_entered.wait()
    await second


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

    assert len(result) == 1
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
async def test_comparison_releases_bundle_when_cancelled() -> None:
    observed = _input(file_digest=_B, note_digest=_C)
    adapter = _Adapter(observed)
    adapter.comparison_release = asyncio.Event()
    owner = _owner(adapter, _root())
    token = _install_review(owner, observed)
    task = asyncio.create_task(owner.compare_conflict("root-1", token, "binding-1"))
    await adapter.comparison_started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

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
        adapter.live_bundles.add(stale_token)
        return observed

    adapter.observe_root = observe_with_stale_bundle
    monkeypatch.setattr(
        runtime_module, "plan_reconciliation", lambda _value: _EqualPlan()
    )

    with pytest.raises(ValueError, match="stale_review"):
        await owner.compare_conflict("root-1", reviewed.observation_token, "binding-1")

    assert adapter.comparison_builds == 0
    assert adapter.live_bundles == set()
