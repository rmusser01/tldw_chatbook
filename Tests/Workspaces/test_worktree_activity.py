"""Complete activity claims follow native physical ownership (ADR-285)."""

import threading
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.Agents.execution_capacity import (
    CapacityRefused,
    RuntimeCapacity,
    WorkOrigin,
)
from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain
from tldw_chatbook.Utils.path_validation import validate_path
from tldw_chatbook.Workspaces.worktree_activity import (
    WorktreeActivity,
    WorktreeFootprint,
    WorktreeScope,
)


def _scope(root, relative=".", *, subtree=False):
    chain = capture_directory_chain(root)
    target = validate_path(relative, chain.canonical_root, allow_hidden=True)
    return WorktreeScope(path=target, subtree=subtree, root=chain)


@pytest.fixture
def native():
    capacity = RuntimeCapacity()
    owners = []

    def begin(*, child=False):
        owner = capacity.begin_execution(
            origin=WorkOrigin.MANUAL, conversation_id="c", child=child
        )
        owners.append(owner)
        return owner

    yield capacity, begin
    for owner in owners:
        owner.finish_root()


@pytest.mark.parametrize(
    ("first_write", "second_write", "allowed"),
    [
        (False, False, True),
        (True, False, False),
        (False, True, False),
        (True, True, False),
    ],
)
def test_readers_share_but_any_writer_conflicts(
    tmp_path, native, first_write, second_write, allowed
):
    _, begin = native
    activity = WorktreeActivity()
    scope = _scope(tmp_path, "file")
    first = WorktreeFootprint(**{"writes" if first_write else "reads": (scope,)})
    second = WorktreeFootprint(**{"writes" if second_write else "reads": (scope,)})
    assert activity.try_acquire(owner=begin(), footprint=first) is not None
    assert (
        activity.try_acquire(owner=begin(), footprint=second) is not None
    ) is allowed


@pytest.mark.parametrize(
    "relative,conflict", [("new", True), ("child/new", True), ("../other", False)]
)
def test_directory_listing_conflicts_with_creates_below_it(
    tmp_path, native, relative, conflict
):
    _, begin = native
    directory = tmp_path / "directory"
    directory.mkdir()
    listing = WorktreeFootprint(reads=(_scope(tmp_path, "directory", subtree=True),))
    activity = WorktreeActivity()
    assert activity.try_acquire(owner=begin(), footprint=listing)
    create = WorktreeFootprint(
        writes=(_scope(tmp_path, str(Path("directory") / relative)),)
    )
    assert (activity.try_acquire(owner=begin(), footprint=create) is None) is conflict


def test_exact_directory_scope_does_not_claim_unrelated_descendants(tmp_path, native):
    _, begin = native
    activity = WorktreeActivity()
    assert activity.try_acquire(
        owner=begin(), footprint=WorktreeFootprint(reads=(_scope(tmp_path),))
    )
    assert activity.try_acquire(
        owner=begin(), footprint=WorktreeFootprint(writes=(_scope(tmp_path, "file"),))
    )


def test_alias_and_nested_root_claim_the_same_canonical_target(tmp_path, native):
    _, begin = native
    root = tmp_path / "root"
    nested = root / "nested"
    nested.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    activity = WorktreeActivity()
    assert activity.try_acquire(
        owner=begin(),
        footprint=WorktreeFootprint(writes=(_scope(alias, "nested/file"),)),
    )
    assert (
        activity.try_acquire(
            owner=begin(), footprint=WorktreeFootprint(reads=(_scope(nested, "file"),))
        )
        is None
    )
    assert (
        activity.try_acquire(
            owner=begin(),
            footprint=WorktreeFootprint(reads=(_scope(root, subtree=True),)),
        )
        is None
    )


def test_refused_multi_root_admission_leaves_no_partial_claim(tmp_path, native):
    _, begin = native
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    activity = WorktreeActivity()
    assert activity.try_acquire(
        owner=begin(), footprint=WorktreeFootprint(writes=(_scope(b, subtree=True),))
    )
    assert (
        activity.try_acquire(
            owner=begin(),
            footprint=WorktreeFootprint(
                writes=(_scope(a, subtree=True), _scope(b, subtree=True))
            ),
        )
        is None
    )
    assert activity.try_acquire(
        owner=begin(), footprint=WorktreeFootprint(writes=(_scope(a, subtree=True),))
    )


def test_simultaneous_multi_root_admission_has_one_complete_winner(tmp_path, native):
    _, begin = native
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    scopes = (_scope(a, subtree=True), _scope(b, subtree=True))
    owners = [begin(), begin()]
    barrier = threading.Barrier(3)
    activity = WorktreeActivity()
    claims = []

    def acquire(owner, roots):
        barrier.wait(timeout=5)
        claims.append(
            activity.try_acquire(owner=owner, footprint=WorktreeFootprint(writes=roots))
        )

    workers = [
        threading.Thread(
            target=acquire, args=(owners[i], scopes[:: 1 if i == 0 else -1])
        )
        for i in range(2)
    ]
    for worker in workers:
        worker.start()
    barrier.wait(timeout=5)
    for worker in workers:
        worker.join(5)
        assert not worker.is_alive()
    assert sum(claim is not None for claim in claims) == 1


@pytest.mark.parametrize("opaque_first", [False, True])
def test_opaque_conflicts_runtime_wide_even_on_disjoint_roots(
    tmp_path, native, opaque_first
):
    _, begin = native
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    opaque = WorktreeFootprint(writes=(_scope(a, subtree=True),), opaque=True)
    ordinary = WorktreeFootprint(reads=(_scope(b, "file"),))
    first, second = (opaque, ordinary) if opaque_first else (ordinary, opaque)
    activity = WorktreeActivity()
    assert activity.try_acquire(owner=begin(), footprint=first)
    assert activity.try_acquire(owner=begin(), footprint=second) is None


@pytest.mark.parametrize("settle_first", [False, True])
def test_release_needs_positive_drain_and_durable_settlement(
    tmp_path, native, settle_first
):
    _, begin = native
    activity = WorktreeActivity()
    footprint = WorktreeFootprint(writes=(_scope(tmp_path, subtree=True),))
    owner = begin()
    claim = activity.try_acquire(owner=owner, footprint=footprint)
    contender = begin()
    claim.settle(durable=False)
    if settle_first:
        claim.settle(durable=True)
    else:
        owner.finish_root()
    assert activity.try_acquire(owner=contender, footprint=footprint) is None
    if settle_first:
        owner.finish_root()
    else:
        claim.settle(durable=True)
    claim.settle(durable=True)
    owner.finish_root()
    assert activity.try_acquire(owner=contender, footprint=footprint)


def test_cleanup_unproven_never_releases_even_after_settlement(tmp_path, native):
    _, begin = native
    activity = WorktreeActivity()
    footprint = WorktreeFootprint(writes=(_scope(tmp_path, subtree=True),))
    owner = begin()
    claim = activity.try_acquire(owner=owner, footprint=footprint)
    owner.mark_cleanup_unproven()
    claim.settle(durable=True)
    owner.finish_root()
    assert activity.try_acquire(owner=begin(), footprint=footprint) is None


def test_prelaunch_reserves_before_child_capacity_and_unwinds_refusal(tmp_path):
    capacity = RuntimeCapacity(max_child_executions=1, reserved_manual_children=0)
    occupied = capacity.begin_execution(
        origin=WorkOrigin.MANUAL, conversation_id="busy", child=True
    )
    activity = WorktreeActivity()
    footprint = WorktreeFootprint(writes=(_scope(tmp_path, subtree=True),))
    try:
        with (
            pytest.raises(CapacityRefused),
            activity.prelaunch(footprint=footprint) as claim,
        ):
            assert claim is not None
            assert capacity.snapshot().child_executions == 1
            with activity.prelaunch(footprint=footprint) as conflicting:
                assert conflicting is None
            claim.bind_owner(
                capacity.begin_execution(
                    origin=WorkOrigin.MANUAL, conversation_id="next", child=True
                )
            )
        with activity.prelaunch(footprint=footprint) as retry:
            assert retry is not None
    finally:
        occupied.finish_root()


def test_bound_prelaunch_claim_survives_scope_exit_and_exception(tmp_path, native):
    _, begin = native
    activity = WorktreeActivity()
    footprint = WorktreeFootprint(writes=(_scope(tmp_path, subtree=True),))
    owner = begin(child=True)
    with (
        pytest.raises(RuntimeError, match="failed start"),
        activity.prelaunch(footprint=footprint) as claim,
    ):
        claim.bind_owner(owner)
        raise RuntimeError("failed start")
    with activity.prelaunch(footprint=footprint) as refused:
        assert refused is None
    owner.finish_root()
    claim.settle(durable=True)
    with activity.prelaunch(footprint=footprint) as retry:
        assert retry is not None


def test_timed_out_tool_keeps_claim_until_its_physical_worker_finishes(
    tmp_path, native
):
    from tldw_chatbook.Agents.agent_models import ToolResult
    from tldw_chatbook.Agents.agent_service import _call_with_timeout

    _, begin = native
    activity = WorktreeActivity()
    footprint = WorktreeFootprint(writes=(_scope(tmp_path, subtree=True),))
    owner = begin()
    claim = activity.try_acquire(owner=owner, footprint=footprint)
    entered, release = threading.Event(), threading.Event()
    workers = []

    def tool():
        workers.append(threading.current_thread())
        entered.set()
        assert release.wait(5)
        return ToolResult(ok=True, content="finished")

    try:
        result = _call_with_timeout(tool, 0.02, "held", owner=owner)
        assert not result.ok and "timed out" in result.error
        assert entered.wait(5)
        claim.settle(durable=True)
        owner.finish_root()
        assert activity.try_acquire(owner=begin(), footprint=footprint) is None
    finally:
        release.set()
        for worker in workers:
            worker.join(5)
            assert not worker.is_alive()
    assert activity.try_acquire(owner=begin(), footprint=footprint)


@pytest.mark.bootstrap_profile
def test_runtime_activity_is_lazy_shared_and_cannot_be_created_after_disposal():
    import asyncio

    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

    runtime = ConsoleRuntime(SimpleNamespace())
    assert runtime._worktree_activity is None
    activity = runtime.worktree_activity
    assert runtime.worktree_activity is activity
    asyncio.run(runtime.dispose())
    assert runtime.worktree_activity is activity
    fresh = ConsoleRuntime(SimpleNamespace())
    asyncio.run(fresh.dispose())
    with pytest.raises(RuntimeError, match="disposed"):
        _ = fresh.worktree_activity


def test_scope_and_footprint_are_immutable(tmp_path):
    scope = _scope(tmp_path)
    footprint = WorktreeFootprint(reads=(scope,))
    with pytest.raises(FrozenInstanceError):
        scope.subtree = True
    with pytest.raises(FrozenInstanceError):
        footprint.opaque = True


def test_parent_settlement_cannot_substitute_for_drained_child_journal(
    tmp_path, native
):
    _, begin = native
    activity = WorktreeActivity()
    footprint = WorktreeFootprint(writes=(_scope(tmp_path, subtree=True),))
    parent, child = begin(), begin(child=True)
    claim = activity.try_acquire(owner=parent, footprint=footprint)
    child_settlement = claim.attach_child(child)
    claim.settle(durable=True)
    parent.finish_root()
    child.finish_root()
    assert activity.try_acquire(owner=begin(), footprint=footprint) is None
    child_settlement.settle(durable=False)
    assert activity.try_acquire(owner=begin(), footprint=footprint) is None
    child_settlement.settle(durable=True)
    assert activity.try_acquire(owner=begin(), footprint=footprint)


def test_child_worker_holds_settled_parent_claim_until_positive_drain(tmp_path, native):
    _, begin = native
    activity = WorktreeActivity()
    footprint = WorktreeFootprint(writes=(_scope(tmp_path, subtree=True),))
    parent, child = begin(), begin(child=True)
    claim = activity.try_acquire(owner=parent, footprint=footprint)
    child_settlement = claim.attach_child(child)
    operation = child.reserve_tool()
    parent.finish_root()
    claim.settle(durable=True)
    child.finish_root()
    child_settlement.settle(durable=True)
    try:
        assert activity.try_acquire(owner=begin(), footprint=footprint) is None
    finally:
        operation.finish()
    assert activity.try_acquire(owner=begin(), footprint=footprint)


def test_cleanup_unproven_child_keeps_entire_tree_claim(tmp_path, native):
    _, begin = native
    activity = WorktreeActivity()
    footprint = WorktreeFootprint(writes=(_scope(tmp_path, subtree=True),))
    parent, child = begin(), begin(child=True)
    claim = activity.try_acquire(owner=parent, footprint=footprint)
    settlement = claim.attach_child(child)
    parent.finish_root()
    claim.settle(durable=True)
    child.mark_cleanup_unproven()
    child.finish_root()
    settlement.settle(durable=True)
    assert activity.try_acquire(owner=begin(), footprint=footprint) is None


def test_late_drain_callback_registration_does_not_release_before_settlement(
    tmp_path, native, monkeypatch
):
    _, begin = native
    owner = begin()
    original_on_drained = owner.on_drained

    def drain_before_registration(callback):
        owner.finish_root()
        original_on_drained(callback)

    monkeypatch.setattr(owner, "on_drained", drain_before_registration)
    activity = WorktreeActivity()
    footprint = WorktreeFootprint(writes=(_scope(tmp_path, subtree=True),))
    claim = activity.try_acquire(owner=owner, footprint=footprint)
    assert activity.try_acquire(owner=begin(), footprint=footprint) is None
    claim.settle(durable=True)
    assert activity.try_acquire(owner=begin(), footprint=footprint)


def test_unbound_claim_cannot_attach_settle_or_bind_after_context_exit(
    tmp_path, native
):
    _, begin = native
    activity = WorktreeActivity()
    footprint = WorktreeFootprint(writes=(_scope(tmp_path, subtree=True),))
    with activity.prelaunch(footprint=footprint) as claim:
        with pytest.raises(ValueError):
            claim.settle(durable=True)
        with pytest.raises(ValueError):
            claim.attach_child(begin(child=True))
    with pytest.raises(ValueError):
        claim.bind_owner(begin())


def test_bound_claim_cannot_rebind_or_attach_nonchild_or_foreign_capacity(
    tmp_path, native
):
    _, begin = native
    activity = WorktreeActivity()
    claim = activity.try_acquire(
        owner=begin(),
        footprint=WorktreeFootprint(writes=(_scope(tmp_path, subtree=True),)),
    )
    with pytest.raises(ValueError):
        claim.bind_owner(begin())
    with pytest.raises(ValueError):
        claim.attach_child(begin())
    foreign_capacity = RuntimeCapacity()
    foreign = foreign_capacity.begin_execution(
        origin=WorkOrigin.MANUAL, conversation_id="foreign", child=True
    )
    try:
        with pytest.raises(ValueError):
            claim.attach_child(foreign)
    finally:
        foreign.finish_root()


def test_duplicate_child_attachment_returns_same_settlement(tmp_path, native):
    _, begin = native
    activity = WorktreeActivity()
    footprint = WorktreeFootprint(writes=(_scope(tmp_path, subtree=True),))
    parent, child = begin(), begin(child=True)
    claim = activity.try_acquire(owner=parent, footprint=footprint)
    first = claim.attach_child(child)
    assert claim.attach_child(child) is first
    claim.settle(durable=True)
    first.settle(durable=True)
    parent.finish_root()
    child.finish_root()
    with pytest.raises(ValueError):
        claim.attach_child(begin(child=True))
    assert activity.try_acquire(owner=begin(), footprint=footprint)


def test_file_hardlinks_conflict_even_with_disjoint_admitted_roots(tmp_path, native):
    _, begin = native
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "file").write_text("shared inode")
    (b / "link").hardlink_to(a / "file")
    activity = WorktreeActivity()
    assert activity.try_acquire(
        owner=begin(), footprint=WorktreeFootprint(writes=(_scope(a, "file"),))
    )
    assert (
        activity.try_acquire(
            owner=begin(), footprint=WorktreeFootprint(reads=(_scope(b, "link"),))
        )
        is None
    )


def test_captured_root_identity_reconciles_rename(tmp_path, native):
    _, begin = native
    original, renamed = tmp_path / "original", tmp_path / "renamed"
    original.mkdir()
    before = _scope(original, "file")
    original.rename(renamed)
    after = _scope(renamed, "file")
    activity = WorktreeActivity()
    assert activity.try_acquire(
        owner=begin(), footprint=WorktreeFootprint(writes=(before,))
    )
    assert (
        activity.try_acquire(owner=begin(), footprint=WorktreeFootprint(reads=(after,)))
        is None
    )


def test_case_aliases_conservatively_serialize(tmp_path, native):
    _, begin = native
    activity = WorktreeActivity()
    assert activity.try_acquire(
        owner=begin(), footprint=WorktreeFootprint(writes=(_scope(tmp_path, "Case"),))
    )
    assert (
        activity.try_acquire(
            owner=begin(),
            footprint=WorktreeFootprint(reads=(_scope(tmp_path, "case"),)),
        )
        is None
    )


def test_scope_rejects_relative_or_unconfined_paths(tmp_path):
    root = capture_directory_chain(tmp_path)
    for path in (
        Path("relative"),
        tmp_path.parent / "outside",
        tmp_path / ".." / "escape",
    ):
        with pytest.raises(ValueError):
            WorktreeScope(path=path, subtree=False, root=root)


def test_unknown_target_metadata_refuses_scope(tmp_path, monkeypatch):
    import tldw_chatbook.Workspaces.worktree_activity as module

    root = capture_directory_chain(tmp_path)
    target = tmp_path / "unknown"
    real_lstat = module.os.lstat

    def lstat(path):
        if path == target:
            raise PermissionError("unavailable")
        return real_lstat(path)

    monkeypatch.setattr(module.os, "lstat", lstat)
    with pytest.raises(ValueError, match="metadata"):
        WorktreeScope(path=target, subtree=False, root=root)


def test_never_bound_exception_or_invalid_owner_unwinds_only_its_reservation(
    tmp_path, native
):
    _, begin = native
    activity = WorktreeActivity()
    first = WorktreeFootprint(writes=(_scope(tmp_path, "held"),))
    second = WorktreeFootprint(writes=(_scope(tmp_path, "new"),))
    assert activity.try_acquire(owner=begin(), footprint=first)
    with pytest.raises(RuntimeError), activity.prelaunch(footprint=second) as claim:
        assert claim is not None
        raise RuntimeError("no launch")
    with pytest.raises(TypeError), activity.prelaunch(footprint=second) as claim:
        claim.bind_owner(object())
    with activity.prelaunch(footprint=second) as retry:
        assert retry is not None
    assert activity.try_acquire(owner=begin(), footprint=first) is None


def test_prelaunch_success_binds_actual_child_only_after_complete_reservation(tmp_path):
    capacity = RuntimeCapacity(max_child_executions=1, reserved_manual_children=0)
    activity = WorktreeActivity()
    footprint = WorktreeFootprint(writes=(_scope(tmp_path, subtree=True),))
    with activity.prelaunch(footprint=footprint) as claim:
        assert claim is not None
        assert capacity.snapshot().child_executions == 0
        with activity.prelaunch(footprint=footprint) as refused:
            assert refused is None
        owner = capacity.begin_execution(
            origin=WorkOrigin.MANUAL, conversation_id="child", child=True
        )
        claim.bind_owner(owner)
        assert capacity.snapshot().child_executions == 1
    try:
        with activity.prelaunch(footprint=footprint) as refused:
            assert refused is None
        claim.settle(durable=True)
    finally:
        owner.finish_root()
    with activity.prelaunch(footprint=footprint) as retry:
        assert retry is not None


def test_refused_opaque_claim_leaves_disjoint_root_available(tmp_path, native):
    _, begin = native
    activity = WorktreeActivity()
    a, b = _scope(tmp_path, "a"), _scope(tmp_path, "b")
    assert activity.try_acquire(owner=begin(), footprint=WorktreeFootprint(writes=(a,)))
    with activity.prelaunch(
        footprint=WorktreeFootprint(writes=(b,), opaque=True)
    ) as refused:
        assert refused is None
    assert activity.try_acquire(owner=begin(), footprint=WorktreeFootprint(writes=(b,)))


def test_child_draining_before_callback_registration_still_requires_settlement(
    tmp_path, native, monkeypatch
):
    _, begin = native
    activity = WorktreeActivity()
    footprint = WorktreeFootprint(writes=(_scope(tmp_path, subtree=True),))
    parent, child = begin(), begin(child=True)
    claim = activity.try_acquire(owner=parent, footprint=footprint)
    original_on_drained = child.on_drained

    def drain_before_registration(callback):
        child.finish_root()
        original_on_drained(callback)

    monkeypatch.setattr(child, "on_drained", drain_before_registration)
    settlement = claim.attach_child(child)
    parent.finish_root()
    claim.settle(durable=True)
    assert activity.try_acquire(owner=begin(), footprint=footprint) is None
    settlement.settle(durable=True)
    assert activity.try_acquire(owner=begin(), footprint=footprint)


def test_path_replacement_cannot_evade_a_captured_inode_claim(tmp_path, native):
    _, begin = native
    target = tmp_path / "file"
    target.write_text("old")
    old = _scope(tmp_path, "file")
    replacement = tmp_path / "replacement"
    replacement.write_text("new")
    replacement.replace(target)
    activity = WorktreeActivity()
    assert activity.try_acquire(
        owner=begin(), footprint=WorktreeFootprint(writes=(old,))
    )
    assert (
        activity.try_acquire(
            owner=begin(),
            footprint=WorktreeFootprint(reads=(_scope(tmp_path, "file"),)),
        )
        is None
    )


def test_pure_module_import_does_not_load_execution_or_config():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import tldw_chatbook.Workspaces.worktree_activity; assert 'tldw_chatbook.config' not in sys.modules; assert 'tldw_chatbook.Agents.execution_capacity' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_reparse_target_metadata_refuses_scope(tmp_path, monkeypatch):
    from dataclasses import replace

    import tldw_chatbook.Workspaces.worktree_activity as module

    target = tmp_path / "file"
    target.write_text("content")
    root = capture_directory_chain(tmp_path)
    actual = module.directory_identity_from_stat(target.lstat())
    monkeypatch.setattr(
        module,
        "directory_identity_from_stat",
        lambda metadata: replace(actual, reparse=True),
    )
    with pytest.raises(ValueError, match="changed"):
        WorktreeScope(path=target, subtree=False, root=root)


@pytest.mark.parametrize("child", [False, True])
@pytest.mark.parametrize("fabricated", [False, True])
def test_fabricated_or_retired_owners_cannot_bind_or_attach(
    tmp_path, native, child, fabricated
):
    from tldw_chatbook.Agents.execution_capacity import ExecutionOwner

    capacity, begin = native
    owner = (
        ExecutionOwner(capacity, WorkOrigin.MANUAL, "fabricated", child)
        if fabricated
        else begin(child=child)
    )
    if not fabricated:
        owner.finish_root()
    activity = WorktreeActivity()
    footprint = WorktreeFootprint(writes=(_scope(tmp_path, subtree=True),))
    with pytest.raises(ValueError, match="admitted"):
        activity.try_acquire(owner=owner, footprint=footprint)
    parent = begin()
    claim = activity.try_acquire(owner=parent, footprint=footprint)
    assert claim is not None
    if child:
        with pytest.raises(ValueError, match="admitted"):
            claim.attach_child(owner)
    claim.settle(durable=True)
    parent.finish_root()
    with activity.prelaunch(footprint=footprint) as unlaunched:
        assert unlaunched is not None


@pytest.mark.parametrize("subtree", [False, True])
@pytest.mark.parametrize("rename_root", [False, True])
def test_absent_unicode_aliases_conflict_without_changing_admitted_paths(
    tmp_path, native, subtree, rename_root
):
    _, begin = native
    root = tmp_path / "root"
    root.mkdir()
    first_name = "\u00e9" if subtree else "\u00e9.txt"
    second_name = "e\u0301/child.txt" if subtree else "e\u0301.txt"
    first = _scope(root, first_name, subtree=subtree)
    original_path = first.path
    if rename_root:
        renamed = tmp_path / "renamed"
        root.rename(renamed)
        root = renamed
    second = _scope(root, second_name)
    assert not first.path.exists()
    assert not second.path.exists()
    assert first.path == original_path
    assert second.path == root.resolve() / second_name
    activity = WorktreeActivity()
    assert activity.try_acquire(
        owner=begin(), footprint=WorktreeFootprint(writes=(first,))
    )
    assert (
        activity.try_acquire(
            owner=begin(), footprint=WorktreeFootprint(writes=(second,))
        )
        is None
    )
