"""Runtime recovery denials and actual physical release notifications."""

import pytest

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]


async def test_unknown_recovery_barrier_blocks_alias_without_execution_grant(tmp_path):
    from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain
    from tldw_chatbook.Workspaces.worktree_activity import (
        WorktreeActivity,
        WorktreeFootprint,
        WorktreeScope,
    )

    root = tmp_path / "root"
    root.mkdir()
    nested = root / "nested"
    nested.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    activity = WorktreeActivity()
    chain = capture_directory_chain(root)
    barrier = WorktreeFootprint(writes=(WorktreeScope(root, True, chain),))
    activity.protect_recovery("closed-unknown", barrier)
    activity.protect_recovery("closed-unknown", barrier)
    nested_chain = capture_directory_chain(alias / "nested")
    with activity.prelaunch(
        footprint=WorktreeFootprint(
            reads=(
                WorktreeScope(
                    nested_chain.canonical_root / "data", False, nested_chain
                ),
            )
        )
    ) as claim:
        assert claim is None
    assert activity.recovery_blockers(barrier) == ("closed-unknown",)
    unrelated = tmp_path / "other"
    unrelated.mkdir()
    other_chain = capture_directory_chain(unrelated)
    with activity.prelaunch(
        footprint=WorktreeFootprint(
            writes=(WorktreeScope(unrelated, True, other_chain),)
        )
    ) as claim:
        assert claim is not None


async def test_opaque_recovery_barrier_also_blocks_scratch_without_an_owner():
    from tldw_chatbook.Workspaces.worktree_activity import (
        WorktreeActivity,
        WorktreeFootprint,
    )

    activity = WorktreeActivity()
    activity.protect_recovery("unknown-opaque", WorktreeFootprint(opaque=True))
    with activity.prelaunch(footprint=WorktreeFootprint()) as claim:
        assert claim is None


async def test_activity_release_event_is_outside_lock_and_waits_for_both_proofs(
    tmp_path,
):
    from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity, WorkOrigin
    from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain
    from tldw_chatbook.Workspaces.worktree_activity import (
        WorktreeActivity,
        WorktreeFootprint,
        WorktreeScope,
    )

    activity = WorktreeActivity()
    capacity = RuntimeCapacity()
    footprint = WorktreeFootprint(
        writes=(WorktreeScope(tmp_path, True, capture_directory_chain(tmp_path)),)
    )
    owner = capacity.begin_execution(
        origin=WorkOrigin.MANUAL, conversation_id="ordinary"
    )
    observed = []

    def changed():
        observed.append(activity.recovery_blockers(footprint))

    unsubscribe = activity.subscribe(changed)
    with activity.prelaunch(footprint=footprint) as claim:
        assert claim is not None
        claim.bind_owner(owner)
    owner.finish_root()
    assert observed == []
    claim.settle(durable=True)
    assert observed == [()]
    claim.settle(durable=True)
    assert observed == [()]
    unsubscribe()


async def test_reviewed_history_release_preserves_the_actual_native_claim(tmp_path):
    from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity, WorkOrigin
    from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain
    from tldw_chatbook.Workspaces.worktree_activity import (
        WorktreeActivity,
        WorktreeFootprint,
        WorktreeScope,
    )

    activity = WorktreeActivity()
    capacity = RuntimeCapacity()
    footprint = WorktreeFootprint(
        writes=(WorktreeScope(tmp_path, True, capture_directory_chain(tmp_path)),)
    )
    owner = capacity.begin_execution(origin=WorkOrigin.MANUAL, conversation_id="live")
    with activity.prelaunch(footprint=footprint) as live:
        assert live is not None
        live.bind_owner(owner)
    activity.protect_recovery("historical", footprint)
    assert activity.release_recovery("historical") is True
    assert activity.release_recovery("historical") is False
    assert activity.recovery_blockers(footprint) == ()
    with activity.prelaunch(footprint=footprint) as blocked:
        assert blocked is None
    owner.finish_root()
    live.settle(durable=True)
    with activity.prelaunch(footprint=footprint) as available:
        assert available is not None


@pytest.mark.parametrize("replacement", ["directory", "symlink", "missing", "file"])
@pytest.mark.parametrize("gate", ["prelaunch", "blockers"])
@pytest.mark.parametrize("native_live", [False, True])
async def test_stale_history_retires_without_releasing_a_native_claim(
    tmp_path, replacement, gate, native_live
):
    from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity, WorkOrigin
    from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain
    from tldw_chatbook.Workspaces.worktree_activity import (
        WorktreeActivity,
        WorktreeFootprint,
        WorktreeScope,
    )

    root = tmp_path / "project"
    root.mkdir()
    activity = WorktreeActivity()
    prior = WorktreeFootprint(
        writes=(WorktreeScope(root, True, capture_directory_chain(root)),)
    )
    owner = claim = None
    if native_live:
        owner = RuntimeCapacity().begin_execution(
            origin=WorkOrigin.MANUAL, conversation_id="still-live"
        )
        with activity.prelaunch(footprint=prior) as claim:
            claim.bind_owner(owner)
    activity.protect_recovery("unreviewed-history", prior)
    root.rename(tmp_path / "old-project")
    if replacement == "directory":
        root.mkdir()
    elif replacement == "symlink":
        target = tmp_path / "new-project"
        target.mkdir()
        root.symlink_to(target, target_is_directory=True)
    if replacement == "file":
        root.write_text("replacement file")
    if replacement in {"missing", "file"}:
        requested = WorktreeFootprint(opaque=True)
    else:
        chain = capture_directory_chain(root)
        requested = WorktreeFootprint(
            writes=(WorktreeScope(chain.canonical_root, True, chain),)
        )
    notifications = []

    def changed():
        assert activity._lock.acquire(blocking=False), (
            "recovery notification ran under the admission lock"
        )
        activity._lock.release()
        notifications.append(True)

    unsubscribe = activity.subscribe(changed)
    try:
        if gate == "blockers":
            assert activity.recovery_blockers(requested) == ()
        with activity.prelaunch(footprint=requested) as admitted:
            assert (admitted is None) is (native_live and replacement != "symlink")
        assert notifications and not activity._recovery
        if native_live:
            assert owner.is_admitted and claim.id in activity._claims
    finally:
        unsubscribe()
        if owner is not None:
            owner.finish_root()
            claim.settle(durable=True)


async def test_stale_history_scope_does_not_drop_other_current_roots(tmp_path):
    from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain
    from tldw_chatbook.Workspaces.worktree_activity import (
        WorktreeActivity,
        WorktreeFootprint,
        WorktreeScope,
    )

    old, current = tmp_path / "old", tmp_path / "current"
    old.mkdir()
    current.mkdir()
    old_scope = WorktreeScope(old, True, capture_directory_chain(old))
    current_scope = WorktreeScope(current, True, capture_directory_chain(current))
    activity = WorktreeActivity()
    activity.protect_recovery(
        "history", WorktreeFootprint(writes=(old_scope, current_scope))
    )
    old.rename(tmp_path / "renamed")
    old.mkdir()
    replacement = WorktreeFootprint(
        writes=(WorktreeScope(old, True, capture_directory_chain(old)),)
    )
    assert activity.recovery_blockers(replacement) == ()
    assert activity.recovery_blockers(WorktreeFootprint(reads=(current_scope,))) == (
        "history",
    )


@pytest.mark.parametrize("failure", ["eio", "permission", "one-shot", "value"])
@pytest.mark.parametrize("gate", ["prelaunch", "blockers"])
async def test_unreadable_metadata_preserves_an_established_recovery_fence(
    tmp_path, monkeypatch, failure, gate
):
    import errno
    import os
    from pathlib import Path

    from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain
    from tldw_chatbook.Workspaces import worktree_activity

    root = tmp_path / "project"
    root.mkdir()
    chain = capture_directory_chain(root)
    footprint = worktree_activity.WorktreeFootprint(
        writes=(worktree_activity.WorktreeScope(chain.canonical_root, True, chain),)
    )
    activity = worktree_activity.WorktreeActivity()
    activity.protect_recovery("unreviewed", footprint)
    original = os.lstat
    failed = False

    def unreadable(path, *args, **kwargs):
        nonlocal failed
        if Path(path) == chain.canonical_root and (failure != "one-shot" or not failed):
            failed = True
            if failure == "permission":
                raise PermissionError(errno.EACCES, "metadata temporarily unreadable")
            raise OSError(errno.EIO, "metadata temporarily unreadable")
        return original(path, *args, **kwargs)

    def invalid_metadata(path):
        if path == chain.canonical_root:
            raise ValueError("directory metadata temporarily incomplete")
        return capture_directory_chain(path)

    with monkeypatch.context() as faults:
        if failure == "value":
            faults.setattr(
                worktree_activity, "capture_directory_chain", invalid_metadata
            )
        else:
            faults.setattr(os, "lstat", unreadable)
        if gate == "blockers":
            assert activity.recovery_blockers(footprint) == ("unreviewed",)
        else:
            with activity.prelaunch(footprint=footprint) as claim:
                assert claim is None
    assert capture_directory_chain(root) == chain
    assert activity.recovery_blockers(footprint) == ("unreviewed",)
    with activity.prelaunch(footprint=footprint) as claim:
        assert claim is None, "metadata recovery silently reviewed prior effects"
