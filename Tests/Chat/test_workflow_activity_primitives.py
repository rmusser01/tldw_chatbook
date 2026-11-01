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
