"""Reviewed revision transitions use the real service's exact work custody."""

import asyncio

import pytest

from Tests.Plugins.test_revocation_persistence import (
    revocation_case as _revocation_case,
)

revocation_case = _revocation_case


@pytest.mark.asyncio
async def test_revision_drain_closes_new_admission_but_finishes_owned_work(
    revocation_case,
    native_package,
):
    case = revocation_case
    service = case.service
    old = await service._call(service._coordinator.published_snapshot)
    original = next(
        row for row in old["revisions"] if row["installation_id"] == case.installation
    )
    maximum = service.capture_maximum("a")
    assert maximum["available_skills"]
    await service.check_entries(case.entries["a"])
    await service.check_entries(case.entries["b"])
    import json

    from Tests.Chat.test_provider_continuation import _checkpoint
    from tldw_chatbook.Chat.provider_continuation import (
        parse_provider_continuation_json,
    )

    pin = await asyncio.to_thread(
        service.capture_resume_pin,
        case.entries["a"],
        "root-a",
        "conversation",
        "message",
    )
    sealed = await asyncio.to_thread(
        service.seal_resume_checkpoint,
        pin,
        parse_provider_continuation_json(json.dumps(_checkpoint())),
        "conversation",
        "message",
    )
    assert (await service.resume_maximum(maximum, sealed, "conversation", "message"))[
        "available_skills"
    ]
    root = native_package()
    skill = root / "skills/review/SKILL.md"
    skill.write_text(skill.read_text() + "\nREVIEWED NEW REVISION\n")
    review = await service.review_revision(case.installation, root)
    # Merely reviewing does not fence the running revision.
    admitted = await service.admit(maximum, "review-still-open")
    assert admitted["available_skills"]
    apply = asyncio.create_task(service.apply_revision(review, review.operation_id))
    try:
        for _ in range(100):
            if service.revision_drain.blockers(review.drain_token):
                break
            await asyncio.sleep(0.01)
        blockers = service.revision_drain.blockers(review.drain_token)
        assert {row["run_id"] for row in blockers} == {"root-a", "root-b"}
        assert not apply.done()
        assert not case.cancelled_b.is_set()
        assert case.child.poll() is None
        await service.check_entries(case.entries["a"])
        await service.check_entries(case.entries["b"])
        with pytest.raises(PermissionError, match="drain"):
            await service.admit(maximum, "late-admission")
        with pytest.raises(PermissionError, match="drain"):
            await asyncio.to_thread(
                service.bind_run,
                case.entries["a"],
                "late-child",
                lambda: None,
                "handle",
                parent_run_id="root-a",
            )
        apply.cancel()
        with pytest.raises(asyncio.CancelledError):
            await apply
        assert service.revision_drain.blockers(review.drain_token)
        case.child.terminate()
        await asyncio.to_thread(case.child.wait, 2)
        await asyncio.to_thread(service.complete_run, "root-a")
        await asyncio.to_thread(service.complete_run, "root-b")
        receipt = await service.apply_revision(review, review.operation_id)
        assert receipt.committed
        new = await service._call(service._coordinator.published_snapshot)
        installed = next(
            row
            for row in new["installations"]
            if row["installation_id"] == case.installation
        )
        assert installed["revision_digest"] == review.inspection.effective_digest
        retained = next(
            row
            for row in new["revisions"]
            if row["revision_digest"] == original["revision_digest"]
        )
        assert retained == original
        assert (
            installed["alias"]
            == next(
                row
                for row in old["installations"]
                if row["installation_id"] == case.installation
            )["alias"]
        )
        fresh = await service.admit(service.capture_maximum("a"), "after-update")
        assert fresh["available_skills"]
        await service.check_entries(fresh["available_skills"])
        with pytest.raises(PermissionError):
            await service.resume_maximum(
                service.capture_maximum("a"), sealed, "conversation", "message"
            )
        with pytest.raises(PermissionError):
            await service.check_entries(case.entries["a"])
    finally:
        if not apply.done():
            service.revision_drain.cancel(review.drain_token)
            await asyncio.gather(apply, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancel_proposal_does_not_cancel_work_or_another_drain(
    revocation_case, native_package
):
    case = revocation_case
    service = case.service
    root = native_package()
    skill = root / "skills/review/SKILL.md"
    skill.write_text(skill.read_text() + "\nREPLACEMENT\n")
    review = await service.review_revision(case.installation, root)
    maximum = service.capture_maximum("a")
    other = service.revision_drain.begin(case.installation, review.previous_revision)
    task = asyncio.create_task(service.apply_revision(review, review.operation_id))
    try:
        for _ in range(100):
            if service.revision_drain.blockers(review.drain_token):
                break
            await asyncio.sleep(0.01)
        service.revision_drain.cancel(review.drain_token)
        with pytest.raises(ValueError, match="cancelled"):
            await task
        assert case.child.poll() is None and not case.cancelled_b.is_set()
        await service.check_entries(case.entries["a"])
        with pytest.raises(PermissionError, match="drain"):
            await service.admit(maximum, "still-blocked")
        service.revision_drain.cancel(other)
        assert (await service.admit(maximum, "reopened"))["available_skills"]
        with pytest.raises(ValueError, match="cancelled"):
            await service.apply_revision(review, review.operation_id)
    finally:
        if not task.done():
            service.revision_drain.cancel(review.drain_token)
            await asyncio.gather(task, return_exceptions=True)
        service.revision_drain.cancel(other)


@pytest.mark.asyncio
async def test_explicit_cancel_work_retains_survivors_until_confirmed(
    revocation_case, native_package
):
    case = revocation_case
    case.keep_alive()
    root = native_package()
    skill = root / "skills/review/SKILL.md"
    skill.write_text(skill.read_text() + "\nREPLACEMENT\n")
    review = await case.service.review_revision(case.installation, root)
    task = asyncio.create_task(case.service.apply_revision(review, review.operation_id))
    try:
        for _ in range(100):
            if case.service.revision_drain.blockers(review.drain_token):
                break
            await asyncio.sleep(0.01)
        case.service.revision_drain.cancel_work(review.drain_token)
        await asyncio.wait_for(case.cleanup_started.wait(), 2)
        assert case.cancelled_b.is_set() and case.child.poll() is None
        assert not task.done()
        case.child.terminate()
        await asyncio.to_thread(case.child.wait, 2)
        await asyncio.to_thread(case.service.complete_run, "root-a")
        await asyncio.to_thread(case.service.complete_run, "root-b")
        assert (await task).committed
    finally:
        if not task.done():
            case.service.revision_drain.cancel(review.drain_token)
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_rollback_retains_current_disables_and_new_selection_exclusions(
    revocation_case, native_package
):
    from pathlib import Path

    case = revocation_case
    service = case.service
    root = native_package()
    skill = root / "skills/review/SKILL.md"
    skill.write_text(skill.read_text() + "\nREPLACEMENT\n")
    # A newly discovered supported skill must stay outside the old selection.
    import json

    manifest = json.loads((root / "plugin.json").read_text())
    (root / "skills/new").mkdir()
    (root / "skills/new/SKILL.md").write_text(
        "---\nname: new\ndescription: Newly added capability.\n---\nNew instructions.\n"
    )
    (root / "plugin.json").write_text(json.dumps(manifest))
    review = await service.review_revision(case.installation, root)
    assert review.selection == ("skill:review",)
    case.child.terminate()
    await asyncio.to_thread(case.child.wait, 2)
    await asyncio.to_thread(service.complete_run, "root-a")
    await asyncio.to_thread(service.complete_run, "root-b")
    assert (await service.apply_revision(review, review.operation_id)).committed
    disabled = await service.review_activation(
        case.installation, workspace_id="b", intent="disabled"
    )
    assert (await service.commit(disabled, disabled.operation_id)).committed
    rollback = await service.review_rollback(
        case.installation, review.previous_revision
    )
    assert rollback.rollback and rollback.data_compatibility == "unknown"
    assert (await service.apply_revision(rollback, rollback.operation_id)).committed
    assert not service.capture_maximum("b")["available_skills"]
    assert service.capture_maximum("a")["available_skills"]
    snapshot = await service._call(service._coordinator.published_snapshot)
    assert all(
        Path(row["materialized_identity"]).is_dir() for row in snapshot["revisions"]
    )
    assert snapshot["installations"][0]["revision_digest"] == review.previous_revision
