"""Real owner status, selection and surviving view awaiters."""

import asyncio

import pytest

from Tests.Plugins.conftest import native_console as _native_console
from Tests.Plugins.conftest import native_package as _native_package

native_console = _native_console
native_package = _native_package
from Tests.Plugins.test_revocation_persistence import (
    revocation_case as _revocation_case,
)

revocation_case = _revocation_case


@pytest.mark.asyncio
async def test_selection_preserves_untrusted_revision_and_durable_history(
    native_console, native_package
):
    service = native_console.service
    review = await service.review_install(
        native_package(), selection=("skill:review",), workspace_id="workspace-a"
    )
    await service.commit(review, review.operation_id)
    selection = await service.review_selection(review.installation_id, selection=())
    assert selection.kind == "select"
    receipt = await service.commit(selection, selection.operation_id)
    assert receipt.committed
    rows = await service.management_snapshot("workspace-a")
    assert not rows[0].trusted and rows[0].selected_components == ()
    details = await service.management_details(
        review.installation_id, workspace_id="workspace-a"
    )
    assert details["inspection"].effective_digest == review.inspection.effective_digest
    history = await service.list_operations(installation_id=review.installation_id)
    assert any(row["receipt"].operation_id == receipt.operation_id for row in history)


@pytest.mark.asyncio
async def test_detached_presentation_waiter_does_not_cancel_submitted_commit(
    native_console, monkeypatch
):
    from tldw_chatbook.UI.Plugins_Modules.operations import PluginOperations

    service = native_console.service
    installed = await native_console.install(enable=False)
    review = await service.review_activation(
        installed.installation_id, workspace_id="workspace-a", intent="enabled"
    )
    operations = PluginOperations(service)
    release = asyncio.Event()
    commit = service.commit

    async def held(*args):
        await release.wait()
        return await commit(*args)

    monkeypatch.setattr(service, "commit", held)
    job = operations.submit(
        "commit",
        review,
        installation_id=review.installation_id,
        workspace_id="workspace-a",
    )

    async def view():
        await asyncio.shield(job.task)

    waiter = asyncio.create_task(view())
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert not job.task.cancelled() and not job.task.done()
    release.set()
    await job.task
    assert job.receipt.committed
    await operations.refresh()
    assert (await service.management_snapshot("workspace-a"))[0].effective_active


@pytest.mark.asyncio
async def test_reconfirming_same_disable_review_reuses_actual_request(native_console):
    from tldw_chatbook.UI.Plugins_Modules.operations import PluginOperations

    installed = await native_console.install()
    operations = PluginOperations(native_console.service)
    review = await native_console.service.review_activation(
        installed.installation_id, workspace_id="workspace-a", intent="disabled"
    )
    first = operations.submit(
        "disable",
        review,
        installation_id=installed.installation_id,
        workspace_id="workspace-a",
    )
    again = operations.submit(
        "disable",
        review,
        installation_id=installed.installation_id,
        workspace_id="workspace-a",
    )
    try:
        assert first is again
    finally:
        await asyncio.gather(*(job.task for job in operations.jobs.values()))


@pytest.mark.asyncio
@pytest.mark.parametrize("uninstall", [False, True])
async def test_reviewed_removal_retains_original_identity_and_rejects_forgery(
    native_console, uninstall
):
    from dataclasses import replace

    service = native_console.service
    installed = await native_console.install()
    review = await service.review_removal(
        installed.installation_id, uninstall=uninstall
    )
    with pytest.raises(ValueError, match="review"):
        service.begin_reviewed_removal(replace(review, installation_id="other"))
    request = service.begin_reviewed_removal(review)
    assert service.begin_reviewed_removal(review) is request
    receipt = await service.finish_revocation(request)
    assert receipt.committed and receipt.operation_id == review.operation_id
    rows = await service.management_snapshot("workspace-a")
    assert not rows if uninstall else not rows[0].effective_active


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["activation", "selection", "revision"])
async def test_reviewed_removal_never_retargets_changed_authority(
    native_console, native_package, change
):
    from tldw_chatbook.Plugins.revocation import RevocationFailure

    service = native_console.service
    installed = await native_console.install()
    review = await service.review_removal(installed.installation_id, uninstall=True)
    if change == "activation":
        newer = await service.review_activation(
            installed.installation_id, workspace_id="workspace-b", intent="enabled"
        )
    elif change == "selection":
        newer = await service.review_selection(installed.installation_id, selection=())
    else:
        package = native_package()
        (package / "skills/review/SKILL.md").write_text(
            (package / "skills/review/SKILL.md").read_text()
            + "\nNew reviewed revision.\n"
        )
        newer = await service.review_revision(installed.installation_id, package)
    await service.commit(newer, newer.operation_id)
    request = service.begin_reviewed_removal(review)
    with pytest.raises(RevocationFailure):
        await service.finish_revocation(request)
    receipt = service.revocation_status(request)
    assert not receipt.committed and receipt.persistence_error
    assert (
        await service.management_details(
            installed.installation_id, workspace_id="workspace-a"
        )
    )["installation"]


@pytest.mark.asyncio
async def test_reviewed_removal_seals_before_held_worker_and_keeps_uncertain_custody(
    revocation_case,
):
    case = revocation_case
    service = case.service
    review = await service.review_removal(case.installation, uninstall=True)
    case.keep_alive()
    await asyncio.to_thread(case.block_persistence)
    try:
        request = service.begin_reviewed_removal(review)
        assert not case.admission_allows_a() and not case.admission_allows_b()
        await asyncio.wait_for(case.cleanup_started.wait(), 2)
        assert case.child.poll() is None
        receipt = service.revocation_status(request)
        assert not receipt.committed and receipt.cleanup_pending
        assert service.begin_reviewed_removal(review) is request
    finally:
        case.release_persistence()
    receipt = await service.finish_revocation(request)
    assert (
        receipt.committed
        and receipt.cleanup_pending
        and receipt.runtime_stopped is False
    )


@pytest.mark.asyncio
async def test_reviewed_removal_expiry_rejects_before_seal(native_console, monkeypatch):
    import time

    service = native_console.service
    installed = await native_console.install()
    review = await service.review_removal(installed.installation_id, uninstall=False)
    with monkeypatch.context() as patch:
        patch.setattr(time, "monotonic", lambda: review.expires_at + 1)
        with pytest.raises(ValueError, match="expired"):
            service.begin_reviewed_removal(review)
    assert not service.fences.operations
    assert (await service.management_snapshot("workspace-a"))[0].effective_active


@pytest.mark.asyncio
async def test_reviewed_removal_failed_persistence_stays_session_only(revocation_case):
    from tldw_chatbook.Plugins.revocation import RevocationFailure

    case = revocation_case
    review = await case.service.review_removal(case.installation, uninstall=True)

    def fail(phase):
        if phase == "materialized":
            raise OSError("private failure")

    await case.service._call(
        lambda: setattr(case.service._coordinator, "progress", fail)
    )
    try:
        request = case.service.begin_reviewed_removal(review)
        with pytest.raises(RevocationFailure) as caught:
            await case.service.finish_revocation(request)
        assert not caught.value.receipt.committed
        assert not case.admission_allows_a() and not case.admission_allows_b()
        assert "private" not in repr(caught.value.receipt)
    finally:
        await case.service._call(
            lambda: setattr(case.service._coordinator, "progress", None)
        )


@pytest.mark.asyncio
async def test_pending_selection_status_exposes_real_blockers_and_cancel(
    revocation_case,
):
    from tldw_chatbook.UI.Plugins_Modules.operations import PluginOperations

    case = revocation_case
    owner = PluginOperations(case.service)
    review = await case.service.review_selection(case.installation, selection=())
    job = owner.submit(
        "select", review, installation_id=case.installation, workspace_id="a"
    )
    try:
        for _ in range(100):
            state = await case.service.operation_status(review.operation_id)
            if state["receipt"].phase == "waiting":
                break
            await asyncio.sleep(0.01)
        assert state["cancel_operation"] and state["blockers"]
        await owner.refresh()
        assert not job.task.done()
        assert job.runtime_status["cancel_operation"]
        assert job.runtime_status["blockers"] == state["blockers"]
        await owner.cancel(review.operation_id)
        await asyncio.wait_for(asyncio.shield(job.task), 5)
        assert not job.receipt.committed
        assert case.child.poll() is None and not case.cancelled_b.is_set()
    finally:
        case.service.cancel_revision(review)
        await asyncio.wait_for(asyncio.shield(job.task), 5)
