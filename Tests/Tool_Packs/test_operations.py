"""Admitted Tool Profile writes outlive their disposable UI observers."""

import asyncio
import threading
from dataclasses import replace

import pytest

from tldw_chatbook.Tool_Packs.activation import (
    InstalledToolProfile,
    ToolPackActivationResult,
)
from tldw_chatbook.Tool_Packs.contracts import ToolPackError
from tldw_chatbook.Tool_Packs.operations import (
    ToolProfileOperations,
    ToolProfileWriteUnavailable,
)
from tldw_chatbook.Tool_Packs.publication import ToolPackPublicationResult
from tldw_chatbook.Tool_Packs.removal import (
    RemovedToolProfile,
    ToolProfileRemovalResult,
)


def result_for(operation):
    return {
        "import": ToolPackActivationResult(
            InstalledToolProfile("research", "a" * 64, 1, "receipt"), "generation"
        ),
        "export": ToolPackPublicationResult("a" * 64, True, False),
        "remove": ToolProfileRemovalResult(
            RemovedToolProfile(
                "research", "tombstone", 2, "a" * 64, "receipt", "b" * 64
            ),
            "generation",
        ),
    }[operation]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["import", "export", "remove"])
async def test_cancelled_observer_cannot_recall_or_replace_admitted_write(operation):
    owner = ToolProfileOperations()
    entered, release = threading.Event(), threading.Event()

    def write(cancelled):
        entered.set()
        assert release.wait(5)
        return result_for(operation)

    task = owner.start(operation, "research", write)
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        pending = owner.pending(operation)
        assert pending.profile_id == "research" and pending.in_progress
        with pytest.raises(ToolProfileWriteUnavailable, match="busy"):
            owner.start(operation, "another", write)

        async def observe():
            return await asyncio.shield(task)

        observer = asyncio.create_task(observe())
        await asyncio.sleep(0)
        observer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observer
        assert not task.done()
        assert owner.pending(operation) is pending
    finally:
        release.set()
        outcome = await task
    assert outcome.result == result_for(operation)
    assert outcome is owner.state
    assert owner.pending(operation) is None
    assert owner.completion_revision == outcome.revision > pending.revision


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["import", "export", "remove"])
@pytest.mark.parametrize(
    "failure",
    ["service", "unexpected", "invalid_result", "malformed_result", "cancelled"],
)
async def test_outcomes_keep_only_compact_results_or_stable_categories(
    operation, failure
):
    owner = ToolProfileOperations()

    def write(cancelled):
        if failure == "service":
            raise ToolPackError(operation, "review_stale")
        if failure == "unexpected":
            raise OSError("private destination /secret/path")
        if failure == "cancelled":
            raise asyncio.CancelledError("private destination /secret/path")
        if failure == "malformed_result":
            return replace(
                result_for(operation),
                **{
                    "import": {"installed": None},
                    "export": {"archive_sha256": None},
                    "remove": {"tombstone": None},
                }[operation],
            )
        return {"path": "/secret/path"}

    outcome = await owner.start(operation, "research", write)
    assert not outcome.in_progress and outcome.result is None
    assert outcome.error_category
    assert "/secret/path" not in repr(outcome)
    assert owner.pending(operation) is None


@pytest.mark.asyncio
async def test_completion_revision_survives_newer_pending_state():
    owner = ToolProfileOperations()
    first = await owner.start(
        "import", "research", lambda cancelled: result_for("import")
    )
    release = threading.Event()

    def write(cancelled):
        assert release.wait(5)
        return result_for("export")

    second = owner.start("export", "research", write)
    try:
        assert owner.state.in_progress
        assert owner.state.revision > first.revision
        assert owner.completion_revision == first.revision
        assert first.result == result_for("import")
    finally:
        release.set()
        await second


@pytest.mark.asyncio
async def test_different_kinds_keep_their_exact_results_in_reverse_completion_order():
    owner = ToolProfileOperations()
    release = threading.Event()

    def import_profile(cancelled):
        assert release.wait(5)
        return result_for("import")

    first = owner.start("import", "research", import_profile)
    first_pending = owner.pending("import")
    try:
        second = await owner.start(
            "export", "research", lambda cancelled: result_for("export")
        )
        assert owner.state is second
        assert owner.pending("import") is first_pending
        assert owner.pending("export") is None
    finally:
        release.set()
        imported = await first
    assert imported.revision > second.revision
    assert imported.result == result_for("import")
    assert second.result == result_for("export")
    assert owner.state is imported
    fresh = await owner.start(
        "import", "research", lambda cancelled: result_for("import")
    )
    assert fresh.revision > imported.revision


@pytest.mark.asyncio
async def test_cancelled_shutdown_closes_admission_and_retains_thread_until_retry():
    owner = ToolProfileOperations()
    entered, release, saw_cancel = (threading.Event() for _ in range(3))

    def write(cancelled):
        entered.set()
        assert release.wait(5)
        if cancelled():
            saw_cancel.set()
        return result_for("export")

    task = owner.start("export", "research", write)
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        drain = asyncio.create_task(owner.close_and_drain())
        await asyncio.sleep(0)
        with pytest.raises(ToolProfileWriteUnavailable, match="shutdown"):
            owner.start("import", "another", write)
        assert not drain.done()
        drain.cancel()
        with pytest.raises(asyncio.CancelledError):
            await drain
        assert not task.done()
    finally:
        release.set()
        await owner.close_and_drain()
    assert saw_cancel.is_set()
    assert task.result().result.committed
    assert owner.pending("export") is None
