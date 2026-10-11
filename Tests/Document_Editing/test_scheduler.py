"""Bounded active-editor leases and permanent result retirement."""

import asyncio

import pytest

from tldw_chatbook.Document_Editing.models import CheckResult, FormatResult


class ControlledWorker:
    def __init__(self):
        self.requests = []
        self.first_started = asyncio.Event()
        self.release = asyncio.Event()
        self.second_done = asyncio.Event()

    async def check(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            self.first_started.set()
            await self.release.wait()
        else:
            self.second_done.set()
        return CheckResult.for_request(request, "checked", True, (), False, True)

    async def format(self, request):
        check = await self.check(request)
        return FormatResult(request.key, "unchanged", None, check)


@pytest.mark.asyncio
async def test_only_latest_pending_revision_survives(request_for):
    from tldw_chatbook.Document_Editing.scheduler import DocumentCheckScheduler

    worker = ControlledWorker()
    results = []
    scheduler = DocumentCheckScheduler(worker, results.append)
    scheduler.submit(request_for("{}", draft_revision=1))
    await asyncio.wait_for(worker.first_started.wait(), 1)
    for revision in range(2, 50):
        scheduler.submit(request_for("{}", draft_revision=revision))
    worker.release.set()
    await asyncio.wait_for(worker.second_done.wait(), 1)
    await asyncio.sleep(0)
    assert [r.key.draft_revision for r in worker.requests] == [1, 49]
    assert [r.key.draft_revision for r in results] == [49]
    await scheduler.close()


@pytest.mark.asyncio
async def test_format_shares_lane_and_superseded_format_is_not_replayed(request_for):
    from tldw_chatbook.Document_Editing.scheduler import DocumentCheckScheduler

    worker = ControlledWorker()
    scheduler = DocumentCheckScheduler(worker, lambda result: None)
    scheduler.submit(request_for("{}"), immediate=True)
    await asyncio.wait_for(worker.first_started.wait(), 1)
    formatting = asyncio.create_task(scheduler.format(request_for("{}")))
    await asyncio.sleep(0)
    latest = request_for("{}", draft_revision=8)
    scheduler.submit(latest, immediate=True)
    assert await formatting is None
    worker.release.set()
    await asyncio.wait_for(worker.second_done.wait(), 1)
    assert worker.requests[-1] == latest and len(worker.requests) == 2
    await scheduler.close()


@pytest.mark.asyncio
async def test_retirement_is_permanent_even_for_identical_source_key(request_for):
    from tldw_chatbook.Document_Editing.scheduler import DocumentCheckScheduler

    worker = ControlledWorker()
    results = []
    scheduler = DocumentCheckScheduler(worker, results.append)
    request = request_for("{}")
    formatting = asyncio.create_task(scheduler.format(request))
    await asyncio.wait_for(worker.first_started.wait(), 1)
    scheduler.retire()
    scheduler.submit(request, immediate=True)
    worker.release.set()
    assert await formatting is None
    await scheduler.close()
    assert results == [] and len(worker.requests) == 1
    assert scheduler._pending is None


@pytest.mark.asyncio
async def test_new_active_lease_reaps_old_before_new_work(request_for):
    from tldw_chatbook.Document_Editing.worker import DocumentWorker

    worker = DocumentWorker()
    first_results, second_results = [], []
    first = await worker.acquire(first_results.append)
    first.submit(request_for("{}"))
    second = await worker.acquire(second_results.append)
    first.submit(request_for("{}"), immediate=True)
    result = await second.format(request_for("{}"))
    assert result.state == "unchanged" and first_results == []
    assert first._pending is None
    await worker.close()


@pytest.mark.asyncio
async def test_overlapping_navigation_cannot_revive_older_lease(monkeypatch):
    from tldw_chatbook.Document_Editing.worker import DocumentWorker

    owner = DocumentWorker()
    first = await owner.acquire(lambda result: None)
    closing = asyncio.Event()
    release = asyncio.Event()
    original_close = first.close

    async def held_close():
        closing.set()
        await release.wait()
        await original_close()

    monkeypatch.setattr(first, "close", held_close)
    second_call = asyncio.create_task(owner.acquire(lambda result: None))
    await asyncio.wait_for(closing.wait(), 1)
    third_call = asyncio.create_task(owner.acquire(lambda result: None))
    await asyncio.sleep(0)
    release.set()
    second, third = await asyncio.gather(second_call, third_call)
    assert second._retired and not third._retired
    await owner.close()
