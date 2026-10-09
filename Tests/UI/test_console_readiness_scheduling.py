"""A refused presentation producer retires its coroutine and stays retryable."""

import asyncio

import pytest

from Tests.UI.test_console_readiness_config_projection import _read_result, _screen
from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
    ConsoleReadinessConfigProjection,
)

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.asyncio
@pytest.mark.parametrize("expired", [False, True])
async def test_refused_readiness_refresh_retires_then_retries_same_owner(
    monkeypatch, expired
):
    screen, _owner, _identity, tasks, _published = _screen(monkeypatch)
    value, rendered, refused = {"current": 1}, [], []
    projection = ConsoleReadinessConfigProjection(
        screen, read_current=lambda: _read_result(dict(value))
    )
    schedule = screen.run_worker
    if expired:
        assert projection.run(lambda: None) is False
        await asyncio.gather(*tasks)
        projection.at -= projection.max_age + 1
    value["current"] = 2

    def refuse(operation, **_):
        refused.append(operation)
        raise RuntimeError("worker manager is closed")

    screen.run_worker = refuse
    failure = None
    try:
        try:
            result = projection.run(lambda: rendered.append(projection.value))
        except Exception as error:
            failure, result = error, None
        retired = all(operation.cr_frame is None for operation in refused)
    finally:
        for operation in refused:
            operation.close()  # Failed controls still retire their own test coroutine.
    assert failure is None
    assert result is expired
    assert len(refused) == 1 and retired
    assert not projection.pending and projection._settled.is_set()
    assert rendered == ([{"current": 1}] if expired else [])
    screen.run_worker = schedule
    projection.run(lambda: None)
    await asyncio.gather(*tasks)
    assert projection.run(lambda: rendered.append(projection.value))
    assert rendered[-1] == {"current": 2}


@pytest.mark.asyncio
async def test_refused_readiness_publication_retires_after_finite_read(monkeypatch):
    screen, _owner, _identity, tasks, published = _screen(monkeypatch)
    schedule, refused = screen.run_worker, []

    def schedule_read_only(operation, **kwargs):
        if kwargs["group"] == "console-readiness-publication":
            refused.append(operation)
            raise RuntimeError("worker manager is closed")
        return schedule(operation, **kwargs)

    screen.run_worker = schedule_read_only
    projection = ConsoleReadinessConfigProjection(
        screen, read_current=lambda: _read_result({"current": 1})
    )
    try:
        assert projection.run(lambda: None) is False
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        retired = all(operation.cr_frame is None for operation in refused)
    finally:
        for operation in refused:
            operation.close()
    assert outcomes == [None]
    assert len(refused) == 1 and retired and not published
    assert not projection.pending and projection._settled.is_set()
    assert projection.value == {"current": 1}
    assert projection.run(lambda: None)


@pytest.mark.asyncio
async def test_refresh_scheduling_cancellation_retires_and_propagates(monkeypatch):
    screen, _owner, _identity, _tasks, _published = _screen(monkeypatch)
    refused = []

    def cancel(operation, **_):
        refused.append(operation)
        raise asyncio.CancelledError

    screen.run_worker = cancel
    projection = ConsoleReadinessConfigProjection(
        screen, read_current=lambda: pytest.fail("cancelled producer entered")
    )
    try:
        with pytest.raises(asyncio.CancelledError):
            projection.run(lambda: pytest.fail("cold cancelled presentation rendered"))
        retired = all(operation.cr_frame is None for operation in refused)
    finally:
        for operation in refused:
            operation.close()
    assert len(refused) == 1 and retired
    assert not projection.pending and projection._settled.is_set()
