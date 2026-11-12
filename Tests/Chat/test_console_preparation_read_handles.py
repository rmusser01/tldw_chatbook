"""Native preparation exposes its owned read, including cancelled result delivery."""

import asyncio
import functools
import threading

import pytest

from tldw_chatbook.Chat import console_preparation_reads as native


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_worker_exposes_the_read_that_retires_before_result_or_cancellation(
    cancel,
):
    creator, source = object(), object()
    reads, observed = set(), []
    entered, release = threading.Event(), threading.Event()

    def callback():
        read = getattr(native, "preparation_read_for", lambda creator: None)(creator)
        observed.append(read)
        entered.set()
        assert release.wait(5)
        return "native result"

    task = asyncio.create_task(
        native.run_preparation_read(
            callback,
            creator=creator,
            session_id="session",
            reads=reads,
            require_current=lambda: None,
            source=source,
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        assert observed[0] is not None
        read = observed[0]
        assert read in reads and read.source is source
        assert not read.retired.done()
        assert native.preparation_read_for(creator) is None
        if cancel:
            task.cancel()
            await asyncio.sleep(0)
            assert not read.retired.done()
        release.set()
        if cancel:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 3)
        else:
            assert await asyncio.wait_for(task, 3) == "native result"
        assert read.retired.done() and not read.retired.cancelled()
        assert read.callback_returned and read.result == "native result"
        assert read._producer.done() and read._producer.result() == "native result"
        assert reads == set()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped", [False, True])
async def test_workflow_observes_actual_native_read_retirement(monkeypatch, wrapped):
    from tldw_chatbook.Chat.console_workflow_runs import (
        _capture_workflow_read,
        _prove_workflow_read,
    )

    creator, source = object(), object()
    reads, captured = set(), []
    issuing = asyncio.current_task()
    if wrapped:
        original = native.run_preparation_read

        @functools.wraps(original)
        async def run(*args, **kwargs):
            return await original(*args, **kwargs)

        monkeypatch.setattr(native, "run_preparation_read", run)

    def callback():
        proof = _capture_workflow_read(creator, source, issuing)
        captured.append(proof)
        with pytest.raises(RuntimeError):
            _prove_workflow_read(proof, creator, source, issuing)
        with pytest.raises(PermissionError):
            _capture_workflow_read(creator, object(), issuing)
        return "retired result"

    assert (
        await native.run_preparation_read(
            callback,
            creator=creator,
            session_id="session",
            reads=reads,
            require_current=lambda: None,
            source=source,
        )
        == "retired result"
    )
    _prove_workflow_read(captured[0], creator, source, issuing)
    with pytest.raises(RuntimeError):
        _prove_workflow_read(captured[0], creator, object(), issuing)


@pytest.mark.asyncio
async def test_post_callback_revocation_keeps_the_actual_retired_result():
    from tldw_chatbook.Chat.console_workflow_runs import (
        _capture_workflow_read,
        _prove_workflow_read,
    )

    creator, source, result = object(), object(), object()
    reads, captured = set(), []
    issuing = asyncio.current_task()
    revoked = threading.Event()

    def current():
        if revoked.is_set():
            raise PermissionError("selection revoked")

    def callback():
        captured.append(_capture_workflow_read(creator, source, issuing))
        revoked.set()
        return result

    with pytest.raises(PermissionError, match="selection revoked"):
        await native.run_preparation_read(
            callback,
            creator=creator,
            session_id="s",
            reads=reads,
            require_current=current,
            source=source,
        )
    _prove_workflow_read(captured[0], creator, source, issuing)
    assert captured[0][0].result is result
    assert not reads
