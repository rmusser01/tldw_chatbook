"""Finite discovery reads preserve cancellation and publication ownership."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "cancel", "source", "owner", "error"])
async def test_context_read_retains_native_work_until_exit(outcome):
    from tldw_chatbook.UI.Console_Modules.skill import _read_console_skill_context

    entered, release = threading.Event(), threading.Event()
    reads = set()
    main = threading.get_ident()
    state = {"source": True, "owner": True}
    observed = []

    async def get_context(*, mode):
        observed.append((mode, threading.get_ident()))
        entered.set()
        assert release.wait(5)
        if outcome == "error":
            raise OSError("checked reader failed")
        return {"available_skills": [{"name": "checked"}]}

    source = SimpleNamespace(runtime_reads=reads)
    task = asyncio.create_task(
        _read_console_skill_context(
            get_context, source, lambda: state["source"], lambda: state["owner"]
        )
    )
    try:
        async with asyncio.timeout(3):
            while not entered.is_set():
                await asyncio.sleep(0.005)
        assert observed == [("local", observed[0][1])]
        assert observed[0][1] != main
        assert len(reads) == 1
        if outcome == "cancel":
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done() and len(reads) == 1
        elif outcome in {"source", "owner"}:
            state[outcome] = False
    finally:
        release.set()
    if outcome == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await task
    elif outcome in {"source", "owner"}:
        with pytest.raises(RuntimeError, match="console_skill_context_changed"):
            await task
    elif outcome == "error":
        with pytest.raises(OSError, match="checked reader failed"):
            await task
    else:
        assert await task == {"available_skills": [{"name": "checked"}]}
    assert not reads
