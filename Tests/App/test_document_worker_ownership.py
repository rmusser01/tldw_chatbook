"""First editor use owns one lazy lane; app shutdown cannot reopen it."""

import asyncio

import pytest

from Tests.UI.app_factory import _build_test_app

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]


@pytest.mark.parametrize("phase", ["drain", "exit", "unmount"])
async def test_unused_document_lane_is_not_constructed_by_shutdown(phase):
    app = _build_test_app(configured_default="chat")
    if phase == "exit":
        app._exit = True
    elif phase == "unmount":
        app._shutting_down = True
    else:
        await app._shutdown_app_owned_lifecycles()
    try:
        assert app.document_worker is None
    finally:
        await app._shutdown_app_owned_lifecycles()


async def test_first_document_use_reuses_one_lane_and_app_close_reaps_its_child(
    tmp_path, monkeypatch
):
    app = _build_test_app(configured_default="chat")
    worker = app.document_worker
    assert worker is not None
    from tldw_chatbook.Document_Editing import worker as module
    from tldw_chatbook.Document_Editing.models import DocumentRequest, RequestKey

    request = DocumentRequest(RequestKey("owner", "note", 1, 1, 1, 0), "{}", "json")
    assert (await worker.check(request)).format_allowed
    assert app.document_worker is worker
    stall = tmp_path / "document-stall.py"
    stall.write_text("import time\ntime.sleep(60)\n")
    monkeypatch.setattr(module, "WORKER_PATH", stall)
    pending = asyncio.create_task(worker.check(request))
    async with asyncio.timeout(2):
        while worker._process is None:
            await asyncio.sleep(0.01)
    process = worker._process
    try:
        await app._shutdown_app_owned_lifecycles()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert process.returncode is not None
        assert worker._process is None
        assert app.document_worker is worker
        assert (await worker.check(request)).state == "failed"
        assert worker._process is None
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
        await worker.close()
