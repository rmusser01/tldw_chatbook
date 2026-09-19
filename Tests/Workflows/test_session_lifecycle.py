"""Barrier-based physical settlement and quit fencing qualification."""

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from Tests.Workflows import test_session as session_tests
from Tests.Workflows.helpers import prompt_definition
from Tests.Workflows.test_session import launch, until
from Tests.Workflows.test_session_admission import file_definition, revision_of

harness = session_tests.harness


async def entered(event):
    assert await asyncio.to_thread(event.wait, 10), "worker never reached barrier"


async def test_quit_fence_preserves_completed_file_and_abort_continues(
    harness, monkeypatch
):
    from tldw_chatbook.Workflows import session as module

    h = harness
    inside, release, exited = threading.Event(), threading.Event(), threading.Event()
    read = module.read_local_text

    def gated(*args, **kwargs):
        result = read(*args, **kwargs)
        inside.set()
        try:
            assert release.wait(10)
            return result
        finally:
            exited.set()

    monkeypatch.setattr(module, "read_local_text", gated)
    _, _, run_id = await launch(h)
    try:
        await entered(inside)
        h.session.begin_close()
        generation = h.session.view().generation
        release.set()
        await entered(exited)
        await until(h.session, lambda v: v.message_code == "close_fenced")
        assert h.requests == []
        assert h.session.view().state != "cancelled"
        assert h.session.view().generation >= generation
        h.session.abort_close()
        await until(h.session, lambda v: v.state == "review")
        assert len(h.requests) == 1
        h.session.answer_review(run_id, "review", accept=False)
        await until(h.session, lambda v: v.state == "rejected")
    finally:
        release.set()


@pytest.mark.parametrize(
    "after_commit,lost_receipt", [(False, False), (True, False), (True, True)]
)
async def test_cancelled_writer_retains_slot_and_close_waiter(
    harness, monkeypatch, after_commit, lost_receipt
):
    from tldw_chatbook.Workflows.session import SessionError

    h = harness
    inside, release, exited = threading.Event(), threading.Event(), threading.Event()
    save = h.scope.save_note

    async def gated(**kwargs):
        try:
            if after_commit:
                result = await save(**kwargs)
            inside.set()
            assert release.wait(10)
            if not after_commit:
                result = await save(**kwargs)
            if lost_receipt:
                raise RuntimeError("private-lost-receipt")
            return result
        finally:
            exited.set()

    monkeypatch.setattr(h.scope, "save_note", gated)
    revision = h.documents.create(json.dumps(file_definition()))
    old = await h.session.prepare(revision, {}, h.setup)
    ticket = await h.session.prepare(revision, {}, h.setup)
    run_id = h.session.start(ticket)
    try:
        await until(h.session, lambda v: v.state == "review")
        h.session.answer_review(run_id, "review", accept=True)
        await entered(inside)
        h.session.cancel(run_id)
        assert h.session.view().state == "stopping"
        assert h.session.view().run_id == run_id
        with pytest.raises(SessionError):
            h.session.start(old)
        close_waiter = asyncio.create_task(h.session.close())
        await asyncio.sleep(0)
        close_waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await close_waiter
        assert not exited.is_set()
        assert h.session.view().state == "stopping"
        release.set()
        await h.session.close()
        assert exited.is_set()
        view = h.session.view()
        assert view.state == "cancelled"
        assert view.message_code == "saved_after_cancel"
        assert h.rows()[0]["id"] == view.note_id
        h.session.abort_close()
        with pytest.raises(SessionError):
            await h.session.prepare(revision, {}, h.setup)
    finally:
        release.set()


@pytest.mark.parametrize(
    "outcome", ["confirmed", "wrong_content", "revoked", "owner_changed", "cleanup"]
)
async def test_postcommit_readback_is_exact_and_cleanup_failure_stays_observable(
    harness, monkeypatch, outcome
):
    from tldw_chatbook.Workflows.local_steps import LocalNoteCleanupError
    from tldw_chatbook.Workflows.session import SessionError

    h = harness
    save = h.scope.save_note

    async def lost_receipt(**kwargs):
        if outcome == "wrong_content":
            kwargs["content"] = "not accepted"
        await save(**kwargs)
        if outcome == "revoked":
            h.set_permission("create_note", "deny")
        if outcome == "owner_changed":
            h.scope.local_notes_service = object()
        if outcome == "cleanup":
            raise LocalNoteCleanupError()
        raise RuntimeError("private-canary-after-commit")

    monkeypatch.setattr(h.scope, "save_note", lost_receipt)
    revision, _, run_id = await launch(h)
    await until(h.session, lambda v: v.state == "review")
    h.session.answer_review(run_id, "review", accept=True)
    view = await until(
        h.session, lambda v: v.state in {"completed", "uncertain", "failed"}
    )
    assert len(h.rows()) == 1
    assert "private-canary" not in str(view)
    if outcome == "confirmed":
        assert view.state == "completed" and view.note_id == h.rows()[0]["id"]
    elif outcome == "cleanup":
        assert view.state == "failed" and view.message_code == "note_cleanup_failed"
        assert view.note_id == h.rows()[0]["id"]
        for _ in range(2):
            with pytest.raises(SessionError, match="note_cleanup_failed"):
                await h.session.close()
        with pytest.raises(SessionError):
            await h.session.prepare(revision, {}, h.setup)
    else:
        assert view.state == "uncertain" and view.note_id is None


async def test_cancelled_setup_waiter_is_retained_and_never_launches(
    harness, monkeypatch
):
    from tldw_chatbook.Workflows import session as module

    h = harness
    inside, release, exited = threading.Event(), threading.Event(), threading.Event()
    resolve = module.resolve_llama_loopback_url

    def gated(url):
        inside.set()
        try:
            assert release.wait(10)
            return resolve(url)
        finally:
            exited.set()

    monkeypatch.setattr(module, "resolve_llama_loopback_url", gated)
    pending = asyncio.create_task(
        h.session.prepare(revision_of(prompt_definition()), {}, h.setup)
    )
    try:
        await entered(inside)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        close = asyncio.create_task(h.session.close())
        await asyncio.sleep(0)
        assert not close.done() and not exited.is_set()
        release.set()
        await close
        assert exited.is_set() and h.requests == [] and h.rows() == []
    finally:
        release.set()


async def test_cancel_before_worker_entry_does_not_read(harness, monkeypatch):
    from tldw_chatbook.Workflows import session as module

    h = harness
    revision = h.documents.create(json.dumps(file_definition()))
    ticket = await h.session.prepare(revision, {}, h.setup)
    loop = asyncio.get_running_loop()
    original = loop.run_in_executor
    gate = threading.Event()
    reads = []
    read = module.read_local_text

    def observed(*args, **kwargs):
        reads.append(True)
        return read(*args, **kwargs)

    monkeypatch.setattr(module, "read_local_text", observed)
    with ThreadPoolExecutor(max_workers=1) as pool:
        occupied = pool.submit(gate.wait, 10)
        monkeypatch.setattr(
            loop,
            "run_in_executor",
            lambda executor, function, *args: original(
                pool if executor is None else executor, function, *args
            ),
        )
        try:
            run_id = h.session.start(ticket)
            await asyncio.sleep(0)
            h.session.cancel(run_id)
            close = asyncio.create_task(h.session.close())
            await asyncio.sleep(0)
            assert not close.done()
            gate.set()
            await close
            assert occupied.result() and reads == []
            assert h.session.view().state == "cancelled"
        finally:
            gate.set()


async def test_model_cleanup_cancelled_once_retained_and_failure_blocks_close(
    harness, monkeypatch
):
    from tldw_chatbook.Workflows.session import SessionError

    h = harness
    inside, release = asyncio.Event(), asyncio.Event()
    cancellations = []

    async def transport(request):
        inside.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancellations.append(True)
            await release.wait()
            raise RuntimeError("private-cleanup-failed")

    monkeypatch.setattr(
        httpx, "AsyncHTTPTransport", lambda **kwargs: httpx.MockTransport(transport)
    )
    _, _, run_id = await launch(h)
    try:
        await asyncio.wait_for(inside.wait(), 10)
        h.session.cancel(run_id)
        h.session.cancel(run_id)
        waiter = asyncio.create_task(h.session.close())
        await asyncio.sleep(0)
        assert not waiter.done()
        assert h.session.view().state == "stopping"
        release.set()
        with pytest.raises(SessionError):
            await waiter
        assert cancellations == [True]
        assert h.session.view().state == "failed"
        assert h.rows() == []
    finally:
        release.set()


async def test_independent_sessions_do_not_share_slot_or_resume_review(harness):
    h = harness
    _, _, run_id = await launch(h)
    await until(h.session, lambda v: v.state == "review")
    second = h.new_session()
    assert second.view() is None
    ticket = await second.prepare(revision_of(prompt_definition()), {}, h.setup)
    second_id = second.start(ticket)
    await until(second, lambda v: v.state == "completed")
    assert second_id != run_id
    assert h.session.view().state == "review"


async def test_attempt_timeout_drains_live_source_and_retains_reason(
    harness, monkeypatch
):
    from tldw_chatbook.Workflows import session as module

    h = harness
    inside, release, exited = threading.Event(), threading.Event(), threading.Event()
    read = module.read_local_text

    def gated(*args, **kwargs):
        text = read(*args, **kwargs)
        inside.set()
        try:
            assert release.wait(10)
            return text
        finally:
            exited.set()

    monkeypatch.setattr(module, "read_local_text", gated)
    document = file_definition()
    document["steps"][0]["timeout_seconds"] = 1
    _, _, run_id = await launch(h, document)
    try:
        await entered(inside)
        await until(h.session, lambda v: v.state == "stopping")
        assert not exited.is_set()
        assert h.session.view().message_code == "attempt_timeout"
        release.set()
        view = await until(h.session, lambda v: v.state == "failed")
        assert view.message_code == "attempt_timeout"
        assert exited.is_set() and h.requests == []
        assert not h.session.answer_review(run_id, "review", accept=True)
    finally:
        release.set()


async def test_fence_completed_note_result_resumes_without_duplicate_write(
    harness, monkeypatch
):
    h = harness
    inside, release, exited = threading.Event(), threading.Event(), threading.Event()
    save = h.scope.save_note

    async def gated(**kwargs):
        result = await save(**kwargs)
        inside.set()
        try:
            assert release.wait(10)
            return result
        finally:
            exited.set()

    monkeypatch.setattr(h.scope, "save_note", gated)
    _, _, run_id = await launch(h)
    try:
        await until(h.session, lambda v: v.state == "review")
        h.session.answer_review(run_id, "review", accept=True)
        await entered(inside)
        h.session.begin_close()
        release.set()
        await entered(exited)
        await until(h.session, lambda v: v.message_code == "close_fenced")
        assert len(h.rows()) == 1
        assert h.session.view().note_id == h.rows()[0]["id"]
        h.session.abort_close()
        await until(h.session, lambda v: v.state == "completed")
        assert len(h.rows()) == 1
    finally:
        release.set()


async def test_setup_discard_and_supersession_keep_capture_owned(harness, monkeypatch):
    from tldw_chatbook.Workflows import session as module

    h = harness
    inside, release = threading.Event(), threading.Event()
    resolve = module.resolve_llama_loopback_url
    calls = []

    def gated(url):
        calls.append(True)
        if len(calls) == 1:
            inside.set()
            assert release.wait(10)
        return resolve(url)

    monkeypatch.setattr(module, "resolve_llama_loopback_url", gated)
    revision = revision_of(prompt_definition())
    first = asyncio.create_task(h.session.prepare(revision, {}, h.setup))
    try:
        await entered(inside)
        h.session.discard_setup()
        second = await h.session.prepare(revision, {}, h.setup)
        release.set()
        with pytest.raises(module.SessionError, match="setup_stale"):
            await first
        h.session.start(second)
        await until(h.session, lambda v: v.state == "completed")
    finally:
        release.set()


async def test_real_notes_close_failure_remains_failed_after_confirmed_readback(
    harness, monkeypatch
):
    from tldw_chatbook.Workflows.session import SessionError

    h = harness
    revision = h.documents.create(json.dumps(file_definition()))
    ticket = await h.session.prepare(revision, {}, h.setup)
    db = h.session.bindings(ticket).notes.db
    close = db.close_connection
    fail = [True]

    def close_once():
        close()
        if fail[0]:
            fail[0] = False
            raise RuntimeError("private-cleanup-error")

    monkeypatch.setattr(db, "close_connection", close_once)
    run_id = h.session.start(ticket)
    await until(h.session, lambda v: v.state == "review")
    h.session.answer_review(run_id, "review", accept=True)
    view = await until(h.session, lambda v: v.state == "failed")
    assert view.message_code == "note_cleanup_failed"
    assert view.note_id == h.rows()[0]["id"]
    with pytest.raises(SessionError, match="note_cleanup_failed"):
        await h.session.close()


async def test_current_app_binding_getters_stay_on_app_loop(harness):
    from tldw_chatbook.Workflows.session import WorkflowSession

    h = harness

    def scope():
        assert threading.current_thread() is threading.main_thread()
        return h.scope

    def user():
        assert threading.current_thread() is threading.main_thread()
        return "reader"

    session = WorkflowSession(h.permissions, notes_scope=scope, notes_user=user)
    try:
        ticket = await session.prepare(revision_of(file_definition()), {}, h.setup)
        run_id = session.start(ticket)
        await until(session, lambda v: v.state == "review")
        session.answer_review(run_id, "review", accept=True)
        await until(session, lambda v: v.state == "completed")
        assert len(h.rows()) == 1
    finally:
        await session.close()


async def test_setup_cleanup_failure_is_observable_after_waiter_cancel(
    harness, monkeypatch
):
    from tldw_chatbook.Workflows import session as module

    h = harness
    inside, release = threading.Event(), threading.Event()
    capture = module.capture_local_note_destination

    def failed_cleanup(*args, **kwargs):
        capture(*args, **kwargs)
        inside.set()
        assert release.wait(10)
        raise module.LocalNoteCleanupError()

    monkeypatch.setattr(module, "capture_local_note_destination", failed_cleanup)
    pending = asyncio.create_task(
        h.session.prepare(revision_of(prompt_definition()), {}, h.setup)
    )
    try:
        await entered(inside)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        release.set()
        with pytest.raises(module.SessionError, match="note_cleanup_failed"):
            await h.session.close()
        h.session.abort_close()
        with pytest.raises(module.SessionError):
            await h.session.prepare(revision_of(prompt_definition()), {}, h.setup)
    finally:
        release.set()


async def test_model_cleanup_failure_without_cancellation_blocks_next_run(
    harness, monkeypatch
):
    from tldw_chatbook.Workflows.session import SessionError

    h = harness

    class FailedClose(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"choices":[{"message":{"content":"answer"},"finish_reason":"stop"}]}'

        async def aclose(self):
            raise RuntimeError("private-model-close-error")

    async def transport(request):
        return httpx.Response(200, stream=FailedClose())

    monkeypatch.setattr(
        httpx, "AsyncHTTPTransport", lambda **kwargs: httpx.MockTransport(transport)
    )
    revision, _, _ = await launch(h)
    view = await until(h.session, lambda v: v.state == "failed")
    assert view.message_code == "model_cleanup_failed"
    with pytest.raises(SessionError, match="model_cleanup_failed"):
        await h.session.close()
    with pytest.raises(SessionError):
        await h.session.prepare(revision, {}, h.setup)
    assert h.rows() == []


async def test_active_budget_counts_permission_worker_before_effect(
    harness, monkeypatch
):
    from tldw_chatbook.Workflows import session as module

    h = harness
    inside, release = threading.Event(), threading.Event()
    check = h.permissions.check

    def gated(effect, **kwargs):
        decision = check(effect, **kwargs)
        inside.set()
        assert release.wait(10)
        return decision

    monkeypatch.setattr(h.permissions, "check", gated)
    await launch(h)
    try:
        await entered(inside)
        monkeypatch.setattr(module, "_ACTIVE_SECONDS", 0)
        release.set()
        view = await until(h.session, lambda v: v.state == "failed")
        assert view.message_code == "active_budget"
        assert h.requests == [] and h.rows() == []
    finally:
        release.set()
