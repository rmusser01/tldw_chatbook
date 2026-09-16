"""The app owns workflow loss and physical drain even off the Workflows screen."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from Tests.UI.test_app_quit_guard import _ConfirmationHarness, _ConfirmationScreen
from Tests.Workflows import test_session as session_tests
from Tests.Workflows.test_session import launch, until
from tldw_chatbook.app import TldwCli

harness = session_tests.harness


class WorkflowQuitHarness(_ConfirmationHarness):
    _shutdown_workflow_session = TldwCli._shutdown_workflow_session

    async def push_screen_wait(self, dialog):
        return self.decision

    def __init__(self, session):
        super().__init__(_ConfirmationScreen())
        self._workflow_session = session
        self.decision = False


async def test_quit_other_screen_stay_retains_review(harness):
    _, _, run_id = await launch(harness)
    await until(harness.session, lambda v: v.state == "review")
    app = WorkflowQuitHarness(harness.session)
    await app._confirm_and_quit()
    assert app.cleanup_calls == 0
    assert harness.session.update_review(run_id, "review", "still here")


async def test_quit_flush_cancellation_reopens_unaccepted_fence(harness):
    _, _, run_id = await launch(harness)
    await until(harness.session, lambda v: v.state == "review")
    app = WorkflowQuitHarness(harness.session)
    app.decision = True
    entered = asyncio.Event()

    async def flush():
        entered.set()
        await asyncio.Event().wait()

    app._workflow_authoring = SimpleNamespace(flush=flush)
    waiter = asyncio.create_task(app._confirm_and_quit())
    await asyncio.wait_for(entered.wait(), 5)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert app.cleanup_calls == 0
    assert harness.session.update_review(
        run_id, "review", "editable after aborted quit"
    )


async def test_failed_flush_preserves_review_and_normal_close_drains_first(harness):
    _, _, run_id = await launch(harness)
    await until(harness.session, lambda v: v.state == "review")
    app = WorkflowQuitHarness(harness.session)
    app.decision = True

    async def failed():
        raise OSError("draft write refused")

    app._workflow_authoring = SimpleNamespace(flush=failed)
    await app._confirm_and_quit()
    assert app.cleanup_calls == 0 and not app._shutting_down
    assert harness.session.update_review(run_id, "review", "still editable")
    order = []

    async def flush():
        order.append(harness.session.view().state)

    app._workflow_authoring = SimpleNamespace(flush=flush)
    await app._shutdown_workflow_session()
    assert order == ["review"]
    assert harness.session.view().state == "cancelled"
    await app._shutdown_workflow_session()


async def test_cancelled_accepted_close_retains_physical_writer_and_fence(
    harness, monkeypatch
):
    from tldw_chatbook.Workflows.session import SessionError

    h = harness
    entered, release, exited = threading.Event(), threading.Event(), threading.Event()
    save = h.scope.save_note

    async def held(**kwargs):
        result = await save(**kwargs)
        entered.set()
        try:
            assert release.wait(10)
            return result
        finally:
            exited.set()

    monkeypatch.setattr(h.scope, "save_note", held)
    revision, _, run_id = await launch(h)
    await until(h.session, lambda v: v.state == "review")
    h.session.answer_review(run_id, "review", accept=True)
    assert await asyncio.to_thread(entered.wait, 5)
    app = WorkflowQuitHarness(h.session)
    app.decision = True
    waiter = asyncio.create_task(app._confirm_and_quit())
    try:
        await until(h.session, lambda v: v.state == "stopping")
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert not exited.is_set() and app.cleanup_calls == 0
        with pytest.raises(SessionError):
            await h.session.prepare(revision, {}, h.setup)
    finally:
        release.set()
    await app._shutdown_workflow_session()
    assert exited.is_set()
    assert h.session.view().note_id == h.rows()[0]["id"]
    assert h.session.view().message_code == "saved_after_cancel"
    await app._confirm_and_quit()
    assert app.cleanup_calls == 1


async def test_failed_physical_drain_never_enters_exit_cleanup(harness, monkeypatch):
    from tldw_chatbook.Workflows.local_steps import LocalNoteCleanupError

    async def failed(**kwargs):
        raise LocalNoteCleanupError()

    monkeypatch.setattr(harness.scope, "save_note", failed)
    _, _, run_id = await launch(harness)
    await until(harness.session, lambda v: v.state == "review")
    harness.session.answer_review(run_id, "review", accept=True)
    await until(harness.session, lambda v: v.state == "failed")
    app = WorkflowQuitHarness(harness.session)
    app.decision = True
    await app._confirm_and_quit()
    assert app.cleanup_calls == 0 and not app._shutting_down
    assert any("physical drain failed" in text for text, _ in app.notifications)


async def test_confirmation_changes_do_not_cancel_newer_review(harness):
    _, _, run_id = await launch(harness)
    await until(harness.session, lambda v: v.state == "review")
    app = WorkflowQuitHarness(harness.session)
    calls = []

    async def respond(dialog):
        calls.append(dialog)
        if len(calls) == 1:
            harness.session.update_review(
                run_id, "review", "Changed during confirmation"
            )
            return True
        return False

    app.push_screen_wait = respond
    await app._confirm_and_quit()
    assert len(calls) == 2
    assert app.cleanup_calls == 0
    assert harness.session.update_review(run_id, "review", "Still editable")


async def test_production_composition_is_lazy_single_owner(harness):
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    assert app._workflow_session is None
    app.notes_scope_service = harness.scope
    app.notes_user_id = "reader"
    owner = app.ensure_workflow_session()
    assert app.ensure_workflow_session() is owner
    assert owner.view() is None
    assert app._workflow_authoring is None
    await app._shutdown_workflow_session()
