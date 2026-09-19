"""The app owns workflow loss and physical drain even off the Workflows screen."""

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from Tests.UI.test_app_quit_guard import _ConfirmationHarness, _ConfirmationScreen
from Tests.Workflows import test_session as session_tests
from Tests.Workflows.test_session import launch, until
from tldw_chatbook.app import TldwCli

harness = session_tests.harness


@pytest.fixture
async def console_runtime(tmp_path):
    from tldw_chatbook.Agents.run_hooks import RunHooksConfig, RunHooksEngine
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

    runtime = ConsoleRuntime(SimpleNamespace())
    runtime._run_hooks_engine = RunHooksEngine(RunHooksConfig, lambda: str(tmp_path))
    _ = runtime.voice_promotion_owner  # Exercise an existing reversible quit fence.
    try:
        yield runtime
    finally:
        await runtime.dispose()


def assert_console_reopened(runtime):
    assert runtime.accepts_raw_cli_refusal_callbacks
    assert not runtime.voice_promotion_owner.status.quit_fenced
    hooks = runtime.run_hooks_engine
    assert runtime.ensure_run_hooks() is hooks
    assert not hooks.fire("PreToolUse", session_id="retained-console").blocked


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

    app._workflow_authoring = SimpleNamespace(
        prepare_quit=flush, abort_quit=lambda: None
    )
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

    app._workflow_authoring = SimpleNamespace(
        prepare_quit=failed, abort_quit=lambda: None
    )
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
    harness, monkeypatch, console_runtime
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
    app.console_runtime = console_runtime
    app.decision = True
    waiter = asyncio.create_task(app._confirm_and_quit())
    try:
        await until(h.session, lambda v: v.state == "stopping")
        assert console_runtime.accepts_raw_cli_refusal_callbacks
        assert console_runtime.voice_promotion_owner.status.quit_fenced
        assert not h.session.reopen_after_drained_quit()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert not exited.is_set() and app.cleanup_calls == 0
        assert_console_reopened(console_runtime)
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
    assert not console_runtime.accepts_raw_cli_refusal_callbacks
    assert console_runtime.voice_promotion_owner.status.quit_fenced
    assert console_runtime.run_hooks_engine.fire(
        "PreToolUse", session_id="retained-console"
    ).blocked


async def test_failed_physical_drain_never_enters_exit_cleanup(
    harness, monkeypatch, console_runtime
):
    from tldw_chatbook.Workflows.local_steps import LocalNoteCleanupError

    async def failed(**kwargs):
        raise LocalNoteCleanupError()

    monkeypatch.setattr(harness.scope, "save_note", failed)
    _, _, run_id = await launch(harness)
    await until(harness.session, lambda v: v.state == "review")
    harness.session.answer_review(run_id, "review", accept=True)
    await until(harness.session, lambda v: v.state == "failed")
    app = WorkflowQuitHarness(harness.session)
    app.console_runtime = console_runtime
    app.decision = True
    await app._confirm_and_quit()
    assert app.cleanup_calls == 0 and not app._shutting_down
    assert any("physical drain failed" in text for text, _ in app.notifications)
    assert_console_reopened(console_runtime)
    assert not harness.session.reopen_after_drained_quit()


async def test_failed_authoring_prepare_keeps_console_and_review_usable(
    tmp_path, harness, console_runtime, monkeypatch
):
    from tldw_chatbook.Workflows.authoring import WorkflowAuthoring

    authoring = WorkflowAuthoring(lambda: tmp_path / "failed-prepare.sqlite3")
    await authoring.create("Retained")
    write = authoring.documents.put_draft

    def failed(*args, **kwargs):
        raise OSError("draft persistence refused")

    monkeypatch.setattr(authoring.documents, "put_draft", failed)
    pending = authoring.drafts.update('{"unfinished":')
    _, _, run_id = await launch(harness)
    await until(harness.session, lambda v: v.state == "review")
    app = WorkflowQuitHarness(harness.session)
    app.console_runtime = console_runtime
    app.decision = True

    app._workflow_authoring = authoring
    try:
        await app._confirm_and_quit()
        assert app.cleanup_calls == 0 and not app._shutting_down
        assert any("safe shutdown" in text for text, _ in app.notifications)
        assert_console_reopened(console_runtime)
        assert authoring.drafts.current == pending
        assert harness.session.update_review(run_id, "review", "Still editable")
        authoring.drafts.update('{"unfinished": ')
    finally:
        monkeypatch.setattr(authoring.documents, "put_draft", write)
        await authoring.close()


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


@pytest.mark.parametrize("cancel_waiter", [False, True])
@pytest.mark.parametrize("waiting_state", ["review", "approval"])
async def test_reconfirmation_abort_restores_same_workflow_owners(
    tmp_path, harness, console_runtime, cancel_waiter, waiting_state
):
    from tldw_chatbook.Chat.console_chat_models import ConsoleLifecycleImpact
    from tldw_chatbook.Workflows.authoring import WorkflowAuthoring
    from tldw_chatbook.Workflows.models import DraftWriteFailed
    from tldw_chatbook.Workflows.session import SessionError

    authoring = WorkflowAuthoring(lambda: tmp_path / "authoring.sqlite3")
    await authoring.create("Before quit")
    drafts, documents = authoring.drafts, authoring.documents
    raw = json.loads(drafts.current.raw_text)
    raw["description"] = "Retained before quit"
    pending = drafts.update(json.dumps(raw))
    if waiting_state == "approval":
        harness.set_permission("workflow_read_file", "ask")
    _, old_ticket, old_run = await launch(harness)
    old_view = await until(harness.session, lambda v: v.state == waiting_state)
    app = WorkflowQuitHarness(harness.session)
    app.console_runtime = console_runtime
    app._workflow_authoring = authoring
    app.decision = True
    impact = ConsoleLifecycleImpact(1, 0, 0, 0, 0)
    console_runtime.set_chat_controller(
        SimpleNamespace(lifecycle_impact=lambda: impact)
    )

    def workflow_changed():
        nonlocal impact
        if harness.session.view().state == "cancelled":
            impact = ConsoleLifecycleImpact(2, 1, 0, 0, 0)

    release = harness.session.subscribe(workflow_changed)
    reconfirming = asyncio.Event()

    async def reconfirm(dialog):
        assert "Live agent runs: 1" in dialog.message
        with pytest.raises(DraftWriteFailed):
            drafts.update(pending.raw_text + " ")
        reconfirming.set()
        if cancel_waiter:
            await asyncio.Event().wait()
        return False

    app._await_console_quit_confirmation = reconfirm
    waiter = asyncio.create_task(app._confirm_and_quit())
    try:
        await asyncio.wait_for(reconfirming.wait(), 5)
        if cancel_waiter:
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
        else:
            await waiter
        assert app.cleanup_calls == 0 and not app._quit_in_progress
        assert_console_reopened(console_runtime)
        assert app._workflow_authoring is authoring
        assert app._workflow_session is harness.session
        assert harness.session.view().state == "cancelled"
        assert not harness.session.answer_review(old_run, "review", accept=True)
        if old_view.pending_effect:
            assert not harness.session.answer_effect(
                old_run,
                old_view.step_id,
                old_view.pending_effect.payload_json,
                approve=True,
            )
        with pytest.raises(SessionError):
            harness.session.start(old_ticket)
        await authoring.open()
        assert authoring.drafts is drafts and authoring.documents is documents
        assert drafts.current.raw_text == pending.raw_text
        created = await authoring.create("After aborted quit")
        document = session_tests.file_definition()
        document["metadata"]["tldw_workflow"].update(
            workflow_id=created.workflow_id, revision_id=created.revision_id
        )
        drafts.update(json.dumps(document))
        saved = await drafts.save_revision()
        assert documents.get_head(created.workflow_id) == saved
        harness.set_permission("workflow_read_file", "allow")
        ticket = await harness.session.prepare(saved, {}, harness.setup)
        new_run = harness.session.start(ticket)
        assert new_run != old_run
        await until(harness.session, lambda v: v.state == "review")
        assert harness.rows() == []
    finally:
        if not waiter.done():
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
        release()
        console_runtime.set_chat_controller(None)
        await authoring.close()
