"""Hook review happens before new draft custody, with one pinned continuation."""

import asyncio
from dataclasses import replace

import pytest

from Tests.Agents.test_hook_permissions import hook_file as _hook_file
from Tests.Chat.test_console_chat_controller import (
    ConsoleChatController,
    ConsoleChatStore,
    RecordingStreamingGateway,
)
from tldw_chatbook.Agents.hook_permissions import HookPermissions
from tldw_chatbook.UI.Console_Modules.hooks import (
    ConsoleHooksController,
    HookReviewResult,
)
from tldw_chatbook.UI.Console_Modules.prompt_queue import (
    ConsolePromptDispatchResult,
    ConsolePromptDispatchStatus,
)
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleDraftStash

pytestmark = pytest.mark.bootstrap_profile

hook_file = _hook_file


@pytest.mark.asyncio
async def test_pending_review_refuses_before_echo_then_approved_send_succeeds(
    hook_file,
):
    owner = HookPermissions()
    store = ConsoleChatStore()
    session = store.ensure_session()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(
        store=store, provider_gateway=gateway, hook_permissions_accessor=lambda: owner
    )
    refused = await controller.submit_draft("draft", session_id=session.id)
    assert not refused.accepted and not refused.should_clear_draft
    assert store.messages_for_session(session.id) == []
    assert gateway.messages_seen is None
    pending = owner.snapshot()
    owner.approve(pending, [pending.rows[0].entry.key])
    accepted = await controller.submit_draft("draft", session_id=session.id)
    assert accepted.accepted and gateway.messages_seen


@pytest.mark.asyncio
async def test_controller_without_owner_cannot_bypass_enabled_hooks(hook_file):
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    refused = await controller.submit_draft("draft")
    assert not refused.accepted and gateway.messages_seen is None
    assert store.sessions() == []


@pytest.mark.asyncio
async def test_pending_review_cannot_enter_prompt_queue(hook_file):
    owner = HookPermissions()
    store = ConsoleChatStore()
    session = store.ensure_session()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=RecordingStreamingGateway(),
        hook_permissions_accessor=lambda: owner,
    )
    registry = controller.prompt_queue_registry
    chain = registry.begin_chain(
        session.id,
        context_epoch=store.conversation_context_epoch(session.id),
        expected_revision=registry.snapshot(session.id).revision,
    )
    refused = await controller.queue_prompt(
        session.id, text="queued", expected_revision=chain.snapshot.revision
    )
    assert not refused.applied and registry.snapshot(session.id).total_count == 0
    pending = owner.snapshot()
    owner.approve(pending, [pending.rows[0].entry.key])
    accepted = await controller.queue_prompt(
        session.id, text="queued", expected_revision=chain.snapshot.revision
    )
    assert accepted.applied and registry.snapshot(session.id).total_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["cancel", "edit", "session", "return"])
async def test_late_ready_cannot_send_a_cancelled_or_changed_draft(hook_file, change):
    owner = HookPermissions()
    future = asyncio.get_running_loop().create_future()
    entered = asyncio.Event()
    stash = ConsoleDraftStash([], "original", False, edit_serial=1, generation=1)
    current = {"session": "a", "stash": stash}
    sent = []

    async def review(snapshot, waiting, on_cancel):
        entered.set()
        return await future

    async def dispatch():
        sent.append("original")
        return ConsolePromptDispatchResult(
            ConsolePromptDispatchStatus.SENT, session_id="a"
        )

    hooks = ConsoleHooksController(
        hook_permissions_accessor=lambda: owner,
        request_review=review,
        current_session=lambda: current["session"],
        current_stash=lambda: current["stash"],
        on_state=lambda snapshot: None,
        notify=lambda message, severity: None,
    )
    task = asyncio.create_task(
        hooks.dispatch("original", session_id="a", stash=stash, dispatch=dispatch)
    )
    await entered.wait()
    duplicate = await hooks.dispatch(
        "original", session_id="a", stash=stash, dispatch=dispatch
    )
    assert not duplicate.accepted
    if change == "edit":
        current["stash"] = replace(stash, edit_serial=2)
    elif change == "session":
        current["session"] = "b"
    else:
        hooks.cancel_pending()
    pending = owner.snapshot()
    approved = owner.approve(pending, [pending.rows[0].entry.key])
    future.set_result(HookReviewResult("ready", approved))
    assert not (await task).accepted
    assert sent == [] and current["stash"].text == "original"


@pytest.mark.asyncio
async def test_ready_dispatches_the_captured_send_once(hook_file):
    owner = HookPermissions()
    stash = ConsoleDraftStash([], "original", False, edit_serial=1, generation=1)
    sent = []

    async def review(snapshot, waiting, on_cancel):
        approved = owner.approve(snapshot, [snapshot.rows[0].entry.key])
        return HookReviewResult("ready", approved)

    async def dispatch():
        sent.append(stash.text)
        return ConsolePromptDispatchResult(
            ConsolePromptDispatchStatus.SENT, session_id="a"
        )

    hooks = ConsoleHooksController(
        hook_permissions_accessor=lambda: owner,
        request_review=review,
        current_session=lambda: "a",
        current_stash=lambda: stash,
        on_state=lambda snapshot: None,
        notify=lambda message, severity: None,
    )
    assert (
        await hooks.dispatch(stash.text, session_id="a", stash=stash, dispatch=dispatch)
    ).accepted
    assert sent == ["original"]


@pytest.mark.asyncio
async def test_hooks_added_after_acceptance_retain_the_durable_recovery_owner(
    hook_file,
):
    import toml

    raw = toml.loads(hook_file.read_text())
    raw["hooks"]["enabled"] = False
    hook_file.write_text(toml.dumps(raw))
    owner = HookPermissions()
    from Tests.Chat.test_console_chat_controller import FakePersistence

    store = ConsoleChatStore(persistence=FakePersistence())
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(
        store=store, provider_gateway=gateway, hook_permissions_accessor=lambda: owner
    )

    def enable_after_acceptance():
        raw["hooks"]["enabled"] = True
        hook_file.write_text(toml.dumps(raw))

    controller.on_submission_accepted = enable_after_acceptance
    accepted = await controller.submit_draft("retained")
    assert accepted.accepted and accepted.preparation_id
    assert gateway.messages_seen is None
    pending = owner.snapshot()
    owner.approve(pending, [pending.rows[0].entry.key])
    resumed = await controller.resume_durable_postcommit(accepted.preparation_id)
    assert resumed.accepted and gateway.messages_seen
    assert [
        row.content
        for row in store.messages_for_session(accepted.session_id)
        if row.role.value == "user"
    ] == ["retained"]


async def test_indicator_refresh_after_runtime_disposal_is_a_noop():
    states = []

    def disposed():
        raise RuntimeError("Console runtime is disposed.")

    hooks = ConsoleHooksController(
        hook_permissions_accessor=disposed,
        request_review=lambda *args: None,
        current_session=lambda: "a",
        current_stash=lambda: None,
        on_state=states.append,
        notify=lambda *args: None,
    )
    await hooks.refresh()
    assert states == []


def _handoff_controller(owner, review, started):
    stash = ConsoleDraftStash([], "original", False, edit_serial=1, generation=1)
    sent: list[str] = []

    async def dispatch():
        sent.append(stash.text)
        return ConsolePromptDispatchResult(
            ConsolePromptDispatchStatus.SENT, session_id="a"
        )

    hooks = ConsoleHooksController(
        hook_permissions_accessor=lambda: owner,
        request_review=review,
        current_session=lambda: "a",
        current_stash=lambda: stash,
        on_state=lambda snapshot: None,
        notify=lambda message, severity: None,
        start_worker=started.append,
    )
    return hooks, stash, dispatch, sent


@pytest.mark.asyncio
async def test_a_send_outside_a_worker_hands_its_review_to_a_worker(hook_file):
    """TASK-33621.28: a non-worker caller -- a handler, or Enter's
    app.call_later callback on the APP pump -- must not wait for the review.
    It gets AWAITING_REVIEW at once; the continuation owns the review, holds
    the Send busy until it settles, and dispatches the captured Send once."""
    owner = HookPermissions()
    answer = asyncio.get_running_loop().create_future()
    reviewed = []

    async def review(snapshot, waiting, on_cancel):
        reviewed.append(waiting)
        return await answer

    started: list = []
    hooks, stash, dispatch, sent = _handoff_controller(owner, review, started)
    result = await hooks.dispatch(
        stash.text, session_id="a", stash=stash, dispatch=dispatch
    )
    assert result.status is ConsolePromptDispatchStatus.AWAITING_REVIEW
    assert not result.accepted
    assert len(started) == 1 and reviewed == [] and sent == []
    assert hooks._busy
    duplicate = await hooks.dispatch(
        stash.text, session_id="a", stash=stash, dispatch=dispatch
    )
    assert duplicate.status is ConsolePromptDispatchStatus.REFUSED
    assert len(started) == 1

    continuation = asyncio.create_task(started[0])
    await asyncio.sleep(0)
    assert reviewed == [True] and hooks.review_open
    pending = owner.snapshot()
    answer.set_result(
        HookReviewResult("ready", owner.approve(pending, [pending.rows[0].entry.key]))
    )
    assert (await continuation).accepted
    assert sent == ["original"] and not hooks._busy and not hooks.review_open


@pytest.mark.asyncio
async def test_a_cancelled_handed_off_review_refuses_and_releases_the_send(hook_file):
    owner = HookPermissions()

    async def review(snapshot, waiting, on_cancel):
        return HookReviewResult("cancel")

    started: list = []
    notices: list[str] = []
    hooks, stash, dispatch, sent = _handoff_controller(owner, review, started)
    hooks._notify = lambda message, severity: notices.append(message)
    await hooks.dispatch(stash.text, session_id="a", stash=stash, dispatch=dispatch)
    result = await started[0]
    assert result.status is ConsolePromptDispatchStatus.REFUSED
    assert notices == ["Send cancelled; draft kept."]
    assert sent == [] and not hooks._busy


@pytest.mark.asyncio
@pytest.mark.parametrize("caller", ["ready-snapshot", "worker"])
async def test_a_ready_send_or_a_worker_caller_never_hands_off(hook_file, caller):
    """Nothing to review, or a caller that IS a worker (spoken "send"): the
    Send is awaited inline and its caller gets the settled outcome."""
    import contextvars

    from textual.worker import active_worker

    owner = HookPermissions()
    if caller == "ready-snapshot":
        pending = owner.snapshot()
        owner.approve(pending, [pending.rows[0].entry.key])

    async def review(snapshot, waiting, on_cancel):
        return HookReviewResult(
            "ready", owner.approve(snapshot, [snapshot.rows[0].entry.key])
        )

    started: list = []
    hooks, stash, dispatch, sent = _handoff_controller(owner, review, started)
    context = contextvars.copy_context()
    if caller == "worker":
        context.run(active_worker.set, object())
    result = await asyncio.get_running_loop().create_task(
        hooks.dispatch(stash.text, session_id="a", stash=stash, dispatch=dispatch),
        context=context,
    )
    assert result.status is ConsolePromptDispatchStatus.SENT
    assert started == [] and sent == ["original"] and not hooks._busy


@pytest.mark.asyncio
async def test_a_worker_that_cannot_start_releases_the_send(hook_file):
    owner = HookPermissions()

    def refuse(continuation):
        raise RuntimeError("screen is closing")

    hooks, stash, dispatch, sent = _handoff_controller(
        owner, lambda *args: None, []
    )
    hooks._start_worker = refuse
    with pytest.raises(RuntimeError, match="screen is closing"):
        await hooks.dispatch(stash.text, session_id="a", stash=stash, dispatch=dispatch)
    assert not hooks._busy and sent == []
