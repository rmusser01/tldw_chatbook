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
@pytest.mark.parametrize("later", ["nothing", "navigation", "typed"])
async def test_ready_without_configured_hooks_sends_the_captured_stash_once(later):
    """A ready, empty hook inventory sends exactly the captured draft.

    Lead ruling (TASK-33620.15): what is sent is what the composer held at
    the press; whatever happens in the composer after it belongs to the next
    draft. dev (3601b7134f) refused a ready send whose composer was edited
    after the capture. Keys now flow while the hook snapshot is read, so that
    turned ordinary type-ahead into "Draft, chat or hooks changed; Send
    again." The captured draft is sent and committed; text typed after it
    stays, and caret navigation is no new draft, so the sent draft leaves the
    composer. (3601b7134f's own fix, the agent handoff commit in
    ``session.py``, is unchanged; a review still re-checks the draft.)
    """
    from textual.events import Key

    from tldw_chatbook.Widgets.Console import ConsoleComposerBar

    owner = HookPermissions()
    snapshot = owner.snapshot()
    assert snapshot.ready and snapshot.rows == ()
    composer = ConsoleComposerBar()
    composer.load_draft("original")
    captured = composer.capture_draft_for_send()
    if later == "typed":
        composer.insert_text(" suffix")
    elif later == "navigation":
        assert composer.handle_console_key(Key("left", None))
    sent = []

    async def review(_snapshot, _waiting, _on_cancel):
        raise AssertionError("An empty ready inventory must not request review")

    async def dispatch():
        # As the dispatcher does once runtime custody accepts the turn.
        sent.append(captured.text)
        composer.commit_captured_draft(captured)
        return ConsolePromptDispatchResult(
            ConsolePromptDispatchStatus.SENT, session_id="a"
        )

    hooks = ConsoleHooksController(
        hook_permissions_accessor=lambda: owner,
        request_review=review,
        current_session=lambda: "a",
        current_stash=composer.capture_draft_for_send,
        on_state=lambda _snapshot: None,
        notify=lambda _message, _severity: None,
    )
    result = await hooks.dispatch(
        "original", session_id="a", stash=captured, dispatch=dispatch
    )
    assert result.accepted
    assert sent == ["original"]
    assert composer.draft_text() == (" suffix" if later == "typed" else "")


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


def _approving_review(owner):
    async def review(snapshot, waiting, on_cancel):
        return HookReviewResult(
            "ready", owner.approve(snapshot, [snapshot.rows[0].entry.key])
        )

    return review


@pytest.mark.asyncio
async def test_a_ready_send_never_hands_off(hook_file):
    """Nothing to review: the Send is awaited inline even off a worker."""
    owner = HookPermissions()
    pending = owner.snapshot()
    owner.approve(pending, [pending.rows[0].entry.key])
    started: list = []
    hooks, stash, dispatch, sent = _handoff_controller(
        owner, _approving_review(owner), started
    )
    result = await hooks.dispatch(
        stash.text, session_id="a", stash=stash, dispatch=dispatch
    )
    assert result.status is ConsolePromptDispatchStatus.SENT
    assert started == [] and sent == ["original"] and not hooks._busy


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["typed", "session"])
async def test_a_ready_send_rechecks_its_chat_but_not_text_typed_after_capture(
    hook_file, change
):
    """TASK-33620.15: keys flow while the hook snapshot is read.

    Text typed after the capture belongs to the next draft (TASK-340), so it
    must not refuse the captured send; with no review, the draft is not
    re-checked. The chat still is: the dispatcher's send gate reads the
    visible chat right after this check.
    """
    owner = HookPermissions()
    pending = owner.snapshot()
    owner.approve(pending, [pending.rows[0].entry.key])
    hooks, stash, dispatch, sent = _handoff_controller(
        owner, _approving_review(owner), []
    )
    if change == "typed":
        typed = replace(stash, text="original x", edit_serial=2)
        hooks._stash = lambda: typed
    else:
        hooks._session = lambda: "b"
    result = await hooks.dispatch(
        stash.text, session_id="a", stash=stash, dispatch=dispatch
    )
    assert result.accepted is (change == "typed")
    assert sent == (["original"] if change == "typed" else [])


async def _dispatch_from_textual(caller, hooks, stash, dispatch):
    """Run ``hooks.dispatch`` from a real Textual caller (no ChatScreen).

    ``worker`` is a worker's own task, as a spoken "send" is. ``screen-pump``
    is a handler on a screen pushed from the app pump (``ConsoleHarness``).
    ``navigated-screen-pump`` is a handler on a screen pushed from a WORKER,
    as tab navigation pushes the Console (``TldwCli._dispatch_screen_
    navigation`` runs it in the ``screen-navigation`` worker). Textual starts
    a pump with ``create_task``, which copies the caller's contextvars, so
    that pump's handlers still find the navigation worker through
    ``get_current_worker()``.

    Returns:
        The dispatch result and the worker ``get_current_worker()`` answered
        with on the calling task (``None`` for ``NoActiveWorker``).
    """
    from textual.app import App
    from textual.screen import Screen
    from textual.worker import NoActiveWorker, get_current_worker

    seen: list = []

    async def send():
        try:
            seen.append(get_current_worker())
        except NoActiveWorker:
            seen.append(None)
        seen.append(
            await hooks.dispatch(
                stash.text, session_id="a", stash=stash, dispatch=dispatch
            )
        )

    app = App()
    async with app.run_test():
        if caller == "worker":
            await app.run_worker(send(), group="test-hook-send").wait()
        else:
            screen = Screen()
            if caller == "navigated-screen-pump":

                async def navigate():
                    await app.push_screen(screen)

                await app.run_worker(navigate(), group="screen-navigation").wait()
            else:
                await app.push_screen(screen)
            screen.call_later(send)
            async with asyncio.timeout(5):
                while len(seen) < 2:
                    await asyncio.sleep(0.01)
    return seen[1], seen[0]


@pytest.mark.asyncio
async def test_a_worker_caller_awaits_its_review_and_gets_the_outcome(hook_file):
    """A caller that IS a worker's task (spoken "send") waits for the review
    and gets the settled outcome, so its acknowledgement can be true."""
    owner = HookPermissions()
    started: list = []
    hooks, stash, dispatch, sent = _handoff_controller(
        owner, _approving_review(owner), started
    )
    result, worker = await _dispatch_from_textual("worker", hooks, stash, dispatch)
    assert worker is not None
    assert result.status is ConsolePromptDispatchStatus.SENT
    assert started == [] and sent == ["original"] and not hooks._busy


@pytest.mark.asyncio
@pytest.mark.parametrize("caller", ["screen-pump", "navigated-screen-pump"])
async def test_a_screen_handler_hands_its_review_to_a_worker(hook_file, caller):
    """TASK-33621.28 review: a handler must never await the review on its
    pump -- including on a Console reached by tab navigation, whose pump
    inherits the navigation worker as ``active_worker``. Asking
    ``get_current_worker()`` alone answered "in a worker" there, so the Send
    button and Workbench send awaited the whole review on the Console pump."""
    owner = HookPermissions()
    started: list = []
    hooks, stash, dispatch, sent = _handoff_controller(
        owner, _approving_review(owner), started
    )
    result, worker = await _dispatch_from_textual(caller, hooks, stash, dispatch)
    for continuation in started:
        continuation.close()
    # The negative control: only the navigated pump inherits a worker, so
    # this case really is the one get_current_worker() alone gets wrong.
    assert (worker is not None) is (caller == "navigated-screen-pump")
    assert result.status is ConsolePromptDispatchStatus.AWAITING_REVIEW
    assert len(started) == 1 and sent == []


@pytest.mark.asyncio
async def test_a_worker_that_cannot_start_releases_the_send(hook_file):
    owner = HookPermissions()

    def refuse(continuation):
        raise RuntimeError("screen is closing")

    hooks, stash, dispatch, sent = _handoff_controller(owner, lambda *args: None, [])
    hooks._start_worker = refuse
    with pytest.raises(RuntimeError, match="screen is closing"):
        await hooks.dispatch(stash.text, session_id="a", stash=stash, dispatch=dispatch)
    assert not hooks._busy and sent == []


@pytest.mark.asyncio
async def test_a_continuation_that_fails_before_its_review_releases_the_send(
    hook_file, monkeypatch
):
    """TASK-33621.28 review: the continuation's FIRST step must already be
    inside the ``try`` whose ``finally`` releases the Send. Its deferred
    diagnostics import sat above that ``try``, so a raise there left
    ``_busy`` set and refused every later Send as "already in progress"
    until restart. A ``None`` in ``sys.modules`` makes that import raise."""
    import sys

    owner = HookPermissions()
    started: list = []
    hooks, stash, dispatch, sent = _handoff_controller(
        owner, _approving_review(owner), started
    )
    await hooks.dispatch(stash.text, session_id="a", stash=stash, dispatch=dispatch)
    assert len(started) == 1 and hooks._busy
    monkeypatch.setitem(
        sys.modules, "tldw_chatbook.Chat.console_send_diagnostics", None
    )
    with pytest.raises(ImportError):
        await started[0]
    assert not hooks._busy and sent == []


@pytest.mark.asyncio
@pytest.mark.parametrize("factory", ["lazy", "eager"])
async def test_in_worker_task_is_true_only_on_a_workers_own_task(factory):
    """``in_worker_task`` under both task factories.

    Textual's own ``App.run_async`` installs ``asyncio.eager_task_factory``;
    ``run_test`` does not. Under the eager factory a worker's first step
    runs inside ``create_task``, before ``Worker._task`` is assigned -- the
    live hfrf1 run showed the review guard logging a false ERROR from
    exactly there, while every (lazy) test passed.
    """
    from textual.app import App
    from textual.screen import Screen

    from tldw_chatbook.UI.Console_Modules.hooks import in_worker_task

    seen: dict[str, bool] = {}

    async def in_worker():
        seen["worker first step"] = in_worker_task()
        await asyncio.sleep(0)
        seen["worker after a suspension"] = in_worker_task()

    def on_pump(label):
        return lambda: seen.__setitem__(label, in_worker_task())

    app = App()
    async with app.run_test():
        loop = asyncio.get_running_loop()
        previous = loop.get_task_factory()
        if factory == "eager":
            loop.set_task_factory(asyncio.eager_task_factory)
        try:
            await app.run_worker(in_worker(), group="probe").wait()
            pushed, navigated = Screen(), Screen()
            await app.push_screen(pushed)
            pushed.call_later(on_pump("screen pushed from the app pump"))

            async def navigate():
                await app.push_screen(navigated)

            await app.run_worker(navigate(), group="screen-navigation").wait()
            navigated.call_later(on_pump("screen pushed from a worker"))
            async with asyncio.timeout(5):
                while len(seen) < 4:
                    await asyncio.sleep(0.01)
        finally:
            loop.set_task_factory(previous)
    assert seen == {
        "worker first step": True,
        "worker after a suspension": True,
        "screen pushed from the app pump": False,
        "screen pushed from a worker": False,
    }
