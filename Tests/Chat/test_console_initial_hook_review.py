"""Resident initial hook review keeps real consent and finite native ownership."""

from __future__ import annotations

import asyncio
import gc
import threading
import weakref
from types import SimpleNamespace

import pytest

from Tests.Agents.test_hook_permissions import hook_file as _hook_file
from Tests.UI.test_console_hook_refresh_lifetime import _OriginalVisit, _until
from tldw_chatbook.Agents.hook_permissions import HookPermissions
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_runtime import (
    CONSOLE_RUNTIME_SHUTDOWN_GRACE_SECONDS,
    CONSOLE_SESSION_CLOSE_GRACE_SECONDS,
    ConsoleRuntime,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]
hook_file = _hook_file


class _View:
    def __init__(self, app):
        self.app = app

    def console_view_hooks(self):
        return {}


@pytest.fixture
async def review_runtime(hook_file):
    store = ConsoleChatStore()
    session = store.ensure_session()
    app = SimpleNamespace(
        screen=object(),
        app_config={},
        call_from_thread=lambda callback, *args: callback(*args),
    )
    runtime = ConsoleRuntime(app)
    owner = runtime.ensure_hook_permissions()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=None,
        hook_permissions_accessor=runtime.ensure_hook_permissions,
    )
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    state = SimpleNamespace(
        runtime=runtime,
        controller=controller,
        host=controller._interrupt_host,
        store=store,
        session=session,
        app=app,
        owner=owner,
        snapshot=owner.snapshot(),
        tasks=[],
    )
    try:
        yield state
    finally:
        await runtime.dispose()
        await asyncio.gather(*state.tasks, return_exceptions=True)


async def _request(state, review_id="review", generation=1, *, session=None):
    session = session or state.session
    task = asyncio.create_task(
        state.runtime.request_initial_hook_review(
            session.id, review_id, generation, state.snapshot
        )
    )
    state.tasks.append(task)
    assert await _until(
        lambda: review_id in state.host.registries.get("hook_review", {}), 5
    ), "the request never published its resident record"
    return task


def _present(state, review_id="review", generation=1):
    view = _View(state.app)
    state.app.screen = view
    attachment = state.runtime.attach_view(view)
    assert state.runtime.finish_view_reconciliation(view, attachment)
    projection = state.host.claim_hook_review_presentation(
        review_id, generation, attachment
    )
    assert projection is not None
    state.app.screen = SimpleNamespace(_console_hook_review_projection=projection)
    # A real modal suspends its Console after the initial claim.
    state.runtime._reconciled_view = None
    return view, attachment, projection


def _keys(snapshot):
    return tuple(row.entry.key for row in snapshot.rows if row.entry)


async def test_cancelled_waiter_and_collected_view_leave_exact_review_answerable(
    review_runtime,
):
    from tldw_chatbook.Chat.console_hook_review import HookReviewResult

    state = review_runtime
    waiter = await _request(state)
    view, attachment, old = _present(state)
    reference = weakref.ref(view)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert state.host.release_hook_review_presentation(
        "review", 1, old.presentation_token
    )
    assert state.runtime.detach_view(view, attachment)
    state.app.screen = object()
    del view
    gc.collect()
    assert reference() is None, "the resident review retained a disposable view"
    pending = state.controller.pending_decision_projection(state.session.id)
    assert pending is not None and pending.decision_id == "review"

    _view, _attachment, current = _present(state)
    assert current.presentation_token is not old.presentation_token
    assert not state.runtime.resolve_initial_hook_review(
        "review",
        1,
        HookReviewResult("cancel"),
        presentation_token=old.presentation_token,
    )
    with pytest.raises(RuntimeError):
        await state.runtime.apply_hook_review_action(
            "review",
            1,
            "approve",
            old.snapshot,
            _keys(old.snapshot),
            presentation_token=old.presentation_token,
        )
    assert not state.owner.snapshot().ready
    assert state.runtime.resolve_initial_hook_review(
        "review",
        1,
        HookReviewResult("cancel"),
        presentation_token=current.presentation_token,
    )
    assert await _until(
        lambda: state.controller.pending_decision_projection(state.session.id) is None,
        5,
    )
    assert state.host.hook_review_retirements() == ()


async def test_live_permission_callback_and_ready_use_actual_fresh_consent(
    review_runtime,
    monkeypatch,
):
    from tldw_chatbook.Chat.console_hook_review import HookReviewResult

    state = review_runtime
    waiter = await _request(state)
    _view, _attachment, projection = _present(state)
    original_approve = state.owner.approve
    calls = []

    def observed_approve(expected, keys):
        calls.append((expected, keys, threading.current_thread()))
        return original_approve(expected, keys)

    # Replacement after request creation must be observed at action time.
    monkeypatch.setattr(state.owner, "approve", observed_approve)
    keys = _keys(state.snapshot)
    approved = await state.runtime.apply_hook_review_action(
        "review",
        1,
        "approve",
        state.snapshot,
        keys,
        presentation_token=projection.presentation_token,
    )
    assert approved.ready and not waiter.done()
    assert len(calls) == 1 and calls[0][:2] == (state.snapshot, keys)
    assert calls[0][2] is not threading.current_thread()
    other = HookPermissions()
    assert not other.revoke(other.snapshot(), keys[0]).ready

    probe = _OriginalVisit(state.owner)
    with probe.installed():
        try:
            assert state.runtime.resolve_initial_hook_review(
                "review",
                1,
                HookReviewResult("ready", approved),
                presentation_token=projection.presentation_token,
            )
            assert await _until(probe.entered.is_set, 5)
            probe.assert_live()
            assert not waiter.done(), "Ready settled before the fresh native read"
            probe.release.set()
            assert await _until(probe.retired, 5)
            assert await _until(lambda: not state.host.hook_review_retirements(), 5)
            assert not waiter.done() or waiter.result().kind != "ready"
        finally:
            probe.release.set()
    probe.assert_retired()
    assert not state.owner.snapshot().ready


@pytest.mark.parametrize("ending", ["close", "dispose"])
async def test_original_consent_operation_survives_remount_cancellation_and_close(
    review_runtime,
    ending,
):
    from tldw_chatbook.Chat.console_hook_review import HookReviewResult

    state = review_runtime
    waiter = await _request(state)
    view, attachment, old = _present(state)
    # Reuse the existing passive raw-config/lease hold inside ORIGINAL _decision.
    probe = _OriginalVisit(state.owner)
    probe.snapshot_code = HookPermissions._decision.__code__
    action = ending_task = None
    with probe.installed():
        try:
            existing_tasks = asyncio.all_tasks()
            action = asyncio.create_task(
                state.runtime.apply_hook_review_action(
                    "review",
                    1,
                    "approve",
                    state.snapshot,
                    _keys(state.snapshot),
                    presentation_token=old.presentation_token,
                )
            )
            state.tasks.append(action)
            assert await _until(probe.entered.is_set, 5)
            probe.assert_live()
            retirements = state.host.hook_review_retirements(state.session.id)
            assert len(retirements) == 1 and not retirements[0].done()
            owned_tasks = asyncio.all_tasks() - existing_tasks
            # View projection may start this unrelated policy reader too.
            owned_tasks.discard(state.runtime._canvas_policy_read_task)
            assert state.host.release_hook_review_presentation(
                "review", 1, old.presentation_token
            )
            assert state.runtime.detach_view(view, attachment)
            _new_view, _new_attachment, current = _present(state)
            assert current.busy
            with pytest.raises(RuntimeError):
                await state.runtime.apply_hook_review_action(
                    "review",
                    1,
                    "approve",
                    state.snapshot,
                    _keys(state.snapshot),
                    presentation_token=current.presentation_token,
                )
            assert not state.runtime.resolve_initial_hook_review(
                "review",
                1,
                HookReviewResult("ready", state.snapshot),
                presentation_token=current.presentation_token,
            )
            # Include helper-created tasks: cancelling a to_thread wrapper
            # does not retire the original native consent write.
            for _ in range(2):
                for owned_task in owned_tasks:
                    owned_task.cancel()
                await asyncio.sleep(0)
            assert not retirements[0].done(), "Task cancellation retired live consent"
            if ending == "close":
                revision = state.controller.lifecycle_impact(
                    session_id=state.session.id
                ).revision
                ending_task = asyncio.create_task(
                    state.runtime.close_session(
                        state.session.id, expected_revision=revision
                    )
                )
                grace = CONSOLE_SESSION_CLOSE_GRACE_SECONDS
            else:
                ending_task = asyncio.create_task(state.runtime.dispose())
                grace = CONSOLE_RUNTIME_SHUTDOWN_GRACE_SECONDS
            state.tasks.append(ending_task)
            await asyncio.sleep(grace + 0.1)
            ending_task.cancel()
            await asyncio.sleep(0)
            ending_task.cancel()
            await asyncio.sleep(0)
            assert (
                not ending_task.done()
            ), "Close abandoned an issued native consent operation"
            assert not retirements[0].done()
            probe.assert_live()
        finally:
            probe.release.set()
            if action is not None:
                await asyncio.gather(action, return_exceptions=True)
            if ending_task is not None:
                await asyncio.gather(ending_task, return_exceptions=True)
            assert await _until(probe.retired, 5)
    probe.assert_retired()
    assert retirements[0].done() and not retirements[0].cancelled()
    assert waiter.done() and waiter.result().kind == "cancel"
    assert state.host.hook_review_retirements() == ()
    assert state.controller.pending_decision_projection(state.session.id) is None
    # Physical consent completion survives cancellation; no invented rollback.
    assert HookPermissions().snapshot().ready


@pytest.mark.parametrize("queues_first", [False, True])
async def test_submission_failure_retires_reserved_operation_without_permission_call(
    review_runtime,
    monkeypatch,
    queues_first,
):
    state = review_runtime
    await _request(state)
    _view, _attachment, projection = _present(state)
    calls = []
    original = state.owner.approve

    def observe(expected, keys):
        calls.append(True)
        return original(expected, keys)

    queued = []

    def refuse_submission(_executor, callback, *args):
        if queues_first:
            queued.append((callback, args))
        raise RuntimeError("executor refused submission")

    with monkeypatch.context() as patch:
        patch.setattr(state.owner, "approve", observe)
        patch.setattr(asyncio.get_running_loop(), "run_in_executor", refuse_submission)
        with pytest.raises(RuntimeError):
            await state.runtime.apply_hook_review_action(
                "review",
                1,
                "approve",
                state.snapshot,
                _keys(state.snapshot),
                presentation_token=projection.presentation_token,
            )
    for callback, args in queued:
        callback(*args)
    assert calls == []
    assert state.host.hook_review_retirements() == ()
    state.host.release_hook_review_presentation(
        "review", 1, projection.presentation_token
    )
    _view, _attachment, refreshed = _present(state)
    assert not refreshed.busy and not state.owner.snapshot().ready


@pytest.mark.parametrize("changed", ["owner", "binding", "ephemeral"])
async def test_changed_request_source_refuses_before_original_permission_callback(
    review_runtime,
    monkeypatch,
    changed,
):
    state = review_runtime
    await _request(state)
    _view, _attachment, projection = _present(state)
    calls = []
    original = state.owner.approve

    def observe(expected, keys):
        calls.append(True)
        return original(expected, keys)

    monkeypatch.setattr(state.owner, "approve", observe)
    if changed == "owner":
        monkeypatch.setattr(state.runtime, "_hook_permissions", HookPermissions())
    elif changed == "binding":
        state.session.conversation_binding_revision += 1
    else:
        state.session.ephemeral = not state.session.ephemeral
    with pytest.raises(RuntimeError):
        await state.runtime.apply_hook_review_action(
            "review",
            1,
            "approve",
            state.snapshot,
            _keys(state.snapshot),
            presentation_token=projection.presentation_token,
        )
    assert calls == []
    assert not state.owner.snapshot().ready
    assert state.host.hook_review_retirements() == ()


async def test_mixed_fifo_duplicate_request_and_late_retirement_keep_exact_successor(
    review_runtime,
):
    from tldw_chatbook.Chat.console_hook_review import HookReviewResult

    state = review_runtime
    earlier = {"event": threading.Event(), "session_id": state.session.id}
    state.controller._pending_approval_rounds["earlier"] = earlier
    assert state.controller._publish_pending_decision(
        round_state=earlier,
        payload={
            "session_id": state.session.id,
            "phase": "finishing",
            "run_id": "run",
            "calls": [{"call_id": "call", "llm_name": "tool"}],
        },
        decision_type="approval",
        decision_id="earlier",
        timeout_seconds=0,
        retained_store=state.controller._parked_approval_payloads,
    )
    first = await _request(state)
    assert (
        state.controller.pending_decision_projection(state.session.id).decision_id
        == "earlier"
    )
    view = _View(state.app)
    state.app.screen = view
    attachment = state.runtime.attach_view(view)
    assert state.runtime.finish_view_reconciliation(view, attachment)
    assert state.host.claim_hook_review_presentation("review", 1, attachment) is None
    with pytest.raises(RuntimeError):
        await state.runtime.request_initial_hook_review(
            state.session.id, "review", 1, state.snapshot
        )
    state.host.complete_definitive_tool("run", "call", "tool")
    _view, _attachment, projection = _present(state)
    assert state.runtime.resolve_initial_hook_review(
        "review",
        1,
        HookReviewResult("cancel"),
        presentation_token=projection.presentation_token,
    )
    assert (await first).kind == "cancel"
    assert await _until(lambda: "review" not in state.host.registries["hook_review"], 5)
    successor = await _request(state, generation=2)
    assert not state.host.retire_hook_review("review", 1)
    assert not successor.done()
    assert (
        state.controller.pending_decision_projection(state.session.id).decision_id
        == "review"
    )


async def test_closing_one_session_keeps_other_review_and_shared_consent_owner(
    review_runtime,
):
    state = review_runtime
    first = await _request(state)
    other = state.store.create_session(activate=False)
    second = await _request(state, "other-review", session=other)
    revision = state.controller.lifecycle_impact(session_id=state.session.id).revision
    await state.runtime.close_session(state.session.id, expected_revision=revision)
    assert (await first).kind == "cancel"
    assert not second.done()
    assert state.runtime.ensure_hook_permissions() is state.owner
    pending = state.controller.pending_decision_projection(other.id)
    assert pending is not None and pending.decision_id == "other-review"
    state.host.cancel_hook_reviews(other.id)
    assert (await second).kind == "cancel"
    assert state.host.hook_review_retirements() == ()
    fresh = state.owner.snapshot()
    assert state.owner.approve(fresh, _keys(fresh)).ready


@pytest.mark.parametrize("replace_controller", [False, True])
async def test_driver_cancelled_before_first_step_retires_exact_reservation(
    review_runtime,
    replace_controller,
):
    from tldw_chatbook.Chat.console_hook_review import HookReviewResult

    state = review_runtime
    waiter = await _request(state)
    _view, _attachment, projection = _present(state)
    assert state.runtime.resolve_initial_hook_review(
        "review",
        1,
        HookReviewResult("ready", state.snapshot),
        presentation_token=projection.presentation_token,
    )
    operation = state.host.registries["hook_review"]["review"]["operation"]
    assert operation.task is not None
    if replace_controller:
        state.runtime.set_chat_controller(
            ConsoleChatController(
                store=ConsoleChatStore(),
                provider_gateway=None,
                hook_permissions_accessor=state.runtime.ensure_hook_permissions,
            )
        )
    operation.task.cancel()
    state.tasks.append(operation.task)
    try:
        assert await _until(
            lambda: operation.retired.done(), 5
        ), "a driver cancelled before entry stranded its reservation"
        assert state.host.hook_review_retirements() == ()
        assert not waiter.done() and not state.owner.snapshot().ready
    finally:
        state.runtime.set_chat_controller(state.controller)
        # No native body has started; retire a failed pre-entry RED for teardown.
        state.host.finish_hook_review_operation(
            operation, None, error=asyncio.CancelledError()
        )


async def test_eager_task_factory_failure_cannot_retire_live_native_consent(
    review_runtime,
):
    state = review_runtime
    await _request(state)
    _view, _attachment, projection = _present(state)
    probe = _OriginalVisit(state.owner)
    probe.snapshot_code = HookPermissions._decision.__code__
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    factory_tasks = []
    action = None

    def starts_then_raises(owner_loop, coroutine, **kwargs):
        if coroutine.cr_code is ConsoleRuntime._execute_hook_review_operation.__code__:
            task = asyncio.Task(
                coroutine,
                loop=owner_loop,
                context=kwargs.get("context"),
                eager_start=True,
            )
            factory_tasks.append(task)
            raise RuntimeError("factory failed after eager native issuance")
        if previous_factory is not None:
            return previous_factory(owner_loop, coroutine, **kwargs)
        return asyncio.Task(coroutine, loop=owner_loop, **kwargs)

    with probe.installed():
        try:
            loop.set_task_factory(starts_then_raises)
            action = asyncio.Task(
                state.runtime.apply_hook_review_action(
                    "review",
                    1,
                    "approve",
                    state.snapshot,
                    _keys(state.snapshot),
                    presentation_token=projection.presentation_token,
                ),
                loop=loop,
            )
            state.tasks.append(action)
            assert await _until(probe.entered.is_set, 5)
            probe.assert_live()
            retirements = state.host.hook_review_retirements(state.session.id)
            assert (
                len(retirements) == 1 and not retirements[0].done()
            ), "an eager factory failure retired the original native write"
        finally:
            loop.set_task_factory(previous_factory)
            probe.release.set()
            if action is not None:
                await asyncio.gather(action, *factory_tasks, return_exceptions=True)
            assert await _until(probe.retired, 5)
    probe.assert_retired()
    assert action.result().ready
    assert state.host.hook_review_retirements() == ()


async def test_new_modal_token_survives_delayed_old_modal_unmount(review_runtime):
    from tldw_chatbook.Chat.console_hook_review import HookReviewResult
    from tldw_chatbook.Widgets.Console.console_hooks_review_modal import (
        project_runtime_hook_review,
    )

    state = review_runtime
    waiter = await _request(state)
    view = _View(state.app)
    state.app.screen = view
    state.app.screen_stack = [view]

    def push_screen(modal):
        state.app.screen_stack.append(modal)
        state.app.screen = modal

    state.app.push_screen = push_screen
    attachment = state.runtime.attach_view(view)
    assert state.runtime.finish_view_reconciliation(view, attachment)
    pending = state.controller.pending_decision_projection(state.session.id)
    assert project_runtime_hook_review(state.runtime, pending)
    old_modal = state.app.screen
    old = old_modal._console_hook_review_projection

    # Textual posts ScreenResume before the popped modal's Unmount finishes.
    state.app.screen_stack.remove(old_modal)
    state.app.screen = view
    state.runtime._reconciled_view = view
    assert project_runtime_hook_review(state.runtime, pending)
    new_modal = state.app.screen
    current = new_modal._console_hook_review_projection
    assert new_modal is not old_modal
    assert current.presentation_token is not old.presentation_token
    old_modal.on_unmount()
    assert project_runtime_hook_review(state.runtime, pending)
    assert state.app.screen is new_modal
    assert new_modal._runtime_result(HookReviewResult("cancel"))
    assert (await waiter).kind == "cancel"
    assert state.controller.pending_decision_projection(state.session.id) is None
