"""A Console send's start lasts from its own acceptance to its own provider request.

TASK-33620.15.4 keeps three display reads on the UI loop during a send's
start (the Library activity rows and the Inspector's registry and appearance
reads): off the loop each opened its own SQLite connection, which delayed
first sends. Its review (TASK-36610) found the window keyed to the wrong
things: the screen remembered only its latest acceptance, and the start
ended at the process's latest provider request, whoever made it. So another
session's provider request ended this send's start early, and a queued
prompt's dispatch or an automatic submission (an agent wake, a goal
iteration, an agent's chat start) never opened one.

Most tests drive the real ChatScreen, runtime and controller on the mounted
send harness. Each accepted send waits just before its provider request on
its own gate, and the gateway double notes that request inside the send's
own work, as the real gateway does at adapter entry. One drives the real
gateway (only the provider call replaced), and the chat-start case runs
screen-free on a real controller. The window's 3 s limit is lifted so a slow
host cannot end a start before the assertion that reads it.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from Tests.UI.test_console_native_chat_flow import (
    _ReadyResolutionGateway,
    _build_console_send_test_app,
    _select_llamacpp_console,
)
from Tests.UI.test_console_send_acknowledgement import (
    REPLY,
    HeldGateway,
    Timeline,
    eager_tasks,
    press,
    ready_console,
    until,
)
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.UI.Console_Modules import config_sync

pytestmark = pytest.mark.bootstrap_profile


class StagedGateway(HeldGateway):
    """Holds each accepted send just before its provider request, on its own gate.

    The hold is in the stream, after the send's durable acceptance (so a
    prompt can queue behind it). Released, the stream notes the provider
    request (``provider_entry``) in the send's own work before it yields,
    where the real gateway notes it at adapter entry.
    """

    def __init__(self, timeline: Timeline) -> None:
        super().__init__(timeline)
        self.gates: list[asyncio.Event] = []

    async def resolve_for_send(self, selection):
        return await _ReadyResolutionGateway.resolve_for_send(self, selection)

    async def stream_chat(self, resolution, messages, **kwargs):
        from tldw_chatbook.Chat.console_send_diagnostics import record_send_stage

        gate = asyncio.Event()
        self.gates.append(gate)
        await gate.wait()
        record_send_stage("provider_entry")
        self.stream_calls += 1
        yield REPLY

    def release_all(self) -> None:
        for gate in self.gates:
            gate.set()


def _build() -> tuple[ConsoleHarness, StagedGateway]:
    app = _build_console_send_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    host = ConsoleHarness(app)
    gateway = StagedGateway(Timeline(host))
    app.console_provider_gateway_factory = lambda: gateway
    return host, gateway


def _in_progress(console) -> bool:
    return config_sync.send_start_in_progress(console)


@pytest.fixture
def no_limit(monkeypatch):
    """Only a send's own provider request (or its end) closes its start."""
    monkeypatch.setattr(config_sync, "SEND_START_LIMIT_SECONDS", 600.0)


async def _settle_custody(console) -> None:
    """End every turn still in runtime custody before the app tears down."""
    runtime = console._console_runtime()
    tasks = [
        record.task
        for record in tuple(runtime._turn_custody.values())
        if record.task is not None and not record.task.done()
    ]
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 15)


@pytest.mark.asyncio
async def test_a_sends_start_ends_at_its_own_provider_request_not_another_sessions(
    no_limit,
):
    """Another session's provider request leaves this send's start open.

    Session A's send is accepted first and waits; session B's send is
    accepted after it and waits too. A then reaches its provider. B has not,
    so B's start (and the live reads it keeps) must still be open; B's own
    provider request ends it. On the base the start ended at any provider
    request in the process made after the screen's latest acceptance.
    """
    host, gateway = _build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, _composer = await ready_console(host, pilot, gateway)
            store = console._console_chat_store
            session_a = store.active_session_id
            try:
                press(host, "enter", "\r")  # Session A's send.
                await until(lambda: len(gateway.gates) == 1, timeout=30)
                console.action_new_console_tab()
                await until(
                    lambda: store.active_session_id != session_a
                    and console._console_visible_draft_session_id
                    == store.active_session_id,
                    timeout=30,
                )
                await _select_llamacpp_console(console)  # B's own settings.
                await pilot.pause(0.3)
                composer = console._console_composer_or_none()
                composer.focus()
                composer.load_draft("session B's own send")
                await pilot.pause()
                press(host, "enter", "\r")  # Session B's send.
                await until(lambda: len(gateway.gates) == 2, timeout=30)
                assert _in_progress(console), "B's acceptance opened no send start"

                gateway.gates[0].set()  # A reaches its provider; B has not.
                await until(lambda: gateway.stream_calls == 1, timeout=30)
                assert _in_progress(console), (
                    "another session's provider request ended this send's start"
                )
                assert console._library_activity.live_reads() is True

                gateway.gates[1].set()  # B's own provider request ends it.
                await until(lambda: gateway.stream_calls == 2, timeout=30)
                assert not _in_progress(console), "B's start outlived its request"
                assert console._library_activity.live_reads() is False
            finally:
                gateway.release_all()
                await _settle_custody(console)


@pytest.mark.asyncio
async def test_a_queued_prompts_dispatch_opens_its_send_start_at_acceptance(
    no_limit, monkeypatch
):
    """The queue's dispatch of a waiting prompt is a send's acceptance too.

    A prompt queued behind a running send is dispatched when that run ends
    (``ConsoleRuntime._submit_queued_turn``). Its start must be open from
    there until its own provider request. On the base only the composer's
    two send paths noted an acceptance, so the dispatched prompt's display
    reads went off the loop for its whole start.
    """
    from tldw_chatbook.UI.Console_Modules.prompt_queue import (
        ConsolePromptDispatchStatus,
    )

    async def no_archive_refusal(_app, _conversation_id):
        # The harness has no local conversation service, so the runtime's
        # archived-conversation check before a turn with a saved
        # conversation (this queued one) would refuse it. Not this test's
        # subject.
        return None

    monkeypatch.setattr(
        "tldw_chatbook.Chat.conversation_archive_actions.conversation_send_refusal",
        no_archive_refusal,
    )
    host, gateway = _build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, _composer = await ready_console(host, pilot, gateway)
            store = console._console_chat_store
            session_id = store.active_session_id
            controller = console._ensure_console_chat_controller()
            try:
                press(host, "enter", "\r")
                await until(lambda: len(gateway.gates) == 1, timeout=30)
                outcome = await console._prompt_queue.dispatch(
                    "a queued prompt", session_id=session_id
                )
                assert outcome.status is ConsolePromptDispatchStatus.QUEUED

                gateway.gates[0].set()  # The running send finishes...
                await until(lambda: gateway.stream_calls == 1, timeout=30)
                # ...and the queue dispatches the waiting prompt, which is
                # accepted and now waits before its provider request.
                await until(lambda: len(gateway.gates) == 2, timeout=30)
                assert (
                    controller.prompt_queue_registry.snapshot(session_id).total_count
                    == 0
                )
                assert _in_progress(console), (
                    "a queued prompt's dispatch opened no send start"
                )
                assert console._library_activity.live_reads() is True

                gateway.gates[1].set()
                await until(lambda: gateway.stream_calls == 2, timeout=30)
                assert not _in_progress(console)
            finally:
                gateway.release_all()
                await _settle_custody(console)


class _StandInAuthorization(SimpleNamespace):
    """Stands in for a coordinator-issued authorization.

    Acceptance is synchronous inside ``ConsoleRuntime.accept_turn`` and checks
    no authorization; the controller refuses the stand-in afterwards, and the
    test cancels what is left.
    """


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", ["agent_wake", "goal_iteration"])
async def test_an_automatic_submissions_acceptance_opens_its_send_start(
    no_limit, origin
):
    """An agent wake or a goal iteration opens a start when it is accepted.

    Both reach the runtime through its own submitters
    (``_submit_fleet_wake``, ``_submit_goal_iteration``), never through the
    composer, so on the base neither noted an acceptance.
    """
    host, gateway = _build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, _composer = await ready_console(host, pilot, gateway)
            runtime = console._console_runtime()
            session_id = console._console_chat_store.active_session_id
            goal = None
            try:
                assert not _in_progress(console)
                if origin == "agent_wake":
                    runtime._submit_fleet_wake(
                        "wake notice",
                        session_id=session_id,
                        wake_authorization=_StandInAuthorization(
                            acceptance_started=False, accepted=False
                        ),
                        on_terminal=lambda _accepted: None,
                    )
                else:
                    goal = asyncio.ensure_future(
                        runtime._submit_goal_iteration(
                            "goal handoff",
                            session_id=session_id,
                            goal_authorization=_StandInAuthorization(),
                        )
                    )
                    await until(
                        lambda: runtime.has_custodied_turns(session_id), timeout=30
                    )
                assert _in_progress(console), (
                    f"an accepted {origin} submission opened no send start"
                )
            finally:
                gateway.release_all()
                await _settle_custody(console)
                if goal is not None:
                    goal.cancel()
                    await asyncio.gather(goal, return_exceptions=True)


@pytest.mark.asyncio
async def test_a_received_sends_start_ends_at_its_own_real_provider_request(
    no_limit, monkeypatch
):
    """The stock send (received intent) opens its start at acceptance, and the
    real gateway's adapter entry, on the worker that calls the provider, ends it.

    Measured live 2026-10-23 (TASK-33620.15.4): a first version marked only
    the dispatched path's draft commit, so the stock send never opened one.
    This drives the real gateway over a file-backed store; only the provider
    call is replaced, and it reads the window while the send is inside that
    call (its task still running, so only its provider request can have
    ended the start).
    """
    from Tests.UI.test_console_received_intent_feedback import (
        _held_received_record,
        _received_console_case,
        _send,
    )

    async with _received_console_case(monkeypatch, "send-start", durable=True) as case:
        seen_in_call: list[bool] = []

        def provider(**kwargs):
            seen_in_call.append(_in_progress(case.console))
            case.provider_calls.append(kwargs)
            return "received intent reply"

        monkeypatch.setattr("tldw_chatbook.Chat.Chat_Functions.chat_api_call", provider)
        assert not _in_progress(case.console)
        _send(case, "enter")
        await _held_received_record(case)  # Received; no provider request yet.
        assert _in_progress(case.console), "a received send opened no send start"
        assert case.console._library_activity.live_reads() is True
        case.probe.release.set()
        await until(lambda: bool(case.provider_calls), timeout=30)
        assert seen_in_call == [False], (
            "the send's own provider request did not end its start"
        )


def test_a_send_start_ends_on_its_own_after_its_limit():
    """A send that never reaches its provider cannot hold reads live for long."""
    from tldw_chatbook.Chat.console_send_start import submission_send_start

    class _Store:
        pass

    screen = SimpleNamespace(_console_chat_store=_Store())
    assert not _in_progress(screen)
    with submission_send_start(screen._console_chat_store) as start:
        assert _in_progress(screen)
        start.accepted_at -= config_sync.SEND_START_LIMIT_SECONDS + 0.01
        assert not _in_progress(screen)
    assert not _in_progress(screen)


@pytest.mark.asyncio
@pytest.mark.requires_cleanup
async def test_an_agents_chat_start_opens_a_send_start_until_it_is_done(tmp_path):
    """An agent's chat start is an automatic submission too.

    It submits its opening prompt straight to the controller
    (``ConsoleChatStartCoordinator`` -> ``submit_draft`` with origin
    ``AGENT_CHAT_START``), never through the runtime's acceptance, so on the
    base it opened no start. Screen-free: the window is read as the Console
    reads it (``send_start_in_progress`` over the screen's store) during the
    start's durable commit, after its acceptance and before its provider
    request, and again once the start has finished.
    """
    import threading

    from Tests.Chat.test_console_chat_start import _create_handoff, _restore_handoff
    from Tests.Chat.test_console_agent_swap import _controller
    from tldw_chatbook.Chat.console_chat_start import (
        AgentChatStartRequest,
        ConsoleChatStartCoordinator,
    )
    from tldw_chatbook.Chat.console_library_policy import (
        ConsoleLibraryPolicyCandidate,
    )

    controller, store, runs = _controller(tmp_path, [["target answer"]])
    screen = SimpleNamespace(_console_chat_store=store)  # What the Console reads.
    source = store.create_session(workspace_id="global", activate=True)
    source.persisted_conversation_id = store.persistence.create_conversation(
        conversation_title="source"
    )
    chain = runs.automatic_work.create_chain(
        source.persisted_conversation_id, root_submission_id="human-root"
    )
    source_run = runs.create_run(
        conversation_id=source.persisted_conversation_id,
        agent_kind="primary",
        work_chain_id=chain,
    )
    controller._agent_bridge._live_primary_runs[source.persisted_conversation_id] = (
        source_run
    )
    controller._active_cancel_events[source.id] = threading.Event()  # A live source.
    controller._active_assistant_message_ids[source.id] = "source-message"
    conversation = _create_handoff(store, draft="hello", source_run_id=source_run)
    defaults = store._library_policy_defaults
    store.persistence.console_library_policy_repository.insert(
        conversation,
        ConsoleLibraryPolicyCandidate(
            defaults.auto_retrieve, defaults.assistant_access
        ),
    )
    target = _restore_handoff(store, conversation)
    coordinator = ConsoleChatStartCoordinator(controller)
    controller._chat_start = coordinator
    await store.library_policy_coordinator.capture_for_execution(target.id)
    request = AgentChatStartRequest(
        "send-start",
        source_run,
        source.id,
        source.incarnation_id,
        conversation,
        target.id,
        target.incarnation_id,
        1,
        store.conversation_context_epoch(target.id),
        "hello",
        controller.resolve_turn_configuration_snapshot(target.id),
        target.workspace_id,
    )
    during_commit: list[bool] = []
    original_commit = store.commit_durable_turn

    def recording_commit(acceptance):
        during_commit.append(config_sync.send_start_in_progress(screen))
        return original_commit(acceptance)

    store.commit_durable_turn = recording_commit
    # The scripted gateway makes no provider request of its own, so its stream
    # notes one where the real gateway does (adapter entry, in the send's own
    # work: here on the agent loop's thread), reading the window either side.
    from tldw_chatbook.Chat.console_send_diagnostics import record_send_stage

    gateway = controller.provider_gateway
    original_stream = gateway.stream_chat
    around_entry: list[tuple[bool, bool]] = []

    async def stream_noting_entry(resolution, messages, **kwargs):
        before = config_sync.send_start_in_progress(screen)
        record_send_stage("provider_entry")
        around_entry.append((before, config_sync.send_start_in_progress(screen)))
        async for chunk in original_stream(resolution, messages, **kwargs):
            yield chunk

    gateway.stream_chat = stream_noting_entry
    try:
        assert not config_sync.send_start_in_progress(screen)
        outcome = await coordinator.start(request)
        assert outcome.launch_status == "started", outcome
        await asyncio.gather(*coordinator.tasks())
        assert during_commit == [True], (
            "the chat start's submission had no send start open after its "
            f"acceptance (read during its durable commit: {during_commit})"
        )
        assert around_entry == [(True, False)], (
            "the chat start's own provider request did not end its send start "
            f"(open before and after that request: {around_entry})"
        )
        assert not config_sync.send_start_in_progress(screen), (
            "the chat start's send start outlived its submission"
        )
    finally:
        await controller.shutdown()
        runs.close()
        store.persistence.db.close_connection()
