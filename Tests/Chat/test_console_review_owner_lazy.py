"""Ordinary Console lifecycle paths never construct the code-review owner.

``ConsoleChatController.code_review`` builds the coordinator (and imports the
review modules) on first access. Stop, close, shutdown and automatic-wake
bookkeeping must leave it unbuilt when no review ever started.
"""

import pytest

from Tests.Chat.test_console_close_fence_composition import rig


def test_stop_leaves_review_owner_unbuilt():
    _store, session, controller, _bridge = rig()
    controller._signal_stop(session_id=session.id)
    assert controller._code_review is None


def test_session_close_leaves_review_owner_unbuilt():
    _store, session, controller, _bridge = rig()
    revision = controller.lifecycle_impact(session_id=session.id).revision
    controller.begin_session_close(session.id, expected_revision=revision)
    assert controller._code_review is None


@pytest.mark.asyncio
async def test_wake_pause_and_shutdown_leave_review_owner_unbuilt():
    _store, _session, controller, _bridge = rig()
    await controller.fleet_wake._pause("saved-chat", None, "automatic_wake_disabled")
    assert controller._code_review is None
    await controller.shutdown()
    assert controller._code_review is None


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
@pytest.mark.parametrize(
    ("draft", "gate_on", "explicit"),
    [
        ("hello", True, False),
        ("Review my changes", True, True),
        # Composition's own admission: no bound root needs code wording, and
        # a verify without a held review is ordinary chat.
        ("Review my resume", True, False),
        ("verify", True, False),
        ("Review my changes", False, False),
    ],
    ids=["ordinary", "explicit-review", "resume", "bare-verify", "gate-off"],
)
async def test_unexpected_review_composition_failure_keeps_agent_turn(
    tmp_path, monkeypatch, draft, gate_on, explicit
):
    """Review tools are optional; their composition never fails an agent turn.

    A turn composition would admit as a review request is told why it got no
    review (a safe reason, never the exception text); every other turn stays
    silent.
    """
    from Tests.console_provider_doubles import provider_resolution
    from tldw_chatbook.Chat import console_code_review
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
    from tldw_chatbook.Chat.console_turn_context import (
        ConsoleTurnConfigurationSnapshot,
    )
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    class Gateway(ConsoleProviderGateway):
        def __init__(self):
            super().__init__(config_provider=dict, environ={})
            self.dispatched = []

        async def resolve_for_send(self, _selection):
            return provider_resolution(provider="llama_cpp", model="local")

        async def stream_chat(self, _resolution, messages, **_kwargs):
            self.dispatched.append(messages)
            yield "plain answer"

    def broken(*_args, **_kwargs):
        raise RuntimeError("unexpected composition failure")

    monkeypatch.setattr(console_code_review, "compose_native_review_provider", broken)
    monkeypatch.setattr(console_code_review, "_review_tools_enabled", lambda: gate_on)
    gateway = Gateway()
    store = ConsoleChatStore()
    session = store.create_session(ephemeral=True)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_bridge=ConsoleAgentBridge(
            agent_runs_db=AgentRunsDB(tmp_path / "runs.db", client_id="t"),
            store=store,
            provider_gateway=gateway,
        ),
        agent_runtime_enabled=True,
        provider="llama_cpp",
        model="local",
    )
    base = controller.resolve_runtime_turn_configuration_snapshot(session.id)
    configuration = ConsoleTurnConfigurationSnapshot.capture(
        session_id=session.id,
        provider_selection=base.provider_selection,
        scratch_space=base.scratch_space,
        session_settings=base.session_settings,
        project_authority=base.project_authority,
        tool_configuration={**base.tool_configuration, "agent_runtime_enabled": True},
    )

    await controller.submit_draft(
        draft, session_id=session.id, configuration=configuration
    )

    assert len(gateway.dispatched) == 1
    contents = [message.content for message in store.messages_for_session(session.id)]
    assert "plain answer" in contents
    notice = (
        "Review needs attention: review_tools_unavailable. "
        "No review stage was admitted."
    )
    attention = [text for text in contents if text.startswith("Review needs")]
    assert attention == ([notice] if explicit else [])
    assert not any("unexpected composition failure" in text for text in contents)


def test_review_offer_on_its_owner_loop_presents_without_deadlock():
    """The offer presenter must not block its own loop on a coroutine it owns."""
    import asyncio
    import threading

    store, session, controller, _bridge = rig()
    loop = asyncio.new_event_loop()
    controller._fleet_wake._loop = loop
    results = []

    async def offer_on_loop():
        results.append(controller._present_review_offer(session.id, "review-1"))

    worker = threading.Thread(
        target=lambda: loop.run_until_complete(offer_on_loop()), daemon=True
    )
    worker.start()
    worker.join(5)

    assert results == [True]
    loop.close()
    assert "Verification is available for review review-1" in (
        store.messages_for_session(session.id)[-1].content
    )


@pytest.mark.asyncio
async def test_view_attach_installs_presenter_and_presents_pending_originals():
    """Only a fresh view claim re-presents pending originals, one pass at a time.

    ``attach_app`` runs on every ensure call; a refresh of the same claim must
    schedule nothing, a claim made mid-pass is owed one pass after it, and a
    failed pass is retrieved and logged.
    """
    import asyncio
    import threading
    from types import SimpleNamespace

    from loguru import logger

    from tldw_chatbook.UI.Console_Modules import wiring

    calls, release, fail = [], threading.Event(), []

    def present_pending(session_id):
        calls.append(session_id)
        release.wait(5)
        if fail:
            raise RuntimeError("PRIVATE failure text")

    runtime = SimpleNamespace(view=object(), _attached_generation=1)
    app = SimpleNamespace(console_runtime=runtime)
    coordinator = SimpleNamespace(
        _reviews={"review-1": object()}, present_pending=present_pending
    )
    controller = SimpleNamespace(_code_review=coordinator)

    wiring.attach_review_presenter(controller, app)
    assert controller.app is app
    assert controller.review_report_presenter is (
        wiring.present_code_review_original
    )
    first = controller._review_present_task
    runtime._attached_generation = 2  # a fresh claim while the first pass runs
    wiring.attach_review_presenter(controller, app)
    assert controller._review_present_task is first  # at most one in flight
    release.set()
    await first
    owed = controller._review_present_task
    assert owed is not first  # the mid-pass claim is owed one pass after it
    await owed
    assert calls == [None, None]

    for _ in range(3):  # refreshes of the same claim schedule nothing
        wiring.attach_review_presenter(controller, app)
        await asyncio.sleep(0)
    assert calls == [None, None]

    runtime._attached_generation = 3
    fail.append(True)
    logged = []
    sink = logger.add(lambda record: logged.append(str(record)), level="WARNING")
    try:
        wiring.attach_review_presenter(controller, app)
        task = controller._review_present_task
        await asyncio.wait({task})
        await asyncio.sleep(0)
    finally:
        logger.remove(sink)
    assert calls == [None, None, None]
    assert any("RuntimeError" in line for line in logged)
    assert not any("PRIVATE" in line for line in logged)

    # No loaded review owner: nothing is built, nothing is scheduled.
    bare = SimpleNamespace(_code_review=None)
    wiring.attach_review_presenter(bare, app)
    assert bare.review_report_presenter is wiring.present_code_review_original
    assert getattr(bare, "_review_present_task", None) is None


def test_activation_with_no_owner_loop_is_left_unhandled():
    """An off-loop activation with no running owner loop records no switch.

    The next real switch or fresh claim must still present the chat's pending
    originals, so the switch is not marked handled.
    """
    from types import SimpleNamespace

    from tldw_chatbook.UI.Console_Modules import wiring

    calls = []
    controller = SimpleNamespace(
        store=SimpleNamespace(active_session_id="chat"),
        _fleet_wake=SimpleNamespace(_loop=None),
        _code_review=SimpleNamespace(
            _reviews={"review-1": object()}, present_pending=calls.append
        ),
    )

    wiring.review_session_activated(controller)

    assert getattr(controller, "_review_active_session", None) is None
    assert getattr(controller, "_review_present_task", None) is None
    assert calls == []
