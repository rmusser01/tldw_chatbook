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
async def test_unexpected_review_composition_failure_keeps_agent_turn(
    tmp_path, monkeypatch
):
    """Review tools are optional; their composition never fails an agent turn."""
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
        "hello", session_id=session.id, configuration=configuration
    )

    assert len(gateway.dispatched) == 1
    assert store.messages_for_session(session.id)[-1].content == "plain answer"


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
