"""Real Console navigation with accepted provider work and retained decisions."""

from __future__ import annotations

import asyncio
import threading

import pytest

from Tests.UI.test_console_screen_reuse import (
    _boot_settled,
    _press_until_screen,
    _scratch_env,
)


async def _until(
    predicate, *, seconds=15.0, detail=lambda: "Console transition stalled"
):
    deadline = asyncio.get_running_loop().time() + seconds
    while not predicate():
        assert asyncio.get_running_loop().time() < deadline, detail()
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
@pytest.mark.ui
@pytest.mark.parametrize("kind", ("approval", "skill_install", "skill_script"))
async def test_hidden_finite_decisions_notify_once_and_remain_answerable(
    monkeypatch, tmp_path, kind
):
    """A retained screen cannot spend the user's decision time or hide its notice."""
    _scratch_env(monkeypatch, tmp_path)
    from textual.screen import ModalScreen

    from Tests.Chat.test_console_runtime_lifetime import _pending_call
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen

    app = TldwCli()
    notices = []
    app.notify = lambda message, **kwargs: notices.append(str(message))
    async with app.run_test(size=(170, 48)) as pilot:
        await _boot_settled(app, pilot)
        await _press_until_screen(pilot, "ctrl+2", "ChatScreen")
        console = app.screen
        controller = console._ensure_console_chat_controller()
        session_id = controller.store.ensure_session().id
        controller.mcp_approval_timeout_seconds = lambda: 1.0
        controller.skill_install_confirm_timeout_seconds = lambda: 1.0
        controller.skill_script_confirm_timeout_seconds = lambda: 1.0
        controller._interrupt_host.POLL_SECONDS = 0.01

        def request():
            if kind == "approval":
                return controller.request_mcp_approvals(
                    [_pending_call()], session_id=session_id
                )
            if kind == "skill_install":
                return controller.request_skill_install_confirm(
                    "https://example.com/test-skill", session_id=session_id
                )
            return controller.request_skill_script_confirm(
                {"skill_name": "demo", "script_path": "scripts/demo.py"},
                session_id=session_id,
            )

        for away in ("navigation", "modal", "armed_navigation"):
            if away == "navigation":
                await _press_until_screen(pilot, "ctrl+1", "HomeScreen")
            elif away == "modal":
                await app.push_screen(ModalScreen())
            notices.clear()
            decision = asyncio.create_task(asyncio.to_thread(request))
            try:
                await _until(lambda: controller._interrupt_host.pending_total() == 1)
                if away == "armed_navigation":
                    await _until(
                        lambda: (
                            controller._interrupt_host.head_round_payload(
                                kind, session_id
                            )
                            is not None
                        )
                    )
                    app.post_message(NavigateToScreen("home"))
                    await _until(lambda: type(app.screen).__name__ == "HomeScreen")
                await asyncio.sleep(1.2)
                assert not decision.done(), (
                    "A hidden decision consumed its finite budget"
                )
                attention = [
                    message for message in notices if "Open Console" in message
                ]
                assert len(attention) == 1
                # Repeated visibility notifications must not repeat a round's notice.
                console.on_screen_suspend()
                assert (
                    len([message for message in notices if "Open Console" in message])
                    == 1
                )
                if away != "modal":
                    app.post_message(NavigateToScreen("chat"))
                    await _until(lambda: app.screen is console)
                else:
                    await app.pop_screen()
                payload = controller._interrupt_host.head_round_payload(
                    kind, session_id
                )
                assert payload is not None
                if kind == "approval":
                    controller.resolve_pending_approval(
                        [], round_id=payload["round_id"]
                    )
                elif kind == "skill_install":
                    controller.resolve_pending_skill_install(
                        True, request_id=payload["request_id"]
                    )
                else:
                    controller.resolve_pending_skill_script(
                        True, False, request_id=payload["request_id"]
                    )
                result = await asyncio.wait_for(decision, 3.0)
                if kind == "skill_install":
                    assert result is True
                elif kind == "skill_script":
                    assert result == {"allow": True, "remember": False}
                assert controller._interrupt_host.pending_total() == 0
            finally:
                if not decision.done():
                    controller.begin_shutdown()
                    await asyncio.wait_for(decision, 3.0)


@pytest.mark.asyncio
@pytest.mark.ui
@pytest.mark.parametrize("settlement", ("complete", "stop", "shutdown"))
async def test_navigation_preserves_real_stream_and_queue_without_retargeting(
    monkeypatch, tmp_path, settlement
):
    """Screen-worker cancellation or ambient session routing loses a real response."""
    _scratch_env(monkeypatch, tmp_path)
    from textual.widgets import Button

    from Tests.UI.test_console_native_chat_flow import (
        _configure_native_ready_console,
        _ReadyResolutionGateway,
        _select_llamacpp_console,
    )
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Chat.console_chat_models import ConsoleRunStatus
    from tldw_chatbook.Widgets.Console import ConsoleComposerBar

    class Gateway(_ReadyResolutionGateway):
        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()

        async def stream_chat(self, resolution, messages, **kwargs):
            yield "partial"
            self.started.set()
            while not self.release.is_set():
                await asyncio.sleep(0.01)
            yield " done"

        async def aclose(self):
            pass

    gateway = Gateway()
    app = TldwCli()
    _configure_native_ready_console(app)
    app.console_provider_gateway_factory = lambda: gateway
    async with app.run_test(size=(170, 48)) as pilot:
        await _boot_settled(app, pilot)
        await _press_until_screen(pilot, "ctrl+2", "ChatScreen")
        console = app.screen
        _select_llamacpp_console(console)
        controller = console._ensure_console_chat_controller()
        session_id = controller.store.active_session_id
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("First navigation message")
        console.query_one("#console-send-message", Button).press()
        try:
            await _until(gateway.started.is_set)
            queued = await console._prompt_queue.dispatch("Second navigation message")
            assert queued.status.value == "queued"
            await _press_until_screen(pilot, "ctrl+1", "HomeScreen")
            assert controller.in_flight_run_count() == 1
            if settlement != "complete":
                if settlement == "stop":
                    assert controller.stop_active_run() is True
                else:
                    await controller.shutdown()
                await _until(lambda: controller.in_flight_run_count() == 0)
                messages = controller.store.messages_for_session(session_id)
                assert messages[1].status == "stopped"
                assert not any(
                    m.content == "Second navigation message" for m in messages
                )
                return
            other_session = controller.new_session().id
            gateway.release.set()
            await _until(
                lambda: (
                    sum(
                        message.content == "partial done"
                        for message in controller.store.messages_for_session(session_id)
                    )
                    == 2
                    and controller.in_flight_run_count() == 0
                ),
                detail=lambda: (
                    controller.run_state_for(session_id),
                    controller.prompt_queue_registry.snapshot(session_id),
                    [
                        (m.role, m.content)
                        for m in controller.store.messages_for_session(session_id)
                    ],
                ),
            )
            assert type(app.screen).__name__ == "HomeScreen"
            messages = controller.store.messages_for_session(session_id)
            assert [message.content for message in messages] == [
                "First navigation message",
                "partial done",
                "Second navigation message",
                "partial done",
            ]
            assert not controller.store.messages_for_session(other_session)
            await _press_until_screen(pilot, "ctrl+2", "ChatScreen")
            assert app.screen is console
            assert controller.store.active_session_id == other_session
            assert (
                controller.run_state_for(session_id).status
                is ConsoleRunStatus.COMPLETED
            )
        finally:
            gateway.release.set()
            await controller.shutdown()
