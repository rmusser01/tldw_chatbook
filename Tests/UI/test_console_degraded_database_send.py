"""TASK-22030: a refused send must be visible, never silent.

If the ChaChaNotes database cannot be opened, `TldwCli.__init__` leaves
``chachanotes_db = None`` and ``ConsoleRuntime.ensure_chat_store`` builds the
Console store with ``persistence=None``. Since `56db75386` a durable turn
(every non-ephemeral manual or queued send) fails closed there -- correctly --
but the refusal was returned as a bare ``ConsoleSubmitResult``: no run state,
no transcript row, no toast. Pressing Send did *nothing at all*.

These tests drive the mounted Console against exactly that state and assert
the surfaces a user can actually see, not the return value.
"""

from __future__ import annotations

import asyncio

import pytest
from textual.widgets import Button

from Tests.UI.app_factory import attach_chachanotes_db
from Tests.UI.test_console_native_chat_flow import (
    WaitingGateway,
    _ASYNC_SETTLE_TIMEOUT,
    _select_llamacpp_console,
    _wait_for_text,
)
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleRunStatus,
)
from tldw_chatbook.Widgets.Console import ConsoleComposerBar


def _degraded_app(notifications: list[tuple[str, dict]]):
    """Build a Console app whose ChaChaNotes database could not be opened."""

    app = _build_test_app()
    # `_build_test_app` already patches `get_chachanotes_db_lazy` to None, so
    # this is the production post-failure shape verbatim -- no double.
    assert getattr(app, "chachanotes_db", None) is None
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.notify = lambda message, **kwargs: notifications.append((message, kwargs))
    return app


@pytest.mark.asyncio
async def test_unopenable_database_refuses_send_visibly_and_keeps_the_draft():
    notifications: list[tuple[str, dict]] = []
    gateway = WaitingGateway()
    app = _degraded_app(notifications)
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")

        console.query_one("#console-send-message", Button).press()
        await pilot.pause()
        await pilot.pause(0.2)

        controller = console._ensure_console_chat_controller()
        session_id = controller.store.active_session_id
        assert session_id is not None

        # 1. A transcript row the user can read, naming the real cause.
        rows = [
            message
            for message in controller.store.messages_for_session(session_id)
            if message.role is ConsoleMessageRole.SYSTEM
        ]
        assert len(rows) == 1, [row.content for row in rows]
        copy = rows[0].content
        assert "not sent" in copy.lower()
        assert "database could not be opened" in copy.lower()
        assert "temporary chat" in copy.lower()

        # 2. A toast carrying the same explanation.
        assert notifications, "the refusal raised no toast at all"
        assert any(
            "database could not be opened" in message.lower()
            for message, _kwargs in notifications
        ), notifications

        # 3. A blocked run state, so the control bar is not left reading idle.
        assert (
            controller.run_state_for(session_id).status is ConsoleRunStatus.BLOCKED
        )

        # 4. The draft the user typed is still there.
        assert composer.draft_text() == "hello"

        # 5. Nothing reached the provider.
        assert gateway.started.is_set() is False


@pytest.mark.asyncio
async def test_unopenable_database_still_sends_a_temporary_conversation():
    """A temporary chat needs no durable commit, so it must keep working."""

    notifications: list[tuple[str, dict]] = []
    gateway = WaitingGateway()
    app = _degraded_app(notifications)
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        store = console._ensure_console_chat_store()
        settings = store.sessions()[0].settings
        temporary = store.create_session(
            title="Temp chat", settings=settings, ephemeral=True, activate=True
        )
        await pilot.pause()

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello from a temporary chat")
        console.query_one("#console-send-message", Button).press()
        await asyncio.wait_for(gateway.started.wait(), timeout=_ASYNC_SETTLE_TIMEOUT)
        await _wait_for_text(console, pilot, "partial")

        assert temporary.ephemeral is True
        roles = [
            message.role for message in store.messages_for_session(temporary.id)
        ]
        assert ConsoleMessageRole.USER in roles
        assert ConsoleMessageRole.SYSTEM not in roles
        assert not any(
            "database could not be opened" in message.lower()
            for message, _kwargs in notifications
        ), notifications

        gateway.release.set()
        await _wait_for_text(console, pilot, "partial done")


@pytest.mark.asyncio
async def test_working_database_is_unaffected_by_the_degraded_path():
    """Control arm: with a usable database the refusal never fires."""

    notifications: list[tuple[str, dict]] = []
    gateway = WaitingGateway()
    app = _degraded_app(notifications)
    attach_chachanotes_db(app)
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")

        console.query_one("#console-send-message", Button).press()
        await asyncio.wait_for(gateway.started.wait(), timeout=_ASYNC_SETTLE_TIMEOUT)
        await _wait_for_text(console, pilot, "partial")

        controller = console._ensure_console_chat_controller()
        session_id = controller.store.active_session_id
        assert session_id is not None
        assert not [
            message
            for message in controller.store.messages_for_session(session_id)
            if message.role is ConsoleMessageRole.SYSTEM
        ]
        assert not any(
            "database could not be opened" in message.lower()
            for message, _kwargs in notifications
        ), notifications
        assert composer.draft_text() == ""

        gateway.release.set()
        await _wait_for_text(console, pilot, "partial done")
