"""Console live-work launch and staged-context handoff boundary tests."""

from unittest.mock import Mock

import pytest
from textual.app import App

from Tests.UI.test_destination_shells import DestinationHarness
from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


class ConsoleHarness(App):
    def __init__(self, app_instance):
        super().__init__()
        self.app_instance = app_instance

    async def on_mount(self) -> None:
        await self.push_screen(ChatScreen(self.app_instance))


def _active_console_screen(host: ConsoleHarness):
    return host.screen_stack[-1]


def test_app_exposes_open_console_for_live_work_helper():
    app = _build_test_app()

    assert hasattr(app, "open_console_for_live_work")


def test_open_console_for_live_work_routes_to_chat_route():
    app = _build_test_app()
    seen = []
    app.post_message = lambda message: seen.append(getattr(message, "screen_name", None))

    app.open_console_for_live_work(source="workflows", title="Daily digest")

    assert seen == ["chat"]
    assert app.pending_console_launch == {
        "source": "workflows",
        "title": "Daily digest",
        "payload": {},
    }


@pytest.mark.parametrize(
    ("route", "button_id"),
    [
        ("watchlists_collections", "watchlists-follow-in-console"),
        ("schedules", "schedules-follow-in-console"),
        ("workflows", "workflows-launch-in-console"),
        ("acp", "acp-follow-in-console"),
    ],
)
@pytest.mark.asyncio
async def test_destination_live_work_actions_open_console(route, button_id):
    app = _build_test_app()
    app.open_console_for_live_work = Mock()
    host = DestinationHarness(app, route)

    async with host.run_test(size=(180, 40)) as pilot:
        await pilot.pause(0.1)
        await pilot.click(f"#{button_id}")
        await pilot.pause(0.1)

    app.open_console_for_live_work.assert_called_once()
    assert app.open_console_for_live_work.call_args.kwargs["source"] == route


@pytest.mark.parametrize(
    ("route", "button_id"),
    [
        ("library", "library-use-in-console"),
        ("artifacts", "artifacts-use-in-console"),
        ("personas", "personas-attach-to-console"),
        ("skills", "skills-attach-to-console"),
    ],
)
@pytest.mark.asyncio
async def test_staged_context_actions_use_chat_handoff_not_live_launch(route, button_id):
    app = _build_test_app()
    app.open_chat_with_handoff = Mock()
    app.open_console_for_live_work = Mock()
    host = DestinationHarness(app, route)

    async with host.run_test(size=(180, 40)) as pilot:
        await pilot.pause(0.1)
        await pilot.click(f"#{button_id}")
        await pilot.pause(0.1)

    app.open_chat_with_handoff.assert_called_once()
    payload = app.open_chat_with_handoff.call_args.args[0]
    assert isinstance(payload, ChatHandoffPayload)
    assert payload.source == route
    app.open_console_for_live_work.assert_not_called()
    assert getattr(app, "pending_console_launch", None) in (None, {})


@pytest.mark.asyncio
async def test_console_renders_pending_launch_context():
    app = _build_test_app()
    app.pending_console_launch = {
        "source": "workflows",
        "title": "Daily digest",
        "payload": {},
    }
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 40)) as pilot:
        await pilot.pause(0.1)
        screen = _active_console_screen(host)

        assert screen.query_one("#console-pending-launch-card")
