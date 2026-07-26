from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pytest

import tldw_chatbook.app as app_module
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Navigation.pending_handoff_store import (
    HandoffChannel,
    PendingHandoffStore,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


def _chat_payload(title: str = "handoff") -> ChatHandoffPayload:
    return ChatHandoffPayload(
        source="tests",
        item_type="document",
        title=title,
        body="context",
        metadata={"nested": {"items": ["original"]}},
    )


def _configure_startup(
    app: TldwCli,
    monkeypatch: pytest.MonkeyPatch,
    route: str = "home",
) -> None:
    app.app_config["_first_run"] = False
    app._initial_tab_value = route
    real_get_cli_setting = app_module.get_cli_setting

    def get_cli_setting_without_splash(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return real_get_cli_setting(section, key, default)

    monkeypatch.setattr(
        app_module,
        "get_cli_setting",
        get_cli_setting_without_splash,
    )


@asynccontextmanager
async def _mounted_app(
    app: TldwCli,
    monkeypatch: pytest.MonkeyPatch,
    route: str = "home",
):
    _configure_startup(app, monkeypatch, route)
    _screen_name, canonical_route, screen_class = app._resolve_screen_navigation_target(
        route
    )
    assert screen_class is not None

    async with app.run_test(size=(170, 48)) as pilot:
        for _ in range(150):
            if getattr(app, "_initial_screen_pushed", False) and isinstance(
                app.screen,
                screen_class,
            ):
                assert app.current_tab == canonical_route
                yield pilot
                return
            await pilot.pause(0.01)
        raise AssertionError("full app did not mount its configured production screen")


async def _wait_for_chat_screen(app: TldwCli, pilot) -> ChatScreen:
    for _ in range(150):
        if isinstance(app.screen, ChatScreen):
            return app.screen
        await pilot.pause(0.01)
    raise AssertionError("full app did not navigate to the production Chat screen")


@pytest.mark.asyncio
async def test_full_app_constructs_pending_handoff_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch):
        assert isinstance(app.pending_handoffs, PendingHandoffStore)


@pytest.mark.asyncio
async def test_full_app_chat_producer_stages_before_navigation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()
    payload = _chat_payload()

    async with _mounted_app(app, monkeypatch):
        messages: list[NavigateToScreen] = []

        def observe_navigation(message: Any) -> None:
            assert isinstance(message, NavigateToScreen)
            claim = app.pending_handoffs.claim(HandoffChannel.CHAT)
            assert claim is not None
            assert claim.value.title == "handoff"
            messages.append(message)
            assert app.pending_handoffs.release(claim) is True

        monkeypatch.setattr(app, "post_message", observe_navigation)
        app.open_chat_with_handoff(payload)
        payload.metadata["nested"]["items"].append("producer-change")

        retry = app.pending_handoffs.claim(HandoffChannel.CHAT)
        assert retry is not None
        assert retry.value.metadata["nested"]["items"] == ["original"]
        assert [message.screen_name for message in messages] == ["chat"]


@pytest.mark.asyncio
async def test_full_app_chat_tabs_gate_prevents_stage_and_navigation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch):
        messages: list[Any] = []
        notifications: list[tuple[str, str]] = []
        current_get_cli_setting = app_module.get_cli_setting

        def tabs_disabled(section, key=None, default=None):
            if section == "chat_defaults" and key == "enable_tabs":
                return False
            return current_get_cli_setting(section, key, default)

        monkeypatch.setattr(app_module, "get_cli_setting", tabs_disabled)
        monkeypatch.setattr(app, "post_message", messages.append)
        monkeypatch.setattr(
            app,
            "notify",
            lambda message, *, severity="information", **_kwargs: notifications.append(
                (message, severity)
            ),
        )

        app.open_chat_with_handoff(_chat_payload())

        assert app.pending_handoffs.claim(HandoffChannel.CHAT) is None
        assert messages == []
        assert notifications == [
            ("Use in Chat requires chat tabs to be enabled.", "warning")
        ]


@pytest.mark.asyncio
async def test_full_app_console_producers_stage_before_navigation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch):
        observed: list[tuple[HandoffChannel, Any]] = []
        expected_channels = iter(
            (
                HandoffChannel.CONSOLE_PROMPT_INSERT,
                HandoffChannel.CONSOLE_LIVE_WORK,
            )
        )

        def observe_navigation(message: Any) -> None:
            assert isinstance(message, NavigateToScreen)
            channel = next(expected_channels)
            claim = app.pending_handoffs.claim(channel)
            assert claim is not None
            observed.append((channel, claim.value))
            assert app.pending_handoffs.acknowledge(claim) is True

        monkeypatch.setattr(app, "post_message", observe_navigation)

        app.stage_console_prompt_insert("  exact prompt\n")
        app.open_console_for_live_work(
            source="tests",
            title="live work",
            payload={"nested": {"items": ["original"]}},
        )

        assert observed[0] == (
            HandoffChannel.CONSOLE_PROMPT_INSERT,
            "  exact prompt\n",
        )
        assert observed[1][0] is HandoffChannel.CONSOLE_LIVE_WORK
        assert observed[1][1].payload == {"nested": {"items": ["original"]}}


@pytest.mark.asyncio
async def test_full_app_producer_normalization_failure_does_not_navigate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CopyFailure:
        def __deepcopy__(self, _memo):
            raise TypeError("private-copy-failure")

    app = TldwCli()

    async with _mounted_app(app, monkeypatch):
        messages: list[Any] = []
        notifications: list[tuple[str, str]] = []
        monkeypatch.setattr(app, "post_message", messages.append)
        monkeypatch.setattr(
            app,
            "notify",
            lambda message, *, severity="information", **_kwargs: notifications.append(
                (message, severity)
            ),
        )

        app.stage_console_prompt_insert("   ")
        app.open_console_for_live_work(
            source="tests",
            title="copy failure",
            payload={"nested": CopyFailure()},
        )
        app.open_chat_with_handoff(object())  # type: ignore[arg-type]

        assert messages == []
        assert len(notifications) == 3
        assert all(severity == "warning" for _message, severity in notifications)
        assert app.pending_handoffs.claim(HandoffChannel.CONSOLE_PROMPT_INSERT) is None
        assert app.pending_handoffs.claim(HandoffChannel.CONSOLE_LIVE_WORK) is None
        assert app.pending_handoffs.claim(HandoffChannel.CHAT) is None


@pytest.mark.asyncio
async def test_full_app_producer_replaces_latest_unclaimed_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch):
        monkeypatch.setattr(app, "post_message", lambda _message: None)

        app.open_console_for_live_work(source="tests", title="first")
        app.open_console_for_live_work(source="tests", title="second")

        claim = app.pending_handoffs.claim(HandoffChannel.CONSOLE_LIVE_WORK)
        assert claim is not None
        assert claim.value.title == "second"


@pytest.mark.asyncio
async def test_full_app_valid_chat_producer_navigates_to_production_screen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch) as pilot:
        app.open_chat_with_handoff(_chat_payload())

        screen = await _wait_for_chat_screen(app, pilot)
        assert screen is app.screen
        assert app.current_tab == "chat"
