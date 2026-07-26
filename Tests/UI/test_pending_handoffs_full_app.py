from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from loguru import logger
import pytest

import tldw_chatbook.app as app_module
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Navigation.pending_handoff_store import (
    HandoffChannel,
    PendingHandoffStore,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar


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


async def _mounted_chat(app: TldwCli, pilot) -> tuple[ChatScreen, ConsoleComposerBar]:
    screen = await _wait_for_chat_screen(app, pilot)
    for _ in range(150):
        composers = list(screen.query("#console-native-composer"))
        if composers and isinstance(composers[0], ConsoleComposerBar):
            return screen, composers[0]
        await pilot.pause(0.01)
    raise AssertionError("production Chat screen did not mount its Console composer")


def _intercept_navigation(
    app: TldwCli,
    monkeypatch: pytest.MonkeyPatch,
    callback,
) -> None:
    real_post_message = app.post_message

    def post_message(message: Any):
        if isinstance(message, NavigateToScreen):
            callback(message)
            return True
        return real_post_message(message)

    monkeypatch.setattr(app, "post_message", post_message)


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

        _intercept_navigation(app, monkeypatch, observe_navigation)
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
        _intercept_navigation(app, monkeypatch, messages.append)
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

        _intercept_navigation(app, monkeypatch, observe_navigation)

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
        _intercept_navigation(app, monkeypatch, messages.append)
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
        _intercept_navigation(app, monkeypatch, lambda _message: None)

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


@pytest.mark.asyncio
async def test_full_app_console_launch_consumer_transfers_and_acknowledges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch, route="chat") as pilot:
        screen, _composer = await _mounted_chat(app, pilot)
        app.pending_handoffs.stage(
            HandoffChannel.CONSOLE_LIVE_WORK,
            ConsoleLiveWorkLaunch.from_values(
                source="tests",
                title="transferred",
                payload={"nested": {"items": ["original"]}},
            ),
        )

        transferred = screen._consume_pending_console_launch()

        assert transferred is screen._pending_console_launch_context
        assert transferred is not None
        assert transferred.title == "transferred"
        assert screen._pending_console_launch_auto_open_inspector is True
        assert app.pending_handoffs.claim(HandoffChannel.CONSOLE_LIVE_WORK) is None


@pytest.mark.asyncio
async def test_full_app_console_launch_replacement_survives_older_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch, route="chat") as pilot:
        screen, _composer = await _mounted_chat(app, pilot)
        app.pending_handoffs.stage(
            HandoffChannel.CONSOLE_LIVE_WORK,
            ConsoleLiveWorkLaunch.from_values(source="tests", title="first"),
        )
        real_acknowledge = app.pending_handoffs.acknowledge

        def acknowledge_after_replacement(claim) -> bool:
            app.pending_handoffs.stage(
                HandoffChannel.CONSOLE_LIVE_WORK,
                ConsoleLiveWorkLaunch.from_values(source="tests", title="second"),
            )
            return real_acknowledge(claim)

        monkeypatch.setattr(
            app.pending_handoffs,
            "acknowledge",
            acknowledge_after_replacement,
        )

        transferred = screen._consume_pending_console_launch()
        replacement = app.pending_handoffs.claim(HandoffChannel.CONSOLE_LIVE_WORK)

        assert transferred is not None
        assert transferred.title == "first"
        assert replacement is not None
        assert replacement.value.title == "second"


@pytest.mark.asyncio
async def test_full_app_console_prompt_consumer_appends_and_acknowledges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch, route="chat") as pilot:
        screen, composer = await _mounted_chat(app, pilot)
        monkeypatch.setattr(screen, "_console_setup_blocked_reason", lambda: "")
        composer.load_draft("existing")
        app.pending_handoffs.stage(
            HandoffChannel.CONSOLE_PROMPT_INSERT,
            "inserted",
        )

        await screen._consume_pending_console_prompt_insert()

        assert composer.draft_text() == "existing\ninserted"
        assert app.pending_handoffs.claim(HandoffChannel.CONSOLE_PROMPT_INSERT) is None


@pytest.mark.asyncio
async def test_full_app_console_prompt_setup_block_releases_for_later_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch, route="chat") as pilot:
        screen, composer = await _mounted_chat(app, pilot)
        notifications: list[tuple[str, str]] = []
        monkeypatch.setattr(
            app,
            "notify",
            lambda message, *, severity="information", **_kwargs: notifications.append(
                (message, severity)
            ),
        )
        monkeypatch.setattr(
            screen,
            "_console_setup_blocked_reason",
            lambda: "provider setup is incomplete",
        )
        app.pending_handoffs.stage(
            HandoffChannel.CONSOLE_PROMPT_INSERT,
            "retry after setup",
        )

        await screen._consume_pending_console_prompt_insert()

        released = app.pending_handoffs.claim(HandoffChannel.CONSOLE_PROMPT_INSERT)
        assert released is not None
        assert released.value == "retry after setup"
        assert app.pending_handoffs.release(released) is True
        assert composer.draft_text() == ""
        assert notifications == [
            (screen._LIBRARY_PROMPT_INSERT_BLOCKED_COPY, "warning")
        ]

        monkeypatch.setattr(screen, "_console_setup_blocked_reason", lambda: "")
        await screen._consume_pending_console_prompt_insert()

        assert composer.draft_text() == "retry after setup"
        assert app.pending_handoffs.claim(HandoffChannel.CONSOLE_PROMPT_INSERT) is None


@pytest.mark.asyncio
async def test_full_app_console_prompt_missing_composer_releases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch, route="chat") as pilot:
        screen, _composer = await _mounted_chat(app, pilot)
        monkeypatch.setattr(screen, "_console_setup_blocked_reason", lambda: "")
        insert_attempts: list[tuple[str, bool]] = []

        def reject_insert(text, *, replace):
            insert_attempts.append((text, replace))
            return False

        monkeypatch.setattr(
            screen,
            "_insert_prompt_text_into_composer",
            reject_insert,
        )
        app.pending_handoffs.stage(
            HandoffChannel.CONSOLE_PROMPT_INSERT,
            "retry when mounted",
        )

        await screen._consume_pending_console_prompt_insert()

        released = app.pending_handoffs.claim(HandoffChannel.CONSOLE_PROMPT_INSERT)
        assert insert_attempts == [("retry when mounted", False)]
        assert released is not None
        assert released.value == "retry when mounted"


@pytest.mark.asyncio
async def test_full_app_console_prompt_cancellation_releases_and_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch, route="chat") as pilot:
        screen, _composer = await _mounted_chat(app, pilot)
        monkeypatch.setattr(screen, "_console_setup_blocked_reason", lambda: "")

        def cancel_insert(_text, *, replace):
            raise asyncio.CancelledError

        monkeypatch.setattr(
            screen,
            "_insert_prompt_text_into_composer",
            cancel_insert,
        )
        app.pending_handoffs.stage(
            HandoffChannel.CONSOLE_PROMPT_INSERT,
            "retry after cancellation",
        )

        with pytest.raises(asyncio.CancelledError):
            await screen._consume_pending_console_prompt_insert()

        released = app.pending_handoffs.claim(HandoffChannel.CONSOLE_PROMPT_INSERT)
        assert released is not None
        assert released.value == "retry after cancellation"


@pytest.mark.asyncio
async def test_full_app_console_prompt_failure_releases_without_logging_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "TASK-645-PRIVATE-PROMPT-SENTINEL"
    app = TldwCli()

    async with _mounted_app(app, monkeypatch, route="chat") as pilot:
        screen, _composer = await _mounted_chat(app, pilot)
        monkeypatch.setattr(screen, "_console_setup_blocked_reason", lambda: "")

        def fail_insert(_text, *, replace):
            raise RuntimeError(sentinel)

        monkeypatch.setattr(
            screen,
            "_insert_prompt_text_into_composer",
            fail_insert,
        )
        app.pending_handoffs.stage(
            HandoffChannel.CONSOLE_PROMPT_INSERT,
            sentinel,
        )
        messages: list[str] = []
        sink_id = logger.add(messages.append, format="{message}")
        try:
            await screen._consume_pending_console_prompt_insert()
        finally:
            logger.remove(sink_id)

        released = app.pending_handoffs.claim(HandoffChannel.CONSOLE_PROMPT_INSERT)
        combined = "\n".join(messages)
        assert released is not None
        assert released.value == sentinel
        assert "channel=console_prompt_insert" in combined
        assert "exception_category=RuntimeError" in combined
        assert sentinel not in combined


@pytest.mark.asyncio
async def test_full_app_console_prompt_replacement_survives_older_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch, route="chat") as pilot:
        screen, _composer = await _mounted_chat(app, pilot)
        monkeypatch.setattr(screen, "_console_setup_blocked_reason", lambda: "")

        def insert_after_replacement(text, *, replace):
            assert text == "first"
            app.pending_handoffs.stage(
                HandoffChannel.CONSOLE_PROMPT_INSERT,
                "second",
            )
            return True

        monkeypatch.setattr(
            screen,
            "_insert_prompt_text_into_composer",
            insert_after_replacement,
        )
        app.pending_handoffs.stage(
            HandoffChannel.CONSOLE_PROMPT_INSERT,
            "first",
        )

        await screen._consume_pending_console_prompt_insert()
        replacement = app.pending_handoffs.claim(HandoffChannel.CONSOLE_PROMPT_INSERT)

        assert replacement is not None
        assert replacement.value == "second"
