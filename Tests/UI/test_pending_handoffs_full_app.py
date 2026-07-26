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
from tldw_chatbook.Widgets.Chat_Widgets.chat_handoff_card import ChatHandoffCard
from tldw_chatbook.Widgets.Chat_Widgets.chat_tab_container import ChatTabContainer
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


async def _mount_production_chat_tabs(
    app: TldwCli,
    screen: ChatScreen,
    pilot,
    monkeypatch: pytest.MonkeyPatch,
) -> ChatTabContainer:
    tab_container = ChatTabContainer(app, id="task-645-production-chat-tabs")
    await screen.mount(tab_container)
    for _ in range(150):
        if tab_container.is_mounted and tab_container.sessions:
            monkeypatch.setattr(screen, "_get_tab_container", lambda: tab_container)
            return tab_container
        await pilot.pause(0.01)
    raise AssertionError("production ChatTabContainer did not finish mounting")


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
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentinel = "TASK-645-PRODUCER-NORMALIZATION-PRIVATE-SENTINEL"

    class CopyFailure:
        def __deepcopy__(self, _memo):
            raise TypeError(sentinel)

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
        messages: list[str] = []
        sink_id = logger.add(messages.append, format="{message}")
        try:
            app.stage_console_prompt_insert("   ")
            app.open_console_for_live_work(
                source="tests",
                title="copy failure",
                payload={"nested": CopyFailure()},
            )
            app.open_chat_with_handoff(object())  # type: ignore[arg-type]
        finally:
            logger.remove(sink_id)

        assert messages == []
        assert len(notifications) == 3
        assert all(severity == "warning" for _message, severity in notifications)
        assert app.pending_handoffs.claim(HandoffChannel.CONSOLE_PROMPT_INSERT) is None
        assert app.pending_handoffs.claim(HandoffChannel.CONSOLE_LIVE_WORK) is None
        assert app.pending_handoffs.claim(HandoffChannel.CHAT) is None
        assert sentinel not in caplog.text


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


@pytest.mark.asyncio
async def test_full_app_chat_native_transfer_acknowledges_after_local_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch, route="chat") as pilot:
        screen, _composer = await _mounted_chat(app, pilot)
        app.pending_handoffs.stage(HandoffChannel.CHAT, _chat_payload("native"))
        real_acknowledge = app.pending_handoffs.acknowledge

        def acknowledge_after_transfer(claim) -> bool:
            assert screen._pending_console_launch_context is not None
            assert screen._pending_console_launch_context.title == "native"
            return real_acknowledge(claim)

        monkeypatch.setattr(
            app.pending_handoffs,
            "acknowledge",
            acknowledge_after_transfer,
        )

        await screen._consume_pending_chat_handoff()

        assert app.pending_handoffs.claim(HandoffChannel.CHAT) is None


@pytest.mark.asyncio
async def test_full_app_chat_native_rag_recovery_does_not_log_payload(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import tldw_chatbook.Chat.citation_evidence_models as evidence_models

    sentinel = "TASK-645-NATIVE-RAG-PRIVATE-SENTINEL"

    class RejectEvidenceBundle:
        def __init__(self, **_kwargs):
            raise ValueError(sentinel)

    app = TldwCli()

    async with _mounted_app(app, monkeypatch, route="chat") as pilot:
        screen, _composer = await _mounted_chat(app, pilot)
        monkeypatch.setattr(evidence_models, "EvidenceBundle", RejectEvidenceBundle)
        payload = ChatHandoffPayload(
            source="rag-search",
            item_type="document",
            title="private RAG handoff",
            body="private context",
            metadata={"nested": {"private": sentinel}},
        )
        app.pending_handoffs.stage(HandoffChannel.CHAT, payload)
        messages: list[str] = []
        sink_id = logger.add(messages.append, format="{message}")
        try:
            await screen._consume_pending_chat_handoff()
        finally:
            logger.remove(sink_id)

        combined = "\n".join(messages)
        assert screen._pending_console_launch_context is not None
        assert app.pending_handoffs.claim(HandoffChannel.CHAT) is None
        assert "exception_category=ValueError" in combined
        assert sentinel not in combined
        assert sentinel not in caplog.text


@pytest.mark.asyncio
async def test_full_app_chat_cancellation_before_create_releases_without_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch, route="chat") as pilot:
        screen, _composer = await _mounted_chat(app, pilot)
        tab_container = await _mount_production_chat_tabs(
            app,
            screen,
            pilot,
            monkeypatch,
        )
        cleanup_calls: list[str] = []

        async def cancel_create(*, session_data):
            raise asyncio.CancelledError

        async def record_cleanup(tab_id: str) -> None:
            cleanup_calls.append(tab_id)

        monkeypatch.setattr(tab_container, "create_new_tab", cancel_create)
        monkeypatch.setattr(tab_container, "close_tab", record_cleanup)
        app.pending_handoffs.stage(HandoffChannel.CHAT, _chat_payload("cancel-create"))

        with pytest.raises(asyncio.CancelledError):
            await screen._consume_pending_chat_handoff()

        released = app.pending_handoffs.claim(HandoffChannel.CHAT)
        assert released is not None
        assert released.value.title == "cancel-create"
        assert cleanup_calls == []


@pytest.mark.asyncio
async def test_full_app_chat_empty_created_id_warns_and_releases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch, route="chat") as pilot:
        screen, _composer = await _mounted_chat(app, pilot)
        tab_container = await _mount_production_chat_tabs(
            app,
            screen,
            pilot,
            monkeypatch,
        )
        create_calls = 0
        notifications: list[tuple[str, str]] = []

        async def reject_create(*, session_data):
            nonlocal create_calls
            create_calls += 1
            return ""

        monkeypatch.setattr(tab_container, "create_new_tab", reject_create)
        monkeypatch.setattr(
            app,
            "notify",
            lambda message, *, severity="information", **_kwargs: notifications.append(
                (message, severity)
            ),
        )
        app.pending_handoffs.stage(HandoffChannel.CHAT, _chat_payload("no-id"))

        await screen._consume_pending_chat_handoff()

        released = app.pending_handoffs.claim(HandoffChannel.CHAT)
        assert create_calls == 1
        assert released is not None
        assert released.value.title == "no-id"
        assert notifications == [
            ("Could not create a chat session for this context.", "error")
        ]


@pytest.mark.asyncio
async def test_full_app_chat_switch_failure_rolls_back_exact_created_tab(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch, route="chat") as pilot:
        screen, _composer = await _mounted_chat(app, pilot)
        tab_container = await _mount_production_chat_tabs(
            app,
            screen,
            pilot,
            monkeypatch,
        )
        preexisting_ids = set(tab_container.sessions)
        created_ids: list[str] = []
        cleanup_calls: list[str] = []
        real_create = tab_container.create_new_tab
        real_close = tab_container.close_tab

        async def record_create(*, session_data):
            tab_id = await real_create(session_data=session_data)
            created_ids.append(tab_id)
            return tab_id

        async def fail_switch(_tab_id: str) -> None:
            raise RuntimeError("switch failed")

        async def record_cleanup(tab_id: str) -> None:
            cleanup_calls.append(tab_id)
            await real_close(tab_id)

        monkeypatch.setattr(tab_container, "create_new_tab", record_create)
        monkeypatch.setattr(tab_container, "switch_to_tab_async", fail_switch)
        monkeypatch.setattr(tab_container, "close_tab", record_cleanup)
        app.pending_handoffs.stage(HandoffChannel.CHAT, _chat_payload("switch"))

        await screen._consume_pending_chat_handoff()

        released = app.pending_handoffs.claim(HandoffChannel.CHAT)
        assert len(created_ids) == 1
        assert cleanup_calls == created_ids
        assert created_ids[0] not in tab_container.sessions
        assert preexisting_ids <= set(tab_container.sessions)
        assert released is not None
        assert released.value.title == "switch"


@pytest.mark.asyncio
async def test_full_app_chat_switch_cancellation_rolls_back_and_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch, route="chat") as pilot:
        screen, _composer = await _mounted_chat(app, pilot)
        tab_container = await _mount_production_chat_tabs(
            app,
            screen,
            pilot,
            monkeypatch,
        )
        created_ids: list[str] = []
        cleanup_calls: list[str] = []
        real_create = tab_container.create_new_tab
        real_close = tab_container.close_tab

        async def record_create(*, session_data):
            tab_id = await real_create(session_data=session_data)
            created_ids.append(tab_id)
            return tab_id

        async def cancel_switch(_tab_id: str) -> None:
            raise asyncio.CancelledError

        async def record_cleanup(tab_id: str) -> None:
            cleanup_calls.append(tab_id)
            await real_close(tab_id)

        monkeypatch.setattr(tab_container, "create_new_tab", record_create)
        monkeypatch.setattr(tab_container, "switch_to_tab_async", cancel_switch)
        monkeypatch.setattr(tab_container, "close_tab", record_cleanup)
        app.pending_handoffs.stage(HandoffChannel.CHAT, _chat_payload("cancel-switch"))

        with pytest.raises(asyncio.CancelledError):
            await screen._consume_pending_chat_handoff()

        released = app.pending_handoffs.claim(HandoffChannel.CHAT)
        assert cleanup_calls == created_ids
        assert created_ids[0] not in tab_container.sessions
        assert released is not None
        assert released.value.title == "cancel-switch"


@pytest.mark.asyncio
async def test_full_app_chat_apply_cancellation_rolls_back_and_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch, route="chat") as pilot:
        screen, _composer = await _mounted_chat(app, pilot)
        tab_container = await _mount_production_chat_tabs(
            app,
            screen,
            pilot,
            monkeypatch,
        )
        created_ids: list[str] = []
        cleanup_calls: list[str] = []
        real_create = tab_container.create_new_tab
        real_close = tab_container.close_tab

        async def record_create(*, session_data):
            tab_id = await real_create(session_data=session_data)
            created_ids.append(tab_id)
            return tab_id

        async def cancel_apply(_session, _payload) -> None:
            raise asyncio.CancelledError

        async def record_cleanup(tab_id: str) -> None:
            cleanup_calls.append(tab_id)
            await real_close(tab_id)

        monkeypatch.setattr(tab_container, "create_new_tab", record_create)
        monkeypatch.setattr(screen, "_apply_handoff_to_chat_session", cancel_apply)
        monkeypatch.setattr(tab_container, "close_tab", record_cleanup)
        app.pending_handoffs.stage(HandoffChannel.CHAT, _chat_payload("cancel-apply"))

        with pytest.raises(asyncio.CancelledError):
            await screen._consume_pending_chat_handoff()

        released = app.pending_handoffs.claim(HandoffChannel.CHAT)
        assert cleanup_calls == created_ids
        assert created_ids[0] not in tab_container.sessions
        assert released is not None
        assert released.value.title == "cancel-apply"


@pytest.mark.asyncio
async def test_full_app_chat_missing_created_session_releases_after_exact_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch, route="chat") as pilot:
        screen, _composer = await _mounted_chat(app, pilot)
        tab_container = await _mount_production_chat_tabs(
            app,
            screen,
            pilot,
            monkeypatch,
        )
        created_ids: list[str] = []
        cleanup_calls: list[str] = []
        real_create = tab_container.create_new_tab
        real_switch = tab_container.switch_to_tab_async
        real_close = tab_container.close_tab

        async def record_create(*, session_data):
            tab_id = await real_create(session_data=session_data)
            created_ids.append(tab_id)
            return tab_id

        async def remove_during_switch(tab_id: str) -> None:
            await real_switch(tab_id)
            await real_close(tab_id)

        async def record_cleanup(tab_id: str) -> None:
            cleanup_calls.append(tab_id)
            await real_close(tab_id)

        monkeypatch.setattr(tab_container, "create_new_tab", record_create)
        monkeypatch.setattr(
            tab_container,
            "switch_to_tab_async",
            remove_during_switch,
        )
        monkeypatch.setattr(tab_container, "close_tab", record_cleanup)
        app.pending_handoffs.stage(HandoffChannel.CHAT, _chat_payload("missing"))

        await screen._consume_pending_chat_handoff()

        released = app.pending_handoffs.claim(HandoffChannel.CHAT)
        assert cleanup_calls == created_ids
        assert created_ids[0] not in tab_container.sessions
        assert released is not None
        assert released.value.title == "missing"


@pytest.mark.asyncio
async def test_full_app_chat_apply_failure_rolls_back_without_logging_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "TASK-645-PRIVATE-CHAT-SENTINEL"
    app = TldwCli()

    async with _mounted_app(app, monkeypatch, route="chat") as pilot:
        screen, _composer = await _mounted_chat(app, pilot)
        tab_container = await _mount_production_chat_tabs(
            app,
            screen,
            pilot,
            monkeypatch,
        )
        created_ids: list[str] = []
        cleanup_calls: list[str] = []
        real_create = tab_container.create_new_tab
        real_close = tab_container.close_tab

        async def record_create(*, session_data):
            tab_id = await real_create(session_data=session_data)
            created_ids.append(tab_id)
            return tab_id

        async def fail_apply(session, _payload) -> None:
            assert tab_container.tab_bar is not None
            tab_container.tab_bar.update_tab_title(
                session.session_data.tab_id,
                sentinel,
            )
            raise RuntimeError(sentinel)

        async def record_cleanup(tab_id: str) -> None:
            cleanup_calls.append(tab_id)
            await real_close(tab_id)

        monkeypatch.setattr(tab_container, "create_new_tab", record_create)
        monkeypatch.setattr(screen, "_apply_handoff_to_chat_session", fail_apply)
        monkeypatch.setattr(tab_container, "close_tab", record_cleanup)
        payload = _chat_payload(sentinel)
        payload.metadata["nested"]["items"] = [sentinel]
        app.pending_handoffs.stage(HandoffChannel.CHAT, payload)
        messages: list[str] = []
        sink_id = logger.add(messages.append, format="{message}")
        try:
            await screen._consume_pending_chat_handoff()
        finally:
            logger.remove(sink_id)

        released = app.pending_handoffs.claim(HandoffChannel.CHAT)
        combined = "\n".join(messages)
        assert cleanup_calls == created_ids
        assert created_ids[0] not in tab_container.sessions
        assert released is not None
        assert released.value.title == sentinel
        assert "channel=chat" in combined
        assert "exception_category=RuntimeError" in combined
        assert sentinel not in combined


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_mode", ("raises", "retains"))
async def test_full_app_chat_cleanup_failure_acknowledges_to_prevent_duplicate(
    monkeypatch: pytest.MonkeyPatch,
    cleanup_mode: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentinel = f"TASK-645-CLEANUP-{cleanup_mode}-PRIVATE-SENTINEL"
    app = TldwCli()

    async with _mounted_app(app, monkeypatch, route="chat") as pilot:
        screen, _composer = await _mounted_chat(app, pilot)
        tab_container = await _mount_production_chat_tabs(
            app,
            screen,
            pilot,
            monkeypatch,
        )
        created_ids: list[str] = []
        cleanup_calls: list[str] = []
        notifications: list[tuple[str, str]] = []
        real_create = tab_container.create_new_tab

        async def record_create(*, session_data):
            tab_id = await real_create(session_data=session_data)
            created_ids.append(tab_id)
            return tab_id

        async def fail_apply(_session, _payload) -> None:
            raise RuntimeError(sentinel)

        async def retain_partial_tab(tab_id: str) -> None:
            cleanup_calls.append(tab_id)
            if cleanup_mode == "raises":
                raise ValueError(sentinel)

        monkeypatch.setattr(tab_container, "create_new_tab", record_create)
        monkeypatch.setattr(screen, "_apply_handoff_to_chat_session", fail_apply)
        monkeypatch.setattr(tab_container, "close_tab", retain_partial_tab)
        monkeypatch.setattr(
            app,
            "notify",
            lambda message, *, severity="information", **_kwargs: notifications.append(
                (message, severity)
            ),
        )
        app.pending_handoffs.stage(HandoffChannel.CHAT, _chat_payload("partial"))

        messages: list[str] = []
        sink_id = logger.add(messages.append, format="{message}")
        try:
            await screen._consume_pending_chat_handoff()
            await screen._consume_pending_chat_handoff()
        finally:
            logger.remove(sink_id)

        combined = "\n".join(messages)
        assert cleanup_calls == created_ids
        assert len(created_ids) == 1
        assert created_ids[0] in tab_container.sessions
        assert app.pending_handoffs.claim(HandoffChannel.CHAT) is None
        assert "channel=chat" in combined
        assert "exception_category=RuntimeError" in combined
        if cleanup_mode == "raises":
            assert "outcome=exception" in combined
            assert "exception_category=ValueError" in combined
        else:
            assert "outcome=tab_retained" in combined
        assert sentinel not in combined
        assert sentinel not in caplog.text
        assert notifications == [
            (
                "Chat context could not be applied cleanly. "
                "Close the incomplete tab before trying again.",
                "warning",
            )
        ]


@pytest.mark.asyncio
async def test_full_app_chat_success_acknowledges_only_after_context_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch, route="chat") as pilot:
        screen, _composer = await _mounted_chat(app, pilot)
        tab_container = await _mount_production_chat_tabs(
            app,
            screen,
            pilot,
            monkeypatch,
        )
        applied = False
        acknowledged_after_apply = False
        real_apply = screen._apply_handoff_to_chat_session
        real_acknowledge = app.pending_handoffs.acknowledge

        async def record_apply(session, payload) -> None:
            nonlocal applied
            assert app.pending_handoffs.claim(HandoffChannel.CHAT) is None
            await real_apply(session, payload)
            applied = True

        def record_acknowledge(claim) -> bool:
            nonlocal acknowledged_after_apply
            acknowledged_after_apply = applied
            return real_acknowledge(claim)

        monkeypatch.setattr(screen, "_apply_handoff_to_chat_session", record_apply)
        monkeypatch.setattr(
            app.pending_handoffs,
            "acknowledge",
            record_acknowledge,
        )
        payload = _chat_payload("success")
        app.pending_handoffs.stage(HandoffChannel.CHAT, payload)

        await screen._consume_pending_chat_handoff()

        active_id = tab_container.active_session_id
        assert applied is True
        assert acknowledged_after_apply is True
        assert app.pending_handoffs.claim(HandoffChannel.CHAT) is None
        assert active_id is not None
        active_session = tab_container.sessions[active_id]
        cards = list(active_session.query(ChatHandoffCard))
        assert len(cards) == 1
        assert cards[0].payload.title == "success"
        assert active_session.get_chat_input().text == payload.default_prompt()


@pytest.mark.asyncio
async def test_full_app_chat_replacement_staged_while_old_claim_waits_survives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch, route="chat") as pilot:
        screen, _composer = await _mounted_chat(app, pilot)
        tab_container = await _mount_production_chat_tabs(
            app,
            screen,
            pilot,
            monkeypatch,
        )
        create_entered = asyncio.Event()
        continue_create = asyncio.Event()
        real_create = tab_container.create_new_tab

        async def paused_create(*, session_data):
            create_entered.set()
            await continue_create.wait()
            return await real_create(session_data=session_data)

        monkeypatch.setattr(tab_container, "create_new_tab", paused_create)
        app.pending_handoffs.stage(HandoffChannel.CHAT, _chat_payload("first"))

        consumer = asyncio.create_task(screen._consume_pending_chat_handoff())
        await create_entered.wait()
        app.pending_handoffs.stage(HandoffChannel.CHAT, _chat_payload("second"))
        continue_create.set()
        await consumer

        replacement = app.pending_handoffs.claim(HandoffChannel.CHAT)
        assert replacement is not None
        assert replacement.value.title == "second"
