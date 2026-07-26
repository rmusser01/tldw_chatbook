from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from loguru import logger
import pytest

import tldw_chatbook.app as app_module
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel
from tldw_chatbook.UI.Screens.study_scope_models import (
    StudyScopeContext,
    StudySourceItem,
)
from tldw_chatbook.UI.Screens.study_screen import StudyScreen


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


async def _wait_for_study_screen(app: TldwCli, pilot) -> StudyScreen:
    for _ in range(300):
        if isinstance(app.screen, StudyScreen) and app.screen.is_mounted:
            await pilot.pause(0.01)
            return app.screen
        await pilot.pause(0.01)
    raise AssertionError("full app did not mount the production Study screen")


@pytest.mark.asyncio
async def test_full_app_study_producer_stages_both_channels_before_navigation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()
    scope = StudyScopeContext(material_title="Private study material")

    async with _mounted_app(app, monkeypatch):
        observed: list[str] = []

        def observe_navigation(message: NavigateToScreen) -> None:
            assert message.screen_name == "study"
            scope_claim = app.pending_handoffs.claim(HandoffChannel.STUDY_SCOPE)
            section_claim = app.pending_handoffs.claim(
                HandoffChannel.STUDY_INITIAL_SECTION
            )
            assert scope_claim is not None
            assert scope_claim.value == scope
            assert section_claim is not None
            assert section_claim.value == "flashcards"
            assert app.pending_handoffs.acknowledge(scope_claim) is True
            assert app.pending_handoffs.acknowledge(section_claim) is True
            observed.append(message.screen_name)

        _intercept_navigation(app, monkeypatch, observe_navigation)

        app.open_study_screen(scope, initial_section="flashcards")

        assert observed == ["study"]


@pytest.mark.asyncio
async def test_full_app_study_producer_clears_omitted_optional_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch):
        app.pending_handoffs.stage(
            HandoffChannel.STUDY_SCOPE,
            StudyScopeContext(material_title="stale"),
        )
        app.pending_handoffs.stage(
            HandoffChannel.STUDY_INITIAL_SECTION,
            "quizzes",
        )

        def observe_navigation(message: NavigateToScreen) -> None:
            assert message.screen_name == "study"
            assert app.pending_handoffs.claim(HandoffChannel.STUDY_SCOPE) is None
            assert (
                app.pending_handoffs.claim(HandoffChannel.STUDY_INITIAL_SECTION) is None
            )

        _intercept_navigation(app, monkeypatch, observe_navigation)

        app.open_study_screen()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "target_id", "expected_route", "channel"),
    [
        (
            "Artifacts",
            "local:chatbook:77",
            "artifacts",
            HandoffChannel.ARTIFACT_CHATBOOK_TARGET,
        ),
        (
            "ACP",
            "local:acp_session:session-1",
            "acp",
            HandoffChannel.ACP_SESSION_TARGET,
        ),
    ],
)
async def test_full_app_console_primary_action_stages_exact_target_before_navigation(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    target_id: str,
    expected_route: str,
    channel: HandoffChannel,
) -> None:
    app = TldwCli()
    launch = ConsoleLiveWorkLaunch.from_values(
        source=source,
        title="Live work",
        payload={"target_id": target_id},
    )

    async with _mounted_app(app, monkeypatch):
        observed: list[str] = []

        def observe_navigation(message: NavigateToScreen) -> None:
            assert message.screen_name == expected_route
            claim = app.pending_handoffs.claim(channel)
            assert claim is not None
            assert claim.value == target_id
            assert app.pending_handoffs.acknowledge(claim) is True
            observed.append(message.screen_name)

        _intercept_navigation(app, monkeypatch, observe_navigation)

        assert app.open_console_live_work_primary_action(launch) is True
        assert observed == [expected_route]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "target_id", "channel"),
    [
        (
            "Artifacts",
            "remote:chatbook:77",
            HandoffChannel.ARTIFACT_CHATBOOK_TARGET,
        ),
        (
            "ACP",
            "remote:acp_session:session-1",
            HandoffChannel.ACP_SESSION_TARGET,
        ),
    ],
)
async def test_full_app_console_primary_action_rejects_noncanonical_target(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    target_id: str,
    channel: HandoffChannel,
) -> None:
    app = TldwCli()
    launch = ConsoleLiveWorkLaunch.from_values(
        source=source,
        title="Live work",
        payload={"target_id": target_id},
    )

    async with _mounted_app(app, monkeypatch):
        navigation: list[NavigateToScreen] = []
        notifications: list[tuple[str, str]] = []
        _intercept_navigation(app, monkeypatch, navigation.append)
        monkeypatch.setattr(
            app,
            "notify",
            lambda message, *, severity="information", **_kwargs: notifications.append(
                (message, severity)
            ),
        )

        assert app.open_console_live_work_primary_action(launch) is False
        assert app.pending_handoffs.claim(channel) is None
        assert navigation == []
        assert notifications == [
            ("Console action target could not be opened. Try again.", "warning")
        ]


@pytest.mark.asyncio
async def test_full_app_study_handoff_overrides_restored_scope_and_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()
    incoming = StudyScopeContext(
        material_title="Incoming material",
        source_items=(
            StudySourceItem(
                source_type="media",
                source_id="media-1",
                locator={"page": 7},
            ),
        ),
    )

    async with _mounted_app(app, monkeypatch) as pilot:
        app.screen_state_store.save(
            "study",
            {
                "study_scope": {
                    "scope_type": "global",
                    "material_title": "Restored material",
                },
                "study_section": "quizzes",
            },
            app._current_runtime_identity(),
        )

        app.open_study_screen(incoming, initial_section="flashcards")
        screen = await _wait_for_study_screen(app, pilot)

        for _ in range(300):
            if (
                screen.scope_state.material_title == "Incoming material"
                and screen.current_section == "flashcards"
            ):
                break
            await pilot.pause(0.01)

        assert screen.scope_state.material_title == "Incoming material"
        assert screen.scope_state.source_items[0].locator == {"page": "7"}
        assert screen.current_section == "flashcards"
        assert app.pending_handoffs.claim(HandoffChannel.STUDY_SCOPE) is None
        assert app.pending_handoffs.claim(HandoffChannel.STUDY_INITIAL_SECTION) is None


@pytest.mark.asyncio
async def test_full_app_study_scope_failure_releases_only_scope_and_redacts_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "TASK-646-STUDY-PRIVATE-SENTINEL"
    app = TldwCli()

    async with _mounted_app(app, monkeypatch, route="study") as pilot:
        screen = await _wait_for_study_screen(app, pilot)
        app.pending_handoffs.stage(
            HandoffChannel.STUDY_SCOPE,
            StudyScopeContext(material_title=sentinel),
        )
        app.pending_handoffs.stage(
            HandoffChannel.STUDY_INITIAL_SECTION,
            "flashcards",
        )

        async def fail_scope(*_args, **_kwargs) -> None:
            raise ValueError(sentinel)

        monkeypatch.setattr(screen, "_apply_scope_context_and_refresh", fail_scope)
        messages: list[str] = []
        sink_id = logger.add(messages.append, format="{message}")
        try:
            await screen.on_screen_resume()
        finally:
            logger.remove(sink_id)

        scope_retry = app.pending_handoffs.claim(HandoffChannel.STUDY_SCOPE)
        assert scope_retry is not None
        assert scope_retry.value.material_title == sentinel
        assert app.pending_handoffs.claim(HandoffChannel.STUDY_INITIAL_SECTION) is None
        assert screen.current_section == "flashcards"
        assert any("exception_category=ValueError" in message for message in messages)
        assert all(sentinel not in message for message in messages)


@pytest.mark.asyncio
async def test_full_app_study_scope_cancellation_releases_and_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch, route="study") as pilot:
        screen = await _wait_for_study_screen(app, pilot)
        app.pending_handoffs.stage(
            HandoffChannel.STUDY_SCOPE,
            StudyScopeContext(material_title="cancelled"),
        )
        started = asyncio.Event()
        never_finish = asyncio.Event()

        async def block_scope(*_args, **_kwargs) -> None:
            started.set()
            await never_finish.wait()

        monkeypatch.setattr(screen, "_apply_scope_context_and_refresh", block_scope)
        resume = asyncio.create_task(screen.on_screen_resume())
        await asyncio.wait_for(started.wait(), timeout=2)
        assert app.pending_handoffs.claim(HandoffChannel.STUDY_SCOPE) is None
        resume.cancel()

        with pytest.raises(asyncio.CancelledError):
            await resume

        retry = app.pending_handoffs.claim(HandoffChannel.STUDY_SCOPE)
        assert retry is not None
        assert retry.value.material_title == "cancelled"


@pytest.mark.asyncio
async def test_full_app_study_newer_scope_survives_older_consumer_acknowledge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()

    async with _mounted_app(app, monkeypatch, route="study") as pilot:
        screen = await _wait_for_study_screen(app, pilot)
        app.pending_handoffs.stage(
            HandoffChannel.STUDY_SCOPE,
            StudyScopeContext(material_title="first"),
        )
        started = asyncio.Event()
        continue_apply = asyncio.Event()

        async def block_scope(*_args, **_kwargs) -> None:
            started.set()
            await continue_apply.wait()

        monkeypatch.setattr(screen, "_apply_scope_context_and_refresh", block_scope)
        resume = asyncio.create_task(screen.on_screen_resume())
        await asyncio.wait_for(started.wait(), timeout=2)
        assert app.pending_handoffs.claim(HandoffChannel.STUDY_SCOPE) is None

        app.pending_handoffs.stage(
            HandoffChannel.STUDY_SCOPE,
            StudyScopeContext(material_title="replacement"),
        )
        continue_apply.set()
        await resume

        replacement = app.pending_handoffs.claim(HandoffChannel.STUDY_SCOPE)
        assert replacement is not None
        assert replacement.value.material_title == "replacement"
