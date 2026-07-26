from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import threading
from typing import Any

from loguru import logger
import pytest

import tldw_chatbook.app as app_module
from tldw_chatbook.ACP_Interop.runtime_session import (
    ACPRuntimeSessionState,
    acp_session_record_id,
)
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel
from tldw_chatbook.UI.Screens.acp_screen import ACPScreen
from tldw_chatbook.UI.Screens.artifacts_screen import ArtifactsScreen
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


async def _wait_for_artifacts_screen(app: TldwCli, pilot) -> ArtifactsScreen:
    for _ in range(300):
        if isinstance(app.screen, ArtifactsScreen) and app.screen.is_mounted:
            await pilot.pause(0.01)
            return app.screen
        await pilot.pause(0.01)
    raise AssertionError("full app did not mount the production Artifacts screen")


async def _wait_for_acp_screen(app: TldwCli, pilot) -> ACPScreen:
    for _ in range(300):
        if isinstance(app.screen, ACPScreen) and app.screen.is_mounted:
            await pilot.pause(0.01)
            return app.screen
        await pilot.pause(0.01)
    raise AssertionError("full app did not mount the production ACP screen")


async def _wait_for_artifact_context(
    screen: ArtifactsScreen,
    pilot,
    *,
    title: str | None = None,
) -> None:
    for _ in range(300):
        launch = screen._latest_chatbook_console_launch
        if screen._chatbook_context_loaded and (
            title is None or (launch is not None and launch["title"] == title)
        ):
            await pilot.pause(0.01)
            return
        await pilot.pause(0.01)
    raise AssertionError("production Artifacts screen did not load expected context")


def _chatbook_record(chatbook_id: int | str, title: str) -> dict[str, Any]:
    return {
        "id": str(chatbook_id),
        "chatbook_id": chatbook_id,
        "name": title,
        "description": f"{title} description",
        "updated_at": "2026-07-26T12:00:00+00:00",
    }


def _acp_runtime_state(session_id: str = "session-1") -> ACPRuntimeSessionState:
    return ACPRuntimeSessionState(
        runtime_id="runtime-1",
        runtime_label="Production runtime",
        session_id=session_id,
        session_title="Current session",
        session_status="running",
        session_payload={"pid": 646},
    )


class InjectedChatbookService:
    """Controllable Chatbook collaborator used by the mounted production app."""

    def __init__(
        self,
        *,
        listed: list[dict[str, Any]] | None = None,
        exact: dict[str, dict[str, Any] | BaseException] | None = None,
        barriers: dict[str, tuple[threading.Event, threading.Event]] | None = None,
    ) -> None:
        self.listed = list(listed or [])
        self.exact = dict(exact or {})
        self.barriers = dict(barriers or {})
        self.list_calls: list[dict[str, Any]] = []
        self.get_calls: list[str] = []

    async def list_chatbooks(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.list_calls.append(dict(kwargs))
        return list(self.listed)

    async def get_chatbook(self, chatbook_id: int | str) -> dict[str, Any]:
        normalized_id = str(chatbook_id)
        self.get_calls.append(normalized_id)
        barrier = self.barriers.get(normalized_id)
        if barrier is not None:
            started, finish = barrier
            started.set()
            if not finish.wait(timeout=5):
                raise TimeoutError("injected Chatbook lookup barrier timed out")
        result = self.exact.get(normalized_id)
        if isinstance(result, BaseException):
            raise result
        if result is None:
            raise KeyError(normalized_id)
        return dict(result)


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


@pytest.mark.asyncio
async def test_full_app_artifact_handoff_uses_exact_lookup_not_latest_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()
    service = InjectedChatbookService(
        listed=[_chatbook_record(99, "Latest")],
        exact={"77": _chatbook_record(77, "Requested")},
    )
    app.local_chatbook_service = service

    async with _mounted_app(app, monkeypatch) as pilot:
        app.pending_handoffs.stage(
            HandoffChannel.ARTIFACT_CHATBOOK_TARGET,
            "local:chatbook:77",
        )
        app.post_message(NavigateToScreen("artifacts"))
        screen = await _wait_for_artifacts_screen(app, pilot)
        await _wait_for_artifact_context(screen, pilot, title="Requested")

        assert service.get_calls == ["77"]
        assert service.list_calls == []
        assert screen._latest_chatbook_console_launch is not None
        assert (
            screen._latest_chatbook_console_launch["payload"]["target_id"]
            == "local:chatbook:77"
        )
        assert "Requested" in str(screen.query_one("#artifacts-detail-ready").render())
        assert (
            app.pending_handoffs.claim(HandoffChannel.ARTIFACT_CHATBOOK_TARGET) is None
        )


@pytest.mark.asyncio
async def test_full_app_missing_artifact_is_terminal_with_explicit_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()
    service = InjectedChatbookService(
        listed=[_chatbook_record(99, "Latest")],
        exact={},
    )
    app.local_chatbook_service = service

    async with _mounted_app(app, monkeypatch) as pilot:
        notifications: list[tuple[str, str]] = []
        monkeypatch.setattr(
            app,
            "notify",
            lambda message, *, severity="information", **_kwargs: notifications.append(
                (message, severity)
            ),
        )
        app.pending_handoffs.stage(
            HandoffChannel.ARTIFACT_CHATBOOK_TARGET,
            "local:chatbook:77",
        )
        app.post_message(NavigateToScreen("artifacts"))
        screen = await _wait_for_artifacts_screen(app, pilot)
        await _wait_for_artifact_context(screen, pilot)

        assert service.get_calls == ["77"]
        assert service.list_calls == []
        assert screen._latest_chatbook_console_launch is None
        assert "no longer exists" in str(
            screen.query_one("#artifacts-chatbook-target-missing").render()
        )
        assert notifications == [
            (
                "The requested local Chatbook artifact no longer exists.",
                "warning",
            )
        ]
        assert (
            app.pending_handoffs.claim(HandoffChannel.ARTIFACT_CHATBOOK_TARGET) is None
        )


@pytest.mark.asyncio
async def test_full_app_artifact_lookup_failure_releases_without_private_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "TASK-646-ARTIFACT-PRIVATE-SENTINEL"
    app = TldwCli()
    service = InjectedChatbookService(
        exact={"77": RuntimeError(sentinel)},
    )
    app.local_chatbook_service = service

    async with _mounted_app(app, monkeypatch) as pilot:
        messages: list[str] = []
        sink_id = logger.add(messages.append, format="{message}")
        try:
            app.pending_handoffs.stage(
                HandoffChannel.ARTIFACT_CHATBOOK_TARGET,
                "local:chatbook:77",
            )
            app.post_message(NavigateToScreen("artifacts"))
            screen = await _wait_for_artifacts_screen(app, pilot)
            await _wait_for_artifact_context(screen, pilot)
        finally:
            logger.remove(sink_id)

        retry = app.pending_handoffs.claim(HandoffChannel.ARTIFACT_CHATBOOK_TARGET)
        assert retry is not None
        assert retry.value == "local:chatbook:77"
        assert screen._latest_chatbook_console_launch is None
        assert any("exception_category=RuntimeError" in message for message in messages)
        assert all(sentinel not in message for message in messages)
        assert all("local:chatbook:77" not in message for message in messages)


@pytest.mark.asyncio
async def test_full_app_mismatched_artifact_record_releases_without_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()
    service = InjectedChatbookService(
        exact={"77": _chatbook_record(78, "Wrong record")},
    )
    app.local_chatbook_service = service

    async with _mounted_app(app, monkeypatch) as pilot:
        app.pending_handoffs.stage(
            HandoffChannel.ARTIFACT_CHATBOOK_TARGET,
            "local:chatbook:77",
        )
        app.post_message(NavigateToScreen("artifacts"))
        screen = await _wait_for_artifacts_screen(app, pilot)
        await _wait_for_artifact_context(screen, pilot)

        retry = app.pending_handoffs.claim(HandoffChannel.ARTIFACT_CHATBOOK_TARGET)
        assert retry is not None
        assert retry.value == "local:chatbook:77"
        assert service.list_calls == []
        assert screen._latest_chatbook_console_launch is None


@pytest.mark.asyncio
async def test_full_app_artifacts_without_handoff_keeps_latest_list_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()
    service = InjectedChatbookService(
        listed=[
            {
                **_chatbook_record(1, "Older"),
                "updated_at": "2026-07-25T12:00:00+00:00",
            },
            _chatbook_record(99, "Latest"),
        ],
    )
    app.local_chatbook_service = service

    async with _mounted_app(app, monkeypatch, route="artifacts") as pilot:
        screen = await _wait_for_artifacts_screen(app, pilot)
        await _wait_for_artifact_context(screen, pilot, title="Latest")

        assert service.get_calls == []
        assert service.list_calls == [{"q": None, "limit": 25, "offset": 0}]
        assert screen._latest_chatbook_console_launch is not None
        assert (
            screen._latest_chatbook_console_launch["payload"]["target_id"]
            == "local:chatbook:99"
        )


@pytest.mark.asyncio
async def test_full_app_artifact_replacement_survives_awaited_exact_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    finish = threading.Event()
    app = TldwCli()
    service = InjectedChatbookService(
        exact={"77": _chatbook_record(77, "Requested")},
        barriers={"77": (started, finish)},
    )
    app.local_chatbook_service = service

    async with _mounted_app(app, monkeypatch) as pilot:
        app.pending_handoffs.stage(
            HandoffChannel.ARTIFACT_CHATBOOK_TARGET,
            "local:chatbook:77",
        )
        app.post_message(NavigateToScreen("artifacts"))
        screen = await _wait_for_artifacts_screen(app, pilot)
        assert await asyncio.to_thread(started.wait, 2)
        assert (
            app.pending_handoffs.claim(HandoffChannel.ARTIFACT_CHATBOOK_TARGET) is None
        )

        app.pending_handoffs.stage(
            HandoffChannel.ARTIFACT_CHATBOOK_TARGET,
            "local:chatbook:78",
        )
        real_acknowledge = app.pending_handoffs.acknowledge
        acknowledged_after_install: list[bool] = []

        def acknowledge_after_install(claim) -> bool:
            launch = screen._latest_chatbook_console_launch
            acknowledged_after_install.append(
                launch is not None
                and launch["payload"]["target_id"] == "local:chatbook:77"
            )
            return real_acknowledge(claim)

        monkeypatch.setattr(
            app.pending_handoffs,
            "acknowledge",
            acknowledge_after_install,
        )
        finish.set()
        await _wait_for_artifact_context(screen, pilot, title="Requested")

        assert acknowledged_after_install == [True]
        replacement = app.pending_handoffs.claim(
            HandoffChannel.ARTIFACT_CHATBOOK_TARGET
        )
        assert replacement is not None
        assert replacement.value == "local:chatbook:78"


@pytest.mark.asyncio
async def test_full_app_absent_artifact_service_releases_for_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()
    app.local_chatbook_service = object()

    async with _mounted_app(app, monkeypatch) as pilot:
        app.pending_handoffs.stage(
            HandoffChannel.ARTIFACT_CHATBOOK_TARGET,
            "local:chatbook:77",
        )
        app.post_message(NavigateToScreen("artifacts"))
        screen = await _wait_for_artifacts_screen(app, pilot)
        await _wait_for_artifact_context(screen, pilot)

        retry = app.pending_handoffs.claim(HandoffChannel.ARTIFACT_CHATBOOK_TARGET)
        assert retry is not None
        assert retry.value == "local:chatbook:77"
        assert "Service unavailable" in str(
            screen.query_one("#artifacts-console-unavailable").render()
        )


@pytest.mark.asyncio
async def test_full_app_artifact_unmount_releases_claim_and_ignores_late_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    finish = threading.Event()
    app = TldwCli()
    service = InjectedChatbookService(
        exact={"77": _chatbook_record(77, "Late result")},
        barriers={"77": (started, finish)},
    )
    app.local_chatbook_service = service

    async with _mounted_app(app, monkeypatch) as pilot:
        app.pending_handoffs.stage(
            HandoffChannel.ARTIFACT_CHATBOOK_TARGET,
            "local:chatbook:77",
        )
        app.post_message(NavigateToScreen("artifacts"))
        screen = await _wait_for_artifacts_screen(app, pilot)
        assert await asyncio.to_thread(started.wait, 2)

        app.post_message(NavigateToScreen("home"))
        for _ in range(300):
            if not isinstance(app.screen, ArtifactsScreen):
                break
            await pilot.pause(0.01)
        assert not isinstance(app.screen, ArtifactsScreen)

        retry = app.pending_handoffs.claim(HandoffChannel.ARTIFACT_CHATBOOK_TARGET)
        assert retry is not None
        finish.set()
        await pilot.pause(0.05)
        assert screen._latest_chatbook_console_launch is None


@pytest.mark.asyncio
async def test_full_app_artifact_restart_releases_old_claim_and_rejects_late_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_started = threading.Event()
    first_finish = threading.Event()
    second_started = threading.Event()
    second_finish = threading.Event()
    app = TldwCli()
    service = InjectedChatbookService(
        exact={
            "77": _chatbook_record(77, "Old result"),
            "78": _chatbook_record(78, "Replacement"),
        },
        barriers={
            "77": (first_started, first_finish),
            "78": (second_started, second_finish),
        },
    )
    app.local_chatbook_service = service

    async with _mounted_app(app, monkeypatch) as pilot:
        app.pending_handoffs.stage(
            HandoffChannel.ARTIFACT_CHATBOOK_TARGET,
            "local:chatbook:77",
        )
        app.post_message(NavigateToScreen("artifacts"))
        screen = await _wait_for_artifacts_screen(app, pilot)
        assert await asyncio.to_thread(first_started.wait, 2)

        app.pending_handoffs.stage(
            HandoffChannel.ARTIFACT_CHATBOOK_TARGET,
            "local:chatbook:78",
        )
        screen._start_chatbook_refresh()
        assert await asyncio.to_thread(second_started.wait, 2)
        assert (
            app.pending_handoffs.claim(HandoffChannel.ARTIFACT_CHATBOOK_TARGET) is None
        )

        second_finish.set()
        await _wait_for_artifact_context(screen, pilot, title="Replacement")
        first_finish.set()
        await pilot.pause(0.05)

        assert screen._latest_chatbook_console_launch is not None
        assert screen._latest_chatbook_console_launch["title"] == "Replacement"
        assert (
            app.pending_handoffs.claim(HandoffChannel.ARTIFACT_CHATBOOK_TARGET) is None
        )


@pytest.mark.asyncio
async def test_full_app_artifact_worker_cancellation_releases_active_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    finish = threading.Event()
    app = TldwCli()
    service = InjectedChatbookService(
        exact={"77": _chatbook_record(77, "Cancelled result")},
        barriers={"77": (started, finish)},
    )
    app.local_chatbook_service = service

    async with _mounted_app(app, monkeypatch) as pilot:
        app.pending_handoffs.stage(
            HandoffChannel.ARTIFACT_CHATBOOK_TARGET,
            "local:chatbook:77",
        )
        app.post_message(NavigateToScreen("artifacts"))
        screen = await _wait_for_artifacts_screen(app, pilot)
        assert await asyncio.to_thread(started.wait, 2)

        worker = screen._chatbook_refresh_worker
        assert worker is not None
        worker.cancel()
        retry = None
        for _ in range(300):
            retry = app.pending_handoffs.claim(HandoffChannel.ARTIFACT_CHATBOOK_TARGET)
            if retry is not None:
                break
            await pilot.pause(0.01)
        finish.set()

        assert retry is not None
        assert retry.value == "local:chatbook:77"


@pytest.mark.asyncio
async def test_full_app_acp_consumes_exact_current_target_after_real_mount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()
    app.acp_runtime_session_state = _acp_runtime_state()
    target_id = acp_session_record_id("session-1")
    assert target_id is not None

    async with _mounted_app(app, monkeypatch) as pilot:
        notifications: list[tuple[str, str]] = []
        monkeypatch.setattr(
            app,
            "notify",
            lambda message, *, severity="information", **_kwargs: notifications.append(
                (message, severity)
            ),
        )
        app.pending_handoffs.stage(HandoffChannel.ACP_SESSION_TARGET, target_id)

        app.post_message(NavigateToScreen("acp"))
        screen = await _wait_for_acp_screen(app, pilot)
        for _ in range(300):
            if (
                not app.pending_handoffs.has_pending(HandoffChannel.ACP_SESSION_TARGET)
                and notifications
            ):
                break
            await pilot.pause(0.01)

        assert screen.query_one("#acp-session-list-row").has_class(
            "acp-selected-session-row"
        )
        assert screen.query_one("#acp-detail-pane").is_mounted
        assert notifications == [
            ("Opened the current ACP session details.", "information")
        ]
        assert app.pending_handoffs.claim(HandoffChannel.ACP_SESSION_TARGET) is None


@pytest.mark.asyncio
async def test_full_app_acp_focus_preserves_newer_target_and_runtime_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TldwCli()
    original_state = _acp_runtime_state()
    app.acp_runtime_session_state = original_state
    current_target = acp_session_record_id("session-1")
    replacement_target = acp_session_record_id("session-2")
    assert current_target is not None
    assert replacement_target is not None

    async with _mounted_app(app, monkeypatch, route="acp") as pilot:
        screen = await _wait_for_acp_screen(app, pilot)
        detail = screen.query_one("#acp-detail-pane")
        scroll_calls: list[bool] = []
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

        def scroll_visible(*, animate: bool) -> None:
            scroll_calls.append(animate)
            app.pending_handoffs.stage(
                HandoffChannel.ACP_SESSION_TARGET,
                replacement_target,
            )

        monkeypatch.setattr(detail, "scroll_visible", scroll_visible)
        app.pending_handoffs.stage(
            HandoffChannel.ACP_SESSION_TARGET,
            current_target,
        )

        screen._consume_pending_session_target()

        assert scroll_calls == [False]
        assert screen.query_one("#acp-session-list-row").has_class(
            "acp-selected-session-row"
        )
        assert app.acp_runtime_session_state is original_state
        assert navigation == []
        assert notifications == [
            ("Opened the current ACP session details.", "information")
        ]
        replacement = app.pending_handoffs.claim(HandoffChannel.ACP_SESSION_TARGET)
        assert replacement is not None
        assert replacement.value == replacement_target


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state",
    [
        ACPRuntimeSessionState(
            runtime_id="runtime-1",
            runtime_label="Production runtime",
        ),
        _acp_runtime_state("different-session"),
    ],
    ids=["no-current-session", "different-current-session"],
)
async def test_full_app_acp_stale_target_acknowledges_with_current_only_recovery(
    monkeypatch: pytest.MonkeyPatch,
    state: ACPRuntimeSessionState,
) -> None:
    app = TldwCli()
    app.acp_runtime_session_state = state
    requested_target = acp_session_record_id("requested-session")
    assert requested_target is not None

    async with _mounted_app(app, monkeypatch, route="acp") as pilot:
        screen = await _wait_for_acp_screen(app, pilot)
        notifications: list[tuple[str, str]] = []
        navigation: list[NavigateToScreen] = []
        _intercept_navigation(app, monkeypatch, navigation.append)
        monkeypatch.setattr(
            app,
            "notify",
            lambda message, *, severity="information", **_kwargs: notifications.append(
                (message, severity)
            ),
        )
        app.pending_handoffs.stage(
            HandoffChannel.ACP_SESSION_TARGET,
            requested_target,
        )

        screen._consume_pending_session_target()

        assert app.acp_runtime_session_state is state
        assert navigation == []
        assert notifications == [
            (
                "Only the current ACP runtime session is available. "
                "Return to Console and choose it again.",
                "warning",
            )
        ]
        probe_target = acp_session_record_id("probe-session")
        assert probe_target is not None
        app.pending_handoffs.stage(
            HandoffChannel.ACP_SESSION_TARGET,
            probe_target,
        )
        probe = app.pending_handoffs.claim(HandoffChannel.ACP_SESSION_TARGET)
        assert probe is not None
        assert probe.value == probe_target
        assert app.pending_handoffs.acknowledge(probe) is True


@pytest.mark.asyncio
async def test_full_app_acp_missing_detail_is_terminal_and_private(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "TASK-646-ACP-PRIVATE-SENTINEL"
    app = TldwCli()
    state = _acp_runtime_state(sentinel)
    app.acp_runtime_session_state = state
    requested_target = acp_session_record_id(sentinel)
    assert requested_target is not None

    async with _mounted_app(app, monkeypatch, route="acp") as pilot:
        screen = await _wait_for_acp_screen(app, pilot)
        real_query_one = screen.query_one
        notifications: list[tuple[str, str]] = []
        monkeypatch.setattr(
            app,
            "notify",
            lambda message, *, severity="information", **_kwargs: notifications.append(
                (message, severity)
            ),
        )

        def query_one(selector, *args, **kwargs):
            if selector == "#acp-detail-pane":
                raise RuntimeError("detail focus unavailable")
            return real_query_one(selector, *args, **kwargs)

        monkeypatch.setattr(screen, "query_one", query_one)
        app.pending_handoffs.stage(
            HandoffChannel.ACP_SESSION_TARGET,
            requested_target,
        )
        messages: list[str] = []
        sink_id = logger.add(messages.append, format="{message}")
        try:
            screen._consume_pending_session_target()
        finally:
            logger.remove(sink_id)

        assert app.acp_runtime_session_state is state
        assert notifications == [
            (
                "Only the current ACP runtime session is available. "
                "Return to Console and choose it again.",
                "warning",
            )
        ]
        probe_target = acp_session_record_id("probe-session")
        assert probe_target is not None
        app.pending_handoffs.stage(
            HandoffChannel.ACP_SESSION_TARGET,
            probe_target,
        )
        probe = app.pending_handoffs.claim(HandoffChannel.ACP_SESSION_TARGET)
        assert probe is not None
        assert probe.value == probe_target
        assert app.pending_handoffs.acknowledge(probe) is True
        assert all(sentinel not in message for message in messages)
