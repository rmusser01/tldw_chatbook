"""ModelCatalogConsentModal dismisses with the user's allow/deny choice."""

from __future__ import annotations

import asyncio
import time
from types import MethodType
from unittest.mock import MagicMock

import pytest
from textual.app import App
from textual.screen import Screen

from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.UI.Screens.home_screen import HomeScreen
from tldw_chatbook.UI.Screens.model_catalog_consent import ModelCatalogConsentModal


class _ConsentHost(App):
    def __init__(self):
        super().__init__()
        self.results = []

    def on_mount(self):
        self.push_screen(
            ModelCatalogConsentModal(), lambda result: self.results.append(result)
        )


@pytest.mark.asyncio
async def test_allow_button_dismisses_true():
    app = _ConsentHost()
    async with app.run_test() as pilot:
        await pilot.click("#model-catalog-consent-allow")
        await pilot.pause()
    assert app.results == [True]


@pytest.mark.asyncio
async def test_deny_button_dismisses_false():
    app = _ConsentHost()
    async with app.run_test() as pilot:
        await pilot.click("#model-catalog-consent-deny")
        await pilot.pause()
    assert app.results == [False]


@pytest.mark.asyncio
async def test_escape_dismisses_false():
    app = _ConsentHost()
    async with app.run_test() as pilot:
        await pilot.press("escape")
        await pilot.pause()
    assert app.results == [False]


@pytest.mark.asyncio
async def test_app_push_suppresses_modal_in_headless_runs():
    """run_test() is headless: no user can answer, so nothing is pushed.

    This is the guard that keeps full-app UI tests (GGUF source modes,
    first-run flows, ...) free of an interleaved consent dialog.
    """
    from tldw_chatbook.app import TldwCli

    class HostApp(App):
        pass

    app = HostApp()
    async with app.run_test() as pilot:
        TldwCli._push_model_catalog_consent_modal(app)
        await pilot.pause()
        assert not isinstance(app.screen, ModelCatalogConsentModal)


@pytest.mark.asyncio
async def test_splash_finishes_before_catalog_consent_and_deny_returns_home():
    """Consent stays topmost after splash startup and dismisses to Home."""
    app = _build_test_app(configured_default="home")
    project_skills_offer = MagicMock(name="project_skills_offer")
    app._maybe_offer_project_skills_import = project_skills_offer

    observed: list[tuple[bool, tuple[type[Screen], ...]]] = []

    def observe_stack() -> tuple[type[Screen], ...]:
        stack = tuple(type(screen) for screen in app.screen_stack)
        snapshot = (bool(app.splash_screen_active), stack)
        if not observed or observed[-1] != snapshot:
            observed.append(snapshot)
        return stack

    def push_catalog_consent(self) -> None:
        observe_stack()
        self.push_screen(
            ModelCatalogConsentModal(), self._handle_model_catalog_consent
        )

    app._push_model_catalog_consent_modal = MethodType(push_catalog_consent, app)

    async with app.run_test(size=(120, 40)) as pilot:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            stack = observe_stack()
            if any(screen is HomeScreen for screen in stack) and any(
                screen is ModelCatalogConsentModal for screen in stack
            ):
                break
            await pilot.pause(0.01)
        else:
            pytest.fail(
                "startup did not mount both Home and model-catalog consent; "
                f"observed={[(active, [item.__name__ for item in stack]) for active, stack in observed]}"
            )

        final_stack = tuple(app.screen_stack)
        assert isinstance(final_stack[-2], HomeScreen), (
            "Home must be immediately below consent after startup; "
            f"stack={[type(screen).__name__ for screen in final_stack]}"
        )
        assert isinstance(final_stack[-1], ModelCatalogConsentModal), (
            "consent must remain the topmost actionable screen; "
            f"stack={[type(screen).__name__ for screen in final_stack]}"
        )

        first_consent = next(
            index
            for index, (_, stack) in enumerate(observed)
            if ModelCatalogConsentModal in stack
        )
        assert any(
            active and stack == (Screen,)
            for active, stack in observed[:first_consent]
        ), f"splash/default Screen was not observed before consent: {observed}"
        assert not observed[first_consent][0], (
            "consent appeared before the splash completed; "
            f"observed={observed}"
        )
        assert not any(
            ModelCatalogConsentModal in stack
            and stack[-1] is not ModelCatalogConsentModal
            for _, stack in observed
        ), f"consent was buried by a later startup screen: {observed}"

        home = final_stack[-2]
        assert await pilot.click("#model-catalog-consent-deny") is True

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            observe_stack()
            if app.screen is home:
                break
            await pilot.pause(0.01)
        else:
            pytest.fail(
                "Deny did not dismiss consent back to the same HomeScreen; "
                f"stack={[type(screen).__name__ for screen in app.screen_stack]}"
            )

        assert app.screen is home
        assert home.is_mounted
        assert home.query("#home-triage-grid")
        project_skills_offer.assert_not_called()


@pytest.mark.asyncio
async def test_first_run_completion_finishes_chat_navigation_before_consent():
    """Completed first-run navigation finishes before consent can be offered."""
    from tldw_chatbook.app import TldwCli

    class CompletionHost:
        current_tab = "home"
        focus_mode = False

        def __init__(self) -> None:
            self.events: list[str] = []
            self.worker_coroutines = []

        async def handle_screen_navigation(self, _message) -> None:
            self.events.append("navigation-start")
            await asyncio.sleep(0)
            self.events.append("navigation-finish")

        def run_worker(self, work, **_kwargs) -> None:
            self.worker_coroutines.append(work)

        def _schedule_startup_model_catalog_refresh(self, **_kwargs) -> None:
            self.events.append("consent")

        def post_message(self, _message) -> None:
            raise AssertionError(
                "first-run completion must not post navigation; "
                f"events={self.events!r}, workers={len(self.worker_coroutines)}"
            )

    host = CompletionHost()
    try:
        TldwCli._handle_first_run_wizard_result(
            host,
            {"completed": True, "exit_route": "chat", "exit_context": None},
        )

        assert len(host.worker_coroutines) == 1
        await host.worker_coroutines[0]
        assert host.events == [
            "navigation-start",
            "navigation-finish",
            "consent",
        ]
    finally:
        for worker in host.worker_coroutines:
            worker.close()


@pytest.mark.asyncio
async def test_first_run_navigation_failure_still_schedules_consent_after_attempt():
    """A failed first-run route still releases the pending consent offer."""
    from tldw_chatbook.app import TldwCli

    class FailingNavigationHost:
        current_tab = "home"
        focus_mode = False

        def __init__(self) -> None:
            self.events: list[str] = []
            self.worker_coroutines = []
            self._schedule_startup_model_catalog_refresh = MagicMock(
                side_effect=lambda **_kwargs: self.events.append("consent")
            )

        async def handle_screen_navigation(self, _message) -> None:
            self.events.append("navigation-attempt")
            raise RuntimeError("injected navigation failure")

        def run_worker(self, work, **_kwargs) -> None:
            self.worker_coroutines.append(work)

        def post_message(self, _message) -> None:
            raise AssertionError("completed navigation must use its worker")

    host = FailingNavigationHost()
    try:
        TldwCli._handle_first_run_wizard_result(
            host,
            {"completed": True, "exit_route": "chat", "exit_context": None},
        )

        assert len(host.worker_coroutines) == 1
        with pytest.raises(RuntimeError, match="injected navigation failure"):
            await host.worker_coroutines[0]

        host._schedule_startup_model_catalog_refresh.assert_called_once_with(
            after_setup_completion=True
        )
        assert host.events == ["navigation-attempt", "consent"]
    finally:
        for worker in host.worker_coroutines:
            worker.close()


@pytest.mark.asyncio
async def test_first_run_navigation_cancellation_does_not_schedule_consent():
    """Cancelling first-run navigation also cancels its pending consent offer."""
    from tldw_chatbook.app import TldwCli

    class CancelledNavigationHost:
        current_tab = "home"
        focus_mode = False

        def __init__(self) -> None:
            self.worker_coroutines = []
            self._schedule_startup_model_catalog_refresh = MagicMock()

        async def handle_screen_navigation(self, _message) -> None:
            raise asyncio.CancelledError

        def run_worker(self, work, **_kwargs) -> None:
            self.worker_coroutines.append(work)

        def post_message(self, _message) -> None:
            raise AssertionError("completed navigation must use its worker")

    host = CancelledNavigationHost()
    try:
        TldwCli._handle_first_run_wizard_result(
            host,
            {"completed": True, "exit_route": "chat", "exit_context": None},
        )

        assert len(host.worker_coroutines) == 1
        with pytest.raises(asyncio.CancelledError):
            await host.worker_coroutines[0]

        host._schedule_startup_model_catalog_refresh.assert_not_called()
    finally:
        for worker in host.worker_coroutines:
            worker.close()
