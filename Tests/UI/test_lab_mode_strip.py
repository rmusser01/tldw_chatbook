"""Lab mode strip: presence, active state, and cross-screen navigation.

The Lab destination seats three screens -- Models (llm), Speech (stts),
Evals (evals). Each mounts a LabModeStrip under its DestinationHeader whose
chips post NavigateToScreen for the other modes' routes; the chip for the
owning screen is highlighted and inert. Before the strip existed, the Evals
inline workbench was unreachable from the rest of the shell.
"""

from __future__ import annotations

import time
from importlib import import_module
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import pytest
from textual import on
from textual.app import App
from textual.widgets import Button

from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Screens.lab_mode_strip import LAB_MODE_CHIPS, LabModeStrip
from tldw_chatbook.UI.Workbench.workbench_widgets import DestinationHeader
from Tests.UI.test_screen_navigation import _build_test_app


# (route, owning screen class name, module path, active chip id)
_LAB_SCREENS = (
    ("llm", "LLMScreen", "tldw_chatbook.UI.Screens.llm_screen", "lab-mode-models"),
    ("stts", "STTSScreen", "tldw_chatbook.UI.Screens.stts_screen", "lab-mode-speech"),
    ("evals", "EvalsScreen", "tldw_chatbook.UI.Screens.evals_screen", "lab-mode-evals"),
)


class _StripHarness(App[None]):
    """Bare harness mounting only the strip; records navigation requests."""

    def __init__(self, active_route: str):
        super().__init__()
        self._active_route = active_route
        self.navigated: list[str] = []

    def compose(self):
        yield LabModeStrip(active_route=self._active_route, id="lab-mode-strip")

    @on(NavigateToScreen)
    def _record_navigation(self, message: NavigateToScreen) -> None:
        self.navigated.append(message.screen_name)


@pytest.mark.parametrize(("route", "class_name", "module", "active_chip"), _LAB_SCREENS)
def test_lab_screen_composes_mode_strip_under_destination_header(
    route, class_name, module, active_chip
):
    app = _build_test_app()
    screen = getattr(import_module(module), class_name)(app)

    widgets = list(screen.compose_content())

    assert isinstance(widgets[0], DestinationHeader), route
    strip = widgets[1]
    assert isinstance(strip, LabModeStrip), route
    assert strip.id == "lab-mode-strip"
    assert strip.active_route == route


@pytest.mark.asyncio
@pytest.mark.parametrize(("route", "class_name", "module", "active_chip"), _LAB_SCREENS)
async def test_active_chip_reflects_current_screen(
    route, class_name, module, active_chip
):
    app = _StripHarness(route)

    async with app.run_test() as pilot:
        await pilot.pause()
        for mode_id, _label, mode_route, _tooltip in LAB_MODE_CHIPS:
            chip = app.query_one(f"#lab-mode-{mode_id}", Button)
            assert chip.has_class("is-active") == (mode_route == route), mode_id
        # Exactly one chip is active, and it is the owning screen's.
        active = [
            button.id
            for button in app.query(".lab-mode-chip")
            if button.has_class("is-active")
        ]
        assert active == [active_chip]


@pytest.mark.asyncio
async def test_inactive_chips_post_navigation_to_their_routes():
    app = _StripHarness("llm")

    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#lab-mode-evals", Button).press()
        await pilot.pause()
        app.query_one("#lab-mode-speech", Button).press()
        await pilot.pause()

    assert app.navigated == ["evals", "stts"]


@pytest.mark.asyncio
async def test_active_chip_press_is_a_noop():
    app = _StripHarness("evals")

    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#lab-mode-evals", Button).press()
        await pilot.pause()

    assert app.navigated == []


def _prepare_clean_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for env_var, path_name in (
        ("HOME", "home"),
        ("XDG_CONFIG_HOME", "xdg-config"),
        ("XDG_DATA_HOME", "xdg-data"),
        ("XDG_CACHE_HOME", "xdg-cache"),
    ):
        path = tmp_path / path_name
        path.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(env_var, str(path))


def _test_cli_setting(section: str, key: str, default=None):
    if section == "splash_screen" and key == "enabled":
        return False
    return default


async def _wait_until(
    pilot,
    condition: Callable[[], bool],
    *,
    timeout_seconds: float = 10.0,
    interval_seconds: float = 0.05,
    context: str = "",
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if condition():
            return
        await pilot.pause(interval_seconds)
    if condition():
        return
    context_suffix = f" for {context}" if context else ""
    raise AssertionError(
        f"condition was not met within {timeout_seconds:.1f}s{context_suffix}"
    )


@pytest.mark.asyncio
async def test_lab_route_and_mode_strip_navigate_the_real_shell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """End to end: NavigateToScreen("lab") seats Models, and the strip moves
    between Lab screens while the Lab nav button stays boxed."""
    from tldw_chatbook.UI.Screens.evals_screen import EvalsScreen
    from tldw_chatbook.UI.Screens.home_screen import HomeScreen
    from tldw_chatbook.UI.Screens.llm_screen import LLMScreen

    _prepare_clean_environment(monkeypatch, tmp_path)
    # Isolate the navigation test from the Models screen's HuggingFace
    # widgets: their init-fired reactives and scan/download workers schedule
    # deferred DOM updates (call_later/thread completion) that race child
    # mounting and screen switches under run_test -- a pre-existing family of
    # races in those widgets, unrelated to shell navigation.
    from tldw_chatbook.Widgets.HuggingFace.download_manager import DownloadManager
    from tldw_chatbook.Widgets.HuggingFace.local_models_widget import LocalModelsWidget
    from tldw_chatbook.Widgets.HuggingFace.model_search_widget import ModelSearchWidget

    async def _noop_async_update(self, *args):
        return None

    monkeypatch.setattr(ModelSearchWidget, "perform_search", lambda self: None)
    monkeypatch.setattr(ModelSearchWidget, "_update_results_list", _noop_async_update)
    monkeypatch.setattr(LocalModelsWidget, "scan_models", lambda self: None)
    monkeypatch.setattr(LocalModelsWidget, "_refresh_model_list", _noop_async_update)
    monkeypatch.setattr(LocalModelsWidget, "_update_summary", lambda self: None)
    monkeypatch.setattr(DownloadManager, "_refresh_downloads_list", _noop_async_update)
    monkeypatch.setattr(DownloadManager, "_update_summary", lambda self: None)
    app = _build_test_app()
    app.app_config["_first_run"] = True
    app._initial_tab_value = "chat"

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(160, 45)) as pilot:
            await _wait_until(
                pilot,
                lambda: isinstance(app.screen, HomeScreen),
                context="initial home",
            )

            # The critique repro: the "lab" destination id must seat Lab's
            # primary route (Models) instead of leaving the app on MCP.
            app.post_message(NavigateToScreen("lab"))
            await _wait_until(
                pilot, lambda: isinstance(app.screen, LLMScreen), context="lab -> llm"
            )
            assert app.screen.query_one("#lab-mode-models", Button).has_class(
                "is-active"
            )
            assert app.screen.query_one("#nav-lab", Button).has_class("is-active")

            app.screen.query_one("#lab-mode-evals", Button).press()
            await _wait_until(
                pilot,
                lambda: isinstance(app.screen, EvalsScreen),
                context="chip -> evals",
            )
            assert app.screen.query_one("#lab-mode-evals", Button).has_class(
                "is-active"
            )
            assert app.screen.query_one("#nav-lab", Button).has_class("is-active")

            app.screen.query_one("#lab-mode-models", Button).press()
            await _wait_until(
                pilot, lambda: isinstance(app.screen, LLMScreen), context="chip -> llm"
            )
            assert app.screen.query_one("#lab-mode-models", Button).has_class(
                "is-active"
            )
            assert app.screen.query_one("#nav-lab", Button).has_class("is-active")
