"""Product maturity Phase 1.2 first-run walkthrough contract."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, Input, OptionList, Static

from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.app import TldwCli, setup_owns_startup_networking
from tldw_chatbook.Chat.local_server_discovery import DiscoveredLocalServer
from tldw_chatbook.LLM_Provider_Catalog.model_auto_refresh import RefreshReport
from tldw_chatbook.UI.Navigation.main_navigation import MainNavigationBar
from tldw_chatbook.UI.Navigation.shell_destinations import SHELL_DESTINATION_ORDER
from tldw_chatbook.UI.Wizards.first_run_setup_state import STEP_PROVIDER
from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
    FirstRunSetupWizard,
    ProviderStep,
    SetupWizardContainer,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path(
    "Docs/superpowers/qa/product-maturity/phase-1/2026-05-05-phase-1-2-first-run-walkthrough.md"
)
TRACKER = Path("Docs/superpowers/trackers/product-maturity-roadmap.md")
PHASE_1_README = Path("Docs/superpowers/qa/product-maturity/phase-1/README.md")
TASK = Path(
    "backlog/tasks/task-8.2 - Product-Maturity-Phase-1.2-Clean-First-Run-Launch-And-Configuration-Walkthrough.md"
)
TOP_LEVEL_DESTINATION_IDS = tuple(
    destination.destination_id for destination in SHELL_DESTINATION_ORDER
)
LOCAL_PATH_PREFIXES = (
    "/Users/",
    "/home/",
    "/var/home/",
    "/private/var/folders/",
    "C:\\Users\\",
    "C:/Users/",
)


def _text(path: Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _assert_no_local_path_prefixes(text: str) -> None:
    leaked_prefixes = [prefix for prefix in LOCAL_PATH_PREFIXES if prefix in text]
    assert not leaked_prefixes, (
        f"evidence contains local filesystem prefix(es): {leaked_prefixes}"
    )


def _screen_text(app) -> str:
    pieces: list[str] = []
    for widget in app.screen.query(Static):
        pieces.append(str(widget.renderable))
    for widget in app.screen.query(Button):
        pieces.append(str(widget.label).strip())
    return "\n".join(pieces)


def _assert_widget_in_view(widget, width: int, height: int) -> None:
    region = widget.region
    assert region.width > 0 and region.height > 0
    assert region.x >= 0 and region.y >= 0
    assert region.right <= width and region.bottom <= height


def _test_cli_setting(section: str, key: str, default=None):
    if section == "splash_screen" and key == "enabled":
        return False
    return default


def _prepare_clean_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> dict[str, str]:
    paths = {
        "HOME": tmp_path / "home",
        "XDG_CONFIG_HOME": tmp_path / "xdg-config",
        "XDG_DATA_HOME": tmp_path / "xdg-data",
        "XDG_CACHE_HOME": tmp_path / "xdg-cache",
    }
    for env_var, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(env_var, str(path))
    # ADR-020 amendment: the consent dialog self-suppresses in headless
    # runs (see _push_model_catalog_consent_modal), so full-app tests here
    # never see it; consent gating has dedicated unit tests in
    # test_app_model_catalog_wiring.py.
    return {env_var: str(path) for env_var, path in paths.items()}


def _build_clean_first_run_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _prepare_clean_environment(monkeypatch, tmp_path)
    app = _build_test_app()
    app.app_config["_first_run"] = True
    app._initial_tab_value = "chat"
    return app


def _unfinished_setup_config(*, resume_attempted: bool) -> dict[str, object]:
    return {
        "first_run": {
            "setup_started": True,
            "setup_completed": False,
            "draft_version": 1,
            "draft_track": "quick",
            "active_step_id": "model",
            "draft_values": {
                "welcome": {"track": "quick"},
                "provider": {
                    "provider_key": "openai",
                    "provider_value": "openai",
                },
            },
            "resume_attempted": resume_attempted,
        }
    }


class _CatalogRefreshScheduleHost:
    def __init__(self, app_config: dict[str, object]) -> None:
        self.app_config = app_config
        self.run_worker = MagicMock()
        self.post_message = MagicMock()
        self.call_after_refresh = MagicMock()
        self.push_screen = MagicMock()
        self._push_model_catalog_consent_modal = MagicMock()
        self._refresh_model_catalogs = AsyncMock()

    _schedule_startup_model_catalog_refresh = (
        TldwCli._schedule_startup_model_catalog_refresh
    )


def _pin_consented_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "tldw_chatbook.app.load_settings",
        lambda: {"model_catalog": {"refresh_consent_recorded": True}},
    )


@pytest.mark.parametrize(
    ("app_config", "expected_action"),
    [
        ({}, "offer"),
        (_unfinished_setup_config(resume_attempted=False), "prompt"),
        (_unfinished_setup_config(resume_attempted=True), "home"),
    ],
)
def test_setup_network_owner_actions_suppress_startup_catalog_refresh(
    app_config: dict[str, object],
    expected_action: str,
) -> None:
    from tldw_chatbook.UI.Wizards.first_run_setup_state import setup_recovery_action

    assert setup_recovery_action(app_config, {}) == expected_action
    assert setup_owns_startup_networking(app_config, {}) is True

    host = _CatalogRefreshScheduleHost(app_config)
    assert host._schedule_startup_model_catalog_refresh(environ={}) is False
    host.run_worker.assert_not_called()


def test_normal_startup_schedules_catalog_refresh_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pin_consented_settings(monkeypatch)
    host = _CatalogRefreshScheduleHost(
        {"first_run": {"setup_completed": True}}
    )

    assert host._schedule_startup_model_catalog_refresh(environ={}) is True
    assert host._schedule_startup_model_catalog_refresh(environ={}) is False

    host.run_worker.assert_called_once_with(
        host._refresh_model_catalogs,
        exclusive=True,
        group="model-catalog-refresh",
    )


def test_unconsented_startup_shows_consent_modal_instead_of_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ADR-020 amendment: no consent on file means a dialog, not network I/O."""
    monkeypatch.setattr("tldw_chatbook.app.load_settings", lambda: {})
    host = _CatalogRefreshScheduleHost(
        {"first_run": {"setup_completed": True}}
    )

    assert host._schedule_startup_model_catalog_refresh(environ={}) is True
    assert host._schedule_startup_model_catalog_refresh(environ={}) is False

    host.run_worker.assert_not_called()
    host.call_after_refresh.assert_called_once()
    assert host.call_after_refresh.call_args.args[0] is (
        host._push_model_catalog_consent_modal
    )


def test_completed_first_run_schedules_deferred_catalog_refresh_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pin_consented_settings(monkeypatch)
    host = _CatalogRefreshScheduleHost({})

    TldwCli._handle_first_run_wizard_result(
        host,
        {"completed": True, "exit_route": None, "exit_context": None},
    )
    TldwCli._handle_first_run_wizard_result(
        host,
        {"completed": True, "exit_route": None, "exit_context": None},
    )

    host.run_worker.assert_called_once_with(
        host._refresh_model_catalogs,
        exclusive=True,
        group="model-catalog-refresh",
    )
    host.post_message.assert_not_called()


@pytest.mark.parametrize(
    "result",
    [
        None,
        {"completed": False, "exit_route": None, "exit_context": None},
        {
            "completed": False,
            "exit_route": "settings",
            "exit_context": {"category": "providers-models"},
        },
    ],
)
def test_incomplete_first_run_result_does_not_schedule_catalog_refresh(
    result: dict[str, object] | None,
) -> None:
    host = _CatalogRefreshScheduleHost({})

    TldwCli._handle_first_run_wizard_result(host, result)

    host.run_worker.assert_not_called()


@pytest.mark.asyncio
async def test_clean_first_run_mount_suppresses_global_catalog_refresh(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _prepare_clean_environment(monkeypatch, tmp_path)
    app = _build_test_app(first_run_setup_completed=False)
    app.app_config["_first_run"] = True
    app._initial_tab_value = "chat"
    app.model_catalog_disk_store = MagicMock()
    catalog_refresh = AsyncMock(return_value=RefreshReport())
    app.local_llm_provider_catalog_service.refresh_stale_configured_providers = (
        catalog_refresh
    )

    with (
        patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting),
        patch.object(app, "notify", wraps=app.notify) as notify_spy,
    ):
        async with app.run_test(size=(140, 40)) as pilot:
            await _wait_until(
                pilot,
                lambda: type(app.screen).__name__ == "FirstRunSetupWizard",
            )
            await pilot.pause(0.1)

    catalog_refresh.assert_not_awaited()
    catalog_notifications = [
        call
        for call in notify_spy.call_args_list
        if call.kwargs.get("title") == "Model catalog"
    ]
    assert catalog_notifications == []


@pytest.mark.asyncio
async def test_initial_first_run_mount_has_no_unrelated_provider_catalog_calls() -> None:
    app_instance = MagicMock(app_config={})
    scope_service = MagicMock()
    scope_service.discover_models = AsyncMock()
    app_instance.llm_provider_catalog_scope_service = scope_service
    wizard = FirstRunSetupWizard(app_instance)

    class _WizardHost(App):
        def compose(self) -> ComposeResult:
            yield wizard

    host = _WizardHost()
    async with host.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.1)
        provider_step = wizard.query_one(SetupWizardContainer).query_one(ProviderStep)
        assert provider_step.selected_provider_key == ""

    scope_service.discover_models.assert_not_awaited()


@pytest.mark.parametrize("size", [(100, 32), (120, 40), (177, 45)])
@pytest.mark.asyncio
async def test_provider_connection_controls_scroll_without_displacing_footer(
    size: tuple[int, int],
) -> None:
    from unittest.mock import AsyncMock

    app_instance = MagicMock(
        app_config={},
        llm_provider_catalog_scope_service=None,
    )
    wizard = FirstRunSetupWizard(app_instance)

    class _StyledWizardHost(App):
        CSS_PATH = str(REPO_ROOT / "tldw_chatbook/css/tldw_cli_modular.tcss")

        def compose(self) -> ComposeResult:
            yield from ()

        async def on_mount(self) -> None:
            self.push_screen(wizard)

    width, height = size
    host = _StyledWizardHost()
    async with host.run_test(size=size) as pilot:
        await pilot.pause(0.2)
        container = wizard.query_one(SetupWizardContainer)
        provider_index = container._step_index_for_id(STEP_PROVIDER)
        assert provider_index is not None
        container.show_step(provider_index)
        await pilot.pause(0.1)
        provider = container.steps[provider_index]
        assert isinstance(provider, ProviderStep)
        provider._local_discover = AsyncMock(
            return_value=tuple(
                DiscoveredLocalServer(
                    "llama_cpp",
                    f"http://127.0.0.1:{8080 + index}",
                    (f"model-{index}",),
                )
                for index in range(8)
            )
        )
        provider.select_provider("llama_cpp")
        endpoint = provider.query_one("#setup-provider-endpoint", Input)
        endpoint.focus()
        await pilot.pause(0.1)
        connection = provider.query_one("#setup-provider-connection")
        assert connection.display, (
            f"connection hidden after provider selection: classes={connection.classes}, "
            f"style={connection.styles.display}"
        )
        assert connection.region.height > 0, (
            f"connection has no layout: region={connection.region}, "
            f"height={connection.styles.height}, min_height={connection.styles.min_height}, "
            f"provider_region={provider.region}, provider_display={provider.display}"
        )
        _assert_widget_in_view(endpoint, width, height)

        effective = provider.query_one("#setup-provider-effective-chat", Static)
        effective.scroll_visible()
        await pilot.pause(0.1)
        _assert_widget_in_view(effective, width, height)
        assert "v1/chat/completions" in str(effective.renderable)

        provider.query_one("#setup-provider-detect", Button).press()
        await pilot.pause(0.1)
        results = provider.query_one("#setup-provider-detection-results", OptionList)
        results.focus()
        await pilot.pause(0.1)
        _assert_widget_in_view(results, width, height)
        assert results.option_count == 10

        for selector in ("#wizard-back", "#wizard-next", "#wizard-cancel"):
            button = wizard.query_one(selector, Button)
            _assert_widget_in_view(button, width, height)
            assert button in host.screen._compositor.visible_widgets


async def _wait_until(
    pilot,
    condition: Callable[[], bool],
    *,
    timeout_seconds: float = 10.0,
    interval_seconds: float = 0.05,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if condition():
            return
        await pilot.pause(interval_seconds)
    if condition():
        return
    raise AssertionError(f"condition was not met within {timeout_seconds:.1f}s")


def _nav_button_is_clickable(app, button_id: str) -> bool:
    """True when a nav destination is where a real mouse click could land it.

    "Clickable" here means fully inside the destination strip's scroll
    viewport and not disabled -- i.e. the button is actually painted, in
    full, on screen.
    """
    try:
        button = app.screen.query_one(f"#{button_id}", Button)
        strip = app.screen.query_one("#nav-destination-strip", Horizontal)
    except Exception:
        return False
    region = button.region
    viewport = strip.region
    return (
        not button.disabled
        and region.width > 0
        and region.x >= viewport.x
        and region.right <= viewport.right
    )


async def _click_nav_destination(
    pilot,
    app,
    button_id: str,
    *,
    timeout_seconds: float = 10.0,
) -> None:
    """Bring a nav destination into view, then press it -- what a click needs.

    task-3200 (backlog task-3224) made a genuinely clip-ghosted nav button
    ``disabled``: at 140 columns not every destination fits the strip, and
    the one straddling the viewport edge is painted as blank space, so a
    real mouse click could never land on it and Enter on an invisible
    button was the exact defect that change closed. This test used to press
    ``#nav-settings`` by id with the strip still anchored at its default
    scroll position, which silently no-opped once that button was ghosted
    and then timed out waiting for a screen transition that was never going
    to happen.

    The contract this test protects is "every one of these destinations is
    reachable from the nav bar and renders its copy" -- not "a programmatic
    press works on an off-screen widget" -- so this helper reveals the
    target first, using the product's own affordance ("More ›", the pager
    the bar shows exactly when destinations overflow), and only presses
    once the button is genuinely on screen.

    The retry loop is not defensive padding: ``MainNavigationBar``'s 0.5s
    settle interval re-anchors the strip on the ACTIVE destination, so a
    paged-in destination can scroll back out before the press lands. Each
    pass re-checks and pages again if needed.
    """
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if _nav_button_is_clickable(app, button_id):
            app.screen.query_one(f"#{button_id}", Button).press()
            return
        overflow_hint = app.query_one("#nav-overflow-hint", Button)
        if not overflow_hint.display:
            # Nothing overflows: the button is already as reachable as it
            # will ever get, so press it and let the assertions speak.
            app.screen.query_one(f"#{button_id}", Button).press()
            return
        overflow_hint.press()
        await pilot.pause(0.05)
        if app.screen.__class__.__name__ == "NavOverflowMenu":
            destination_id = button_id.removeprefix("nav-")
            app.screen.query_one(f"#nav-overflow-{destination_id}", Button).press()
            return
    raise AssertionError(
        f"#{button_id} never became clickable within {timeout_seconds:.1f}s"
    )


@pytest.mark.asyncio
async def test_clean_first_run_launches_home_and_exposes_setup_orientation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app = _build_clean_first_run_app(monkeypatch, tmp_path)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(140, 40)) as pilot:
            await _wait_until(
                pilot,
                # Nav strip + docked hint mount a tick after the screen swap;
                # wait for the full chrome before asserting/clicking.
                lambda: (
                    app.current_tab == "home"
                    and app.screen.__class__.__name__ == "HomeScreen"
                    and len(app.screen.query(".nav-button"))
                    == len(TOP_LEVEL_DESTINATION_IDS)
                    and len(app.screen.query("#nav-overflow-hint")) == 1
                ),
            )

            nav_buttons = list(
                app.screen.query(MainNavigationBar).first().query(Button)
            )
            nav_ids = [button.id for button in nav_buttons]
            assert "nav-home" in nav_ids
            assert "nav-console" in nav_ids
            assert "nav-library" in nav_ids
            assert "nav-settings" in nav_ids

            home_title = app.screen.query_one("#home-canvas-title", Static)
            primary_action = app.screen.query_one("#home-primary-action", Button)
            nav_overflow_hint = app.screen.query_one("#nav-overflow-hint", Button)
            assert str(home_title.renderable).strip() == "Set up Console model"
            assert str(primary_action.label).strip() == "Set up Console model"
            assert str(primary_action.label).strip() != "Start in Console"
            assert str(nav_overflow_hint.label).strip() == "More ▾"

            for button_id, current_tab, screen_name, required_copy in (
                (
                    "nav-console",
                    "chat",
                    "ChatScreen",
                    ("Console", "Conversation"),
                ),
                (
                    "nav-library",
                    "library",
                    "LibraryScreen",
                    ("Library", "Import / Export", "Search / RAG"),
                ),
                (
                    "nav-settings",
                    "settings",
                    "SettingsScreen",
                    ("Settings", "Global preferences", "Appearance"),
                ),
            ):
                await _click_nav_destination(pilot, app, button_id)
                await _wait_until(
                    pilot,
                    lambda current_tab=current_tab, screen_name=screen_name: (
                        app.current_tab == current_tab
                        and app.screen.__class__.__name__ == screen_name
                    ),
                )
                # Some destinations (Settings categories) populate a beat
                # after the screen switch; wait for the copy, then assert.
                await _wait_until(
                    pilot,
                    lambda required_copy=required_copy: all(
                        copy in _screen_text(app) for copy in required_copy
                    ),
                )
                screen_text = _screen_text(app)
                for copy in required_copy:
                    assert copy in screen_text


@pytest.mark.parametrize("size", [(100, 32), (180, 50)])
@pytest.mark.asyncio
async def test_clean_first_run_home_survives_supported_terminal_sizes(
    size: tuple[int, int],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app = _build_clean_first_run_app(monkeypatch, tmp_path)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=size) as pilot:
            await _wait_until(
                pilot,
                # Nav strip + docked hint mount a tick after the screen swap;
                # wait for the full chrome before asserting.
                lambda: (
                    app.current_tab == "home"
                    and app.screen.__class__.__name__ == "HomeScreen"
                    and len(app.screen.query(".nav-button"))
                    == len(TOP_LEVEL_DESTINATION_IDS)
                    and len(app.screen.query("#nav-overflow-hint")) == 1
                ),
            )

            primary_action = app.screen.query_one("#home-primary-action", Button)
            nav_overflow_hint = app.screen.query_one("#nav-overflow-hint", Button)
            assert app.current_tab == "home"
            assert app.screen.__class__.__name__ == "HomeScreen"
            assert str(primary_action.label).strip() == "Set up Console model"
            assert str(nav_overflow_hint.label).strip() == "More ▾"


@pytest.mark.asyncio
async def test_fresh_config_auto_offers_wizard_over_initial_screen(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Pins the task-11 app-level contract this file's other tests
    deliberately opt OUT of: on a truly fresh config (no configured
    provider, no first_run state at all), the setup wizard must be
    auto-offered on top of whatever the initial screen is -- not silently
    skipped. Every other test in this file builds via
    _build_test_app()'s default (first_run_setup_completed=True, task-11's
    fix for the regression this auto-offer caused here) so they can assert
    against Home's content directly, exactly as they did before the wizard
    existed; this is the one test in the file that intentionally leaves
    the auto-offer live, so the new contract stays pinned at the real App
    level rather than only at the pure-function level
    (first_run_setup_state.should_offer_wizard, covered separately in
    Tests/Wizards/test_first_run_setup_wizard.py::TestAppOfferGating).
    """
    _prepare_clean_environment(monkeypatch, tmp_path)
    app = _build_test_app(first_run_setup_completed=False)
    app.app_config["_first_run"] = True
    app._initial_tab_value = "chat"

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(140, 40)) as pilot:
            await _wait_until(
                pilot,
                lambda: type(app.screen).__name__ == "FirstRunSetupWizard",
            )
            assert type(app.screen).__name__ == "FirstRunSetupWizard"
            # The initial screen is still there, underneath -- the wizard is
            # pushed ON TOP of it (per the approved design), not swapped in
            # place of it.
            assert app.current_tab == "home"


@pytest.mark.parametrize("prefix", LOCAL_PATH_PREFIXES)
def test_local_path_guard_rejects_common_home_and_temp_prefixes(prefix: str) -> None:
    with pytest.raises(AssertionError):
        _assert_no_local_path_prefixes(f"Fresh HOME: {prefix}developer/project")


def test_local_path_guard_allows_sanitized_temp_placeholders() -> None:
    _assert_no_local_path_prefixes("Fresh HOME: <tmp>/home")
