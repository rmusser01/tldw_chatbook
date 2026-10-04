"""Real Console recovery layout must not rebuild CSS for unchanged speech state."""

from __future__ import annotations

import pytest
from textual.app import ComposeResult
from textual.widgets import Button
from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from tldw_chatbook.Chat.console_display_state import ConsoleControlState
from tldw_chatbook.Widgets.Console.console_control_bar import ConsoleControlBar

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_internals_decomposition import (
    _configure_native_ready_console,
)
from Tests.UI.test_console_narrow_layout import _compositor_text

pytestmark = pytest.mark.bootstrap_profile


class ControlBarHarness(ConsolidatedCSSApp):
    """Mount the complete real bar under the production stylesheet."""

    CSS_PATH = [str(path) for path in APP_STYLESHEETS]

    def __init__(self, app_instance):
        super().__init__()
        self.bar = ConsoleControlBar(
            ConsoleControlState.from_values(),
            app_instance,
            id="console-control-bar",
            classes="console-control-bar",
        )

    def compose(self) -> ComposeResult:
        yield self.bar


def _ready_host():
    from tldw_chatbook.config import save_settings_to_cli_config

    assert save_settings_to_cli_config(
        {
            "splash_screen": {"enabled": False},
            "first_run": {"setup_completed": True},
            "_first_run": {"setup_completed": True},
            "chat_defaults": {"provider": "llama_cpp", "model": "local-model"},
            "api_settings": {
                "llama_cpp": {
                    "api_url": "http://127.0.0.1:9099",
                    "model": "local-model",
                }
            },
        }
    )
    app = _build_test_app()
    _configure_native_ready_console(app)
    return ControlBarHarness(app)


@pytest.mark.asyncio
@pytest.mark.parametrize("visible", [False, True])
@private_profile_test
async def test_repeated_speech_sync_preserves_painted_geometry_without_css_work(
    monkeypatch,
    request,
    visible: bool,
) -> None:
    host = _ready_host()
    async with host.run_test(size=(90, 30)) as pilot:
        await pilot.pause()
        bar = host.bar
        bar.sync_auto_speak(enabled=True, paused=visible, retry_available=True)
        await pilot.pause()
        expected_region = bar.region
        assert expected_region.height == (2 if visible else 1)
        retry = bar.query_one("#console-auto-speak-retry", Button)
        if visible:
            assert retry.region.width > 0
            assert host.get_widget_at(*retry.region.center)[0] is retry
        calls = []
        stylesheet_type = type(host.stylesheet)
        original = stylesheet_type.apply

        def observed_apply(stylesheet, node, *args, **kwargs):
            calls.append(node)
            return original(stylesheet, node, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(stylesheet_type, "apply", observed_apply)
            for _ in range(8):
                bar.sync_auto_speak(enabled=True, paused=visible, retry_available=True)
            assert calls == [], "unchanged speech state re-applied stylesheet rules"
        await pilot.pause()
        assert bar.region == expected_region
        assert retry.display is visible
        if visible:
            painted = _compositor_text(host.export_screenshot(simplify=True))
            assert "Retry speech" in painted
            assert "Resume auto-speak" in painted


@pytest.mark.asyncio
@private_profile_test
async def test_recovery_height_corrects_external_class_and_inline_constraint_changes(
    request,
):
    host = _ready_host()
    async with host.run_test(size=(90, 30)) as pilot:
        await pilot.pause()
        bar = host.bar
        for visible in (True, False):
            bar.add_class("h-3")
            bar.styles.height = 7
            bar.styles.min_height = 7
            bar.styles.max_height = 7
            bar.sync_auto_speak(enabled=True, paused=visible, retry_available=True)
            await pilot.pause()
            assert bar.region.height == (2 if visible else 1)
            assert {name for name in bar.classes if name.startswith("h-")} == {
                "h-2" if visible else "h-1"
            }
            assert not bar.styles.inline.has_rule("height")


def test_recovery_height_repairs_percent_constraints_with_equal_numeric_values():
    from textual.css.scalar import Scalar

    bar = ConsoleControlBar(ConsoleControlState.from_values(), object())
    bar.styles.min_height = "1%"
    bar.styles.max_height = "1%"
    bar.sync_auto_speak(enabled=False, paused=False)
    assert bar.styles.min_height == Scalar.from_number(1)
    assert bar.styles.max_height == Scalar.from_number(1)
