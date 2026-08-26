"""TASK-2031 (live-UAT defect): provider/model chips must refresh on Apply.

The session model popover's Apply updated the session and the left-rail
Model summary, but the status chips kept showing the OLD provider/model
until a session/tab switch — the user watches "Provider: Anthropic" while
the run is actually served by the newly-applied provider.
"""
from __future__ import annotations

from dataclasses import replace

import pytest
from textual.app import App

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.widgets import Static

from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


class _ConsoleHarness(ConsolidatedCSSApp):
    def __init__(self, app_instance) -> None:
        super().__init__()
        self.app_instance = app_instance

    async def on_mount(self) -> None:
        await self.push_screen(ChatScreen(self.app_instance))


async def _wait_for(pilot, predicate, what: str, timeout: float = 8.0):
    import time as _t

    deadline = _t.monotonic() + timeout
    while _t.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        await pilot.pause(0.05)
    raise AssertionError(f"timed out waiting for {what}")


@pytest.mark.asyncio
async def test_popover_apply_refreshes_the_provider_chip():
    """Applying new session settings must refresh the provider chip
    without a session switch — the tick's control-bar sync path."""
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {"provider": "llama_cpp", "model": "local-model"},
        "api_settings": {
            "llama_cpp": {
                "api_url": "http://127.0.0.1:9099",
                "model": "local-model",
            },
            "vllm": {
                "api_url": "http://127.0.0.1:9098",
                "model": "served-model",
            },
        },
        "providers": {
            "llama_cpp": ["local-model"],
            "vLLM": ["served-model"],
        },
    }
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "local-model"
    harness = _ConsoleHarness(app)
    async with harness.run_test(size=(160, 48)) as pilot:
        await pilot.pause()
        chat_screen = harness.screen_stack[-1]
        assert isinstance(chat_screen, ChatScreen)
        chat_screen._ensure_console_chat_controller()

        from textual.css.query import NoMatches

        def chip_text() -> str:
            try:
                chip = chat_screen.query_one("#console-provider-chip", Static)
            except NoMatches:
                return ""
            return str(chip.renderable)

        await _wait_for(pilot, chip_text, "initial provider chip")

        settings = chat_screen._session._ensure_active_console_session_settings()
        chat_screen._apply_console_model_popover_result(
            replace(
                settings,
                provider="vllm",
                model="served-model",
                source="user",
            )
        )
        # The regular tick calls this; the chip must be fresh WITHOUT a
        # session switch.
        chat_screen._sync_console_control_bar()
        await pilot.pause()

        await _wait_for(
            pilot,
            lambda: "vllm" in chip_text().lower(),
            f"chip to show the applied provider (still: {chip_text()!r})",
        )
