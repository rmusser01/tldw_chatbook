"""task-17653: the footer token counter is retired on Console.

The cost chip is the single token/cost surface on Console (owner ruling
2026-08-17). The footer's `#footer-token-count` used to be mounted armed
on the chat screen and one write away from duplicating the chip — any
caller of `update_token_count` (including db_status_manager's "Token
count error" path) would reveal it unconditionally.
"""

from __future__ import annotations

import pytest

from Tests.UI.test_console_native_chat_flow import (
    _configure_native_ready_console,
)
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus


@pytest.mark.asyncio
async def test_console_footer_token_counter_cannot_appear() -> None:
    """No write can surface the footer token counter on Console."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(150, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#screen-footer-status")
        footer = console.query_one("#screen-footer-status", AppFooterStatus)
        token_display = footer.query_one("#footer-token-count")

        assert footer._show_token_count is False
        assert token_display.display is False

        footer.update_token_count("🟢 Tokens: 2,700 / 128,000 (2%)")
        await pilot.pause()
        assert token_display.display is False

        footer.update_token_count("Token count error")
        await pilot.pause()
        assert token_display.display is False


@pytest.mark.asyncio
async def test_console_footer_word_count_still_works() -> None:
    """Retiring the token counter must not touch the footer's other chips."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(150, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#screen-footer-status")
        footer = console.query_one("#screen-footer-status", AppFooterStatus)

        footer.update_word_count(42)
        await pilot.pause()
        word_display = footer.query_one("#footer-word-count")
        assert "Words: 42" in str(word_display.renderable)
