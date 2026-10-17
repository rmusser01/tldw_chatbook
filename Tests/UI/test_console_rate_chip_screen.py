"""Screen-level tests for the Console rate chip (TASK-35100).

Wires together the tracker math (Tests/Chat/test_console_token_rate.py),
the chip widget (Tests/UI/test_console_rate_chip.py), and the real
ChatScreen poll path: a stub gateway streams a reply in two chunks with a
mid-stream stall, and the assertions ride the screen's own 0.2s sync --
live rate visible while the stream holds, then the provider-accurate
settled figure once usage attaches.

Harness: mirrors ``Tests/UI/test_console_cost_chip_screen.py`` -- a real
mounted ``ChatScreen`` behind ``ConsoleHarness``, a stub provider gateway
injected via ``app.console_provider_gateway_factory`` (never a real
network call).
"""

from __future__ import annotations

import asyncio

import pytest
from textual.widgets import Button

from Tests.UI.app_factory import attach_chachanotes_db
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_console_native_chat_flow import (
    _build_console_send_test_app as _build_test_app,
)
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)

from tldw_chatbook.Widgets.Console import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_status_chips import ConsoleStatusText
from tldw_chatbook.config import load_settings, save_settings_to_cli_config
from Tests.console_provider_doubles import provider_resolution

pytestmark = pytest.mark.bootstrap_profile

_USAGE = {
    "input_tokens": 1000,
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 0,
    "output_tokens": 200,
}


def _configure_anthropic_ready_console(app, model: str = "claude-sonnet-4-6") -> None:
    assert save_settings_to_cli_config(
        {
            "chat_defaults": {"provider": "anthropic", "model": model},
            "api_settings.anthropic": {"api_key": "test-anthropic-key"},
        }
    )
    app.app_config = load_settings()
    app.chat_api_provider_value = "anthropic"
    app.chat_api_model_value = model


class _TwoChunkAnthropicGateway:
    """Stub gateway: one chunk, hold until released, then the rest.

    The hold is the streaming window: the 0.2s transcript poll keeps
    ticking while the run is active, so the tracker accumulates real
    observations without any sleeps of its own.
    """

    def __init__(self, usage_payload: dict, *, first: str, rest: str) -> None:
        self._usage_payload = usage_payload
        self._first = first
        self._rest = rest
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def cached_context_window(self, settings):
        from tldw_chatbook.Utils.token_counter import resolve_context_window

        return resolve_context_window(settings.provider, settings.model or "")

    async def resolve_context_window(self, settings):
        return self.cached_context_window(settings)

    async def resolve_for_send(self, selection):
        return provider_resolution(
            provider="anthropic",
            base_url=selection.base_url or "",
            model=selection.explicit_model
            or selection.configured_model
            or "claude-sonnet-4-6",
            ready=True,
            visible_copy="",
            prompt_caching=True,
        )

    async def stream_chat(self, resolution, messages, **kwargs):
        signals = kwargs.get("signals")
        if signals is not None:
            signals.record_usage_payload(self._usage_payload)
        yield self._first
        self.started.set()
        await self.release.wait()
        yield self._rest


@pytest.mark.asyncio
async def test_rate_rides_run_chip_live_then_settles_to_badge():
    """Redesign 2026-10-16: streaming widens nothing -- the live rate is a
    fragment on the existing run chip; only after the run goes terminal
    does the compact settled badge appear in the vacated slot."""
    gateway = _TwoChunkAnthropicGateway(
        _USAGE, first="The answer begins, ", rest="and finishes here."
    )
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_anthropic_ready_console(app)
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)
    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("count my tokens")
        console.query_one("#console-send-message", Button).press()

        # Streaming window: the run segment carries the live rate inside the
        # one-line strip; no separate rate slot exists while streaming.
        await asyncio.wait_for(gateway.started.wait(), timeout=4.0)
        line = console.query_one("#console-status-text", ConsoleStatusText)

        def line_text() -> str:
            renderable = line.renderable
            return (
                renderable.plain
                if hasattr(renderable, "plain")
                else str(line.render())
            )

        deadline = asyncio.get_event_loop().time() + 4.0
        while asyncio.get_event_loop().time() < deadline:
            if "tok/s" in line_text():
                break
            await pilot.pause(0.05)
        else:
            raise AssertionError(
                f"status line never carried a live tok/s rate: {line_text()!r}"
            )
        assert "tok/s avg" not in line_text()

        # Settle: the stream finishes, usage attaches, and the line resolves
        # to the provider-accurate overall figure, marked "avg".
        gateway.release.set()
        await pilot.pause(0.5)
        await console._sync_native_console_chat_ui()
        badge_deadline = asyncio.get_event_loop().time() + 4.0
        while asyncio.get_event_loop().time() < badge_deadline:
            await console._sync_native_console_chat_ui()
            if "tok/s avg" in line_text():
                break
            await pilot.pause(0.05)
        else:
            raise AssertionError(
                f"settled avg figure never showed: {line_text()!r}"
            )
        assert "95 tok" not in line_text()  # detail lives in the destinations


@pytest.mark.asyncio
async def test_rate_readout_config_off_hides_and_resets():
    gateway = _TwoChunkAnthropicGateway(
        _USAGE, first="The answer begins, ", rest="and finishes here."
    )
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_anthropic_ready_console(app)
    app.app_config.setdefault("console", {})["show_tokens_per_second"] = False
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)
    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("count my tokens")
        console.query_one("#console-send-message", Button).press()

        await asyncio.wait_for(gateway.started.wait(), timeout=4.0)
        gateway.release.set()
        await pilot.pause(0.5)
        await console._sync_native_console_chat_ui()
        await pilot.pause(0.2)
        await console._sync_native_console_chat_ui()
        line = console.query_one("#console-status-text", ConsoleStatusText)
        renderable = line.renderable
        text = (
            renderable.plain if hasattr(renderable, "plain") else str(line.render())
        )
        assert "tok/s" not in text
