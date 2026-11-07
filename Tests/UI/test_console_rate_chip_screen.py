"""Screen-level tests for the Console rate readout (TASK-35100).

Wires together the tracker math (Tests/Chat/test_console_token_rate.py),
the strip segment (Tests/UI/test_console_rate_chip.py), and the real
ChatScreen poll path: a stub gateway streams a reply at a steady cadence,
and the assertions ride the screen's own 0.2s sync -- the live rate rides
the run segment while tokens flow (stale decayed fragments are
suppressed), then the provider-accurate settled figure lands once usage
attaches.

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

_REPLY = (
    "Sure -- watch the status line while this reply streams: the run "
    "segment carries my live tokens-per-second rate, and when the reply "
    "finishes it settles to the overall average figure."
)
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


class _SteadyAnthropicGateway:
    """Stub gateway: a steady multi-chunk stream (no stall).

    The fragment floor (review follow-up) suppresses decayed sub-floor
    rates, so the live-fragment assertion needs a stream that keeps
    tokens flowing -- this one emits ~20-char chunks on a fixed cadence,
    the same shape as the demo recorder.
    """

    def __init__(self, usage_payload: dict, *, reply: str) -> None:
        self._usage_payload = usage_payload
        self._reply = reply
        self.started = asyncio.Event()
        self.done = asyncio.Event()

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
        from tldw_chatbook.Utils.token_counter import estimate_tokens

        signals = kwargs.get("signals")
        if signals is not None:
            signals.record_usage_payload(
                {**self._usage_payload,
                 "output_tokens": estimate_tokens(self._reply)}
            )
        pieces = [self._reply[i : i + 20] for i in range(0, len(self._reply), 20)]
        for piece in pieces:
            yield piece
            self.started.set()
            await asyncio.sleep(0.09)
        self.done.set()



@pytest.mark.asyncio
async def test_rate_rides_run_chip_live_then_settles_to_badge():
    """Redesign 2026-10-16: streaming widens nothing -- the live rate is a
    fragment on the existing run chip; only after the run goes terminal
    does the compact settled badge appear in the vacated slot."""
    gateway = _SteadyAnthropicGateway(_USAGE, reply=_REPLY)
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

        # Settle: the stream finishes on its own, usage attaches, and the
        # line resolves to the provider-accurate overall figure ("avg").
        await pilot.pause(0.6)
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
    gateway = _SteadyAnthropicGateway(_USAGE, reply=_REPLY)
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
        await pilot.pause(0.8)
        await console._sync_native_console_chat_ui()
        await pilot.pause(0.2)
        await console._sync_native_console_chat_ui()
        line = console.query_one("#console-status-text", ConsoleStatusText)
        renderable = line.renderable
        text = (
            renderable.plain if hasattr(renderable, "plain") else str(line.render())
        )
        assert "tok/s" not in text
        assert console._console_rate_tracker is None


def test_rate_tracker_waits_for_an_assistant_and_resets_without_reallocation():
    from types import SimpleNamespace

    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    chips = []
    screen = SimpleNamespace(
        _console_rate_tracker=None,
        _console_rate_reading=None,
        _show_console_rate=lambda: True,
        _console_active_run_copy=lambda: "",
        query_one=lambda *args: SimpleNamespace(
            sync_rate_chip=lambda *values: chips.append(values)
        ),
    )
    ChatScreen._sync_console_rate_chip(screen, [])
    assert screen._console_rate_tracker is None
    message = SimpleNamespace(
        id="assistant",
        role=ConsoleMessageRole.ASSISTANT,
        content="A streaming answer",
        status="streaming",
        usage=None,
    )
    ChatScreen._sync_console_rate_chip(screen, [message])
    tracker = screen._console_rate_tracker
    assert tracker is not None and screen._console_rate_reading is not None
    ChatScreen._sync_console_rate_chip(screen, [])
    assert (
        screen._console_rate_tracker is tracker and screen._console_rate_reading is None
    )
    screen._show_console_rate = lambda: False
    ChatScreen._sync_console_rate_chip(screen, [message])
    assert (
        screen._console_rate_tracker is tracker and screen._console_rate_reading is None
    )
    assert chips[-1] == (False, "", "")


@pytest.mark.parametrize(
    "enabled,reading", [(False, None), (False, object()), (True, None)]
)
def test_run_copy_skips_rate_import_without_enabled_reading(
    monkeypatch, enabled, reading
):
    import builtins
    from types import SimpleNamespace

    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    rendered = []
    screen = SimpleNamespace(
        _pending_console_launch_context=None,
        _build_console_control_state=lambda context: None,
        _console_mode_summary=lambda state: "Console",
        _native_run_status_copy=lambda: None,
        _console_active_run_copy=lambda: "Streaming response.",
        _console_rate_reading=reading,
        _show_console_rate=lambda: enabled,
        _console_chat_controller=None,
        _console_chat_store=None,
        query_one=lambda *args: SimpleNamespace(
            update=lambda value: None,
            sync_run_chip=lambda *values: rendered.append(values),
        ),
    )
    original = builtins.__import__

    def without_rate_helpers(name, *args, **kwargs):
        assert not name.endswith("console_token_rate")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_rate_helpers)
    ChatScreen._sync_console_mode_bar.__wrapped__(screen)
    assert rendered == [(True, "Streaming response.")]
