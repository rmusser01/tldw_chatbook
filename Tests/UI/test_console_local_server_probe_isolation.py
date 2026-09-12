"""Mounting the Console must not probe a live localhost server (task-15111).

The defect, measured on a dev machine with an `audiocpp` server bound to
`127.0.0.1:8080`: a record-only socket shim over `Tests/UI` logged real TCP
connections to `127.0.0.1:8080` and `127.0.0.1:11434` from **every** test that
mounted `ChatScreen` with an unconfigured provider -- 56 tests apiece in the
first 6% of the suite alone.

Mechanism: an unconfigured provider makes the setup card blocking, which starts
`_maybe_start_console_local_discovery` -> `discover_local_servers`. Its
candidate list ALWAYS leads with the two well-known localhost defaults
regardless of config (`build_local_server_candidates`), and
`probe_models_endpoint` builds a real `httpx.AsyncClient` when none is
injected. Only ONE test in the suite (`test_console_local_server_discovery_card`)
ever stubbed the `console_local_server_discovery` app seam; everything else
fell through to the network.

These tests pin both halves of the fix: the probe chokepoint is neutralized for
tests (`Tests/conftest.py::_no_local_server_probes`) and the socket guard
(`Tests/network_guard.py`) proves it rather than trusting it.
"""

from __future__ import annotations

import pytest
# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp

from Tests import network_guard
from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.Chat.local_server_discovery import (
    DEFAULT_LLAMACPP_DISCOVERY_URL,
    DEFAULT_OLLAMA_DISCOVERY_URL,
    build_local_server_candidates,
    discover_local_servers,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console.console_settings_modal import (
    ConsoleModelDiscoveryIdentity,
    ConsoleUnverifiedModelDecision,
)


class ConsoleHarness(ConsolidatedCSSApp):
    """Real ChatScreen harness (mirrors test_console_setup_lock_polish)."""

    def __init__(self, app_instance):
        super().__init__()
        self.app_instance = app_instance

    async def on_mount(self) -> None:
        await self.push_screen(ChatScreen(self.app_instance))


def test_unverified_model_decision_is_not_endpoint_agnostic() -> None:
    """An approval for one canonical endpoint cannot authorize another endpoint."""
    first = ConsoleModelDiscoveryIdentity(
        provider_key="vllm",
        connection_identity=("vllm", "http://127.0.0.1:8000"),
        draft_generation=3,
    )
    second = ConsoleModelDiscoveryIdentity(
        provider_key="vllm",
        connection_identity=("vllm", "http://127.0.0.1:8001"),
        draft_generation=4,
    )

    assert ConsoleUnverifiedModelDecision(first, "custom") != (
        ConsoleUnverifiedModelDecision(second, "custom")
    )


def _blocked_provider_app():
    """Test app whose Console provider is blocked (empty OpenAI key).

    The blocked state is what makes the setup card blocking, which is what
    starts local-server discovery.
    """
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {"provider": "OpenAI", "model": "gpt-4.1-2025-04-14"},
        "api_settings": {"openai": {"api_key": ""}},
    }
    app.chat_api_provider_value = "OpenAI"
    app.chat_api_model_value = "gpt-4.1-2025-04-14"
    return app


async def _wait_for(predicate, pilot, attempts: int = 300) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await pilot.pause(0.01)
    raise AssertionError("condition was not met in time")


def test_discovery_candidates_still_target_the_wellknown_localhost_ports() -> None:
    """Anti-vacuity: the candidate list really does lead with 8080 and 11434.

    If this ever stops being true the isolation tests below could pass for the
    wrong reason (nothing to probe), so it is asserted explicitly.
    """
    candidates = [candidate.base_url for candidate in build_local_server_candidates({})]

    assert candidates[:2] == [
        DEFAULT_LLAMACPP_DISCOVERY_URL,
        DEFAULT_OLLAMA_DISCOVERY_URL,
    ]
    assert DEFAULT_LLAMACPP_DISCOVERY_URL == "http://127.0.0.1:8080"


@pytest.mark.local_server_probe
async def test_unguarded_discovery_really_reaches_localhost_8080_and_11434() -> None:
    """The defect itself, pinned: without the probe guard, egress is attempted.

    Marked `local_server_probe` to opt OUT of `_no_local_server_probes`, i.e.
    it runs the production probe exactly as `chat_screen` does. The socket
    guard catches the attempt, so this documents the escape without performing
    it -- and it is the reason the next test is not vacuous.
    """
    servers = await discover_local_servers({})

    attempted = {address for _call, address in network_guard.drain_blocked_attempts()}
    assert {"127.0.0.1:8080", "127.0.0.1:11434"} <= attempted, (
        f"expected probes at the two well-known ports, saw {sorted(attempted)}"
    )
    # Blocked egress degrades down the ordinary "nothing listening" path.
    assert servers == ()


async def test_guarded_discovery_makes_no_connection_attempt() -> None:
    """With the autouse guard in place (default), nothing leaves the process."""
    servers = await discover_local_servers({})

    assert servers == ()
    assert network_guard.blocked_attempts() == ()


@pytest.mark.asyncio
async def test_mounting_the_blocked_console_probes_no_live_server(monkeypatch) -> None:
    """End-to-end: the blocking setup card starts discovery and touches nothing.

    Asserts the worker actually ran (`_console_local_discovery_started`) so the
    "no connection" half cannot pass by simply never getting there.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    host = ConsoleHarness(_blocked_provider_app())

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for(lambda: console._console_setup_modal_blocking(), pilot)
        await _wait_for(lambda: console._console_local_discovery_started, pilot)
        # Let the discovery worker run to completion.
        for _ in range(20):
            await pilot.pause(0.01)

        assert console._console_local_discovery_started is True
        assert console._console_detected_local_server is None

    assert network_guard.blocked_attempts() == ()
