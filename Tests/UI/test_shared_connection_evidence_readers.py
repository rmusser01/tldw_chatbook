"""TASK-33005.9: Home, the Library lock pre-check and the first-run wizard
share the Console's connection evidence.

Phase 5 left three readers config-only: after a refused test, Home and the
Library pre-check still read Ready while the Console read Not ready, and the
wizard kept its results in a private store. Each test below settles a refused
result and shows it reaching the surface, through the real readiness builder
and the app's real evidence owner.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from textual.widgets import Button, Input

from Tests.private_profile import private_profile_test
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    build_console_settings_readiness,
    build_default_console_session_settings,
    console_send_connection,
)
from tldw_chatbook.Chat.provider_test_evidence import (
    ProviderTestEvidence,
    provider_connection_evidence,
)

pytestmark = pytest.mark.asyncio

_LLAMA_URL = "http://127.0.0.1:9099"
_LLAMA_CONFIG = {
    "api_settings": {"llama_cpp": {"api_url": _LLAMA_URL}},
    "chat_defaults": {"provider": "llama_cpp", "model": "model-a"},
}


def _no_network(monkeypatch) -> None:
    """Reading shared evidence is memory only: any HTTP send fails the test."""

    def refuse(*_args, **_kwargs):
        raise AssertionError("a readiness read made a network request")

    monkeypatch.setattr(httpx.Client, "send", refuse)
    monkeypatch.setattr(httpx.AsyncClient, "send", refuse)


def _refuse_the_default_connection(running_app, config) -> None:
    """Settle a refused test of the connection a new chat sends to."""
    connection = console_send_connection(
        build_default_console_session_settings(config), app_config=config, environ={}
    )
    assert connection is not None
    assert provider_connection_evidence(running_app).publish(
        ProviderTestEvidence(connection, "unreachable", (), "connection_refused"),
        order=1,
    )


@private_profile_test
async def test_a_refused_test_reaches_home_and_the_library_lock_pre_check(
    request, monkeypatch
):
    """AC#1/#3: Home's Model badge seam and Library's "Use in Console" lock
    pre-check read the refused result of the default connection, as the
    Console does, without a request; Home's header then says Not ready, the
    Console's word for a known failure, not Not set up."""
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_home_screen import (
        HOME_MOUNT_PAUSE,
        HOME_TEST_SIZE,
        HomeHarness,
        _active_home_screen,
    )
    from tldw_chatbook.UI.Screens import home_screen as home_screen_module
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    app = _build_test_app()
    app.app_config.update(_LLAMA_CONFIG)
    # The snapshot's fresh-config read sees this same config.
    monkeypatch.setattr(home_screen_module, "load_settings", lambda: app.app_config)
    host = HomeHarness(app)
    async with host.run_test(size=HOME_TEST_SIZE) as pilot:
        await pilot.pause(HOME_MOUNT_PAUSE)
        home = _active_home_screen(host)
        # The pre-check's own body, on the running app (it needs no mount).
        library = SimpleNamespace(app_instance=app, app=host)

        def home_ready() -> bool:
            return home._home_console_model_state(
                background_credentials=False, allow_fresh_load=False
            )[0]

        assert home_ready() is True
        assert LibraryScreen._console_setup_would_block(library) is False

        _refuse_the_default_connection(host, app.app_config)
        _no_network(monkeypatch)

        assert home_ready() is False
        assert LibraryScreen._console_setup_would_block(library) is True

        await home._refresh_home_content_snapshot().wait()
        await pilot.pause()
        header = str(home.query_one("#home-header-line").render())
        assert header == "Home | Not ready \u00b7 Local"
        # The next action under it asks for a reconnect, not a setup.
        assert home._current_dashboard.next_action.label == "Reconnect Console model"

        # After the first send no Get started card returns, so the Library
        # warns of none (search-and-rag.md says so): Console reads Not ready.
        console_cfg = app.app_config.setdefault("console", {})
        console_cfg.setdefault("onboarding", {})["first_send_completed"] = True
        assert LibraryScreen._console_setup_would_block(library) is False


def _refused_outcome(category: str = "connection_refused"):
    from tldw_chatbook.UI.Screens.settings_endpoint_probe import (
        SettingsEndpointProbeOutcome,
    )

    return SettingsEndpointProbeOutcome(
        state="unreachable", category=category, summary="Refused."
    )


@pytest.mark.parametrize(
    ("provider", "environ", "config", "category", "blocker"),
    [
        pytest.param(
            "llama_cpp",
            {},
            {"api_settings": {"llama_cpp": {"api_url": _LLAMA_URL}}},
            "connection_refused",
            "endpoint_unreachable",
            id="keyless-local",
        ),
        pytest.param(
            "openai",
            {"OPENAI_API_KEY": "sk-wizard-env-test-key"},
            {"api_settings": {"openai": {}}},
            "unauthorized",
            "credential_rejected",
            id="environment-key",
        ),
    ],
)
@private_profile_test
async def test_a_refused_wizard_test_reaches_the_console(
    request, provider, environ, config, category, blocker
):
    """AC#2/#3: the wizard's Test settles into the app's shared owner under
    the connection the Console keys, so the Console reads it refused. The
    environment-key case is the one a private edit counter could never key:
    the owner keys the sent key's digest."""
    from Tests.Wizards.test_first_run_setup_wizard import _provider_step, _StepHost

    probe = AsyncMock(return_value=_refused_outcome(category))
    step = _provider_step(probe=probe, environ=environ)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider(provider)
        await pilot.pause()
        if provider == "llama_cpp":
            step.query_one("#setup-provider-endpoint", Input).value = _LLAMA_URL
            await pilot.pause()
        step.query_one("#setup-provider-test", Button).press()
        await pilot.pause(0.1)
        probe.assert_awaited_once()

        readiness = build_console_settings_readiness(
            ConsoleSessionSettings(provider=provider, model="model-a"),
            app_config=config,
            environ=environ,
            connection_evidence=provider_connection_evidence(app),
        )

    assert readiness.blocker == blocker
    assert readiness.endpoint_category == category
    assert readiness.native_send_supported is False
