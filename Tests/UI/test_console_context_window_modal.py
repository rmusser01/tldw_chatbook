"""Mounted capacity refresh must remain responsive and reject stale targets."""

import asyncio

import pytest
from textual.widgets import Button, Select, Static

from Tests.UI.test_console_session_settings import ModalHarness
from tldw_chatbook.Chat.console_context_window import ContextWindowResolution
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    build_console_context_estimate,
)
from tldw_chatbook.Widgets.Console.console_settings_modal import ConsoleSettingsModal


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["switch", "return", "dismiss"])
async def test_full_modal_refreshes_capacity_and_ignores_late_previous_model(finish):
    entered, release = asyncio.Event(), asyncio.Event()
    first = True

    async def resolve(settings):
        nonlocal first
        if first:
            first = False
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
            return ContextWindowResolution(9000, "server metadata", True)
        return ContextWindowResolution(64000, "server metadata", True)

    settings = ConsoleSessionSettings(provider="openai", model="gpt-4o")
    modal = ConsoleSettingsModal(
        settings=settings,
        app_config={},
        providers_models={
            "openai": ["gpt-4o"],
            "anthropic": ["claude-3-opus-20240229"],
        },
        context_estimate=build_console_context_estimate(
            [], settings.provider, settings.model
        ),
        context_window_resolver=resolve,
        can_save=True,
    )
    app = ModalHarness()
    try:
        async with app.run_test(size=(160, 48)) as pilot:
            await app.push_screen(modal)
            await asyncio.wait_for(entered.wait(), 2)
            modal.query_one("#console-settings-streaming", Button).press()
            await pilot.pause()
            assert modal._streaming_draft is False
            modal.query_one("#console-settings-provider", Select).value = "anthropic"
            await pilot.pause()
            assert modal._context_estimate.token_limit == 64000
            if finish == "return":
                modal.query_one("#console-settings-provider", Select).value = "openai"
                await pilot.pause()
            elif finish == "dismiss":
                await pilot.press("escape")
                assert modal not in app.screen_stack
            expected = modal._context_estimate.token_limit
            release.set()
            await pilot.pause()
            assert modal._context_estimate.token_limit == expected
            if finish != "dismiss":
                assert f"{expected:,}" in str(
                    modal.query_one("#console-context-model-window", Static).render()
                )
    finally:
        release.set()
