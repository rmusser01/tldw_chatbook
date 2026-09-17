"""Quick settings resolve serving capacity without blocking or stale publication."""

import asyncio
from dataclasses import replace

import pytest
from textual.widgets import Button, Static

from Tests.Chat.test_console_settings_apply import _rebase, _state
from Tests.UI.test_console_session_settings import ModalHarness
from tldw_chatbook.Chat.console_context_policy import (
    ConsoleContextPolicyOverrides,
    ContextBudgetMode,
)
from tldw_chatbook.Chat.console_context_window import ContextWindowResolution
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsContextEstimate,
    build_console_settings_readiness,
)
from tldw_chatbook.Chat.console_settings_apply import ConsoleSettingsOrigin
from tldw_chatbook.Widgets.Console.console_context_controls import (
    build_console_context_control_state,
)
from tldw_chatbook.Widgets.Console.console_model_popover import ConsoleModelPopover


def _popover(resolver):
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        base_url="http://127.0.0.1:9099",
        max_tokens=1000,
    )
    config = {
        "api_settings": {"llama_cpp": {"api_url": settings.base_url}},
        "chat_defaults": {"max_tokens": 1000},
    }
    state = build_console_context_control_state(
        settings=settings,
        estimate=ConsoleSettingsContextEstimate(600, 4000, "600 / 4,000"),
        global_overrides=ConsoleContextPolicyOverrides(
            budget_mode=ContextBudgetMode.CUSTOM,
            custom_budget_tokens=9000,
            trigger_ratio=0.75,
        ),
        conversation_tokens=400,
        request_overhead_tokens=200,
    )
    return ConsoleModelPopover(
        origin=ConsoleSettingsOrigin("session", None, 0),
        app_config=config,
        initial_draft=_state(settings),
        providers_models={"llama_cpp": ["model-a", "model-b"]},
        context_state=state,
        scope_copy="This chat",
        durability_copy="Unsaved",
        draft_rebaser=_rebase,
        live_committer=lambda _submission: None,
        default_readiness_resolver=lambda provider, model: (
            build_console_settings_readiness(
                replace(settings, provider=provider, model=model), app_config=config
            )
        ),
        context_window_resolver=resolver,
    )


@pytest.mark.asyncio
async def test_popover_updates_serving_window_and_keeps_policy_while_resolver_waits():
    started, release = asyncio.Event(), asyncio.Event()

    async def resolve(_settings):
        started.set()
        await release.wait()
        return ContextWindowResolution(12000, "server metadata", True)

    popover = _popover(resolve)
    app = ModalHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(popover)
        await asyncio.wait_for(started.wait(), timeout=2)
        await pilot.pause()
        assert popover._context_state.model_window_tokens == 32000
        assert "estimated" in str(
            popover.query_one("#console-popover-model-window", Static).render()
        )
        before = popover._streaming
        popover.query_one("#console-popover-streaming", Button).press()
        await pilot.pause()
        assert popover._streaming is not before
        release.set()
        await popover.workers.wait_for_complete()
        await pilot.pause()

        state = popover._context_state
        assert state.model_window_tokens == 12000
        assert state.model_window_source == "server metadata"
        assert state.model_window_verified
        assert state.safe_input_ceiling_tokens == 10488
        assert state.conversation_budget_tokens == 9000
        assert state.compaction_trigger_tokens == 6750
        assert "12,000" in str(
            popover.query_one("#console-popover-model-window", Static).render()
        )
        assert "10,488" in str(
            popover.query_one("#console-popover-request-usage", Static).render()
        )


@pytest.mark.asyncio
async def test_popover_rejects_late_window_after_model_a_b_a():
    started, release = asyncio.Event(), asyncio.Event()
    stale_finished = asyncio.Event()
    first = True

    async def resolve(settings):
        nonlocal first
        if first:
            first = False
            started.set()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    continue
            stale_finished.set()
            return ContextWindowResolution(8000, "server metadata", True)
        return ContextWindowResolution(
            16000 if settings.model == "model-a" else 24000,
            "server metadata",
            True,
        )

    popover = _popover(resolve)
    app = ModalHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        try:
            await app.push_screen(popover)
            await asyncio.wait_for(started.wait(), timeout=2)
            picker = popover.query_one("#console-popover-model-search")
            picker.set_model_value("model-b")
            picker.post_message(picker.ModelSelected("model-b"))
            await pilot.pause()
            assert popover._context_state.model_window_tokens == 24000
            picker.set_model_value("model-a")
            picker.post_message(picker.ModelSelected("model-a"))
            await pilot.pause()
            assert popover._context_state.model_window_tokens == 16000
            release.set()
            await asyncio.wait_for(stale_finished.wait(), timeout=2)
            await pilot.pause()
            assert popover._context_state.model_window_tokens == 16000
        finally:
            release.set()


@pytest.mark.asyncio
async def test_popover_resolver_failure_keeps_estimated_fallback():
    async def resolve(_settings):
        raise OSError("server unavailable")

    popover = _popover(resolve)
    app = ModalHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(popover)
        await pilot.pause()
        await popover.workers.wait_for_complete()
        assert popover._context_state.model_window_tokens == 32000
        assert not popover._context_state.model_window_verified
        assert "estimated" in str(
            popover.query_one("#console-popover-model-window", Static).render()
        )


@pytest.mark.asyncio
async def test_popover_ignores_context_result_after_dismissal():
    started, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def resolve(_settings):
        started.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                continue
        finished.set()
        return ContextWindowResolution(12000, "server metadata", True)

    popover = _popover(resolve)
    app = ModalHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        try:
            await app.push_screen(popover)
            await asyncio.wait_for(started.wait(), timeout=2)
            await pilot.press("escape")
            await pilot.pause()
            assert app.screen is not popover
            release.set()
            await asyncio.wait_for(finished.wait(), timeout=2)
            await pilot.pause()
            assert popover._context_state.model_window_tokens == 32000
        finally:
            release.set()
