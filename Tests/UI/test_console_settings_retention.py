"""Mounted regressions for conversation settings retained across modal visits."""

from dataclasses import replace

import pytest
from textual.widgets import Button, Input

from Tests.private_profile import private_profile_test
from Tests.UI.test_console_provider_apply_defaults_flow import (
    _ConsoleFlowHarness,
    _drain_settings_tasks,
    _persisted_console_app,
    _reset_default_intent_state,  # noqa: F401 - shared autouse fixture
)
from Tests.UI.test_destination_shells import _wait_for_selector
from tldw_chatbook.Chat.console_session_settings import (
    build_target_default_console_session_settings,
)
from tldw_chatbook.config import load_settings
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


async def _open_settings(console, harness, pilot, surface):
    if surface == "quick":
        await console.action_open_console_model_popover()
    else:
        await console._open_console_settings(focus_model=True)
    await pilot.pause()
    return harness.screen


async def _submit_settings(modal, console, harness, pilot, surface, action):
    if surface == "quick":
        if action == "default":
            modal.query_one("#console-popover-defaults", Button).press()
            await pilot.pause()
            button_id = "console-popover-make-new-chat-default"
        else:
            button_id = "console-popover-apply"
    else:
        button_id = (
            "console-settings-make-default"
            if action == "default"
            else "console-settings-save"
        )
    button = modal.query_one(f"#{button_id}", Button)
    assert not button.disabled
    button.press()
    await pilot.pause()
    assert harness.screen is console
    await _drain_settings_tasks(harness.app_instance)


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["quick", "full"])
@pytest.mark.parametrize("second_action", ["apply", "default"])
@private_profile_test
async def test_reopen_keeps_previously_applied_temperature(
    request, surface, second_action
):
    """A later Streaming edit must not replace the displayed conversation value."""

    app = _persisted_console_app()
    harness = _ConsoleFlowHarness(app)
    async with harness.run_test(size=(160, 48)) as pilot:
        console = harness.screen
        assert isinstance(console, ChatScreen)
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        prefix = "console-popover" if surface == "quick" else "console-settings"

        modal = await _open_settings(console, harness, pilot, surface)
        modal.query_one(f"#{prefix}-temperature", Input).value = "0.23"
        await pilot.pause()
        await _submit_settings(modal, console, harness, pilot, surface, "apply")
        assert store.session_settings(session_id).temperature == pytest.approx(0.23)

        modal = await _open_settings(console, harness, pilot, surface)
        assert float(modal.query_one(f"#{prefix}-temperature", Input).value) == (
            pytest.approx(0.23)
        )
        modal.query_one(f"#{prefix}-streaming", Button).press()
        await pilot.pause()
        await _submit_settings(modal, console, harness, pilot, surface, second_action)

        current = store.session_settings(session_id)
        assert current.temperature == pytest.approx(0.23)
        if second_action == "default":
            assert app.console_default_durability_state.failure_phase is None
            loaded = load_settings(force_reload=True)
            saved = build_target_default_console_session_settings(
                loaded, current.provider, current.model
            )
            assert saved.temperature == pytest.approx(0.23)
            assert saved.streaming is current.streaming
            assert loaded["chat_defaults"]["provider"] == current.provider
            assert loaded["chat_defaults"]["model"] == current.model
            assert loaded["api_settings"]["llama_cpp"]["api_url"] == (
                "http://127.0.0.1:9099"
            )
            await console._session._create_native_console_session_from_active_context()
            new_settings = store.session_settings(store.active_session_id)
            assert new_settings.temperature == pytest.approx(0.23)
            assert new_settings.streaming is current.streaming


@pytest.mark.asyncio
@private_profile_test
async def test_quick_apply_retains_hidden_generation_values_and_live_endpoint(
    request,
):
    """Opening quick settings cannot reset a full-settings conversation snapshot."""

    app = _persisted_console_app()
    harness = _ConsoleFlowHarness(app)
    async with harness.run_test(size=(160, 48)) as pilot:
        console = harness.screen
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        store.replace_session_settings(
            session_id,
            replace(
                store.session_settings(session_id),
                top_p=0.43,
                min_p=0.12,
                top_k=17,
                max_tokens=1234,
                seed=19,
                presence_penalty=0.31,
                frequency_penalty=0.29,
                base_url="http://127.0.0.1:9101",
            ),
        )

        modal = await _open_settings(console, harness, pilot, "quick")
        modal.query_one("#console-popover-streaming", Button).press()
        await pilot.pause()
        await _submit_settings(modal, console, harness, pilot, "quick", "apply")

        current = store.session_settings(session_id)
        assert current.top_p == pytest.approx(0.43)
        assert current.min_p == pytest.approx(0.12)
        assert current.top_k == 17
        assert current.max_tokens == 1234
        assert current.seed == 19
        assert current.presence_penalty == pytest.approx(0.31)
        assert current.frequency_penalty == pytest.approx(0.29)
        assert current.base_url == "http://127.0.0.1:9101"


@pytest.mark.asyncio
@private_profile_test
async def test_full_default_after_quick_transfer_saves_newly_exposed_values(
    request,
):
    """Full settings must preserve values absent from the quick field mask."""

    app = _persisted_console_app()
    harness = _ConsoleFlowHarness(app)
    async with harness.run_test(size=(160, 48)) as pilot:
        console = harness.screen
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        store.replace_session_settings(
            session_id, replace(store.session_settings(session_id), top_p=0.43)
        )
        revision = store.session_settings_revision(session_id)

        quick = await _open_settings(console, harness, pilot, "quick")
        quick.query_one("#console-popover-temperature", Input).value = "0.23"
        await pilot.pause()
        quick.query_one("#console-popover-full-settings", Button).press()
        await pilot.pause()
        full = harness.screen
        assert full.query_one("#console-settings-temperature", Input).value == "0.23"
        assert full.query_one("#console-settings-top-p", Input).value == "0.43"
        assert store.session_settings_revision(session_id) == revision
        await _submit_settings(full, console, harness, pilot, "full", "default")

        current = store.session_settings(session_id)
        assert current.temperature == pytest.approx(0.23)
        assert current.top_p == pytest.approx(0.43)
        saved = build_target_default_console_session_settings(
            load_settings(force_reload=True), current.provider, current.model
        )
        assert saved.temperature == pytest.approx(0.23)
        assert saved.top_p == pytest.approx(0.43)
