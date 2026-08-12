"""Models' Ollama button gating survives the async probe conversion
(task-15473 AC#1: "Models availability UX unchanged").

`LLMManagementWindow._update_ollama_api_state`/`_ollama_api_available` were
converted from plain sync methods to coroutines so the probe they call
(`_probe_local_server`) can `await` a non-blocking connect instead of
freezing the event loop. This file pins the OBSERVABLE end-to-end behavior
that conversion must not change: with the service down, every dependent
Ollama button is disabled with the "requires a running service" tooltip;
with it up, none of them are.

`_probe_local_server` is patched to a fixed async result here -- no real
socket is opened, so this file needs no `allow_network` opt-in (unlike
`test_llm_screen_ollama_probe_nonblocking.py`, which pins the probe's own
real-socket up/refused/timeout behavior and loop responsiveness).
"""

from __future__ import annotations

import pytest
from textual.widgets import Button

from tldw_chatbook.config import get_cli_setting as _real_get_cli_setting
from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow
from tldw_chatbook.UI.Screens import llm_screen as llm_screen_module
from tldw_chatbook.UI.Screens.llm_screen import LLMScreen
from Tests.UI.app_factory import _build_test_app

#: Buttons `_update_ollama_api_state` deliberately never gates (they must
#: stay usable regardless of service availability -- starting the service,
#: or fixing the configured executable path, are exactly what an unavailable
#: state needs).
_UNGATED_BUTTON_IDS = {
    "ollama-start-service-button",
    "ollama-stop-service-button",
    "ollama-browse-exec-button",
}

_DOWN_TOOLTIP = "Requires a running Ollama service — start it above."


@pytest.fixture(autouse=True)
def _deterministic_models_mount(monkeypatch):
    """Same rationale as ``test_llm_screen_lab_adoption.py``'s identically
    named fixture: neutralize the splash race so the Models screen mounts
    deterministically.
    """

    def fake_get_cli_setting(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return _real_get_cli_setting(section, key, default)

    monkeypatch.setattr("tldw_chatbook.app.get_cli_setting", fake_get_cli_setting)


async def _mount_models_with_probe_result(monkeypatch, pilot_app, *, available: bool):
    """Mount the Models screen with the Ollama probe patched to a fixed
    result and return its `LLMManagementWindow`.

    Patches the module attribute `_probe_local_server` resolves via its
    local import in `_ollama_api_available` -- no real socket is ever
    opened.
    """

    async def fake_probe(host: str = "127.0.0.1", port: int = 11434) -> bool:
        return available

    monkeypatch.setattr(llm_screen_module, "_probe_local_server", fake_probe)

    screen = LLMScreen(pilot_app)
    await pilot_app.push_screen(screen)
    return screen.query_one(LLMManagementWindow)


def _gated_buttons(window: LLMManagementWindow) -> list[Button]:
    view = window.query_one("#llm-view-ollama")
    return [
        button
        for button in view.query(Button)
        if button.id and button.id not in _UNGATED_BUTTON_IDS
    ]


@pytest.mark.asyncio
async def test_ollama_controls_disabled_when_probe_reports_down(monkeypatch):
    app = _build_test_app()
    async with app.run_test(size=(120, 40)) as pilot:
        window = await _mount_models_with_probe_result(monkeypatch, app, available=False)
        await pilot.pause()

        gated = _gated_buttons(window)
        assert gated, "test premise: at least one gated Ollama button exists"
        assert all(button.disabled for button in gated), (
            "every gated Ollama button should be disabled while the "
            "service is unavailable"
        )
        assert all(button.tooltip == _DOWN_TOOLTIP for button in gated)

        # "Start Ollama Service" is excluded from the availability gate (a
        # user must be able to start the service from an unavailable
        # state) and starts enabled at compose time.
        start_button = window.query_one("#ollama-start-service-button", Button)
        assert not start_button.disabled


@pytest.mark.asyncio
async def test_ollama_controls_enabled_when_probe_reports_up(monkeypatch):
    app = _build_test_app()
    async with app.run_test(size=(120, 40)) as pilot:
        window = await _mount_models_with_probe_result(monkeypatch, app, available=True)
        await pilot.pause()

        gated = _gated_buttons(window)
        assert gated, "test premise: at least one gated Ollama button exists"
        assert not any(button.disabled for button in gated), (
            "no gated Ollama button should be disabled while the service "
            "is available"
        )


@pytest.mark.asyncio
async def test_ollama_controls_flip_when_availability_changes_between_ticks(monkeypatch):
    """The same coroutine drives both the one-shot post-mount check and
    the periodic `set_interval(3.0, ...)` tick -- exercise both directly
    rather than waiting a real 3s for the timer, and prove the gate is not
    stuck from whichever result the FIRST call happened to see.
    """
    app = _build_test_app()
    async with app.run_test(size=(120, 40)) as pilot:
        window = await _mount_models_with_probe_result(monkeypatch, app, available=False)
        await pilot.pause()
        assert all(button.disabled for button in _gated_buttons(window))

        async def fake_probe_up(host: str = "127.0.0.1", port: int = 11434) -> bool:
            return True

        monkeypatch.setattr(llm_screen_module, "_probe_local_server", fake_probe_up)
        await window._update_ollama_api_state()
        await pilot.pause()

        assert not any(button.disabled for button in _gated_buttons(window)), (
            "the gate should flip to enabled on the very next probe result"
        )
