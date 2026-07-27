"""Models' adoption of the Lab frame, and its rail lift."""

from __future__ import annotations

import pytest
from textual.widgets import Button, Static

from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow
from tldw_chatbook.UI.Screens.llm_screen import LLMScreen
from Tests.UI.test_screen_navigation import _build_test_app


async def _models_screen(pilot_app):
    screen = LLMScreen(pilot_app)
    await pilot_app.push_screen(screen)
    return screen


def _app():
    """Build the test app.

    No CSS bundle: every assertion here is behavioural (class membership,
    reactive values, chip text), not rendered styling. Rail-row styling is
    asserted in test_lab_workbench.py against a class-level CSS_PATH -- a
    post-construction `app.CSS_PATH = ...` would silently do nothing, since
    App.__init__ reads CSS_PATH once at construction.
    """
    return _build_test_app()


def _rail_rows(screen):
    return list(screen.query(".lab-rail-row").results(Button))


@pytest.mark.asyncio
async def test_all_nine_provider_rows_live_in_the_rail():
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        await pilot.pause()
        await pilot.pause()
        keys = [row.lab_view_key for row in _rail_rows(screen)]
        assert keys == [
            "llama-cpp",
            "llamafile",
            "ollama",
            "vllm",
            "onnx",
            "transformers",
            "mlx-lm",
            "local-models",
            "download-models",
        ]


@pytest.mark.asyncio
async def test_the_window_no_longer_carries_nav_buttons():
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        await pilot.pause()
        await pilot.pause()
        window = screen.query_one(LLMManagementWindow)
        assert not window.query(".llm-nav-button")


@pytest.mark.asyncio
async def test_the_rail_is_highlighted_on_arrival_before_any_press():
    """LLMManagementWindow.on_mount sets active_view itself, so a
    press-only implementation would leave the rail unhighlighted here."""
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        await pilot.pause()
        await pilot.pause()
        active = [r for r in _rail_rows(screen) if "is-active" in r.classes]
        assert len(active) == 1
        assert active[0].lab_view_key == "llama-cpp"


@pytest.mark.asyncio
async def test_pressing_a_rail_row_moves_both_the_body_and_the_highlight():
    """The highlight half fails SILENTLY -- query() returns empty rather than
    raising -- so a body-only assertion would pass with it dead."""
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        await pilot.pause()
        await pilot.pause()

        ollama = next(r for r in _rail_rows(screen) if r.lab_view_key == "ollama")
        ollama.press()
        await pilot.pause()

        window = screen.query_one(LLMManagementWindow)
        assert window.active_view == "ollama"
        assert "-active" in window.query_one("#llm-view-ollama").classes

        active = [r for r in _rail_rows(screen) if "is-active" in r.classes]
        assert len(active) == 1, "exactly one rail row must be highlighted"
        assert active[0].lab_view_key == "ollama"


@pytest.mark.asyncio
async def test_the_status_row_reports_running_servers():
    app = _app()
    app.llamacpp_server_process = None
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        await pilot.pause()
        await pilot.pause()
        chip = screen.query_one("#lab-status-chip-servers", Static)
        assert "Servers: none running" in str(chip.renderable)

        class _Alive:
            def poll(self):
                return None

        app.llamacpp_server_process = _Alive()
        screen.refresh_lab_status()
        await pilot.pause()
        assert "Servers: 1 running" in str(chip.renderable)
