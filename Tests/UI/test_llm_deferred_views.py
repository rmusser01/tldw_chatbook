"""Lab ▸ Models mounts provider bodies lazily and caches first-used views.

All eleven stable shells exist so navigation and CSS remain predictable, but
only the initial llama.cpp body composes on arrival. Every other body mounts
on first selection and stays cached to preserve in-progress form state.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from textual.widgets import Input, Static

from tldw_chatbook.config import get_cli_setting as _real_get_cli_setting
from tldw_chatbook.UI.LLM_Management_Window import (
    LLMManagementWindow,
    OllamaServiceView,
)
from tldw_chatbook.UI.Screens.llm_screen import LLMScreen
from Tests.UI.app_factory import _build_test_app

pytestmark = pytest.mark.asyncio

_DEFERRED_VIEW_IDS = (
    "llm-view-ollama",
    "llm-view-curated",
    "llm-view-installed",
    "llm-view-external",
    "llm-view-remote",
)

_FIRST_PAINT_VIEW_IDS = (
    "llm-view-llama-cpp",
    "llm-view-llamafile",
    "llm-view-vllm",
    "llm-view-onnx",
    "llm-view-transformers",
    "llm-view-mlx-lm",
)
_ALL_VIEW_IDS = _FIRST_PAINT_VIEW_IDS + _DEFERRED_VIEW_IDS


def _mounted_view_ids(window: LLMManagementWindow) -> tuple[str | None, ...]:
    return tuple(
        sorted(
            (view.id for view in window.query(".llm-view")),
            key=lambda view_id: view_id or "",
        )
    )


@pytest.fixture(autouse=True)
def _no_splash(monkeypatch):
    def fake_get_cli_setting(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return _real_get_cli_setting(section, key, default)

    monkeypatch.setattr("tldw_chatbook.app.get_cli_setting", fake_get_cli_setting)


async def test_initial_load_populates_only_llamacpp_body():
    """The default pane is usable before any inactive pane body is mounted."""
    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        screen = LLMScreen(app)
        await app.push_screen(screen)
        for _ in range(8):
            await pilot.pause()

        window = screen.query_one(LLMManagementWindow)
        assert _mounted_view_ids(window) == tuple(sorted(_ALL_VIEW_IDS))
        assert set(window.view_mapping.values()) == set(_ALL_VIEW_IDS)
        assert window.active_view == "llama-cpp"
        assert screen.query_one("#llm-view-llama-cpp").has_class("-active")
        assert screen.query_one("#llamacpp-start-server-button").is_mounted

        inactive_body_selectors = (
            "#ollama-exec-path",
            "#llamafile-start-server-button",
            "#vllm-start-server-button",
            "#onnx-start-server-button",
            "#transformers-models-dir-path",
            "#mlx-start-server-button",
            "#curated-models-view",
            "#installed-models-view",
            "#external-models-view",
            "#remote-models-view",
        )
        assert all(not list(screen.query(selector)) for selector in inactive_body_selectors)


async def test_first_selection_mounts_and_caches_only_requested_view():
    """First use populates one pane; revisiting reuses that pane's state."""
    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        screen = LLMScreen(app)
        await app.push_screen(screen)
        for _ in range(8):
            await pilot.pause()

        window = screen.query_one(LLMManagementWindow)
        llama_exec = screen.query_one("#llamacpp-exec-path", Input)
        llama_exec.value = "/tmp/keep-this-value"

        window.active_view = "ollama"
        for _ in range(8):
            await pilot.pause()

        ollama_exec = screen.query_one("#ollama-exec-path", Input)
        ollama_exec.value = "/tmp/keep-ollama-value"
        assert not list(screen.query("#remote-models-view"))

        window.active_view = "llama-cpp"
        await pilot.pause()
        assert screen.query_one("#llamacpp-exec-path", Input) is llama_exec
        assert llama_exec.value == "/tmp/keep-this-value"

        window.active_view = "ollama"
        await pilot.pause()
        assert screen.query_one("#ollama-exec-path", Input) is ollama_exec
        assert ollama_exec.value == "/tmp/keep-ollama-value"


async def test_ollama_view_activates_and_renders_after_deferral():
    """The extracted ollama view must render its real content when shown —
    the one view whose body moved out of `compose` (single-substitution
    extraction: the prereq line)."""
    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        screen = LLMScreen(app)
        await app.push_screen(screen)
        for _ in range(6):
            await pilot.pause()

        window = screen.query_one(LLMManagementWindow)
        window.active_view = "ollama"
        await pilot.pause()

        view = screen.query_one("#llm-view-ollama")
        assert view.has_class("-active")
        assert list(view.query("#ollama-exec-path")), (
            "extracted ollama view lost its executable-path input"
        )
        prereq = [str(w.renderable) for w in view.query(".prereq-hint").results()]
        assert any("Requires: Ollama" in text for text in prereq), (
            f"prereq line missing from extracted view: {prereq}"
        )


async def test_inactive_view_compose_cannot_stall_models_navigation(monkeypatch):
    """An expensive inactive pane must not block the event loop on arrival."""

    def blocking_ollama_compose(self):
        time.sleep(0.35)
        yield Static("deliberately slow inactive pane")

    monkeypatch.setattr(OllamaServiceView, "compose", blocking_ollama_compose)

    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        gaps: list[float] = []
        running = True

        async def heartbeat() -> None:
            loop = asyncio.get_running_loop()
            previous = loop.time()
            while running:
                await asyncio.sleep(0.005)
                now = loop.time()
                gaps.append(now - previous)
                previous = now

        heartbeat_task = asyncio.create_task(heartbeat())
        await asyncio.sleep(0)
        screen = LLMScreen(app)
        await app.push_screen(screen)
        for _ in range(8):
            await pilot.pause()
        await asyncio.sleep(0.05)
        running = False
        await heartbeat_task

        assert screen.query_one("#llamacpp-start-server-button").is_mounted
        assert gaps
        assert max(gaps) < 0.25, f"event loop stalled for {max(gaps):.3f}s"
