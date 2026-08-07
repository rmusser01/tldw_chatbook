"""task-2900: Lab ▸ Models defers its five heavy hidden views past first paint.

Same pattern as task-2725 (Roleplay): the ollama view (58 widgets) and the
four library views (download-models 76, curated, installed, remote) arrive
`display: none` behind the CSS `-active` mechanism anyway; mounting them
after first paint takes their CSS-application cost off the click→paint
critical path. `watch_active_view` already tolerates absent views.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.config import get_cli_setting as _real_get_cli_setting
from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow
from tldw_chatbook.UI.Screens.llm_screen import LLMScreen
from Tests.UI.app_factory import _build_test_app

pytestmark = pytest.mark.asyncio

_DEFERRED_VIEW_IDS = (
    "llm-view-ollama",
    "llm-view-curated",
    "llm-view-installed",
    "llm-view-remote",
    "llm-view-download-models",
)

_ALL_VIEW_IDS = _DEFERRED_VIEW_IDS + (
    "llm-view-llama-cpp",
    "llm-view-llamafile",
    "llm-view-vllm",
    "llm-view-onnx",
    "llm-view-transformers",
    "llm-view-mlx-lm",
)


@pytest.fixture(autouse=True)
def _no_splash(monkeypatch):
    def fake_get_cli_setting(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return _real_get_cli_setting(section, key, default)

    monkeypatch.setattr("tldw_chatbook.app.get_cli_setting", fake_get_cli_setting)


async def test_first_paint_excludes_the_deferred_views(monkeypatch):
    """Compose alone must not mount the deferred views (the perf mechanism)."""
    monkeypatch.setattr(
        LLMManagementWindow,
        "_mount_deferred_views",
        AsyncMock(),
        raising=False,
    )

    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        screen = LLMScreen(app)
        await app.push_screen(screen)
        await pilot.pause()

        for view_id in _DEFERRED_VIEW_IDS:
            assert not list(screen.query(f"#{view_id}")), (
                f"#{view_id} mounted during compose — back on the "
                "click→paint critical path"
            )


async def test_load_mounts_every_view_with_llamacpp_active():
    """After the real deferred mount, all eleven views exist, exactly one is
    active (the initial llama-cpp), and the deferred ones stay CSS-hidden."""
    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        screen = LLMScreen(app)
        await app.push_screen(screen)
        for _ in range(6):
            await pilot.pause()

        for view_id in _ALL_VIEW_IDS:
            found = list(screen.query(f"#{view_id}"))
            assert len(found) == 1, f"#{view_id} missing after load"

        active = [
            view_id
            for view_id in _ALL_VIEW_IDS
            if screen.query_one(f"#{view_id}").has_class("-active")
        ]
        assert active == ["llm-view-llama-cpp"]


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
        prereq = [
            str(w.renderable)
            for w in view.query(".prereq-hint").results()
        ]
        assert any("Requires: Ollama" in text for text in prereq), (
            f"prereq line missing from extracted view: {prereq}"
        )
