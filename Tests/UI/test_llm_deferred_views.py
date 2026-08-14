"""task-2900: Lab ▸ Models defers its five heavy hidden views past first paint.

Same pattern as task-2725 (Roleplay): the ollama view and the four library
views (curated, installed, external, remote) arrive
`display: none` behind the CSS `-active` mechanism anyway; mounting them
after first paint takes their CSS-application cost off the click→paint
critical path. `watch_active_view` already tolerates absent views.
"""

from __future__ import annotations

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


async def test_first_paint_excludes_the_deferred_views(monkeypatch):
    """First paint has exactly the six eager views and five deferred views."""
    scheduled: list[object] = []

    def hold_after_refresh(self, callback, *args, **kwargs):
        scheduled.append(callback)

    monkeypatch.setattr(
        LLMManagementWindow,
        "call_after_refresh",
        hold_after_refresh,
    )

    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        screen = LLMScreen(app)
        await app.push_screen(screen)
        await pilot.pause()

        window = screen.query_one(LLMManagementWindow)
        mounted_ids = _mounted_view_ids(window)
        mapped_ids = set(window.view_mapping.values())

        assert [callback.__name__ for callback in scheduled] == [
            "_finish_deferred_mount"
        ]
        assert mounted_ids == tuple(sorted(_FIRST_PAINT_VIEW_IDS))
        assert mapped_ids - set(mounted_ids) == set(_DEFERRED_VIEW_IDS)
        assert len(mapped_ids) == len(_ALL_VIEW_IDS) == 11


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

        window = screen.query_one(LLMManagementWindow)
        mounted_ids = _mounted_view_ids(window)
        assert mounted_ids == tuple(sorted(_ALL_VIEW_IDS))
        assert len(mounted_ids) == len(_ALL_VIEW_IDS) == 11
        assert set(window.view_mapping.values()) == set(_ALL_VIEW_IDS)

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
        prereq = [str(w.renderable) for w in view.query(".prereq-hint").results()]
        assert any("Requires: Ollama" in text for text in prereq), (
            f"prereq line missing from extracted view: {prereq}"
        )
