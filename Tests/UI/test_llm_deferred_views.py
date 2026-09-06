"""Lab ▸ Models mounts provider bodies lazily and caches first-used views.

All eleven stable shells exist so navigation and CSS remain predictable, but
only the initial llama.cpp body composes on arrival. Every other body mounts
on first selection and stays cached to preserve in-progress form state.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

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
        manager = window.query_one("#llamacpp-snapshot-manager")
        assert manager.parent.id == "llm-view-llama-cpp"

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


async def test_failed_server_mount_retains_body_for_retry():
    """A transient mount failure must not consume the deferred server body."""

    window = LLMManagementWindow.__new__(LLMManagementWindow)
    body = (SimpleNamespace(is_mounted=False),)
    window._lazy_server_bodies = {"llamafile": body}
    window._populated_views = set()
    window.view_mapping = {"llamafile": "llm-view-llamafile"}
    window.call_after_refresh = lambda *_args: None

    class FlakyPane:
        def __init__(self) -> None:
            self.attempts = 0

        async def mount_all(self, widgets) -> None:
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("transient mount failure")
            assert tuple(widgets) == body

    pane = FlakyPane()
    window.query_one = lambda _selector: pane

    with pytest.raises(RuntimeError, match="transient mount failure"):
        await window._mount_deferred_views("llamafile")

    assert window._lazy_server_bodies["llamafile"] == body
    assert "llamafile" not in window._populated_views

    await window._mount_deferred_views("llamafile")

    assert "llamafile" not in window._lazy_server_bodies
    assert "llamafile" in window._populated_views


@pytest.mark.parametrize(
    ("view_name", "populated", "populating", "expected_schedules"),
    (
        ("missing", set(), set(), 0),
        ("remote", {"remote"}, set(), 0),
        ("remote", set(), {"remote"}, 0),
        ("remote", set(), set(), 1),
    ),
)
async def test_population_scheduler_guards_duplicate_and_invalid_requests(
    view_name,
    populated,
    populating,
    expected_schedules,
):
    """Only one valid, not-yet-started pane population may be scheduled."""

    window = LLMManagementWindow.__new__(LLMManagementWindow)
    window.view_mapping = {"remote": "llm-view-remote"}
    window._populated_views = set(populated)
    window._populating_views = set(populating)
    scheduled = []

    def schedule(awaitable, **kwargs):
        scheduled.append(kwargs)
        awaitable.close()

    window.run_worker = schedule

    window.ensure_view_populated(view_name)

    assert len(scheduled) == expected_schedules
    if expected_schedules:
        assert view_name in window._populating_views
    else:
        assert window._populating_views == set(populating)
    if scheduled:
        assert scheduled == [
            {
                "group": "llm-view-mount-remote",
                "exclusive": True,
                "exit_on_error": False,
            }
        ]


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


@pytest.mark.parametrize(
    "view_name,expected", [("ollama", "ollama"), ("private-view-token", "unknown")]
)
async def test_lazy_mount_failure_logs_only_bounded_view_context(
    monkeypatch, view_name, expected
):
    from loguru import logger

    window = LLMManagementWindow(_build_test_app())
    records = []

    async def fail_mount(_view_name):
        raise RuntimeError("private-provider-secret")

    monkeypatch.setattr(window, "_mount_deferred_views", fail_mount)
    sink = logger.add(lambda message: records.append(message.record))
    try:
        await window._activate_deferred_view(view_name)
    finally:
        logger.remove(sink)
    failures = [
        r for r in records if r["message"].startswith("Lazy LLM view mount failed")
    ]
    assert len(failures) == 1
    assert failures[0]["message"].endswith("view=" + expected)
    assert failures[0]["exception"] is None
    assert "private-provider-secret" not in str(failures[0])
    assert "private-view-token" not in str(failures[0])
