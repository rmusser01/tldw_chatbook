"""task-22220 item 2: the 3 s Ollama tick must construct nothing on an
inactive screen.

`LLMManagementWindow.on_mount` arms `set_interval(3.0,
_schedule_ollama_api_state)`, and the scheduler used to unconditionally
construct the `_update_ollama_api_state()` coroutine and schedule a worker
around it -- just so the coroutine's own first line (`is_attached` /
`screen.is_active`) could drop it. On a hidden tab that is one worker
construction every 3 seconds, forever. The gate is hoisted into
`_schedule_ollama_api_state`; the coroutine keeps its own pre-await guard
(the scheduling->running race) and post-await re-check (mid-probe
deactivation, task-15473).

Born red: the inactive-screen test observed 5 worker constructions over 5
ticks against the pre-fix tree. The active-screen test pins the preserved
worker semantics (`exclusive=True`, `group="ollama-api-state"`,
`exit_on_error=False`) and was green on both sides of the fix.

`_probe_local_server` is patched to a fixed async result -- no real socket.
"""

from __future__ import annotations

import pytest
from textual.screen import Screen

from tldw_chatbook.config import get_cli_setting as _real_get_cli_setting
from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow
from tldw_chatbook.UI.Screens import llm_screen as llm_screen_module
from tldw_chatbook.UI.Screens.llm_screen import LLMScreen
from Tests.UI.app_factory import _build_test_app

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _deterministic_models_mount(monkeypatch):
    """Neutralize the splash race so the Models screen mounts
    deterministically (same rationale as
    ``test_llm_screen_ollama_ux_unchanged.py``'s identically named fixture).
    """

    def fake_get_cli_setting(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return _real_get_cli_setting(section, key, default)

    monkeypatch.setattr("tldw_chatbook.app.get_cli_setting", fake_get_cli_setting)


async def _mount_models(monkeypatch, pilot) -> LLMManagementWindow:
    """Mount the Models screen with the Ollama probe patched down.

    Two ``pilot.pause()`` calls, not one: the established convention for
    this mount (`on_mount` defers the heavy views via `call_after_refresh`;
    a single pause intermittently landed before that chain drained).
    """

    async def fake_probe(host: str = "127.0.0.1", port: int = 11434) -> bool:
        return False

    monkeypatch.setattr(llm_screen_module, "_probe_local_server", fake_probe)

    screen = LLMScreen(pilot.app)
    await pilot.app.push_screen(screen)
    await pilot.pause()
    await pilot.pause()
    return screen.query_one(LLMManagementWindow)


def _shadow_run_worker(window: LLMManagementWindow) -> list[dict]:
    """Shadow ``window.run_worker`` with a recorder that schedules nothing.

    Instance-attribute shadowing is sufficient here: the probes call
    ``_schedule_ollama_api_state`` directly, and that method resolves
    ``self.run_worker`` through the instance. Any coroutine handed over is
    closed so the recorder never leaves an un-awaited coroutine behind.
    """
    constructions: list[dict] = []

    def counting_run_worker(work, *args, **kwargs):
        constructions.append(dict(kwargs))
        close = getattr(work, "close", None)
        if callable(close):
            close()
        return None

    window.run_worker = counting_run_worker
    return constructions


async def test_inactive_screen_tick_constructs_no_worker(monkeypatch):
    """Five ticks against a covered Models screen must construct zero
    workers (pre-fix: five)."""
    app = _build_test_app()
    async with app.run_test(size=(120, 40)) as pilot:
        window = await _mount_models(monkeypatch, pilot)

        await pilot.app.push_screen(Screen())
        await pilot.pause()
        assert window.is_attached, "test premise: the window is still mounted"
        assert not window.screen.is_active, (
            "test premise: the Models screen is covered"
        )

        constructions = _shadow_run_worker(window)
        for _ in range(5):
            window._schedule_ollama_api_state()

        assert constructions == [], (
            f"{len(constructions)} worker(s) constructed over 5 ticks with "
            "the screen inactive; the gate must be hoisted into the scheduler"
        )


async def test_active_screen_tick_schedules_one_exclusive_worker(monkeypatch):
    """The active case keeps its worker semantics: one worker per tick,
    exclusive within the ``ollama-api-state`` group, errors contained."""
    app = _build_test_app()
    async with app.run_test(size=(120, 40)) as pilot:
        window = await _mount_models(monkeypatch, pilot)
        assert window.screen.is_active, (
            "test premise: the Models screen is on top"
        )

        constructions = _shadow_run_worker(window)
        window._schedule_ollama_api_state()

        assert len(constructions) == 1, (
            "an active screen's tick must still schedule its worker"
        )
        kwargs = constructions[0]
        assert kwargs.get("exclusive") is True
        assert kwargs.get("group") == "ollama-api-state"
        assert kwargs.get("exit_on_error") is False
