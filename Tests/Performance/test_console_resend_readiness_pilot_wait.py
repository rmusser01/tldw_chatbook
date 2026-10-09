"""Pure original Pilot body: unrelated screen settlement is not readiness."""

import ast
import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from Tests.Performance.test_console_resend_startup_readiness import (
    _load_helpers,
    _Projection,
    _Screen,
)


_SPEC = importlib.util.find_spec("textual")
assert _SPEC is not None and _SPEC.origin is not None
_DEPENDENCY = Path(_SPEC.origin).parent / "pilot.py"


def _original_pause():
    source = _DEPENDENCY.read_bytes()
    owner = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "Pilot"
    )
    pause = next(
        node
        for node in owner.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "pause"
    )
    namespace = {"asyncio": asyncio}
    exec(
        compile(ast.Module(body=[pause], type_ignores=[]), str(_DEPENDENCY), "exec"),
        namespace,
    )
    assert source == _DEPENDENCY.read_bytes()
    assert pause.body[1].value.value.func.attr == "_wait_for_screen"
    return namespace["pause"]


class _OwnedScreenWaitPilot:
    pause = _original_pause()

    def __init__(self):
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.calls = self.paint = 0
        self.app = SimpleNamespace(screen=SimpleNamespace(_on_timer_update=self._paint))

    def _paint(self):
        self.paint += 1

    async def _wait_for_screen(self, timeout=30.0):
        self.calls += 1
        self.entered.set()
        await self.release.wait()


async def _held_screen_readiness_case():
    screen, pilot = _Screen(), _OwnedScreenWaitPilot()
    projection = _Projection(screen)
    screen._console_readiness_config_projection = projection
    entered, release = asyncio.Event(), asyncio.Event()
    reader = asyncio.create_task(projection._refresh(entered, release))
    unrelated = asyncio.create_task(pilot.pause(0.01))
    barrier = None
    try:
        await entered.wait()
        await pilot.entered.wait()
        barrier = asyncio.create_task(
            _load_helpers()["_await_selected_console_readiness"](
                screen, pilot, deadline=time.monotonic() + 0.15
            )
        )
        await asyncio.sleep(0)
        assert not barrier.done() and screen.control_calls == 0
        release.set()
        await reader
        await asyncio.wait_for(asyncio.shield(barrier), timeout=0.15)
        assert pilot.calls == 1 and not unrelated.done() and not unrelated.cancelled()
        assert screen.control_calls == 1 and screen.button.disabled
    finally:
        release.set()
        pilot.release.set()
        await reader
        await unrelated
        if barrier is not None and not barrier.done():
            barrier.cancel()
        if barrier is not None:
            await asyncio.gather(barrier, return_exceptions=True)


@pytest.mark.unit
def test_resend_readiness_does_not_wait_for_original_pilot_screen_settlement():
    asyncio.run(asyncio.wait_for(_held_screen_readiness_case(), timeout=1.0))
