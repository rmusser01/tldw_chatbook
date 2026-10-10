"""Pure ownership controls for the three Resend fixtures' setup barrier."""

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
import time

import pytest


_HELPER_PATH = (
    Path(__file__).resolve().parents[2] / "Tests/UI/test_console_turn_resend_ui.py"
)


class _Screen:
    def __init__(self):
        self.workers = []
        self._console_sync_in_progress = self._console_sync_requested = False
        self._control_ready = False
        self.control_calls = 0
        self.button = SimpleNamespace(disabled=True)
        self._console_readiness_config_projection = None

    async def _sync_native_console_chat_ui(self, entered, release):
        self._console_sync_in_progress = True
        entered.set()
        try:
            await release.wait()
        finally:
            self._console_sync_in_progress = False

    def _sync_console_control_bar(self, rail_state=None):
        self.control_calls += 1
        return self._control_ready


class _Projection:
    def __init__(self, screen):
        self.screen, self.pending = screen, True

    async def _refresh(self, entered, release):
        entered.set()
        try:
            await release.wait()
            if self.screen._console_readiness_config_projection is self:
                self.screen._control_ready = True
        finally:
            self.pending = False


class _Pilot:
    async def pause(self, duration):
        await asyncio.sleep(duration)


def _load_helpers():
    source = _HELPER_PATH.read_bytes()
    nodes = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name
        in {"_await_selected_console_readiness", "_select_ready_llamacpp_console"}
    ]
    assert len(nodes) == 2
    namespace = {
        "asyncio": asyncio,
        "time": time,
        "ChatScreen": _Screen,
        "ConsoleReadinessConfigProjection": _Projection,
    }
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(_HELPER_PATH), "exec"),
        namespace,
    )
    assert source == _HELPER_PATH.read_bytes()
    return namespace


async def _scheduled_not_completed(barrier):
    delivered = asyncio.Event()
    asyncio.get_running_loop().call_soon(delivered.set)
    await delivered.wait()
    await asyncio.sleep(0)
    assert not barrier.done()


async def _projection_case(queued=False, replace=False, cancel=False, drift=False):
    screen, pilot = _Screen(), _Pilot()
    projection = _Projection(screen)
    screen._console_readiness_config_projection = projection
    entered, release = asyncio.Event(), asyncio.Event()
    original = projection._refresh(entered, release)
    screen.workers.append(SimpleNamespace(node=screen, _work=original))
    owner = None if queued else asyncio.create_task(original)
    barrier, replacement_owner = None, None
    old_defaults = _Screen._sync_console_control_bar.__defaults__
    replacement_release = asyncio.Event()
    try:
        if owner is not None:
            await entered.wait()
        barrier = asyncio.create_task(
            _load_helpers()["_await_selected_console_readiness"](
                screen, pilot, deadline=time.monotonic() + 0.5
            )
        )
        await _scheduled_not_completed(barrier)
        assert screen.control_calls == 0
        if queued:
            owner = asyncio.create_task(original)
            await entered.wait()
        if cancel:
            barrier.cancel()
            with pytest.raises(asyncio.CancelledError):
                await barrier
            assert not owner.done() and not owner.cancelled() and projection.pending
            return
        if drift:
            _Screen._sync_console_control_bar.__defaults__ = ("foreign",)
            with pytest.raises(AssertionError, match="source changed"):
                await barrier
            assert not owner.done() and not owner.cancelled()
            return
        if replace:
            replacement = _Projection(screen)
            screen._console_readiness_config_projection = replacement
            await _scheduled_not_completed(barrier)
            assert screen.control_calls == 0
            release.set()
            await owner
            assert not screen._control_ready
            await _scheduled_not_completed(barrier)
            replacement_entered = asyncio.Event()
            replacement_owner = asyncio.create_task(
                replacement._refresh(replacement_entered, replacement_release)
            )
            await replacement_entered.wait()
            replacement_release.set()
            await replacement_owner
        else:
            release.set()
            await owner
        await barrier
        assert screen.control_calls >= 1 and screen._control_ready
        assert screen.button.disabled, "Fixture must not force the actual Button state"
    finally:
        _Screen._sync_console_control_bar.__defaults__ = old_defaults
        release.set()
        replacement_release.set()
        if owner is None:
            original.close()
        else:
            await owner
        if replacement_owner is not None:
            await replacement_owner
        if barrier is not None and not barrier.done():
            barrier.cancel()
            await asyncio.gather(barrier, return_exceptions=True)


async def _direct_or_pump_case(pump=False, foreign=False):
    screen, other, pilot = _Screen(), _Screen(), _Pilot()
    target = other if foreign else screen
    entered, release, pump_release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    ended = asyncio.Event()

    async def run():
        await target._sync_native_console_chat_ui(entered, release)
        ended.set()
        if pump:
            await pump_release.wait()

    screen._control_ready = True
    owner = asyncio.create_task(run())
    barrier = None
    try:
        await entered.wait()
        barrier = asyncio.create_task(
            _load_helpers()["_await_selected_console_readiness"](
                screen, pilot, deadline=time.monotonic() + 0.5
            )
        )
        await barrier
        assert (
            not owner.done()
        ), "Current readiness does not await unrelated whole-sync or its flags"
        if not foreign:
            assert screen._console_sync_in_progress
        assert screen.control_calls == 1
        assert screen.button.disabled
    finally:
        release.set()
        pump_release.set()
        await owner
        if barrier is not None and not barrier.done():
            barrier.cancel()
            await asyncio.gather(barrier, return_exceptions=True)


async def _deadline_case():
    screen, pilot = _Screen(), _Pilot()
    projection = _Projection(screen)
    screen._console_readiness_config_projection = projection
    entered, release = asyncio.Event(), asyncio.Event()
    owner = asyncio.create_task(projection._refresh(entered, release))
    try:
        await entered.wait()
        with pytest.raises(AssertionError, match="original startup budget"):
            await _load_helpers()["_await_selected_console_readiness"](
                screen, pilot, deadline=time.monotonic() + 0.01
            )
        assert not owner.done() and not owner.cancelled() and projection.pending
        assert screen.button.disabled
    finally:
        release.set()
        await owner


async def _disabled_case():
    screen, pilot = _Screen(), _Pilot()
    screen._control_ready = True
    await _load_helpers()["_await_selected_console_readiness"](
        screen, pilot, deadline=time.monotonic() + 0.5
    )
    assert screen.control_calls == 1 and screen.button.disabled


async def _shared_setup_budget_case():
    namespace = _load_helpers()
    calls = []

    async def selector(screen, pilot, selected):
        calls.append(("selector", selected))

    def select(screen):
        calls.append(("original_sync_selection", screen))

    async def publication(screen, pilot, *, deadline):
        calls.append(("owned_publication", deadline))

    namespace.update(
        time=SimpleNamespace(monotonic=lambda: 7.0),
        _wait_for_selector=selector,
        _select_llamacpp_console=select,
        _await_selected_console_readiness=publication,
    )
    screen, pilot = object(), object()
    await namespace["_select_ready_llamacpp_console"](screen, pilot)
    assert calls == [
        ("selector", "#console-native-composer"),
        ("original_sync_selection", screen),
        ("owned_publication", 9.0),
    ]


CASES = [
    lambda: _projection_case(),
    lambda: _projection_case(queued=True),
    lambda: _projection_case(replace=True),
    lambda: _projection_case(cancel=True),
    lambda: _projection_case(drift=True),
    lambda: _direct_or_pump_case(),
    lambda: _direct_or_pump_case(pump=True),
    lambda: _direct_or_pump_case(foreign=True),
    _deadline_case,
    _disabled_case,
    _shared_setup_budget_case,
]


@pytest.mark.unit
@pytest.mark.parametrize(
    "case",
    CASES,
    ids=[
        "projection",
        "queued",
        "owner_drift",
        "cancellation",
        "source_drift",
        "unrelated_same_screen_whole_sync",
        "long_lived_pump",
        "foreign_owner",
        "deadline",
        "disabled_unchanged",
        "shared_two_seconds",
    ],
)
def test_resend_setup_waits_for_original_current_readiness_ownership(case):
    asyncio.run(asyncio.wait_for(case(), timeout=1.0))
