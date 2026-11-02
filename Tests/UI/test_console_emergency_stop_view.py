"""TASK-33620.1 review follow-ups: the Console's cached emergency-stop view.

``emergency_stop_hold`` keeps one copy of the stop on the screen. These tests
pin the view against the real stop file (the production reader and writer,
in a temporary directory) and a stand-in screen, without mounting the app:

* an arrival read that started before this Console's own write must not
  overwrite the newer state when it lands;
* a send that would take the received-intent fast path before the arrival
  read landed must not resolve the stop path (a guarded config read) on the
  UI loop.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.emergency_stop import (
    EmergencyStopState,
    read_emergency_stop,
    set_emergency_stop,
)
from tldw_chatbook.UI.Console_Modules import emergency_stop_hold

# ``Tests/UI/conftest.py`` imports the app (a guarded config read) for every
# test here; the per-test profile redirect fails that admission closed. Each
# test writes only under ``tmp_path`` (``Tests/conftest.py``
# ``bootstrap_profile``).
pytestmark = pytest.mark.bootstrap_profile


class _Screen:
    """The attributes ``emergency_stop_hold`` reads on the Console screen."""

    _console_emergency_stop_copy = ""
    is_mounted = True

    def __init__(self) -> None:
        self.notices: list[str] = []
        self.app_instance = SimpleNamespace(
            console_emergency_stop_write=None,
            notify=lambda message, **_kwargs: self.notices.append(str(message)),
        )
        self.drafts_cleared = 0

    def _clear_console_composer_draft(self) -> None:
        self.drafts_cleared += 1

    def _console_composer_or_none(self):
        return None

    def _sync_console_control_bar(self) -> None:
        """The repaint every changed view triggers (a no-op here)."""


async def _wait_for(event: threading.Event) -> None:
    for _ in range(200):
        if event.is_set():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("the arrival read never started")


@pytest.mark.asyncio
@pytest.mark.parametrize("stale_read_active", [True, False])
async def test_an_arrival_read_older_than_a_write_does_not_overwrite_it(
    tmp_path: Path, monkeypatch, stale_read_active: bool
) -> None:
    """The arrival read and a write each record their result when they return.

    The read here starts first and returns last, carrying the state it saw
    before the write: the hold must not come back after a Clear, nor vanish
    after a fresh stop.
    """
    path = tmp_path / "emergency_stop.json"
    if stale_read_active:
        set_emergency_stop(path, reason="before the clear")
    started = threading.Event()
    proceed = threading.Event()

    def slow_read() -> tuple[Path, EmergencyStopState]:
        state = read_emergency_stop(path)
        started.set()
        proceed.wait(5)
        return path, state

    monkeypatch.setattr(emergency_stop_hold, "_read_default_stop", slow_read)
    screen = _Screen()
    screen._console_emergency_stop_path = path
    arrival = asyncio.create_task(emergency_stop_hold.refresh(screen))
    await _wait_for(started)
    if stale_read_active:
        screen._console_emergency_stop_copy = emergency_stop_hold.HELD_COPY
        await emergency_stop_hold.clear_from_strip(screen)
    else:
        await emergency_stop_hold.run_command(screen, SimpleNamespace(args="freeze"))
    proceed.set()
    await arrival

    assert read_emergency_stop(path).active is not stale_read_active
    expected = "" if stale_read_active else emergency_stop_hold.HELD_COPY
    assert (
        screen._console_emergency_stop_copy == expected
    ), "an arrival read that started before the write overwrote its state"


def test_an_unknown_stop_path_is_not_resolved_on_the_ui_loop(
    tmp_path: Path, monkeypatch
) -> None:
    """Before the arrival read lands the send leaves the fast path instead.

    Resolving the default path is a guarded config read (measured at up to
    98 ms in TASK-33620.15.24); the dispatch path reads the stop itself.
    """
    resolved: list[Path] = []

    def resolve() -> Path:
        resolved.append(tmp_path / "emergency_stop.json")
        return resolved[-1]

    monkeypatch.setattr(emergency_stop_hold, "default_emergency_stop_path", resolve)
    screen = _Screen()

    assert emergency_stop_hold.received_send_held(screen) is True
    assert resolved == [], "the stop path was resolved on the UI loop"
