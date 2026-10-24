"""Extended arms for TASK-34000.20 -- NOT in the UI PR-lane census.

The gated sibling (``test_library_ingest_tilde_path_and_enter_waits.py``)
carries the three 160x45 boots. These are the slower / wider arms:

- the AC#1 and AC#2 walks at 100x30 (the review's second size);
- owner ruling 2026-10-23 (question 1): the Start import BUTTON waits for
  the pre-check exactly as Enter does -- one shared seam;
- Review Focus 2: a text edit while a submit is armed drops the armed
  submit; the new text imports only on its own Enter.
"""

from __future__ import annotations

import threading

import pytest
from textual.widgets import Button, Input

from Tests.UI.test_library_ingest_tilde_path_and_enter_waits import (
    EXISTING,
    MISSING,
    PATH_PROBLEM_PREFIX,
    _home_with_fixture,
    _import_canvas,
    _painted,
    _type_path,
    _wait_for_precheck,
    _wait_painted_once,
)
from Tests.UI.test_library_shell import (
    _active_library_screen,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.UI.Screens import library_screen as library_screen_module

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

SMALL = (100, 30)


async def test_an_existing_tilde_path_imports_at_100x30(tmp_path, monkeypatch):
    handbook = _home_with_fixture(tmp_path, monkeypatch)
    submitted: list = []
    toasts: list[str] = []
    app, screen, host = _import_canvas(tmp_path, submitted, monkeypatch, toasts)
    async with host.run_test(size=SMALL) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-ingest-path")
        await _type_path(screen, pilot, EXISTING)
        await _wait_for_precheck(screen, pilot)
        assert screen._ingest_state.form.preflight.errors == []
        assert "dangerous" not in _painted(host)
        await pilot.press("enter")
        await _wait_for_condition(
            pilot, lambda: submitted, timeout=10.0, message="Enter never imported"
        )
        assert submitted[0]["source_path"] == str(handbook)
        assert toasts == []


async def test_a_missing_tilde_path_is_one_sentence_at_100x30(tmp_path, monkeypatch):
    _home_with_fixture(tmp_path, monkeypatch)
    submitted: list = []
    toasts: list[str] = []
    app, screen, host = _import_canvas(tmp_path, submitted, monkeypatch, toasts)
    async with host.run_test(size=SMALL) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-ingest-path")
        await _type_path(screen, pilot, MISSING)
        await _wait_for_precheck(screen, pilot)
        assert "dangerous" not in _painted(host)
        painted = await _wait_painted_once(host, pilot, PATH_PROBLEM_PREFIX)
        assert "dangerous" not in painted
        assert "Path not found" not in painted
        await pilot.press("enter")
        await pilot.pause(0.3)
        assert submitted == []
        assert toasts == []


def _gate_precheck(monkeypatch) -> threading.Event:
    gate = threading.Event()
    real_analyze = library_screen_module.analyze_path

    def gated_analyze(path, **kwargs):
        if str(path).startswith("~"):
            gate.wait(timeout=10)
        return real_analyze(path, **kwargs)

    monkeypatch.setattr(library_screen_module, "analyze_path", gated_analyze)
    return gate


async def test_start_import_while_checking_waits_too(tmp_path, monkeypatch):
    """Owner ruling (question 1): Start import is gated at the same seam as Enter."""
    handbook = _home_with_fixture(tmp_path, monkeypatch)
    submitted: list = []
    toasts: list[str] = []
    gate = _gate_precheck(monkeypatch)
    app, screen, host = _import_canvas(tmp_path, submitted, monkeypatch, toasts)
    async with host.run_test(size=(160, 45)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-ingest-path")
        await _type_path(screen, pilot, EXISTING)
        start = screen.query_one("#library-ingest-start", Button)
        assert start.disabled is False
        start.press()
        await pilot.pause(1.0)
        assert submitted == [], "Start imported before the pre-check had a verdict"
        assert toasts == []
        gate.set()
        await _wait_for_condition(
            pilot,
            lambda: submitted,
            timeout=10.0,
            message="the armed Start never fired once the check landed",
        )
        await pilot.pause(0.3)
        assert len(submitted) == 1
        assert submitted[0]["source_path"] == str(handbook)


async def test_a_text_edit_while_a_submit_is_armed_drops_it(tmp_path, monkeypatch):
    """Review Focus 2: an armed submit dies with the text it was armed for."""
    handbook = _home_with_fixture(tmp_path, monkeypatch)
    submitted: list = []
    toasts: list[str] = []
    gate = _gate_precheck(monkeypatch)
    app, screen, host = _import_canvas(tmp_path, submitted, monkeypatch, toasts)
    async with host.run_test(size=(160, 45)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-ingest-path")
        # Arm against text A (an existing file) while its check is held...
        await _type_path(screen, pilot, EXISTING)
        await pilot.press("enter")
        await pilot.pause(0.2)
        assert screen._ingest_state.submit_armed_generation is not None
        # ...then edit the field to text B before A's check can land.
        await _type_path(screen, pilot, MISSING)
        assert screen._ingest_state.submit_armed_generation is None
        gate.set()
        await _wait_for_precheck(screen, pilot)
        await pilot.pause(0.5)
        assert submitted == [], "A's armed submit survived the edit to B"
        assert toasts == []
        await _wait_painted_once(host, pilot, PATH_PROBLEM_PREFIX)
        # B is wrong, so B's own Enter imports nothing either.
        await pilot.press("enter")
        await pilot.pause(0.3)
        assert submitted == []
        # And a real file typed afterwards imports only on ITS Enter.
        await _type_path(screen, pilot, EXISTING)
        await _wait_for_precheck(screen, pilot)
        assert submitted == []
        await pilot.press("enter")
        await _wait_for_condition(
            pilot, lambda: submitted, timeout=10.0, message="Enter never imported"
        )
        assert submitted[0]["source_path"] == str(handbook)
        path_input = screen.query_one("#library-ingest-path", Input)
        assert path_input is not None
