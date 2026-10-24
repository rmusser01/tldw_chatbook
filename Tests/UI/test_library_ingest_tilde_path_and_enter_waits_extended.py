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

import pytest
from textual.screen import Screen
from textual.widgets import Button, Input

from Tests.UI.test_library_ingest_tilde_path_and_enter_waits import (
    EXISTING,
    MISSING,
    PATH_PROBLEM_PREFIX,
    _home_with_fixture,
    _import_canvas,
    _painted,
    _gate_precheck,
    _no_submit_for,
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


async def _leave_canvas_with_escape(screen, pilot) -> None:
    """Esc in the ingest canvas: back to the hub through the rail-row seam."""
    await pilot.press("escape")
    await _wait_for_condition(
        pilot,
        lambda: not screen.query("#library-ingest-path"),
        timeout=10.0,
        message="Esc did not leave the ingest canvas",
    )


async def _return_to_canvas(screen, pilot) -> None:
    """``i`` on the hub re-selects the Import row; the staged form persists."""
    await pilot.press("i")
    await _wait_for_selector(screen, pilot, "#library-ingest-path")
    await pilot.pause()


async def test_start_while_checking_then_leaving_and_returning_imports_only_on_a_press(
    tmp_path, monkeypatch
):
    """Fix round 1, C1 (a): Start while Checking…, Esc to the hub, the check
    lands, return. A single Tab out of the field must import NOTHING (the
    reviewer saw it import); the stuck "Checking…" must not leave Enter dead."""
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
        screen.query_one("#library-ingest-start", Button).press()
        await pilot.pause(0.2)
        assert submitted == []
        await _leave_canvas_with_escape(screen, pilot)
        gate.set()
        await _no_submit_for(pilot, submitted, 1.0, "imported while another canvas showed")
        await _return_to_canvas(screen, pilot)
        assert screen._ingest_state.form.path == EXISTING, "the staged form was lost"
        assert screen._ingest_state.form.preflight_checking is False
        screen.query_one("#library-ingest-path", Input).focus()
        await pilot.pause()
        await pilot.press("tab")
        await _no_submit_for(pilot, submitted, 1.0, "a Tab out of the field imported")
        assert toasts == []
        # Enter is not dead after the round trip.
        screen.query_one("#library-ingest-path", Input).focus()
        await pilot.pause()
        await pilot.press("enter")
        await _wait_for_condition(
            pilot, lambda: submitted, timeout=10.0, message="Enter was dead on return"
        )
        assert len(submitted) == 1
        assert submitted[0]["source_path"] == str(handbook)


async def test_enter_while_checking_then_leaving_never_imports_on_another_canvas(
    tmp_path, monkeypatch
):
    """Fix round 1, C1 (b): Enter while Checking…, then Esc. The blur
    re-check must not carry the arm; nothing imports while the hub shows."""
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
        await pilot.press("enter")
        await pilot.pause(0.2)
        assert submitted == []
        await _leave_canvas_with_escape(screen, pilot)
        gate.set()
        await _no_submit_for(
            pilot, submitted, 1.5, "the armed Enter imported while the hub showed"
        )
        assert toasts == []
        await _return_to_canvas(screen, pilot)
        screen.query_one("#library-ingest-path", Input).focus()
        await pilot.pause()
        await pilot.press("enter")
        await _wait_for_condition(
            pilot, lambda: submitted, timeout=10.0, message="Enter was dead on return"
        )
        assert len(submitted) == 1
        assert submitted[0]["source_path"] == str(handbook)


async def test_a_modal_covering_the_screen_disarms_a_waiting_submit(
    tmp_path, monkeypatch
):
    """Fix round 1, C1 rule 3 / M2: a result landing while another screen
    covers the Library must not import; Enter works once it is uncovered."""
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
        await pilot.press("enter")
        await pilot.pause(0.2)
        assert submitted == []
        cover = Screen()
        host.push_screen(cover)
        await _wait_for_condition(
            pilot, lambda: host.screen is cover, timeout=5.0, message="cover not shown"
        )
        gate.set()
        await _no_submit_for(pilot, submitted, 1.5, "imported under a covering screen")
        host.pop_screen()
        await _wait_for_condition(
            pilot, lambda: host.screen is screen, timeout=5.0, message="cover not popped"
        )
        screen.query_one("#library-ingest-path", Input).focus()
        await pilot.pause()
        await pilot.press("enter")
        await _wait_for_condition(
            pilot, lambda: submitted, timeout=10.0, message="Enter was dead after the cover"
        )
        assert len(submitted) == 1
        assert submitted[0]["source_path"] == str(handbook)


async def test_surrounding_whitespace_checks_and_imports_the_same_path(
    tmp_path, monkeypatch
):
    """Fix round 1, I1: a fast Enter on ``"  ~/…/team-handbook.md  "`` must
    check and import the same path a slow Enter would (the trigger used to
    pass the unstripped text, so a leading space defeated the expansion)."""
    handbook = _home_with_fixture(tmp_path, monkeypatch)
    submitted: list = []
    toasts: list[str] = []
    gate = _gate_precheck(monkeypatch)
    app, screen, host = _import_canvas(tmp_path, submitted, monkeypatch, toasts)
    async with host.run_test(size=(160, 45)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-ingest-path")
        await _type_path(screen, pilot, "  " + EXISTING + "  ")
        await pilot.press("enter")
        gate.set()
        await _wait_for_condition(
            pilot,
            lambda: submitted,
            timeout=10.0,
            message=lambda: (
                "the padded path never imported; verdict: "
                f"{screen._ingest_state.form.preflight}"
            ),
        )
        assert len(submitted) == 1
        assert submitted[0]["source_path"] == str(handbook)
        assert toasts == []


async def test_enter_on_a_stale_not_found_verdict_rechecks_the_path(
    tmp_path, monkeypatch
):
    """Fix round 1, I3: the footer says "enter check this path"; with a
    current not-found verdict, creating the file and pressing Enter must
    re-check (and then import), not leave the stale sentence on screen."""
    _home_with_fixture(tmp_path, monkeypatch)
    submitted: list = []
    toasts: list[str] = []
    app, screen, host = _import_canvas(tmp_path, submitted, monkeypatch, toasts)
    async with host.run_test(size=(160, 45)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-ingest-path")
        await _type_path(screen, pilot, MISSING)
        await _wait_for_precheck(screen, pilot)
        await _wait_painted_once(host, pilot, PATH_PROBLEM_PREFIX)
        created = tmp_path / "fixtures" / "import-docs" / "nope.md"
        created.write_text("# Now it exists\n", encoding="utf-8")
        await pilot.press("enter")
        await _wait_for_condition(
            pilot,
            lambda: submitted,
            timeout=10.0,
            message="Enter did not re-check the path and import the new file",
        )
        assert submitted[0]["source_path"] == str(created)
        assert PATH_PROBLEM_PREFIX not in _painted(host)
        assert toasts == []
