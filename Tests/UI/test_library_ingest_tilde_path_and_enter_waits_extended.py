"""Extended arms for TASK-34000.20 -- NOT in the UI PR-lane census.

The gated sibling (``test_library_ingest_tilde_path_and_enter_waits.py``)
carries the five 160x45 boots. These are the slower / wider arms:

- the AC#1 and AC#2 walks at 100x30 (the review's second size);
- owner ruling 2026-10-23 (question 1): the Start import BUTTON waits for
  the pre-check exactly as Enter does -- one shared seam;
- Review Focus 2: a text edit while a submit is armed drops the armed
  submit; the new text imports only on its own Enter;
- the fix-round disarm routes (leave/return, modal, cancelled Browse) and
  the re-check-on-Enter pins (a stale not-found verdict; an empty folder
  that gained a file). The suspend/resume disarm arm moved to the gated
  sibling (P1-C final fix round, I-D ruling).
"""

from __future__ import annotations

import pytest
from textual.screen import Screen
from textual.widgets import Button, Input

from Tests.UI.test_library_ingest_tilde_path_and_enter_waits import (
    EXISTING,
    MISSING,
    PATH_PROBLEM_PREFIX,
    _arm_enter_under_held_check,
    _enter_imports_after,
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
    """Fix round 1, C1 (a): Start while Checking…, Esc to the hub, return,
    the check lands. A single Tab out of the field must import NOTHING (the
    reviewer saw it import); the stuck "Checking…" must not leave Enter dead.

    Fix round 2 (R2-2): focus sits on the Start BUTTON before the press and
    stays there through Esc, so the path field never blurs and no blur
    re-check can re-point the arm -- the only thing that can drop it is the
    rail-row pause's own disarm. Mutation-checked: with that disarm replaced
    by ``pass`` this arm fails at the Tab assertion.
    """
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
        start.focus()
        await pilot.pause()
        assert screen.focused is start
        start.press()
        await pilot.pause(0.2)
        assert submitted == []
        assert screen._ingest_state.submit_armed_generation is not None
        await _leave_canvas_with_escape(screen, pilot)
        await _return_to_canvas(screen, pilot)
        assert screen._ingest_state.form.path == EXISTING, "the staged form was lost"
        assert screen._ingest_state.form.preflight_checking is False
        gate.set()
        await pilot.pause(0.5)
        screen.query_one("#library-ingest-path", Input).focus()
        await pilot.pause()
        await pilot.press("tab")
        await _no_submit_for(pilot, submitted, 1.5, "a Tab out of the field imported")
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


async def test_enter_on_a_current_empty_folder_verdict_rechecks_and_imports_once(
    tmp_path, monkeypatch
):
    """P1-C final fix round, I-B: the footer says "enter check this path" for
    EVERY shut gate, but only a path problem re-checked. An empty folder that
    has since gained a file must re-check on the first Enter and then import
    once from that press -- not toast "This folder is empty" twice and import
    nothing. The same seam covers an all-unsupported folder, a lone 0-byte
    file and a selection the targeted server takes nothing of (one flag,
    ``selection_has_nothing_importable``)."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    submitted: list = []
    toasts: list[str] = []
    app, screen, host = _import_canvas(tmp_path, submitted, monkeypatch, toasts)
    async with host.run_test(size=(160, 45)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-ingest-path")
        await _type_path(screen, pilot, str(inbox))
        await _wait_for_precheck(screen, pilot)
        assert screen._ingest_state.form.preflight.total_files == 0
        await _wait_painted_once(host, pilot, "This folder is empty")
        (inbox / "a.md").write_text("# Now there is a file\n", encoding="utf-8")
        await pilot.press("enter")
        await _wait_for_condition(
            pilot,
            lambda: submitted,
            timeout=10.0,
            message="Enter did not re-check the folder and import the new file",
        )
        await pilot.pause(0.3)
        assert len(submitted) == 1, "the one Enter imported more than once"
        assert submitted[0]["source_path"] == str(inbox)
        assert toasts == [], toasts
        # A successful submit clears the path (and with it the stale verdict).
        assert screen._ingest_state.form.path == ""
        assert "This folder is empty" not in _painted(host)


async def test_a_cancelled_browse_picker_drops_a_waiting_submit(tmp_path, monkeypatch):
    """Fix round 2, R2-1: Browse… opened and CANCELLED (no path adopted) still
    suspended the screen; the check landing afterwards must not import."""
    handbook = _home_with_fixture(tmp_path, monkeypatch)
    submitted: list = []
    toasts: list[str] = []
    gate = _gate_precheck(monkeypatch)
    app, screen, host = _import_canvas(tmp_path, submitted, monkeypatch, toasts)
    async with host.run_test(size=(160, 45)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-ingest-path")
        await _arm_enter_under_held_check(screen, pilot, submitted)
        await pilot.click("#library-ingest-browse")
        await _wait_for_condition(
            pilot, lambda: host.screen is not screen, timeout=10.0, message="no picker"
        )
        host.screen.dismiss(None)
        await _wait_for_condition(
            pilot, lambda: host.screen is screen, timeout=10.0, message="picker stayed"
        )
        assert screen._ingest_state.form.path == EXISTING
        gate.set()
        await _no_submit_for(pilot, submitted, 1.5, "imported after a cancelled Browse…")
        assert toasts == []
        await _enter_imports_after(screen, pilot, submitted, handbook)


async def test_a_modal_dismissed_before_the_check_lands_drops_a_waiting_submit(
    tmp_path, monkeypatch
):
    """Fix round 2, R2-1: a modal pushed and popped while the check is still
    running; the result lands on the uncovered screen and must not import."""
    handbook = _home_with_fixture(tmp_path, monkeypatch)
    submitted: list = []
    toasts: list[str] = []
    gate = _gate_precheck(monkeypatch)
    app, screen, host = _import_canvas(tmp_path, submitted, monkeypatch, toasts)
    async with host.run_test(size=(160, 45)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-ingest-path")
        await _arm_enter_under_held_check(screen, pilot, submitted)
        cover = Screen()
        host.push_screen(cover)
        await _wait_for_condition(
            pilot, lambda: host.screen is cover, timeout=5.0, message="cover not shown"
        )
        host.pop_screen()
        await _wait_for_condition(
            pilot, lambda: host.screen is screen, timeout=5.0, message="cover not popped"
        )
        gate.set()
        await _no_submit_for(pilot, submitted, 1.5, "imported after a dismissed modal")
        assert toasts == []
        await _enter_imports_after(screen, pilot, submitted, handbook)


async def test_a_result_landing_between_a_keystroke_and_its_handler_imports_nothing(
    tmp_path, monkeypatch
):
    """Fix round 2, R2-3: the fire-time text check reads the LIVE Input. The
    form echo lags the keystroke until ``Input.Changed`` is handled; a result
    delivered in that gap must see the new text and disarm."""
    from tldw_chatbook.Library.ingest_types import PreflightResult

    handbook = _home_with_fixture(tmp_path, monkeypatch)
    submitted: list = []
    toasts: list[str] = []
    gate = _gate_precheck(monkeypatch)
    app, screen, host = _import_canvas(tmp_path, submitted, monkeypatch, toasts)
    async with host.run_test(size=(160, 45)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-ingest-path")
        await _arm_enter_under_held_check(screen, pilot, submitted)
        generation = screen._ingest_state.preflight_generation
        accepted = PreflightResult(
            type_groups={"generic": [str(handbook)]},
            warnings=[],
            errors=[],
            total_size=handbook.stat().st_size,
            truncated=False,
            total_files=1,
        )
        path_input = screen.query_one("#library-ingest-path", Input)
        # The keystroke's Changed message is queued but NOT yet handled when
        # the worker's result is applied -- no await in between.
        path_input.value = MISSING
        assert screen._ingest_state.form.path == EXISTING  # the echo lags
        screen._apply_library_ingest_preflight_result(accepted, generation, EXISTING)
        await _no_submit_for(
            pilot, submitted, 1.0, "imported the OLD text while the field showed new text"
        )
        assert screen._ingest_state.submit_armed_generation is None
        assert toasts == []
        # Release the held worker thread before the app exits (its result is
        # for a superseded generation and is dropped).
        gate.set()
        await pilot.pause(0.5)
