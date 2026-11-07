"""Library Import accepts ``~/`` paths and Enter waits for the pre-check (TASK-34000.20).

Review L-05 (``Docs/QA/notes-library-ux-review-2026-10-02/verify/nl-v-l-05``):
the pre-check validated the RAW field text, so ``~/fixtures/…/team-handbook.md``
-- a file that exists -- was called "Path contains dangerous pattern: ~/"
while the submit path expanded the same text and imported it; a missing
``~/`` path painted three sentences for one typo; and Enter inside the 0.8 s
debounce imported a path the pre-check had not yet seen (capture 07).

Five boots at 160x45 against the real ``LibraryScreen`` with the app
stylesheet (``LibraryHarness``). Every claim is what the user sees (painted
text) or what reached the app's ``submit_library_ingest_job`` seam. The
fourth and fifth boots pin the two disarm routes the whole-branch review
ruled into a PR lane (fix round 1 C1 (c): a Browse pick; fix round 2 R2-1: a
screen suspend); the other disarm arms live in the ``_extended`` sibling.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from textual.screen import Screen
from textual.widgets import Input

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_shell import (
    LibraryHarness,
    _active_library_screen,
    _seed_conversations,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.Constants import LIBRARY_NAV_CONTEXT_INGEST
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Library.library_ingest_jobs import LibraryIngestJobRegistry
from tldw_chatbook.UI.Screens import library_screen as library_screen_module
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

SIZE = (160, 45)
PATH_PROBLEM_PREFIX = "Can't find that path"
EXISTING = "~/fixtures/import-docs/team-handbook.md"
MISSING = "~/fixtures/import-docs/nope.md"


def _home_with_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Make ``tmp_path`` the user's home and plant the review's fixture file."""
    monkeypatch.setenv("HOME", str(tmp_path))
    docs = tmp_path / "fixtures" / "import-docs"
    docs.mkdir(parents=True)
    handbook = docs / "team-handbook.md"
    handbook.write_text("# Team handbook\n\nWelcome.\n", encoding="utf-8")
    return handbook


def _import_canvas(
    tmp_path: Path, submitted: list, monkeypatch: pytest.MonkeyPatch, toasts: list[str]
):
    app = _build_test_app()
    _seed_conversations(app, ())
    # Start is gated on a media DB and a job registry being present (the
    # faked TldwCli ships neither); real ones, as the directory journeys do.
    app.media_db = MediaDatabase(str(tmp_path / "media.db"), client_id="tilde-ui")
    app.library_ingest_jobs = LibraryIngestJobRegistry()
    # The real seam's shape: ``_do_submit_ingest`` calls
    # ``submit_library_ingest_job(**submit_kwargs)`` with ``source_path=``.
    app.submit_library_ingest_job = lambda **kwargs: (
        submitted.append(kwargs) or "job-34000"
    )
    # The warning toast seam is ``self.app_instance.notify`` (the screen's
    # ``_notify_library_ingest_warning``); record without rendering, since
    # ``app`` is the faked TldwCli, not the running harness.
    monkeypatch.setattr(
        app, "notify", lambda message, *a, **k: toasts.append(str(message))
    )
    screen = LibraryScreen(app)
    screen.apply_navigation_context({LIBRARY_NAV_CONTEXT_INGEST: True})
    host = LibraryHarness(app, screen=screen)
    return app, screen, host


def _painted(host) -> str:
    return "\n".join(strip.text for strip in host.screen._compositor.render_strips())


async def _type_path(screen, pilot, text: str) -> None:
    path_input = screen.query_one("#library-ingest-path", Input)
    path_input.focus()
    await pilot.pause()
    path_input.value = text  # posts Input.Changed exactly as typing does
    await pilot.pause()


def _gate_precheck(monkeypatch: pytest.MonkeyPatch) -> threading.Event:
    """Hold every pre-check of a ``~``-prefixed path until the gate is set.

    Only the PRE-CHECK worker sees the raw ``~/`` text; the readiness worker
    that runs after a submit receives the expanded source and must not be
    held, or the base race could never show itself.
    """
    gate = threading.Event()
    real_analyze = library_screen_module.analyze_path

    def gated_analyze(path, **kwargs):
        if str(path).lstrip().startswith("~"):
            gate.wait(timeout=10)
        return real_analyze(path, **kwargs)

    monkeypatch.setattr(library_screen_module, "analyze_path", gated_analyze)
    return gate


async def _no_submit_for(pilot, submitted: list, seconds: float, why: str) -> None:
    """Pin an absence: nothing reaches the seam for ``seconds`` wall-clock."""
    await pilot.pause(seconds)
    assert submitted == [], why


def _precheck_is_current(screen) -> bool:
    form = screen._ingest_state.form
    # ``preflight_path`` is this task's new field; on the base commit the
    # attribute does not exist, so degrade to "a result landed" there and
    # let the assertions after the wait name the real failure.
    described = getattr(form, "preflight_path", None)
    return (
        not form.preflight_checking
        and form.preflight is not None
        and (described is None or described == form.path.strip())
    )


async def _wait_for_precheck(screen, pilot) -> None:
    await _wait_for_condition(
        pilot,
        lambda: _precheck_is_current(screen),
        timeout=10.0,
        message="the pre-check for the current text never landed",
    )


async def _wait_painted_once(host, pilot, text: str) -> str:
    """Wait until ``text`` is painted exactly once, then return the paint.

    The summary under the field is mounted in place after the result lands,
    so the paint can lag the state by a frame (seen at 100x30: the ``Static``
    was in the DOM with the right content before it had a layout).
    """
    await _wait_for_condition(
        pilot,
        lambda: _painted(host).count(text) == 1,
        timeout=10.0,
        message=lambda: f"{text!r} was painted {_painted(host).count(text)} times",
    )
    return _painted(host)


async def _wait_painted(host, pilot, text: str) -> str:
    """Wait until ``text`` is painted at least once, then return the paint.

    (P1-C final fix round, M4) A "not painted" belt read on the frame the
    state landed carried no weight: the summary had not been painted yet
    either way. Read it after the verdict's OWN copy is on screen.
    """
    await _wait_for_condition(
        pilot,
        lambda: text in _painted(host),
        timeout=10.0,
        message=lambda: f"{text!r} never painted",
    )
    return _painted(host)


async def _wait_for_recheck(screen, pilot, before) -> None:
    """Wait for a verdict that is NEWER than ``before`` and describes the field.

    (P1-C final fix round, I-C) Enter on a current verdict starts a re-check
    whose worker lands later; ``_wait_for_precheck`` alone is vacuous while
    the stored verdict is still the old, current one. A re-check produces a
    new ``PreflightResult`` object, so identity tells the two apart.
    """
    await _wait_for_condition(
        pilot,
        lambda: screen._ingest_state.form.preflight is not before
        and _precheck_is_current(screen),
        timeout=10.0,
        message="Enter did not re-check the path",
    )


async def test_an_existing_tilde_path_passes_the_precheck_and_enter_imports_it(
    tmp_path, monkeypatch
):
    """AC#1: ``~/`` is the user's home; the pre-check reads it and Enter imports it."""
    handbook = _home_with_fixture(tmp_path, monkeypatch)
    submitted: list = []
    toasts: list[str] = []
    app, screen, host = _import_canvas(tmp_path, submitted, monkeypatch, toasts)
    async with host.run_test(size=SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-ingest-path")

        await _type_path(screen, pilot, EXISTING)
        await _wait_for_precheck(screen, pilot)
        assert screen._ingest_state.form.preflight.errors == []
        # (M4) The belts read AFTER the verdict's own copy is painted.
        painted = await _wait_painted(host, pilot, "plain text file")
        assert "dangerous" not in painted
        assert PATH_PROBLEM_PREFIX not in painted

        await pilot.press("enter")
        await _wait_for_condition(
            pilot, lambda: submitted, timeout=10.0, message="Enter never imported"
        )
        assert len(submitted) == 1
        assert submitted[0]["source_path"] == str(handbook)
        assert toasts == []


async def test_a_missing_tilde_path_shows_exactly_one_not_found_message_and_enter_imports_nothing(
    tmp_path, monkeypatch
):
    """AC#2 + AC#3: one plain sentence under the field; Enter imports nothing, toasts nothing."""
    _home_with_fixture(tmp_path, monkeypatch)
    submitted: list = []
    toasts: list[str] = []
    app, screen, host = _import_canvas(tmp_path, submitted, monkeypatch, toasts)
    async with host.run_test(size=SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-ingest-path")

        await _type_path(screen, pilot, MISSING)
        await _wait_for_precheck(screen, pilot)
        # (M4) The belts read AFTER the one sentence is painted, not on the
        # frame the state landed.
        painted = await _wait_painted_once(host, pilot, PATH_PROBLEM_PREFIX)
        assert "dangerous" not in painted
        assert "Path not found" not in painted

        # (fix round 1, I3) Enter on a CURRENT not-found verdict re-checks the
        # path -- the footer promises "enter check this path" -- and the
        # re-check's verdict decides. (P1-C final fix round, I-C) Wait for
        # THAT verdict and its paint before reading anything: on a timer the
        # field still showed "Checking…" at 0.3 s under CI load (PR #3234's
        # failure class), and ``submitted == []`` was vacuous until the
        # re-check had landed.
        before = screen._ingest_state.form.preflight
        await pilot.press("enter")
        await _wait_for_recheck(screen, pilot, before)
        assert screen._ingest_state.form.preflight.path_invalid
        painted = await _wait_painted_once(host, pilot, PATH_PROBLEM_PREFIX)
        assert "dangerous" not in painted
        assert "Path not found" not in painted
        await _no_submit_for(
            pilot, submitted, 0.3, "Enter imported a path the re-check refused"
        )
        assert toasts == []
        assert _painted(host).count(PATH_PROBLEM_PREFIX) == 1


async def test_enter_before_the_precheck_lands_waits_for_it_and_imports_only_an_accepted_path(
    tmp_path, monkeypatch
):
    """AC#3 both halves, with the pre-check held open by a gate.

    Arm B first (existing path): on the base commit Enter inside the debounce
    imports before any verdict exists (the review's capture 07). Arm A
    (missing path): Enter must neither import nor toast, and when the held
    check lands saying the path is wrong, the armed submit fires nothing.
    The gate holds past the 0.8 s debounce on purpose: a debounce fire while
    a deliberate check is in flight must not drop the armed submit.
    """
    handbook = _home_with_fixture(tmp_path, monkeypatch)
    submitted: list = []
    toasts: list[str] = []
    gate = _gate_precheck(monkeypatch)
    app, screen, host = _import_canvas(tmp_path, submitted, monkeypatch, toasts)
    async with host.run_test(size=SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-ingest-path")

        # Arm B: an existing path, Enter before any verdict.
        await _type_path(screen, pilot, EXISTING)
        await pilot.press("enter")
        await pilot.pause(1.0)
        assert submitted == [], "Enter imported before the pre-check had a verdict"
        gate.set()
        await _wait_for_condition(
            pilot,
            lambda: submitted,
            timeout=10.0,
            message="the armed submit never fired once the check landed",
        )
        await pilot.pause(0.3)
        assert len(submitted) == 1
        assert submitted[0]["source_path"] == str(handbook)
        assert toasts == []

        # Arm A: a missing path, Enter before any verdict.
        gate.clear()
        await _type_path(screen, pilot, MISSING)
        await pilot.press("enter")
        await pilot.pause(1.0)
        assert len(submitted) == 1
        assert toasts == []
        gate.set()
        await _wait_for_precheck(screen, pilot)
        await pilot.pause(0.3)
        assert len(submitted) == 1, "the armed submit imported a path the check refused"
        assert toasts == []
        await _wait_painted_once(host, pilot, PATH_PROBLEM_PREFIX)


async def test_a_browse_pick_while_a_submit_is_armed_imports_nothing(
    tmp_path, monkeypatch
):
    """Fix round 1, C1 (c): Enter on A while A's check is held, then Browse…
    picks B. B must NOT be submitted without a press: the arm was for A's
    exact text, and a different path through the picker drops it. B still
    imports on its own Enter afterwards.
    """
    handbook = _home_with_fixture(tmp_path, monkeypatch)
    other = handbook.with_name("glossary.md")
    other.write_text("# Glossary\n", encoding="utf-8")
    submitted: list = []
    toasts: list[str] = []
    gate = _gate_precheck(monkeypatch)
    app, screen, host = _import_canvas(tmp_path, submitted, monkeypatch, toasts)
    async with host.run_test(size=SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-ingest-path")

        await _type_path(screen, pilot, EXISTING)
        await pilot.press("enter")
        await pilot.pause(0.2)
        assert submitted == []
        # The real Browse… flow: the button pushes the picker, the picker's
        # result runs the controller's callback (adopt + re-check).
        await pilot.click("#library-ingest-browse")
        await _wait_for_condition(
            pilot,
            lambda: host.screen is not screen,
            timeout=10.0,
            message="Browse… did not open the picker",
        )
        host.screen.dismiss(other)
        await _wait_for_condition(
            pilot,
            lambda: host.screen is screen
            and screen._ingest_state.form.path == str(other),
            timeout=10.0,
            message="the picker did not hand the path back",
        )
        gate.set()
        await _wait_for_precheck(screen, pilot)
        await _no_submit_for(
            pilot, submitted, 1.0, "a Browse pick imported B under A's armed Enter"
        )
        assert toasts == []
        # B imports only on its own press.
        screen.query_one("#library-ingest-path", Input).focus()
        await pilot.pause()
        await pilot.press("enter")
        await _wait_for_condition(
            pilot, lambda: submitted, timeout=10.0, message="Enter never imported B"
        )
        assert len(submitted) == 1
        assert submitted[0]["source_path"] == str(other)


async def _arm_enter_under_held_check(screen, pilot, submitted) -> None:
    await _type_path(screen, pilot, EXISTING)
    await pilot.press("enter")
    await pilot.pause(0.2)
    assert submitted == []
    assert screen._ingest_state.submit_armed_generation is not None


async def _enter_imports_after(screen, pilot, submitted, handbook) -> None:
    """Enter still works once the user is back on the canvas."""
    screen.query_one("#library-ingest-path", Input).focus()
    await pilot.pause()
    await pilot.press("enter")
    await _wait_for_condition(
        pilot, lambda: submitted, timeout=10.0, message="Enter was dead afterwards"
    )
    assert len(submitted) == 1
    assert submitted[0]["source_path"] == str(handbook)


async def test_a_screen_suspend_and_resume_drops_a_waiting_submit(tmp_path, monkeypatch):
    """Fix round 2, R2-1: another tab (``switch_screen``) suspends the Library
    screen; the check landing AFTER resume must not import.

    (P1-C final fix round, I-D ruling) Moved here from the ``_extended``
    sibling so the suspend disarm -- the last C1 route -- runs on every pull
    request; the Browse arm above stays green without it, so it alone could
    not carry the guarantee.
    """
    handbook = _home_with_fixture(tmp_path, monkeypatch)
    submitted: list = []
    toasts: list[str] = []
    gate = _gate_precheck(monkeypatch)
    app, screen, host = _import_canvas(tmp_path, submitted, monkeypatch, toasts)
    async with host.run_test(size=SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-ingest-path")
        await _arm_enter_under_held_check(screen, pilot, submitted)
        other = Screen()
        host.install_screen(screen, name="library")
        host.install_screen(other, name="other-tab")
        host.switch_screen("other-tab")
        await _wait_for_condition(
            pilot, lambda: host.screen is other, timeout=5.0, message="tab not shown"
        )
        host.switch_screen("library")
        await _wait_for_condition(
            pilot, lambda: host.screen is screen, timeout=5.0, message="tab not resumed"
        )
        gate.set()
        await _no_submit_for(pilot, submitted, 1.5, "imported after a suspend/resume")
        assert toasts == []
        await _enter_imports_after(screen, pilot, submitted, handbook)
