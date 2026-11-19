"""TASK-33620.15.20: tray patches between structural deltas match a rebuild.

The Conversations tray's in-place patch (TASK-36012) decides from a record
that compose writes, indexed by each row's place in the browser state
(``_BrowserRowPaint``). Only a rebuild may move a row: rows added, removed,
renamed or reordered while a run is in flight must rebuild, and every patch
after one must land what a fresh build of the same state shows, in both glyph
modes. Kept apart from ``test_console_tray_patch_styles.py`` so that each file
stays well inside the UI PR lane's per-file budget. On origin/dev 3efe8476ce
it fails at the colour steps only, for the reason that file gives: a plain
``button.label =`` write drops a style-only change.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

# The UI fixture imports the app after selecting a per-test profile. Import it
# under the collection bootstrap first, before config freezes its source owner.
import tldw_chatbook.app  # noqa: F401
from tldw_chatbook.Widgets import glyph_fallback
from Tests.UI.test_console_run_tick_layout_requests import _LayoutRequests, _names
from Tests.UI.test_console_send_transition_layout_requests import (
    _row,
    _settled_tray,
    _tray_state,
    _TrayHost,
)
from Tests.UI.test_console_tray_patch_styles import _remount, _styled_witness

pytestmark = [pytest.mark.bootstrap_profile]


def _r(index: int, *, running: bool = False, **changes):
    """Row ``index`` from the shared fixtures, with ``changes`` applied."""
    return replace(_row(index, running=running), **changes)


def _rows_state(*rows):
    """The idle tray state with its one section holding ``rows``."""
    state = _tray_state()
    browser = state.conversation_browser
    section = replace(browser.sections[0], rows=rows, count=len(rows))
    return replace(state, conversation_browser=replace(browser, sections=(section,)))


_RED, _BLUE = "#ff0000", "#0000ff"

#: (step, rows, whether the patch takes it). Each step starts from the one
#: before it; the first starts from rows 0-2 idle.
_SEQUENCE = (
    ("row 0 starts", (_r(0, running=True), _r(1), _r(2)), True),
    ("reorder while running", (_r(1), _r(0, running=True), _r(2)), False),
    ("row 1 starts too", (_r(1, running=True), _r(0, running=True), _r(2)), True),
    (
        "a row is added",
        (_r(1, running=True), _r(0, running=True), _r(2), _r(3)),
        False,
    ),
    (
        "an idle row's colour",
        (_r(1, running=True), _r(0, running=True), _r(2, color=_RED), _r(3)),
        True,
    ),
    (
        # The attention glyph owns the slot, so the colour shows nothing yet.
        "a running row's colour",
        (
            _r(1, running=True),
            _r(0, running=True, color=_BLUE),
            _r(2, color=_RED),
            _r(3),
        ),
        True,
    ),
    (
        "row 0 ends in its colour",
        (_r(1, running=True), _r(0, color=_BLUE), _r(2, color=_RED), _r(3)),
        True,
    ),
    ("a row is removed", (_r(1, running=True), _r(0, color=_BLUE), _r(3)), False),
    (
        "a running row is renamed",
        (_r(1, running=True, title="Renamed"), _r(0, color=_BLUE), _r(3)),
        False,
    ),
    (
        "another row takes index 2",
        (_r(1, running=True, title="Renamed"), _r(0, color=_BLUE), _r(4)),
        False,
    ),
    (
        "an icon while running and idle",
        (
            _r(1, running=True, title="Renamed", icon="🔬"),
            _r(0, color=_BLUE, icon="📚"),
            _r(4),
        ),
        True,
    ),
    (
        "row 4 cannot be opened",
        (
            _r(1, running=True, title="Renamed", icon="🔬"),
            _r(0, color=_BLUE, icon="📚"),
            _r(4, openable=False),
        ),
        False,
    ),
    (
        "the unopenable row starts",
        (
            _r(1, running=True, title="Renamed", icon="🔬"),
            _r(0, color=_BLUE, icon="📚"),
            _r(4, running=True, openable=False),
        ),
        True,
    ),
    *(
        (
            f"rows 1 and 0 trade the run ({turn})",
            (
                _r(1, running=turn % 2 == 1, title="Renamed", icon="🔬"),
                _r(0, running=turn % 2 == 0, color=_BLUE, icon="📚"),
                _r(4, running=True, openable=False),
            ),
            True,
        )
        for turn in range(4)
    ),
    (
        "every run ends",
        (
            _r(1, title="Renamed", icon="🔬"),
            _r(0, color=_BLUE, icon="📚"),
            _r(4, openable=False),
        ),
        True,
    ),
)


async def test_patches_between_structural_deltas_match_a_rebuild(monkeypatch) -> None:
    """Rows added, removed, renamed or reordered mid-run rebuild; later patches hold.

    The patch indexes its record by each row's place in the browser state.
    Only a rebuild may move a row, so every patch after one must still land
    what a fresh build of the same state shows. In each glyph mode one tray
    runs the whole sequence; each step's result is then compared with a fresh
    build. One app serves both modes.
    """
    requests = _LayoutRequests(monkeypatch)
    failures: list[str] = []
    host = _TrayHost(_rows_state(_r(0), _r(1), _r(2)))
    async with host.run_test(size=(40, 30)) as pilot:
        await _settled_tray(host, pilot)
        for mode, ascii_mode in (("unicode", False), ("ascii", True)):
            monkeypatch.setattr(glyph_fallback, "_ASCII_MODE", ascii_mode)
            tray = await _remount(host, pilot, _rows_state(_r(0), _r(1), _r(2)))
            seen: list[list] = []
            for step, rows, patched in _SEQUENCE:
                incoming = _rows_state(*rows)
                with requests.counting() as calls:
                    tray.sync_state(incoming)
                rebuilt = tray in calls
                for _ in range(4):
                    await pilot.pause()
                if rebuilt is patched:
                    failures.append(
                        f"{mode}, {step}: expected a "
                        f"{'patch' if patched else 'rebuild'}, "
                        f"layout requests {_names(calls)}"
                    )
                if tray.state != incoming:
                    failures.append(f"{mode}, {step}: the tray did not adopt the state")
                seen.append(_styled_witness(tray))
            # A state the sequence revisits is built once.
            fresh: list[tuple[object, list]] = []
            for (step, rows, _patched), result in zip(_SEQUENCE, seen):
                state = _rows_state(*rows)
                built = next((w for s, w in fresh if s == state), None)
                if built is None:
                    built = _styled_witness(await _remount(host, pilot, state))
                    fresh.append((state, built))
                if result != built:
                    failures.append(
                        f"{mode}, {step}: the result differs from a fresh build"
                    )
    assert failures == []
