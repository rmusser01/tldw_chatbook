"""Phase-C task 2.5 structural pins: the sync storm's four measured wastes.

``test_library_phase_c_switch_residency.py`` pins the SHAPE of a rail-mode
switch (no whole-screen recompose, mount/unmount ceilings).
``test_library_phase_c_resident_canvas.py`` pins the hazards residency creates.
This file pins the four specific pieces of per-switch work that phase-C task
2.5 measured as provably wasted, each with the number that made it a target.

## Where these pins come from

Task 2 removed the structural cause of the freeze (0 whole-screen recomposes,
mounts 177 -> 81/26) and the longest main-thread block did not move (98 ms).
Task 2.5 opened, as its plan requires, with an attribution measurement rather
than a fix: ``Helper_Scripts/library_restyle_attribution_probe.py`` wraps every
Textual entry point that reaches ``Stylesheet.apply`` and attributes each apply
to its trigger and to the application line that caused it. Measured on this
worktree, per switch:

    media (switch-back)    423 applies / 86 ms restyle
                           238 (56%) from apply_route's TWO marker-class flips
                            89 (21%) mount-proportional
    notes (switch, 1st)    463 applies / 92 ms restyle
                           205 (44%) from sync_layout's pane ``disabled`` flips
                           192 (41%) from apply_route's two marker-class flips
                            30 (6%)  mount-proportional
    notes (switch, later)  333 applies / 65 ms restyle
                           206 (62%) from apply_route's two marker-class flips

A class flip on the shell restyles the shell's ENTIRE subtree (Textual's
``DOMNode.update_node_styles`` -> ``App.update_styles`` ->
``stylesheet.update_nodes(node.walk_children(with_self=True))``), which is
96-119 applies per flip here. That is why the arm with the FEWEST mounts ran
the MOST applies -- the finding the design record had to retract a claim over.

The two mount-side wastes come from the same measurement run
(``sync_state`` calls instrumented with the canvas's ``display`` and the
calling stack):

    media (switch-back)    21 of 81 mounts rebuild the OUTGOING Notes canvas,
                           from ``_supersede_library_notes_navigation`` two
                           statements BEFORE the destination row is set
    notes (switch, later)  25 of 62 mounts rebuild the destination canvas
                           while ``display`` is still False, from the notes
                           tree initial load; the route swap repaints it again
                           when it shows it

Each test below names the exact number it was written against. They are
structural (counts of restyles and rebuilds), not wall-clock, so they are
deterministic on a loaded machine.
"""
from __future__ import annotations

import re

import pytest
from textual.app import App
from textual.widgets import Button

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _seed_conversations,
    _two_conversations,
    _two_media_items,
    _wait_for_library_shell,
)
from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_ROW_BROWSE_MEDIA,
    LIBRARY_ROW_BROWSE_NOTES,
)
from tldw_chatbook.Widgets.Library.library_browse_reader_shell import (
    LIBRARY_ROUTE_MARKER_CLASSES,
)


async def _settle(pilot, passes: int = 40, delay: float = 0.01) -> None:
    for _ in range(passes):
        await pilot.pause(delay)


async def _open_library(host, pilot):
    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    await _settle(pilot, passes=30)
    if screen.query("#library-rail-explore-all"):
        screen.query_one("#library-rail-explore-all", Button).press()
        await _settle(pilot, passes=30)
    return screen


async def _press_rail_row(screen, pilot, row_id: str) -> None:
    screen.query_one(f"#library-row-{row_id}", Button).press()
    await _settle(pilot)


def _seeded_host():
    app = _build_test_app()
    _seed_conversations(
        app, _two_conversations(), notes=None, media=_two_media_items()
    )
    return LibraryHarness(app)


class _SubtreeRestyleCounter:
    """Counts ``App.update_styles`` fires per node id, while armed."""

    def __init__(self) -> None:
        self.armed = False
        self.fires: list[str] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        original = App.update_styles

        def update_styles(app_self, node, animate: bool = True):
            if self.armed:
                self.fires.append(getattr(node, "id", None) or type(node).__name__)
            return original(app_self, node, animate=animate)

        monkeypatch.setattr(App, "update_styles", update_styles)

    def count(self, node_id: str) -> int:
        return sum(1 for fire in self.fires if fire == node_id)


@pytest.mark.asyncio
async def test_route_marker_classes_have_no_stylesheet_rules() -> None:
    """The marker classes are query markers, so nothing may style them.

    This is the guard that licenses ``apply_route`` flipping them with
    ``update=False`` (Textual's "do not restyle" flag). If someone adds a rule
    that mentions one of these class names -- on the shell itself or as an
    ancestor in a descendant selector -- the flip stops being free and this
    test fails, naming the seam that has to be restored. Scanning the parsed
    stylesheet rather than the ``.tcss`` sources on purpose: widget
    ``DEFAULT_CSS`` is part of the same stylesheet and a grep of the css/
    directory would miss it.
    """
    host = _seeded_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        await _open_library(host, pilot)
        patterns = {
            marker: re.compile(rf"\.{re.escape(marker)}(?![\w-])")
            for marker in LIBRARY_ROUTE_MARKER_CLASSES
        }
        offenders = [
            f"{marker}: {rule.selectors}"
            for rule in host.stylesheet.rules
            for marker, pattern in patterns.items()
            if pattern.search(rule.selectors)
        ]
    assert not offenders, (
        "A stylesheet rule now depends on a Library route MARKER class, so "
        "flipping it without a restyle is no longer correct. Either drop the "
        "rule or restore the restyle in "
        "LibraryBrowseReaderShell.apply_route:\n  " + "\n  ".join(offenders)
    )


@pytest.mark.asyncio
async def test_route_switch_does_not_restyle_the_whole_shell_subtree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rail-mode switch must not restyle the browse shell's subtree.

    Measured before this pin: ``apply_route`` fired ``App.update_styles`` on
    ``#library-browse-reader-shell`` TWICE per switch (one ``set_class`` per
    marker), and each fire applied the stylesheet to all 96-119 nodes of the
    shell subtree -- 238 of the 423 apply calls on a media switch-back, 43 ms
    of its 86 ms of restyle, for two classes no stylesheet rule references.
    """
    host = _seeded_host()
    counter = _SubtreeRestyleCounter()
    counter.install(monkeypatch)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _open_library(host, pilot)
        await _press_rail_row(screen, pilot, LIBRARY_ROW_BROWSE_MEDIA)
        await _press_rail_row(screen, pilot, LIBRARY_ROW_BROWSE_NOTES)

        measured = {}
        for label, row_id in (
            ("media (switch-back)", LIBRARY_ROW_BROWSE_MEDIA),
            ("notes (switch)", LIBRARY_ROW_BROWSE_NOTES),
        ):
            counter.fires = []
            counter.armed = True
            await _press_rail_row(screen, pilot, row_id)
            counter.armed = False
            measured[label] = counter.count("library-browse-reader-shell")

        # Liveness: the switch has to have switched, or zero restyles is what
        # a dead UI scores (the lesson the acceptance pin's own
        # ``_assert_switch_did_something`` records).
        assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_NOTES
        assert screen.query_one("#library-notes-canvas").display

    assert measured == {"media (switch-back)": 0, "notes (switch)": 0}, (
        "a rail-mode switch restyles the whole browse-shell subtree: "
        f"{measured}"
    )
