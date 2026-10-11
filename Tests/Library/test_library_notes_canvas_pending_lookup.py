"""TASK-34000.25 fix round 2: ``_pending_widget_lookup`` reads Textual's
compose-time child registry, so a Textual change to it fails HERE by name.

``LibraryNotesCanvas._compose_editor`` gates every editor surface on its
last line, over the composed-but-unmounted tree, through this lookup: a
child yielded under ``with Container():`` must be reachable from the
container before anything mounts (``compose_add_child`` ->
``_pending_children`` in Textual 8.2.8). Without this pin a Textual bump
would surface as an app crash in the first UI test that opens a note.

PR #3055 review (Important 1): the pin above is the LOUD half. The second
test is the soft half -- when the lookup cannot resolve the gated nodes
(an empty registry is exactly what a renamed attribute yields), compose
must not raise and must not skip the gates: it logs once at WARNING and
gates the mounted tree one refresh later.

TASK-34789: the third test is the same rule for the canvas's OWN recompose.
While it removes and remounts its children the header row can be in the DOM
without its children, and a shell resize handled on the screen's pump in
that window (``apply_work_width``) must not raise -- the shape it asked for
lands on the new children when the recompose completes.

Fast lane: a bare Textual app (``run_test``), no product app boot; the
fallback and recompose tests mount one ``LibraryNotesCanvas`` under the app
stylesheets (``ConsolidatedCSSApp``) because their claims are paint-level.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
from loguru import logger
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widgets import Button, Static

from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from Tests.Widgets.Library.test_library_notes_canvas import _editor_state
from tldw_chatbook.Widgets.Library import library_notes_canvas as canvas_module
from tldw_chatbook.Widgets.Library.library_notes_canvas import (
    LibraryNotesCanvas,
    _pending_widget_lookup,
)


class _Probe(Static):
    """Composes a nested tree and snapshots the lookup before mounting."""

    found: dict[str, object] = {}
    missing: type[Exception] | None = None

    def compose(self) -> ComposeResult:
        roots = []
        with Vertical(id="outer") as outer:
            roots.append(outer)
            with Horizontal(id="row"):
                yield Button("One", id="one")
                yield Button("Two", id="two")
            yield Static("leaf", id="leaf")
        lookup = _pending_widget_lookup(roots)
        _Probe.found = {
            selector: lookup(selector) for selector in ("#outer", "#row", "#one", "#two", "#leaf")
        }
        try:
            lookup("#absent")
        except NoMatches as exc:
            _Probe.missing = type(exc)
        # No re-yield: a closed ``with`` container is composed by its own
        # ``__exit__`` -- exactly how ``_compose_editor``'s ``root()`` only
        # records the node (yielding it again mounts it twice).


class _App(App):
    def compose(self) -> ComposeResult:
        yield _Probe()


@pytest.mark.asyncio
async def test_pending_lookup_finds_every_child_yielded_under_a_with_container():
    _Probe.found = {}
    _Probe.missing = None
    app = _App()
    async with app.run_test(size=(40, 10)):
        assert set(_Probe.found) == {"#outer", "#row", "#one", "#two", "#leaf"}
        # The same objects that then mounted -- not copies, not stale.
        for selector, widget in _Probe.found.items():
            assert app.query_one(selector) is widget, selector
        assert _Probe.missing is NoMatches


class _EditorCanvasApp(ConsolidatedCSSApp):
    """One editor-mode canvas under the app stylesheets (paint-level claim)."""

    CSS_PATH: ClassVar[list[str]] = [str(path) for path in APP_STYLESHEETS]

    def compose(self) -> ComposeResult:
        shell = Vertical(id="library-shell-grid")
        shell.styles.width = 100
        shell.styles.height = 30
        with shell:
            yield LibraryNotesCanvas(mode="editor", presentation_state=_editor_state())


@pytest.mark.asyncio
async def test_an_empty_pending_lookup_never_aborts_compose_and_gates_after_one_refresh(
    monkeypatch,
):
    """PR #3055 review (Important 1): a lookup that resolves nothing -- the
    shape a renamed Textual registry attribute produces -- must not raise
    inside ``compose`` (a compose exception is the nav-freeze shape) and
    must not silently skip the gates. It logs ONCE at WARNING and the same
    gates land on the mounted tree after one refresh: the Info and Preview
    surfaces of an Edit-region note are not painted.

    RED on bbc7c2c6d2: ``compose`` raised ``NoMatches`` for the first gated
    selector and the app never came up.
    """
    original = canvas_module._pending_widget_lookup
    monkeypatch.setattr(
        canvas_module, "_pending_widget_lookup", lambda roots: original(())
    )
    monkeypatch.setattr(LibraryNotesCanvas, "_pending_lookup_warned", False, raising=False)
    warnings: list[str] = []
    sink = logger.add(
        lambda message: warnings.append(message.record["message"]),
        level="WARNING",
        filter=lambda record: "surface gating" in record["message"],
    )
    try:
        app = _EditorCanvasApp()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await pilot.pause()
            canvas = app.query_one(LibraryNotesCanvas)
            painted = app.screen._compositor.visible_widgets
            editor = canvas.query_one("#library-note-editor-region")
            context = canvas.query_one("#library-note-context-region")
            preview = canvas.query_one("#library-note-preview-region")
            assert editor.display is True
            assert editor in painted, "the Edit surface is not painted"
            assert context.display is False
            assert context not in painted, "the Info surface is painted"
            assert preview.display is False
            assert preview not in painted, "the Preview surface is painted"
            assert len(warnings) == 1, warnings
            # A second compose under the same failure stays quiet: once.
            await canvas.remove()
            await app.query_one("#library-shell-grid").mount(
                LibraryNotesCanvas(mode="editor", presentation_state=_editor_state())
            )
            await pilot.pause()
            await pilot.pause()
            again = app.query_one(LibraryNotesCanvas)
            assert again.query_one("#library-note-context-region").display is False
            assert len(warnings) == 1, warnings
    finally:
        logger.remove(sink)


#: Either side of ``_HEADER_ONE_ROW_MIN_WIDTH`` (100): the editor header is
#: one row at the first width and stacked at the second.
ONE_ROW_WORK_WIDTH = 120
STACKED_WORK_WIDTH = 60


@pytest.mark.asyncio
async def test_a_resize_landing_mid_recompose_is_applied_when_the_recompose_completes():
    """TASK-34789: ``apply_work_width`` on a half-remounted canvas never raises.

    The half-mounted state is driven directly, not by timing: the resize is
    delivered from inside the recompose's own ``mount_all``, after the new
    top-level children are registered and before their children exist --
    the state a notes-tree or trash reload leaves the work pane in for as
    long as its recompose runs, while the screen's pump is free to handle
    ``AdaptiveReaderShellResized``.

    RED on dev 1115b492f8: ``NoMatches: No nodes match
    '#library-note-status' on LibraryNotesCanvas(...)`` from
    ``apply_compact_presentation`` -- the UI Fast Lane crash of
    ``test_new_note_with_saved_content_survives_a_rail_round_trip``.
    """
    app = _EditorCanvasApp()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        canvas = app.query_one(LibraryNotesCanvas)
        canvas.apply_work_width(ONE_ROW_WORK_WIDTH)
        one_row = canvas.query_one("#library-note-header-second-row")
        assert one_row.styles.layout.name == "horizontal"

        real_mount_all = canvas.mount_all
        landed: list[str] = []

        def mount_all_then_resize(widgets, **kwargs):
            pending = real_mount_all(widgets, **kwargs)
            if not landed:
                # Registered, not yet awaited: the row is there, its
                # children are still pending.
                assert canvas.query("#library-note-header-second-row")
                assert not canvas.query("#library-note-status")
                landed.append("resized")
                canvas.apply_work_width(STACKED_WORK_WIDTH)
            return pending

        canvas.mount_all = mount_all_then_resize
        canvas.refresh(recompose=True)
        for _ in range(100):
            if landed and canvas.query("#library-note-status"):
                break
            await pilot.pause(0.02)
        await pilot.pause()

        assert landed == ["resized"]
        stacked = canvas.query_one("#library-note-header-second-row")
        assert stacked is not one_row, "the canvas never recomposed"
        # The shape the mid-recompose resize asked for, on the NEW children.
        assert stacked.styles.layout.name == "vertical"
        assert canvas.query_one("#library-note-status").has_class("h-1")
