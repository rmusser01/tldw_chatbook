"""Library export canvas: disabled-with-reason button + durable receipt.

Covers task-2858 AC#3 (LIB-11/LIB-12): the "Export bundle (.zip)" button must
never be a silent no-op (disabled state + tooltip wired to the SAME
predicate as the "Nothing to export in this scope." line, and a real click
on a disabled button must genuinely fire nothing), and a successful export
must leave a durable "Last export: <path> · <relative time>" receipt that
survives leaving and re-entering the export canvas within the session.

Widget-render and click-dispatch assertions use a real Textual ``Pilot``
(``App.run_test()``) -- rendered-geometry/DOM-state truth, not a mock of
Textual's own click handling. The screen-level in-place-patch tests use the
``SimpleNamespace`` unbound-method pattern already established in
``test_library_export_cancel.py`` (no full ``LibraryScreen`` Pilot mount
required for pure state-transition logic). The one test that exercises a
REAL export (not a mock/fake service) lives in
``Tests/Library/test_library_export_roundtrip.py`` alongside its sibling
round-trip tests.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from textual.app import App

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.widgets import Button, Static

from tldw_chatbook.Library.library_export_scope import ExportScope
from tldw_chatbook.Library.library_export_state import (
    EXPORT_BUTTON_COUNTING_TOOLTIP,
    EXPORT_BUTTON_NO_DESTINATION_TOOLTIP,
    EXPORT_BUTTON_READY_TOOLTIP,
    build_library_export_form_state,
)
from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_INGEST_EXPORT
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library.library_export_canvas import LibraryExportCanvas


def _state(**overrides):
    base = dict(
        scope=ExportScope(kind="everything"),
        counts={"media": 1, "conversations": 0, "notes": 0},
        name="x",
        description="",
        media_quality="thumbnail",
        destination="/tmp/out.zip",
    )
    base.update(overrides)
    return build_library_export_form_state(**base)


class _Host(ConsolidatedCSSApp):
    """Minimal host: mounts one export canvas, nothing else."""

    def __init__(self, state):
        super().__init__()
        self._state = state
        self.presses: list[str] = []

    def compose(self):
        yield LibraryExportCanvas(self._state, id="library-export-canvas")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.presses.append(event.button.id or "")


# --- Compose-time disabled + tooltip wiring (LIB-11) -------------------------


@pytest.mark.asyncio
async def test_submit_button_disabled_and_tooltipped_for_empty_scope():
    state = _state(counts={"media": 0, "conversations": 0, "notes": 0})
    assert state.export_enabled is False

    app = _Host(state)
    async with app.run_test() as pilot:
        button = pilot.app.query_one("#library-export-submit", Button)
        assert button.disabled is True
        assert button.tooltip == state.empty_scope_line


@pytest.mark.asyncio
async def test_submit_button_disabled_and_tooltipped_for_missing_destination():
    state = _state(destination="")
    assert state.export_enabled is False

    app = _Host(state)
    async with app.run_test() as pilot:
        button = pilot.app.query_one("#library-export-submit", Button)
        assert button.disabled is True
        assert button.tooltip == EXPORT_BUTTON_NO_DESTINATION_TOOLTIP


@pytest.mark.asyncio
async def test_submit_button_enabled_with_ready_tooltip_when_export_is_valid():
    state = _state()
    assert state.export_enabled is True

    app = _Host(state)
    async with app.run_test() as pilot:
        button = pilot.app.query_one("#library-export-submit", Button)
        assert button.disabled is False
        assert button.tooltip == EXPORT_BUTTON_READY_TOOLTIP


# --- Click-swallow prevention (LIB-11) ---------------------------------------


@pytest.mark.asyncio
async def test_disabled_export_button_click_fires_no_pressed_event():
    """task-2858 AC#3 (LIB-11): a click on the Export button while the
    SAME predicate that drives "Nothing to export in this scope." is
    False must genuinely fire nothing -- not merely "look" disabled. This
    exercises Textual's REAL click dispatch (Pilot), not a direct method
    call: Button.press() itself refuses to post Pressed while `disabled`
    is set, and this pins that the canvas's compose()-time wiring is what
    makes that refusal actually happen for the empty-scope case.
    """
    state = _state(counts={"media": 0, "conversations": 0, "notes": 0})
    assert state.export_enabled is False

    app = _Host(state)
    async with app.run_test() as pilot:
        button = pilot.app.query_one("#library-export-submit", Button)
        assert button.disabled is True
        await pilot.click("#library-export-submit")
        await pilot.pause()
        assert app.presses == []


@pytest.mark.asyncio
async def test_enabled_export_button_click_fires_a_pressed_event():
    """Control case: the same click DOES reach a Pressed handler once the
    predicate is satisfied -- proves the previous test's empty result is
    the disabled gate, not a broken click harness."""
    state = _state()
    assert state.export_enabled is True

    app = _Host(state)
    async with app.run_test() as pilot:
        await pilot.click("#library-export-submit")
        await pilot.pause()
        assert app.presses == ["library-export-submit"]


# --- Receipt Static: presence/absence (LIB-12) --------------------------------


@pytest.mark.asyncio
async def test_last_export_line_hidden_before_any_export_this_session():
    state = _state(last_export_line="")

    app = _Host(state)
    async with app.run_test() as pilot:
        widget = pilot.app.query_one("#library-export-last-line", Static)
        assert widget.display is False


@pytest.mark.asyncio
async def test_last_export_line_visible_and_rendered_when_present():
    receipt = "Last export: /tmp/prior.zip · 5m ago"
    state = _state(last_export_line=receipt)

    app = _Host(state)
    async with app.run_test() as pilot:
        widget = pilot.app.query_one("#library-export-last-line", Static)
        assert widget.display is True
        assert receipt in str(widget.render())


# --- In-place patchers keep disabled/tooltip/receipt in sync (LIB-11/LIB-12) --


@pytest.mark.asyncio
async def test_apply_library_export_counts_patches_tooltip_alongside_disabled():
    """task-2858 AC#3 (LIB-11): before this task, ``_apply_library_export_
    counts`` only patched ``disabled`` in place -- the (newly added)
    tooltip would have gone stale the instant counts landed. Starts with
    counts still loading (tooltip: "waiting"), lands a non-empty count
    with NO destination chosen, and asserts the tooltip flips to the
    actual new blocker (no destination) in place, without a recompose.
    """
    scope = ExportScope(kind="everything")
    initial_state = build_library_export_form_state(
        scope=scope,
        counts=None,
        name="x",
        description="",
        media_quality="thumbnail",
        destination="",
    )

    app = _Host(initial_state)
    async with app.run_test() as pilot:
        button = pilot.app.query_one("#library-export-submit", Button)
        assert button.disabled is True
        assert button.tooltip == EXPORT_BUTTON_COUNTING_TOOLTIP

        fake = SimpleNamespace(
            is_mounted=True,
            _library_selected_row_id=LIBRARY_ROW_INGEST_EXPORT,
            _library_export_scope=scope,
            _library_export_counts=None,
            _library_export_counts_request_id=1,
            _library_snapshot_state_generation=0,
            _library_export_form={
                "name": "x",
                "description": "",
                "quality": "thumbnail",
                "destination": "",
                "destination_exists": False,
            },
            _library_export_running=False,
            # task-15790: production gained this flag (library_screen
            # __init__); the double predates it -- the stale-double class.
            _library_export_quality_choices_visible=False,
            _library_export_status="",
            _library_export_error="",
            _library_export_last_path="",
            _library_export_last_at=None,
            query_one=pilot.app.query_one,
        )
        fake._library_entry_route_key = lambda: (LIBRARY_ROW_INGEST_EXPORT,)
        fake._library_entry_reconcile_is_current = lambda *_args: True
        fake._build_library_export_state = (
            lambda: LibraryScreen._build_library_export_state(fake)
        )

        LibraryScreen._apply_library_export_counts(
            fake,
            scope,
            {"media": 1, "conversations": 0, "notes": 0},
            request_id=1,
        )

        assert button.disabled is True  # still blocked: no destination now
        assert button.tooltip == EXPORT_BUTTON_NO_DESTINATION_TOOLTIP


@pytest.mark.asyncio
async def test_update_library_export_canvas_after_run_patches_receipt_and_tooltip():
    """task-2858 AC#3 (LIB-12): the run-completion in-place patcher must
    render the freshly-set receipt AND re-enable the button with its
    ready tooltip, mirroring how ``_apply_library_export_success`` sets
    ``_library_export_last_path``/``_last_at`` just before this runs.
    """
    scope = ExportScope(kind="everything")
    initial_state = build_library_export_form_state(
        scope=scope,
        counts={"media": 1, "conversations": 0, "notes": 0},
        name="x",
        description="",
        media_quality="thumbnail",
        destination="/tmp/out.zip",
        running=True,
        status_line="Exporting… (1 items)",
    )

    app = _Host(initial_state)
    async with app.run_test() as pilot:
        last_line = pilot.app.query_one("#library-export-last-line", Static)
        assert last_line.display is False

        fake = SimpleNamespace(
            is_mounted=True,
            _library_selected_row_id=LIBRARY_ROW_INGEST_EXPORT,
            _library_export_scope=scope,
            _library_export_counts={"media": 1, "conversations": 0, "notes": 0},
            _library_export_form={
                "name": "x",
                "description": "",
                "quality": "thumbnail",
                "destination": "/tmp/out.zip",
                "destination_exists": False,
            },
            _library_export_running=False,
            # task-15790: production gained this flag (library_screen
            # __init__); the double predates it -- the stale-double class.
            _library_export_quality_choices_visible=False,
            _library_export_status="",
            _library_export_error="",
            _library_export_last_path="/tmp/out.zip",
            _library_export_last_at=1000.0,
            query_one=pilot.app.query_one,
        )
        fake._build_library_export_state = (
            lambda: LibraryScreen._build_library_export_state(fake)
        )

        LibraryScreen._update_library_export_canvas_after_run(fake)

        last_line = pilot.app.query_one("#library-export-last-line", Static)
        assert last_line.display is True
        assert "Last export: /tmp/out.zip" in str(last_line.render())
        button = pilot.app.query_one("#library-export-submit", Button)
        assert button.disabled is False
        assert button.tooltip == EXPORT_BUTTON_READY_TOOLTIP


# --- Receipt survives leaving/re-entering the canvas (LIB-12) ----------------


def test_reset_library_export_transient_state_preserves_the_receipt():
    """task-2858 AC#3 (LIB-12): every OTHER export field resets on entry
    (a fresh form each visit is correct), but the receipt must NOT --
    it has to survive the user leaving the Export row (e.g. via another
    rail row) and coming back within the same session."""
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    screen = LibraryScreen(app)
    screen._library_export_last_path = "/tmp/prior.zip"
    screen._library_export_last_at = 12345.0
    screen._library_export_form["name"] = "edited but about to be reset"

    screen._reset_library_export_transient_state()

    assert screen._library_export_last_path == "/tmp/prior.zip"
    assert screen._library_export_last_at == 12345.0
    # Proof the reset genuinely ran (form fields DID reset) -- otherwise
    # the receipt fields surviving would be trivially true.
    assert screen._library_export_form["name"] != "edited but about to be reset"


def test_build_library_export_state_includes_receipt_after_reset():
    """The receipt must actually reach the rendered state -- not just
    survive as a dangling, unused instance attribute."""
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    screen = LibraryScreen(app)
    screen._library_export_last_path = "/tmp/prior.zip"
    screen._library_export_last_at = 12345.0
    screen._reset_library_export_transient_state()

    state = screen._build_library_export_state()

    assert state.last_export_line.startswith("Last export: /tmp/prior.zip")


def test_build_library_export_state_has_no_receipt_before_any_export():
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    screen = LibraryScreen(app)

    state = screen._build_library_export_state()

    assert state.last_export_line == ""


# --- Receipt round-trips through save_state/restore_state --------------------


def test_save_state_and_restore_state_round_trip_the_receipt():
    """task-2858 AC#3 (LIB-12): extends the receipt's durability past a
    full navigate-away-and-back to Library (not just an in-session canvas
    switch) via the screen's already-existing save_state/restore_state
    seam."""
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    screen = LibraryScreen(app)
    screen._library_export_last_path = "/tmp/prior.zip"
    screen._library_export_last_at = 12345.0

    saved = screen.save_state()

    restored = LibraryScreen(app)
    restored.restore_state(saved)

    assert restored._library_export_last_path == "/tmp/prior.zip"
    assert restored._library_export_last_at == 12345.0


def test_restore_state_degrades_gracefully_with_no_prior_receipt():
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    screen = LibraryScreen(app)

    saved = screen.save_state()

    restored = LibraryScreen(app)
    restored.restore_state(saved)

    assert restored._library_export_last_path == ""
    assert restored._library_export_last_at is None
