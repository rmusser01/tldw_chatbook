"""TASK-21242: the four correctness gaps the TASK-21116 conversion left behind.

One test per gap, each verified RED against the pre-fix behaviour (checkout
swap of the three production files) before the fixes landed:

* gap 1 -- a SUPERSEDED/FAILED browse-route swap re-arms the media reader
  builder's one-shot side effects (the ingest "Matched an existing item"
  arrival note and the composed-detail marker) instead of consuming them;
* gap 2 -- a ``kind == "notes"`` canvas sync refuses when the FILES source
  owns the route, instead of painting a database-notes canvas that is not
  the route's owner;
* gap 3 -- a strict canvas-child replacement whose widget has no
  owner-driven sync (loading Static, media viewer) runs the COMPOSED
  follow-up, so ``restore_focus`` is returned after that path;
* gap 4 -- a canvas-scoped sync re-requests an armed list-entry focus
  after the canvas's own recompose, the way ``compose_content`` did for
  every whole-screen recompose.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from textual.widgets import Button, Static

from Tests.UI.app_factory import _build_test_app as _build_tldw_test_app
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _seed_conversations,
    _two_conversations,
    _two_media_items,
    _two_notes,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.Library.library_notes_state import LibraryNotesFocusIdentity  # noqa: F401 -- re-exported seam symbol users may pin
from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_ROW_BROWSE_MEDIA,
    LIBRARY_ROW_BROWSE_NOTES,
    build_library_shell_state,
)
from tldw_chatbook.UI.Library_Modules import library_browse_route_swap
from tldw_chatbook.UI.Library_Modules.canvas_sync import _sync_library_canvas
from tldw_chatbook.UI.Library_Modules.screen_constants import (
    LIBRARY_NOTES_SOURCE_DATABASE,
    LIBRARY_NOTES_SOURCE_FILES,
)
from tldw_chatbook.UI.Screens.library_screen import (
    LibraryEntryReconcileResult,
    LibraryScreen,
)
from tldw_chatbook.Widgets.Library.library_canvas_sync import PostRecomposeCallback


def _build_test_app():
    """Build the legacy/full Library surface these seam tests exercise."""
    app = _build_tldw_test_app()
    app.library_new_profile_admission = False
    return app


ARRIVAL_NOTE = "Matched an existing item — nothing new was imported."


async def _goto_notes_list(screen, pilot) -> None:
    """Navigate the live Library to the database Notes list route."""
    if screen.query("#library-rail-explore-all"):
        screen.query_one("#library-rail-explore-all", Button).press()
    await _wait_for_selector(screen, pilot, "#library-row-browse-notes")
    screen.query_one("#library-row-browse-notes", Button).press()
    await _wait_for_selector(screen, pilot, "#library-notes-canvas")


# ---------------------------------------------------------------------------
# gap 1 -- reader builder side effects survive a failed/superseded swap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
async def test_failed_browse_route_swap_re_arms_reader_arrival_note() -> None:
    """A swap that discards a freshly built Reader must un-consume its state.

    The Notes->Media browse swap builds the permanent Reader (which pops the
    ingest arrival note and advances the composed-detail marker) BEFORE the
    adopt step can fail. On failure the caller takes the whole-screen
    fallback recompose, whose compose rebuilds the Reader -- from state the
    discarded build already consumed, so the "Matched an existing item"
    notice would never reach the user.
    """
    app = _build_test_app()
    _seed_conversations(
        app,
        _two_conversations(),
        notes=_two_notes(),
        media=_two_media_items(),
    )
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        # Visit Media first so its canvas is RESIDENT; the failing adopt then
        # runs the resident-repaint branch that can report failure.
        if screen.query("#library-rail-explore-all"):
            screen.query_one("#library-rail-explore-all", Button).press()
        await _wait_for_selector(screen, pilot, "#library-row-browse-media")
        screen.query_one("#library-row-browse-media", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-canvas")
        # ...then leave it resident-but-hidden from the Notes route.
        await _goto_notes_list(screen, pilot)

        controller = screen._media_controller
        controller._library_media_arrival_note = ARRIVAL_NOTE
        controller._library_media_composed_detail = None
        before = controller._snapshot_library_media_reader_side_effects()
        assert before == (ARRIVAL_NOTE, None)

        # Prove the swap really builds (and consumes) the Reader: wrap the
        # builder and record the note value AFTER it ran.
        consumed_after_build: list[str] = []
        real_builder = controller._build_library_media_reader

        def _recording_builder():
            viewer = real_builder()
            consumed_after_build.append(controller._library_media_arrival_note)
            return viewer

        shell = build_library_shell_state(
            screen._build_library_shell_input(),
            selected_row_id=LIBRARY_ROW_BROWSE_MEDIA,
        )
        with (
            patch.object(controller, "_build_library_media_reader", _recording_builder),
            patch.object(
                library_browse_route_swap, "_sync_library_canvas", return_value=False
            ) as adopt_sync,
        ):
            adopted = await screen._replace_library_browse_canvas(shell)
        assert adopted is False
        assert adopt_sync.called, "the swap must have reached the adopt repaint"
        assert consumed_after_build == [""], (
            "precondition: the Reader build consumed the arrival note"
        )
        after = controller._snapshot_library_media_reader_side_effects()
        assert after == before, (
            "a failed swap must re-arm the arrival note and the composed "
            f"marker; got {after!r}, expected {before!r}"
        )


# ---------------------------------------------------------------------------
# gap 2 -- the notes arm honours the files source
# ---------------------------------------------------------------------------


def _forbidden_query_one(*args, **kwargs):
    raise AssertionError(
        "kind == 'notes' must not resolve #library-notes-canvas while the "
        "files source owns the route: the canvas is not the route's owner"
    )


def test_notes_canvas_sync_refuses_when_files_source_owns_the_route() -> None:
    """A files-source route refuses the database-notes canvas sync outright.

    Under Files the route's canvas owner is the file-notes workspace; the
    database-notes canvas is not mounted. The refusal must come BEFORE any
    DOM resolution (and must not take the whole-screen fallback): there is
    nothing on this route for the sync to recover.
    """
    screen = SimpleNamespace(
        _notes_state=SimpleNamespace(source=LIBRARY_NOTES_SOURCE_FILES),
        _library_selected_row_id=LIBRARY_ROW_BROWSE_NOTES,
        _library_canvas_projection_depth=0,
        query_one=_forbidden_query_one,
    )
    messages: list[str] = []
    with (
        patch(
            "tldw_chatbook.UI.Library_Modules.canvas_sync.logger.debug",
            side_effect=lambda message, *a, **k: messages.append(str(message)),
        ),
    ):
        assert _sync_library_canvas(screen, "notes") is False
    assert any("files source owns the route" in m for m in messages)


def test_notes_canvas_sync_does_not_refuse_the_database_source_early() -> None:
    """The refusal is source-specific: database routes are not refused there.

    The database double is allowed to fail later (its harness is
    incomplete); what is pinned is that the FILES refusal never fires for
    it -- the refusal message names the files source and must not appear.
    """

    def _boom(*args, **kwargs):
        raise AssertionError("unreachable: the arm must proceed past the source check")

    screen = SimpleNamespace(
        _notes_state=SimpleNamespace(source=LIBRARY_NOTES_SOURCE_DATABASE),
        _library_selected_row_id=LIBRARY_ROW_BROWSE_NOTES,
        _library_canvas_projection_depth=0,
        query_one=_boom,
        refresh=lambda *a, **k: None,
    )
    messages: list[str] = []
    with patch(
        "tldw_chatbook.UI.Library_Modules.canvas_sync.logger.debug",
        side_effect=lambda message, *a, **k: messages.append(str(message)),
    ):
        result = _sync_library_canvas(screen, "notes", allow_screen_fallback=False)
    assert result is False  # the incomplete double fails downstream, not refused
    assert not any("files source owns the route" in message for message in messages), (
        "the source refusal must not fire for the database source"
    )


# ---------------------------------------------------------------------------
# gap 3 -- the sync_kind-None branch runs the composed follow-up
# ---------------------------------------------------------------------------


class _FakeCanvasHost:
    """Minimal awaitable host for ``_project_library_canvas_child``."""

    def __init__(self) -> None:
        self.children: tuple = ()
        self.mounted: list = []

    async def remove_children(self, children) -> tuple:
        return ()

    async def mount(self, widget) -> None:
        self.mounted.append(widget)


def _gap3_screen() -> LibraryScreen:
    """A LibraryScreen shell with only the seams the projection touches."""
    screen = object.__new__(LibraryScreen)

    async def _noop_async(*args, **kwargs):
        return None

    screen._library_entry_reconcile_is_current = lambda *a, **k: True  # type: ignore[method-assign]
    screen.children_safe_to_await_removing = lambda parent: ()  # type: ignore[method-assign]
    screen._repair_library_entry_canvas_owner = _noop_async  # type: ignore[method-assign]
    screen._apply_library_notes_stage_visibility = lambda *a, **k: None  # type: ignore[method-assign]
    screen._apply_library_notes_footer_context = lambda *a, **k: None  # type: ignore[method-assign]
    screen._library_canvas_projection_depth = 0
    screen.call_after_refresh = lambda fn, *a, **k: fn(*a)  # type: ignore[method-assign]
    screen._library_pending_list_entry_focus = False  # type: ignore[attr-defined]
    return screen


@pytest.mark.asyncio
async def test_no_owner_sync_projection_returns_captured_focus() -> None:
    """A loading-Static replacement runs restore_focus, not just ``then``.

    ``_project_library_canvas_child`` composes ``restore_focus`` and ``then``
    into one follow-up. When the widget has no owner-driven sync
    (``sync_kind is None`` -- loading Static, media viewer, collections) the
    branch used to run bare ``then``, dropping the restore; with ``then``
    None the computed restore was discarded outright.
    """
    screen = _gap3_screen()
    host = _FakeCanvasHost()
    widget = Static(
        "Loading local Library sources…",
        id="library-canvas-loading",
        markup=False,
    )
    restore_focus = Mock(return_value=True)
    then = Mock()

    result = await screen._project_library_canvas_child(  # type: ignore[arg-type]
        widget,
        canvas_host=host,  # type: ignore[arg-type]
        generation=7,
        route_key=("probe",),
        focus_identity=SimpleNamespace(identity="focused-row"),  # type: ignore[arg-type]
        restore_focus=restore_focus,  # type: ignore[arg-type]
        then=then,  # type: ignore[arg-type]
    )
    assert result is LibraryEntryReconcileResult.APPLIED
    assert host.mounted == [widget]
    restore_focus.assert_called_once()
    then.assert_called_once()


@pytest.mark.asyncio
async def test_no_owner_sync_projection_returns_focus_with_no_then() -> None:
    """The restore alone also runs when the projection carries no ``then``."""
    screen = _gap3_screen()
    host = _FakeCanvasHost()
    widget = Static(
        "Loading local Library sources…",
        id="library-canvas-loading",
        markup=False,
    )
    restore_focus = Mock(return_value=True)

    result = await screen._project_library_canvas_child(  # type: ignore[arg-type]
        widget,
        canvas_host=host,  # type: ignore[arg-type]
        generation=7,
        route_key=("probe",),
        focus_identity=SimpleNamespace(identity="focused-row"),  # type: ignore[arg-type]
        restore_focus=restore_focus,  # type: ignore[arg-type]
        then=None,
    )
    assert result is LibraryEntryReconcileResult.APPLIED
    restore_focus.assert_called_once()


# ---------------------------------------------------------------------------
# gap 4 -- an armed list-entry focus is re-requested after a canvas sync
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
async def test_armed_list_entry_focus_re_requested_after_canvas_scoped_sync() -> None:
    """A canvas-scoped notes sync re-fires the armed entry-focus channel.

    ``compose_content`` re-requests a pending entry focus on every
    whole-screen recompose; the canvas-scoped seam bypasses it, so rows
    rebuilt by a canvas-level recompose left the armed channel unfired.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _goto_notes_list(screen, pilot)

        generation = screen._library_list_entry_focus_generation
        screen._library_pending_list_entry_focus = True
        screen._library_list_entry_focus_deadline = None  # no timer interplay
        calls: list[int] = []

        def _spy(focus_generation: int) -> None:
            calls.append(focus_generation)

        # Flip a kwargs input so the canvas sync genuinely recomposes.
        screen._notes_state.sort_choices_visible = (
            not screen._notes_state.sort_choices_visible
        )
        with patch.object(
            screen, "_focus_library_list_entry_if_current", side_effect=_spy
        ):
            assert _sync_library_canvas(screen, "notes") is True
            for _ in range(20):
                await pilot.pause(0.02)
        assert calls, (
            "the armed channel must be re-requested after the canvas's own "
            "recompose, the way compose_content did on every whole-screen one"
        )
        assert max(calls) >= generation


def test_re_request_is_a_default_not_an_eviction() -> None:
    """The re-request queues through the DEFAULT slot, never evicting intent.

    Pins the queueing discipline at the unit level: a canvas carrying a
    pending action intent must keep it, with the re-request composed ahead
    of it (defaults run first, the action's focus choice wins).
    """

    class _RecordingCanvas(PostRecomposeCallback):
        def __init__(self) -> None:
            self.fired: list = []

        def recompose(self):
            callback = self._post_recompose_callback
            self._post_recompose_callback = None
            if callback is not None:
                callback()
            return self

    canvas = _RecordingCanvas()
    intent = Mock()
    canvas.queue_after_recompose(intent)
    pending_before = canvas._post_recompose_callback
    re_request = Mock()
    canvas.queue_default_after_recompose(re_request)
    assert canvas._post_recompose_callback is not pending_before
    canvas.recompose()
    re_request.assert_called_once()
    assert intent.called, "the action intent queued first must still run"
