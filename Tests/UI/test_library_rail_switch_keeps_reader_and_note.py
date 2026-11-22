"""TASK-34000.25: Library rail switches keep the open media reader and a new
note, and the Media reader gains a Note action (review finding S-02).

The reader and the note editor share one work pane, so a researcher
switches rail rows constantly -- and every switch threw state away in both
directions. On the base (``d2c074cb87``):

- ``_select_library_rail_row_after_source_admission`` resets Media to the
  list on every rail press ("A rail-row press is always a fresh entry"), so
  Media ▸ open item ▸ rail Notes ▸ rail Media shows "Select a media item to
  read it here." beside a row still marked ``loaded`` (L-16);
- the Notes retention gate requires ``session_blank_id is None``, and a
  note born from New note keeps that id until an EXPLICIT Save (an autosave
  deliberately does not clear it), so a New-note note with autosaved content
  is flushed and closed by the next rail press;
- ``n`` in the Media reader does nothing: the bare ``n`` branch in ``on_key``
  is gated on ``library_notes_new``, false on the Media row.

This is the lean PR-gated core (``scripts/ui_pr_gate_census.txt``): one
real ``LibraryHarness`` boot per test (``CSS_PATH`` = the app bundle), the
Notes side backed by a REAL ChaChaNotes database behind the real Notes scope
service (``Tests/UI/library_quit_guard_support._NotesProfile``), so every
data-integrity claim is read from the DB, not from widget text. The slower
arms (the Info-tab round trip, the real-external-edit conflict twin and its
positive twin -- the dirty-return arm with two autosaves, moved out on the
PR #3055 review for lane headroom -- ``n`` inside the Find input, the
server-item refusal, the vetoed title, the untouched-blank GC and ‹ Back at
100x30) live in ``test_library_rail_switch_keeps_reader_and_note_extended.py``,
outside the lane.
"""

from __future__ import annotations

import asyncio

import pytest
from textual.containers import VerticalScroll
from textual.widgets import Button, Input, TextArea

from Tests.UI.library_quit_guard_support import (
    _NotesProfile,
    _armed_editor,
    _new_blank_note,
    _scaled_autosave,
    _type,
    _type_at_end,
    _until,
)
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    LibraryProductionCSSHarness,
    _active_library_screen,
    _build_test_app,
    _seed_conversations,
    _two_conversations,
    _two_media_items,
    _two_notes,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
from tldw_chatbook.Library.collections_capture_models import CaptureSaveRequest
from tldw_chatbook.Prompt_Management.prompt_scope_service import (
    LocalPromptService,
    PromptScopeService,
)
from tldw_chatbook.UI.Library_Modules.library_media_browse_controller import (
    LibraryMediaBrowseController,
)
from tldw_chatbook.UI.Library_Modules.screen_constants import (
    LIBRARY_LIST_ENTRY_FOCUS_EXPIRY_RETRIES,
)
from tldw_chatbook.UI.Screens import library_screen as library_screen_module
from tldw_chatbook.Widgets.Library import LIBRARY_ADAPTIVE_READER_GRIP_CLASS
from tldw_chatbook.Widgets.Library.library_media_canvas import LibraryMediaRowScroll
from tldw_chatbook.Widgets.Library.library_media_content import (
    LibraryMediaContentBody,
)

pytestmark = pytest.mark.bootstrap_profile

#: The review's wide repro size.
SIZE = (160, 45)

MEDIA_UUID = "4b0e1c2d-7a53-4f0e-9b61-2c8d5e7f9a10"
MEDIA_TITLE = "A field guide to Markdown tables"
MEDIA_ROW = "#library-media-row-1"
LOADED_ID = "local:media:1"
VIEWER_TITLE = "#library-media-viewer-title"
NOTE_BODY = "#library-note-body"
NOTE_TITLE = "#library-note-title"

#: Well past the viewport at 160x45, so a restored offset is a real restore.
SCROLL_Y = 10


def _media_items() -> list[dict]:
    """Two local items; the first is long enough to scroll and carries a uuid."""
    body = "\n".join(
        f"Paragraph {n} of the field guide, one plain line of prose." for n in range(1, 160)
    )
    return [
        {
            "id": "media-1",
            "uuid": MEDIA_UUID,
            "title": MEDIA_TITLE,
            "type": "article",
            "last_modified": "2026-07-06T08:00:00Z",
            "author": "Jordan Lee",
            "keywords": ["markdown"],
            "content": body,
            "version": 1,
        },
        {
            "id": "media-2",
            "uuid": "0f1e2d3c-4b5a-4968-8776-5544332211aa",
            "title": "Product Demo Video",
            "type": "video",
            "last_modified": "2026-07-06T10:00:00Z",
            "author": "Morgan Lee",
            "keywords": ["demo"],
            "content": "Full transcript: the product demo video walks through the new dashboard.",
            "version": 2,
        },
    ]


def _host(tmp_path) -> tuple[LibraryHarness, _NotesProfile]:
    """The Library screen over a real notes DB and the static media service."""
    app = _build_test_app()
    _seed_conversations(app, [], media=_media_items())
    profile = _NotesProfile(tmp_path)
    app.chachanotes_db = profile.db
    app.notes_scope_service = profile.scope_service
    app.notes_service = profile.interop
    return LibraryHarness(app), profile


async def _library(host, pilot):
    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    return screen


async def _open_media_item(screen, pilot) -> LibraryMediaContentBody:
    """Browse Media, open the first item, wait for the Reader to settle."""
    screen.query_one("#library-row-browse-media", Button).press()
    row = await _wait_for_selector(screen, pilot, MEDIA_ROW)
    row.press()
    await _wait_for_selector(screen, pilot, VIEWER_TITLE)

    def _settled() -> bool:
        session = screen._media_state.reader_session
        return (
            session.pending_request is None
            and session.loaded_id == LOADED_ID
            and screen._media_state.detail is not None
        )

    await _until(pilot, _settled, "the Reader to settle on the first item")
    await pilot.pause()
    return screen.query_one("#library-media-viewer-content", LibraryMediaContentBody)


async def _select_reader_mode(screen, pilot, mode: str) -> None:
    """Press a Reader tab and wait for the REBUILT children, not the session.

    Fix round 2 (review 2 §4): ``handle_library_media_reader_mode`` sets
    ``reader_session.mode`` synchronously and then recomposes the viewer
    (remove, then mount_all under ``batch()``); a wait on the session alone
    is satisfied mid-flight, so the next DOM read missed the tab button
    (``NoMatches``) or scrolled the OUTGOING body ("max 287" that never
    reached y=10). The "(selected)" label exists only on the rebuilt
    children, so it is the signal that the new tree is mounted.
    """
    screen.query_one(f"#library-media-reader-select-{mode}", Button).press()
    await _until(
        pilot,
        lambda: screen._media_state.reader_session.mode == mode,
        f"the {mode} tab in the session",
    )
    selector = f"#library-media-reader-select-{mode}"
    await _wait_for_selector(screen, pilot, selector)
    await _until(
        pilot,
        lambda: "(selected)" in str(screen.query_one(selector, Button).label),
        f"the rebuilt {mode} tab to be the selected one",
    )
    await pilot.pause()


async def _scroll_reader(screen, pilot, body: LibraryMediaContentBody, y: int) -> None:
    """Scroll the Read body to ``y`` once it has laid out enough to get there.

    The Raw view builds its wrap index on its own layout pass, a pump or two
    after the session settles, so an immediate ``scroll_to`` clamps to 0
    (``test_library_media_reader_scroller_resolution.py``).
    """
    await _until(
        pilot,
        lambda: body.scroller.max_scroll_y >= y,
        f"the Read body to lay out past y={y} (max {body.scroller.max_scroll_y})",
        timeout=10.0,
    )
    body.scroller.scroll_to(y=y, animate=False, immediate=True)
    await _until(
        pilot,
        lambda: int(body.scroller.scroll_y) == y,
        f"the Reader to scroll to y={y} (max {body.scroller.max_scroll_y})",
        timeout=5.0,
    )


async def _rail(screen, pilot, row_id: str) -> None:
    screen.query_one(f"#library-row-browse-{row_id}", Button).press()
    await pilot.pause()
    await pilot.pause()


def _reader_scroll_y(screen) -> int | None:
    try:
        body = screen.query_one("#library-media-viewer-content", LibraryMediaContentBody)
    except Exception:  # noqa: BLE001 -- absent while the Reader shows its empty state
        return None
    return int(body.scroller.scroll_y)


def _loaded_rows(screen) -> list[str]:
    return [
        str(row.id)
        for row in screen.query(".library-media-row").results(Button)
        if " · loaded" in str(row.label)
    ]


def _saved(screen) -> bool:
    snapshot = screen._library_note_session.snapshot
    return (
        snapshot is not None
        and not snapshot.dirty
        and not snapshot.saving
        and screen._notes_state.autosave_state == "saved"
    )


def _session(screen):
    snapshot = screen._library_note_session.snapshot
    assert snapshot is not None, "no note session is open"
    return snapshot


# --- AC#2 / AC#5: the open item survives Media -> Notes -> Media -------------


@pytest.mark.asyncio
async def test_media_item_survives_a_rail_round_trip(tmp_path):
    """The same item, the Read tab and the scroll offset come back; the row's
    ``loaded`` marker names exactly that item (L-16)."""
    host, profile = _host(tmp_path)
    async with host.run_test(size=SIZE) as pilot:
        screen = await _library(host, pilot)
        await _open_media_item(screen, pilot)
        await _select_reader_mode(screen, pilot, "info")
        await _select_reader_mode(screen, pilot, "read")
        # Re-queried AFTER the rebuilt tree is in: the body from before the
        # mode presses is the outgoing one.
        body = screen.query_one("#library-media-viewer-content", LibraryMediaContentBody)
        await _scroll_reader(screen, pilot, body, SCROLL_Y)

        await _rail(screen, pilot, "notes")
        await _wait_for_selector(screen, pilot, ".library-notes-tree-note-row")
        await _rail(screen, pilot, "media")

        assert screen.query(VIEWER_TITLE), (
            "the Reader shows its empty state after the round trip: "
            f"view={screen._media_state.view!r}"
        )
        session = screen._media_state.reader_session
        assert session.loaded_id == LOADED_ID
        assert session.mode == "read"
        assert screen._media_state.view == "viewer"
        await _until(
            pilot,
            lambda: _reader_scroll_y(screen) == SCROLL_Y,
            f"the reading position to be restored (now {_reader_scroll_y(screen)})",
            timeout=10.0,
        )
        assert _loaded_rows(screen) == [MEDIA_ROW.lstrip("#")]
    profile.db.close_connection()


# --- AC#1 / AC#5: a New-note note with autosaved text survives ---------------


@pytest.mark.asyncio
async def test_new_note_with_saved_content_survives_a_rail_round_trip(
    tmp_path, monkeypatch
):
    """Capture 28: the note created from New note stays open with its text."""
    _scaled_autosave(monkeypatch)
    mounted = _record_surface_flags_at_mount(monkeypatch)
    host, profile = _host(tmp_path)
    async with host.run_test(size=SIZE) as pilot:
        screen = await _library(host, pilot)
        await _new_blank_note(screen, pilot)
        note_id = _session(screen).note_id
        await _type_at_end(pilot, screen.query_one(NOTE_BODY, TextArea), "s02 paper notes")
        await _until(pilot, lambda: _saved(screen), "the autosave to land")
        assert profile.note(note_id)["content"] == "s02 paper notes"

        await _rail(screen, pilot, "media")
        await _wait_for_selector(screen, pilot, "#library-media-canvas")
        await _rail(screen, pilot, "notes")

        await _until(
            pilot,
            lambda: bool(screen.query(NOTE_BODY)),
            "the note editor to be open again after the round trip",
            timeout=5.0,
        )
        # Fix round 1: the pane exists and nothing has synced it yet -- its
        # ``display`` flags are exactly what compose gave them. On the base
        # they were composed True and flipped only by a later sync (a race
        # the live captures at both sizes never won), so this is the
        # deterministic half; the paint checks below are the user's half.
        _assert_editor_surfaces_composed(screen, "edit")
        # ...and as sampled at the rebuilt pane's own mount, before ANY sync
        # could run: deterministic by construction (5 of 5 on the base).
        assert mounted, "the work pane never mounted an editor"
        assert mounted[-1] == EXPECTED_EDIT_FLAGS_AT_MOUNT, mounted[-1]
        # One pause: the first painted frame of the rebuilt pane.
        await pilot.pause()
        assert screen._notes_state.view == "editor"
        assert _session(screen).note_id == note_id
        assert screen.query_one(NOTE_BODY, TextArea).text == "s02 paper notes"
        assert screen.query_one(NOTE_TITLE, Input).value in {"", "Untitled"}
        # AC#3, the live sequence (nothing typed after the return): the
        # review's false "changed elsewhere" render was the rebuilt work
        # pane's conflict callout, composed visible and hidden only by a
        # later presentation sync that an untouched note never triggered.
        # Fix round 1: the SAME cause showed Edit, Preview and Info stacked,
        # two Back buttons and the bulk strip with the body unpainted --
        # asserted on the compositor, immediately, before any later sync.
        _assert_editor_surfaces_painted(screen, "edit")
        _assert_no_conflict_rendered(screen)
        assert _session(screen).in_conflict is False
    profile.db.close_connection()


EDITOR_SURFACES = {
    "edit": {"painted": ("#library-note-body", "#library-note-editor-region"), "hidden": ("#library-note-preview-region", "#library-note-context-region")},
    "preview": {"painted": ("#library-note-preview-region",), "hidden": ("#library-note-body", "#library-note-context-region")},
    "context": {"painted": ("#library-note-context-region",), "hidden": ("#library-note-body", "#library-note-preview-region")},
}


#: ``display`` of the gated surfaces a retained Edit-mode note must carry the
#: instant its rebuilt pane mounts (the first frame paints from these).
EXPECTED_EDIT_FLAGS_AT_MOUNT = {
    "#library-note-editor-region": True,
    "#library-note-preview-region": False,
    "#library-note-context-region": False,
    "#library-note-back": True,
    "#library-note-context-back": False,
    "#library-note-bulk-status": False,
    "#library-note-conflict-region": False,
    "#library-note-wide-utilities": False,
    "#library-note-delete-confirmation": False,
}


def _record_surface_flags_at_mount(monkeypatch) -> list[dict[str, bool]]:
    """Sample the gated surfaces' ``display`` at every editor pane's mount.

    ``LibraryNotesCanvas._apply_post_compose_state`` runs from ``on_mount``,
    after the children exist and before any later presentation sync can
    land, so what it sees is exactly what compose produced. Recording there
    makes the base's race a certainty: on ``02b60c5e38`` the rebuilt pane
    composed every surface visible and a later sync hid them only
    sometimes (the live captures at both sizes caught the wrong frame).
    """
    from tldw_chatbook.Widgets.Library.library_notes_canvas import LibraryNotesCanvas

    samples: list[dict[str, bool]] = []
    real = LibraryNotesCanvas._apply_post_compose_state

    def sampled(self):
        if self.mode == "editor" and self.query("#library-note-body"):
            samples.append(
                {
                    selector: bool(self.query_one(selector).display)
                    for selector in EXPECTED_EDIT_FLAGS_AT_MOUNT
                }
            )
        return real(self)

    monkeypatch.setattr(LibraryNotesCanvas, "_apply_post_compose_state", sampled)
    return samples


def _assert_editor_surfaces_composed(screen, mode: str) -> None:
    """The gated surfaces' ``display`` flags as composed, before any sync."""
    for selector in EDITOR_SURFACES[mode]["painted"]:
        assert screen.query_one(selector).display is True, f"{selector} composed hidden"
    for selector in EDITOR_SURFACES[mode]["hidden"]:
        assert screen.query_one(selector).display is False, f"{selector} composed shown"
    assert screen.query_one("#library-note-bulk-status").display is False
    assert screen.query_one("#library-note-conflict-region").display is False
    assert screen.query_one("#library-note-wide-utilities").display is False
    assert screen.query_one("#library-note-delete-confirmation").display is False
    backs = [
        selector
        for selector in ("#library-note-back", "#library-note-context-back")
        if screen.query_one(selector).display
    ]
    assert len(backs) == 1, f"back buttons composed shown: {backs}"


def _painted(screen, selector) -> bool:
    widget = screen.query_one(selector)
    return widget in screen._compositor.visible_widgets and widget.region.height > 0


def _assert_editor_surfaces_painted(screen, mode: str) -> None:
    """Exactly the surfaces of ``mode`` are painted: one region, one Back,
    no bulk strip -- the retained editor as the user left it, in the first
    frame after the return (no later sync may be relied on)."""
    for selector in EDITOR_SURFACES[mode]["painted"]:
        assert _painted(screen, selector), f"{selector} is not painted in {mode}"
    if mode == "edit":
        assert screen.query_one("#library-note-body").region.height >= 6
    for selector in EDITOR_SURFACES[mode]["hidden"]:
        assert not _painted(screen, selector), f"{selector} is painted in {mode}"
    backs = [
        selector
        for selector in ("#library-note-back", "#library-note-context-back")
        if _painted(screen, selector)
    ]
    assert len(backs) == 1, f"back buttons painted: {backs}"
    assert not _painted(screen, "#library-note-bulk-status")
    assert not _painted(screen, "#library-note-wide-utilities")
    assert not _painted(screen, "#library-note-delete-confirmation")


def _assert_no_conflict_rendered(screen) -> None:
    """Neither the conflict callout nor the status line says 'changed elsewhere'.

    The callout (``#library-note-conflict-region``) is always composed and
    shown by ``display``; the review's false render is a VISIBLE one, so the
    claim is about the compositor, not the DOM.
    """
    region = screen.query_one("#library-note-conflict-region")
    assert region.display is False, "the conflict callout is displayed"
    assert region not in screen._compositor.visible_widgets, (
        "the conflict callout is painted"
    )
    assert "changed elsewhere" not in screen._notes_controller._library_note_status_line()


# --- AC#4: `n` in the Reader creates a note naming the document --------------


@pytest.mark.asyncio
async def test_n_in_the_reader_creates_a_note_naming_the_document_and_the_reader_keeps_its_place(
    tmp_path,
):
    """A note titled after the document, its first line the media:// source
    link, the caret after it; back on Media the item is at the same place."""
    host, profile = _host(tmp_path)
    async with host.run_test(size=SIZE) as pilot:
        screen = await _library(host, pilot)
        body = await _open_media_item(screen, pilot)
        await _scroll_reader(screen, pilot, body, SCROLL_Y)
        screen.set_focus(body.scroller)
        await pilot.pause()
        rows_before = len(profile.rows())

        await pilot.press("n")

        await _armed_editor(screen, pilot)
        await pilot.pause()
        snapshot = _session(screen)
        assert screen.query_one(NOTE_TITLE, Input).value == MEDIA_TITLE
        source_line = f"[{MEDIA_TITLE}](media://{MEDIA_UUID})"
        note_body = screen.query_one(NOTE_BODY, TextArea)
        assert note_body.text.startswith(source_line + "\n\n"), note_body.text
        assert note_body.cursor_location == note_body.document.end
        assert len(profile.rows()) == rows_before + 1
        row = profile.note(snapshot.note_id)
        assert row["title"] == MEDIA_TITLE
        assert row["content"].startswith(source_line)
        assert "Note from" in screen._notes_controller._library_note_status_line()

        await _type(pilot, "x")
        await _rail(screen, pilot, "media")

        assert screen.query(VIEWER_TITLE), "the Reader lost the item after n"
        assert screen._media_state.reader_session.loaded_id == LOADED_ID
        await _until(
            pilot,
            lambda: _reader_scroll_y(screen) == SCROLL_Y,
            f"the reading position to be restored (now {_reader_scroll_y(screen)})",
            timeout=10.0,
        )
        assert _loaded_rows(screen) == [MEDIA_ROW.lstrip("#")]
    profile.db.close_connection()


@pytest.mark.asyncio
async def test_programmatic_media_row_restore_keeps_reader_until_user_keys(tmp_path):
    from textual._context import active_message_pump
    from textual.events import DescendantFocus

    watched_screen = restored_target = None
    restore_focus_handled = False

    def after_restore_focus():
        nonlocal restore_focus_handled
        restore_focus_handled = True

    def observe_focus(message):
        if (
            isinstance(message, DescendantFocus)
            and watched_screen is not None
            and active_message_pump.get() is watched_screen
            and message.widget is restored_target
            and message.widget is watched_screen.focused
        ):
            watched_screen.call_later(after_restore_focus)

    host, profile = _host(tmp_path)
    try:
        async with host.run_test(size=SIZE, message_hook=observe_focus) as pilot:
            screen = await _library(host, pilot)
            body = await _open_media_item(screen, pilot)
            await pilot.press("tab")  # Real input retires initial resize suppression.
            screen.set_focus(body.scroller)
            await pilot.pause()

            def detail_retired():
                return (
                    screen._media_controller._library_media_selection_timer is None
                    and screen._media_state.reader_session.pending_request is None
                    and not any(
                        worker.group == "library_media_detail"
                        and not worker.is_finished
                        for worker in screen.workers
                    )
                )

            await _until(
                pilot,
                lambda: (
                    not screen._library_media_browse_controller.loading
                    and not screen._library_media_browse_controller.facet_loading
                    and not screen.query_one(
                        "#library-media-canvas"
                    ).has_pending_recompose_callback
                    and not screen._library_pending_list_entry_focus
                    and not screen._media_controller._library_media_find_focus_pending
                    and not screen._library_notes_restoring_focus
                    and not screen._notes_state.resize_settling
                    and screen._library_notes_programmatic_focus_target is None
                    and detail_retired()
                ),
                "the original Media entry/publication owners to retire",
            )
            session = screen._media_state.reader_session
            assert screen._media_state.selected_media_id == LOADED_ID
            assert session.selected_id == session.loaded_id == LOADED_ID
            restored_target = screen.query_one("#library-media-row-0", Button)
            assert str(restored_target.media_id) == "local:media:2"
            watched_screen = screen

            screen.set_focus(None)  # Framework/recompose focus-loss setup.
            screen._media_controller._restore_library_media_focus(
                "#library-media-row-0"
            )
            await _until(
                pilot, lambda: restore_focus_handled, "the real restore focus event"
            )
            await _until(
                pilot, detail_retired, "any focus-triggered detail work to retire"
            )
            assert screen.focused is restored_target
            assert screen._library_notes_programmatic_focus_target is None
            assert screen._media_state.selected_media_id == LOADED_ID
            assert screen._media_state.reader_session.selected_id == LOADED_ID
            assert screen._media_state.reader_session.loaded_id == LOADED_ID

            # Independent genuine-user phase: Down reaches Reader1's row,
            # Up must select/load Reader2, then Down must select/load Reader1.
            for key, expected in (
                ("down", LOADED_ID),
                ("up", "local:media:2"),
                ("down", LOADED_ID),
            ):
                await pilot.press(key)
                await _until(
                    pilot,
                    lambda expected=expected: (
                        str(getattr(screen.focused, "media_id", "")) == expected
                        and screen._media_state.selected_media_id == expected
                        and screen._media_state.reader_session.selected_id == expected
                        and screen._media_state.reader_session.loaded_id == expected
                        and detail_retired()
                    ),
                    "the real row key to focus, select and finish loading its item",
                )
            assert host._exception is None
    finally:
        profile.db.close_connection()


def test_programmatic_media_restore_releases_only_its_failed_focus_marker():
    from types import SimpleNamespace

    from textual.widget import Widget

    from tldw_chatbook.UI.Library_Modules.library_media_controller import (
        LibraryMediaController,
    )

    target = SimpleNamespace(focusable=True, has_class=lambda _: True)
    later_owner = object()

    class FocusAttempt:
        _library_pending_list_entry_focus = False
        _library_media_find_focus_pending = False
        _library_notes_programmatic_focus_target = None

        def __init__(self, outcome):
            self.outcome = outcome
            self.focused = target if outcome == "already-focused" else None
            self.focus_calls = 0

        def query_one(self, selector, widget_type):
            assert selector == "#restore-target"
            assert widget_type is Widget
            return target

        def set_focus(self, widget, *, scroll_visible):
            self.focus_calls += 1
            assert widget is target and scroll_visible is False
            assert self._library_notes_programmatic_focus_target is target
            if self.outcome == "later-owner":
                self._library_notes_programmatic_focus_target = later_owner
            if self.outcome != "noop":
                raise RuntimeError("focus refused")

    for outcome in ("already-focused", "noop", "raises", "later-owner"):
        attempt = FocusAttempt(outcome)
        if outcome in {"raises", "later-owner"}:
            with pytest.raises(RuntimeError, match="focus refused"):
                LibraryMediaController._restore_library_media_focus(
                    attempt, "#restore-target"
                )
        else:
            LibraryMediaController._restore_library_media_focus(
                attempt, "#restore-target"
            )
        expected = later_owner if outcome == "later-owner" else None
        assert attempt._library_notes_programmatic_focus_target is expected, outcome
        assert attempt.focus_calls == int(outcome != "already-focused"), outcome


# --- TASK-36952.3.3: controller-hosted Media syncs re-request armed focus -----


@pytest.mark.asyncio
async def test_controller_hosted_media_sync_re_requests_an_armed_list_entry_focus(
    tmp_path, monkeypatch
):
    """A Media canvas sync hosted by the media controller re-requests an armed
    list-entry focus with the arm's generation, as a screen-hosted sync does.

    ``_sync_library_canvas`` probes its host for
    ``_focus_library_list_entry_if_current``. TASK-36952.3.3 bound that name on
    ``LibraryMediaController`` for its moved browse sync, so the controller's
    other Media syncs (sort, sort choice, select all, select clear, delete
    selected, cancel bulk delete, bulk-delete confirm, receipt dismiss) now
    queue the re-request after the recompose too. Without the binding the
    probe finds nothing and no re-request is queued.
    """
    from types import SimpleNamespace

    host, profile = _host(tmp_path)
    async with host.run_test(size=SIZE) as pilot:
        screen = await _library(host, pilot)
        await _rail(screen, pilot, "media")
        await _wait_for_selector(screen, pilot, MEDIA_ROW)
        await screen.workers.wait_for_complete()
        await pilot.pause()
        requested: list[int] = []
        monkeypatch.setattr(
            screen, "focus_library_list_entry_if_current", requested.append
        )
        screen._library_pending_list_entry_focus = True
        generation = screen._library_list_entry_focus_generation

        screen._media_controller.handle_library_media_select_clear(
            SimpleNamespace(stop=lambda: None)
        )
        await _until(
            pilot,
            lambda: bool(requested),
            "the armed list-entry re-request after the controller-hosted sync",
            timeout=5.0,
        )
        for _ in range(4):
            await pilot.pause()

        assert requested == [generation]
    profile.db.close_connection()

#: The first row each destination lands on, once settled.
_FIRST_ROW = {"media": "library-media-row-0", "notes": "library-notes-tree-note-1"}


@pytest.mark.parametrize(
    ("notes", "captures", "setup", "switch", "expected"),
    [
        # TASK-41504: each switch the libfocus stream saw land on a pane grip
        # or the rail row, and where it must land: the first row, or the empty
        # list's documented control (the Notes filter, task-31946; Collections'
        # Quick Capture, or its filter when a scope or filter narrowed it).
        pytest.param(
            True, "", (), ("click", "notes"), _FIRST_ROW["notes"], id="rail-click-notes"
        ),
        pytest.param(
            False, "", (), ("press", "notes"), "library-notes-filter",
            id="press-empty-notes",
        ),
        pytest.param(
            False, "", (), ("click", "notes"), "library-notes-filter",
            id="click-empty-notes",
        ),
        pytest.param(
            False, "", ("media",), ("click", "notes"), "library-notes-filter",
            id="media-click-empty-notes",
        ),
        pytest.param(
            True, "", ("media",), ("press", "collections"),
            "library-collections-quick-capture",
            id="media-press-collections",
        ),
        pytest.param(
            True, "saved", ("media",), ("press", "collections"),
            "library-collections-row-0",
            id="media-press-collections-with-captures",
        ),
        pytest.param(
            True, "saved-archived-scope", ("media",), ("press", "collections"),
            "library-collections-filter",
            id="media-press-collections-empty-scope",
        ),
        pytest.param(
            True, "capture-unavailable", ("media",), ("press", "collections"),
            "library-collections-filter",
            id="media-press-collections-quick-capture-disabled",
        ),
        pytest.param(
            True, "", ("media", "notes"), ("press", "media"), _FIRST_ROW["media"],
            id="notes-press-resident-media",
        ),
        # TASK-42422: the same switch with the Media rows held mid-repaint.
        pytest.param(
            True, "", ("media", "notes"), ("press-rows-held", "media"),
            _FIRST_ROW["media"],
            id="notes-press-resident-media-rows-held",
        ),
        # Review of TASK-42422: and with the page result held, so only the
        # running repaint (no queued one) stands between the entry and ``type:``.
        pytest.param(
            True, "", ("media", "notes"), ("press-rows-and-page-held", "media"),
            _FIRST_ROW["media"],
            id="notes-press-resident-media-rows-and-page-held",
        ),
        pytest.param(
            True, "", ("notes", "media"), ("click", "notes"), _FIRST_ROW["notes"],
            id="media-click-resident-notes",
        ),
    ],
)
async def test_each_library_switch_lands_on_its_first_row_never_a_grip(
    notes, captures, setup, switch, expected, monkeypatch
):
    """TASK-41504: a rail click or a code-driven ``press()`` lands entry focus
    on the destination's first row (or its empty list's control), reachable,
    and still there once the entry window has closed -- never a pane grip
    or the rail row. Production stylesheet at 170x48, where grips were seen.
    """
    app = _build_test_app()
    _seed_conversations(
        app,
        _two_conversations(),
        notes=_two_notes() if notes else [],
        media=_two_media_items(),
    )
    scope = app.collections_capture_scope_service
    if captures == "capture-unavailable":
        # No capability answer, so Quick Capture is drawn disabled.
        async def no_capabilities():
            raise RuntimeError("capabilities unavailable")

        scope.capabilities = no_capabilities
    elif captures:
        # Two saved captures, so the entry has rows to wait for.
        for title in ("Alpha", "Beta"):
            await scope.save_capture(
                CaptureSaveRequest(
                    scope.active_authority.key,
                    f"https://example.test/{title.lower()}",
                    title=title,
                    text_content=f"Readable {title} body.",
                )
            )
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        if captures == "saved-archived-scope":
            # Nothing is archived, so the page is empty but narrowed.
            screen._collections_state.active_scope = "archived"

        def on(widget_id: str) -> bool:
            focused = screen.focused
            return (
                focused is not None
                and focused.id == widget_id
                and focused in screen.focus_chain
            )

        def where() -> str:
            focused = screen.focused
            reachable = focused is not None and focused in screen.focus_chain
            return f"focus on {focused!r} (in the focus chain: {reachable})"

        async def settle_on(widget_id: str) -> None:
            await _wait_for_condition(
                pilot, lambda: on(widget_id), message=lambda: f"{widget_id}: {where()}"
            )
            await _wait_for_condition(
                pilot,
                lambda: not screen._library_pending_list_entry_focus,
                message="The entry window never closed.",
            )

        def put_focus_on(widget_id: str) -> bool:
            """Focus ``widget_id`` until it holds, whatever the entry did."""
            if on(widget_id) and not screen._library_pending_list_entry_focus:
                return True
            for widget in screen.query(f"#{widget_id}"):
                screen.set_focus(widget)
            return False

        for row in setup:
            # The user is on the first row before the switch under test.
            screen.query_one(f"#library-row-browse-{row}").press()
            first = _FIRST_ROW[row]
            for _ in range(2):  # again after a pause, should a late repaint move it
                await _wait_for_condition(
                    pilot,
                    lambda first=first: put_focus_on(first),
                    message=lambda first=first: f"setup {first}: {where()}",
                )
                await pilot.pause(0.3)
        how, row = switch
        if captures == "saved":
            # Review r16fo-rev2: this entry waits out the whole capture
            # loader, the closest of these to the 2 s window; give it room.
            monkeypatch.setattr(
                library_screen_module, "LIBRARY_LIST_ENTRY_FOCUS_ARMED_SECONDS", 6.0
            )
        rows_absent_attempts: list[None] = []
        release_held = asyncio.Event()
        held = how in {"press-rows-held", "press-rows-and-page-held"}
        page_held = how == "press-rows-and-page-held"
        if held:
            # TASK-42422 (CI's UI Fast Lane 33): the Media repaint mounts its
            # toolbar before its rows. An entry attempt in that gap found no
            # row, landed on the empty list's ``type:`` control, and the next
            # repaint removed it, so Textual parked focus on the Library grip.
            # Hold the rows' mount until an attempt has run in that gap.
            compose_rows = LibraryMediaRowScroll._compose
            enter_list = type(screen)._focus_library_list_entry

            async def held_rows(self):
                await release_held.wait()
                await compose_rows(self)

            def counted_entry(self):
                # The first ``type:`` match, as the old one can linger.
                found = self.query("#library-media-type-filter")
                control = found.first() if found else None
                if (
                    control is not None
                    and not control._pruning
                    and control in self.focus_chain
                    and not self.query(".library-media-row")
                    # With the page held no second repaint is queued, so only
                    # the running one can tell this attempt to wait.
                    and not (
                        page_held
                        and any(
                            getattr(node, "_recompose_required", False)
                            for node in control.ancestors
                        )
                    )
                ):
                    rows_absent_attempts.append(None)
                return enter_list(self)

            monkeypatch.setattr(LibraryMediaRowScroll, "_compose", held_rows)
            monkeypatch.setattr(
                type(screen), "_focus_library_list_entry", counted_entry
            )
        if page_held:
            # Review of TASK-42422: hold the page and facet results too, as
            # when they are slow, so nothing queues the next repaint.
            for name in ("_search", "_load_facets"):
                load = getattr(LibraryMediaBrowseController, name)

                async def held_load(self, *args, _load=load, **kwargs):
                    await release_held.wait()
                    return await _load(self, *args, **kwargs)

                monkeypatch.setattr(LibraryMediaBrowseController, name, held_load)
        held_focus = None
        try:
            if how == "click":
                await pilot.click(f"#library-row-browse-{row}")
            else:
                screen.query_one(f"#library-row-browse-{row}").press()
            if held:
                # Not ``pilot.pause``: it waits for every pump, held ones too.
                for _ in range(1500):
                    if rows_absent_attempts:
                        break
                    await asyncio.sleep(0.01)
                held_focus = screen.focused
        finally:
            release_held.set()  # never leave a held mount for teardown
        if held:
            assert rows_absent_attempts, "No entry attempt saw the rows held."
            # Nothing lands on a control the repaint is still replacing.
            assert held_focus is None, f"mid-repaint focus on {held_focus!r}"
        await settle_on(expected)
        await pilot.pause(0.5)  # a late restore or repaint would show here
        assert on(expected), where()
        assert not screen.focused.has_class(LIBRARY_ADAPTIVE_READER_GRIP_CLASS)


@pytest.mark.asyncio
async def test_prompts_list_entry_waits_out_a_running_repaint(tmp_path, monkeypatch):
    """Review of TASK-42422: the Prompts list canvas is not a
    ``RecomposeCaptureGuard``, so ``recompose_in_flight`` never marked its
    repaint as running. Its filter mounts before its rows, like Media's
    toolbar. With the rows held mid-repaint and nothing else queued, a
    re-armed list entry must wait for the repaint, then land on the first
    row, not the filter the repaint is still building.
    """
    app = _build_test_app()
    app.library_new_profile_admission = False
    _seed_conversations(app, _two_conversations())
    prompts_db = PromptsDatabase(
        tmp_path / "entry-prompts.db", client_id="entry-prompts-test"
    )
    prompt_id, _uuid, _msg = prompts_db.add_prompt(
        name="Summarize",
        author="Alice",
        details="A summarizer",
        system_prompt="You are concise.",
        user_prompt="Summarize: {text}",
    )
    app.prompt_scope_service = PromptScopeService(
        local_service=LocalPromptService(prompts_db), server_service=None
    )
    host = LibraryProductionCSSHarness(app)
    row_id = f"library-prompt-row-{prompt_id}"
    release_rows = asyncio.Event()
    filter_attempts: list[None] = []
    try:
        async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            screen.query_one("#library-row-browse-prompts").press()
            await _wait_for_selector(screen, pilot, f"#{row_id}")
            await _wait_for_condition(
                pilot,
                lambda: not screen._library_pending_list_entry_focus,
                message="The Prompts entry window never closed.",
            )
            compose_list = VerticalScroll._compose
            enter_list = type(screen)._focus_library_list_entry

            async def held_list(self):
                if self.id == "library-prompts-list":
                    await release_rows.wait()
                await compose_list(self)

            def counted_entry(self):
                found = self.query("#library-prompts-filter")
                control = found.first() if found else None
                if (
                    control is not None
                    and not control._pruning
                    and control in self.focus_chain
                    and not self.query(".library-prompt-row")
                    and not any(
                        getattr(node, "_recompose_required", False)
                        for node in control.ancestors
                    )
                ):
                    filter_attempts.append(None)
                return enter_list(self)

            monkeypatch.setattr(VerticalScroll, "_compose", held_list)
            monkeypatch.setattr(
                type(screen), "_focus_library_list_entry", counted_entry
            )
            held_focus = None
            try:
                screen.set_focus(None)
                # A "back to the list" seam re-arms while the list repaints.
                screen._arm_library_list_entry_focus()
                screen.query_one("#library-prompts-canvas").refresh(recompose=True)
                # Not ``pilot.pause``: it waits for every pump, held ones too.
                for _ in range(1500):
                    if filter_attempts:
                        break
                    await asyncio.sleep(0.01)
                held_focus = screen.focused
            finally:
                release_rows.set()  # never leave a held mount for teardown
            assert filter_attempts, "No entry attempt saw the rows held."
            assert held_focus is None, f"mid-repaint focus on {held_focus!r}"
            await _wait_for_condition(
                pilot,
                lambda: getattr(screen.focused, "id", None) == row_id,
                message=lambda: f"{row_id}: focus on {screen.focused!r}",
            )
    finally:
        prompts_db.close_connection()


@pytest.mark.asyncio
async def test_notes_entry_window_close_waits_for_a_filter_being_repainted():
    """Review of TASK-42422: when the entry window closes on an empty Notes
    list with nothing focused, the expiry lands on the filter. A filter a
    queued repaint is about to replace must not take that landing (Textual's
    reset would move it to a pane grip); the expiry waits for the repaint and
    lands on the new filter.
    """
    app = _build_test_app()
    _seed_conversations(
        app, _two_conversations(), notes=[], media=_two_media_items()
    )
    host = LibraryProductionCSSHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_condition(
            pilot,
            lambda: getattr(screen.focused, "id", None) == "library-notes-filter"
            and not screen._library_pending_list_entry_focus,
            message=lambda: f"Notes entry never settled: {screen.focused!r}",
        )
        old_filter = screen.focused
        screen.set_focus(None)
        # The window is still open, and the canvas holding the filter has a
        # repaint queued, when the window's timer fires.
        screen._library_pending_list_entry_focus = True
        screen.query_one("#library-notes-canvas").refresh(recompose=True)
        screen._expire_library_list_entry_focus()
        assert screen.focused is None, f"landed on {screen.focused!r} mid-repaint"

        def on_new_filter() -> bool:
            focused = screen.focused
            return (
                getattr(focused, "id", None) == "library-notes-filter"
                and focused is not old_filter
                and focused in screen.focus_chain
                and not screen._library_pending_list_entry_focus
            )

        await _wait_for_condition(
            pilot,
            on_new_filter,
            message=lambda: f"The window closed on {screen.focused!r}",
        )
        await pilot.pause(0.5)  # a late reset would show here
        assert on_new_filter(), repr(screen.focused)
        assert not screen.focused.has_class(LIBRARY_ADAPTIVE_READER_GRIP_CLASS)


@pytest.mark.asyncio
async def test_list_entry_lands_on_mounted_rows_while_the_screen_lock_is_held():
    """Re-review of TASK-42422: the running-repaint check reads each
    ancestor's lock but stops below the Screen. Rows a whole-screen recompose
    has mounted are final, and waiting for the whole screen to mount could
    outlast the entry window and leave focus unset. With the screen's lock
    held and the Media rows mounted, a re-armed entry still lands on row 0.
    """
    app = _build_test_app()
    _seed_conversations(
        app, _two_conversations(), notes=_two_notes(), media=_two_media_items()
    )
    host = LibraryProductionCSSHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-media").press()
        await _wait_for_condition(
            pilot,
            lambda: getattr(screen.focused, "id", None) == _FIRST_ROW["media"]
            and not screen._library_pending_list_entry_focus,
            message=lambda: f"Media entry never settled: {screen.focused!r}",
        )
        release_screen = asyncio.Event()

        async def hold_screen_lock() -> None:
            async with screen.lock:  # as a running screen recompose holds it
                await release_screen.wait()

        holder = asyncio.create_task(hold_screen_lock())
        landed = None
        try:
            for _ in range(100):
                if screen.lock.is_locked:
                    break
                await asyncio.sleep(0.01)
            assert screen.lock.is_locked
            screen.set_focus(None)
            screen._arm_library_list_entry_focus()
            for _ in range(150):  # 1.5 s, inside the 2 s window
                if screen.focused is not None:
                    break
                await asyncio.sleep(0.01)
            landed = screen.focused
        finally:
            release_screen.set()
            await holder
        assert getattr(landed, "id", None) == _FIRST_ROW["media"], repr(landed)


@pytest.mark.asyncio
async def test_notes_window_close_lands_anyway_after_its_retry_cap(monkeypatch):
    """Re-review of TASK-42422: the window-close wait for a repainting Notes
    filter is bounded. With the filter always reading as replaced, the
    expiry retries ``LIBRARY_LIST_ENTRY_FOCUS_EXPIRY_RETRIES`` times, then
    lands on the filter and disarms, and schedules nothing more.
    """
    app = _build_test_app()
    _seed_conversations(
        app, _two_conversations(), notes=[], media=_two_media_items()
    )
    host = LibraryProductionCSSHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_condition(
            pilot,
            lambda: getattr(screen.focused, "id", None) == "library-notes-filter"
            and not screen._library_pending_list_entry_focus,
            message=lambda: f"Notes entry never settled: {screen.focused!r}",
        )
        calls: list[int] = []
        expire = type(screen)._expire_library_list_entry_focus

        def counted_expire(self, *args):
            calls.append(len(calls))
            return expire(self, *args)

        monkeypatch.setattr(
            library_screen_module, "_library_entry_target_replaced", lambda _w: True
        )
        monkeypatch.setattr(
            type(screen), "_expire_library_list_entry_focus", counted_expire
        )
        screen.set_focus(None)
        screen._library_pending_list_entry_focus = True
        screen._expire_library_list_entry_focus()
        assert screen.focused is None  # waiting, not landed
        cap = LIBRARY_LIST_ENTRY_FOCUS_EXPIRY_RETRIES
        await _wait_for_condition(
            pilot,
            lambda: not screen._library_pending_list_entry_focus,
            message=lambda: f"The window never closed after {len(calls)} calls.",
        )
        assert len(calls) == cap + 1, calls
        assert getattr(screen.focused, "id", None) == "library-notes-filter"
        await pilot.pause(0.3)  # nothing left scheduled
        assert len(calls) == cap + 1, calls
