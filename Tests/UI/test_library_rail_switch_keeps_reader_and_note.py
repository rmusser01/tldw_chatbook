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

import pytest
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
    LibraryHarness,
    _active_library_screen,
    _build_test_app,
    _seed_conversations,
    _wait_for_library_shell,
    _wait_for_selector,
)
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
