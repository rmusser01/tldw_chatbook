"""Shared harness for the TASK-34000.11 note-link Pilot tests.

Three files use it: the two PR-gated census files
``test_library_note_links_picker_and_preview.py`` (insert + Escape) and
``test_library_note_links_preview_opens_in_app.py`` (Preview links), and the
``_extended.py`` sibling outside the census. Not a test module, so
pytest does not collect it (no ``test_`` prefix).

Every test drives the real ``TldwCli`` booted into Library over a real
ChaChaNotes database (the quit-guard profile plus the review's two notes),
with the production stylesheet, at the review's size. The OS URL handler is
replaced by a recorder so "never reaches ``app.open_url``" is a counted
assertion rather than an absence nobody checked.
"""

from __future__ import annotations

import asyncio
import time
import webbrowser

from textual import events
from textual.widgets import Button, Static, TextArea

from tldw_chatbook.Library.library_notes_tree_paging import NotesBranchKey
from tldw_chatbook.UI.Library_Modules.screen_constants import (
    LIBRARY_ROW_BROWSE_NOTES,
    LIBRARY_ROW_CREATE_NOTE,
)

from Tests.UI.app_factory import _build_test_app
from Tests.UI.library_quit_guard_support import (
    _armed_editor,
    _library as _mounted_library,
    _NotesProfile,
    _until,
)
from Tests.UI.test_library_shell import _seed_conversations

LISBON = "Trip planning: Lisbon"
INDEX = "Index — start here"
WEB = "Web pointer"
STALE = "Stale link"
TWO = "Two links"
WEB_URL = "https://example.test/x"

#: The notices (``markup=False``), verbatim from the task brief.
DELETED_NOTICE = "That note isn't in the Library — it may have been deleted."
SELF_NOTICE = "This link points at the note you are reading."


class _LinksProfile(_NotesProfile):
    """The quit-guard profile plus the review's linked pair and three probes."""

    def __init__(self, tmp_path) -> None:
        super().__init__(tmp_path)
        self.lisbon_id = self.db.add_note(LISBON, "itinerary")
        self.index_id = self.db.add_note(
            INDEX, f"Start with [[{LISBON}]](note://{self.lisbon_id}) today."
        )
        self.web_id = self.db.add_note(WEB, f"See [docs]({WEB_URL}) for more.")
        gone_id = self.db.add_note("Gone", "soon deleted")
        gone = self.db.get_note_by_id(gone_id)
        assert self.db.soft_delete_note(gone_id, gone["version"]) is True
        self.gone_id = gone_id
        self.stale_id = self.db.add_note(STALE, f"Old: [[Gone]](note://{gone_id}).")
        self.two_id = self.db.add_note(
            TWO,
            f"First [[{LISBON}]](note://{self.lisbon_id}), then "
            f"[[{INDEX}]](note://{self.index_id}).",
        )
        assert all(
            (self.lisbon_id, self.index_id, self.web_id, self.stale_id, self.two_id)
        )


def _app(tmp_path):
    """The production app booted straight into Library over the links corpus."""
    app = _build_test_app(configured_default="library")
    _seed_conversations(app, [])
    profile = _LinksProfile(tmp_path)
    app.chachanotes_db = profile.db
    app.notes_scope_service = profile.scope_service
    app.notes_service = profile.interop
    return app, profile


async def _library(app, pilot):
    """Enter the link journey only after a healthy, published source snapshot."""
    screen = await _mounted_library(app, pilot)
    for _ in range(2):
        if screen._library_lookup_error is None:
            break
        await _until(
            pilot,
            lambda: screen._library_lookup_error is None
            or bool(screen.query("#library-source-retry")),
            "the real source Retry control",
        )
        if screen._library_lookup_error is None:
            break
        previous_generation = screen._library_snapshot_state_generation
        screen.query_one("#library-source-retry", Button).press()
        await _until(
            pilot,
            lambda: screen._library_snapshot_state_generation > previous_generation,
            "source Retry to publish its completed result",
        )
    assert (
        screen._library_loaded and screen._library_lookup_error is None
    ), "Notes link setup never obtained a healthy source snapshot"
    await _until(
        pilot,
        lambda: screen._library_loaded
        and screen._library_lookup_error is None
        and screen._library_snapshot_rendered_generation
        == screen._library_snapshot_state_generation
        and not screen._library_entry_reconcile_dirty
        and not screen.query("#library-canvas-error")
        and bool(screen.query("#library-row-browse-notes"))
        and screen.query_one("#library-row-browse-notes").is_parent_active,
        "the healthy Library source DOM",
    )
    return screen


def _record_open_url(monkeypatch, app) -> list[str]:
    """Replace the OS URL handler with a recorder; the list is the evidence."""
    opened: list[str] = []
    monkeypatch.setattr(app, "open_url", lambda url, *a, **k: opened.append(str(url)))
    return opened


def _record_browser(monkeypatch) -> list[str]:
    """Replace ``webbrowser.open`` (the http(s) policy's own seam)."""
    calls: list[str] = []

    def _open(url, *args, **kwargs):
        calls.append(str(url))
        return True

    monkeypatch.setattr(webbrowser, "open", _open)
    return calls


def _painted_text_in_region(app, region) -> str:
    """Only the compositor cells painted inside ``region``, one line."""
    strips = list(app.screen._compositor.render_strips())
    rows = (
        strips[y].crop(region.x, region.right).text
        for y in range(max(region.y, 0), min(region.bottom, len(strips)))
    )
    return " ".join(" ".join(rows).split())


def _painted(app, screen, selector: str) -> str:
    return _painted_text_in_region(app, screen.query_one(selector).region)


async def _until_painted(app, screen, pilot, selector, needle, *, timeout=15.0):
    def _found() -> bool:
        return bool(screen.query(selector)) and needle in _painted(
            app, screen, selector
        )

    try:
        await _until(
            pilot, _found, f"{needle!r} painted in {selector}", timeout=timeout
        )
    except AssertionError as exc:
        current = (
            _painted(app, screen, selector) if screen.query(selector) else "<absent>"
        )
        raise AssertionError(f"{exc}; {selector} paints: {current!r}") from None


def _painted_offset(app, region, needle: str) -> tuple[int, int] | None:
    """Screen offset of the first painted ``needle`` inside ``region``."""
    strips = list(app.screen._compositor.render_strips())
    for y in range(max(region.y, 0), min(region.bottom, len(strips))):
        text = strips[y].text
        index = text.find(needle, region.x)
        if 0 <= index < region.right:
            return (index + 1, y)
    return None


def _picker_or_none(screen):
    found = screen.query("#library-note-link-picker")
    return found.first() if found else None


def _picker_open(screen) -> bool:
    picker = _picker_or_none(screen)
    return picker is not None and bool(picker.display)


def _notes_tree_publication(screen):
    """The current settled visit and its live, fully published note rows."""
    state = screen._notes_state
    if (
        state.view != "list"
        or screen._library_selected_row_id != LIBRARY_ROW_BROWSE_NOTES
        or screen._library_note_session.snapshot is not None
        or state.selected_note_id
    ):
        return None
    keys = tuple(
        NotesBranchKey(parent, kind)
        for parent in (None, *sorted(state.tree_expanded_ids))
        for kind in ("folders", "placements")
    )
    branches = tuple(state.tree_branches.get(key) for key in keys)
    if any(
        branch is None
        or branch.loading
        or branch.error
        or branch.freshness != "fresh"
        or branch.topology_epoch != state.tree_topology_epoch
        or branch.generation != state.tree_request_generations.get(key)
        for key, branch in zip(keys, branches)
    ):
        return None
    canvases = list(screen.query("#library-notes-canvas, #library-note-work-pane"))
    rows = tuple(screen.query(".library-notes-tree-note-row").results(Button))
    if len(canvases) != 2 or any(
        canvas._recompose_required or canvas.recompose_in_flight for canvas in canvases
    ):
        return None
    for widget in (*canvases, *rows):
        if not widget.is_parent_active or any(
            not node.is_attached or node._pruning or node._closing
            for node in widget.ancestors_with_self
            if node is not screen.app
        ):
            return None
    projection = screen._build_library_notes_tree_projection()
    if projection is None:
        return None
    expected = tuple(
        (row.placement_id, row.note_id) for row in projection.rows if row.kind == "note"
    )
    if tuple((row.placement_id, row.note_id) for row in rows) != expected:
        return None
    return state.tree_topology_epoch, branches, expected, rows


async def _open_notes(screen, pilot, *, deadline: float | None = None) -> None:
    """Press the rail's Notes row (``Button.press()``, as the quit-guard
    support does). NOT ``pilot.click``: from inside an open note the rail
    row is not displayed, its region is empty, and a click on an empty
    region lands at the screen origin -- the nav bar's Home tab (measured:
    the app switched to ``HomeScreen`` mid-test)."""
    deadline = time.monotonic() + 30.0 if deadline is None else deadline
    prior_epoch = screen._notes_state.tree_topology_epoch
    prior_navigation = screen._library_navigation_context_generation
    # Retained Create -> Browse acknowledges the current visit; other rail
    # entries request a fresh visit, including pressing an already-list route.
    fresh_visit = screen._library_selected_row_id != LIBRARY_ROW_CREATE_NOTE
    screen.query_one("#library-row-browse-notes", Button).press()
    await _until(
        pilot,
        lambda: screen._library_navigation_context_generation > prior_navigation
        and (not fresh_visit or screen._notes_state.tree_topology_epoch > prior_epoch)
        and _notes_tree_publication(screen) is not None,
        "the acknowledged Notes visit and its tree DOM publication",
        timeout=max(0.0, deadline - time.monotonic()),
    )


async def _open_note(screen, pilot, title: str, note_id: str) -> TextArea:
    """The list row press, as ``library_quit_guard_support._open_note`` does."""
    deadline = time.monotonic() + 30.0
    await _open_notes(screen, pilot, deadline=deadline)

    def _row():
        for button in screen.query(".library-notes-tree-note-row").results(Button):
            if button.note_id == note_id and title in str(button.label):
                return button
        return None

    while True:
        assert (
            time.monotonic() < deadline
        ), f"timed out waiting for the published row for {title!r}"
        await _until(
            pilot,
            lambda: _notes_tree_publication(screen) is not None and _row() is not None,
            f"the published row for {title!r}",
            timeout=max(0.0, deadline - time.monotonic()),
        )
        publication = _notes_tree_publication(screen)
        row = _row()
        await pilot.pause()
        # Reacquire after a native layout/message turn. A retired Button can
        # still be attached while its queued Pressed loses its parent.
        if publication == _notes_tree_publication(screen) and _row() is row:
            row.press()
            break
    await _armed_editor(screen, pilot, note_id)
    return screen.query_one("#library-note-body", TextArea)


async def _preview(screen, pilot, needle: str, app) -> None:
    """Press Preview; await its current painted DOM over a native turn."""

    def _publication():
        snapshot = screen._library_note_session.snapshot
        regions = list(screen.query("#library-note-preview-region"))
        bodies = list(screen.query("#library-note-preview-body"))
        canvases = list(screen.query("#library-notes-canvas, #library-note-work-pane"))
        if (
            snapshot is None
            or not screen._notes_state.preview
            or len(regions) != 1
            or len(bodies) != 1
            or len(canvases) != 2
            or any(c._recompose_required or c.recompose_in_flight for c in canvases)
        ):
            return None
        region, body = regions[0], bodies[0]
        if not region.display or not body.display or screen.focused is not region:
            return None
        for widget in (*canvases, region, body):
            if not widget.is_parent_active or any(
                not node.is_attached or node._pruning or node._closing
                for node in widget.ancestors_with_self
                if node is not screen.app
            ):
                return None
        if needle not in _painted(app, screen, "#library-note-preview-body"):
            return None
        return snapshot.note_id, snapshot.body, tuple(canvases), region, body

    try:
        async with asyncio.timeout(30.0):
            screen.query_one("#library-note-preview", Button).press()
            await _until(
                pilot,
                lambda: bool(screen.query("#library-note-preview-region"))
                and screen.query_one("#library-note-preview-region").display,
                "Preview to show",
            )
            await _until_painted(
                app, screen, pilot, "#library-note-preview-body", needle
            )
            while True:
                await _until(
                    pilot,
                    lambda: _publication() is not None,
                    "the current Preview DOM publication",
                )
                publication = _publication()
                await pilot.pause()
                if publication == _publication():
                    return
    except TimeoutError as exc:
        raise AssertionError(
            "timed out waiting for the current Preview DOM publication"
        ) from exc


async def _click_preview_link(app, screen, pilot, label: str) -> None:
    offset = None

    def _link_is_painted() -> bool:
        nonlocal offset
        preview = screen.query("#library-note-preview-body")
        if not preview:
            return False
        offset = _painted_offset(app, preview.first().region, label)
        return offset is not None

    await _until(pilot, _link_is_painted, f"{label!r} painted for clicking")
    assert offset is not None, f"{label!r} is not painted in the preview body"
    await pilot.click(offset=offset)


def _title_value(screen) -> str:
    from textual.widgets import Input

    return screen.query_one("#library-note-title", Input).value


def _backlink_header(screen) -> str:
    found = screen.query("#library-note-context-backlinks-title")
    if not found:
        return ""
    widget = found.first()
    assert isinstance(widget, Static)
    return str(widget.renderable)


async def _info_backlinks(screen, pilot, expected_header: str) -> list[str]:
    """Open Info, wait for ``expected_header``, return the row labels."""
    screen.query_one("#library-note-context", Button).press()
    await _until(
        pilot,
        lambda: _backlink_header(screen) == expected_header,
        f"Info to read {expected_header!r} (reads {_backlink_header(screen)!r})",
    )
    return [
        str(button.label)
        for button in screen.query(".library-note-backlink").results(Button)
    ]


def _send_key(app, key: str, character: str | None = None) -> None:
    """Deliver one key through the driver WITHOUT a pilot pause.

    Two or more of these in a row arrive in the same read, the way a real
    terminal delivers a fast burst (the reviewer's probe technique):
    ``pilot.press`` pauses between keys and can never reproduce it.
    """
    event = events.Key(key, character)
    event.set_sender(app)
    app._driver.send_message(event)


def _burst(app, text: str, *trailing: str) -> None:
    """Send ``text`` one key per character, then ``trailing`` keys, as one burst."""
    for char in text:
        _send_key(
            app,
            "left_square_bracket"
            if char == "["
            else ("space" if char == " " else char),
            char,
        )
    for key in trailing:
        _send_key(app, key, None)
