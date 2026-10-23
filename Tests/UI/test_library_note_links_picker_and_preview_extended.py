"""TASK-34000.11, the slower arms (NOT in ``scripts/ui_pr_gate_census.txt``).

Enter on the focused Preview follows the note's one link or lists several;
the picker at 80x24 stays inside the pane; a 30-title corpus caps the rows
at eight and Down wraps; a title carrying ``[b]`` paints literally in the
picker and lands in the inserted link.
"""

from __future__ import annotations

import pytest
from textual.widgets import OptionList, TextArea

from Tests.UI.library_note_links_support import (
    INDEX,
    LISBON,
    TWO,
    _app,
    _armed_editor,
    _burst,
    _open_note,
    _painted,
    _painted_offset,
    _picker_open,
    _picker_or_none,
    _preview,
    _record_browser,
    _record_open_url,
    _send_key,
    _title_value,
    _until,
    _until_painted,
)
from Tests.UI.library_quit_guard_support import (
    SIZE,
    _library,
    _new_blank_note,
    _record_toasts,
    _type,
)
from textual.widgets import Markdown
from Tests.UI.test_library_notes_files_sync_journey import (
    _isolate_actor_pack_recovery,  # noqa: F401 - autouse fixture, re-exported
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

COMPACT = (80, 24)


def _footer_pairs(screen) -> tuple[tuple[str, str], ...]:
    return tuple(screen._notes_controller._library_notes_footer_shortcuts())


async def _body_focused(screen, pilot) -> TextArea:
    await _new_blank_note(screen, pilot)
    await _type(pilot, "T")
    body = screen.query_one("#library-note-body", TextArea)
    await pilot.click("#library-note-body")
    await _until(pilot, lambda: screen.focused is body, "the body to take focus")
    return body


async def test_enter_on_the_focused_preview_follows_the_only_link_and_lists_several(
    tmp_path, monkeypatch
) -> None:
    """Sharper AC#3 (owner question 1)."""
    app, profile = _app(tmp_path)
    opened = _record_open_url(monkeypatch, app)
    async with app.run_test(size=SIZE) as pilot:
        screen = await _library(app, pilot)
        await _open_note(screen, pilot, INDEX, profile.index_id)
        await _preview(screen, pilot, LISBON, app)
        region = screen.query_one("#library-note-preview-region")
        await _until(pilot, lambda: screen.focused is region, "Preview to take focus")
        assert ("enter", "follow link") in _footer_pairs(screen), _footer_pairs(screen)

        await pilot.press("enter")
        await _armed_editor(screen, pilot, profile.lisbon_id)
        assert _title_value(screen) == LISBON
        assert opened == []

        # A note with no link: no chip, and Enter leaves the Preview alone.
        await _preview(screen, pilot, "itinerary", app)
        await _until(pilot, lambda: screen.focused is region_of(screen), "Preview focus")
        assert ("enter", "follow link") not in _footer_pairs(screen)
        await pilot.press("enter")
        await pilot.pause(0.4)
        assert screen._library_note_session.snapshot.note_id == profile.lisbon_id
        assert screen.query_one("#library-note-preview-region").display

        # Two links: Enter lists both labels; Down + Enter opens the second.
        await _open_note(screen, pilot, TWO, profile.two_id)
        await _preview(screen, pilot, INDEX, app)
        await _until(pilot, lambda: screen.focused is region_of(screen), "Preview focus")
        await pilot.press("enter")
        await _until_painted(app, screen, pilot, "#library-note-link-picker", INDEX)
        painted = _painted(app, screen, "#library-note-link-picker")
        assert LISBON in painted and INDEX in painted, painted
        await pilot.press("down")
        await pilot.press("enter")
        await _armed_editor(screen, pilot, profile.index_id)
        assert _title_value(screen) == INDEX
        assert not _picker_open(screen)
        assert opened == []


def region_of(screen):
    found = screen.query("#library-note-preview-region")
    return found.first() if found else None


LONG_TITLE = (
    "A very long note title that keeps going so we can test how the tree, "
    "the editor header and the Info pane cope with it at every width"
)


async def test_the_picker_fits_inside_the_pane_at_80x24(tmp_path, monkeypatch) -> None:
    """The picker stays inside the editor pane, keeps the CARET ROW visible
    (fix round 1, M-5: with the caret on the body's last row the box was
    clamped over it), and a title wider than the picker takes ONE row, cut
    with an ellipsis (live at 160x45 and 80x24 the seed's long title wrapped
    to two rows and hid the eighth row)."""
    app, profile = _app(tmp_path)
    _record_open_url(monkeypatch, app)
    assert profile.db.add_note(LONG_TITLE, "wide")
    tall_id = profile.db.add_note("Tall body", "\n".join(f"line {i}" for i in range(12)) + "\n")
    async with app.run_test(size=COMPACT) as pilot:
        screen = await _library(app, pilot)
        body = await _open_note(screen, pilot, "Tall body", tall_id)
        await pilot.click("#library-note-body")
        await _until(pilot, lambda: screen.focused is body, "the body to take focus")
        await pilot.press("ctrl+end")
        await pilot.pause(0.2)
        await _type(pilot, "[[")
        await _until_painted(app, screen, pilot, "#library-note-link-picker", LISBON)
        await pilot.pause(0.2)
        picker = _picker_or_none(screen)
        caret_row = body.cursor_screen_offset.y
        assert not (picker.region.y <= caret_row < picker.region.bottom), (
            f"the picker {picker.region} covers the caret row {caret_row}"
        )
        pane = screen.query_one("#library-note-editor-region").region
        assert picker.region.x >= pane.x and picker.region.right <= pane.right, (
            picker.region,
            pane,
        )
        assert picker.region.y >= 0 and picker.region.bottom <= app.size.height
        assert picker.region.width <= app.size.width
        options = screen.query_one("#library-note-link-picker-options", OptionList)
        await _until(pilot, lambda: options.option_count >= 2, "the rows")
        # Never taller than rows + the two border rows (no title takes a
        # second row); M-5 may cut it SHORTER to keep the caret row visible.
        assert 3 <= picker.region.height <= options.option_count + 2, (
            picker.region,
            options.option_count,
        )
        painted = _painted(app, screen, "#library-note-link-picker")
        assert "A very long note title" in painted and "…" in painted, painted
        assert "every width" not in painted, painted


async def test_thirty_titles_cap_the_rows_at_eight_and_down_wraps(
    tmp_path, monkeypatch
) -> None:
    app, profile = _app(tmp_path)
    _record_open_url(monkeypatch, app)
    for index in range(30):
        assert profile.db.add_note(f"Topic {index:02d}", "corpus")
    async with app.run_test(size=SIZE) as pilot:
        screen = await _library(app, pilot)
        await _body_focused(screen, pilot)
        # Fix round 1 (M-6): a one-character query is matched whole by Task
        # 3's rule (Review Focus 3), so it must not claim "No note titled";
        # the picker asks for more and the next letter lists the corpus.
        await _type(pilot, "[[t")
        await _until_painted(app, screen, pilot, "#library-note-link-picker", "Keep typing")
        assert "No note titled" not in _painted(app, screen, "#library-note-link-picker")
        await _type(pilot, "o")
        await _until_painted(app, screen, pilot, "#library-note-link-picker", "Topic")
        options = screen.query_one("#library-note-link-picker-options", OptionList)
        await _until(pilot, lambda: options.option_count == 8, "eight rows")
        assert options.highlighted == 0
        for _ in range(9):
            await pilot.press("down")
        await _until(pilot, lambda: options.highlighted == 1, "Down x9 to wrap to row 1")


async def test_a_title_with_markup_paints_literally_and_lands_in_the_link(
    tmp_path, monkeypatch
) -> None:
    """Review Focus 5: ``[b]`` is painted as text, never parsed as markup; the
    inserted link is one parseable ``[[…]](note://id)``."""
    from tldw_chatbook.Notes.note_import_plan_models import stored_note_link

    app, profile = _app(tmp_path)
    _record_open_url(monkeypatch, app)
    bold_id = profile.db.add_note("Bold [b] title", "markup")
    assert bold_id
    async with app.run_test(size=SIZE) as pilot:
        screen = await _library(app, pilot)
        body = await _body_focused(screen, pilot)
        await _type(pilot, "[[Bold")
        await _until_painted(app, screen, pilot, "#library-note-link-picker", "Bold [b] title")
        await pilot.press("enter")
        expected = stored_note_link("Bold [b] title", bold_id)
        await _until(pilot, lambda: body.text == expected, f"the body to read {expected!r}")
        assert "(b)" in expected and "[b]" not in expected


# ---------------------------------------------------------------------------
# Fix round 1 pins
# ---------------------------------------------------------------------------


async def test_a_down_enter_burst_in_follow_mode_opens_the_highlighted_link(
    tmp_path, monkeypatch
) -> None:
    """I-1: Down and Enter arriving in one read must open the SECOND link.
    Every follow key used to go through an exclusive worker, so Enter's
    cancelled the pending Down (3 of 3 probe runs opened Lisbon)."""
    app, profile = _app(tmp_path)
    opened = _record_open_url(monkeypatch, app)
    async with app.run_test(size=SIZE) as pilot:
        screen = await _library(app, pilot)
        for _attempt in range(2):
            await _open_note(screen, pilot, TWO, profile.two_id)
            await _preview(screen, pilot, INDEX, app)
            region = screen.query_one("#library-note-preview-region")
            await _until(pilot, lambda: screen.focused is region, "Preview focus")
            await pilot.press("enter")
            await _until_painted(app, screen, pilot, "#library-note-link-picker", INDEX)
            _send_key(app, "down")
            _send_key(app, "enter")

            def _opened_one() -> bool:
                snapshot = screen._library_note_session.snapshot
                return (
                    snapshot is not None
                    and snapshot.note_id in (profile.index_id, profile.lisbon_id)
                    and screen._notes_state.editor_armed
                )

            await _until(pilot, _opened_one, "a link to open")
            assert screen._library_note_session.snapshot.note_id == profile.index_id, (
                "the burst opened the FIRST link"
            )
        assert opened == []


async def test_a_type_and_enter_burst_never_inserts_a_stale_row(tmp_path, monkeypatch) -> None:
    """I-2: while a narrower query's load is in flight, Enter must not accept
    the rows of the previous query (the probe inserted ``[[Burst 0]]`` for
    ``Lisb``+Enter and autosave then recorded a WRONG backlink). The accept
    is held until the current answer lands, then inserts Lisbon."""
    from tldw_chatbook.Notes.note_import_plan_models import stored_note_link

    app, profile = _app(tmp_path)
    _record_open_url(monkeypatch, app)
    ids = [profile.db.add_note(f"Burst {i}", "") for i in range(2)]
    expected = stored_note_link(LISBON, profile.lisbon_id)
    async with app.run_test(size=SIZE) as pilot:
        screen = await _library(app, pilot)
        # Variant B: the picker is open on the recent rows, then "Lisb"+Enter
        # arrive together.
        body = await _open_note(screen, pilot, "Burst 0", ids[0])
        await pilot.click("#library-note-body")
        await _until(pilot, lambda: screen.focused is body, "focus")
        _burst(app, "[[")
        await _until(pilot, lambda: _picker_open(screen), "the picker on [[")
        _burst(app, "Lisb", "enter")
        await _until(pilot, lambda: body.text == expected, f"the body to read {expected!r}")
        assert "Burst" not in body.text
        # Variant A: everything in one read. No picker has shown by the time
        # Enter lands, so Enter is the editor's own newline -- and nothing
        # stale is ever inserted (the probe's base result, kept as the
        # negative control).
        body = await _open_note(screen, pilot, "Burst 1", ids[1])
        await pilot.click("#library-note-body")
        await _until(pilot, lambda: screen.focused is body, "focus")
        _burst(app, "[[Lisb", "enter")
        await _until(pilot, lambda: body.text == "[[Lisb\n", "the newline to land")
        await pilot.pause(0.8)
        assert body.text == "[[Lisb\n" and "note://" not in body.text
        assert not _picker_open(screen)


async def test_escape_latch_clears_on_a_new_context_and_a_note_switch(tmp_path, monkeypatch) -> None:
    """M-12: the Escape latch is the dismissed context's START; a DIFFERENT
    context (another line, another note) must open the picker again."""
    app, profile = _app(tmp_path)
    _record_open_url(monkeypatch, app)
    latch_id = profile.db.add_note("Latch probe", "[[Tri\n[[Lisb")
    async with app.run_test(size=SIZE) as pilot:
        screen = await _library(app, pilot)
        body = await _open_note(screen, pilot, "Latch probe", latch_id)
        await pilot.click("#library-note-body")
        await _until(pilot, lambda: screen.focused is body, "focus")
        await pilot.press("ctrl+home")
        await pilot.press("end")
        await _until_painted(app, screen, pilot, "#library-note-link-picker", LISBON)
        await pilot.press("escape")
        await _until(pilot, lambda: not _picker_open(screen), "Escape to close")
        assert body.cursor_location == (0, 5)
        await pilot.press("down")
        await _until(pilot, lambda: body.cursor_location == (1, 5), "the caret on line 1")
        await _until_painted(app, screen, pilot, "#library-note-link-picker", LISBON)
        # The note switch: latch line 1, open another note, come back.
        await pilot.press("escape")
        await _until(pilot, lambda: not _picker_open(screen), "Escape to close")
        await _open_note(screen, pilot, LISBON, profile.lisbon_id)
        body = await _open_note(screen, pilot, "Latch probe", latch_id)
        await pilot.click("#library-note-body")
        await _until(pilot, lambda: screen.focused is body, "focus")
        await pilot.press("ctrl+end")
        await _until(pilot, lambda: body.cursor_location == (1, 6), "the caret at the end")
        await _until_painted(app, screen, pilot, "#library-note-link-picker", LISBON)


HOSTILE_RENDERED = (
    ("QUPPERQ", "HTTP://up.example/"),
    ("QPROTOQ", "//evil.example/"),
    ("QFTPQ", "ftp://a/[b]x[/]"),
    ("QCTRLQ", "https://a.example/%1B]0;x%07"),
    ("QMARKQ", "https://x.example/[b]bold[/]"),
)
HOSTILE_POSTED = (
    "javascript:alert(1)",
    "file:///etc/passwd",
    "data:text/html,hi",
    "vbscript:msgbox(1)",
    "https://a.example/\x1b]0;x\x07",
    "https://a.example/with space",
)


def _has_control_chars(text: str) -> bool:
    return any(ord(ch) < 32 or 127 <= ord(ch) <= 159 for ch in text)


async def test_hostile_preview_links_are_refused_and_never_reach_the_os(tmp_path, monkeypatch) -> None:
    """I-5 / M-9: every non-http(s) scheme, a scheme-relative link, an
    uppercase scheme and an http(s) href carrying control characters or
    whitespace are refused with the notice (control characters stripped
    from it); the one clean http link reaches the browser exactly once;
    ``app.open_url`` is never called; the body is unchanged; the app lives.
    markdown-it renders ``javascript:``/``file:``/``data:``/``vbscript:``
    links as literal text, so those arrive as posted ``LinkClicked`` events."""
    app, profile = _app(tmp_path)
    body_text = "\n\n".join(f"[{label}]({href})" for label, href in HOSTILE_RENDERED)
    hostile_id = profile.db.add_note("Hostile links", body_text)
    opened = _record_open_url(monkeypatch, app)
    browser = _record_browser(monkeypatch)
    clean_http = "https://x.example/[b]bold[/]"
    async with app.run_test(size=SIZE) as pilot:
        screen = await _library(app, pilot)
        toasts = _record_toasts(monkeypatch, app)
        await _open_note(screen, pilot, "Hostile links", hostile_id)
        await _preview(screen, pilot, "QUPPERQ", app)
        region = screen.query_one("#library-note-preview-body").region
        for label, href in HOSTILE_RENDERED:
            before = len(toasts)
            offset = _painted_offset(app, region, label)
            assert offset is not None, f"{label} not painted"
            await pilot.click(offset=offset)
            await _until(pilot, lambda: len(toasts) > before, f"a notice for {href!r}")
            notice = toasts[before]
            if href == clean_http:
                assert notice == f"Opened in browser: {href}"
                await _until(pilot, lambda: browser == [clean_http], "the one browser call")
            else:
                assert notice.startswith("Link not opened (unsupported scheme): "), notice
                assert browser == [], (href, browser)
            assert not _has_control_chars(notice), repr(notice)
        md = screen.query_one("#library-note-preview-body", Markdown)
        for href in HOSTILE_POSTED:
            before = len(toasts)
            md.post_message(Markdown.LinkClicked(md, href))
            await _until(pilot, lambda: len(toasts) > before, f"a notice for {href!r}")
            notice = toasts[before]
            assert notice.startswith("Link not opened (unsupported scheme): "), notice
            assert not _has_control_chars(notice), repr(notice)
        await pilot.pause(0.5)
        assert browser == [clean_http], browser
        assert opened == [], opened
        assert profile.db.get_note_by_id(hostile_id)["content"] == body_text
        assert screen._library_note_session.snapshot.body == body_text
        assert app.is_running and app._exception is None


async def test_the_picker_hides_when_the_body_loses_focus(tmp_path, monkeypatch) -> None:
    """M-4: a click on the Title or Shift+Tab out of the body closes the
    picker; the probe found it stayed open with the body's key routing armed."""
    app, profile = _app(tmp_path)
    _record_open_url(monkeypatch, app)
    async with app.run_test(size=SIZE) as pilot:
        screen = await _library(app, pilot)
        body = await _body_focused(screen, pilot)
        await _type(pilot, "[[")
        await _until_painted(app, screen, pilot, "#library-note-link-picker", LISBON)
        await pilot.click("#library-note-title")
        await _until(
            pilot, lambda: getattr(screen.focused, "id", None) == "library-note-title", "the title"
        )
        await _until(pilot, lambda: not _picker_open(screen), "the picker to close on blur")
        assert body.link_picker_open is False
        await pilot.click("#library-note-body")
        await _until(pilot, lambda: screen.focused is body, "focus")
        await pilot.press("end")
        await _type(pilot, "x [[")
        await _until_painted(app, screen, pilot, "#library-note-link-picker", LISBON)
        await pilot.press("shift+tab")
        await _until(pilot, lambda: screen.focused is not body, "focus to leave the body")
        await _until(pilot, lambda: not _picker_open(screen), "the picker to close on Shift+Tab")
        assert body.link_picker_open is False


async def test_accept_consumes_a_closer_a_click_accepts_and_enter_on_no_rows_inserts_a_newline(
    tmp_path, monkeypatch
) -> None:
    """M-11: (a) accepting with the caret inside ``[[Lisb]]`` leaves no stray
    ``]]``; (b) a mouse click on a row accepts it; (c) Enter on the disabled
    "No note titled" row closes the picker and the editor inserts its newline."""
    from tldw_chatbook.Notes.note_import_plan_models import stored_note_link

    app, profile = _app(tmp_path)
    _record_open_url(monkeypatch, app)
    lisbon_link = stored_note_link(LISBON, profile.lisbon_id)
    index_link = stored_note_link(INDEX, profile.index_id)
    async with app.run_test(size=SIZE) as pilot:
        screen = await _library(app, pilot)
        body = await _body_focused(screen, pilot)
        await _type(pilot, "[[Lisb]]")
        await pilot.press("left", "left")
        await _until(pilot, lambda: body.cursor_location == (0, 6), "the caret inside the closer")
        await _until_painted(app, screen, pilot, "#library-note-link-picker", LISBON)
        await pilot.press("enter")
        await _until(pilot, lambda: body.text == lisbon_link, f"the body to read {lisbon_link!r}")
        assert body.text.count("]]") == 1

        await pilot.press("end")
        await _type(pilot, " [[Ind")
        await _until_painted(app, screen, pilot, "#library-note-link-picker", INDEX)
        picker_region = screen.query_one("#library-note-link-picker").region
        offset = _painted_offset(app, picker_region, INDEX)
        assert offset is not None
        await pilot.click(offset=offset)
        expected = f"{lisbon_link} {index_link}"
        await _until(pilot, lambda: body.text == expected, f"the click to insert {index_link!r}")
        assert not _picker_open(screen)

        await _type(pilot, " [[zzzz")
        await _until_painted(app, screen, pilot, "#library-note-link-picker", "No note titled")
        await pilot.press("enter")
        await _until(pilot, lambda: body.text == f"{expected} [[zzzz\n", "Enter to insert the newline")
        assert not _picker_open(screen)
