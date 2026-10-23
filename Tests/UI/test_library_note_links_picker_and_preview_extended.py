"""TASK-34000.11, the slower arms (NOT in ``scripts/ui_pr_gate_census.txt``).

Enter on the focused Preview follows the note's one link or lists several;
the picker at 80x24 stays inside the pane; a 30-title corpus caps the rows
at eight and Down wraps; a title carrying ``[b]`` paints literally in the
picker and lands in the inserted link.
"""

from __future__ import annotations

import pytest
from textual.widgets import Markdown, OptionList, TextArea

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
        options_now = screen.query_one("#library-note-link-picker-options", OptionList)
        await _until(pilot, lambda: options_now.option_count >= 2, "the rows")
        cut_height = picker.region.height
        assert cut_height < options_now.option_count + 2, "the pin needs a cut box"
        # Fix round 2 (R2-5): once the rows had been cut to fit beside the
        # caret, the same rows shown again kept the cut height. When the
        # terminal grows, the box grows back to its rows -- and still never
        # covers the caret row.
        await pilot.resize_terminal(80, 45)
        await _until(
            pilot,
            lambda: _picker_open(screen)
            and picker.region.height == options_now.option_count + 2,
            f"the picker to grow back from {cut_height} rows after the resize",
            timeout=10.0,
        )
        caret_row = body.cursor_screen_offset.y
        assert not (picker.region.y <= caret_row < picker.region.bottom), (
            f"the picker {picker.region} covers the caret row {caret_row} after the resize"
        )
        assert picker.region.bottom <= app.size.height
        await pilot.resize_terminal(*COMPACT)
        await pilot.pause(0.3)
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


REFUSED = "Link not opened (unsupported scheme): "
#: (label, href as written, the full refusal notice -- Textual unquotes the
#: href before it reaches the handler, and the notice drops C0/C1).
HOSTILE_RENDERED = (
    ("QUPPERQ", "HTTP://up.example/", f"{REFUSED}HTTP://up.example/"),
    ("QPROTOQ", "//evil.example/", f"{REFUSED}//evil.example/"),
    ("QFTPQ", "ftp://a/[b]x[/]", f"{REFUSED}ftp://a/[b]x[/]"),
    ("QCTRLQ", "https://a.example/%1B]0;x%07", f"{REFUSED}https://a.example/]0;x"),
    ("QMARKQ", "https://x.example/[b]bold[/]", "Opened in browser: https://x.example/[b]bold[/]"),
)
HOSTILE_POSTED = (
    ("javascript:alert(1)", f"{REFUSED}javascript:alert(1)"),
    ("file:///etc/passwd", f"{REFUSED}file:///etc/passwd"),
    ("data:text/html,hi", f"{REFUSED}data:text/html,hi"),
    ("vbscript:msgbox(1)", f"{REFUSED}vbscript:msgbox(1)"),
    ("https://a.example/\x1b]0;x\x07", f"{REFUSED}https://a.example/]0;x"),
    ("https://a.example/with space", f"{REFUSED}https://a.example/with space"),
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
    body_text = "\n\n".join(f"[{label}]({href})" for label, href, _notice in HOSTILE_RENDERED)
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
        for label, href, expected_notice in HOSTILE_RENDERED:
            before = len(toasts)
            offset = _painted_offset(app, region, label)
            assert offset is not None, f"{label} not painted"
            await pilot.click(offset=offset)
            await _until(pilot, lambda: len(toasts) > before, f"a notice for {href!r}")
            notice = toasts[before]
            assert notice == expected_notice, (href, notice)
            if href == clean_http:
                await _until(pilot, lambda: browser == [clean_http], "the one browser call")
            else:
                assert browser == [], (href, browser)
            assert not _has_control_chars(notice), repr(notice)
        md = screen.query_one("#library-note-preview-body", Markdown)
        for href, expected_notice in HOSTILE_POSTED:
            before = len(toasts)
            md.post_message(Markdown.LinkClicked(md, href))
            await _until(pilot, lambda: len(toasts) > before, f"a notice for {href!r}")
            notice = toasts[before]
            assert notice == expected_notice, (href, notice)
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


# ---------------------------------------------------------------------------
# Fix round 2 pins
# ---------------------------------------------------------------------------


async def test_a_down_inside_a_burst_never_inserts_a_row_the_user_did_not_see(
    tmp_path, monkeypatch
) -> None:
    """R2-1 (the insert-mode I-1): the picker is open on ``[[``; the burst
    ``Lisbon``+Down+Enter used to insert "Lisbon alpha" (row 0 of the FRESH
    rows) because the Down moved the highlight on the STALE rows and the
    held Enter then accepted row 0. Ruling: a held accept never accepts a row
    the user did not see highlighted in the current set -- the hold is
    CANCELLED, the picker stays open on the fresh rows, nothing is inserted;
    a paced Down+Enter then inserts "Lisbon beta". A row click while the
    load is in flight likewise inserts nothing."""
    import asyncio

    from tldw_chatbook.Notes.note_import_plan_models import stored_note_link

    app, profile = _app(tmp_path)
    _record_open_url(monkeypatch, app)
    ids = [profile.db.add_note(f"Burst {i}", "") for i in range(2)]
    alpha = profile.db.add_note("Lisbon alpha", "a")
    beta = profile.db.add_note("Lisbon beta", "b")
    assert alpha and beta
    # A slowed candidate service keeps every load in flight for 0.4 s, so
    # the keys below land over STALE rows deterministically.
    real = profile.scope_service.list_note_link_candidates

    async def _slow(*args, **kwargs):
        await asyncio.sleep(0.4)
        return await real(*args, **kwargs)

    monkeypatch.setattr(profile.scope_service, "list_note_link_candidates", _slow)
    async with app.run_test(size=SIZE) as pilot:
        screen = await _library(app, pilot)
        body = await _open_note(screen, pilot, "Burst 0", ids[0])
        await pilot.click("#library-note-body")
        await _until(pilot, lambda: screen.focused is body, "focus")
        _burst(app, "[[")
        await _until(pilot, lambda: _picker_open(screen), "the picker on [[")
        await pilot.pause(0.3)
        options = screen.query_one("#library-note-link-picker-options", OptionList)
        _burst(app, "Lisbon", "down", "enter")
        await _until(pilot, lambda: options.option_count == 3, "the three Lisbon rows")
        await pilot.pause(0.8)
        assert body.text == "[[Lisbon", body.text
        assert _picker_open(screen)
        assert options.option_count == 3
        await pilot.press("down")
        await pilot.press("enter")
        expected = stored_note_link("Lisbon beta", beta)
        await _until(pilot, lambda: body.text == expected, f"the paced Down+Enter to insert {expected!r}")

        # The click arm: a row chosen while the narrower load is in flight.
        body = await _open_note(screen, pilot, "Burst 1", ids[1])
        await pilot.click("#library-note-body")
        await _until(pilot, lambda: screen.focused is body, "focus")
        _burst(app, "[[")
        await _until(pilot, lambda: _picker_open(screen), "the picker on [[")
        await pilot.pause(0.3)
        options = screen.query_one("#library-note-link-picker-options", OptionList)
        picker = _picker_or_none(screen)
        _burst(app, "Lisbon")
        await _until(pilot, lambda: body.text == "[[Lisbon", "the keys to land")
        assert picker.generation != screen._notes_controller._library_notes_link_picker_generation, (
            "the narrower load must still be in flight"
        )
        options.action_select()  # what a click on the (stale) highlighted row posts
        await _until(pilot, lambda: options.option_count == 3, "the three Lisbon rows")
        await pilot.pause(0.8)
        assert body.text == "[[Lisbon", body.text
        assert _picker_open(screen)


async def test_tab_moves_focus_when_only_a_hint_row_shows(tmp_path, monkeypatch) -> None:
    """R2-2 (regression from M-11): with "No note titled …" or "Keep typing…"
    showing, Tab must still move focus out of the body (and the picker goes
    with the focus); the Tab claim ignored ``has_rows`` while the body no
    longer intercepted Tab, so Tab did nothing."""
    app, profile = _app(tmp_path)
    _record_open_url(monkeypatch, app)
    async with app.run_test(size=SIZE) as pilot:
        screen = await _library(app, pilot)
        body = await _body_focused(screen, pilot)
        await _type(pilot, "[[zzzz")
        await _until_painted(app, screen, pilot, "#library-note-link-picker", "No note titled")
        await pilot.press("tab")
        await _until(pilot, lambda: screen.focused is not body, "Tab to leave the body")
        await _until(pilot, lambda: not _picker_open(screen), "the picker to close")
        assert body.text == "[[zzzz"
        await pilot.click("#library-note-body")
        await _until(pilot, lambda: screen.focused is body, "focus")
        await pilot.press("end")
        await _type(pilot, " [[q")
        await _until_painted(app, screen, pilot, "#library-note-link-picker", "Keep typing")
        await pilot.press("tab")
        await _until(pilot, lambda: screen.focused is not body, "Tab to leave the body")
        await _until(pilot, lambda: not _picker_open(screen), "the picker to close")
        assert body.text == "[[zzzz [[q"


async def test_the_picker_never_opens_on_a_line_without_two_brackets(tmp_path, monkeypatch) -> None:
    """R2-3: Enter landing after the ``[[Lisb`` sync started its load but
    before it answered used to let that load open the picker at (1, 0) --
    a line with no ``[[`` -- where it took Up/Down and swallowed the next
    Enter. The no-context sync now supersedes the in-flight load. The
    window is made deterministic by delaying the candidate service, so the
    load is always still in flight when Enter lands."""
    import asyncio

    app, profile = _app(tmp_path)
    _record_open_url(monkeypatch, app)
    note_id = profile.db.add_note("Window", "")
    real = profile.scope_service.list_note_link_candidates

    async def _slow(*args, **kwargs):
        await asyncio.sleep(0.4)
        return await real(*args, **kwargs)

    monkeypatch.setattr(profile.scope_service, "list_note_link_candidates", _slow)
    async with app.run_test(size=SIZE) as pilot:
        screen = await _library(app, pilot)
        body = await _open_note(screen, pilot, "Window", note_id)
        await pilot.click("#library-note-body")
        await _until(pilot, lambda: screen.focused is body, "focus")
        # One burst, so no load has answered when Enter lands (paced keys
        # let the slow ``[[`` load open the picker mid-word, which is the
        # held-accept path, not this one).
        _burst(app, "[[Lisb")
        await _until(pilot, lambda: body.text == "[[Lisb", "the text")
        assert not _picker_open(screen), "the slow load must still be in flight"
        _send_key(app, "enter")
        await _until(pilot, lambda: body.text == "[[Lisb\n", "the newline")
        await pilot.pause(1.2)
        assert body.cursor_location == (1, 0)
        assert not _picker_open(screen), "the picker opened on a bare line"
        assert body.link_picker_open is False
        await pilot.press("down")
        await pilot.pause(0.2)
        assert body.cursor_location == (1, 0)
        await pilot.press("enter")
        await _until(pilot, lambda: body.text == "[[Lisb\n\n", "the second Enter to land")


async def test_a_held_enter_with_no_rows_keeps_the_picker_open_on_the_hint(tmp_path, monkeypatch) -> None:
    """R2-4: ``[[Lis`` rows showing, then ``zz``+Enter as one burst: the held
    Enter's load answers with no rows. The picker stays OPEN on the
    "No note titled “Liszz”" hint so the user sees why nothing was inserted;
    no newline is inserted for that Enter; the next Enter is the editor's."""
    app, profile = _app(tmp_path)
    _record_open_url(monkeypatch, app)
    async with app.run_test(size=SIZE) as pilot:
        screen = await _library(app, pilot)
        body = await _body_focused(screen, pilot)
        await _type(pilot, "[[Lis")
        await _until_painted(app, screen, pilot, "#library-note-link-picker", LISBON)
        _burst(app, "zz", "enter")
        await _until_painted(app, screen, pilot, "#library-note-link-picker", "No note titled “Liszz”")
        await pilot.pause(0.8)
        assert body.text == "[[Liszz", body.text
        assert _picker_open(screen)
        await pilot.press("enter")
        await _until(pilot, lambda: body.text == "[[Liszz\n", "the next Enter to be the editor's")
        await _until(pilot, lambda: not _picker_open(screen), "the picker to close")


# ---------------------------------------------------------------------------
# Fix round 3 pins
# ---------------------------------------------------------------------------

LONG_BODY = "\n".join(f"Line {i:03d} of the long body" for i in range(120))


def _picker_geometry(app, screen):
    picker = _picker_or_none(screen)
    body = screen.query_one("#library-note-body", TextArea)
    caret = body.cursor_screen_offset
    region = picker.region
    return {
        "caret": (caret.x, caret.y),
        "picker": tuple(region),
        "covers": region.y <= caret.y < region.bottom,
        "attached": region.y == caret.y + 1 or region.bottom == caret.y,
        "inside_screen": region.x >= 0
        and region.y >= 0
        and region.right <= app.size.width
        and region.bottom <= app.size.height,
        "caret_visible": body.content_region.y <= caret.y < body.content_region.bottom,
        "open": bool(picker.display),
        "painted": _painted(app, screen, "#library-note-link-picker"),
    }


async def test_a_follow_picker_row_click_opens_the_link_after_the_body_was_touched(
    tmp_path, monkeypatch
) -> None:
    """R3-1 (regression from R2-1): the stale-row guard ran before the
    follow-mode branch, and the follow picker shows with generation 0 while
    the controller's generation is non-zero after any body interaction --
    so a mouse click on a follow row did nothing. Insert mode's guard is
    untouched; follow mode opens the clicked link."""
    app, profile = _app(tmp_path)
    opened = _record_open_url(monkeypatch, app)
    async with app.run_test(size=SIZE) as pilot:
        screen = await _library(app, pilot)
        body = await _open_note(screen, pilot, TWO, profile.two_id)
        # The user clicked into the body once (a caret move, no edit).
        await pilot.click("#library-note-body")
        await _until(pilot, lambda: screen.focused is body, "focus")
        await pilot.press("end")
        await pilot.pause(0.3)
        assert screen._notes_controller._library_notes_link_picker_generation > 0
        await _preview(screen, pilot, INDEX, app)
        region = screen.query_one("#library-note-preview-region")
        await _until(pilot, lambda: screen.focused is region, "Preview focus")
        await pilot.press("enter")
        await _until_painted(app, screen, pilot, "#library-note-link-picker", INDEX)
        picker = _picker_or_none(screen)
        offset = _painted_offset(app, picker.region, INDEX[:10])
        assert offset is not None
        await pilot.click(offset=offset)
        await _armed_editor(screen, pilot, profile.index_id)
        assert _title_value(screen) == INDEX
        assert not _picker_open(screen)
        assert opened == []


async def test_a_shrink_that_scrolls_the_caret_off_screen_never_leaves_an_invisible_picker(
    tmp_path, monkeypatch
) -> None:
    """R3-2: Textual's TextArea does not scroll the caret into view on a
    resize. Caret on the last row, 160x45 -> 80x24: the re-anchor used to
    follow the off-screen caret, the picker painted nothing but stayed open,
    and Enter inserted a link invisibly. Ruling: scroll the caret into view
    first; if it still is not visible, close the picker. Never accept from a
    picker the user cannot see."""
    from tldw_chatbook.Notes.note_import_plan_models import stored_note_link

    app, profile = _app(tmp_path)
    _record_open_url(monkeypatch, app)
    note_id = profile.db.add_note("Long body", LONG_BODY)
    async with app.run_test(size=SIZE) as pilot:
        screen = await _library(app, pilot)
        body = await _open_note(screen, pilot, "Long body", note_id)
        await pilot.click("#library-note-body")
        await _until(pilot, lambda: screen.focused is body, "focus")
        await pilot.press("ctrl+end")
        await pilot.pause(0.3)
        await _type(pilot, "[[Lis")
        await _until_painted(app, screen, pilot, "#library-note-link-picker", LISBON)
        await pilot.resize_terminal(80, 24)
        await _until(
            pilot,
            lambda: not _picker_open(screen)
            or (
                _picker_geometry(app, screen)["caret_visible"]
                and _picker_geometry(app, screen)["inside_screen"]
                and _picker_geometry(app, screen)["painted"]
            ),
            "the picker to close or to be visible beside a visible caret",
            timeout=10.0,
        )
        geometry = _picker_geometry(app, screen)
        assert geometry["caret_visible"], geometry
        if geometry["open"]:
            assert geometry["inside_screen"] and geometry["painted"] and not geometry["covers"], geometry
            await pilot.press("enter")
            # Row 0 for "Lis" is "Reading list" (rank ties, title order) --
            # the highlighted row the user SEES is what Enter inserts.
            expected_tail = stored_note_link("Reading list", profile.other_id)
            await _until(
                pilot,
                lambda: body.document.get_line(body.document.line_count - 1).endswith(expected_tail),
                "the visible picker to insert its highlighted row",
            )
        else:
            # Closed: Enter is the editor's own newline, nothing is inserted.
            await pilot.press("enter")
            await _until(pilot, lambda: body.document.line_count == 121, "the newline")
            assert "note://" not in body.text
        await pilot.resize_terminal(*SIZE)
        await pilot.pause(0.3)


async def test_a_body_scroll_re_anchors_or_closes_the_picker_and_idle_makes_no_passes(
    tmp_path, monkeypatch
) -> None:
    """R3-3: at 80x24 with the caret mid-body, a wheel scroll of +3 left the
    picker where it was -- covering the caret row. The picker follows the
    caret on a body scroll; when the scroll takes the caret off the body it
    closes. Idle costs nothing: 0 anchor passes while nothing changes."""
    from tldw_chatbook.Widgets.Library.library_note_link_picker import LibraryNoteLinkPicker

    app, profile = _app(tmp_path)
    _record_open_url(monkeypatch, app)
    note_id = profile.db.add_note("Long body", LONG_BODY)
    passes = {"n": 0}
    real_pass = LibraryNoteLinkPicker._anchor_pass

    def _counted(self):
        passes["n"] += 1
        return real_pass(self)

    monkeypatch.setattr(LibraryNoteLinkPicker, "_anchor_pass", _counted)
    async with app.run_test(size=COMPACT) as pilot:
        screen = await _library(app, pilot)
        body = await _open_note(screen, pilot, "Long body", note_id)
        await pilot.click("#library-note-body")
        await _until(pilot, lambda: screen.focused is body, "focus")
        body.move_cursor((60, 0), center=True)
        await pilot.pause(0.3)
        await _type(pilot, "[[Lis")
        await _until_painted(app, screen, pilot, "#library-note-link-picker", LISBON)
        await pilot.pause(0.4)
        before = _picker_geometry(app, screen)
        assert before["attached"] and not before["covers"], before
        # Idle: nothing changes, no anchor pass.
        idle_start = passes["n"]
        await pilot.pause(1.0)
        assert passes["n"] == idle_start, f"{passes['n'] - idle_start} anchor passes while idle"
        # A scroll that keeps the caret visible: the picker follows it. (+1,
        # not the reviewer's +3: with the caret two rows under the body's
        # top border, +3 puts it ON the border row -- not visible -- and
        # closing is then the right answer, pinned below.)
        body.scroll_relative(y=1, animate=False)
        await _until(
            pilot,
            lambda: _picker_geometry(app, screen)["attached"]
            and not _picker_geometry(app, screen)["covers"]
            and _picker_geometry(app, screen)["caret"] != before["caret"],
            f"the picker to follow the caret after a +1 scroll (was {before})",
        )
        after = _picker_geometry(app, screen)
        assert after["caret_visible"] and after["open"], after
        # A scroll that takes the caret off the body: the picker closes.
        body.scroll_relative(y=12, animate=False)
        await _until(pilot, lambda: not _picker_open(screen), "the picker to close once the caret is off the body")
        assert body.link_picker_open is False
        assert "[[Lis" in body.text and "note://" not in body.text


async def test_the_follow_picker_sits_at_the_preview_top_left_after_a_resize(
    tmp_path, monkeypatch
) -> None:
    """R3-4: follow mode passes no cell provider, so a re-anchor kept the
    160x45 cell -- after 160x45 -> 80x24 the follow picker sat 40 cells right
    of the Preview's top-left. It is recomputed from the Preview region: the
    box sits one cell in and one row down from the region's top-left (the
    round-1 placement), at every size."""
    app, profile = _app(tmp_path)
    _record_open_url(monkeypatch, app)
    async with app.run_test(size=SIZE) as pilot:
        screen = await _library(app, pilot)
        await _open_note(screen, pilot, TWO, profile.two_id)
        await _preview(screen, pilot, INDEX, app)
        region = screen.query_one("#library-note-preview-region")
        await _until(pilot, lambda: screen.focused is region, "Preview focus")
        await pilot.press("enter")
        await _until_painted(app, screen, pilot, "#library-note-link-picker", INDEX)
        picker = _picker_or_none(screen)
        for size in ((80, 24), SIZE, (80, 24)):
            await pilot.resize_terminal(*size)
            await _until(
                pilot,
                lambda: picker.region.x == region.region.x + 1
                and picker.region.y == region.region.y + 1,
                f"the follow picker at the Preview's top-left after {size} "
                f"(picker {picker.region}, preview {region.region})",
                timeout=10.0,
            )
            assert INDEX in _painted(app, screen, "#library-note-link-picker")
        await pilot.resize_terminal(*SIZE)
