"""Critique-10 Reader group: the typing footer, Find, and a rendered analysis.

Covers tasks 32346 (the canvas verbs survive a focused Input), 32348 (Find
opens from the keyboard and refuses the tabs it cannot search) and 32365
(a Markdown analysis renders).
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input, Static

from Tests.UI.test_library_crit9_shell import (
    _library_host,
)
from Tests.UI.test_library_media_render_fixes import (
    _open_first_reader_row,
    _painted,
    _switch_to_analysis,
)
from Tests.UI.test_library_media_side_by_side import (
    _build_media_test_app,
    _open_media_list,
    _two_media_items,
)
from tldw_chatbook.Widgets.Library.library_media_content import (
    LibraryMediaContentBody,
)
from tldw_chatbook.Widgets.Library.library_media_viewer import (
    ANALYSIS_RENDERED_BLOCKED_BY_SEARCH,
    LibraryMediaViewer,
)
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _submit_content_search_query,
    _seed_conversations,
    _two_conversations,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)


def _media_host() -> LibraryProductionCSSHarness:
    """The Library with two local media items, list and Reader both usable."""
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    return LibraryProductionCSSHarness(app)


# --------------------------------------------------------------------------
# task-32346: the footer under a focused Input
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_focused_search_box_keeps_the_canvas_verbs_behind_one_named_key():
    host = _library_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-search").press()
        query = await _wait_for_selector(screen, pilot, "#library-rag-query-input")
        query.focus()
        await pilot.pause()
        chips = screen._library_footer_shortcuts_for_current_state()
        labels = [label for _key, label in chips]
        assert labels[0] == "typing in field", chips
        assert ("esc", "leave field") in chips, chips
        joined = " ".join(labels)
        assert "after esc: u use Library context in Console · o open evidence" in joined, chips
        # AC#2: "F6 next pane" is LAST, so the responsive footer drops it
        # before any canvas verb.
        assert chips[-1] == ("F6", "next pane"), chips
        # AC#1's honesty half: no single printable key is advertised as live.
        assert not [key for key, _label in chips if len(key) == 1 and key.isprintable()], chips


@pytest.mark.asyncio
async def test_the_media_filter_box_keeps_the_list_verbs_the_same_way():
    host = _media_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        box = await _wait_for_selector(screen, pilot, "#library-media-filter")
        box.focus()
        # The list settles asynchronously and re-seats focus; wait for the
        # caret to actually land in the filter box rather than assume it.
        await _wait_for_condition(
            pilot,
            lambda: screen.focused is screen.query_one("#library-media-filter"),
            message="The media filter box never took focus.",
        )
        chips = screen._library_footer_shortcuts_for_current_state()
        joined = " ".join(label for _key, label in chips)
        assert "after esc: " in joined and "s select" in joined, chips
        assert chips[-1] == ("F6", "next pane"), chips


# --------------------------------------------------------------------------
# task-32348: Find
# --------------------------------------------------------------------------


def test_find_is_refused_on_the_tabs_that_have_no_search_bar():
    from tldw_chatbook.Library.library_media_viewer_state import (
        analysis_find_unavailable_reason,
    )

    for mode in ("info", "highlights"):
        assert analysis_find_unavailable_reason(
            mode=mode, analysis="anything", generating=False, editing=False
        ) == "This tab has no text to search · switch to Read or Analysis.", mode
    assert analysis_find_unavailable_reason(
        mode="read", analysis="", generating=False, editing=False
    ) == ""
    assert analysis_find_unavailable_reason(
        mode="analysis", analysis="", generating=False, editing=False
    ) == "No analysis to search yet."


async def _open_first_media_reader(host, pilot):
    """Open the first media item's Reader and return the settled screen."""
    screen = await _open_media_list(host, pilot)
    await _open_first_reader_row(screen, pilot)
    return screen


@pytest.mark.asyncio
async def test_ctrl_f_opens_the_reader_find_bar_and_the_footer_names_it():
    host = _media_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_first_media_reader(host, pilot)
        assert ("ctrl+f", "find") in screen._library_footer_shortcuts_for_current_state()
        # The critique's "Find is not in the Tab order" reading is wrong: the
        # button is an ordinary focusable Button in the Reader's focus chain.
        # What was missing was a KEY, which is what this test pins.
        find_button = screen.query_one("#library-media-reader-find", Button)
        assert find_button.focusable, find_button
        assert find_button in screen.focus_chain, [
            w.id for w in screen.focus_chain
        ]
        await pilot.press("ctrl+f")
        await pilot.pause()
        assert screen.query("#library-media-content-search-controls")
        assert screen._media_state.find_open is True


@pytest.mark.asyncio
async def test_t_never_arms_the_trash_while_find_is_open():
    host = _media_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_first_media_reader(host, pilot)
        await pilot.press("ctrl+f")
        await pilot.pause()
        assert screen.check_action("library_media_move_to_trash", ()) is False
        assert ("t", "trash") not in screen._library_footer_shortcuts_for_current_state()
        await pilot.press("escape")
        await pilot.pause()
        assert screen.check_action("library_media_move_to_trash", ()) is True


@pytest.mark.asyncio
async def test_find_is_refused_on_the_info_tab():
    """task-32348, B D4: the Info tab has no bar to mount, so Find refuses it.

    Review finding 2: this deliberately says nothing about "t". With Find
    CLOSED on Info, ``library_media_move_to_trash`` is still live (its gate
    reads view/substate/pending, never the reader mode), so "t" does arm the
    confirmation there -- the AC#2 guarantee is about a PENDING Find gesture
    and is pinned by ``test_t_never_arms_the_trash_while_find_is_open``.
    """
    host = _media_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_first_media_reader(host, pilot)
        screen.query_one("#library-media-reader-select-info", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-reader-mode-info")
        await pilot.pause()
        assert screen.check_action("library_media_reader_find", ()) is False
        assert ("ctrl+f", "find") not in (
            screen._library_footer_shortcuts_for_current_state()
        )
        await pilot.press("ctrl+f")
        await pilot.pause()
        assert screen._media_state.find_open is False
        assert not screen.query("#library-media-content-search-controls")


# --------------------------------------------------------------------------
# task-32365: a Markdown analysis renders
# --------------------------------------------------------------------------


_MARKDOWN_ANALYSIS = "## Key contributions\n\nA first point, and a second."


def _analysis_host(analysis: str) -> LibraryProductionCSSHarness:
    """Media whose stored analysis is ``analysis``.

    Local media detail never carries ``analysis_content`` at the top level;
    the viewer reads the newest ``versions`` entry.
    """
    app = _build_media_test_app()
    items = _two_media_items()
    for item in items:
        item["versions"] = [{"version_number": 1, "analysis_content": analysis}]
    _seed_conversations(app, _two_conversations(), media=items)
    return LibraryProductionCSSHarness(app)


@pytest.mark.asyncio
async def test_a_stored_analysis_renders_its_markdown_with_a_raw_toggle():
    host = _analysis_host(_MARKDOWN_ANALYSIS)
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_first_media_reader(host, pilot)
        await _switch_to_analysis(screen, pilot)
        body = await _wait_for_selector(screen, pilot, "#library-media-viewer-content")
        painted = _painted(host, body.region)
        assert "Key contributions" in painted, painted
        assert "## Key" not in painted, painted
        toggle = screen.query_one("#library-media-analysis-content-mode-raw", Button)
        toggle.press()
        await pilot.pause()
        await pilot.pause()
        assert "## Key contributions" in _painted(
            host, screen.query_one("#library-media-viewer-content").region
        )


@pytest.mark.asyncio
async def test_a_plain_text_analysis_is_offered_no_toggle():
    """Nothing to render, so no affordance for it (the Read tab's rule)."""
    host = _analysis_host("A flat paragraph of prose, with no markup at all.")
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_first_media_reader(host, pilot)
        await _switch_to_analysis(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-media-viewer-content")
        assert not screen.query("#library-media-analysis-content-mode-raw")


@pytest.mark.asyncio
async def test_an_active_find_query_drops_the_rendered_analysis_to_the_view_that_marks():
    """Review finding 1: rendered mode mounts only the Markdown widget, whose
    ``sync_search`` no-ops (there is no raw view to restyle) and whose
    scroll-to-match is a source-line index applied to a rendered scroller. So
    a Find over a rendered analysis counted matches and marked none. An active
    query now shows the view that can mark, and the toggle says so."""
    host = _analysis_host(_MARKDOWN_ANALYSIS)
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_first_media_reader(host, pilot)
        await _switch_to_analysis(screen, pilot)
        body = await _wait_for_selector(screen, pilot, "#library-media-viewer-content")
        assert body.active_mode == "rendered", body.active_mode

        await _submit_content_search_query(screen, pilot, "point")
        body = await _wait_for_selector(screen, pilot, "#library-media-viewer-content")
        await _wait_for_condition(
            pilot,
            lambda: screen.query_one(
                "#library-media-viewer-content", LibraryMediaContentBody
            ).active_mode == "raw",
            message="An active analysis query never dropped to the Raw view.",
        )
        raw_button = screen.query_one("#library-media-analysis-content-mode-raw", Button)
        assert "Raw (selected)" in str(raw_button.label), raw_button.label
        # ...and Rendered is refused with its reason, never a pressable
        # control that repaints the same body.
        rendered = screen.query_one(
            "#library-media-analysis-content-mode-rendered", Button
        )
        assert rendered.disabled is True
        assert rendered.tooltip == ANALYSIS_RENDERED_BLOCKED_BY_SEARCH
        reason = screen.query_one("#library-media-analysis-content-mode-reason", Static)
        assert str(reason.renderable) == ANALYSIS_RENDERED_BLOCKED_BY_SEARCH

        # Clearing the query hands the rendered view back.
        await _submit_content_search_query(screen, pilot, "")
        await _wait_for_condition(
            pilot,
            lambda: screen.query_one(
                "#library-media-viewer-content", LibraryMediaContentBody
            ).active_mode == "rendered",
            message="Clearing the query never restored the Rendered view.",
        )


@pytest.mark.asyncio
async def test_below_64_columns_a_focused_input_leaves_the_single_chip_alone():
    """Coordinator ruling (Task 6 measurement): below 64 columns the footer
    paints exactly ONE ~24-char context chip, and there Escape returns to
    Library rather than leaving the field. Emitting the wide
    ``typing in field · after esc: …`` form as well makes AppFooterStatus
    elide the whole context to "…", so the Input-focus transform stands down
    on that stage and the narrow block's chip is the entire context.

    The expected set is READ from the blurred state rather than written out,
    so this pins "focusing an Input changes nothing here" without duplicating
    the literal Task 6 owns.
    """
    host = _media_host()
    async with host.run_test(size=(60, 24)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _wait_for_condition(
            pilot,
            lambda: screen._library_narrow_stage_return_active(),
            message="The Library pane never closed at 60 columns.",
        )
        blurred = screen._library_footer_shortcuts_for_current_state()
        # AppFooterStatus paints a PREFIX of the registered set, and Task 6's
        # block puts its Escape chip first so that prefix is that chip.
        assert blurred[0][0] == "esc", blurred

        box = await _wait_for_selector(screen, pilot, "#library-media-filter")
        box.focus()
        await _wait_for_condition(
            pilot,
            lambda: screen.focused is screen.query_one("#library-media-filter"),
            message="The media filter box never took focus.",
        )
        assert screen._library_footer_shortcuts_for_current_state() == blurred


@pytest.mark.asyncio
async def test_below_64_columns_the_viewer_names_escape_not_a_bare_status_word():
    """Task 6's re-review, applied to the surface its own gate stands down on.

    ``_library_narrow_stage_return_active`` yields whenever an earlier Escape
    action owns the key, and its docstring names the 60x24 Media viewer as
    exactly that case -- so the stand-down above does NOT cover the viewer,
    and the wide form ran there. At 60 columns only the first chip is
    painted, so it led with the bare status word "typing in field": true, and
    useless, where the blurred footer had at least named a key.

    The single slot now carries the Escape chip, which is the one thing that
    works from inside the field -- the same "recovery outranks navigation
    below 64 columns" order Task 6's block uses, so the two read as one
    grammar at that width.
    """
    host = _media_host()
    async with host.run_test(size=(60, 24)) as pilot:
        screen = await _open_first_media_reader(host, pilot)
        assert not screen._library_narrow_stage_return_active()
        await pilot.press("ctrl+f")
        await _wait_for_condition(
            pilot,
            lambda: isinstance(screen.focused, Input),
            message="ctrl+f never focused the Find input.",
        )
        chips = screen._library_footer_shortcuts_for_current_state()
        assert chips[0][0] == "esc", chips
        assert chips[1] == ("", "typing in field"), chips


# --------------------------------------------------------------------------
# Qodo review round
# --------------------------------------------------------------------------


def test_find_follows_the_analysis_the_reader_actually_shows():
    """Qodo 1: ``detail_analysis_text`` read only the newest version, while
    ``build_library_media_viewer_state`` prefers a top-level
    ``analysis_content`` -- so on a detail carrying one with no versions the
    Analysis tab displayed text while the Find gate reported none. Both
    callers want "the analysis the Reader shows", and one of them says so in
    its own docstring, so the precedence is fixed at the shared seam."""
    from tldw_chatbook.Library.library_media_viewer_state import (
        build_library_media_viewer_state,
        detail_analysis_text,
    )

    top_level_only = {
        "media_id": "1",
        "title": "T",
        "type": "article",
        "analysis_content": "Top-level analysis",
    }
    assert detail_analysis_text(top_level_only) == "Top-level analysis"
    assert (
        build_library_media_viewer_state(top_level_only).analysis
        == detail_analysis_text(top_level_only)
    )

    # Top level still wins over versions, and versions still answer alone.
    both = dict(top_level_only, versions=[{"version_number": 1, "analysis_content": "V"}])
    assert detail_analysis_text(both) == "Top-level analysis"
    versions_only = {
        "media_id": "1",
        "title": "T",
        "type": "article",
        "versions": [{"version_number": 1, "analysis_content": "V"}],
    }
    assert detail_analysis_text(versions_only) == "V"


def test_an_external_detail_is_searchable_whatever_mode_it_inherited():
    """Qodo 4: ``_compose_active_body`` composes the Read body for EVERY
    external detail (``external_detail or reader_mode == "read"``), so it
    always has a bar to mount. The mode-only refusal treated one opened
    after Info/Highlights as unsearchable -- a regression this round
    introduced, since the gate used to return "" for every non-analysis
    mode."""
    from tldw_chatbook.Library.library_media_viewer_state import (
        analysis_find_unavailable_reason,
    )

    for mode in ("info", "highlights", "read", "analysis"):
        assert analysis_find_unavailable_reason(
            mode=mode,
            analysis="",
            generating=False,
            editing=False,
            external=True,
        ) == "", mode
    # Local details keep the refusal.
    assert analysis_find_unavailable_reason(
        mode="info", analysis="x", generating=False, editing=False
    ) == "This tab has no text to search · switch to Read or Analysis."


@pytest.mark.asyncio
async def test_pressing_raw_during_a_search_does_not_outlive_the_search():
    """Qodo 3: the toggle strip's press handler wrote analysis_content_mode
    unconditionally, so pressing the already-selected Raw during a search
    made the mode stick and the inline promise ("clear the search to read it
    rendered") false. Both controls are inert while the query forces Raw."""
    host = _analysis_host(_MARKDOWN_ANALYSIS)
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_first_media_reader(host, pilot)
        await _switch_to_analysis(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-media-viewer-content")
        await _submit_content_search_query(screen, pilot, "point")
        await _wait_for_condition(
            pilot,
            lambda: screen.query_one(
                "#library-media-viewer-content", LibraryMediaContentBody
            ).active_mode == "raw",
            message="The query never forced the Raw view.",
        )
        raw = screen.query_one("#library-media-analysis-content-mode-raw", Button)
        assert raw.disabled is True, "Raw must not be pressable while it is forced"
        viewer = screen.query_one(LibraryMediaViewer)
        assert viewer.analysis_content_mode == "rendered"

        await _submit_content_search_query(screen, pilot, "")
        await _wait_for_condition(
            pilot,
            lambda: screen.query_one(
                "#library-media-viewer-content", LibraryMediaContentBody
            ).active_mode == "rendered",
            message="Clearing the query never restored the Rendered view.",
        )
