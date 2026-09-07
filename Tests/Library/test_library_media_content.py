"""Mounted behavior tests for the Library media-content widgets."""

import asyncio

import pytest
from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer
from textual.widgets import Button, Input, Markdown, Static

from tldw_chatbook.Library.library_media_viewer_state import find_content_matches
from tldw_chatbook.Widgets.Library.library_media_content import (
    LibraryMediaContentBody,
    LibraryMediaContentSearchControls,
    build_raw_content_match_lines,
)
from tldw_chatbook.Widgets.Library.library_media_raw_view import (
    VirtualizedRawContent,
)


class SearchControlsHarness(App[None]):
    """Mount controls with an active Markdown query."""

    def compose(self) -> ComposeResult:
        yield LibraryMediaContentSearchControls(
            is_markdown=True,
            query="budget",
            matches=(2, 8),
            match_index=0,
            id="controls",
        )


class BlankSearchControlsHarness(App[None]):
    """Mount controls with no active query."""

    def compose(self) -> ComposeResult:
        yield LibraryMediaContentSearchControls(
            is_markdown=True,
            query="",
            matches=(),
            match_index=0,
            id="controls",
        )


class BodyHarness(App[None]):
    """Mount one supplied body instance without rebuilding it."""

    def __init__(self, body: LibraryMediaContentBody) -> None:
        super().__init__()
        self._body = body

    def compose(self) -> ComposeResult:
        yield self._body


class DelayedRenderedBody(LibraryMediaContentBody):
    """Hold the first Rendered mount until the test selects a winner.

    FINAL WHOLE-BRANCH REVIEW FINDING 4: ``sync_mode`` used to call
    ``_ensure_mode_mounted`` twice back-to-back with the same effective
    argument -- removed as dead duplicate code. This checkpoint used to fire
    on the SECOND ever ``_ensure_mode_mounted("raw")`` call because, under
    the old double-call code, ``sync_mode("rendered")``'s own SECOND call
    re-read ``self._desired_mode`` (already overwritten to ``"raw"`` by the
    concurrently-started ``sync_mode("raw")`` before it reached the lock)
    and so consumed the first "raw" count itself -- an accidental
    side-channel, not a deliberate contract. With the duplicate removed,
    ``_ensure_mode_mounted("raw")`` is called exactly once per
    ``sync_mode("raw")`` call, so this fires on the FIRST (and only) one.
    """

    def __init__(
        self,
        *args: object,
        mount_started: asyncio.Event,
        release: asyncio.Event,
        raw_started: asyncio.Event,
        release_raw: asyncio.Event,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._mount_started = mount_started
        self._release = release
        self._raw_started = raw_started
        self._release_raw = release_raw
        self._delay_rendered_mount = True
        self._raw_ensure_calls = 0

    async def _ensure_mode_mounted(self, mode: str) -> None:
        if mode == "rendered" and self._delay_rendered_mount:
            self._delay_rendered_mount = False
            self._mount_started.set()
            await self._release.wait()
        if mode == "raw":
            self._raw_ensure_calls += 1
            if self._raw_ensure_calls == 1:
                self._raw_started.set()
                await self._release_raw.wait()
        await super()._ensure_mode_mounted(mode)


@pytest.mark.asyncio
async def test_body_lazily_mounts_each_mode_once_and_reuses_widgets() -> None:
    """Catch mode changes that recompose an already-mounted content view."""
    body = LibraryMediaContentBody(
        content="# Heading\n\nbudget one\nbudget two",
        is_markdown=True,
        mode="raw",
        query="",
        match_index=0,
        id="library-media-viewer-content",
    )
    async with BodyHarness(body).run_test() as _pilot:
        raw = body.query_one(
            "#library-media-viewer-content-text", VirtualizedRawContent
        )
        assert not body.query("#library-media-viewer-content-markdown")

        await body.sync_mode("rendered")
        markdown = body.query_one("#library-media-viewer-content-markdown", Markdown)
        await body.sync_mode("raw")
        await body.sync_mode("rendered")

        assert body.query_one("#library-media-viewer-content-text") is raw
        assert body.query_one("#library-media-viewer-content-markdown") is markdown


@pytest.mark.asyncio
async def test_body_search_updates_lazily_mounted_raw_rich_highlights() -> None:
    """Catch Raw construction that loses search state set while Rendered is visible.

    task-22500: the virtualized Raw view has no whole-document renderable to
    inspect -- it restyles each row from ``_query``/``_match_index`` when
    Textual paints it -- so this now confirms the search state itself
    reached the newly-constructed widget, plus that both matching rows
    still paint their query text.
    """
    body = LibraryMediaContentBody(
        content="# Heading\n\nbudget one\nbudget two",
        is_markdown=True,
        mode="rendered",
        query="",
        match_index=0,
        id="library-media-viewer-content",
    )
    async with BodyHarness(body).run_test() as pilot:
        body.sync_search("budget", 1)
        await body.sync_mode("raw")
        await pilot.pause()

        raw = body.raw_view
        assert isinstance(raw, VirtualizedRawContent)
        assert raw._query == "budget"
        assert raw._match_index == 1
        assert "budget" in raw.render_line(2).text
        assert "budget" in raw.render_line(3).text


@pytest.mark.asyncio
async def test_body_rapid_mode_changes_leave_latest_mode_visible_once() -> None:
    """Catch an earlier delayed request overriding the latest mode visibility.

    FINDING 4: with the duplicate ``_ensure_mode_mounted`` call removed,
    ``sync_mode("rendered")`` (started first, then artificially delayed)
    applies ITS OWN "rendered" display flags before ``sync_mode("raw")``
    (started second, queued behind the same lock) gets a chance to run --
    so mid-race, Rendered is transiently the visible one even though "raw"
    is the LATER, and therefore intended-final, request. That transient
    state is exactly what this test now pins (it was invisible under the
    old double-call code, which incidentally re-read the already-changed
    ``_desired_mode`` and papered over it). What must still hold, and is
    asserted after both tasks finish, is the actual contract: the LATEST
    requested mode ends up the one visible, and neither mode is ever
    mounted twice.
    """
    mount_started = asyncio.Event()
    release = asyncio.Event()
    raw_started = asyncio.Event()
    release_raw = asyncio.Event()
    body = DelayedRenderedBody(
        content="# Heading",
        is_markdown=True,
        mode="raw",
        query="",
        match_index=0,
        id="library-media-viewer-content",
        mount_started=mount_started,
        release=release,
        raw_started=raw_started,
        release_raw=release_raw,
    )
    async with BodyHarness(body).run_test() as _pilot:
        rendered_task = asyncio.create_task(body.sync_mode("rendered"))
        await mount_started.wait()
        raw_task = asyncio.create_task(body.sync_mode("raw"))
        release.set()
        await raw_started.wait()

        raw = body.query_one(
            "#library-media-viewer-content-text", VirtualizedRawContent
        )
        assert body.query_one("#library-media-viewer-content-markdown", Markdown)
        # Mid-race: sync_mode("rendered") already ran to completion (it was
        # queued FIRST and released first), so its own "rendered" display
        # flags are the ones currently applied -- transiently stale, since
        # sync_mode("raw") (the later, intended-final request) is still
        # queued behind the lock at this exact point.
        assert not raw.display
        # task-22500: Rendered has no wrapper of its own -- the Markdown is
        # a direct child of this body (which scrolls for Rendered), so
        # visibility is the Markdown widget's own `.display`.
        assert body._markdown_widget.display

        release_raw.set()
        await asyncio.gather(rendered_task, raw_task)

        # After both finish, the LATEST request ("raw") wins: its own
        # sync_mode call ran after Rendered's and re-applied the correct
        # display flags, so the transient staleness above self-corrects.
        assert raw.display
        assert not body._markdown_widget.display
        assert len(body.query("#library-media-viewer-content-text")) == 1
        assert len(body.query("#library-media-viewer-content-markdown")) == 1


@pytest.mark.asyncio
async def test_body_exposes_the_active_scroller_per_mode() -> None:
    """``scroller`` must resolve the CURRENT mode's real scroller, not this
    container -- callers used to query this widget as a ``VerticalScroll``
    inside try/except, and silently lost scroll capture/restore the moment
    the type stopped matching (task-22500)."""
    body = LibraryMediaContentBody(
        content="# Heading\n\nbody text",
        is_markdown=True,
        mode="raw",
        query="",
        match_index=0,
        id="library-media-viewer-content",
    )
    async with BodyHarness(body).run_test() as pilot:
        assert isinstance(body.scroller, ScrollableContainer)
        assert isinstance(body.raw_view, VirtualizedRawContent)
        assert body.scroller is body.raw_view
        await body.sync_mode("rendered")
        await pilot.pause()
        assert body.scroller is not body.raw_view


@pytest.mark.asyncio
async def test_match_index_sync_preserves_navigation_identity_and_focus() -> None:
    """Catch a match-index update that recompiles the navigation controls."""
    app = SearchControlsHarness()
    async with app.run_test() as pilot:
        controls = app.query_one("#controls", LibraryMediaContentSearchControls)
        previous = app.query_one("#library-media-content-search-prev", Button)
        next_button = app.query_one("#library-media-content-search-next", Button)
        next_button.focus()

        controls.sync_match_index(matches=(2, 8), match_index=1)
        await pilot.pause()

        assert app.query_one("#library-media-content-search-prev") is previous
        assert app.query_one("#library-media-content-search-next") is next_button
        assert app.focused is next_button
        assert str(app.query_one("#library-media-content-search-status", Static).renderable) == (
            "Match 2 of 2 matches"
        )


@pytest.mark.asyncio
async def test_active_query_sync_preserves_search_and_navigation_identity() -> None:
    """Catch active-query updates that needlessly recompose mounted controls."""
    app = SearchControlsHarness()
    async with app.run_test() as pilot:
        controls = app.query_one("#controls", LibraryMediaContentSearchControls)
        search_input = app.query_one("#library-media-content-search", Input)
        previous = app.query_one("#library-media-content-search-prev", Button)
        next_button = app.query_one("#library-media-content-search-next", Button)

        controls.sync_query_state(
            is_markdown=True,
            query="cost",
            matches=(4,),
            match_index=0,
        )
        await pilot.pause()

        assert app.query_one("#library-media-content-search", Input) is search_input
        assert app.query_one("#library-media-content-search-prev", Button) is previous
        assert app.query_one("#library-media-content-search-next", Button) is next_button
        assert search_input.value == "cost"
        assert str(app.query_one("#library-media-content-search-status", Static).renderable) == (
            "Match 1 of 1 matches"
        )


@pytest.mark.asyncio
async def test_match_index_sync_displays_no_matches() -> None:
    """Catch a formatter that leaves a stale match count for an empty result."""
    app = SearchControlsHarness()
    async with app.run_test() as pilot:
        controls = app.query_one("#controls", LibraryMediaContentSearchControls)

        controls.sync_match_index(matches=(), match_index=0)
        await pilot.pause()

        assert str(app.query_one("#library-media-content-search-status", Static).renderable) == (
            "No matches"
        )


@pytest.mark.asyncio
async def test_match_index_sync_wraps_status_index() -> None:
    """Catch a formatter that exposes an out-of-range rather than wrapped index."""
    app = SearchControlsHarness()
    async with app.run_test() as pilot:
        controls = app.query_one("#controls", LibraryMediaContentSearchControls)

        controls.sync_match_index(matches=(2, 8), match_index=3)
        await pilot.pause()

        assert str(app.query_one("#library-media-content-search-status", Static).renderable) == (
            "Match 2 of 2 matches"
        )


@pytest.mark.asyncio
async def test_blank_query_sync_hides_search_controls() -> None:
    """Catch a blank-query transition that leaves stale controls visible.

    task-28002: the status/nav children are display-gated, never torn down
    -- recomposing here destroyed the focused ``Input`` and stranded screen
    focus on nothing (a live-verified total keyboard deadlock).
    """
    app = SearchControlsHarness()
    async with app.run_test() as pilot:
        controls = app.query_one("#controls", LibraryMediaContentSearchControls)

        controls.sync_query_state(
            is_markdown=True,
            query="",
            matches=(),
            match_index=0,
        )
        await pilot.pause()

        assert not app.query_one("#library-media-content-search-status").display
        assert not app.query_one("#library-media-content-search-nav").display


@pytest.mark.asyncio
async def test_active_query_sync_reveals_controls_and_markdown_placeholder() -> None:
    """Catch a blank-to-active transition that fails to reveal its controls."""
    app = BlankSearchControlsHarness()
    async with app.run_test() as pilot:
        controls = app.query_one("#controls", LibraryMediaContentSearchControls)
        blank_input = app.query_one("#library-media-content-search", Input)

        controls.sync_query_state(
            is_markdown=True,
            query="cost",
            matches=(4,),
            match_index=0,
        )
        await pilot.pause()

        active_input = app.query_one("#library-media-content-search", Input)
        assert active_input is blank_input
        assert active_input.placeholder == "Search content (raw text)…"
        assert app.query_one("#library-media-content-search-prev", Button).display
        assert app.query_one("#library-media-content-search-next", Button).display
        assert str(
            app.query_one("#library-media-content-search-status", Static).renderable
        ) == "Match 1 of 1 matches"


@pytest.mark.asyncio
async def test_activity_flip_preserves_focused_search_input() -> None:
    """task-28002 pinning test: the first submit must not eat keyboard focus.

    Live repro (2026-09-02): typing a first query and pressing Enter
    recomposed the controls, destroying the focused ``Input`` -- screen
    focus became None and EVERY key (Escape included) went dead until a
    mouse click. The input's identity and focus must survive both
    activity flips.
    """
    app = BlankSearchControlsHarness()
    async with app.run_test() as pilot:
        controls = app.query_one("#controls", LibraryMediaContentSearchControls)
        search_input = app.query_one("#library-media-content-search", Input)
        search_input.focus()
        await pilot.pause()
        assert app.focused is search_input

        controls.sync_query_state(
            is_markdown=True,
            query="cost",
            matches=(4,),
            match_index=0,
        )
        await pilot.pause()
        await pilot.pause()

        assert app.query_one("#library-media-content-search", Input) is search_input
        assert app.focused is search_input

        controls.sync_query_state(
            is_markdown=True,
            query="",
            matches=(),
            match_index=0,
        )
        await pilot.pause()
        await pilot.pause()

        assert app.query_one("#library-media-content-search", Input) is search_input
        assert app.focused is search_input


# ---------------------------------------------------------------------------
# TASK-21134: a search refresh restyles, it does not resize
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_refresh_does_not_arm_a_layout_pass(monkeypatch) -> None:
    """``sync_search`` restyles the SAME characters, so no layout is needed.

    task-22500: the virtualized Raw view restyles each row it paints
    directly from ``_query``/``_match_index`` -- there is no whole-document
    renderable to rebuild -- so forwarding a search update only needs to
    repaint, never relayout. ``LibraryMediaContentBody.sync_search`` now
    makes TWO calls against the Raw view per invocation --
    ``set_match_lines`` (feeding the active-match line list) and
    ``sync_search`` itself -- and each triggers its own ``self.refresh()``,
    whose ``layout`` default is already ``False``; this pins that BOTH
    calls never flip it to ``True`` (task-21134's original 84.9 -> 57.7
    ms/click regression guard, now against the widget that replaced the
    plain ``Static``). The exact call COUNT is not pinned -- only that none
    of them ever requests a layout pass.
    """
    body = LibraryMediaContentBody(
        content="alpha needle\nbeta\ngamma needle\n",
        is_markdown=False,
        mode="raw",
        query="",
        match_index=0,
        id="body",
    )
    app = BodyHarness(body)

    async with app.run_test(size=(60, 12)):
        raw_widget = body.raw_view
        assert raw_widget is not None

        seen: list[bool] = []
        real_refresh = VirtualizedRawContent.refresh

        def recording_refresh(
            self, *regions, repaint: bool = True, layout: bool = False, recompose: bool = False
        ):
            if self is raw_widget:
                seen.append(layout)
            return real_refresh(
                self, *regions, repaint=repaint, layout=layout, recompose=recompose
            )

        monkeypatch.setattr(VirtualizedRawContent, "refresh", recording_refresh)

        body.sync_search("needle", 0)
        body.sync_search("needle", 1)

        assert seen and all(layout is False for layout in seen), (
            f"search refresh armed a layout pass: {seen}"
        )


@pytest.mark.asyncio
async def test_search_refresh_still_repaints_the_active_match() -> None:
    """Skipping layout must not skip the repaint the highlight depends on.

    task-22500: there is no cached whole-document renderable left to go
    stale -- ``render_line`` reads ``_query``/``_match_index`` fresh on
    every call -- so this now pins that the Raw view's search state (and
    therefore what the NEXT paint will show) actually advances when
    ``sync_search`` is called twice in a row. Which row paints
    ACTIVE-vs-plain styling is task 7's ``set_match_lines`` wiring, not yet
    reachable here.
    """
    body = LibraryMediaContentBody(
        content="alpha needle\nbeta\ngamma needle\n",
        is_markdown=False,
        mode="raw",
        query="",
        match_index=0,
        id="body",
    )
    app = BodyHarness(body)

    async with app.run_test(size=(60, 12)):
        raw_widget = body.raw_view
        assert raw_widget is not None

        body.sync_search("needle", 0)
        assert raw_widget._query == "needle"
        assert raw_widget._match_index == 0
        assert "needle" in raw_widget.render_line(0).text

        body.sync_search("needle", 1)
        assert raw_widget._match_index == 1
        assert "needle" in raw_widget.render_line(2).text


# ---------------------------------------------------------------------------
# TASK-22500: the whole-document highlight ``Text`` is retired; only the
# match-LINE list survives, feeding ``VirtualizedRawContent.set_match_lines``.
# ---------------------------------------------------------------------------


EQUIVALENCE_DOCUMENTS = {
    "plain": "alpha needle\nbeta\ngamma needle\n",
    "crlf": "alpha needle\r\nbeta\r\nneedle at line start\r\n",
    "no-trailing-newline": "needle first\nlast needle",
    "mixed-case": "NEEDLE shouting\nneedle whispering\nNeEdLe wobbling",
    "repeated-on-one-line": "needle needle needle\nplain\nneedle",
    "unicode": "café needle ☕\nnaïve\nneedle — em dash",
    "blank-lines": "\n\nneedle\n\n\nneedle\n\n",
    "needle-only": "needle",
    "no-match": "nothing to see here\nmove along",
    "empty": "",
}


def test_match_lines_are_derived_without_building_a_document_text() -> None:
    """TASK-22500 step 1: the retired builders are gone; only the line list remains."""
    from tldw_chatbook.Widgets.Library import library_media_content as module

    assert not hasattr(module, "build_raw_content_renderable")
    lines = module.build_raw_content_match_lines(
        "alpha\nbudget here\nomega\nbudget", "budget"
    )
    assert lines == (1, 3)
    assert module.build_raw_content_match_lines("alpha", "") == ()
    assert module.build_raw_content_match_lines("", "x") == ()


def test_match_lines_agree_with_the_shared_matcher() -> None:
    """``build_raw_content_match_lines`` must equal ``find_content_matches``'.

    task-22500: replaces ``test_highlight_plan_match_lines_agree_with_the_
    shared_matcher``, which compared this same oracle against a
    ``RawContentHighlightPlan``'s ``.matches`` -- the plan is gone, but the
    equivalence it guarded (the screen's status count and scroll target
    must point at the same lines the Raw view highlights) still matters.
    """
    for content in EQUIVALENCE_DOCUMENTS.values():
        actual = build_raw_content_match_lines(content, "  NeEdLe ")
        expected = find_content_matches(content, "needle")
        assert actual == expected


def _match_line_activity(
    raw_widget: VirtualizedRawContent, line_count: int = 4
) -> dict[int, bool]:
    """Map every SOURCE line the Raw view currently paints reversed to active-or-not.

    Shared by the active-styling tests below: walks ``render_line`` (the
    same method Textual's compositor calls) and records, per line carrying
    the match style, whether it also carries ``bold`` (the active match) or
    not (a plain match).
    """
    found: dict[int, bool] = {}
    for row in range(line_count):
        strip = raw_widget.render_line(row)
        for segment in strip._segments:
            if segment.style is not None and segment.style.reverse:
                found[row] = bool(segment.style.bold)
    return found


@pytest.mark.asyncio
async def test_sync_search_marks_only_the_active_match_line() -> None:
    """``sync_search`` feeds ``set_match_lines`` so exactly one line paints active.

    task-22500: replaces ``test_highlight_plan_moves_only_the_active_span``,
    whose subject -- ``RawContentHighlightPlan.renderable`` rewriting Rich
    ``Span`` entries in place -- no longer exists. Active-vs-plain styling
    now lives in ``VirtualizedRawContent.render_line``, driven by the
    ``_match_lines``/``_match_index`` that ``LibraryMediaContentBody.
    sync_search`` feeds it via ``set_match_lines``; this pins that wiring
    end to end instead of a ``Text``'s spans.
    """
    body = LibraryMediaContentBody(
        content="needle one\nplain\nneedle two\nneedle three",
        is_markdown=False,
        mode="raw",
        query="",
        match_index=0,
        id="body",
    )
    app = BodyHarness(body)

    async with app.run_test(size=(60, 12)):
        raw_widget = body.raw_view
        assert raw_widget is not None

        body.sync_search("needle", 0)
        assert _match_line_activity(raw_widget) == {0: True, 2: False, 3: False}

        body.sync_search("needle", 1)
        assert _match_line_activity(raw_widget) == {0: False, 2: True, 3: False}

        # Wrapping past the end (3 matches, index 3) lands back on the first.
        body.sync_search("needle", 3)
        assert _match_line_activity(raw_widget) == {0: True, 2: False, 3: False}


@pytest.mark.asyncio
async def test_construction_seeds_the_active_match_before_any_sync_search() -> None:
    """FIX 2: a body mounted with an already-active query paints active on first frame.

    Before this fix, ``_build_raw_widget`` never called ``set_match_lines``,
    so a freshly (re)mounted Raw view with a live query painted every match
    PLAIN -- never bold -- until the NEXT ``sync_search`` call. That is
    visible after any recompose that carries a live query forward (a
    traversal, or the Rendered<->Raw toggle): the active match would flash
    unstyled for one frame. This mounts a body with a non-empty query and
    non-zero match index already set and asserts the active line is
    correctly bold on the very first paint, with ``sync_search`` never
    called.
    """
    body = LibraryMediaContentBody(
        content="needle one\nplain\nneedle two\nneedle three",
        is_markdown=False,
        mode="raw",
        query="needle",
        match_index=1,
        id="seeded-body",
    )
    app = BodyHarness(body)

    async with app.run_test(size=(60, 12)):
        raw_widget = body.raw_view
        assert raw_widget is not None
        # Matches are lines (0, 2, 3); match_index=1 is the second match: line 2.
        assert _match_line_activity(raw_widget) == {0: False, 2: True, 3: False}


@pytest.mark.asyncio
async def test_sync_search_memoizes_match_lines_per_content_and_query(
    monkeypatch,
) -> None:
    """FIX 1: repeated Prev/Next clicks must not re-scan the document.

    task-22500 originally called ``build_raw_content_match_lines``
    unconditionally from ``sync_search``, reopening the exact per-click
    O(document) cost task-22209 shipped to close (a Prev/Next click passes
    the SAME query as the previous call). This pins the memo: six calls
    against the SAME (content, query) cost exactly ONE scan; a newly
    submitted query costs exactly one fresh scan (and repeating THAT query
    costs no more); and a new document (a fresh body instance -- content is
    fixed per instance, so a document swap is always a new instance) costs
    one fresh scan too, even reusing a query string already seen elsewhere.
    A memo that never invalidates would be worse than no memo, so the
    "exactly one fresh call" assertions matter as much as the "no
    re-scan" ones.
    """
    import tldw_chatbook.Widgets.Library.library_media_content as module

    calls: list[str] = []
    original = module.build_raw_content_match_lines

    def counting(content: str, query: str) -> tuple[int, ...]:
        calls.append(query)
        return original(content, query)

    monkeypatch.setattr(module, "build_raw_content_match_lines", counting)

    body = LibraryMediaContentBody(
        content="needle one\nplain\nneedle two\nneedle three",
        is_markdown=False,
        mode="raw",
        query="",
        match_index=0,
        id="memo-body",
    )
    app = BodyHarness(body)

    async with app.run_test(size=(60, 12)):
        assert body.raw_view is not None
        assert calls == [], (
            "constructing/mounting with a BLANK query scanned the document"
        )

        # Six Prev/Next-style calls against the SAME (content, query) --
        # only the FIRST should cost a scan.
        for match_index in (0, 1, 2, 0, 1, 2):
            body.sync_search("needle", match_index)
        assert calls == ["needle"], f"same-query navigation re-scanned: {calls}"

        # A newly submitted query costs exactly one fresh scan...
        body.sync_search("plain", 0)
        assert calls == ["needle", "plain"]
        # ...and repeating that SAME new query costs no more.
        body.sync_search("plain", 0)
        body.sync_search("plain", 0)
        assert calls == ["needle", "plain"], "repeating the same query re-scanned"

    # A new document -- a fresh body instance, since `content` is fixed per
    # instance and never mutated after `__init__` -- costs one fresh scan,
    # even reusing the SAME query string as above ("plain").
    second_body = LibraryMediaContentBody(
        content="plain one\nneedle plain\nplain two",
        is_markdown=False,
        mode="raw",
        query="",
        match_index=0,
        id="memo-body-2",
    )
    second_app = BodyHarness(second_body)

    async with second_app.run_test(size=(60, 12)):
        assert second_body.raw_view is not None
        second_body.sync_search("plain", 0)
        assert calls == ["needle", "plain", "plain"], (
            "a new document reused the previous document's memo"
        )
        second_body.sync_search("plain", 0)
        assert calls == ["needle", "plain", "plain"], (
            "the new document's memo re-scanned on repeat"
        )


@pytest.mark.asyncio
async def test_body_search_updates_reuse_the_same_raw_view() -> None:
    """Catch a search update that rebuilds the Raw view instead of restyling it.

    task-22500: the virtualized widget replaced the whole-document highlight
    plan this used to pin -- each row restyles itself in ``render_line``, so
    there is nothing document-sized left to build or cache at the body
    level. What still matters is that repeated search updates reuse the
    SAME mounted widget (never re-``compose``/re-mount it) and that the
    widget's rendered rows reflect the query currently in effect.
    """
    body = LibraryMediaContentBody(
        content="alpha needle\nbeta beacon\ngamma needle\ndelta beacon",
        is_markdown=False,
        mode="raw",
        query="",
        match_index=0,
        id="body",
    )
    app = BodyHarness(body)

    async with app.run_test(size=(60, 12)):
        raw_widget = body.raw_view
        assert raw_widget is not None

        body.sync_search("needle", 0)
        assert body.raw_view is raw_widget
        assert "needle" in raw_widget.render_line(0).text

        body.sync_search("needle", 1)
        assert body.raw_view is raw_widget
        assert "needle" in raw_widget.render_line(2).text

        # A padded restatement of the same query is the same needle.
        body.sync_search("  needle ", 1)
        assert body.raw_view is raw_widget
        assert raw_widget._query == "needle"

        body.sync_search("beacon", 0)
        assert body.raw_view is raw_widget
        assert raw_widget._query == "beacon"
        assert "beacon" in raw_widget.render_line(1).text


@pytest.mark.asyncio
async def test_body_content_mutation_does_not_retroactively_change_the_mounted_raw_view() -> None:
    """The Raw view's document is frozen at construction (task-22500).

    ``VirtualizedRawContent`` builds its line lists once, in ``__init__``;
    ``sync_search`` only forwards ``query``/``match_index``, never
    ``content``. Production never mutates ``body.content`` after
    construction -- a content change recomposes the whole viewer and builds
    a fresh body -- so this pins that contract directly rather than relying
    on that higher-level recompose to exercise it.
    """
    body = LibraryMediaContentBody(
        content="alpha needle\nbeta",
        is_markdown=False,
        mode="raw",
        query="",
        match_index=0,
        id="body",
    )
    app = BodyHarness(body)

    async with app.run_test(size=(60, 12)):
        raw_widget = body.raw_view
        assert raw_widget is not None
        assert "alpha needle" in raw_widget.render_line(0).text

        body.content = "gamma needle\ndelta needle\nepsilon"
        body.sync_search("needle", 1)

        # Same widget instance, still showing the ORIGINAL document -- a
        # content change requires a new widget, produced by a fresh compose().
        assert body.raw_view is raw_widget
        assert "alpha needle" in raw_widget.render_line(0).text
