"""Mounted behavior tests for the Library media-content widgets."""

import asyncio

import pytest
from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Markdown, Static

import tldw_chatbook.Widgets.Library.library_media_content as media_content_module
from tldw_chatbook.Library.library_media_viewer_state import find_content_matches
from tldw_chatbook.Widgets.Library.library_media_content import (
    LibraryMediaContentBody,
    LibraryMediaContentSearchControls,
    RawContentHighlightPlan,
    build_raw_content_highlight_plan,
    build_raw_content_renderable,
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
    """Hold the first Rendered mount until the test selects a winner."""

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
            if self._raw_ensure_calls == 2:
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
        raw = body.query_one("#library-media-viewer-content-text", Static)
        assert not body.query("#library-media-viewer-content-markdown")

        await body.sync_mode("rendered")
        markdown = body.query_one("#library-media-viewer-content-markdown", Markdown)
        await body.sync_mode("raw")
        await body.sync_mode("rendered")

        assert body.query_one("#library-media-viewer-content-text") is raw
        assert body.query_one("#library-media-viewer-content-markdown") is markdown


@pytest.mark.asyncio
async def test_body_search_updates_lazily_mounted_raw_rich_highlights() -> None:
    """Catch Raw construction that loses search state set while Rendered is visible."""
    body = LibraryMediaContentBody(
        content="# Heading\n\nbudget one\nbudget two",
        is_markdown=True,
        mode="rendered",
        query="",
        match_index=0,
        id="library-media-viewer-content",
    )
    async with BodyHarness(body).run_test() as _pilot:
        body.sync_search("budget", 1)
        await body.sync_mode("raw")

        raw = body.query_one("#library-media-viewer-content-text", Static)
        assert isinstance(raw.renderable, Text)
        budget_spans = [
            span
            for span in raw.renderable.spans
            if raw.renderable.plain[span.start : span.end] == "budget"
        ]
        assert len(budget_spans) == 2
        assert [str(span.style) for span in budget_spans] == [
            "reverse",
            "reverse bold",
        ]


@pytest.mark.asyncio
async def test_body_rapid_mode_changes_leave_latest_mode_visible_once() -> None:
    """Catch an earlier delayed request overriding the latest mode visibility."""
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

        raw = body.query_one("#library-media-viewer-content-text", Static)
        markdown = body.query_one("#library-media-viewer-content-markdown", Markdown)
        assert raw.display
        assert not markdown.display

        release_raw.set()
        await asyncio.gather(rendered_task, raw_task)

        assert raw.display
        assert not markdown.display
        assert len(body.query("#library-media-viewer-content-text")) == 1
        assert len(body.query("#library-media-viewer-content-markdown")) == 1


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
async def test_blank_query_sync_removes_active_search_controls() -> None:
    """Catch a blank-query transition that leaves stale controls mounted."""
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

        assert not app.query("#library-media-content-search-status")
        assert not app.query("#library-media-content-search-prev")
        assert not app.query("#library-media-content-search-next")


@pytest.mark.asyncio
async def test_active_query_sync_mounts_controls_and_markdown_placeholder() -> None:
    """Catch a blank-to-active transition that fails to recompose its controls."""
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
        assert active_input is not blank_input
        assert active_input.placeholder == "Search content (raw text)…"
        assert app.query_one("#library-media-content-search-prev", Button)
        assert app.query_one("#library-media-content-search-next", Button)


# ---------------------------------------------------------------------------
# TASK-21134: a search refresh restyles, it does not resize
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_refresh_does_not_arm_a_layout_pass(monkeypatch) -> None:
    """``sync_search`` restyles the SAME characters, so no layout is needed.

    ``build_raw_content_renderable`` only moves highlight spans around: it
    never adds, removes or rewraps a line, so the Static's size cannot change
    between match-nav clicks. The layout pass ``Static.update()`` arms by
    default cost a measured 84.9 -> 57.7 ms of CPU per click on a 100 KB
    document at 120x40 (10 Layout messages per 10 clicks -> 0).
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
        raw_widget = body._raw_widget
        assert raw_widget is not None

        seen: list[bool] = []
        real_update = Static.update

        def recording_update(self, content="", *, layout: bool = True) -> None:
            if self is raw_widget:
                seen.append(layout)
            real_update(self, content, layout=layout)

        monkeypatch.setattr(Static, "update", recording_update)

        body.sync_search("needle", 0)
        body.sync_search("needle", 1)

        assert seen == [False, False], f"search refresh armed a layout pass: {seen}"


@pytest.mark.asyncio
async def test_search_refresh_still_repaints_the_active_match() -> None:
    """Skipping layout must not skip the repaint the highlight depends on."""
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
        raw_widget = body._raw_widget
        assert raw_widget is not None

        body.sync_search("needle", 0)
        first = raw_widget.renderable
        assert isinstance(first, Text)
        first_active = [
            span for span in first.spans if "bold" in str(span.style)
        ]

        body.sync_search("needle", 1)
        second = raw_widget.renderable
        assert isinstance(second, Text)
        second_active = [
            span for span in second.spans if "bold" in str(span.style)
        ]

        assert first.plain == second.plain  # same characters: no relayout owed
        assert first_active and second_active
        assert first_active[0].start != second_active[0].start


# ---------------------------------------------------------------------------
# TASK-22209: one document pass per (document, query), not per click
# ---------------------------------------------------------------------------


def _reference_renderable(content: str, query: str, match_index: int) -> Text | str:
    """The pre-task-22209 highlighter, kept verbatim as the equivalence oracle.

    The plan-based builder must produce byte-identical text and identical
    spans to this -- the point of task-22209 was to stop RUNNING it per
    click, not to change what it draws.
    """
    display_content = content or "No stored content."
    normalized_query = query.strip()
    if not normalized_query or not content:
        return display_content
    matches = find_content_matches(content, normalized_query)
    if not matches:
        return display_content
    current_line = matches[match_index % len(matches)]
    needle = normalized_query.lower()
    text = Text()
    for line_index, line in enumerate(display_content.split("\n")):
        if line_index:
            text.append("\n")
        hit = line.lower().find(needle)
        if hit < 0:
            text.append(line)
            continue
        text.append(line[:hit])
        text.append(
            line[hit : hit + len(needle)],
            style="reverse bold" if line_index == current_line else "reverse",
        )
        text.append(line[hit + len(needle) :])
    return text


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


@pytest.mark.parametrize("label", sorted(EQUIVALENCE_DOCUMENTS))
@pytest.mark.parametrize("match_index", [0, 1, 5])
def test_highlight_plan_matches_the_previous_renderer(label, match_index) -> None:
    """Catch a plan rebuild that draws anything different from the old loop."""
    content = EQUIVALENCE_DOCUMENTS[label]
    expected = _reference_renderable(content, "needle", match_index)
    actual = build_raw_content_renderable(content, "needle", match_index)

    if isinstance(expected, str):
        assert actual == expected
        return
    assert isinstance(actual, Text)
    assert actual.plain == expected.plain
    assert list(actual.spans) == list(expected.spans)


def test_highlight_plan_match_lines_agree_with_the_shared_matcher() -> None:
    """The plan's own match list must equal ``find_content_matches``'.

    The plan derives its matches in the SAME pass that builds the spans (so
    span ``i`` belongs to match ``i``); if that ever drifted from the
    matcher the screen's status count and scroll target would point at
    different lines than the highlights.
    """
    for content in EQUIVALENCE_DOCUMENTS.values():
        plan = build_raw_content_highlight_plan(content, "  NeEdLe ")
        expected = find_content_matches(content, "needle")
        if isinstance(plan, str):
            assert expected == ()
            continue
        assert plan.matches == expected


def test_highlight_plan_moves_only_the_active_span() -> None:
    """Navigating between matches restyles two spans and rebuilds nothing."""
    plan = build_raw_content_highlight_plan(
        "needle one\nplain\nneedle two\nneedle three", "needle"
    )
    assert isinstance(plan, RawContentHighlightPlan)

    first = plan.renderable(0)
    assert [str(span.style) for span in first.spans] == [
        "reverse bold",
        "reverse",
        "reverse",
    ]

    second = plan.renderable(1)
    assert second is first
    assert [str(span.style) for span in second.spans] == [
        "reverse",
        "reverse bold",
        "reverse",
    ]

    # Wrapping past the end lands back on the first match.
    wrapped = plan.renderable(3)
    assert [str(span.style) for span in wrapped.spans] == [
        "reverse bold",
        "reverse",
        "reverse",
    ]


@pytest.mark.asyncio
async def test_body_reuses_one_plan_per_query_and_rebuilds_on_a_new_one(
    monkeypatch,
) -> None:
    """Catch a body that rebuilds its highlight plan per match-nav click."""
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
        raw_widget = body._raw_widget
        assert raw_widget is not None

        builds: list[tuple[str, str]] = []
        real_builder = build_raw_content_highlight_plan

        def counting_builder(content: str, query: str):
            builds.append((content, query))
            return real_builder(content, query)

        monkeypatch.setattr(
            media_content_module,
            "build_raw_content_highlight_plan",
            counting_builder,
        )

        body.sync_search("needle", 0)
        first = raw_widget.renderable
        body.sync_search("needle", 1)
        body.sync_search("needle", 0)
        assert raw_widget.renderable is first
        assert len(builds) == 1, (
            f"The body rebuilt its highlight plan {len(builds)} times for "
            "one (document, query) pair."
        )

        # A padded restatement of the same query is the same needle.
        body.sync_search("  needle ", 1)
        assert raw_widget.renderable is first
        assert len(builds) == 1

        body.sync_search("beacon", 0)
        assert len(builds) == 2
        second = raw_widget.renderable
        assert second is not first
        assert isinstance(second, Text)
        assert {
            second.plain[span.start : span.end] for span in second.spans
        } == {"beacon"}


@pytest.mark.asyncio
async def test_body_replacing_its_document_rebuilds_the_plan() -> None:
    """The plan key holds the document, not just the query.

    The mounted screen never does this -- a content change recomposes the
    viewer and builds a FRESH body -- so this pins the widget's own
    contract rather than a live screen path: a body asked to highlight a
    document it was not holding before must not answer from the previous
    document's spans.
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
        raw_widget = body._raw_widget
        assert raw_widget is not None

        body.sync_search("needle", 0)
        first = raw_widget.renderable
        assert isinstance(first, Text)
        assert first.plain == "alpha needle\nbeta"

        body.content = "gamma needle\ndelta needle\nepsilon"
        body.sync_search("needle", 1)

        second = raw_widget.renderable
        assert isinstance(second, Text)
        assert second is not first
        assert second.plain == "gamma needle\ndelta needle\nepsilon"
        assert [str(span.style) for span in second.spans] == [
            "reverse",
            "reverse bold",
        ]
