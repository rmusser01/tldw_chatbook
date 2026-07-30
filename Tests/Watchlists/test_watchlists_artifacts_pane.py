"""The Artifacts section: briefings, visible (spec #2 phase 1, task 4).

Tasks 1-3 built the `briefings` tables, the selection and the generation
service; none of it had a surface. These tests drive the real screen, the
real `SubscriptionsDB` behind `_build_test_app()`'s temp-dir profile, the
real selection and the real generation service. **Exactly one seam is
faked: the chat call**, injected at the service boundary (the screen's
`generate_briefing` reference) so everything below it -- selection,
statuses, junction rows, the coverage watermark -- is the shipping code.

Four properties are worth naming, because each one is a defect this task
could plausibly have shipped:

* Opening Artifacts must not mount CONTENT. The reader is the Items tab's
  affordance; every other section takes the full centre width.
* The briefing body is model output written from remote feed content, so
  its links must paint inert. Asserted through a real render, not through
  the source string.
* A `generating` row left behind by a crash must not wedge the Generate
  button forever -- and recovering it must not silently start a second
  generation over the top of one that may still be running elsewhere.
* A database error inside generation must not take the application down.
  `generate_briefing` deliberately lets those propagate (a database error
  is not a briefing outcome), and an exception escaping a Textual worker
  with the default `exit_on_error=True` exits the app.
"""

from __future__ import annotations

import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from io import StringIO
from unittest.mock import Mock

import pytest
from rich.console import Console
from textual.widgets import Button, DataTable, Static

from Tests.UI.test_destination_shells import DestinationHarness
from Tests.UI.test_destination_visual_parity_correction import (
    _visual_destination_harness,
)
from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.Subscriptions import briefing_service
from tldw_chatbook.Subscriptions.item_persist import persist_subscription_item
from tldw_chatbook.UI.Screens import watchlists_collections_screen as screen_module
from tldw_chatbook.UI.Screens.watchlists_collections_screen import (
    WatchlistsCollectionsScreen,
)
from tldw_chatbook.UI.Watchlists_Modules.artifacts_pane import ArtifactsPane
from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import TreeScope
from tldw_chatbook.UI.Watchlists_Modules.watchlists_tab_strip import SECTIONS

# Marked so CI actually runs this file: the unit job selects `-m unit` and
# the UI job runs `Tests/UI` plus `Tests -m ui --ignore=Tests/UI`, so an
# unmarked test in `Tests/Watchlists` is collected by nothing.
pytestmark = pytest.mark.ui


CANNED_BODY = (
    "## This week\n\n"
    "Acme shipped a thing.\n\n"
    "[Anthropic docs](https://evil.test/steal)\n\n"
    "[click](javascript:alert)\n\n"
    "More detail follows.\n"
)


class _FakeChat:
    """The one faked seam: a stand-in for `Chat_Functions.chat_api_call`."""

    def __init__(self, *, reply: object = CANNED_BODY):
        self.reply = reply
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply


def _use_fake_chat(monkeypatch, chat) -> None:
    """Fake the chat call at the service boundary, nothing else.

    `generate_briefing` binds its `chat` default at definition time, so
    patching `briefing_service.chat_api_call` would not reach it. Wrapping
    the screen's own reference instead keeps the whole service real --
    selection, statuses, junction rows, watermark -- with only the provider
    call replaced.
    """

    async def _generate(db, watchlist_id, **kwargs):
        return await briefing_service.generate_briefing(
            db, watchlist_id, chat=chat, **kwargs
        )

    monkeypatch.setattr(screen_module, "generate_briefing", _generate)


def _seed_watchlist(app, *, items: int = 2) -> int:
    """A watchlist with a source and some fresh items, through the real paths."""
    service = app.watchlist_bundle_service
    db = service.db
    watchlist = service.create("Morning AI Brief")
    watchlist_id = watchlist["id"]
    source_id = db.add_subscription(
        name="AI News", type="rss", source="https://ai-news.example/feed.xml"
    )
    service.add_source(watchlist_id, source_id)
    created = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    for index in range(items):
        with db.transaction() as conn:
            persist_subscription_item(
                conn,
                source_id,
                {
                    "url": f"https://ai-news.example/{index}",
                    "title": f"Story {index}",
                    "content": f"body of story {index}",
                    "content_hash": f"hash-{index}",
                    "content_kind": "article",
                    "content_format": "text",
                },
                run_id=None,
                now=created,
            )
    return watchlist_id


@asynccontextmanager
async def _open_artifacts(app, watchlist_id, *, size=(180, 50), visual=False):
    """Open the Watchlists screen on Artifacts, scoped to one watchlist."""
    host = (
        _visual_destination_harness(app, "watchlists_collections")
        if visual
        else DestinationHarness(app, "watchlists_collections")
    )
    async with host.run_test(size=size) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert isinstance(screen, WatchlistsCollectionsScreen)
        if watchlist_id is not None:
            screen.tree_scope = TreeScope(kind="watchlist", watchlist_id=watchlist_id)
        screen.active_section = "artifacts"
        await pilot.pause(0.2)
        yield screen, pilot, host


async def _press_generate(screen, pilot, *, ticks: int = 40):
    """Press the real Generate button and let the worker settle."""
    pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
    pane.query_one("#artifacts-generate-button", Button).press()
    for _ in range(ticks):
        await pilot.pause(0.05)
        if not screen._briefing_in_flight:
            break
    await pilot.pause(0.1)


def _briefing_rows(app, watchlist_id) -> list[dict]:
    return app.watchlist_bundle_service.db.list_briefings(watchlist_id)


def _render_to_console(renderable, *, width: int = 100) -> tuple[str, str]:
    """Render through a real Console and return (plain, ansi).

    The same helper `test_watchlists_content_pane.py` uses, for the same
    reason: the question is what was *interpreted*, and only a real render
    answers that.
    """
    console = Console(
        width=width,
        record=True,
        color_system="standard",
        force_terminal=True,
        file=StringIO(),
    )
    console.print(renderable)
    return console.export_text(clear=False), console.export_text(styles=True)


def _painted(screen, region) -> str:
    """Everything the compositor actually painted inside `region`."""
    strips = screen._compositor.render_strips()
    lines: list[str] = []
    for row in range(region.y, min(region.y + region.height, len(strips))):
        lines.append("".join(segment.text for segment in strips[row]))
    return "\n".join(lines)


def _assert_on_screen(widget, *, size, context: str) -> None:
    """Placement, not merely a non-zero height (the vacuous-guard lesson)."""
    width, height = size
    region = widget.region
    assert region.width > 0, f"{context} has no width"
    assert region.height > 0, f"{context} has no height"
    assert region.x >= 0 and region.y >= 0, f"{context} starts off-screen: {region}"
    assert region.x + region.width <= width, f"{context} runs off the right: {region}"
    assert region.y + region.height <= height, (
        f"{context} runs off the bottom of a {width}x{height} terminal: {region}"
    )


# --- 1. The section exists, and it does not drag CONTENT in with it -------


@pytest.mark.asyncio
async def test_artifacts_is_a_section_and_opening_it_leaves_content_unmounted():
    assert ("artifacts", "Artifacts") in SECTIONS, "the strip must offer Artifacts"

    app = _build_test_app()
    watchlist_id = _seed_watchlist(app)
    async with _open_artifacts(app, watchlist_id) as (screen, pilot, _host):
        assert screen.query_one("#wl-tab-artifacts", Button)
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        detail_pane = screen.query_one("#watchlists-detail-pane")

        # Full centre width, like Sources/Runs/Rules -- the pane fills the
        # region it was routed into rather than sharing it with a reader.
        assert pane.region.width == detail_pane.region.width > 0

        # The CONTENT gate keys on `active_section != "items"` and needed no
        # change for this section -- but nothing was asserting that, so a
        # future edit to the gate could quietly hand Artifacts a reader with
        # nothing to show and take a third of the centre column for it.
        assert not screen.query("#wl-region-content"), (
            "Artifacts must not mount the CONTENT reader"
        )
        assert screen.query_one("#wl-header-content"), (
            "CONTENT must still be reachable as its collapsed header"
        )
        assert screen.query_one("#watchlists-detail-title", Static)


# --- 2. Generate writes a briefing, and its body renders inert -------------


@pytest.mark.asyncio
async def test_generate_records_a_complete_briefing_and_renders_its_body(monkeypatch):
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    chat = _FakeChat()
    _use_fake_chat(monkeypatch, chat)

    # The production stylesheet, because half of what this test asserts is
    # what the compositor painted.
    async with _open_artifacts(app, watchlist_id, visual=True) as (
        screen,
        pilot,
        _host,
    ):
        pane_before = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        await _press_generate(screen, pilot)

        assert len(chat.calls) == 1, "exactly one provider call per briefing"
        rows = _briefing_rows(app, watchlist_id)
        assert [row["status"] for row in rows] == ["complete"]
        assert rows[0]["item_count"] == 2

        # The pane instance survived: a repaint, not a full-screen recompose.
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert pane is pane_before, (
            "generating a briefing must repaint the pane, not rebuild the screen"
        )

        table = pane.query_one("#artifacts-table", DataTable)
        assert table.row_count == 1
        painted_table = _painted(screen, table.region)
        assert "complete" in painted_table

        # The finished briefing is the one on screen, body and all.
        detail = pane.query_one("#artifacts-detail", Static)
        plain, ansi = _render_to_console(detail.renderable, width=detail.region.width)
        assert "Acme shipped a thing" in plain

        # A hostile link paints as inert text. `hyperlinks=False` is the whole
        # of it: no OSC-8 escape is emitted, so no label can hide a
        # destination the reader cannot see.
        assert "\x1b]8;" not in ansi, (
            "a briefing body must never emit a real terminal hyperlink -- the "
            "label is model-written over content fetched from a remote source"
        )
        # The destination is disclosed beside the label instead...
        assert "https://evil.test/steal" in plain
        # ...and the markdown branch really was taken (a plain-text render
        # would still show the raw link syntax).
        assert "[Anthropic docs](" not in plain
        # A `javascript:` link is not even link-shaped to the parser, so it
        # survives as the literal characters the model wrote.
        assert "[click](javascript:alert)" in plain

        # And that is what the terminal actually shows, not merely what the
        # renderable would produce in isolation.
        assert "evil.test/steal" in _painted(screen, detail.region)


# --- 3. A stuck `generating` row is recovered, and says so -----------------


@pytest.mark.asyncio
async def test_a_stuck_generating_row_is_refused_then_recovered(monkeypatch):
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    chat = _FakeChat()
    _use_fake_chat(monkeypatch, chat)

    # A worker that died mid-generation leaves exactly this behind.
    zombie_id = app.watchlist_bundle_service.db.insert_briefing(watchlist_id)

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, _host):
        # First press: refuses, and says why.
        await _press_generate(screen, pilot)

        assert chat.calls == [], "nothing may be generated while a row is in flight"
        assert app.notify.called, "a refusal must be visible, not silent"
        _args, kwargs = app.notify.call_args
        assert kwargs.get("severity") in {"warning", "error"}
        assert kwargs.get("markup") is False, (
            "toast bodies carrying counts and names must not be parsed as markup"
        )
        statuses = {row["id"]: row["status"] for row in _briefing_rows(app, watchlist_id)}
        assert "complete" not in statuses.values()

        # Second press: the zombie has been recovered, so this one proceeds.
        app.notify.reset_mock()
        await _press_generate(screen, pilot)

        assert len(chat.calls) == 1, (
            "after zombie recovery the same button must actually generate"
        )
        rows = _briefing_rows(app, watchlist_id)
        by_id = {row["id"]: row for row in rows}
        assert by_id[zombie_id]["status"] == "failed"
        assert by_id[zombie_id]["error"] == "interrupted"
        assert any(
            row["status"] == "complete" for row in rows if row["id"] != zombie_id
        ), "the second press must have written a real briefing"


# --- 4. A database error is reported, not fatal ----------------------------


@pytest.mark.asyncio
async def test_a_database_error_during_generation_does_not_exit_the_app(monkeypatch):
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)

    async def _explode(db, watchlist_id, **kwargs):
        # `generate_briefing` turns provider failures into `failed` rows but
        # deliberately lets database errors propagate -- see its docstring.
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(screen_module, "generate_briefing", _explode)

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        await _press_generate(screen, pilot)

        assert host.is_running, "a worker failure must not exit the application"
        assert host.screen_stack[-1] is screen, "the screen must still be standing"
        assert screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert app.notify.called, "a failure the user asked for must be reported"
        _args, kwargs = app.notify.call_args
        assert kwargs.get("severity") == "error"
        assert kwargs.get("markup") is False
        assert screen._briefing_in_flight is False, (
            "the in-flight guard must clear even when generation raises"
        )

        # The guard is genuinely re-armed: a second press is accepted.
        app.notify.reset_mock()
        await _press_generate(screen, pilot)
        assert app.notify.called


# --- Every status has a body of its own ------------------------------------


@pytest.mark.asyncio
async def test_failed_and_empty_briefings_explain_themselves():
    """No status renders as a blank pane.

    A `failed` row carries the provider's own message and an `empty` row
    says the window was empty -- both are outcomes the pipeline records
    deliberately (spec §Error-handling ethos), so both have to read as
    outcomes rather than as a body that failed to load.
    """
    app = _build_test_app()
    watchlist_id = _seed_watchlist(app)
    db = app.watchlist_bundle_service.db
    failed_id = db.insert_briefing(watchlist_id)
    db.update_briefing(failed_id, status="failed", error="openai: 429 rate limited")
    empty_id = db.insert_briefing(watchlist_id)
    db.update_briefing(empty_id, status="empty")

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, _host):
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert pane.query_one("#artifacts-table", DataTable).row_count == 2

        # Re-queried after each selection: `selected_briefing` is
        # `recompose=True`, so the previous `Static` is detached by then.
        pane.select_briefing_by_id(str(failed_id))
        await pilot.pause()
        plain, _ansi = _render_to_console(
            pane.query_one("#artifacts-detail", Static).renderable
        )
        assert "openai: 429 rate limited" in plain

        pane.select_briefing_by_id(str(empty_id))
        await pilot.pause()
        plain, _ansi = _render_to_console(
            pane.query_one("#artifacts-detail", Static).renderable
        )
        assert "nothing new arrived" in plain


@pytest.mark.asyncio
async def test_a_bracket_shaped_watchlist_name_paints_instead_of_exploding():
    """`Static` parses Rich markup by default, and the scope line names a
    watchlist the user typed.

    An unclosed tag would raise out of `compose()`, which exits the whole
    application -- so the pane wraps the line in a `Text`, and this pins
    both halves: the app survives, and the name paints verbatim rather than
    being interpreted or backslash-escaped.
    """
    app = _build_test_app()
    app.notify = Mock()
    hostile = "[bold red]Morning [brief"
    watchlist_id = app.watchlist_bundle_service.create(hostile)["id"]

    async with _open_artifacts(app, watchlist_id, visual=True) as (
        screen,
        pilot,
        host,
    ):
        assert host.is_running
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        note = pane.query_one("#artifacts-scope-note", Static)
        plain, ansi = _render_to_console(note.renderable, width=200)
        assert hostile in plain, "the name must paint exactly as it was typed"
        assert "\\[" not in plain, "and must not grow escaping backslashes"
        assert "\x1b[1;31m" not in ansi, "and `[bold red]` must not be applied"


@pytest.mark.asyncio
async def test_moving_the_tree_scope_moves_what_artifacts_is_about():
    """A briefing belongs to exactly one watchlist, so the pane follows the tree.

    Without this the pane would keep showing the previous watchlist's
    briefings while Generate acted on the new one -- the split-brain shape,
    on a surface that spends the user's provider quota.
    """
    app = _build_test_app()
    first = _seed_watchlist(app)
    second = app.watchlist_bundle_service.create("Security Watch")["id"]
    app.watchlist_bundle_service.db.insert_briefing(first)

    async with _open_artifacts(app, first) as (screen, pilot, _host):
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert pane.query_one("#artifacts-table", DataTable).row_count == 1

        screen.tree_scope = TreeScope(kind="watchlist", watchlist_id=second)
        await pilot.pause(0.2)

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert pane.query_one("#artifacts-table", DataTable).row_count == 0
        assert pane.selected_briefing is None
        assert "Security Watch" in pane.scope_label


# --- 5. Geometry, under the production stylesheet --------------------------


@pytest.mark.parametrize("size", [(160, 42), (180, 50)])
@pytest.mark.asyncio
async def test_the_list_the_button_and_the_body_are_all_on_screen(size, monkeypatch):
    """List + Generate + detail placed inside the terminal, real CSS.

    The fixture is what makes this bite: a long body is exactly the state in
    which an unconstrained detail `Static` lays itself out past the bottom
    of the terminal, so this asserts placement rather than a non-zero
    height.
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    long_body = "\n\n".join(f"Paragraph {index} of the briefing." for index in range(40))
    _use_fake_chat(monkeypatch, _FakeChat(reply=long_body))

    async with _open_artifacts(app, watchlist_id, size=size, visual=True) as (
        screen,
        pilot,
        _host,
    ):
        await _press_generate(screen, pilot)
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert pane.selected_briefing is not None, "the fixture must select a body"

        table = pane.query_one("#artifacts-table", DataTable)
        button = pane.query_one("#artifacts-generate-button", Button)
        detail = pane.query_one("#artifacts-detail", Static)

        for widget, context in (
            (table, "the briefing list"),
            (button, "the Generate button"),
            (detail, "the briefing body"),
        ):
            assert widget.display, f"{context} is not displayed"
            _assert_on_screen(widget, size=size, context=context)

        # And in that order, with the body under the list it belongs to.
        assert button.region.y < table.region.y < detail.region.y

        # The Generate label is really painted, not merely placed.
        assert "Generate" in _painted(screen, button.region)
