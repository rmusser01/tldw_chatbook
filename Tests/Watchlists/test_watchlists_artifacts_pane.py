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

import json
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from io import StringIO
from unittest.mock import Mock

import pytest
from rich.console import Console
from textual.coordinate import Coordinate
from textual.widgets import Button, DataTable, Select, Static

from Tests.UI.test_destination_shells import DestinationHarness, _static_text
from Tests.UI.test_destination_visual_parity_correction import (
    _visual_destination_harness,
)
from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.Subscriptions import briefing_cast, briefing_service
from tldw_chatbook.Subscriptions.briefing_cast import dump_roster
from tldw_chatbook.Subscriptions.item_persist import persist_subscription_item
from tldw_chatbook.UI.Screens import watchlists_collections_screen as screen_module
from tldw_chatbook.UI.Screens.watchlists_collections_screen import (
    WatchlistsCollectionsScreen,
)
from tldw_chatbook.UI.Watchlists_Modules.artifacts_pane import (
    ArtifactsPane,
    CastScriptRequested,
    GenerateBriefingRequested,
)
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


async def _press_generate(screen, pilot, app, watchlist_id, *, timeout: float = 20.0):
    """Press the real Generate button and wait until the press is answered.

    Waits on observable state, never on a fixed sleep (fix round 1, Finding
    4). The first loop ends when the press has been *answered* -- the guard
    was claimed (the handler sets `_briefing_in_flight` synchronously, so
    that is the acceptance signal), or a toast refused it before dispatch,
    or the rows already changed. The second waits out the worker. The third
    waits for the repaint to agree with the database, which is the last
    thing a press causes. A generous timeout bounds all three, so a hung
    worker fails as a test failure rather than as a confident assertion
    about a half-finished state.
    """
    db = app.watchlist_bundle_service.db
    rows_before = len(db.list_briefings(watchlist_id))
    notes_before = getattr(app.notify, "call_count", 0)

    pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
    pane.query_one("#artifacts-generate-button", Button).press()

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await pilot.pause(0.02)
        if (
            screen._briefing_in_flight
            or getattr(app.notify, "call_count", 0) > notes_before
            or len(db.list_briefings(watchlist_id)) != rows_before
        ):
            break
    while time.monotonic() < deadline and screen._briefing_in_flight:
        await pilot.pause(0.02)
    while time.monotonic() < deadline:
        await pilot.pause(0.02)
        table = screen.query_one(
            "#watchlists-artifacts-pane", ArtifactsPane
        ).query_one("#artifacts-table", DataTable)
        if table.row_count == len(db.list_briefings(watchlist_id)):
            return


def _briefing_rows(app, watchlist_id) -> list[dict]:
    return app.watchlist_bundle_service.db.list_briefings(watchlist_id)


# --- Task 5: casting a script ------------------------------------------
#
# Same seam discipline as `_use_fake_chat` above: `generate_script` binds
# its `chat` default at definition time too, so the fake is wrapped around
# the SCREEN's own `generate_script` reference (`screen_module.
# generate_script`), keeping the real service -- roster validation, prompt
# building, strict turn parsing, the snapshot -- running underneath, with
# only the provider call replaced.


def _use_fake_cast_chat(monkeypatch, chat) -> None:
    async def _generate(db, briefing_id, **kwargs):
        return await briefing_cast.generate_script(db, briefing_id, chat=chat, **kwargs)

    monkeypatch.setattr(screen_module, "generate_script", _generate)


ONE_SPEAKER_ROSTER = [{"name": "Narrator", "role_prompt": "Calm narration."}]


async def _prepare_cast(screen, pilot, app, watchlist_id, *, roster=None) -> int:
    """Generate a `complete` briefing, then create+select a default preset
    so the Cast button is enabled. Returns the briefing's id.

    Real Generate (not a raw `db.insert_briefing`) so the briefing selected
    afterwards is the one the whole screen already agrees on -- exactly
    what `_generate_briefing`'s own `select_briefing_id=generated_id`
    leaves behind.
    """
    db = app.watchlist_bundle_service.db
    await _press_generate(screen, pilot, app, watchlist_id)
    briefing_id = _briefing_rows(app, watchlist_id)[0]["id"]
    preset_id = db.insert_briefing_preset(
        "Solo", roster_json=dump_roster(roster or ONE_SPEAKER_ROSTER)
    )
    db.set_watchlist_briefing_settings(watchlist_id, default_preset_id=preset_id)
    await screen._load_briefings()
    await pilot.pause()
    return briefing_id


async def _press_cast(screen, pilot, app, briefing_id, *, timeout: float = 20.0):
    """Press the real Cast button and wait until the press is answered.

    Mirrors `_press_generate` exactly, scoped to one briefing's scripts
    rather than a whole watchlist's briefings.
    """
    db = app.watchlist_bundle_service.db
    scripts_before = len(db.list_briefing_scripts(briefing_id))
    notes_before = getattr(app.notify, "call_count", 0)

    pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
    pane.query_one("#artifacts-cast-button", Button).press()

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await pilot.pause(0.02)
        if (
            screen._cast_in_flight
            or getattr(app.notify, "call_count", 0) > notes_before
            or len(db.list_briefing_scripts(briefing_id)) != scripts_before
        ):
            break
    while time.monotonic() < deadline and screen._cast_in_flight:
        await pilot.pause(0.02)
    while time.monotonic() < deadline:
        await pilot.pause(0.02)
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        table = pane.query_one("#artifacts-scripts-table", DataTable)
        if table.row_count == len(db.list_briefing_scripts(briefing_id)):
            return


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
        assert screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)

        # (The full-width claim is asserted in the real-CSS geometry test
        # below, against the width Sources gets on the same terminal --
        # comparing this pane to its own parent under a harness that applies
        # no stylesheet could not have failed. Fix round 1, Minor d.)

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


@pytest.mark.asyncio
async def test_artifacts_says_it_is_local_like_the_notifications_inbox():
    """Parity with the one sibling section that has no server half.

    Briefings are written to, and read from, this device's `SubscriptionsDB`
    whatever the Backend selector says. Offering the choice anyway would be
    a lie about where the rows come from, so the selector is disabled and
    the label states the truth -- exactly what Notifications already does.
    """
    app = _build_test_app()
    watchlist_id = _seed_watchlist(app)
    async with _open_artifacts(app, watchlist_id) as (screen, pilot, _host):
        backend_select = screen.query_one("#watchlists-backend-select")
        assert backend_select.disabled is True
        assert "local" in _static_text(
            screen.query_one("#watchlists-backend-label", Static)
        )

        # And the parity is real, not a coincidence of copy: Sources gets
        # the live selector back.
        screen.active_section = "sources"
        await pilot.pause(0.2)
        assert screen.query_one("#watchlists-backend-select").disabled is False


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
        await _press_generate(screen, pilot, app, watchlist_id)

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
    """The Generate path's OWN recovery-then-refuse-then-recover sequence.

    The zombie is seeded AFTER Artifacts is already open, on purpose:
    whole-branch review fix 3 wired zombie recovery into the Artifacts-load
    path too, so a row seeded BEFORE opening would already be recovered by
    the plain load, before the first press ever runs -- collapsing this
    test's two presses into one and asserting nothing about the Generate
    path's own sweep.

    Seeding post-load is not enough on its own: the worker's `finally`
    reloads Artifacts after `_briefing_in_flight` clears, and the
    load-path recovery would silently recover the zombie between the two
    presses even with the button's own sweep deleted -- every outcome
    assertion here stays green under that mutation. What actually pins
    the Generate path is the recorder below: the load path only ever
    calls `fail_interrupted_briefings` when `_briefing_in_flight` is
    clear (`_fail_interrupted_briefings_if_safe`'s guard), so a call
    observed with the flag CLAIMED can only be the button's own sweep.
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    chat = _FakeChat()
    _use_fake_chat(monkeypatch, chat)

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, _host):
        # A worker that died mid-generation leaves exactly this behind --
        # inserted only now, after the section's own load has already run.
        zombie_id = app.watchlist_bundle_service.db.insert_briefing(watchlist_id)

        # Record `_briefing_in_flight` at every recovery call: True can
        # only come from the Generate worker's own sweep (see docstring).
        in_flight_at_call: list[bool] = []
        real_fail = screen_module.fail_interrupted_briefings

        def _recording_fail(db, watchlist_id=None):
            in_flight_at_call.append(bool(screen._briefing_in_flight))
            return real_fail(db, watchlist_id)

        monkeypatch.setattr(
            screen_module, "fail_interrupted_briefings", _recording_fail
        )

        # First press: refuses, and says why.
        await _press_generate(screen, pilot, app, watchlist_id)

        assert True in in_flight_at_call, (
            "the zombie must be recovered by the Generate path's OWN sweep"
            " (a call with `_briefing_in_flight` claimed), not merely by"
            " the load-path recovery that runs after the flag clears"
        )

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
        await _press_generate(screen, pilot, app, watchlist_id)

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


@pytest.mark.asyncio
async def test_a_zombie_generating_row_is_recovered_on_a_plain_artifacts_load():
    """Whole-branch review fix 3: the spec says a `generating` row not
    backed by a live worker is failed "on the next Generate attempt OR
    Artifacts load" -- only the Generate path was wired. A user who opens
    Artifacts after a crash, without ever touching the Generate button,
    must not see a briefing stuck reading `generating` forever.
    """
    app = _build_test_app()
    watchlist_id = _seed_watchlist(app)
    zombie_id = app.watchlist_bundle_service.db.insert_briefing(watchlist_id)

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, _host):
        # Nothing pressed Generate -- this is a plain section open. Poll
        # rather than trust `_open_artifacts`'s own fixed pause: the sweep
        # and the list read (Qodo round 1, FIX A) both now hop off the UI
        # thread via `asyncio.to_thread`, so under a busy full-suite run
        # the thread-pool dispatch can outlast a short fixed wait even
        # though it always finishes eventually.
        deadline = time.monotonic() + 10.0
        by_id = {}
        while time.monotonic() < deadline:
            by_id = {row["id"]: row for row in _briefing_rows(app, watchlist_id)}
            if by_id.get(zombie_id, {}).get("status") == "failed":
                break
            await pilot.pause(0.05)
        assert by_id[zombie_id]["status"] == "failed"
        assert by_id[zombie_id]["error"] == "interrupted"

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        pane.select_briefing_by_id(str(zombie_id))
        await pilot.pause()
        plain, _ansi = _render_to_console(
            pane.query_one("#artifacts-detail", Static).renderable
        )
        assert "interrupted" in plain
        assert "This briefing is being written now." not in plain


@pytest.mark.asyncio
async def test_a_live_in_flight_row_is_not_failed_by_a_concurrent_load():
    """The other half of fix 3's guard: a row THIS screen's own worker is
    still writing must not be treated as a zombie just because Artifacts
    happens to reload while it is running.

    `fail_interrupted_briefings` fails EVERY `generating` row for a
    watchlist unconditionally -- it has no way to tell a live row from a
    crashed one on its own. `_briefing_in_flight` is the one signal that
    can, so the load path must respect it exactly like the Generate path
    already does.
    """
    app = _build_test_app()
    watchlist_id = _seed_watchlist(app)
    db = app.watchlist_bundle_service.db

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, _host):
        # Stands in for a real live worker's row, claimed AFTER the
        # section's initial load so it survives to the point being tested.
        live_id = db.insert_briefing(watchlist_id)
        screen._briefing_in_flight = True
        try:
            await screen._load_briefings()
        finally:
            screen._briefing_in_flight = False

        assert db.get_briefing(live_id)["status"] == "generating", (
            "a row the guard says is live must survive a concurrent load"
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_briefings_list_read_runs_off_the_event_loop_thread():
    """Qodo round 1, FIX A: pin the load-bearing part of moving the read off
    the UI thread. `run_worker` alone only *schedules* `_load_briefings`
    back onto the SAME event loop -- it is `asyncio.to_thread` wrapping
    `db.list_briefings` that actually gets the SELECT off it. Same pattern
    as `test_the_queue_write_runs_off_the_event_loop_thread` in
    `Tests/UI/test_watchlists_inspector.py`: a mutation that drops
    `to_thread` and calls `list_briefings` directly still ends in the same
    state (rows loaded, pane repainted), so only watching WHICH thread runs
    the call can tell the two apart.
    """
    app = _build_test_app()
    watchlist_id = _seed_watchlist(app)
    db = app.watchlist_bundle_service.db

    loop_thread_id = threading.get_ident()
    read_thread_ids: list[int] = []
    real_list_briefings = db.list_briefings

    def _spy(watchlist_id_arg):
        read_thread_ids.append(threading.get_ident())
        return real_list_briefings(watchlist_id_arg)

    db.list_briefings = _spy

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, _host):
        # The section open already triggers a load; wait for it, then force
        # a second one explicitly so the assertion does not depend on
        # exactly when the initial one ran.
        read_thread_ids.clear()
        await screen._load_briefings()

    assert read_thread_ids, "the read must have run at all"
    assert all(thread_id != loop_thread_id for thread_id in read_thread_ids), (
        "db.list_briefings must run off the event-loop thread "
        "(asyncio.to_thread), not synchronously inside the worker on the "
        "same thread that runs the event loop"
    )


@pytest.mark.asyncio
async def test_the_refusal_toast_names_the_watchlist_actually_generating():
    """Whole-branch review fix 4: `_briefing_in_flight` is deliberately
    screen-global (the `wl-briefing` worker group is `exclusive=True`, so a
    second dispatch would cancel a real generation mid-run -- it must not
    become per-watchlist). But the refusal copy claimed the running
    briefing was "for this watchlist", which is false the moment the
    watchlist actually generating differs from the one on screen.
    """
    app = _build_test_app()
    app.notify = Mock()
    running_id = _seed_watchlist(app)  # named "Morning AI Brief"
    other_id = app.watchlist_bundle_service.create("Other Watch")["id"]

    async with _open_artifacts(app, other_id) as (screen, pilot, _host):
        # A generation for a DIFFERENT watchlist than the one on screen --
        # exactly the scenario the old copy lied about.
        screen._briefing_in_flight = True
        screen._briefing_in_flight_watchlist_id = running_id
        try:
            screen.handle_generate_briefing_requested(GenerateBriefingRequested())
        finally:
            screen._briefing_in_flight = False
            screen._briefing_in_flight_watchlist_id = None

        assert app.notify.called
        args, kwargs = app.notify.call_args
        message = args[0] if args else str(kwargs.get("message", ""))
        assert kwargs.get("markup") is False
        assert "Morning AI Brief" in message, (
            "the toast must name the watchlist actually generating"
        )
        assert "for this watchlist" not in message.lower()


# --- The guard is claimed at dispatch, not inside the worker ---------------


@pytest.mark.asyncio
async def test_two_fast_presses_generate_exactly_once(monkeypatch):
    """Outcome half: hammering Generate must produce one briefing, not two.

    `run_worker` only *schedules*, so a guard checked inside the worker body
    leaves a window in which two presses both pass -- and `exclusive=True`
    then cancels the first mid-generation, leaving the `generating` row the
    guard exists to prevent.
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    chat = _FakeChat()
    _use_fake_chat(monkeypatch, chat)

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, _host):
        button = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane).query_one(
            "#artifacts-generate-button", Button
        )
        # Both in one tick, with nothing awaited between them.
        button.press()
        button.press()

        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            await pilot.pause(0.02)
            if not screen._briefing_in_flight and chat.calls:
                break
        await pilot.pause(0.1)

        assert len(chat.calls) == 1, "two presses must not buy two generations"
        rows = _briefing_rows(app, watchlist_id)
        assert [row["status"] for row in rows] == ["complete"], (
            f"expected exactly one complete briefing, got {[r['status'] for r in rows]}"
        )


@pytest.mark.asyncio
async def test_the_guard_is_claimed_before_the_worker_runs(monkeypatch):
    """Mechanism half, and the deterministic one.

    The outcome test above cannot see the dispatch window: whichever way the
    flag is set, `exclusive=True` collapses two same-tick dispatches to one
    generation, so it passes either way (measured -- see the task report).
    This one pins the invariant itself, with no race in it: the handler is
    synchronous and has no `await`, so when it returns, no worker code can
    yet have run. If the guard is claimed there, it is already True at that
    instant; if it is claimed inside the worker body, it is still False.
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    chat = _FakeChat()
    _use_fake_chat(monkeypatch, chat)

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, _host):
        screen.handle_generate_briefing_requested(GenerateBriefingRequested())

        assert screen._briefing_in_flight is True, (
            "the guard must be claimed by the handler, before `run_worker` "
            "has scheduled anything"
        )
        assert chat.calls == [], "and no worker code can have run yet"

        # A press arriving in that window is refused rather than dispatched.
        app.notify.reset_mock()
        screen.handle_generate_briefing_requested(GenerateBriefingRequested())
        assert app.notify.call_count == 1
        _args, kwargs = app.notify.call_args
        assert kwargs.get("markup") is False

        # Let the one accepted generation finish so the worker is not
        # cancelled out from under the harness at teardown.
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and screen._briefing_in_flight:
            await pilot.pause(0.02)
        assert len(chat.calls) == 1


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
        await _press_generate(screen, pilot, app, watchlist_id)

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

        # The guard is genuinely re-armed. Asserted on the SERVICE, not on
        # "some toast happened" (fix round 1, Finding 3): a refusal toasts
        # identically, so the old assertion could not tell a re-armed button
        # from a permanently wedged one. With the database reachable again,
        # the same button must reach the service and leave a briefing behind.
        chat = _FakeChat()
        _use_fake_chat(monkeypatch, chat)
        app.notify.reset_mock()
        await _press_generate(screen, pilot, app, watchlist_id)

        assert len(chat.calls) == 1, (
            "the second press must have reached the generation service"
        )
        assert any(
            row["status"] == "complete" for row in _briefing_rows(app, watchlist_id)
        ), "and must have left a finished briefing behind"


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
async def test_only_a_focused_tables_highlight_selects(monkeypatch):
    """The `has_focus` gate, asserted rather than assumed (fix round 1, 5e).

    Assigning `briefings` recomposes this pane, which builds a BRAND NEW
    `DataTable` whose cursor starts on row 0 and announces it. Forwarding
    that announcement to selection makes the pane fight its own rebuild --
    the 157-selections-from-one-tab-open shape. A table that has just been
    mounted holds no focus, which is the discriminator
    `highlight_is_user_driven` uses.
    """
    app = _build_test_app()
    watchlist_id = _seed_watchlist(app)
    db = app.watchlist_bundle_service.db
    first = db.insert_briefing(watchlist_id)
    db.update_briefing(first, status="empty")
    second = db.insert_briefing(watchlist_id)
    db.update_briefing(second, status="empty")

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, _host):
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        table = pane.query_one("#artifacts-table", DataTable)
        assert table.row_count == 2
        assert not table.has_focus, "the fixture needs an unfocused table"

        # The rebuild's own row-0 announcement must change nothing.
        assert pane.selected_briefing is None
        assert screen._selected_briefing is None

        # A user-driven highlight on the focused table does select.
        table.focus()
        await pilot.pause()
        table.cursor_coordinate = Coordinate(1, 0)
        await pilot.pause(0.1)

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert pane.selected_briefing is not None
        assert pane.selected_briefing["id"] == db.list_briefings(watchlist_id)[1]["id"]


@pytest.mark.asyncio
async def test_a_bracket_shaped_watchlist_name_paints_instead_of_exploding():
    """`Static` parses Rich markup by default, and the scope line names a
    watchlist the user typed.

    Measured behaviour: with a bare `str` the tag is silently SWALLOWED and
    the name loses characters (Textual tolerated the unclosed `[brief`
    rather than raising). The pane wraps the line in a `Text`, and this pins
    that the name paints verbatim -- neither interpreted nor
    backslash-escaped.
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

        # Poll rather than trust a fixed pause: the reload this scope change
        # triggers now hops off the UI thread for its DB read (Qodo round 1,
        # FIX A, `asyncio.to_thread`), so under a busy full-suite run the
        # thread-pool dispatch can outlast a short fixed wait even though it
        # always finishes eventually.
        deadline = time.monotonic() + 10.0
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        while time.monotonic() < deadline:
            if pane.query_one("#artifacts-table", DataTable).row_count == 0:
                break
            await pilot.pause(0.05)

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
        await _press_generate(screen, pilot, app, watchlist_id)
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

        # Full centre width, asserted against a SIBLING section rather than
        # against this pane's own parent (fix round 1, Minor d): Sources is
        # one of the sections the spec says takes the whole centre column,
        # and it is measured here under the same stylesheet on the same
        # terminal. A reader stealing width from Artifacts, or a stray
        # `max-width` on the pane, would show up as a difference.
        artifacts_width = pane.region.width  # before the section switch
        screen.active_section = "sources"
        await pilot.pause(0.2)
        sources_width = screen.query_one("#watchlists-sources-pane").region.width
        assert artifacts_width == sources_width > size[0] // 2, (
            f"Artifacts is {artifacts_width} columns wide where Sources gets "
            f"{sources_width} on the same {size[0]}x{size[1]} terminal"
        )


# --- 6. Toolbar pickers: selection mode, default preset, Presets… (Task 4) -
#
# Tasks 1-3 built the writer (`set_watchlist_briefing_settings`), the
# preset table/CRUD (`list_briefing_presets`), and the manager modal
# (`BriefingPresetModal`); none of it had a way in from this screen.
# `briefing_selection_mode` had a READER since phase 1
# (`briefing_service._selection_mode`) but no writer anywhere in the UI, so
# `auto` and `curated` were unreachable -- this section is what retires
# that deferral.


def _capture_generate_calls(monkeypatch) -> list[dict]:
    """Fake `generate_briefing` that records its call kwargs and returns a
    minimal row.

    Unlike `_use_fake_chat` (which keeps the real service running, with
    only the provider call replaced), this section's tests are about what
    the SCREEN passes to `generate_briefing` -- specifically `preset_id` --
    not what the service does with it (Task 2's own suite already covers
    that). Bypassing the real service keeps these tests fast and focused on
    the one call-site argument in question.
    """
    calls: list[dict] = []

    async def _fake(db, watchlist_id, **kwargs):
        calls.append(kwargs)
        return {"id": 999}

    monkeypatch.setattr(screen_module, "generate_briefing", _fake)
    return calls


async def _press_generate_button_and_wait_for_a_call(
    screen, pilot, host, calls: list
) -> None:
    """Press Generate and wait for the whole `wl-briefing` worker to finish
    (real completion via `host.workers.wait_for_complete()`, not a
    wall-clock poll -- see this section's own note on why).
    """
    pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
    pane.query_one("#artifacts-generate-button", Button).press()
    await pilot.pause()
    await host.workers.wait_for_complete()
    await pilot.pause()


# Every wait in this section uses `host.workers.wait_for_complete()`
# (already established by `test_destination_shells.py`'s
# `test_mcp_destination_add_server_binding_opens_real_form_end_to_end`)
# rather than a wall-clock poll loop. `_load_briefings` now runs one more
# `to_thread` hop for the picker state on top of its existing zombie-sweep
# + list-read hops, and a `pilot.pause()`/deadline poll races real wall-clock
# time against however long the thread pool takes to be scheduled -- which
# a sufficiently busy machine (this repo's own test suite plus whatever
# else is running on the same host) can push past any fixed bound, MEASURED
# during this task's own verification (a 20s poll still missed, consistently,
# under heavy concurrent load). `wait_for_complete()` has no such bound: it
# awaits the dispatched worker(s) to actual completion, however long that
# takes, so it cannot flake on host speed the way a deadline can.


@pytest.mark.asyncio
async def test_toolbar_pickers_render_only_when_a_watchlist_is_in_scope():
    """The mode/preset `Select`s and the `Presets…` `Button` have nothing to
    act on without a single watchlist in scope, so -- unlike Generate/
    Refresh, which stay visible to explain themselves -- they do not render
    at all when `can_generate` is False.
    """
    app = _build_test_app()
    watchlist_id = _seed_watchlist(app)

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert pane.can_generate is True
        assert pane.query_one("#artifacts-mode-select", Select)
        assert pane.query_one("#artifacts-preset-select", Select)
        assert pane.query_one("#artifacts-presets-button", Button)

        screen.tree_scope = TreeScope(kind="all")
        await pilot.pause()
        await host.workers.wait_for_complete()
        await pilot.pause()

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert pane.can_generate is False
        assert not pane.query("#artifacts-mode-select")
        assert not pane.query("#artifacts-preset-select")
        assert not pane.query("#artifacts-presets-button")
        # Generate/Refresh, unlike the pickers, still explain themselves.
        assert pane.query_one("#artifacts-generate-button", Button)


@pytest.mark.asyncio
async def test_mode_select_shows_the_watchlists_stored_mode_on_load():
    """The read-path pin: Task 1's writer sets `curated` before Artifacts
    ever opens, and the mode Select must reflect it on the very first
    render -- not merely hold it in some screen-private field.
    """
    app = _build_test_app()
    watchlist_id = _seed_watchlist(app)
    db = app.watchlist_bundle_service.db
    db.set_watchlist_briefing_settings(watchlist_id, selection_mode="curated")

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        await host.workers.wait_for_complete()
        await pilot.pause()

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        select = pane.query_one("#artifacts-mode-select", Select)
        assert select.value == "curated"


@pytest.mark.asyncio
async def test_changing_mode_writes_off_loop_and_does_not_rebuild_the_screen():
    """Thread-identity pin (the established `asyncio.to_thread` pattern) plus
    the instance-survival assertion: a picker change must patch the pane in
    place, never rebuild it via `self.refresh(recompose=True)`.
    """
    app = _build_test_app()
    watchlist_id = _seed_watchlist(app)
    db = app.watchlist_bundle_service.db

    loop_thread_id = threading.get_ident()
    write_thread_ids: list[int] = []
    real_set = db.set_watchlist_briefing_settings

    def _spy(watchlist_id_arg, **kwargs):
        write_thread_ids.append(threading.get_ident())
        return real_set(watchlist_id_arg, **kwargs)

    db.set_watchlist_briefing_settings = _spy

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        await host.workers.wait_for_complete()
        pane_before = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        select = pane_before.query_one("#artifacts-mode-select", Select)
        # The fresh-watchlist default, confirmed by the read path above --
        # this is a genuine change, not a same-value mount-time no-op.
        assert select.value == "auto_featured"

        select.value = "curated"
        await pilot.pause()
        await host.workers.wait_for_complete()
        await pilot.pause()

        assert write_thread_ids, "set_watchlist_briefing_settings must have run"
        assert all(tid != loop_thread_id for tid in write_thread_ids), (
            "the write must run off the event-loop thread (asyncio.to_thread)"
        )

        row = db.conn.execute(
            "SELECT briefing_selection_mode FROM watchlists WHERE id = ?",
            (watchlist_id,),
        ).fetchone()
        assert row["briefing_selection_mode"] == "curated"

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert pane is pane_before, (
            "a picker change must repaint the pane, not rebuild the screen"
        )
        assert screen._briefing_selection_mode == "curated"


@pytest.mark.asyncio
async def test_preset_select_lists_presets_and_persists_a_choice():
    """The default-preset picker offers "App default" (`None`) plus every
    loaded preset, and choosing one persists `default_briefing_preset_id`.
    """
    app = _build_test_app()
    watchlist_id = _seed_watchlist(app)
    db = app.watchlist_bundle_service.db
    preset_id = db.insert_briefing_preset("Evening Digest", roster_json="[]")

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert screen._loaded_briefing_presets, "the presets read must have run"

        pane_before = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        preset_select = pane_before.query_one("#artifacts-preset-select", Select)

        # Nothing stored yet: "App default" is the active choice.
        assert preset_select.value is None

        # An id NOT among the loaded presets (nor `None`) is illegal --
        # proving the legal values are exactly what was loaded, not any
        # integer (same technique `BriefingPresetModal`'s own character/
        # voice Select tests use, Task 3).
        with pytest.raises(Exception):
            preset_select.value = preset_id + 999_999

        preset_select.value = preset_id
        await pilot.pause()
        await host.workers.wait_for_complete()
        await pilot.pause()

        row = db.conn.execute(
            "SELECT default_briefing_preset_id FROM watchlists WHERE id = ?",
            (watchlist_id,),
        ).fetchone()
        assert row["default_briefing_preset_id"] == preset_id
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert pane is pane_before, (
            "a picker change must repaint the pane, not rebuild the screen"
        )
        assert screen._briefing_default_preset_id == preset_id


@pytest.mark.asyncio
async def test_generate_casts_the_die_with_the_stored_default_preset(monkeypatch):
    """With a default preset stored, Generate invokes `generate_briefing`
    with `preset_id=<that id>`.
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    db = app.watchlist_bundle_service.db
    preset_id = db.insert_briefing_preset("Evening Digest", roster_json="[]")
    db.set_watchlist_briefing_settings(watchlist_id, default_preset_id=preset_id)
    calls = _capture_generate_calls(monkeypatch)

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert screen._briefing_default_preset_id == preset_id

        await _press_generate_button_and_wait_for_a_call(screen, pilot, host, calls)

    assert calls, "generate_briefing must have been invoked"
    assert calls[-1].get("preset_id") == preset_id


@pytest.mark.asyncio
async def test_generate_casts_the_die_with_no_default_preset(monkeypatch):
    """With no default preset stored, Generate invokes `generate_briefing`
    with `preset_id=None` -- the other half of the die-cast contract.
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    calls = _capture_generate_calls(monkeypatch)

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        assert screen._briefing_default_preset_id is None

        await _press_generate_button_and_wait_for_a_call(screen, pilot, host, calls)

    assert calls, "generate_briefing must have been invoked"
    assert calls[-1].get("preset_id") is None


@pytest.mark.asyncio
async def test_setting_curated_via_the_picker_then_generating_records_it_on_the_row(
    monkeypatch,
):
    """The phase-1 deferral's dead branch, made reachable end to end.

    `briefing_selection_mode` has had a READER since phase 1
    (`briefing_service._selection_mode`) but no writer anywhere in the UI
    until this task. Setting `curated` through the picker and then pressing
    Generate is the first time the two actually meet -- this is the test
    that retires the deferral.
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    chat = _FakeChat()
    _use_fake_chat(monkeypatch, chat)

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        select = pane.query_one("#artifacts-mode-select", Select)
        assert select.value == "auto_featured"

        select.value = "curated"
        await pilot.pause()
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert screen._briefing_selection_mode == "curated"

        await _press_generate(screen, pilot, app, watchlist_id)

    rows = _briefing_rows(app, watchlist_id)
    assert rows, "Generate must have written a briefing row"
    assert rows[0]["selection_mode"] == "curated"


@pytest.mark.asyncio
async def test_presets_button_opens_the_preset_manager(monkeypatch):
    """The toolbar's "Presets…" button calls Task 3's existing opener,
    `_open_briefing_preset_manager`, unchanged.
    """
    app = _build_test_app()
    watchlist_id = _seed_watchlist(app)

    calls: list[None] = []

    async def _recording_open(self):
        calls.append(None)

    monkeypatch.setattr(
        screen_module.WatchlistsCollectionsScreen,
        "_open_briefing_preset_manager",
        _recording_open,
    )

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        pane.query_one("#artifacts-presets-button", Button).press()
        await pilot.pause()
        await host.workers.wait_for_complete()
        await pilot.pause()

    assert calls, "the Presets… button must open the preset manager"


# --- 7. Casting a script (spec #2 phase 2a, Task 5) -------------------------
#
# Tasks 1-4 built the `briefing_scripts` table, the cast service
# (`generate_script`/`fail_interrupted_scripts`), and the picker toolbar this
# section's Cast button reads its default preset from; none of it had a way
# in from the screen. Same seam discipline as the Generate suite above: the
# only faked call is the chat provider, wrapped around the screen's own
# `generate_script` reference -- everything else (roster validation, prompt
# building, strict turn parsing, the snapshot, `fail_interrupted_scripts`)
# is the real service and a real `SubscriptionsDB`.


@pytest.mark.asyncio
async def test_casting_a_complete_briefing_writes_a_script_row_and_the_table_shows_it(
    monkeypatch,
):
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    _use_fake_chat(monkeypatch, _FakeChat())

    async with _open_artifacts(app, watchlist_id, visual=True) as (
        screen,
        pilot,
        _host,
    ):
        briefing_id = await _prepare_cast(screen, pilot, app, watchlist_id)

        cast_chat = _FakeChat(
            reply=json.dumps([{"speaker": "Narrator", "text": "Welcome."}])
        )
        _use_fake_cast_chat(monkeypatch, cast_chat)

        await _press_cast(screen, pilot, app, briefing_id)

        assert len(cast_chat.calls) == 1, "exactly one provider call per cast"
        rows = app.watchlist_bundle_service.db.list_briefing_scripts(briefing_id)
        assert [row["status"] for row in rows] == ["complete"]
        assert rows[0]["preset_name"] == "Solo"

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        table = pane.query_one("#artifacts-scripts-table", DataTable)
        assert table.row_count == 1
        painted = _painted(screen, table.region)
        assert "Solo" in painted
        assert "complete" in painted


@pytest.mark.asyncio
async def test_script_turns_render_as_speaker_labelled_text_never_markup(monkeypatch):
    """The mandatory literal-paint test: a turn containing `[bold red]x[/]`
    must paint as those literal characters -- never interpreted as Rich
    markup, and never escaped into visible backslashes either. Model/turn
    text goes through `rich.text.Text` exactly like a briefing body's
    `error`/status fields already do (`_detail_renderable`), never a markup
    parser.
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    _use_fake_chat(monkeypatch, _FakeChat())

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, _host):
        briefing_id = await _prepare_cast(screen, pilot, app, watchlist_id)

        hostile_turns = [{"speaker": "Narrator", "text": "[bold red]x[/]"}]
        _use_fake_cast_chat(monkeypatch, _FakeChat(reply=json.dumps(hostile_turns)))

        await _press_cast(screen, pilot, app, briefing_id)

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert pane.selected_script is not None, "the fixture must select a script"
        detail = pane.query_one("#artifacts-script-detail", Static)
        plain, ansi = _render_to_console(detail.renderable, width=100)

        assert "[bold red]x[/]" in plain, "the turn must paint exactly as written"
        assert "\\[" not in plain, "and must not grow escaping backslashes"
        assert "\x1b[1;31m" not in ansi, "and `[bold red]` must not be applied"
        # And the speaker label is really there, distinguishing this from a
        # render that merely dumped the whole JSON blob as text.
        assert "Narrator" in plain


@pytest.mark.asyncio
async def test_more_than_200_turns_are_capped_with_an_honest_count(monkeypatch):
    """Spec ethos: never a silent truncation. A script with more than 200
    turns shows the first 200 plus a stated "…N more turns" line.
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    _use_fake_chat(monkeypatch, _FakeChat())

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, _host):
        briefing_id = await _prepare_cast(screen, pilot, app, watchlist_id)

        many_turns = [
            {"speaker": "Narrator", "text": f"Line {index}."} for index in range(210)
        ]
        _use_fake_cast_chat(monkeypatch, _FakeChat(reply=json.dumps(many_turns)))

        await _press_cast(screen, pilot, app, briefing_id)

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        detail = pane.query_one("#artifacts-script-detail", Static)
        plain, _ansi = _render_to_console(detail.renderable, width=100)

        assert "Line 199." in plain, "the 200th turn (index 199) must be shown"
        assert "Line 200." not in plain, "the 201st turn must NOT be shown"
        assert "10 more turns" in plain, "the overflow must be stated, not silent"


@pytest.mark.asyncio
async def test_casting_a_non_complete_briefing_refuses_naming_the_status():
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    db = app.watchlist_bundle_service.db
    briefing_id = db.insert_briefing(watchlist_id)
    db.update_briefing(briefing_id, status="failed", error="boom")
    preset_id = db.insert_briefing_preset(
        "Solo", roster_json=dump_roster(ONE_SPEAKER_ROSTER)
    )
    db.set_watchlist_briefing_settings(watchlist_id, default_preset_id=preset_id)

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        await host.workers.wait_for_complete()
        await pilot.pause()
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        pane.select_briefing_by_id(str(briefing_id))
        await pilot.pause()

        await _press_cast(screen, pilot, app, briefing_id)

        assert app.notify.called, "a refusal must be visible, not silent"
        args, kwargs = app.notify.call_args
        message = args[0] if args else str(kwargs.get("message", ""))
        assert "failed" in message, "the toast must name the briefing's actual status"
        assert kwargs.get("markup") is False
        assert db.list_briefing_scripts(briefing_id) == [], (
            "a pre-flight refusal must never write a row"
        )


@pytest.mark.asyncio
async def test_second_cast_while_in_flight_refuses_naming_the_running_one():
    """Sibling of `test_the_refusal_toast_names_the_watchlist_actually_
    generating`: `_cast_in_flight` is screen-global, so the refusal must
    name which briefing is actually being cast rather than assume it is
    whichever one is on screen right now.
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    db = app.watchlist_bundle_service.db
    briefing_id = db.insert_briefing(watchlist_id)
    db.update_briefing(briefing_id, status="complete", body_markdown="Body")

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, _host):
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        pane.select_briefing_by_id(str(briefing_id))
        await pilot.pause()

        screen._cast_in_flight = True
        screen._cast_in_flight_briefing_id = briefing_id
        try:
            screen.handle_cast_script_requested(CastScriptRequested())
        finally:
            screen._cast_in_flight = False
            screen._cast_in_flight_briefing_id = None

        assert app.notify.called
        args, kwargs = app.notify.call_args
        message = args[0] if args else str(kwargs.get("message", ""))
        assert str(briefing_id) in message
        assert kwargs.get("markup") is False


@pytest.mark.asyncio
async def test_the_cast_guard_is_claimed_before_the_worker_runs(monkeypatch):
    """Mechanism half, the deterministic sibling of `test_the_guard_is_
    claimed_before_the_worker_runs`: the handler is synchronous with no
    `await`, so when it returns, no worker code can yet have run. If the
    guard is claimed there, `_cast_in_flight` is already True at that
    instant; if it is claimed inside the worker body instead, it is still
    False -- this is Step 5 mutation (a)'s target.
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    _use_fake_chat(monkeypatch, _FakeChat())

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, _host):
        briefing_id = await _prepare_cast(screen, pilot, app, watchlist_id)
        cast_chat = _FakeChat(
            reply=json.dumps([{"speaker": "Narrator", "text": "Hi."}])
        )
        _use_fake_cast_chat(monkeypatch, cast_chat)

        screen.handle_cast_script_requested(CastScriptRequested())

        assert screen._cast_in_flight is True, (
            "the guard must be claimed by the handler, before `run_worker` "
            "has scheduled anything"
        )
        assert cast_chat.calls == [], "and no worker code can have run yet"

        app.notify.reset_mock()
        screen.handle_cast_script_requested(CastScriptRequested())
        assert app.notify.call_count == 1
        _args, kwargs = app.notify.call_args
        assert kwargs.get("markup") is False

        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and screen._cast_in_flight:
            await pilot.pause(0.02)
        assert len(cast_chat.calls) == 1, "exactly one cast must have run"


@pytest.mark.asyncio
async def test_a_zombie_generating_script_is_recovered_on_a_plain_artifacts_load(
    monkeypatch,
):
    """Load-path seam: `_load_briefings` sweeps a crashed cast worker's
    `generating` script row, exactly like `test_a_zombie_generating_row_
    is_recovered_on_a_plain_artifacts_load` does for briefings. The
    recorder proves this is the LOAD path's own sweep (`_cast_in_flight`
    clear at call time), not the Cast worker's -- the flag-at-call-time
    lesson carried from the phase-1 zombie test.
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    _use_fake_chat(monkeypatch, _FakeChat())

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, _host):
        briefing_id = await _prepare_cast(screen, pilot, app, watchlist_id)
        db = app.watchlist_bundle_service.db
        preset_id = db.list_briefing_presets()[0]["id"]
        zombie_id = db.insert_briefing_script(
            briefing_id,
            preset_id=preset_id,
            preset_name="Solo",
            roster_snapshot_json=dump_roster(ONE_SPEAKER_ROSTER),
        )

        in_flight_at_call: list[bool] = []
        real_sweep = screen_module.fail_interrupted_scripts

        def _recording_sweep(db_arg, briefing_id_arg=None):
            in_flight_at_call.append(bool(screen._cast_in_flight))
            return real_sweep(db_arg, briefing_id_arg)

        monkeypatch.setattr(screen_module, "fail_interrupted_scripts", _recording_sweep)

        await screen._load_briefings()

        assert in_flight_at_call, "the load path's own sweep must have run at all"
        assert all(not flag for flag in in_flight_at_call), (
            "the load path's sweep must run with `_cast_in_flight` CLEAR -- a "
            "call recorded True could only be the Cast worker's own sweep"
        )
        rows = db.list_briefing_scripts(briefing_id)
        by_id = {row["id"]: row for row in rows}
        assert by_id[zombie_id]["status"] == "failed"
        assert by_id[zombie_id]["error"] == "interrupted"

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        pane.select_script_by_id(str(zombie_id))
        await pilot.pause()
        plain, _ansi = _render_to_console(
            pane.query_one("#artifacts-script-detail", Static).renderable
        )
        assert "interrupted" in plain
        assert "This script is being written now." not in plain


@pytest.mark.asyncio
async def test_casting_recovers_a_zombie_script_via_its_own_sweep(monkeypatch):
    """Cast-path seam, pinned SEPARATELY from the load-path test above: the
    Cast worker sweeps `fail_interrupted_scripts` at its own front, exactly
    where `_sweep_and_guard` runs for Generate. The recorder proves THIS
    call carries `_cast_in_flight` claimed (True) -- only the Cast worker's
    own sweep call can, since the load path's sweep is gated on the flag
    being clear. Unlike briefings, recovering the zombie does not itself
    refuse this attempt: `briefing_scripts` has no one-generating-row-
    per-briefing invariant (a briefing may be cast many times), so the SAME
    press both recovers the zombie AND casts a real script.
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    _use_fake_chat(monkeypatch, _FakeChat())

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, _host):
        briefing_id = await _prepare_cast(screen, pilot, app, watchlist_id)
        db = app.watchlist_bundle_service.db
        preset_id = db.list_briefing_presets()[0]["id"]
        zombie_id = db.insert_briefing_script(
            briefing_id,
            preset_id=preset_id,
            preset_name="Solo",
            roster_snapshot_json=dump_roster(ONE_SPEAKER_ROSTER),
        )

        in_flight_at_call: list[bool] = []
        real_sweep = screen_module.fail_interrupted_scripts

        def _recording_sweep(db_arg, briefing_id_arg=None):
            in_flight_at_call.append(bool(screen._cast_in_flight))
            return real_sweep(db_arg, briefing_id_arg)

        monkeypatch.setattr(screen_module, "fail_interrupted_scripts", _recording_sweep)

        cast_chat = _FakeChat(
            reply=json.dumps([{"speaker": "Narrator", "text": "Hi."}])
        )
        _use_fake_cast_chat(monkeypatch, cast_chat)

        await _press_cast(screen, pilot, app, briefing_id)

        assert True in in_flight_at_call, (
            "the zombie must be recovered by the Cast worker's OWN sweep "
            "(a call with `_cast_in_flight` claimed), not merely by the "
            "load-path recovery that runs after the flag clears"
        )
        rows = db.list_briefing_scripts(briefing_id)
        by_id = {row["id"]: row for row in rows}
        assert by_id[zombie_id]["status"] == "failed"
        assert by_id[zombie_id]["error"] == "interrupted"
        assert any(
            row["status"] == "complete" for row in rows if row["id"] != zombie_id
        ), "the same press must also have cast a real script"


@pytest.mark.asyncio
async def test_a_failed_script_renders_its_error_string(monkeypatch):
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    _use_fake_chat(monkeypatch, _FakeChat())

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, _host):
        briefing_id = await _prepare_cast(screen, pilot, app, watchlist_id)
        _use_fake_cast_chat(monkeypatch, _FakeChat(reply="not json at all"))

        await _press_cast(screen, pilot, app, briefing_id)

        rows = app.watchlist_bundle_service.db.list_briefing_scripts(briefing_id)
        assert rows[0]["status"] == "failed"

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert pane.selected_script is not None, "the fresh cast must be selected"
        assert pane.selected_script["id"] == rows[0]["id"]
        plain, _ansi = _render_to_console(
            pane.query_one("#artifacts-script-detail", Static).renderable
        )
        assert rows[0]["error"] in plain


@pytest.mark.asyncio
async def test_a_failed_cast_leaves_the_briefing_detail_unchanged(monkeypatch):
    """Spec §Error-handling ethos: a script's outcome never touches the
    briefing it was cast from -- asserted byte-for-byte, and by what is
    still painted in the briefing's own detail area.
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    _use_fake_chat(monkeypatch, _FakeChat())

    async with _open_artifacts(app, watchlist_id, visual=True) as (
        screen,
        pilot,
        _host,
    ):
        briefing_id = await _prepare_cast(screen, pilot, app, watchlist_id)
        db = app.watchlist_bundle_service.db
        before = dict(db.get_briefing(briefing_id))

        _use_fake_cast_chat(monkeypatch, _FakeChat(reply="not json at all"))
        await _press_cast(screen, pilot, app, briefing_id)

        rows = db.list_briefing_scripts(briefing_id)
        assert rows and rows[0]["status"] == "failed"

        after = dict(db.get_briefing(briefing_id))
        assert before == after, "casting a script must never touch the briefing row"

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        detail = pane.query_one("#artifacts-detail", Static)
        plain, _ansi = _render_to_console(detail.renderable, width=detail.region.width)
        assert "Acme shipped a thing" in plain


@pytest.mark.asyncio
async def test_cast_is_disabled_until_a_preset_exists():
    """`Cast` starts disabled with a tooltip when there is no default
    preset AND no preset exists at all to pick one from; it enables the
    moment any preset exists.
    """
    app = _build_test_app()
    watchlist_id = _seed_watchlist(app)
    db = app.watchlist_bundle_service.db
    briefing_id = db.insert_briefing(watchlist_id)
    db.update_briefing(briefing_id, status="complete", body_markdown="Body")

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, _host):
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        pane.select_briefing_by_id(str(briefing_id))
        await pilot.pause()

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        cast_button = pane.query_one("#artifacts-cast-button", Button)
        assert cast_button.disabled is True
        assert cast_button.tooltip

        db.insert_briefing_preset("Solo", roster_json=dump_roster(ONE_SPEAKER_ROSTER))
        await screen._load_briefings()
        await pilot.pause()

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        cast_button = pane.query_one("#artifacts-cast-button", Button)
        assert cast_button.disabled is False
