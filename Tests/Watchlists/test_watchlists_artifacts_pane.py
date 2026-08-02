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

import asyncio
import json
import re
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from rich.console import Console
from rich.text import Text
from textual.coordinate import Coordinate
from textual.widgets import Button, DataTable, Select, Static

from Tests.UI.test_destination_shells import DestinationHarness, _static_text
from Tests.UI.test_destination_visual_parity_correction import (
    _visual_destination_harness,
)
from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.Subscriptions import briefing_cast, briefing_service
import tldw_chatbook.Subscriptions.briefing_audio as briefing_audio
from tldw_chatbook.Subscriptions.briefing_audio import AudioGenerationError
from tldw_chatbook.Subscriptions.briefing_cast import dump_roster
from tldw_chatbook.Subscriptions.briefing_export import default_briefing_filename
from tldw_chatbook.Subscriptions.item_persist import persist_subscription_item
from tldw_chatbook.Third_Party.textual_fspicker import FileSave, SelectDirectory
from tldw_chatbook.TTS import audio_player as audio_player_module
from tldw_chatbook.UI.Screens import watchlists_collections_screen as screen_module
from tldw_chatbook.UI.Screens.watchlists_collections_screen import (
    WatchlistsCollectionsScreen,
)
from tldw_chatbook.UI.Watchlists_Modules.artifacts_pane import (
    ArtifactsPane,
    BriefingSelected,
    CastScriptRequested,
    CitationActivated,
    ExportBriefingRequested,
    ExportFeedRequested,
    GenerateBriefingRequested,
    PlayAudioRequested,
    StopAudioRequested,
    SynthesizeAudioRequested,
    _audio_file_is_playable,
    audio_file_path_is_safe,
)
from tldw_chatbook.UI.Watchlists_Modules.content_pane import ContentPane
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


def _seeded_item_rows(app) -> list[sqlite3.Row]:
    """The real `subscription_items` rows `_seed_watchlist` just wrote, id
    ASC -- so a citation test can cite an id the database actually has,
    rather than a number invented in the test itself.
    """
    db = app.watchlist_bundle_service.db
    return list(
        db.conn.execute("SELECT id, title FROM subscription_items ORDER BY id")
    )


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

        def _recording_fail(db, watchlist_id=None, *, exclude=()):
            in_flight_at_call.append(bool(screen._briefing_in_flight))
            return real_fail(db, watchlist_id, exclude=exclude)

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


@pytest.mark.asyncio
async def test_a_claimed_watchlist_survives_an_artifacts_open():
    """Phase 4 Task 1, survey finding (a): the sibling of the test above,
    but for a LIVE claim this screen instance did NOT take -- standing in
    for a scheduled run once phase 4's scheduler exists. Claimed directly
    via the service (`briefing_service._claim_briefing`), not through
    `_briefing_in_flight`: that flag is this screen's own dispatch-time UX
    guard and is deliberately untouched by this scenario, since nothing on
    screen is dispatching anything.
    """
    app = _build_test_app()
    watchlist_id = _seed_watchlist(app)
    db = app.watchlist_bundle_service.db

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, _host):
        live_id = db.insert_briefing(watchlist_id)
        assert not screen._briefing_in_flight, (
            "this scenario is a claim with no screen dispatch behind it"
        )
        with briefing_service._claim_briefing(watchlist_id):
            await screen._load_briefings()
            assert db.get_briefing(live_id)["status"] == "generating", (
                "a claimed watchlist must survive an Artifacts open"
            )

        # Once the claim releases, a plain load recovers it normally --
        # the claim never outlives the process (Locked decision 1).
        await screen._load_briefings()
        assert db.get_briefing(live_id)["status"] == "failed"
        assert db.get_briefing(live_id)["error"] == "interrupted"


@pytest.mark.asyncio
async def test_generate_during_a_claimed_watchlist_refuses_without_falsifying_the_row(
    monkeypatch,
):
    """Phase 4 Task 1, survey finding (b): pressing Generate while another
    in-process caller holds this watchlist's claim must hit the EXISTING
    `blocking` refusal -- not silently mark the live row interrupted and
    start a second generation over the top of it.

    Asserts the SPECIFIC `blocking` toast (`severity="warning"`, "already
    in progress"), not merely "some refusal happened": `generate_briefing`
    itself also refuses a claimed watchlist (`GenerationInFlightError`), so
    a looser assertion would still pass with the screen's OWN `blocking`
    check deleted entirely, as long as the worker went on to call
    `generate_briefing` and hit ITS claim collision instead -- a different,
    generic-error-toast path this test must tell apart from the one it
    names.
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    chat = _FakeChat()
    _use_fake_chat(monkeypatch, chat)
    db = app.watchlist_bundle_service.db

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, _host):
        live_id = db.insert_briefing(watchlist_id)
        with briefing_service._claim_briefing(watchlist_id):
            await _press_generate(screen, pilot, app, watchlist_id)

        assert chat.calls == [], "nothing may be generated while claimed elsewhere"
        assert app.notify.called, "the refusal must be visible, not silent"
        args, kwargs = app.notify.call_args
        message = args[0] if args else str(kwargs.get("message", ""))
        assert kwargs.get("severity") == "warning"
        assert kwargs.get("markup") is False
        assert "already in progress" in message
        assert db.get_briefing(live_id)["status"] == "generating", (
            "the live claim's row must not be falsified as interrupted"
        )


@pytest.mark.asyncio
async def test_generate_in_flight_race_toasts_the_specific_message_not_a_db_error(
    monkeypatch,
):
    """Whole-branch review FIX 2: `_sweep_and_guard` only sees a claim once
    its row lands in the database. If another in-process caller claims the
    watchlist AFTER the sweep reads (finding no `generating` row yet, so
    `blocking` stays empty) but BEFORE it inserts, this attempt proceeds
    into `generate_briefing`'s own claim check and raises
    `GenerationInFlightError` -- a specific, contracted, user-safe message
    (`str(exc)` names the watchlist, per the class's own docstring). The
    bare `except Exception` used to swallow this as "the watchlist
    database could not be reached", which is both untrue (nothing is
    unreachable -- a race was lost) and unhelpful (it tells the user
    nothing about what to do, whereas the real message says a generation
    is already running).
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    db = app.watchlist_bundle_service.db

    async def _raise_in_flight(db_arg, watchlist_id_arg, **kwargs):
        raise briefing_service.GenerationInFlightError(
            f"a briefing is already being generated for watchlist {watchlist_id_arg}"
        )

    monkeypatch.setattr(screen_module, "generate_briefing", _raise_in_flight)

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        rows_before = len(db.list_briefings(watchlist_id))
        await _press_generate(screen, pilot, app, watchlist_id)

        assert host.is_running, "the lost race must not exit the application"
        assert app.notify.called, "the race must be visible, not silent"
        args, kwargs = app.notify.call_args
        message = args[0] if args else str(kwargs.get("message", ""))
        assert kwargs.get("severity") == "warning"
        assert kwargs.get("markup") is False
        assert "already being generated" in message
        assert "could not be reached" not in message, (
            "must not fall through to the generic database-unreachable toast"
        )
        assert len(db.list_briefings(watchlist_id)) == rows_before, (
            "GenerationInFlightError fires before any row insert (the "
            "phase-1 no-orphan-row contract) -- no failed-row side effect"
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
    """The mode/preset/cadence `Select`s and the `Presets…` `Button` have
    nothing to act on without a single watchlist in scope, so -- unlike
    Generate/Refresh, which stay visible to explain themselves -- they do
    not render at all when `can_generate` is False.

    Also pins the cadence picker's default: a fresh watchlist has never had
    `briefing_cadence_seconds` written, so the Select must show "Off"
    (`None`) -- the same fallback `ArtifactsPane.briefing_cadence_seconds`
    itself defaults to.
    """
    app = _build_test_app()
    watchlist_id = _seed_watchlist(app)

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert pane.can_generate is True
        assert pane.query_one("#artifacts-mode-select", Select)
        assert pane.query_one("#artifacts-preset-select", Select)
        cadence_select = pane.query_one("#artifacts-cadence-select", Select)
        assert cadence_select.value is None, "a fresh watchlist defaults to Off"
        assert pane.query_one("#artifacts-presets-button", Button)

        screen.tree_scope = TreeScope(kind="all")
        await pilot.pause()
        await host.workers.wait_for_complete()
        await pilot.pause()

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert pane.can_generate is False
        assert not pane.query("#artifacts-mode-select")
        assert not pane.query("#artifacts-preset-select")
        assert not pane.query("#artifacts-cadence-select")
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


# --- Whole-branch review fix wave, Important #3 -----------------------------
#
# `_write_briefing_selection_mode`/`_write_briefing_default_preset` patch
# screen memory (and the mounted pane's matching reactive) after their own
# `await`, without checking the screen is still scoped to the SAME
# watchlist the write was dispatched for. `handle_generate_briefing_
# requested` reads `_briefing_default_preset_id` at its own dispatch time,
# so a stale write's completion landing after a scope change could hand a
# Generate press for a DIFFERENT watchlist the wrong preset.


@pytest.mark.asyncio
async def test_switching_watchlists_mid_write_does_not_let_the_stale_write_clobber_the_new_one():
    """Deterministic control over exactly when `set_watchlist_briefing_
    settings` resolves comes from a `threading.Event` the fake write blocks
    on (Task 3's own controllable-seam pattern, not a sleep/poll race):
    pick a default preset for watchlist A (the write blocks), switch
    Artifacts to watchlist B while it is still in flight (B's own settings
    load for real -- a different, unblocked read path), release A's write,
    and confirm the screen -- now scoped to B -- keeps B's own default
    preset rather than being clobbered by A's write landing late. The
    write itself is unaffected: A's own row really does end up holding
    A's chosen preset, even though the screen never reflects it.
    """
    app = _build_test_app()
    watchlist_a = _seed_watchlist(app)
    watchlist_b = app.watchlist_bundle_service.create("Security Watch")["id"]
    db = app.watchlist_bundle_service.db
    preset_a = db.insert_briefing_preset("For A", roster_json="[]")
    preset_b = db.insert_briefing_preset("For B", roster_json="[]")
    db.set_watchlist_briefing_settings(watchlist_b, default_preset_id=preset_b)

    async with _open_artifacts(app, watchlist_a) as (screen, pilot, host):
        await host.workers.wait_for_complete()
        await pilot.pause()

        release_write = threading.Event()
        call_started = threading.Event()
        real_set = db.set_watchlist_briefing_settings

        def _blocking_set(watchlist_id_arg, **kwargs):
            call_started.set()
            assert release_write.wait(timeout=5), "test setup: write never released"
            return real_set(watchlist_id_arg, **kwargs)

        db.set_watchlist_briefing_settings = _blocking_set

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        preset_select = pane.query_one("#artifacts-preset-select", Select)
        assert preset_select.value is None, "watchlist A has no default yet"

        # Pick a default preset for A -- this write blocks.
        preset_select.value = preset_a
        await pilot.pause()

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not call_started.is_set():
            await pilot.pause(0.02)
        assert call_started.is_set(), "the write must have started"

        # While A's write is still blocked, switch Artifacts to B.
        screen.tree_scope = TreeScope(kind="watchlist", watchlist_id=watchlist_b)
        await pilot.pause()

        deadline = time.monotonic() + 5.0
        while (
            time.monotonic() < deadline
            and screen._briefing_default_preset_id != preset_b
        ):
            await pilot.pause(0.02)
        assert screen._briefing_default_preset_id == preset_b, (
            "switching scope must load B's own settings"
        )

        # NOW release A's write and let it finish.
        release_write.set()
        await host.workers.wait_for_complete()
        await pilot.pause()

        # Still scoped to B: A's stale completion must not have clobbered
        # B's own in-memory state.
        assert screen._briefing_default_preset_id == preset_b
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert pane.default_preset_id == preset_b

    # The DB write itself is correctly keyed and needed no change: A's own
    # row really did get preset_a, even though the screen never reflected it.
    row = db.conn.execute(
        "SELECT default_briefing_preset_id FROM watchlists WHERE id = ?",
        (watchlist_a,),
    ).fetchone()
    assert row["default_briefing_preset_id"] == preset_a


# --- 6b. Scheduled-briefing cadence picker (spec #2 phase 4, Task 4) --------
#
# Tasks 1-3 of phase 4 built the in-process claims, the `briefing_cadence_
# seconds` column/writer, and the scheduler seam that reads it back through
# `list_briefing_schedules`; none of it had a way in from this screen, and
# the scope note above the toolbar still said "on request" unconditionally
# -- a lie the moment a cadence could be stored at all. This section is what
# retires both gaps: the third picker mirrors the mode/preset pickers'
# established shape (`_briefing_picker_mount_absorbed` instance-keyed
# absorption, `asyncio.to_thread` write, in-place pane patch, no screen
# recompose) exactly, and the scope label test pins the honesty fix.


@pytest.mark.asyncio
async def test_cadence_select_shows_the_watchlists_stored_cadence_on_load():
    """The read-path pin: Task 2's writer sets a cadence before Artifacts
    ever opens, and the cadence Select must reflect it on the very first
    render -- not merely hold it in some screen-private field.
    """
    app = _build_test_app()
    watchlist_id = _seed_watchlist(app)
    db = app.watchlist_bundle_service.db
    db.set_watchlist_briefing_settings(watchlist_id, briefing_cadence_seconds=43_200)

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        await host.workers.wait_for_complete()
        await pilot.pause()

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        select = pane.query_one("#artifacts-cadence-select", Select)
        assert select.value == 43_200


@pytest.mark.asyncio
async def test_changing_cadence_writes_off_loop_and_does_not_rebuild_the_screen():
    """Thread-identity pin (the established `asyncio.to_thread` pattern) plus
    the instance-survival assertion: a picker change must patch the pane in
    place, never rebuild it via `self.refresh(recompose=True)`. Mirrors
    `test_changing_mode_writes_off_loop_and_does_not_rebuild_the_screen`
    exactly, and -- like that test -- drives the REAL mounted `Select`'s
    `value` setter (not a hand-set `pane.briefing_cadence_seconds`), so the
    real `Select.Changed` -> `BriefingCadenceChanged` -> screen-handler
    chain is what is actually under test.
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
        select = pane_before.query_one("#artifacts-cadence-select", Select)
        # The fresh-watchlist default, confirmed by the read path above --
        # this is a genuine change, not a same-value mount-time no-op.
        assert select.value is None

        select.value = 86_400  # "Daily"
        await pilot.pause()
        await host.workers.wait_for_complete()
        await pilot.pause()

        assert write_thread_ids, "set_watchlist_briefing_settings must have run"
        assert all(tid != loop_thread_id for tid in write_thread_ids), (
            "the write must run off the event-loop thread (asyncio.to_thread)"
        )

        row = db.conn.execute(
            "SELECT briefing_cadence_seconds FROM watchlists WHERE id = ?",
            (watchlist_id,),
        ).fetchone()
        assert row["briefing_cadence_seconds"] == 86_400

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert pane is pane_before, (
            "a picker change must repaint the pane, not rebuild the screen"
        )
        assert screen._briefing_cadence_seconds == 86_400
        assert pane.briefing_cadence_seconds == 86_400


@pytest.mark.asyncio
async def test_choosing_off_clears_the_stored_cadence():
    """`Off` maps to `None`, and picking it must clear `briefing_cadence_
    seconds` back to NULL -- the DB column's own "never scheduled" state
    (`set_watchlist_briefing_settings`'s `_UNSET`-sentinel shape: passing
    `None` explicitly clears, distinct from not passing the kwarg at all).
    """
    app = _build_test_app()
    watchlist_id = _seed_watchlist(app)
    db = app.watchlist_bundle_service.db
    db.set_watchlist_briefing_settings(watchlist_id, briefing_cadence_seconds=604_800)

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        await host.workers.wait_for_complete()
        await pilot.pause()

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        select = pane.query_one("#artifacts-cadence-select", Select)
        assert select.value == 604_800, "the seeded weekly cadence must load first"

        select.value = None  # "Off"
        await pilot.pause()
        await host.workers.wait_for_complete()
        await pilot.pause()

        row = db.conn.execute(
            "SELECT briefing_cadence_seconds FROM watchlists WHERE id = ?",
            (watchlist_id,),
        ).fetchone()
        assert row["briefing_cadence_seconds"] is None
        assert screen._briefing_cadence_seconds is None
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert pane.briefing_cadence_seconds is None


@pytest.mark.asyncio
async def test_the_scope_label_states_on_request_or_the_actual_schedule_honestly():
    """The honesty fix this task exists to ship: `_briefing_scope_label`
    used to always say "on request", which stopped being true the moment a
    cadence could be stored at all. Both directions, against the SAME
    watchlist, driven through a real picker change (not a hand-set reactive
    or a direct call to the private label method) -- and the label updates
    in place, without waiting for another full `_load_briefings` reload.
    """
    app = _build_test_app()
    watchlist_id = _seed_watchlist(app)

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        await host.workers.wait_for_complete()
        await pilot.pause()

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert "on request" in pane.scope_label
        assert "scheduled" not in pane.scope_label

        select = pane.query_one("#artifacts-cadence-select", Select)
        select.value = 43_200  # "Every 12h"
        await pilot.pause()
        await host.workers.wait_for_complete()
        await pilot.pause()

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert "scheduled every 12h while the app is open" in pane.scope_label
        assert "on request" not in pane.scope_label


@pytest.mark.asyncio
async def test_an_out_of_catalog_cadence_gets_a_synthetic_select_option_and_an_honest_scope_label():
    """Review round 1, Important #1: the out-of-catalog fallbacks are
    DB-reachable TODAY, not theoretical -- `set_watchlist_briefing_settings`
    (Task 2) validates only `> 0`, never catalog membership, and Task 2's
    own test suite uses `briefing_cadence_seconds=3600` as its standard
    fixture. `3600` is not one of `_CADENCE_OPTIONS` (`Off`/43200/86400/
    604800), so both of this pane's own defensive fallbacks are exercised
    by a value the writer accepts right now, through no path more exotic
    than Task 2's writer itself.

    Mirrors the stale-preset-id fallback test (`test_casting_refuses_
    before_dispatch_when_the_default_preset_is_dangling`, `_preset_select_
    options`'s own precedent) for the Select half, and the mode/preset-
    picker honesty test above for the scope-label half -- both fallbacks,
    seeded once, asserted together.
    """
    app = _build_test_app()
    watchlist_id = _seed_watchlist(app)
    db = app.watchlist_bundle_service.db
    db.set_watchlist_briefing_settings(watchlist_id, briefing_cadence_seconds=3_600)

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        await host.workers.wait_for_complete()
        await pilot.pause()

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)

        # Fallback 1: the Select still renders, holding the real stored
        # value via a synthetic trailing option -- never `InvalidSelect
        # ValueError` for a value this picker never offered.
        cadence_select = pane.query_one("#artifacts-cadence-select", Select)
        option_labels = {value: str(label) for label, value in cadence_select._options}
        assert option_labels[3_600] == "Every 3600s"
        assert cadence_select.value == 3_600

        # Fallback 2: the scope label stays honest -- generic wording, but
        # still names the real cadence and still carries the "while the
        # app is open" promise verbatim, not a silent "on request" lie.
        assert "scheduled every 3600s while the app is open" in pane.scope_label
        assert "on request" not in pane.scope_label


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
async def test_casting_with_presets_but_no_default_refuses_with_actionable_copy(
    monkeypatch,
):
    """Fix round 1, ruling 2: Cast stays ENABLED when presets exist but none
    is chosen as the watchlist's default (`ArtifactsPane`'s disabled
    condition is "no default AND no presets at all" -- presets exist here,
    just none picked). Pressing it in this state must still be refused, but
    with copy that tells the user what to do -- not `generate_script`'s own
    raw `ScriptCastError` text for `preset_id=None`
    ("briefing preset None does not exist"), which names nothing the user
    can act on.
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    _use_fake_chat(monkeypatch, _FakeChat())

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, _host):
        await _press_generate(screen, pilot, app, watchlist_id)
        briefing_id = _briefing_rows(app, watchlist_id)[0]["id"]
        db = app.watchlist_bundle_service.db
        db.insert_briefing_preset("Solo", roster_json=dump_roster(ONE_SPEAKER_ROSTER))
        # Deliberately NOT set as the watchlist's default preset.
        await screen._load_briefings()
        await pilot.pause()
        assert screen._briefing_default_preset_id is None
        assert screen._loaded_briefing_presets, "the fixture needs a real preset"

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        cast_button = pane.query_one("#artifacts-cast-button", Button)
        assert cast_button.disabled is False, "a preset exists, so Cast stays enabled"

        cast_button.press()
        await pilot.pause()

        assert app.notify.called, "a refusal must be visible, not silent"
        args, kwargs = app.notify.call_args
        message = args[0] if args else str(kwargs.get("message", ""))
        assert "default preset" in message.lower(), (
            "the toast must tell the user to choose or create a default preset"
        )
        assert "does not exist" not in message, (
            "must not be generate_script's raw, unactionable ScriptCastError text"
        )
        assert kwargs.get("markup") is False
        assert db.list_briefing_scripts(briefing_id) == [], (
            "this refusal must never reach the service at all"
        )


@pytest.mark.asyncio
async def test_casting_refuses_before_dispatch_when_the_default_preset_is_dangling(
    monkeypatch,
):
    """Whole-branch review fix wave, Important #1.

    A default preset can be hard-deleted (`BriefingPresetModal`, Task 3 --
    no FK enforces the pointer) while it is still a watchlist's stored
    default: `_load_briefings`'s combined read re-reads the watchlist's own
    `default_briefing_preset_id` column verbatim, but reloads the preset
    LIST fresh, so the dangling id survives a reload even though it no
    longer names a real row. Before this fix, pressing Cast in that state
    fell through to `generate_script`'s own raw `ScriptCastError` text
    ("briefing preset <id> does not exist") -- honest, but naming nothing
    the user can act on -- while the toolbar's own preset picker was
    already showing "Preset <id> (deleted)" for the exact same id. This
    test also pins that Select surface, closing the Task 4 report's own
    parked test gap (concern 4).
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    _use_fake_chat(monkeypatch, _FakeChat())

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, _host):
        await _press_generate(screen, pilot, app, watchlist_id)
        briefing_id = _briefing_rows(app, watchlist_id)[0]["id"]
        db = app.watchlist_bundle_service.db
        preset_id = db.insert_briefing_preset(
            "Solo", roster_json=dump_roster(ONE_SPEAKER_ROSTER)
        )
        db.set_watchlist_briefing_settings(watchlist_id, default_preset_id=preset_id)
        await screen._load_briefings()
        await pilot.pause()
        assert screen._briefing_default_preset_id == preset_id

        # Hard-delete the preset (the modal's own path; no FK stops this),
        # then reload exactly as a plain Artifacts refresh would.
        assert db.delete_briefing_preset(preset_id) is True
        await screen._load_briefings()
        await pilot.pause()

        # The dangling id survives the reload; only the loaded preset LIST
        # drops the row.
        assert screen._briefing_default_preset_id == preset_id
        assert all(
            preset.get("id") != preset_id
            for preset in screen._loaded_briefing_presets
        )

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        preset_select = pane.query_one("#artifacts-preset-select", Select)
        option_labels = {value: str(label) for label, value in preset_select._options}
        assert option_labels[preset_id] == f"Preset {preset_id} (deleted)"
        assert preset_select.value == preset_id

        cast_button = pane.query_one("#artifacts-cast-button", Button)
        assert cast_button.disabled is False, "presets is non-empty; Cast stays enabled"

        cast_button.press()
        await pilot.pause()

        assert app.notify.called, "a refusal must be visible, not silent"
        args, kwargs = app.notify.call_args
        message = args[0] if args else str(kwargs.get("message", ""))
        assert "no longer exists" in message.lower()
        assert "does not exist" not in message, (
            "must not be generate_script's raw, unactionable ScriptCastError text"
        )
        assert kwargs.get("markup") is False
        assert db.list_briefing_scripts(briefing_id) == [], (
            "this refusal must never reach the service at all"
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
async def test_a_cast_press_during_a_claimed_briefing_refuses_not_run_concurrently(
    monkeypatch,
):
    """Phase 4 Task 1, survey finding (c): before this fix, Cast had NO
    refusal at all for this case (`watchlists_collections_screen.py`'s own
    comment above `_cast_sweep_is_safe` used to document the absence
    explicitly) -- a press during a genuinely in-flight cast for the SAME
    briefing would start a second, concurrent one. Claimed directly via the
    service (`briefing_cast._claim_cast`), standing in for another
    in-process caster; `_cast_in_flight` is deliberately untouched, since
    this is not the SAME screen instance's own dispatch-time guard being
    exercised (that is `test_second_cast_while_in_flight_refuses_naming_
    the_running_one`, above).

    Asserts the SPECIFIC `blocking` toast (`severity="warning"`, "already
    being cast"), not merely "some refusal happened": `generate_script`
    itself also refuses a claimed briefing (`GenerationInFlightError`), so
    a looser assertion would still pass with the screen's OWN `blocking`
    check deleted entirely, as long as the worker went on to call
    `generate_script` and hit ITS claim collision instead -- a different,
    generic-error-toast path this test must tell apart from the one it
    names (this is what pins mutation (iii)).
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    _use_fake_chat(monkeypatch, _FakeChat())

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, _host):
        briefing_id = await _prepare_cast(screen, pilot, app, watchlist_id)
        db = app.watchlist_bundle_service.db
        live_script_id = db.insert_briefing_script(
            briefing_id, preset_id=None, preset_name="Solo", roster_snapshot_json="[]"
        )
        cast_chat = _FakeChat(
            reply=json.dumps([{"speaker": "Narrator", "text": "Hi."}])
        )
        _use_fake_cast_chat(monkeypatch, cast_chat)

        with briefing_cast._claim_cast(briefing_id):
            await _press_cast(screen, pilot, app, briefing_id)

        assert cast_chat.calls == [], "nothing may be cast while claimed elsewhere"
        assert app.notify.called, "the refusal must be visible, not silent"
        args, kwargs = app.notify.call_args
        message = args[0] if args else str(kwargs.get("message", ""))
        assert kwargs.get("severity") == "warning"
        assert kwargs.get("markup") is False
        assert "already being cast" in message
        assert db.get_briefing_script(live_script_id)["status"] == "generating", (
            "the live claim's row must not be falsified as interrupted"
        )


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

        def _recording_sweep(db_arg, briefing_id_arg=None, *, exclude=()):
            in_flight_at_call.append(bool(screen._cast_in_flight))
            return real_sweep(db_arg, briefing_id_arg, exclude=exclude)

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

        def _recording_sweep(db_arg, briefing_id_arg=None, *, exclude=()):
            in_flight_at_call.append(bool(screen._cast_in_flight))
            return real_sweep(db_arg, briefing_id_arg, exclude=exclude)

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


@pytest.mark.asyncio
async def test_the_briefings_table_keeps_at_least_three_usable_rows(monkeypatch):
    """Fix round 1, ruling 1: no existing test pinned the briefings table's
    USABLE height -- `test_the_list_the_button_and_the_body_are_all_on_
    screen` only asserts `region.height > 0`. Re-weighting the pane's `fr`
    split to 2:6:1:1 (this task's own CSS fix -- see `_watchlists.tcss`)
    trades the briefings list's share down in favour of its own body and
    the new scripts section; pinned here with BOTH a briefing and a script
    actually present, since the scripts section's own fixed rows are part
    of what squeezes the briefings table down toward its floor. `height >=
    4` is a header row plus at least 3 data rows -- not merely "some rows".
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
        _use_fake_cast_chat(
            monkeypatch, _FakeChat(reply=json.dumps([{"speaker": "Narrator", "text": "Hi."}]))
        )
        await _press_cast(screen, pilot, app, briefing_id)

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        table = pane.query_one("#artifacts-table", DataTable)
        assert table.region.height >= 4, (
            f"the briefings table has only {table.region.height} row(s) of "
            "height -- not enough for a header plus 3 usable data rows"
        )


@pytest.mark.asyncio
async def test_switching_the_selected_briefing_clears_stale_scripts_before_the_reload_lands(
    monkeypatch,
):
    """Fix round 1, minor: a briefing row click must not show the PREVIOUS
    briefing's scripts under the NEW selection even for one frame.
    `handle_briefing_selected` re-dispatches `_load_briefings()` to fetch
    the newly selected briefing's own scripts, but that reload is
    asynchronous -- without clearing the pane's `scripts`/`selected_script`
    reactives SYNCHRONOUSLY at click time, the old scripts would still be
    on screen (attached to the wrong briefing) until the worker lands.

    `handle_briefing_selected` is called DIRECTLY here, exactly like
    `test_the_cast_guard_is_claimed_before_the_worker_runs` calls
    `handle_cast_script_requested` directly: the handler has no `await` in
    it, so checking pane state immediately after it returns -- with NO
    `pilot.pause()` at all -- pins the clearing as truly synchronous, not
    merely "fast enough to usually win a race". A version of this test
    that went through the real click path plus one `pilot.pause()` was
    measured to be VACUOUS: `_load_briefings`'s own `asyncio.to_thread`
    hops finished within that single pause often enough that the assertion
    passed for the wrong reason (the reload had already landed) even with
    the clearing code deleted entirely.
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    _use_fake_chat(monkeypatch, _FakeChat())

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        # First briefing, with a real cast script attached to it.
        first_id = await _prepare_cast(screen, pilot, app, watchlist_id)
        _use_fake_cast_chat(
            monkeypatch, _FakeChat(reply=json.dumps([{"speaker": "Narrator", "text": "Hi."}]))
        )
        await _press_cast(screen, pilot, app, first_id)
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert pane.scripts, "the fixture needs the first briefing to have a script"
        assert pane.selected_script is not None

        # A second, scriptless briefing for the same watchlist.
        db = app.watchlist_bundle_service.db
        second_id = db.insert_briefing(watchlist_id)
        db.update_briefing(second_id, status="complete", body_markdown="Second body")
        await screen._load_briefings()
        await pilot.pause()
        await host.workers.wait_for_complete()
        await pilot.pause()
        second_row = next(
            row for row in _briefing_rows(app, watchlist_id) if row["id"] == second_id
        )

        # The handler is synchronous (no `await`), so state checked
        # IMMEDIATELY after it returns -- before `run_worker` has let the
        # reload do anything at all -- proves the clearing itself, not
        # merely that it finishes "soon".
        screen.handle_briefing_selected(BriefingSelected(second_row))

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert pane.selected_script is None, (
            "the stale script selection must clear synchronously, before "
            "the reload worker is even dispatched"
        )
        assert pane.scripts == [], (
            "the stale scripts list must clear synchronously, before the "
            "reload worker is even dispatched"
        )

        # And once the reload actually lands, the SECOND briefing's (empty)
        # scripts are what's shown -- not a stale carry-over.
        await pilot.pause()
        await host.workers.wait_for_complete()
        await pilot.pause()
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert pane.scripts == []


# --- Task 6: citations into the reader + pruned degradation --------------
#
# Retires the phase-1 "citations" deferral: a briefing body's `[item N]`
# markers (`briefing_service.build_briefing_prompt`'s own convention) become
# navigable. `extract_citation_ids`'s own ordering/dedup/ignore-non-numeric
# behaviour is pure and tested directly in `Tests/Subscriptions/
# test_briefing_service.py`; these tests are about the RESOLUTION (the
# screen's `_load_briefings`, via `get_subscription_items_by_ids`) and
# ACTIVATION (`handle_citation_activated`) built on top of it.
#
# `pane.activate_citation_by_id` is called directly rather than fabricating
# a `DataTable` row-selection event -- the same directness this file's
# existing tests already give `select_briefing_by_id`/`select_script_by_id`
# -- but it is still a REAL call on the REAL mounted pane, so it posts a
# REAL `CitationActivated` through the REAL message pump into the REAL
# screen handler; nothing about the screen-side wiring is faked.


@pytest.mark.asyncio
async def test_a_complete_briefings_citations_table_lists_each_cited_id_with_its_title(
    monkeypatch,
):
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app, items=2)
    cited = _seeded_item_rows(app)[0]
    _use_fake_chat(
        monkeypatch,
        _FakeChat(reply=f"## This week\n\n{cited['title']} happened [item {cited['id']}].\n"),
    )

    async with _open_artifacts(app, watchlist_id, visual=True) as (screen, pilot, _host):
        await _press_generate(screen, pilot, app, watchlist_id)
        # `_press_generate` only waits for the briefings TABLE's row count to
        # agree with the database -- but setting `pane.selected_briefing`
        # (inside the generation worker's own `_load_briefings` call) fires
        # `ArtifactsPane.watch_selected_briefing`, which posts
        # `BriefingSelected`, which `handle_briefing_selected` answers by
        # clearing `pane.citations` and dispatching a SECOND, separate
        # `_load_briefings()` -- the identical cascade `_prepare_cast` below
        # already settles with this same extra direct call before reading
        # pane state. Measured directly (a debug harness printing every
        # `_load_briefings` call's own citations): without this, `pane.
        # citations` is observed empty here more often than not, mid-cascade.
        await screen._load_briefings()
        await pilot.pause()

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert pane.citations == [
            {
                "item_id": cited["id"],
                "label": pane.citations[0]["label"],
                "available": True,
            }
        ], "exactly one citation, for the one id the body actually cites"
        label = pane.citations[0]["label"]
        assert isinstance(label, Text), "a title is remote text -- never a bare str"
        assert cited["title"] in label.plain
        assert str(cited["id"]) in label.plain

        table = pane.query_one("#artifacts-citations-table", DataTable)
        assert table.row_count == 1
        painted = _painted(screen, table.region)
        assert cited["title"] in painted
        assert "Available" in painted


@pytest.mark.asyncio
async def test_a_citation_to_a_pruned_item_degrades(monkeypatch):
    """The plan's second named invariant: **citation-to-pruned-item-
    degrades**. A `[item N]` id that does not resolve to a live
    `subscription_items` row (deleted, or simply never existed) must
    degrade honestly -- both in what the citations table shows, and in what
    activating it does. It must never be silently treated as available.
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app, items=1)
    pruned_id = max(row["id"] for row in _seeded_item_rows(app)) + 1000
    _use_fake_chat(
        monkeypatch,
        _FakeChat(reply=f"## This week\n\nSomething happened [item {pruned_id}].\n"),
    )

    async with _open_artifacts(app, watchlist_id, visual=True) as (screen, pilot, _host):
        await _press_generate(screen, pilot, app, watchlist_id)
        # Settle the `BriefingSelected` reload cascade -- see the comment on
        # the identical call in `test_a_complete_briefings_citations_table_
        # lists_each_cited_id_with_its_title` above.
        await screen._load_briefings()
        await pilot.pause()

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert len(pane.citations) == 1
        citation = pane.citations[0]
        assert citation["item_id"] == pruned_id
        assert citation["available"] is False
        assert isinstance(citation["label"], Text)
        assert citation["label"].plain == f"item {pruned_id} — no longer available"

        table = pane.query_one("#artifacts-citations-table", DataTable)
        assert table.row_count == 1
        painted = _painted(screen, table.region)
        assert "no longer available" in painted
        assert "Not available" in painted

        # Activating it: a toast, markup=False, and -- the invariant's other
        # half -- NO section switch. Section-switching is the mechanism a
        # future edit could most plausibly get backwards (treat "pruned" as
        # "switch anyway, then discover there's nothing to show"), so both
        # halves are pinned in the same test.
        notes_before = app.notify.call_count
        section_before = screen.active_section
        pane.activate_citation_by_id(str(pruned_id))
        await pilot.pause(0.2)

        assert screen.active_section == section_before, (
            "a pruned citation must not switch sections"
        )
        assert not screen.query("#wl-region-content"), (
            "a pruned citation must not mount the reader either"
        )
        assert app.notify.call_count > notes_before, "a pruned citation must toast"
        toast_call = app.notify.call_args
        assert toast_call.kwargs.get("markup") is False
        assert str(pruned_id) in str(toast_call.args[0])


@pytest.mark.asyncio
async def test_activating_an_available_citation_opens_it_in_the_reader_and_marks_it_read(
    monkeypatch,
):
    """Activating a resolving citation is an OPEN (design ruling -- do not
    relitigate): it switches to the Items ("Read") section, the reader
    shows that exact item, and -- the same side effect a real click on the
    Items table already has -- the item's status flips from `new` to
    `reviewed`. Pinned here so a future "why did my item get marked read"
    question has this path, not just a mouse click, to find.
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app, items=2)
    db = app.watchlist_bundle_service.db
    # `_build_test_app`'s `patch("tldw_chatbook.app.get_subscriptions_db_path",
    # ...)` is only active while `TldwCli()` itself is constructing (see its
    # nested `with` blocks) -- long enough for the EAGERLY-built
    # `watchlist_bundle_service.db` to see it, but `LocalWatchlistsService`'s
    # own `db_factory` is a lambda that re-resolves `get_subscriptions_db_
    # path()` LAZILY, on its first real call, which happens well after that
    # patch has already been undone -- so, in this harness only,
    # `app.local_watchlists_service._db()` falls through to the real,
    # unpatched path instead of this test's isolated one (confirmed directly:
    # a debug probe printed the two `db_path`s and they differ). That is the
    # SAME database `_mark_item_read_on_open`'s write actually reaches
    # (`_controller` -> `WatchlistScopeService` -> `LocalWatchlistsService.
    # update_item` -> `self._db()`), so without this redirect the write
    # would target a database this test's seeded item was never in. This is
    # a test-harness-only artifact (in the real app both resolve to the same
    # configured path), not something Task 6 introduces, so it is patched
    # here rather than in the shared `_build_test_app`/`_seed_watchlist`
    # helpers, which every other test in this file already relies on as-is.
    monkeypatch.setattr(app.local_watchlists_service, "db_factory", lambda: db)
    cited = _seeded_item_rows(app)[0]
    assert db.get_item_status(cited["id"]) == "new", "fixture precondition"
    _use_fake_chat(
        monkeypatch,
        _FakeChat(reply=f"## This week\n\n{cited['title']} happened [item {cited['id']}].\n"),
    )

    async with _open_artifacts(app, watchlist_id, visual=True) as (screen, pilot, _host):
        await _press_generate(screen, pilot, app, watchlist_id)
        # Settle the `BriefingSelected` reload cascade -- see the comment on
        # the identical call in `test_a_complete_briefings_citations_table_
        # lists_each_cited_id_with_its_title` above.
        await screen._load_briefings()
        await pilot.pause()

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        citation = next(c for c in pane.citations if c["item_id"] == cited["id"])
        assert citation["available"] is True

        assert screen.active_section == "artifacts"
        pane.activate_citation_by_id(str(cited["id"]))
        # Real wall-clock delay: `handle_citation_activated` defers opening
        # the item by a `set_timer(0.05, ...)` (the section switch it makes
        # first is not visible to a query until the NEXT recompose), and a
        # bare `pilot.pause()` waits for CPU idle, not wall-clock time, so
        # it would not reliably let that timer fire.
        await pilot.pause(0.3)

        assert screen.active_section == "items", (
            "an available citation must switch to the Read tab"
        )
        content_pane = screen.query_one("#watchlists-content-pane", ContentPane)
        for _ in range(60):
            await pilot.pause()
            if content_pane.item is not None:
                break
        assert content_pane.item is not None
        # `content_pane.item["id"]` is `normalize_watchlist_item`'s own
        # NAMESPACED id (`"local:watchlist_item:<n>"`), not the bare
        # `[item N]` id the body cited -- `item_id` is that same
        # normalization's bare-id field, and asserting on it (plus the
        # title) is the unambiguous way to confirm this is really the cited
        # item, not merely "the reader now shows *an* item".
        assert content_pane.item["item_id"] == cited["id"]
        assert content_pane.item["title"] == cited["title"]

        for _ in range(60):
            await pilot.pause()
            if db.get_item_status(cited["id"]) == "reviewed":
                break
        assert db.get_item_status(cited["id"]) == "reviewed", (
            "opening a cited item through the citations table must mark it "
            "read, exactly like opening it via a click in the Items table"
        )


@pytest.mark.asyncio
async def test_keyboard_browsing_the_citations_table_does_not_activate_a_row(monkeypatch):
    """Review fix round 1 (Important), confirmed live by the reviewer:
    focusing the citations table and pressing an arrow key must not
    activate anything. `highlight_is_user_driven` (`table_selection.py`)
    filters rebuild-echo noise from a real event -- it does NOT distinguish
    a click from keyboard cursor movement, so if the citations table were
    routed through `RowHighlighted` the same way briefings/scripts still
    are, a single `down` press would switch sections and mark an item read
    on every step of merely BROWSING the list, with no confirmation at all.
    This drives the REAL `DataTable` (focus + `pilot.press("down")`), not
    `pane.activate_citation_by_id`, so it is `on_data_table_row_
    highlighted`'s own citations-table no-op that is under test.
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app, items=2)
    rows = _seeded_item_rows(app)
    body = (
        "## This week\n\n"
        f"{rows[0]['title']} happened [item {rows[0]['id']}]. "
        f"{rows[1]['title']} happened [item {rows[1]['id']}].\n"
    )
    _use_fake_chat(monkeypatch, _FakeChat(reply=body))

    async with _open_artifacts(app, watchlist_id, visual=True) as (screen, pilot, _host):
        await _press_generate(screen, pilot, app, watchlist_id)
        # Settle the `BriefingSelected` reload cascade -- see the comment on
        # the identical call in `test_a_complete_briefings_citations_table_
        # lists_each_cited_id_with_its_title` above.
        await screen._load_briefings()
        await pilot.pause()

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert len(pane.citations) == 2, (
            "the fixture needs two citations to browse between"
        )

        table = pane.query_one("#artifacts-citations-table", DataTable)
        table.focus()
        await pilot.pause(0.2)
        assert table.cursor_row == 0

        notes_before = app.notify.call_count
        await pilot.press("down")
        for _ in range(30):
            await pilot.pause()

        assert table.cursor_row == 1, (
            "the cursor must still move -- browsing itself is not blocked"
        )
        assert screen.active_section == "artifacts", (
            "arrow-key browsing of the citations table must not switch sections"
        )
        assert app.notify.call_count == notes_before, (
            "arrow-key browsing must not toast either (no pruned-citation refusal)"
        )
        db = app.watchlist_bundle_service.db
        assert db.get_item_status(rows[0]["id"]) == "new"
        assert db.get_item_status(rows[1]["id"]) == "new"


@pytest.mark.asyncio
async def test_pressing_enter_on_a_citation_activates_it_through_the_real_table(
    monkeypatch,
):
    """Review fix round 1: closes the gap where every citation test up to
    this point called `pane.activate_citation_by_id` directly, so nothing
    exercised a real `DataTable` input event. Drives the table for real --
    focus, cursor already on the (only) row, `Enter` -- so `on_data_table_
    row_selected`'s own citations-table branch is what is under test.
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app, items=2)
    db = app.watchlist_bundle_service.db
    # See the identical redirect + comment in
    # `test_activating_an_available_citation_opens_it_in_the_reader_and_
    # marks_it_read` above: `local_watchlists_service`'s lazy `db_factory`
    # resolves the real, unpatched db path in this harness, which is a
    # different database than the one this test seeds and asserts against.
    monkeypatch.setattr(app.local_watchlists_service, "db_factory", lambda: db)
    cited = _seeded_item_rows(app)[0]
    _use_fake_chat(
        monkeypatch,
        _FakeChat(reply=f"## This week\n\n{cited['title']} happened [item {cited['id']}].\n"),
    )

    async with _open_artifacts(app, watchlist_id, visual=True) as (screen, pilot, _host):
        await _press_generate(screen, pilot, app, watchlist_id)
        await screen._load_briefings()
        await pilot.pause()

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        table = pane.query_one("#artifacts-citations-table", DataTable)
        table.focus()
        await pilot.pause(0.2)
        assert table.cursor_row == 0

        await pilot.press("enter")
        # Real wall-clock delay for `handle_citation_activated`'s own
        # `set_timer(0.05, ...)` -- see the identical comment above.
        await pilot.pause(0.3)

        assert screen.active_section == "items", (
            "Enter on a citation row must activate it -- switching to the "
            "Read tab, exactly like a direct call to activate_citation_by_id"
        )
        content_pane = screen.query_one("#watchlists-content-pane", ContentPane)
        for _ in range(60):
            await pilot.pause()
            if content_pane.item is not None:
                break
        assert content_pane.item is not None
        assert content_pane.item["item_id"] == cited["id"]

        for _ in range(60):
            await pilot.pause()
            if db.get_item_status(cited["id"]) == "reviewed":
                break
        assert db.get_item_status(cited["id"]) == "reviewed"


@pytest.mark.asyncio
async def test_citations_do_not_shrink_the_briefings_table_below_its_pinned_minimum(
    monkeypatch,
):
    """Task 5's `test_the_briefings_table_keeps_at_least_three_usable_rows`
    already pins `#artifacts-table`'s floor with a script section present;
    this is the same pin with a citations table ALSO present -- the
    scenario the brief specifically warns about ("the citations table must
    not steal below the pinned minimums"). Every OTHER new test in this
    file uses `_seed_watchlist`'s default single-item body with no
    `[item N]` marker, so this is the only one that puts all three sections
    (briefing list, scripts, citations) on screen at once.
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    cited = _seeded_item_rows(app)[0]
    _use_fake_chat(
        monkeypatch,
        _FakeChat(reply=f"## This week\n\n{cited['title']} happened [item {cited['id']}].\n"),
    )

    async with _open_artifacts(app, watchlist_id, visual=True) as (
        screen,
        pilot,
        _host,
    ):
        briefing_id = await _prepare_cast(screen, pilot, app, watchlist_id)
        _use_fake_cast_chat(
            monkeypatch, _FakeChat(reply=json.dumps([{"speaker": "Narrator", "text": "Hi."}]))
        )
        await _press_cast(screen, pilot, app, briefing_id)

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert pane.citations, "the fixture must actually produce a citation"
        table = pane.query_one("#artifacts-table", DataTable)
        assert table.region.height >= 4, (
            f"adding the citations table shrank the briefings table to "
            f"{table.region.height} row(s) -- the pinned floor from Task 5 "
            "must survive a citations table sharing the same budget"
        )


# --- Task 7: synthesizing and playing a script's audio -------------------
#
# Unlike Cast (Task 5), which fakes only the chat call so the REAL cast
# service runs underneath, these tests fake `generate_script_audio` itself
# at the screen's own reference -- the brief's own instruction, mirroring
# `_use_fake_cast_chat`'s seam-choice but one level deeper: the real
# pipeline needs a real TTS service, a real profile service, and real
# per-turn synthesis (Tasks 4-6), none of which a UI-focused test should
# need to stand up just to press a button and read a status back. A
# `complete` script row is built directly via `insert_briefing_script`
# rather than through a real Cast pass, for the identical reason
# `test_casting_a_non_complete_briefing_refuses_naming_the_status` builds
# its briefing row directly -- everything these tests exercise is
# downstream of that row already existing.


class _FakeAudioService:
    """The one faked seam: a stand-in for `briefing_audio.generate_script_
    audio`, called exactly as the screen calls the real thing
    (`db, script_id, *, tts_service, profile_service`).

    Writes a REAL `briefing_audio` row through the REAL DB methods (Task
    1's CRUD), so a test asserting against `db.list_briefing_audio` sees
    exactly what the real pipeline would leave behind -- only the
    synthesis/stitching/voice-resolution machinery in between is skipped.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._status = "complete"
        self._file_path: str | None = None
        self._duration: float | None = 12.3
        self._turn_count: int | None = 1
        self._error: str | None = None
        self._raise: AudioGenerationError | None = None

    def set_next(
        self,
        *,
        status: str = "complete",
        file_path: str | None = None,
        duration: float | None = 12.3,
        turn_count: int | None = 1,
        error: str | None = None,
    ) -> None:
        self._status = status
        self._file_path = file_path
        self._duration = duration
        self._turn_count = turn_count
        self._error = error

    def raise_next(self, message: str) -> None:
        self._raise = AudioGenerationError(message)

    async def __call__(self, db, script_id, *, tts_service, profile_service):
        self.calls.append(
            {
                "script_id": script_id,
                "tts_service": tts_service,
                "profile_service": profile_service,
            }
        )
        if self._raise is not None:
            exc, self._raise = self._raise, None
            raise exc
        audio_id = db.create_briefing_audio(script_id, voice_snapshot_json="[]")
        if self._status == "complete":
            db.update_briefing_audio(
                audio_id,
                status="complete",
                file_path=self._file_path,
                duration_seconds=self._duration,
                turn_count=self._turn_count,
            )
        else:
            db.update_briefing_audio(
                audio_id, status=self._status, error=self._error or "synthesis failed"
            )
        return db.get_briefing_audio(audio_id)


def _use_fake_audio_service(monkeypatch, fake) -> None:
    """Fake `generate_script_audio` at the screen's own reference, exactly
    like `_use_fake_cast_chat` fakes `generate_script` one level up.
    """
    monkeypatch.setattr(screen_module, "generate_script_audio", fake)


class _FakePlayer:
    """A stand-in for `SimpleAudioPlayer`, the two methods `handle_play_
    audio_requested`/`handle_stop_audio_requested` actually call.
    """

    def __init__(self) -> None:
        self.stopped = False
        self._current: Path | None = None

    def get_current_file(self) -> Path | None:
        return self._current

    def set_current(self, path: Path) -> None:
        self._current = path

    def stop(self) -> None:
        self.stopped = True
        self._current = None


def _patch_audio_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Redirect `briefing_audio_dir()` into `tmp_path` (Qodo review round 1,
    FIX B): `_audio_file_is_playable`/`handle_play_audio_requested` now
    validate a row's `file_path` against `briefing_audio_dir()`, which
    calls `briefing_audio.get_user_data_dir()` -- a DIFFERENT name binding
    than `tldw_chatbook.app.get_user_data_dir` (which `_build_test_app`
    already patches), so tests that need a file to validate as safely
    "inside" the audio dir must patch this one too, mirroring `Tests/
    Subscriptions/test_briefing_audio_pipeline.py`'s own `_patch_user_data_
    dir` helper exactly.
    """
    monkeypatch.setattr(briefing_audio, "get_user_data_dir", lambda: tmp_path)


def _seed_complete_script(app, watchlist_id, *, roster=None) -> tuple[int, int]:
    """A `complete` briefing and a `complete` cast script, built directly
    via the DB rather than a real Generate+Cast pass -- these tests fake
    `generate_script_audio` entirely, so nothing downstream of the script
    ROW needs to be real either (unlike Task 5's `_prepare_cast`, whose
    tests exercise the real cast service).

    Returns:
        `(briefing_id, script_id)`.
    """
    db = app.watchlist_bundle_service.db
    briefing_id = db.insert_briefing(watchlist_id)
    db.update_briefing(briefing_id, status="complete", body_markdown="Body")
    script_id = db.insert_briefing_script(
        briefing_id,
        preset_id=None,
        preset_name="Solo",
        roster_snapshot_json=dump_roster(roster or ONE_SPEAKER_ROSTER),
        status="complete",
    )
    return briefing_id, script_id


async def _select_briefing_and_script(screen, pilot, host, briefing_id, script_id) -> None:
    """Select a briefing then its script through the real pane, waiting
    for both of the `_load_briefings` reloads either selection dispatches
    (`handle_briefing_selected`/`handle_script_selected`) to actually land.
    """
    pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
    pane.select_briefing_by_id(str(briefing_id))
    await pilot.pause()
    await host.workers.wait_for_complete()
    await pilot.pause()
    pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
    pane.select_script_by_id(str(script_id))
    await pilot.pause()
    await host.workers.wait_for_complete()
    await pilot.pause()


async def _press_synthesize(screen, pilot, app, script_id, *, timeout: float = 20.0):
    """Press the real Synthesize button and wait until the press is
    answered. Mirrors `_press_cast` exactly, scoped to one script's audio
    rather than one briefing's scripts -- there is no dedicated audio
    TABLE to wait on (the audio state is folded into the script detail
    Static), so the final settle condition is the audio row count/a toast,
    same observable-state discipline as every sibling `_press_*` helper.
    """
    db = app.watchlist_bundle_service.db
    rows_before = len(db.list_briefing_audio(script_id))
    notes_before = getattr(app.notify, "call_count", 0)

    pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
    pane.query_one("#artifacts-synthesize-button", Button).press()

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await pilot.pause(0.02)
        if (
            screen._audio_in_flight
            or getattr(app.notify, "call_count", 0) > notes_before
            or len(db.list_briefing_audio(script_id)) != rows_before
        ):
            break
    while time.monotonic() < deadline and screen._audio_in_flight:
        await pilot.pause(0.02)
    while time.monotonic() < deadline:
        await pilot.pause(0.02)
        if (
            len(db.list_briefing_audio(script_id)) != rows_before
            or getattr(app.notify, "call_count", 0) > notes_before
        ):
            return


@pytest.mark.asyncio
async def test_synthesizing_a_complete_script_writes_an_audio_row_and_the_detail_shows_it(
    monkeypatch, tmp_path
):
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    briefing_id, script_id = _seed_complete_script(app, watchlist_id)

    audio_file = tmp_path / "clip.wav"
    audio_file.write_bytes(b"RIFF....WAVEfmt ")
    fake_audio = _FakeAudioService()
    fake_audio.set_next(file_path=str(audio_file), duration=12.3, turn_count=1)
    _use_fake_audio_service(monkeypatch, fake_audio)

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        await _select_briefing_and_script(screen, pilot, host, briefing_id, script_id)

        await _press_synthesize(screen, pilot, app, script_id)

        assert len(fake_audio.calls) == 1, "exactly one synthesis call per press"
        assert fake_audio.calls[0]["script_id"] == script_id

        db = app.watchlist_bundle_service.db
        rows = db.list_briefing_audio(script_id)
        assert [row["status"] for row in rows] == ["complete"]

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert pane.script_audio is not None
        assert pane.script_audio["id"] == rows[0]["id"]
        plain, _ansi = _render_to_console(
            pane.query_one("#artifacts-script-detail", Static).renderable
        )
        assert "complete" in plain
        assert "12.3s" in plain


@pytest.mark.asyncio
async def test_synthesizing_a_non_complete_script_refuses_naming_the_status():
    """Deliberately does NOT fake `generate_script_audio`: the REAL
    function's own pre-flight refusal (`_load_script_for_audio`) raises
    before ever touching `tts_service`/`profile_service`, exactly mirroring
    why `test_casting_a_non_complete_briefing_refuses_naming_the_status`
    uses the real `generate_script` for the identical reason.

    Seeded directly as `failed`, not `generating`: `_load_briefings`'s OWN
    (pre-existing, Task 5) zombie-script sweep would otherwise flip a
    `generating` row to `failed`/`interrupted` the moment `_select_
    briefing_and_script` re-dispatches its reload -- a real, separate
    behaviour, just the wrong one for THIS test to trip over by accident.
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    db = app.watchlist_bundle_service.db
    briefing_id = db.insert_briefing(watchlist_id)
    db.update_briefing(briefing_id, status="complete", body_markdown="Body")
    script_id = db.insert_briefing_script(
        briefing_id,
        preset_id=None,
        preset_name="Solo",
        roster_snapshot_json=dump_roster(ONE_SPEAKER_ROSTER),
        status="failed",
    )

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        await _select_briefing_and_script(screen, pilot, host, briefing_id, script_id)

        await _press_synthesize(screen, pilot, app, script_id)

        assert app.notify.called, "a refusal must be visible, not silent"
        args, kwargs = app.notify.call_args
        message = args[0] if args else str(kwargs.get("message", ""))
        assert "failed" in message, (
            "the toast must name the script's actual status"
        )
        assert kwargs.get("markup") is False
        assert db.list_briefing_audio(script_id) == [], (
            "a pre-flight refusal must never write a row"
        )


@pytest.mark.asyncio
async def test_second_synthesis_while_in_flight_refuses_naming_the_running_one():
    """Sibling of `test_second_cast_while_in_flight_refuses_naming_the_
    running_one`: `_audio_in_flight` is screen-global, so the refusal must
    name which script is actually being synthesized.
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    briefing_id, script_id = _seed_complete_script(app, watchlist_id)

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        await _select_briefing_and_script(screen, pilot, host, briefing_id, script_id)

        screen._audio_in_flight = True
        screen._audio_in_flight_script_id = script_id
        try:
            screen.handle_synthesize_audio_requested(SynthesizeAudioRequested())
        finally:
            screen._audio_in_flight = False
            screen._audio_in_flight_script_id = None

        assert app.notify.called
        args, kwargs = app.notify.call_args
        message = args[0] if args else str(kwargs.get("message", ""))
        assert str(script_id) in message
        assert kwargs.get("markup") is False


@pytest.mark.asyncio
async def test_the_audio_guard_is_claimed_before_the_worker_runs(monkeypatch):
    """Mechanism half, the deterministic sibling of `test_the_cast_guard_
    is_claimed_before_the_worker_runs`: the handler is synchronous with no
    `await`, so when it returns, no worker code can yet have run. If the
    guard is claimed there, `_audio_in_flight` is already True at that
    instant; if it is claimed inside the worker body instead, it is still
    False -- this is Step 5 mutation (a)'s target.
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    briefing_id, script_id = _seed_complete_script(app, watchlist_id)

    fake_audio = _FakeAudioService()
    _use_fake_audio_service(monkeypatch, fake_audio)

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        await _select_briefing_and_script(screen, pilot, host, briefing_id, script_id)

        screen.handle_synthesize_audio_requested(SynthesizeAudioRequested())

        assert screen._audio_in_flight is True, (
            "the guard must be claimed by the handler, before `run_worker` "
            "has scheduled anything"
        )
        assert fake_audio.calls == [], "and no worker code can have run yet"

        app.notify.reset_mock()
        screen.handle_synthesize_audio_requested(SynthesizeAudioRequested())
        assert app.notify.call_count == 1
        _args, kwargs = app.notify.call_args
        assert kwargs.get("markup") is False

        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and screen._audio_in_flight:
            await pilot.pause(0.02)
        assert len(fake_audio.calls) == 1, "exactly one synthesis must have run"


@pytest.mark.asyncio
async def test_a_database_error_during_synthesis_does_not_exit_the_app(monkeypatch):
    """Sibling of `test_a_database_error_during_generation_does_not_exit_
    the_app`: `generate_script_audio` deliberately lets database errors
    propagate (Task 6's own docstring), and an exception escaping a
    Textual worker with the default `exit_on_error=True` takes the whole
    application down with it -- proven live for Generate in phase 1, not
    theoretical. `_synthesize_audio`'s bare `except Exception` around the
    call is the guard; this drives a REAL raise through the faked seam and
    asserts the app is still standing, not merely that a toast appeared
    (review round 1, Important #1).
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    briefing_id, script_id = _seed_complete_script(app, watchlist_id)
    db = app.watchlist_bundle_service.db

    async def _explode(db_arg, script_id_arg, *, tts_service, profile_service):
        # `generate_script_audio` turns synthesis/voice-resolution
        # failures into a `failed` row but deliberately lets database
        # errors propagate -- see its own docstring.
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(screen_module, "generate_script_audio", _explode)

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        await _select_briefing_and_script(screen, pilot, host, briefing_id, script_id)

        await _press_synthesize(screen, pilot, app, script_id)

        assert host.is_running, "a worker failure must not exit the application"
        assert host.screen_stack[-1] is screen, "the screen must still be standing"
        assert screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert app.notify.called, "a failure the user asked for must be reported"
        _args, kwargs = app.notify.call_args
        assert kwargs.get("severity") == "error"
        assert kwargs.get("markup") is False
        assert screen._audio_in_flight is False, (
            "the in-flight guard must clear even when synthesis raises"
        )
        assert db.list_briefing_audio(script_id) == [], (
            "a pre-flight database error must not leave a `generating` "
            "row behind"
        )

        # The guard is genuinely re-armed. Asserted on the SERVICE, not on
        # "some toast happened" (mirrors the Generate sibling's own fix
        # round 1, Finding 3): a refusal toasts identically, so the old
        # assertion could not tell a re-armed button from a permanently
        # wedged one. With the database reachable again, the same button
        # must reach the service and leave a finished audio row behind.
        fake_audio = _FakeAudioService()
        _use_fake_audio_service(monkeypatch, fake_audio)
        app.notify.reset_mock()
        await _press_synthesize(screen, pilot, app, script_id)

        assert len(fake_audio.calls) == 1, (
            "the second press must have reached the synthesis service"
        )
        assert any(
            row["status"] == "complete" for row in db.list_briefing_audio(script_id)
        ), "and must have left a finished audio row behind"


@pytest.mark.asyncio
async def test_a_zombie_generating_audio_row_is_recovered_on_a_plain_artifacts_load(
    monkeypatch,
):
    """Load-path seam: `_load_briefings` sweeps a crashed synthesis
    worker's `generating` audio row, exactly like the sibling scripts/
    briefings zombie tests. The recorder proves this is the LOAD path's
    own sweep (`_audio_in_flight` clear at call time), not the Synthesize
    worker's -- the flag-at-call-time lesson carried from phase 1: pinned
    SEPARATELY from the dispatch-path test below, since adding this
    recovery could otherwise silently vacate that one.
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    briefing_id, script_id = _seed_complete_script(app, watchlist_id)
    db = app.watchlist_bundle_service.db
    zombie_id = db.create_briefing_audio(script_id, voice_snapshot_json="[]")

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        await _select_briefing_and_script(screen, pilot, host, briefing_id, script_id)

        in_flight_at_call: list[bool] = []
        real_sweep = screen_module.fail_interrupted_audio

        def _recording_sweep(db_arg, script_id_arg=None, *, exclude=()):
            in_flight_at_call.append(bool(screen._audio_in_flight))
            return real_sweep(db_arg, script_id_arg, exclude=exclude)

        monkeypatch.setattr(screen_module, "fail_interrupted_audio", _recording_sweep)

        await screen._load_briefings()

        assert in_flight_at_call, "the load path's own sweep must have run at all"
        assert all(not flag for flag in in_flight_at_call), (
            "the load path's sweep must run with `_audio_in_flight` CLEAR -- "
            "a call recorded True could only be the Synthesize worker's own "
            "sweep"
        )
        row = db.get_briefing_audio(zombie_id)
        assert row["status"] == "failed"
        assert row["error"] == "interrupted"

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert pane.script_audio is not None
        assert pane.script_audio["id"] == zombie_id
        plain, _ansi = _render_to_console(
            pane.query_one("#artifacts-script-detail", Static).renderable
        )
        assert "interrupted" in plain
        assert "This audio is being synthesized now." not in plain


@pytest.mark.asyncio
async def test_synthesizing_recovers_a_zombie_audio_row_via_its_own_sweep(monkeypatch):
    """Synthesize-path seam, pinned SEPARATELY from the load-path test
    above: the Synthesize worker sweeps `fail_interrupted_audio` at its
    own front, exactly where `_cast_script` sweeps for Cast. The recorder
    proves THIS call carries `_audio_in_flight` claimed (True) -- only the
    Synthesize worker's own sweep call can, since the load path's sweep is
    gated on the flag being clear.
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    briefing_id, script_id = _seed_complete_script(app, watchlist_id)
    db = app.watchlist_bundle_service.db
    zombie_id = db.create_briefing_audio(script_id, voice_snapshot_json="[]")

    fake_audio = _FakeAudioService()
    _use_fake_audio_service(monkeypatch, fake_audio)

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        await _select_briefing_and_script(screen, pilot, host, briefing_id, script_id)

        in_flight_at_call: list[bool] = []
        real_sweep = screen_module.fail_interrupted_audio

        def _recording_sweep(db_arg, script_id_arg=None, *, exclude=()):
            in_flight_at_call.append(bool(screen._audio_in_flight))
            return real_sweep(db_arg, script_id_arg, exclude=exclude)

        monkeypatch.setattr(screen_module, "fail_interrupted_audio", _recording_sweep)

        await _press_synthesize(screen, pilot, app, script_id)

        assert True in in_flight_at_call, (
            "the zombie must be recovered by the Synthesize worker's OWN "
            "sweep (a call with `_audio_in_flight` claimed), not merely by "
            "the load-path recovery that runs after the flag clears"
        )
        rows = {row["id"]: row for row in db.list_briefing_audio(script_id)}
        assert rows[zombie_id]["status"] == "failed"
        assert rows[zombie_id]["error"] == "interrupted"
        assert any(
            row["status"] == "complete"
            for audio_id, row in rows.items()
            if audio_id != zombie_id
        ), "the same press must also have synthesized real audio"


@pytest.mark.asyncio
async def test_the_scripts_table_shows_an_audio_indicator_for_every_row_with_a_render():
    """Review round 1, Minor #4: before this fix, a user had to select
    each script row in turn to discover whether it had ever been
    synthesized -- `pane.script_audio` only ever answers that for the
    SELECTED script. The scripts table's own "Audio" column answers it up
    front for every row, via a plain, app-controlled glyph
    (`ArtifactsPane._AUDIO_GLYPH`), never provider/model text -- mirroring
    `ItemsPane._QUEUED_GLYPH`'s own phase-1 precedent exactly.
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    db = app.watchlist_bundle_service.db
    briefing_id = db.insert_briefing(watchlist_id)
    db.update_briefing(briefing_id, status="complete", body_markdown="Body")

    with_audio_id = db.insert_briefing_script(
        briefing_id,
        preset_id=None,
        preset_name="With audio",
        roster_snapshot_json=dump_roster(ONE_SPEAKER_ROSTER),
        status="complete",
    )
    db.create_briefing_audio(with_audio_id, voice_snapshot_json="[]")

    without_audio_id = db.insert_briefing_script(
        briefing_id,
        preset_id=None,
        preset_name="Without audio",
        roster_snapshot_json=dump_roster(ONE_SPEAKER_ROSTER),
        status="complete",
    )

    async with _open_artifacts(app, watchlist_id, visual=True) as (
        screen,
        pilot,
        host,
    ):
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        pane.select_briefing_by_id(str(briefing_id))
        await pilot.pause()
        await host.workers.wait_for_complete()
        await pilot.pause()

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert with_audio_id in pane.scripts_with_audio
        assert without_audio_id not in pane.scripts_with_audio

        # Real `DataTable` cells, not a painted screen region: the table's
        # rendered WIDTH is a separate, already-covered concern (`test_the_
        # briefings_table_keeps_at_least_three_usable_rows`'s own sibling
        # for HEIGHT); reading cells directly proves the glyph logic
        # itself without depending on this fixture's terminal width.
        table = pane.query_one("#artifacts-scripts-table", DataTable)
        audio_column_index = 3  # "Preset", "Status", "Created", "Audio"
        with_audio_row = table.get_row(str(with_audio_id))
        without_audio_row = table.get_row(str(without_audio_id))
        assert str(with_audio_row[audio_column_index]) == ArtifactsPane._AUDIO_GLYPH
        assert str(without_audio_row[audio_column_index]) == ""


# --- Owner decision, task-7 phase 2b follow-up: three-state audio glyph ----
#
# "If synthesis fails, show the audio glyph with a red x" (project owner,
# verbatim). Before this, `scripts_with_audio` only ever answered "has an
# attempt of ANY status" (the review-round-1 fix directly above), so a
# FAILED synthesis painted identically to a successful one -- a reviewer
# independently flagged the same gap. These three tests each drive the
# real load path (`pane.select_briefing_by_id`, the same `_load_briefings`
# reload every other test in this file exercises) for exactly one of
# `ArtifactsPane._audio_cell`'s three readings.


def _seed_briefing_with_three_script_audio_states(app, watchlist_id) -> dict[str, int]:
    """One `complete` briefing, three `complete` cast scripts -- one for
    each state `scripts_with_audio` can carry for a script id: a newest
    audio render that is `complete`, one that is `failed`, and a script
    with no `briefing_audio` row at all. Shared by the three tests below
    so each asserts on exactly ONE state without re-deriving the same
    fixture three times.

    Returns:
        `{"briefing_id": ..., "complete": script_id, "failed": script_id,
        "none": script_id}`.
    """
    db = app.watchlist_bundle_service.db
    briefing_id = db.insert_briefing(watchlist_id)
    db.update_briefing(briefing_id, status="complete", body_markdown="Body")

    def _script(preset_name: str) -> int:
        return db.insert_briefing_script(
            briefing_id,
            preset_id=None,
            preset_name=preset_name,
            roster_snapshot_json=dump_roster(ONE_SPEAKER_ROSTER),
            status="complete",
        )

    complete_script_id = _script("Complete audio")
    complete_audio_id = db.create_briefing_audio(
        complete_script_id, voice_snapshot_json="[]"
    )
    db.update_briefing_audio(complete_audio_id, status="complete")

    failed_script_id = _script("Failed audio")
    failed_audio_id = db.create_briefing_audio(
        failed_script_id, voice_snapshot_json="[]"
    )
    db.update_briefing_audio(
        failed_audio_id, status="failed", error="synthesis failed"
    )

    none_script_id = _script("No audio")

    return {
        "briefing_id": briefing_id,
        "complete": complete_script_id,
        "failed": failed_script_id,
        "none": none_script_id,
    }


async def _load_scripts_table(app, watchlist_id, script_states) -> DataTable:
    """Open Artifacts, select the seeded briefing through the real pane,
    and return the mounted scripts table once the reload has landed.
    """
    async with _open_artifacts(app, watchlist_id, visual=True) as (
        screen,
        pilot,
        host,
    ):
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        pane.select_briefing_by_id(str(script_states["briefing_id"]))
        await pilot.pause()
        await host.workers.wait_for_complete()
        await pilot.pause()
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        table = pane.query_one("#artifacts-scripts-table", DataTable)
        # Copy the one cell each test cares about out of the live table --
        # the `async with` block below tears the screen down on exit, and
        # a `Text` cell value survives that fine (it owns no widget
        # reference), but returning the `DataTable` itself would not.
        return {
            state: table.get_row(str(script_id))[3]
            for state, script_id in script_states.items()
            if state != "briefing_id"
        }


_AUDIO_COLUMN_INDEX = 3  # "Preset", "Status", "Created", "Audio"


@pytest.mark.asyncio
async def test_a_complete_audio_row_shows_the_note_glyph_alone():
    """`STATUS_COMPLETE` -> the note glyph, and nothing else -- no failure
    mark, in either the plain text or the styled render.
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    script_states = _seed_briefing_with_three_script_audio_states(app, watchlist_id)

    cells = await _load_scripts_table(app, watchlist_id, script_states)
    cell = cells["complete"]

    assert cell.plain == ArtifactsPane._AUDIO_GLYPH
    _plain, ansi = _render_to_console(cell)
    assert "\x1b[1;31m" not in ansi, "a complete render must carry no red mark at all"


@pytest.mark.asyncio
async def test_a_failed_audio_row_shows_the_note_glyph_and_a_red_x():
    """`STATUS_FAILED` -> the note glyph PLUS a red `✗` -- the owner's own
    words: "if synthesis fails, show the audio glyph with a red x".

    Asserts the exact combined plain text (not merely "the glyph is
    present", which the ORIGINAL bug -- and a complete row -- would also
    satisfy) and that the `✗` specifically carries an explicit red style,
    never a markup string (`ArtifactsPane._audio_cell` builds this with
    `rich.text.Text.append(..., style=...)`, exactly like `_audio_detail_
    renderable`'s header -- this pane never markup-parses cell content).
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    script_states = _seed_briefing_with_three_script_audio_states(app, watchlist_id)

    cells = await _load_scripts_table(app, watchlist_id, script_states)
    cell = cells["failed"]

    expected = f"{ArtifactsPane._AUDIO_GLYPH} {ArtifactsPane._AUDIO_FAILED_MARK}"
    assert cell.plain == expected, (
        "a failed render must show the note glyph AND the failure mark -- "
        "not the glyph alone, which is what the bug this fixes looked like"
    )
    _plain, ansi = _render_to_console(cell)
    assert "\x1b[1;31m" in ansi, "the ✗ must carry an explicit red style"


@pytest.mark.asyncio
async def test_a_script_with_no_audio_row_renders_a_blank_audio_cell():
    """No `briefing_audio` row at all -> an empty cell -- never a bare
    glyph (that would claim an attempt that never happened) and never a
    failure mark either.
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    script_states = _seed_briefing_with_three_script_audio_states(app, watchlist_id)

    cells = await _load_scripts_table(app, watchlist_id, script_states)
    cell = cells["none"]

    assert cell.plain == ""


@pytest.mark.asyncio
async def test_a_failed_audio_row_renders_its_error_text(monkeypatch):
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    briefing_id, script_id = _seed_complete_script(app, watchlist_id)

    fake_audio = _FakeAudioService()
    fake_audio.set_next(
        status="failed", error="turn 0: no voice assigned for speaker 'Narrator'"
    )
    _use_fake_audio_service(monkeypatch, fake_audio)

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        await _select_briefing_and_script(screen, pilot, host, briefing_id, script_id)

        await _press_synthesize(screen, pilot, app, script_id)

        db = app.watchlist_bundle_service.db
        rows = db.list_briefing_audio(script_id)
        assert rows[0]["status"] == "failed"

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert pane.script_audio is not None
        assert pane.script_audio["id"] == rows[0]["id"]
        plain, _ansi = _render_to_console(
            pane.query_one("#artifacts-script-detail", Static).renderable
        )
        assert rows[0]["error"] in plain


@pytest.mark.asyncio
async def test_a_failed_audio_rows_error_text_paints_literally_never_as_markup(
    monkeypatch,
):
    """Review round 1, Minor #2: the mandatory literal-paint test, the
    exact sibling of `test_script_turns_render_as_speaker_labelled_text_
    never_markup` -- a synthesis/provider error is untrusted (model or
    provider-authored) text, so a bracket-shaped fragment like
    `[bold red]x[/]` must paint as those literal characters, never
    interpreted as Rich markup and never escaped into visible backslashes
    either. `_audio_detail_renderable` appends it via `rich.text.Text`,
    exactly like every other model/provider-derived field on this pane.
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    briefing_id, script_id = _seed_complete_script(app, watchlist_id)

    fake_audio = _FakeAudioService()
    fake_audio.set_next(status="failed", error="[bold red]x[/]")
    _use_fake_audio_service(monkeypatch, fake_audio)

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        await _select_briefing_and_script(screen, pilot, host, briefing_id, script_id)

        await _press_synthesize(screen, pilot, app, script_id)

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert pane.script_audio is not None
        assert pane.script_audio["status"] == "failed"
        plain, ansi = _render_to_console(
            pane.query_one("#artifacts-script-detail", Static).renderable,
            width=100,
        )

        assert "[bold red]x[/]" in plain, "the error must paint exactly as written"
        assert "\\[" not in plain, "and must not grow escaping backslashes"
        assert "\x1b[1;31m" not in ansi, "and `[bold red]` must not be applied"


@pytest.mark.asyncio
async def test_play_calls_the_player_with_the_rows_path_and_stop_stops_it(
    monkeypatch, tmp_path
):
    _patch_audio_dir(monkeypatch, tmp_path)
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    briefing_id, script_id = _seed_complete_script(app, watchlist_id)
    db = app.watchlist_bundle_service.db
    audio_file = briefing_audio.briefing_audio_dir() / "clip.wav"
    audio_file.write_bytes(b"RIFF....WAVEfmt ")
    audio_id = db.create_briefing_audio(script_id, voice_snapshot_json="[]")
    db.update_briefing_audio(
        audio_id,
        status="complete",
        file_path=str(audio_file),
        duration_seconds=5.0,
        turn_count=1,
    )

    play_calls: list[Path] = []
    monkeypatch.setattr(
        screen_module, "play_audio_file", lambda path: play_calls.append(path)
    )
    # `handle_stop_audio_requested` delegates to `tts_events.stop_audio_
    # playback_if_current`, which does its OWN local `from tldw_chatbook.
    # TTS.audio_player import get_audio_player` at call time -- so faking
    # it means patching THAT module's attribute, not the screen's (the
    # screen no longer holds a `get_audio_player` reference at all).
    fake_player = _FakePlayer()
    monkeypatch.setattr(audio_player_module, "get_audio_player", lambda: fake_player)

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        await _select_briefing_and_script(screen, pilot, host, briefing_id, script_id)

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        pane.query_one("#artifacts-play-button", Button).press()
        await pilot.pause()

        assert play_calls == [audio_file], (
            "Play must hand the player exactly this row's own file path"
        )

        fake_player.set_current(audio_file)
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        pane.query_one("#artifacts-stop-button", Button).press()
        await pilot.pause()

        assert fake_player.stopped is True


@pytest.mark.asyncio
async def test_stop_does_not_silence_a_different_currently_playing_file(monkeypatch):
    """Trap #3 (`tts_events.stop_audio_playback_if_current`'s own
    docstring): the shared player is a single-slot APP-WIDE singleton, so
    Stop must compare against what the player actually has loaded before
    touching it -- never a bare, unconditional `.stop()`, which could
    silence a completely unrelated clip.
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    briefing_id, script_id = _seed_complete_script(app, watchlist_id)
    db = app.watchlist_bundle_service.db
    audio_id = db.create_briefing_audio(script_id, voice_snapshot_json="[]")
    db.update_briefing_audio(
        audio_id,
        status="complete",
        file_path="/tmp/this-scripts-clip.wav",
        duration_seconds=5.0,
        turn_count=1,
    )

    fake_player = _FakePlayer()
    fake_player.set_current(Path("/tmp/an-unrelated-clip.wav"))
    monkeypatch.setattr(audio_player_module, "get_audio_player", lambda: fake_player)

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        await _select_briefing_and_script(screen, pilot, host, briefing_id, script_id)

        screen.handle_stop_audio_requested(StopAudioRequested())

        assert fake_player.stopped is False, (
            "Stop must not touch a DIFFERENT file the player is currently "
            "playing"
        )


@pytest.mark.asyncio
async def test_play_is_disabled_when_the_file_is_null_or_missing(monkeypatch, tmp_path):
    """The spec's honest-degradation rule: an artifact whose file was
    deleted underneath us -- or was never written at all (no row yet, or
    a `failed` row) -- must not offer a control that can never do
    anything. A positive control closes the vacuous-guard gap (`_assert_
    on_screen`'s own naming for the lesson): Play really can be enabled,
    so the disabled assertions above are not trivially true because Play
    is simply never enabled at all.

    Also covers Qodo review round 1, FIX B: a `file_path` outside
    `briefing_audio_dir()` must leave Play disabled exactly like a missing
    file, never treated as playable just because SOME file happens to
    exist at that location on disk.
    """
    _patch_audio_dir(monkeypatch, tmp_path)
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    briefing_id, script_id = _seed_complete_script(app, watchlist_id)
    db = app.watchlist_bundle_service.db

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        await _select_briefing_and_script(screen, pilot, host, briefing_id, script_id)

        # Case 1: no audio row at all yet.
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert pane.script_audio is None, "fixture sanity: nothing synthesized yet"
        assert pane.query_one("#artifacts-play-button", Button).disabled is True

        # Case 2: a row exists with `file_path` NULL (e.g. a `failed` row,
        # or the dedicated voice-resolution-failure row that never gets
        # one -- `briefing_audio._record_voice_resolution_failure`).
        db.update_briefing_audio(
            db.create_briefing_audio(script_id, voice_snapshot_json="[]"),
            status="failed",
            error="boom",
        )
        await screen._load_briefings()
        await pilot.pause()
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert pane.script_audio is not None
        assert pane.script_audio.get("file_path") is None
        assert pane.query_one("#artifacts-play-button", Button).disabled is True

        # Case 3: a row claims a `file_path`, but the file has since been
        # deleted from disk.
        missing_file = briefing_audio.briefing_audio_dir() / "deleted.wav"
        assert not missing_file.exists()
        db.update_briefing_audio(
            db.create_briefing_audio(script_id, voice_snapshot_json="[]"),
            status="complete",
            file_path=str(missing_file),
            duration_seconds=5.0,
            turn_count=1,
        )
        await screen._load_briefings()
        await pilot.pause()
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert pane.script_audio["file_path"] == str(missing_file)
        assert pane.query_one("#artifacts-play-button", Button).disabled is True

        # Case 4 (Qodo review round 1, FIX B): a row claims a `file_path`
        # OUTSIDE `briefing_audio_dir()` -- a tampered/corrupted row -- even
        # though a real file exists right there on disk.
        outside_file = tmp_path / "outside" / "clip.wav"
        outside_file.parent.mkdir(parents=True, exist_ok=True)
        outside_file.write_bytes(b"RIFF....WAVEfmt ")
        db.update_briefing_audio(
            db.create_briefing_audio(script_id, voice_snapshot_json="[]"),
            status="complete",
            file_path=str(outside_file),
            duration_seconds=5.0,
            turn_count=1,
        )
        await screen._load_briefings()
        await pilot.pause()
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert pane.script_audio["file_path"] == str(outside_file)
        assert pane.query_one("#artifacts-play-button", Button).disabled is True

        # Positive control: a REAL file, INSIDE briefing_audio_dir(), makes
        # Play enabled.
        real_file = briefing_audio.briefing_audio_dir() / "clip.wav"
        real_file.write_bytes(b"RIFF....WAVEfmt ")
        db.update_briefing_audio(
            db.create_briefing_audio(script_id, voice_snapshot_json="[]"),
            status="complete",
            file_path=str(real_file),
            duration_seconds=5.0,
            turn_count=1,
        )
        await screen._load_briefings()
        await pilot.pause()
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert pane.query_one("#artifacts-play-button", Button).disabled is False


# --- Path validation (Qodo review round 1, FIX B) -----------------------------
#
# `audio_file_path_is_safe`/`_audio_file_is_playable` must confirm a row's
# `file_path` resolves inside `briefing_audio_dir()` BEFORE any `.exists()`
# call or playback -- the DB row is trusted today, but CLAUDE.md requires
# every file path to go through `Utils/path_validation.py`, and a tampered
# or corrupted row must not let this screen probe or play an arbitrary
# path.


def test_audio_file_path_is_safe_rejects_a_path_outside_the_audio_dir(
    monkeypatch, tmp_path
) -> None:
    """A `file_path` that is a plain, unrelated absolute path (not even
    disguised as an in-directory path) must be rejected -- the baseline
    "obviously outside" case the traversal and in-dir tests below
    contrast against."""
    _patch_audio_dir(monkeypatch, tmp_path)

    assert audio_file_path_is_safe("/etc/passwd") is False


def test_audio_file_path_is_safe_rejects_a_traversal_path(monkeypatch, tmp_path) -> None:
    """A path that is textually rooted at `briefing_audio_dir()` but
    escapes it via `..` segments must still be rejected -- a naive
    "starts with the audio dir string" check would wrongly accept this,
    since the check must resolve the path, not just prefix-match it."""
    _patch_audio_dir(monkeypatch, tmp_path)
    audio_dir = briefing_audio.briefing_audio_dir()

    traversal = str(audio_dir / ".." / ".." / "etc" / "passwd")

    assert audio_file_path_is_safe(traversal) is False


def test_audio_file_path_is_safe_accepts_a_normal_in_dir_path(
    monkeypatch, tmp_path
) -> None:
    """The control case: a genuine, well-formed path inside
    `briefing_audio_dir()` must be accepted -- proving the two rejection
    tests above are pinning a real boundary and not a check so strict it
    rejects everything."""
    _patch_audio_dir(monkeypatch, tmp_path)
    audio_dir = briefing_audio.briefing_audio_dir()
    in_dir_path = audio_dir / "script-1-audio-1.wav"

    assert audio_file_path_is_safe(str(in_dir_path)) is True


def test_audio_file_is_playable_never_probes_the_filesystem_for_an_unsafe_path(
    monkeypatch, tmp_path
) -> None:
    """The safety check must run BEFORE `.exists()`: an unsafe path must
    never even reach a filesystem probe. Wrapping the real `Path.exists`
    (rather than registering a callback) catches a call regardless of which
    `Path` instance makes it.
    """
    _patch_audio_dir(monkeypatch, tmp_path)
    calls: list[Path] = []
    original_exists = Path.exists

    def _spy_exists(self, *args, **kwargs):
        calls.append(self)
        return original_exists(self, *args, **kwargs)

    monkeypatch.setattr(Path, "exists", _spy_exists)

    row = {"file_path": "/etc/passwd", "status": "complete"}

    assert _audio_file_is_playable(row) is False
    assert calls == [], "an unsafe path must never be probed with .exists()"


def test_audio_file_is_playable_true_for_a_real_in_dir_file(monkeypatch, tmp_path) -> None:
    """Positive control: `_audio_file_is_playable` still works for the
    ordinary case once path validation is in place."""
    _patch_audio_dir(monkeypatch, tmp_path)
    real_file = briefing_audio.briefing_audio_dir() / "clip.wav"
    real_file.write_bytes(b"RIFF....WAVEfmt ")

    row = {"file_path": str(real_file), "status": "complete"}

    assert _audio_file_is_playable(row) is True


@pytest.mark.asyncio
async def test_handle_play_audio_requested_refuses_an_unsafe_path_with_no_probe_or_playback(
    monkeypatch, tmp_path
) -> None:
    """Defense in depth: even if Play were somehow pressed with a row whose
    `file_path` fails validation -- a race, or a directly-set screen state
    bypassing the disabled button -- `handle_play_audio_requested` must
    refuse it itself, exactly like the "no file at all" case: silently, no
    toast, no exception, and never handing the path to the player.
    """
    _patch_audio_dir(monkeypatch, tmp_path)
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    briefing_id, script_id = _seed_complete_script(app, watchlist_id)

    play_calls: list[Path] = []
    monkeypatch.setattr(
        screen_module, "play_audio_file", lambda path: play_calls.append(path)
    )

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        await _select_briefing_and_script(screen, pilot, host, briefing_id, script_id)

        screen._loaded_script_audio = {
            "file_path": "/etc/passwd",
            "status": "complete",
        }
        screen.handle_play_audio_requested(PlayAudioRequested())
        await pilot.pause()

        assert play_calls == [], "an unsafe path must never reach the player"
        assert not app.notify.called, (
            "an unsafe path is treated exactly like a missing file: silent, "
            "not a toast"
        )


@pytest.mark.asyncio
async def test_handle_play_audio_requested_still_plays_a_normal_in_dir_path(
    monkeypatch, tmp_path
) -> None:
    """The normal case must keep working once the guard is in place."""
    _patch_audio_dir(monkeypatch, tmp_path)
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    briefing_id, script_id = _seed_complete_script(app, watchlist_id)

    audio_file = briefing_audio.briefing_audio_dir() / "clip.wav"
    audio_file.write_bytes(b"RIFF....WAVEfmt ")

    play_calls: list[Path] = []
    monkeypatch.setattr(
        screen_module, "play_audio_file", lambda path: play_calls.append(path)
    )

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        await _select_briefing_and_script(screen, pilot, host, briefing_id, script_id)

        screen._loaded_script_audio = {
            "file_path": str(audio_file),
            "status": "complete",
        }
        screen.handle_play_audio_requested(PlayAudioRequested())
        await pilot.pause()

        assert play_calls == [audio_file]


# --- Task 1 (phase 3): exporting a briefing as markdown --------------------
#
# `ArtifactsPane`'s Export button lives in the EXISTING `#artifacts-toolbar`
# (no new `Horizontal` -- see that compose()-site comment); its disabled
# state, the `FileSave` push it triggers, and the write-path's honest
# toasts are exercised below. `_write_briefing_export_file` is called
# directly for the write-path tests, the same directness `library_screen`'s
# own `_write_library_note_export_file` tests use -- it is a plain async
# method the `FileSave` callback resolves to, not something that needs a
# real dialog driven through the UI to exercise.


def _seed_complete_briefing(app, watchlist_id: int, *, body: str = "Body text") -> int:
    """A `complete` briefing with a real body, seeded directly (no fake
    chat needed -- these tests are about the export flow, not generation).
    """
    db = app.watchlist_bundle_service.db
    briefing_id = db.insert_briefing(watchlist_id)
    db.update_briefing(
        briefing_id,
        status="complete",
        body_markdown=body,
        covers_from_ts="2026-07-25T00:00:00+00:00",
        covers_through_item_id=5,
    )
    return briefing_id


@pytest.mark.asyncio
async def test_export_button_is_disabled_without_a_complete_selection():
    """Export starts disabled with nothing selected, stays disabled for a
    `failed` row, and enables only once a `complete` briefing is selected.
    """
    app = _build_test_app()
    watchlist_id = _seed_watchlist(app)
    db = app.watchlist_bundle_service.db
    failed_id = db.insert_briefing(watchlist_id)
    db.update_briefing(failed_id, status="failed", error="boom")
    complete_id = _seed_complete_briefing(app, watchlist_id)

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        export_button = pane.query_one("#artifacts-export-button", Button)
        assert export_button.disabled is True, "no selection -> disabled"
        assert export_button.compact, "a bordered button costs 3 rows in a height:1 strip"

        pane.select_briefing_by_id(str(failed_id))
        await host.workers.wait_for_complete()
        await pilot.pause()
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert (
            pane.query_one("#artifacts-export-button", Button).disabled is True
        ), "a failed briefing has no body worth exporting"

        pane.select_briefing_by_id(str(complete_id))
        await host.workers.wait_for_complete()
        await pilot.pause()
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert (
            pane.query_one("#artifacts-export-button", Button).disabled is False
        )


@pytest.mark.asyncio
async def test_pressing_export_pushes_a_file_save_dialog_seeded_with_the_default_filename(
    monkeypatch,
):
    """Pressing Export posts `ExportBriefingRequested`, which the screen's
    handler answers by pushing a `FileSave` dialog pre-filled with a
    sanitized default filename -- proven here by its only observable
    effect (the push), the same way `test_presets_button_opens_the_preset_
    manager` proves `ManagePresetsRequested` through ITS handler's effect
    rather than by intercepting the message object.
    """
    app = _build_test_app()
    watchlist_id = _seed_watchlist(app)
    briefing_id = _seed_complete_briefing(app, watchlist_id)

    push_screen_mock = AsyncMock()

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        # `host` (the `DestinationHarness`) is the real Textual `App`
        # subclass driving this pilot -- `screen.app` resolves to IT, not
        # to the `app` (`TldwCli`) fixture, which this screen only reads
        # as `self.app_instance` (see `_notify_watchlists`). The dialog
        # must be patched on the object `self.app.push_screen` actually
        # resolves through.
        monkeypatch.setattr(host, "push_screen", push_screen_mock)

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        pane.select_briefing_by_id(str(briefing_id))
        await host.workers.wait_for_complete()
        await pilot.pause()

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        pane.query_one("#artifacts-export-button", Button).press()
        await pilot.pause()
        await host.workers.wait_for_complete()
        await pilot.pause()

    assert push_screen_mock.await_count == 1, "Export must push exactly one dialog"
    args, kwargs = push_screen_mock.call_args
    dialog = args[0]
    assert isinstance(dialog, FileSave)
    briefing_row = dict(app.watchlist_bundle_service.db.list_briefings(watchlist_id)[0])
    expected_filename = default_briefing_filename(
        {**briefing_row, "watchlist_name": "Morning AI Brief"},
        watchlist_name="Morning AI Brief",
    )
    assert dialog._default_file == expected_filename
    assert callable(kwargs.get("callback"))


@pytest.mark.asyncio
@pytest.mark.parametrize("resolve_via", ["a real path", "cancel"])
async def test_a_second_export_press_while_the_dialog_is_open_is_refused_then_rearms(
    monkeypatch, tmp_path, resolve_via,
):
    """Review round 1 (Important #1): an earlier draft argued Textual
    "refuses to stack" a second `FileSave`. A live repro of two rapid
    presses disproved that -- the screen stack ended up
    `['FileSave', 'FileSave']`, two live dialogs, not one refused.

    Two presses in one tick (before either worker has run -- `run_worker`
    only schedules) must push exactly ONE dialog and refuse the second
    with a toast. Then, once the first dialog resolves -- exercised here
    BOTH via a real path and via a cancel (`resolve_via`) -- a LATER press
    must work again: the re-arm assertion that catches a flag stuck
    `True` forever, which would be worse than the bug being fixed (a
    cancelled export permanently wedging Export shut).
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    briefing_id = _seed_complete_briefing(app, watchlist_id)

    push_screen_mock = AsyncMock()

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        monkeypatch.setattr(host, "push_screen", push_screen_mock)

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        pane.select_briefing_by_id(str(briefing_id))
        await host.workers.wait_for_complete()
        await pilot.pause()

        # Two presses in the same tick: `Button.press()` only POSTS
        # `Button.Pressed` (Textual's message queue is FIFO and each
        # message is handled to completion before the next is dequeued),
        # so the first press's handler -- which claims the guard
        # synchronously, on the UI thread, before `run_worker` -- has
        # already set `_briefing_export_in_flight` by the time the second
        # press's `ExportBriefingRequested` is handled.
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        export_button = pane.query_one("#artifacts-export-button", Button)
        export_button.press()
        export_button.press()
        await pilot.pause()
        await host.workers.wait_for_complete()
        await pilot.pause()

        assert push_screen_mock.await_count == 1, (
            "two rapid presses must push exactly one dialog, not stack two"
        )
        refusals = [
            call
            for call in app.notify.call_args_list
            if "already in progress" in str(call.args[0])
        ]
        assert len(refusals) == 1, "the second press must be refused with a toast"

        # Resolve the FIRST press's dialog -- the callback it was given --
        # either via a real chosen path or via `None` (cancelled).
        _, first_kwargs = push_screen_mock.call_args
        callback = first_kwargs["callback"]
        if resolve_via == "a real path":
            await callback(tmp_path / "export.md")
        else:
            await callback(None)

        # The guard must have re-armed: a THIRD press now pushes ANOTHER
        # real dialog rather than being refused.
        push_screen_mock.reset_mock()
        app.notify.reset_mock()
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        pane.query_one("#artifacts-export-button", Button).press()
        await pilot.pause()
        await host.workers.wait_for_complete()
        await pilot.pause()

        assert push_screen_mock.await_count == 1, (
            "Export must be usable again once the first dialog resolved"
        )


@pytest.mark.asyncio
async def test_write_briefing_export_file_writes_the_document_and_toasts_success(
    tmp_path,
):
    """The write-path (bypassing the dialog UI, exercised separately above)
    writes `briefing_markdown_document`'s output and notifies on success,
    with `markup=False` since the destination's own filename is
    interpolated into the toast.
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    briefing_id = _seed_complete_briefing(app, watchlist_id, body="Body text")
    briefing = dict(app.watchlist_bundle_service.db.list_briefings(watchlist_id)[0])
    briefing["watchlist_name"] = "Morning AI Brief"

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, _host):
        destination = tmp_path / "export.md"
        await screen._write_briefing_export_file(destination, briefing)

    written = destination.read_text(encoding="utf-8")
    assert "Body text" in written
    assert "Morning AI Brief" in written
    app.notify.assert_called_once()
    args, kwargs = app.notify.call_args
    assert "exported successfully" in args[0]
    assert kwargs.get("markup") is False


@pytest.mark.asyncio
async def test_write_briefing_export_file_cancelled_writes_nothing():
    """A `None` path (the user cancelled the dialog) writes nothing and
    toasts a cancellation, not an error. There is no destination to check
    for a stray write against -- `None` means the dialog never returned
    one -- so this only pins the toast.
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    briefing_id = _seed_complete_briefing(app, watchlist_id)
    briefing = dict(app.watchlist_bundle_service.db.list_briefings(watchlist_id)[0])
    briefing["watchlist_name"] = "Morning AI Brief"

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, _host):
        await screen._write_briefing_export_file(None, briefing)

    app.notify.assert_called_once()
    args, kwargs = app.notify.call_args
    assert "cancelled" in args[0].lower()
    assert kwargs.get("severity") == "information"


@pytest.mark.asyncio
async def test_write_briefing_export_file_rejects_an_invalid_path(monkeypatch, tmp_path):
    """A `FileSave`-returned path that fails `validate_path_simple` is
    rejected with a quiet warning toast -- no write, no crash -- rather
    than trusting the dialog's returned path unconditionally.
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    briefing_id = _seed_complete_briefing(app, watchlist_id)
    briefing = dict(app.watchlist_bundle_service.db.list_briefings(watchlist_id)[0])
    briefing["watchlist_name"] = "Morning AI Brief"

    def _reject_path(*_args, **_kwargs):
        raise ValueError("rejected for test")

    monkeypatch.setattr(screen_module, "validate_path_simple", _reject_path)

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, _host):
        destination = tmp_path / "export.md"
        await screen._write_briefing_export_file(destination, briefing)

    assert not destination.exists()
    app.notify.assert_called_once()
    args, kwargs = app.notify.call_args
    assert "Rejected export path" in args[0]
    assert kwargs.get("severity") == "warning"
    assert kwargs.get("markup") is False


@pytest.mark.asyncio
async def test_write_briefing_export_file_write_failure_toasts_the_exception_type(
    monkeypatch, tmp_path
):
    """An `OSError` from the write itself toasts `type(exc).__name__` --
    never the briefing body, never a raw traceback -- and leaves no file
    behind.
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    briefing_id = _seed_complete_briefing(app, watchlist_id)
    briefing = dict(app.watchlist_bundle_service.db.list_briefings(watchlist_id)[0])
    briefing["watchlist_name"] = "Morning AI Brief"

    def _boom(*_args, **_kwargs):
        raise OSError("disk full")

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, _host):
        destination = tmp_path / "export.md"
        with monkeypatch.context() as ctx:
            ctx.setattr(Path, "write_text", _boom)
            await screen._write_briefing_export_file(destination, briefing)

    assert not destination.exists()
    app.notify.assert_called_once()
    args, kwargs = app.notify.call_args
    assert "OSError" in args[0]
    assert kwargs.get("severity") == "error"
    assert kwargs.get("markup") is False


@pytest.mark.asyncio
async def test_write_briefing_export_file_unicode_encode_error_toasts_the_exception_type(
    monkeypatch, tmp_path
):
    """Review round 1 (Important #2): the write's `except` previously
    caught only `OSError`, narrower than both the brief and the
    `library_screen` precedent it claims to mirror. A live repro
    confirmed a `UnicodeEncodeError` -- entirely plausible from model- or
    feed-derived body text -- escaped uncaught: no toast, no
    notification, a silent failure. Broadened to `except Exception`
    (still logging by type only, never the body).
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    briefing_id = _seed_complete_briefing(app, watchlist_id)
    briefing = dict(app.watchlist_bundle_service.db.list_briefings(watchlist_id)[0])
    briefing["watchlist_name"] = "Morning AI Brief"

    def _boom(*_args, **_kwargs):
        raise UnicodeEncodeError("ascii", "x", 0, 1, "boom")

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, _host):
        destination = tmp_path / "export.md"
        with monkeypatch.context() as ctx:
            ctx.setattr(Path, "write_text", _boom)
            await screen._write_briefing_export_file(destination, briefing)

    assert not destination.exists()
    app.notify.assert_called_once()
    args, kwargs = app.notify.call_args
    assert "UnicodeEncodeError" in args[0]
    assert kwargs.get("severity") == "error"
    assert kwargs.get("markup") is False


@pytest.mark.asyncio
async def test_write_briefing_export_file_cancelled_error_propagates_uncaught(
    monkeypatch, tmp_path
):
    """`asyncio.CancelledError` must never be reported as a failed export
    -- a cancelled worker is not the same thing as a write that failed,
    and the broadened `except Exception` above must not accidentally
    swallow it (review round 1, Important #2's own caveat).
    """
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    briefing_id = _seed_complete_briefing(app, watchlist_id)
    briefing = dict(app.watchlist_bundle_service.db.list_briefings(watchlist_id)[0])
    briefing["watchlist_name"] = "Morning AI Brief"

    def _cancel(*_args, **_kwargs):
        raise asyncio.CancelledError()

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, _host):
        destination = tmp_path / "export.md"
        with monkeypatch.context() as ctx:
            ctx.setattr(Path, "write_text", _cancel)
            with pytest.raises(asyncio.CancelledError):
                await screen._write_briefing_export_file(destination, briefing)

    assert not destination.exists()
    app.notify.assert_not_called()


# --- Task 5 (phase 3): exporting a watchlist's podcast feed directory ------
#
# `ArtifactsPane`'s Export Feed button lives in `#artifacts-toolbar` (no
# new `Horizontal` -- see that compose()-site comment) -- the SAME
# watchlist-scoped toolbar Generate/Refresh/Task 1's markdown Export
# already live in, which renders unconditionally once a watchlist is in
# scope. Review round 1, Important #1: an earlier draft placed it in
# `#artifacts-audio-toolbar` instead, which only renders once a SCRIPT is
# selected -- but the feed export is WATCHLIST-scoped, so that button was
# unreachable without first selecting some unrelated script. Its disabled
# state, the `SelectDirectory` push it triggers, and the write-path's
# honest toasts (including an honest, capped partial-export count) are
# exercised below. Mirrors the Task 1 markdown-export section immediately
# above in almost every respect -- same guard shape, same push-then-
# callback split, same re-arm discipline -- so most docstrings below only
# name what is DIFFERENT for the feed flow rather than repeating the full
# rationale.


def _seed_exportable_audio_episode(
    app, watchlist_id: int, *, filename: str = "clip.wav"
) -> tuple[int, int, int, Path]:
    """A `complete` briefing -> `complete` script -> `complete`, file-
    backed `briefing_audio` row -- everything `list_watchlist_audio_
    episodes` (and so `ArtifactsPane.has_audio_episodes`/`export_feed_
    directory`) requires to treat this as one exportable episode.

    Callers must already have redirected `briefing_audio_dir()` into a
    temp directory (`_patch_audio_dir`, above) before calling this, so the
    file this seeds lands somewhere `audio_file_path_is_safe` accepts.

    Returns:
        `(briefing_id, script_id, audio_id, audio_file_path)`.
    """
    briefing_id, script_id = _seed_complete_script(app, watchlist_id)
    db = app.watchlist_bundle_service.db
    audio_file = briefing_audio.briefing_audio_dir() / filename
    audio_file.write_bytes(b"RIFF....WAVEfmt ")
    audio_id = db.create_briefing_audio(script_id, voice_snapshot_json="[]")
    db.update_briefing_audio(
        audio_id,
        status="complete",
        file_path=str(audio_file),
        duration_seconds=12.3,
        turn_count=1,
    )
    return briefing_id, script_id, audio_id, audio_file


@pytest.mark.asyncio
async def test_export_feed_button_is_disabled_without_any_complete_audio_episode(
    monkeypatch, tmp_path
):
    """Export Feed starts disabled with no audio anywhere in the
    watchlist, and enables once a complete, file-backed episode exists
    ANYWHERE in it -- a dead control offering to export nothing is a spec
    violation (phase 2b shipped a disabled Play for exactly this reason).

    Deliberately watchlist-scoped, not script-scoped: no briefing or
    script is ever selected in this test, proving the button's disabled
    state depends only on the watchlist's own audio, never on a
    selection this pane may or may not have.
    """
    _patch_audio_dir(monkeypatch, tmp_path)
    app = _build_test_app()
    watchlist_id = _seed_watchlist(app)
    _briefing_id, script_id = _seed_complete_script(app, watchlist_id)

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, _host):
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        export_feed_button = pane.query_one("#artifacts-export-feed-button", Button)
        assert export_feed_button.disabled is True, "no audio anywhere -> disabled"
        assert export_feed_button.compact, (
            "a bordered button costs 3 rows in a height:1 strip"
        )

        db = app.watchlist_bundle_service.db
        audio_file = briefing_audio.briefing_audio_dir() / "clip.wav"
        audio_file.write_bytes(b"RIFF....WAVEfmt ")
        audio_id = db.create_briefing_audio(script_id, voice_snapshot_json="[]")
        db.update_briefing_audio(
            audio_id,
            status="complete",
            file_path=str(audio_file),
            duration_seconds=1.0,
            turn_count=1,
        )
        await screen._load_briefings()
        await pilot.pause()

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert (
            pane.query_one("#artifacts-export-feed-button", Button).disabled is False
        )


@pytest.mark.asyncio
async def test_export_feed_button_is_visible_and_enabled_with_no_script_selected(
    monkeypatch, tmp_path
):
    """Review round 1, Important #1: pins the discoverability property
    directly. The feed export is WATCHLIST-scoped -- it exports every
    complete episode in the watchlist -- so it must be reachable with
    NOTHING selected at all, not just once a user has happened to click
    into some unrelated script. `#artifacts-toolbar` (where this button
    now lives) renders unconditionally once a watchlist is in scope,
    unlike `#artifacts-audio-toolbar` (gated on `selected_script is not
    None`), where an earlier draft wrongly placed it -- a user could not
    have found it there without first selecting a script that has
    nothing to do with the export.
    """
    _patch_audio_dir(monkeypatch, tmp_path)
    app = _build_test_app()
    watchlist_id = _seed_watchlist(app)
    _seed_exportable_audio_episode(app, watchlist_id)

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, _host):
        assert screen._selected_briefing is None, (
            "the fixture must start with nothing selected"
        )
        assert screen._selected_script is None

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert pane.selected_briefing is None
        assert pane.selected_script is None
        export_feed_button = pane.query_one("#artifacts-export-feed-button", Button)
        assert export_feed_button.disabled is False, (
            "Export Feed must be reachable without selecting a briefing "
            "or script first"
        )


@pytest.mark.asyncio
async def test_a_workbench_rebuild_does_not_reset_has_audio_episodes(
    monkeypatch, tmp_path
):
    """`_build_detail_pane` is a factory the workbench calls on every
    region rebuild -- a freshly built `ArtifactsPane`'s reactives start at
    their class defaults unless the factory explicitly reseeds them from
    screen state, exactly like every sibling field it already seeds
    (`briefings`, `scripts_with_audio`, ...). Missing that one line would
    silently disable Export Feed the moment a user toggled any OTHER
    region (e.g. the rail), even with real exportable audio still on the
    watchlist. Mirrors `test_tree_highlight_survives_a_section_switch_
    and_a_rail_toggle`'s own mechanism (`action_toggle_left_rail` rebuilds
    the whole workbench) for driving a real rebuild rather than a direct,
    unmounted factory call.
    """
    _patch_audio_dir(monkeypatch, tmp_path)
    app = _build_test_app()
    watchlist_id = _seed_watchlist(app)
    briefing_id, script_id, _audio_id, _audio_file = _seed_exportable_audio_episode(
        app, watchlist_id
    )

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        await _select_briefing_and_script(screen, pilot, host, briefing_id, script_id)
        assert screen._watchlist_has_audio_episodes is True

        screen.action_toggle_left_rail()
        await pilot.pause()
        screen.action_toggle_left_rail()
        await pilot.pause()

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert pane.has_audio_episodes is True, (
            "a workbench rebuild must not silently disable Export Feed"
        )


@pytest.mark.asyncio
async def test_pressing_export_feed_pushes_a_select_directory_dialog(
    monkeypatch, tmp_path
):
    """Pressing Export Feed posts `ExportFeedRequested`, which the
    screen's handler answers by pushing a `SelectDirectory` dialog --
    proven here by its only observable effect (the push), the same way
    Task 1's own `test_pressing_export_pushes_a_file_save_dialog...`
    proves `ExportBriefingRequested` through ITS handler's effect.

    Deliberately selects nothing (review round 1, Important #1): the
    button lives in the watchlist-scoped `#artifacts-toolbar`, reachable
    the moment a watchlist is in scope, not gated on any briefing/script
    selection.
    """
    _patch_audio_dir(monkeypatch, tmp_path)
    app = _build_test_app()
    watchlist_id = _seed_watchlist(app)
    _seed_exportable_audio_episode(app, watchlist_id)

    push_screen_mock = AsyncMock()

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        monkeypatch.setattr(host, "push_screen", push_screen_mock)

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert pane.has_audio_episodes, "the fixture needs an exportable episode"
        pane.query_one("#artifacts-export-feed-button", Button).press()
        await pilot.pause()
        await host.workers.wait_for_complete()
        await pilot.pause()

    assert push_screen_mock.await_count == 1, "Export Feed must push exactly one dialog"
    args, kwargs = push_screen_mock.call_args
    dialog = args[0]
    assert isinstance(dialog, SelectDirectory)
    assert callable(kwargs.get("callback"))


@pytest.mark.asyncio
@pytest.mark.parametrize("resolve_via", ["a real path", "cancel"])
async def test_a_second_export_feed_press_while_the_dialog_is_open_is_refused_then_rearms(
    monkeypatch, tmp_path, resolve_via,
):
    """Mirrors Task 1's own `test_a_second_export_press_while_the_dialog_
    is_open_is_refused_then_rearms`: two presses in one tick must push
    exactly ONE dialog and refuse the second with a toast; a LATER press,
    after the first dialog resolves (exercised both via a real path and
    via a cancel), must work again -- the re-arm assertion that catches a
    flag stuck `True` forever.

    Deliberately selects nothing (review round 1, Important #1) -- see
    `test_pressing_export_feed_pushes_a_select_directory_dialog`'s own
    note.
    """
    _patch_audio_dir(monkeypatch, tmp_path)
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    _seed_exportable_audio_episode(app, watchlist_id)

    push_screen_mock = AsyncMock()

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        monkeypatch.setattr(host, "push_screen", push_screen_mock)

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        export_feed_button = pane.query_one("#artifacts-export-feed-button", Button)
        # Two presses in the same tick, mirroring Task 1's own reasoning:
        # `Button.press()` only POSTS `Button.Pressed`, and the first
        # press's handler claims the guard synchronously, on the UI
        # thread, before `run_worker` -- so by the time the second
        # press's `ExportFeedRequested` is handled, `_feed_export_in_
        # flight` is already `True`.
        export_feed_button.press()
        export_feed_button.press()
        await pilot.pause()
        await host.workers.wait_for_complete()
        await pilot.pause()

        assert push_screen_mock.await_count == 1, (
            "two rapid presses must push exactly one dialog, not stack two"
        )
        refusals = [
            call
            for call in app.notify.call_args_list
            if "already in progress" in str(call.args[0])
        ]
        assert len(refusals) == 1, "the second press must be refused with a toast"

        _, first_kwargs = push_screen_mock.call_args
        callback = first_kwargs["callback"]
        if resolve_via == "a real path":
            destination = tmp_path / "export_dest"
            destination.mkdir()
            await callback(destination)
        else:
            await callback(None)

        # The guard must have re-armed: a THIRD press now pushes ANOTHER
        # real dialog rather than being refused.
        push_screen_mock.reset_mock()
        app.notify.reset_mock()
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        pane.query_one("#artifacts-export-feed-button", Button).press()
        await pilot.pause()
        await host.workers.wait_for_complete()
        await pilot.pause()

        assert push_screen_mock.await_count == 1, (
            "Export Feed must be usable again once the first dialog resolved"
        )


@pytest.mark.asyncio
async def test_export_feed_directory_writes_episodes_and_toasts_the_count(
    monkeypatch, tmp_path
):
    """The write-path (bypassing the dialog UI, exercised separately
    above) exports the feed via the REAL service and notifies the episode
    count on success, with `markup=False` since the destination's own
    directory name is interpolated into the toast.
    """
    _patch_audio_dir(monkeypatch, tmp_path)
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    _briefing_id, _script_id, _audio_id, audio_file = _seed_exportable_audio_episode(
        app, watchlist_id
    )
    db = app.watchlist_bundle_service.db
    destination = tmp_path / "export_dest"
    destination.mkdir()

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, _host):
        await screen._export_feed_directory(
            db, watchlist_id, "Morning AI Brief", destination
        )

    assert (destination / "feed.xml").exists()
    assert len(list(destination.glob("*.wav"))) == 1, (
        "the one exportable episode's audio must have been copied in"
    )
    app.notify.assert_called_once()
    args, kwargs = app.notify.call_args
    assert "Exported 1 episode" in args[0]
    assert "of" not in args[0], "a full export must not read like a partial one"
    assert kwargs.get("severity") == "information"
    assert kwargs.get("markup") is False


@pytest.mark.asyncio
async def test_export_feed_directory_cancelled_writes_nothing(monkeypatch, tmp_path):
    """A `None` path (the user cancelled the dialog) writes nothing and
    toasts a cancellation, not an error.
    """
    _patch_audio_dir(monkeypatch, tmp_path)
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    _briefing_id, _script_id, _audio_id, _audio_file = _seed_exportable_audio_episode(
        app, watchlist_id
    )
    db = app.watchlist_bundle_service.db

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, _host):
        await screen._export_feed_directory(db, watchlist_id, "Morning AI Brief", None)

    app.notify.assert_called_once()
    args, kwargs = app.notify.call_args
    assert "cancelled" in args[0].lower()
    assert kwargs.get("severity") == "information"


@pytest.mark.asyncio
async def test_export_feed_directory_partial_export_toasts_the_honest_count(
    monkeypatch, tmp_path
):
    """A partial export (one episode skipped because its source file has
    since vanished) toasts "N of M episodes exported" -- never a plain
    success -- per `FeedExportResult.skipped`'s own named invariant
    (`briefing_export.py`'s module docstring, decision 3).

    Review round 1, Minor #2: the inlined reason must read as plain
    prose, not a raw `audio {id}:` prefix that means nothing to a user --
    `export_feed_directory` writes reasons naming an internal `audio_id`
    for support/debugging, but the TOAST strips that id
    (`_user_facing_skip_reasons`).
    """
    _patch_audio_dir(monkeypatch, tmp_path)
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    _briefing_id, script_id, _audio_id, _audio_file = _seed_exportable_audio_episode(
        app, watchlist_id
    )
    db = app.watchlist_bundle_service.db
    # A second, complete audio row whose source file has since vanished --
    # `export_feed_directory` skips it (module docstring, decision 3)
    # rather than failing the whole export.
    missing_file = briefing_audio.briefing_audio_dir() / "missing.wav"
    missing_audio_id = db.create_briefing_audio(script_id, voice_snapshot_json="[]")
    db.update_briefing_audio(
        missing_audio_id,
        status="complete",
        file_path=str(missing_file),
        duration_seconds=1.0,
        turn_count=1,
    )
    destination = tmp_path / "export_dest"
    destination.mkdir()

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, _host):
        await screen._export_feed_directory(
            db, watchlist_id, "Morning AI Brief", destination
        )

    app.notify.assert_called_once()
    args, kwargs = app.notify.call_args
    assert "Exported 1 of 2 episodes" in args[0]
    assert "successfully" not in args[0], "a partial export must never claim success"
    assert "source file no longer exists" in args[0], (
        "the reason itself must still read as plain prose"
    )
    assert re.search(r"audio \d+:", args[0]) is None, (
        "the internal `audio {id}:` prefix must never reach the user-facing toast"
    )
    assert kwargs.get("severity") == "warning"
    assert kwargs.get("markup") is False


@pytest.mark.asyncio
async def test_export_feed_directory_partial_export_caps_inlined_reasons(
    monkeypatch, tmp_path
):
    """Review round 1, Minor #2: five skipped episodes must not produce a
    toast quoting all five -- `export_feed_directory` can skip
    arbitrarily many, and a toast that inlines every one of them is
    unreadable well before it gets there. Only the first `_MAX_INLINE_
    SKIP_REASONS` (3) are shown, followed by an honest "…and N more"
    trailer; the headline "N of M" count itself is never capped or
    approximated.
    """
    _patch_audio_dir(monkeypatch, tmp_path)
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    _briefing_id, script_id, _audio_id, _audio_file = _seed_exportable_audio_episode(
        app, watchlist_id
    )
    db = app.watchlist_bundle_service.db
    # Five more complete audio rows whose source files were never written
    # -- each one skipped by `export_feed_directory` with its own "source
    # file no longer exists" reason, naming a distinct `audio_id`.
    for index in range(5):
        missing_file = briefing_audio.briefing_audio_dir() / f"missing-{index}.wav"
        missing_audio_id = db.create_briefing_audio(
            script_id, voice_snapshot_json="[]"
        )
        db.update_briefing_audio(
            missing_audio_id,
            status="complete",
            file_path=str(missing_file),
            duration_seconds=1.0,
            turn_count=1,
        )
    destination = tmp_path / "export_dest"
    destination.mkdir()

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, _host):
        await screen._export_feed_directory(
            db, watchlist_id, "Morning AI Brief", destination
        )

    app.notify.assert_called_once()
    args, kwargs = app.notify.call_args
    assert "Exported 1 of 6 episodes" in args[0], (
        "the headline count is never capped, only the inlined reasons are"
    )
    assert args[0].count("source file no longer exists") == 3, (
        "only the first 3 reasons are inlined"
    )
    assert "…and 2 more" in args[0]
    assert re.search(r"audio \d+:", args[0]) is None
    assert kwargs.get("severity") == "warning"
    assert kwargs.get("markup") is False


@pytest.mark.asyncio
async def test_export_feed_directory_rejects_an_invalid_destination(
    monkeypatch, tmp_path
):
    """A destination that fails `export_feed_directory`'s own `validate_
    path_simple(..., require_exists=True)` -- e.g. one that does not exist
    -- is rejected with a quiet warning toast, no write, no crash.
    """
    _patch_audio_dir(monkeypatch, tmp_path)
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    _briefing_id, _script_id, _audio_id, _audio_file = _seed_exportable_audio_episode(
        app, watchlist_id
    )
    db = app.watchlist_bundle_service.db
    destination = tmp_path / "does_not_exist"

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, _host):
        await screen._export_feed_directory(
            db, watchlist_id, "Morning AI Brief", destination
        )

    assert not destination.exists()
    app.notify.assert_called_once()
    args, kwargs = app.notify.call_args
    assert "Rejected export destination" in args[0]
    assert kwargs.get("severity") == "warning"
    assert kwargs.get("markup") is False


@pytest.mark.asyncio
async def test_export_feed_press_survives_an_os_error_from_the_service(
    monkeypatch, tmp_path
):
    """Sibling of `test_a_database_error_during_synthesis_does_not_exit_
    the_app`: drives the REAL press -> real `SelectDirectory` -> callback
    path (never a patched picker, and never a direct method call) so an
    escaping exception would actually have to survive Textual's own
    message-pump dispatch of the dismiss callback, not merely a bare
    coroutine call. `export_feed_directory` deliberately lets an `OSError`
    from the atomic `feed.xml` write propagate (that function's own
    module docstring), so `_export_feed_directory`'s broad `except
    Exception` is what stands between it and the whole app going down --
    exactly the phase-2b app-death lesson (an unwrapped worker with the
    default `exit_on_error=True` took the app down for real, once).

    Deliberately selects nothing (review round 1, Important #1) -- see
    `test_pressing_export_feed_pushes_a_select_directory_dialog`'s own
    note.
    """
    _patch_audio_dir(monkeypatch, tmp_path)
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    _seed_exportable_audio_episode(app, watchlist_id)
    destination = tmp_path / "export_dest"
    destination.mkdir()

    def _boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(screen_module, "export_feed_directory", _boom)

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, host):
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        pane.query_one("#artifacts-export-feed-button", Button).press()
        await pilot.pause()
        await host.workers.wait_for_complete()
        await pilot.pause()

        assert isinstance(host.screen_stack[-1], SelectDirectory), (
            "the real vendored picker must be the one actually pushed"
        )
        host.screen_stack[-1].dismiss(destination)
        await pilot.pause()
        await host.workers.wait_for_complete()
        await pilot.pause()

        assert host.is_running, "a callback failure must not exit the application"
        assert host.screen_stack[-1] is screen, "the screen must still be standing"
        assert screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)

    assert not (destination / "feed.xml").exists()
    app.notify.assert_called()
    args, kwargs = app.notify.call_args
    assert "OSError" in args[0]
    assert kwargs.get("severity") == "error"
    assert kwargs.get("markup") is False

    # The guard is genuinely re-armed, not merely toasting identically
    # while stuck -- mirrors `test_a_database_error_during_synthesis_
    # does_not_exit_the_app`'s own follow-up assertion.
    assert screen._feed_export_in_flight is False


@pytest.mark.asyncio
async def test_export_feed_directory_cancelled_error_propagates_uncaught(
    monkeypatch, tmp_path
):
    """`asyncio.CancelledError` must never be reported as a failed export
    -- mirrors Task 1's own `test_write_briefing_export_file_cancelled_
    error_propagates_uncaught`.
    """
    _patch_audio_dir(monkeypatch, tmp_path)
    app = _build_test_app()
    app.notify = Mock()
    watchlist_id = _seed_watchlist(app)
    _briefing_id, _script_id, _audio_id, _audio_file = _seed_exportable_audio_episode(
        app, watchlist_id
    )
    db = app.watchlist_bundle_service.db
    destination = tmp_path / "export_dest"
    destination.mkdir()

    def _cancel(*_args, **_kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(screen_module, "export_feed_directory", _cancel)

    async with _open_artifacts(app, watchlist_id) as (screen, pilot, _host):
        with pytest.raises(asyncio.CancelledError):
            await screen._export_feed_directory(
                db, watchlist_id, "Morning AI Brief", destination
            )

    assert not (destination / "feed.xml").exists()
    app.notify.assert_not_called()
