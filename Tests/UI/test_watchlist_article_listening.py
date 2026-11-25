"""Mounted Watchlists listening controls with app-owned transport receipts."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from sqlite3 import OperationalError
from threading import get_ident
from types import SimpleNamespace

import pytest
from textual.screen import Screen
from textual.widgets import Button, DataTable, Input, Static

from Tests.UI.app_factory import _build_test_app, swap_app_aliases
from Tests.UI.test_destination_visual_parity_correction import (
    _visual_destination_harness,
)
from tldw_chatbook.Subscriptions.article_listening_service import (
    BriefListeningPreview,
    ListeningRevision,
    ListeningSnapshot,
)
from tldw_chatbook.Subscriptions.item_persist import persist_subscription_item
from tldw_chatbook.TTS import controlled_wav_profile
from tldw_chatbook.UI.Watchlists_Modules.article_list import ArticleListPane
from tldw_chatbook.UI.Watchlists_Modules.artifacts_pane import ArtifactsPane
from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import TreeScope
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

pytestmark = [pytest.mark.ui, pytest.mark.bootstrap_profile]


class Listening:
    """Inert owner that records what a real button or binding sends."""

    def __init__(self):
        self.state = ListeningSnapshot(
            revision=ListeningRevision("owner", 0, None, 0, None)
        )
        self.preview = BriefListeningPreview(2, ("empty_body",), "play")
        self.calls = []

    def snapshot(self):
        return self.state

    async def preview_brief(self, briefing_id, *, runtime_backend):
        assert briefing_id > 0 and runtime_backend == "local"
        return self.preview

    async def listen(self, item_id, *, runtime_backend, expected_revision):
        self.calls.append(("listen", item_id, runtime_backend, expected_revision))

    async def play_brief(
        self, briefing_id, *, runtime_backend, replace, expected_revision
    ):
        self.calls.append(
            ("brief", briefing_id, runtime_backend, replace, expected_revision)
        )

    async def pause(self, *, expected_revision):
        self.calls.append(("pause", expected_revision))

    async def resume(self, *, expected_revision):
        self.calls.append(("resume", expected_revision))

    async def stop(self, *, expected_revision):
        self.calls.append(("stop", expected_revision))

    async def skip(self, *, expected_revision):
        self.calls.append(("skip", expected_revision))

    async def play_again(self, *, expected_revision):
        self.calls.append(("play_again", expected_revision))

    async def retry(self, *, regenerate=False, expected_revision):
        self.calls.append(("retry", regenerate, expected_revision))

    async def clear(self, *, expected_revision):
        self.calls.append(("clear", expected_revision))


async def _until(predicate):
    async with asyncio.timeout(6):
        while not predicate():
            await asyncio.sleep(0.01)


def _text(widget):
    value = widget.renderable
    return getattr(value, "plain", str(value))


def _enabled(screen, selector):
    return (
        bool(screen.query(selector)) and not screen.query_one(selector, Button).disabled
    )


def _painted(screen, widget):
    region = widget.region.intersection(screen.region)
    strips = screen._compositor.render_strips()
    return " ".join(
        strips[y].crop(region.x, region.right).text for y in region.line_range
    )


async def _click(screen, pilot, selector):
    button = screen.query_one(selector, Button)
    assert not button.disabled
    # Textual ignores mouse presses on a button during its active effect.
    await _until(lambda: not button.has_class("-active"))
    button.scroll_visible(animate=False, immediate=True)
    await pilot.pause()
    hit = await pilot.click(selector)
    parent_layout = [
        (node.id, node.region, node.scroll_offset)
        for node in button.ancestors_with_self
        if hasattr(node, "region")
    ]
    assert hit, f"{selector} must be a real hit target: {parent_layout}"
    return button


@asynccontextmanager
async def _watchlists(monkeypatch, size=(120, 40), *, served=False, unavailable=None):
    async def readiness():
        return unavailable

    monkeypatch.setattr(controlled_wav_profile, "controlled_wav_readiness", readiness)
    app = _build_test_app()
    app._served_canvas_mode = served
    service = Listening()
    swap_app_aliases(app, article_listening_service=service)
    db = app.subscriptions_db
    watchlist_id = app.watchlist_bundle_service.create("Listening sources")["id"]
    source = db.add_subscription(
        name="Saved feed", type="rss", source="https://listening.invalid/feed"
    )
    app.watchlist_bundle_service.add_source(watchlist_id, source)
    ids = []
    for index, body in enumerate(("Saved prose", "Next saved prose", None)):
        with db.transaction() as cursor:
            ids.append(
                persist_subscription_item(
                    cursor,
                    source,
                    {
                        "title": f"Story {index}",
                        "url": f"https://listening.invalid/{index}",
                        "content": body,
                        "content_hash": str(index),
                        "content_kind": "article",
                        "content_format": "text",
                    },
                    None,
                    "2026-11-24T08:00:00+00:00",
                )
            )
    brief = db.insert_briefing(watchlist_id)
    db.update_briefing(
        brief,
        status="complete",
        body_markdown=" ".join(f"Source [item {item}]." for item in ids),
        covers_through_item_id=ids[-1],
    )
    host = _visual_destination_harness(app, "watchlists_collections")
    async with host.run_test(size=(140, 42)) as pilot:
        screen = host.screen_stack[-1]
        await _until(lambda: bool(screen.query("#wl-workbench")))
        yield SimpleNamespace(
            app=app,
            db=db,
            service=service,
            host=host,
            screen=screen,
            pilot=pilot,
            watchlist=watchlist_id,
            ids=ids,
            brief=brief,
            size=size,
        )


async def _open_reader(state):
    state.screen.active_section = "items"
    await _until(lambda: bool(state.screen.query("#watchlists-items-pane")))
    pane = state.screen.query_one("#watchlists-items-pane", ArticleListPane)
    await _until(lambda: bool(pane.items))
    pane.select_item_by_id(f"local:watchlist_item:{state.ids[0]}")
    await _until(lambda: bool(state.screen.query("#content-body")))
    await _until(lambda: state.db.get_item_status(state.ids[0]) == "reviewed")
    await state.pilot.resize_terminal(*state.size)
    await state.pilot.pause()


async def _open_brief(state):
    state.screen.tree_scope = TreeScope(kind="watchlist", watchlist_id=state.watchlist)
    state.screen.active_section = "artifacts"
    await _until(
        lambda: bool(state.screen.query("#watchlists-artifacts-pane #artifacts-table"))
    )
    pane = state.screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
    await _until(lambda: bool(pane.briefings))
    pane.select_briefing_by_id(str(state.brief))
    await _until(lambda: bool(state.screen.query("#artifacts-citations-table")))
    await state.pilot.resize_terminal(*state.size)
    await state.pilot.pause()
    return state.screen.query_one("#artifacts-citations-table", DataTable)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (140, 42)])
async def test_reader_listens_with_a_visible_control_and_contextual_key(
    monkeypatch, size
):
    async with _watchlists(monkeypatch, size, served=True) as state:
        await _open_reader(state)
        screen, pilot, service = state.screen, state.pilot, state.service
        assert screen.query("#watchlists-listen"), "Reader needs a Listen control"
        await _until(lambda: _enabled(screen, "#watchlists-listen"))
        note = screen.query_one("#watchlists-listen-note", Static)
        assert "saved" in _text(note).lower() and "computer running Chatbook" in _text(
            note
        )
        button = await _click(screen, pilot, "#watchlists-listen")
        assert "Listen" in _painted(screen, button)
        await _until(lambda: len(service.calls) == 1)
        assert any(key == "l" for key, *_ in screen._footer_shortcut_registration[1])
        await pilot.press("l")
        await _until(lambda: len(service.calls) == 2)
        assert (
            service.calls
            == [
                (
                    "listen",
                    f"local:watchlist_item:{state.ids[0]}",
                    "local",
                    service.state.revision,
                )
            ]
            * 2
        )
        assert state.db.get_item_status(state.ids[0]) == "reviewed"

        # Compact Read is reader-only, so the input-routing witness is wide.
        if size[0] > 80:
            search = screen.query_one("#items-search-input", Input)
            search.focus()
            await pilot.pause()
            await pilot.press("l")
            assert search.value == "l"
            assert len(service.calls) == 2
            registration = screen._footer_shortcut_registration
            assert registration is None or all(
                key != "l" for key, *_ in registration[1]
            )


@pytest.mark.asyncio
async def test_reader_explains_missing_output_before_starting(monkeypatch):
    async with _watchlists(
        monkeypatch, unavailable="qualified_ffplay_unavailable"
    ) as state:
        await _open_reader(state)
        screen = state.screen
        assert screen.query("#watchlists-listen-note"), "Reader needs inline readiness"
        note = screen.query_one("#watchlists-listen-note", Static)
        await _until(lambda: "ffplay" in _text(note).lower())
        assert screen.query_one("#watchlists-listen", Button).disabled
        await state.pilot.press("l")
        assert state.service.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (140, 42)])
async def test_highlighted_citation_listens_and_preserves_the_brief(monkeypatch, size):
    async with _watchlists(monkeypatch, size) as state:
        table = await _open_brief(state)
        screen, pilot, service = state.screen, state.pilot, state.service
        assert screen.query(
            "#artifacts-listen-citation"
        ), "Citations need a separate Listen action"
        table.focus()
        await pilot.press("down")
        assert table.cursor_row == 1
        await _click(screen, pilot, "#artifacts-listen-citation")
        await _until(lambda: bool(service.calls))
        assert service.calls == [
            (
                "listen",
                f"local:watchlist_item:{state.ids[1]}",
                "local",
                service.state.revision,
            )
        ]
        assert screen.active_section == "artifacts"
        assert screen.query_one("#artifacts-citations-table", DataTable) is table
        assert table.cursor_row == 1
        assert state.db.get_item_status(state.ids[1]) == "new"
        note = screen.query_one("#artifacts-listening-note", Static)
        await _until(lambda: "2 playable" in _text(note))
        assert "1 unavailable" in _text(note)
        table.focus()
        await pilot.press("down")
        await _until(
            lambda: screen.query_one("#artifacts-listen-citation", Button).disabled
        )
        assert "saved body" in _text(
            screen.query_one("#artifacts-citation-listen-note", Static)
        )
        await _until(lambda: _enabled(screen, "#artifacts-play-cited"))
        await _click(screen, pilot, "#artifacts-play-cited")
        await _until(lambda: len(service.calls) == 2)
        assert service.calls[-1] == (
            "brief",
            state.brief,
            "local",
            False,
            service.state.revision,
        )
        assert state.host.screen is screen and screen.active_section == "artifacts"

        if size[0] > 80:
            # The existing Open timer and Read page commit can race each
            # other. Keep the replacement snapshot parked until the exact
            # selected article has hydrated and its ordinary read write lands.
            read_page = screen._controller.list_reader_items_page
            pending_page = asyncio.Event()
            release_page = asyncio.Event()

            async def delayed_page(*args, **kwargs):
                pending_page.set()
                await release_page.wait()
                return await read_page(*args, **kwargs)

            monkeypatch.setattr(
                screen._controller, "list_reader_items_page", delayed_page
            )
            try:
                table.focus()
                await pilot.press("up")
                assert table.cursor_row == 1
                await pilot.press("enter")
                await _until(pending_page.is_set)
                assert screen.active_section == "items"
                await _until(
                    lambda: state.db.get_item_status(state.ids[1]) == "reviewed"
                )
                reader = screen.query_one("#watchlists-content-pane")
                assert reader.item["item_id"] == state.ids[1]
                assert reader.item["content"] == "Next saved prose"
            finally:
                release_page.set()
            assert len(service.calls) == 2


@pytest.mark.asyncio
async def test_brief_uses_resume_and_confirms_replacement(monkeypatch):
    async with _watchlists(monkeypatch) as state:
        service = state.service
        revision = ListeningRevision("owner", 1, "queue", 7, "entry")
        service.state = ListeningSnapshot(
            phase="paused",
            revision=revision,
            current_title="Story 0",
            actions=frozenset({"resume", "stop", "clear"}),
        )
        service.preview = BriefListeningPreview(2, ("empty_body",), "resume")
        await _open_brief(state)
        screen, pilot = state.screen, state.pilot
        assert screen.query(
            "#artifacts-play-cited"
        ), "Brief needs a cited playback entry point"
        await _until(lambda: _enabled(screen, "#artifacts-play-cited"))
        button = screen.query_one("#artifacts-play-cited", Button)
        assert "Resume" in button.label.plain
        await _click(screen, pilot, "#artifacts-play-cited")
        await _until(lambda: service.calls == [("resume", revision)])
        paused = service.state
        service.state = replace(
            paused, phase="preparing", actions=frozenset({"pause", "stop"})
        )
        await _until(lambda: button.label.plain == "Queue in progress")
        assert button.disabled

        service.preview = BriefListeningPreview(2, ("empty_body",), "replace")
        service.state = replace(paused, revision=replace(revision, epoch=2))
        await _until(lambda: "Play cited" in button.label.plain and not button.disabled)
        await _click(screen, pilot, "#artifacts-play-cited")
        await _until(lambda: isinstance(state.host.screen, ConfirmationDialog))
        await _click(state.host.screen, pilot, "#cancel-button")
        assert service.calls == [("resume", revision)]
        await _until(lambda: state.host.screen is screen)
        await _click(screen, pilot, "#artifacts-play-cited")
        await _until(lambda: isinstance(state.host.screen, ConfirmationDialog))
        await _click(state.host.screen, pilot, "#confirm-button")
        await _until(lambda: len(service.calls) == 2)
        assert service.calls[-1] == (
            "brief",
            state.brief,
            "local",
            True,
            service.state.revision,
        )
        assert state.host.screen is screen and screen.active_section == "artifacts"
        await _click(screen, pilot, "#artifacts-play-cited")
        await _until(lambda: isinstance(state.host.screen, ConfirmationDialog))
        service.preview = BriefListeningPreview(
            1, ("empty_body", "missing_item"), "replace"
        )
        await _click(state.host.screen, pilot, "#confirm-button")
        note = screen.query_one("#artifacts-listening-note", Static)
        await _until(lambda: "1 playable" in _text(note))
        assert len(service.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (140, 42)])
async def test_transport_projects_progress_and_recovery_without_losing_focus(
    monkeypatch, size
):
    async with _watchlists(monkeypatch, size) as state:
        screen, pilot, service = state.screen, state.pilot, state.service
        await pilot.resize_terminal(*size)
        revision = ListeningRevision("owner", 1, "queue", 7, "entry")
        service.state = ListeningSnapshot(
            phase="preparing",
            revision=revision,
            current_title="Story [literal]",
            suspended_title="Queued story",
            stored_feed_text=True,
            queue_position=2,
            queue_count=3,
            completed_chunks=1,
            total_chunks=4,
            actions=frozenset({"pause", "stop", "skip", "clear"}),
        )
        assert screen.query("#wl-listening"), "Watchlists needs its own fixed transport"
        status = screen.query_one("#wl-listening-status", Static)
        await _until(lambda: "Preparing" in _text(status))
        assert "1/4" in _text(status) and "Story [literal]" in _text(status)
        assert "Suspended" in _text(screen.query_one("#wl-listening-suspended", Static))
        pause = await _click(screen, pilot, "#wl-listening-primary")
        await _until(lambda: service.calls == [("pause", revision)])
        assert screen.focused is pause
        service.state = replace(
            service.state,
            phase="playing",
            elapsed_frames=96000,
            sample_rate=48000,
            duration_frames=480000,
        )
        await _until(lambda: "0:02" in _text(status))
        assert "0:10" in _painted(screen, status)
        assert screen.focused is pause
        await pilot.press("tab", "enter")
        await _until(lambda: service.calls[-1] == ("stop", revision))
        await pilot.press("tab", "enter")
        await _until(lambda: service.calls[-1] == ("skip", revision))
        service.state = replace(
            service.state,
            phase="paused",
            error="article_audio_missing",
            actions=frozenset({"regenerate", "clear"}),
        )
        await _until(lambda: "Regenerate" in pause.label.plain)
        assert (
            "missing" in _text(screen.query_one("#wl-listening-detail", Static)).lower()
        )
        await _click(screen, pilot, "#wl-listening-primary")
        await _until(lambda: service.calls[-1] == ("retry", True, revision))
        await _click(screen, pilot, "#wl-listening-clear")
        await _until(lambda: service.calls[-1] == ("clear", revision))
        service.state = replace(
            service.state,
            phase="paused",
            error=None,
            actions=frozenset({"resume", "clear"}),
        )
        await _until(lambda: pause.label.plain == "Resume saved listening")
        service.state = replace(
            service.state, phase="completed", actions=frozenset({"play_again", "clear"})
        )
        await _until(lambda: pause.label.plain == "Play again")
        await _click(screen, pilot, "#wl-listening-primary")
        await _until(lambda: service.calls[-1] == ("play_again", revision))
        successor = Listening()
        successor.state = replace(
            service.state,
            phase="playing",
            revision=ListeningRevision("successor", 0, "next", 0, "next-entry"),
            actions=frozenset({"pause", "stop"}),
        )
        swap_app_aliases(state.app, article_listening_service=successor)
        await state.host.switch_screen(Screen())
        await _until(lambda: not screen.is_attached)
        assert service.calls.count(("stop", revision)) == 1
        assert successor.calls == []


@pytest.mark.asyncio
async def test_global_palette_uses_the_live_owner_from_another_screen(monkeypatch):
    from tldw_chatbook.app import ArticleListeningCommandProvider

    provider_class = ArticleListeningCommandProvider
    async with _watchlists(monkeypatch) as state:
        assert provider_class in state.app.COMMANDS
        revision = ListeningRevision("owner", 1, "queue", 7, "entry")
        state.service.state = ListeningSnapshot(
            phase="playing",
            revision=revision,
            current_title="Story",
            actions=frozenset({"pause", "stop"}),
        )
        away = Screen()
        await state.host.push_screen(away)
        state.host.services = state.app.services
        provider = provider_class(away)
        hits = [hit async for hit in provider.discover()]
        assert len(hits) == 2
        pause = [hit async for hit in provider.search("pause")]
        assert len(pause) == 1
        pause[0].command()
        await _until(lambda: state.service.calls == [("pause", revision)])
        state.service.state = replace(
            state.service.state, phase="paused", actions=frozenset({"resume", "stop"})
        )
        pause[0].command()
        resume = [hit async for hit in provider.search("resume")]
        assert len(resume) == 1
        resume[0].command()
        await _until(
            lambda: state.service.calls == [("pause", revision), ("resume", revision)]
        )
        assert state.host.screen is away


@pytest.mark.asyncio
@pytest.mark.parametrize("reproject", [False, True])
async def test_play_rechecks_the_guidance_the_user_clicked(monkeypatch, reproject):
    async with _watchlists(monkeypatch) as state:
        service, screen, pilot = state.service, state.screen, state.pilot
        await _open_brief(state)
        await _until(lambda: _enabled(screen, "#artifacts-play-cited"))
        note = screen.query_one("#artifacts-listening-note", Static)
        assert "2 playable" in _text(note)
        before = service.state.revision
        changed = BriefListeningPreview(1, ("empty_body", "missing_item"), "play")
        entered, released = asyncio.Event(), asyncio.Event()

        async def preview(*args, **kwargs):
            if not entered.is_set():
                entered.set()
                await released.wait()
            return changed

        monkeypatch.setattr(service, "preview_brief", preview)
        try:
            await _click(screen, pilot, "#artifacts-play-cited")
            await _until(entered.is_set)
            if reproject:
                service.state = replace(
                    service.state, revision=replace(before, epoch=1)
                )
                await _until(lambda: "1 playable" in _text(note))
            released.set()
            await state.host.workers.wait_for_complete()
            assert service.calls == []
            assert "1 playable" in _text(note)
            await _click(screen, pilot, "#artifacts-play-cited")
            await _until(lambda: len(service.calls) == 1)
            assert service.calls[0] == (
                "brief",
                state.brief,
                "local",
                False,
                service.state.revision,
            )
        finally:
            released.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("citation", [False, True])
async def test_unmount_withdraws_pending_article_on_the_same_owner(
    monkeypatch, citation
):
    async with _watchlists(monkeypatch, (140, 42)) as state:
        if citation:
            table = await _open_brief(state)
            table.focus()
            await state.pilot.press("down")
        else:
            await _open_reader(state)
        selector = "#artifacts-listen-citation" if citation else "#watchlists-listen"
        started, release, withdrawn = (asyncio.Event() for _ in range(3))
        service = state.service

        async def listen(*args, **kwargs):
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                withdrawn.set()
                raise
            service.calls.append("output selected")

        monkeypatch.setattr(service, "listen", listen)
        try:
            await _until(lambda: _enabled(state.screen, selector))
            await _click(state.screen, state.pilot, selector)
            await _until(started.is_set)
            await state.host.switch_screen(Screen())
            await _until(lambda: not state.screen.is_attached)
            assert withdrawn.is_set()
            assert service.calls == []
        finally:
            release.set()


@pytest.mark.asyncio
async def test_delayed_transport_resume_needs_the_view_after_readiness(monkeypatch):
    async with _watchlists(monkeypatch) as state:
        await _open_reader(state)
        await _until(lambda: _enabled(state.screen, "#watchlists-listen"))
        service = state.service
        service.state = replace(
            service.state,
            phase="paused",
            current_title="Saved",
            actions=frozenset({"resume"}),
        )
        button = state.screen.query_one("#wl-listening-primary", Button)
        await _until(
            lambda: button.label.plain == "Resume saved listening"
            and button.region.width > 0
        )
        started, release = asyncio.Event(), asyncio.Event()

        async def probe():
            started.set()
            await release.wait()

        monkeypatch.setattr(controlled_wav_profile, "controlled_wav_readiness", probe)
        try:
            await _click(state.screen, state.pilot, "#wl-listening-primary")
            await _until(started.is_set)
            await state.host.switch_screen(Screen())
            await _until(lambda: not state.screen.is_attached)
        finally:
            release.set()
        await state.host.workers.wait_for_complete()
        assert service.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("during_click", [False, True])
async def test_preview_storage_error_can_be_rechecked_without_playing(
    monkeypatch, during_click
):
    async with _watchlists(monkeypatch) as state:
        service, screen, pilot = state.service, state.screen, state.pilot
        successful = service.preview_brief

        async def unavailable(*args, **kwargs):
            raise OperationalError("private diagnostic")

        if during_click:
            await _open_brief(state)
            await _until(lambda: _enabled(screen, "#artifacts-play-cited"))
            monkeypatch.setattr(service, "preview_brief", unavailable)
            await _click(screen, pilot, "#artifacts-play-cited")
        else:
            monkeypatch.setattr(service, "preview_brief", unavailable)
            await _open_brief(state)
        button = screen.query_one("#artifacts-play-cited", Button)
        note = screen.query_one("#artifacts-listening-note", Static)
        await _until(lambda: "storage is unavailable" in _text(note))
        assert "private diagnostic" not in _text(note)
        assert button.label.plain == "Recheck cited excerpts"
        assert service.calls == []
        monkeypatch.setattr(service, "preview_brief", successful)
        await _click(screen, pilot, "#artifacts-play-cited")
        await _until(lambda: "2 playable" in _text(note))
        assert service.calls == []
        await _click(screen, pilot, "#artifacts-play-cited")
        await _until(lambda: len(service.calls) == 1)


@pytest.mark.asyncio
async def test_long_legal_titles_keep_compact_host_notice_and_actions_visible(
    monkeypatch,
):
    async with _watchlists(monkeypatch, (80, 24), served=True) as state:
        await state.pilot.resize_terminal(*state.size)
        await _until(lambda: state.screen.size.width == 80)
        service = state.service
        service.state = replace(
            service.state,
            phase="paused",
            current_title="current [literal] " * 111,
            suspended_title="suspended [literal] " * 100,
            actions=frozenset({"resume", "stop", "clear"}),
            queue_count=2,
        )
        screen = state.screen
        details = screen.query_one("#wl-listening-detail", Static)
        await _until(lambda: "computer running Chatbook" in _text(details))
        await state.pilot.pause()
        transport = screen.query_one("#wl-listening")
        assert screen.region.contains_region(transport.region)
        assert "Audio plays on the computer running Chatbook." in _painted(
            screen, details
        )
        await _click(screen, state.pilot, "#wl-listening-primary")
        await _until(lambda: service.calls == [("resume", service.state.revision)])


@pytest.mark.asyncio
async def test_body_availability_is_calculated_off_loop_once_per_view_selection(
    monkeypatch,
):
    inspected = []

    class Body(str):
        def strip(self, *args):
            inspected.append(get_ident())
            return super().strip(*args)

        def isspace(self):
            inspected.append(get_ident())
            return super().isspace()

    async with _watchlists(monkeypatch) as state:
        screen = state.screen
        body = Body(" " * 8192 + "Saved text")

        async def load_body(**kwargs):
            return body

        monkeypatch.setattr(screen._controller, "get_item_content", load_body)
        await _open_reader(state)
        await _until(lambda: _enabled(screen, "#watchlists-listen"))
        await state.pilot.pause(0.65)
        assert inspected and get_ident() not in inspected
        count = len(inspected)
        screen.query_one("#items-search-input", Input).focus()
        await state.pilot.pause(0.65)
        assert len(inspected) == count

        original_citations = state.db.get_subscription_items_by_ids

        def load_citations(ids):
            rows = original_citations(ids)
            rows[state.ids[1]]["content"] = body
            return rows

        monkeypatch.setattr(state.db, "get_subscription_items_by_ids", load_citations)
        table = await _open_brief(state)
        table.focus()
        await state.pilot.press("down")
        await _until(lambda: _enabled(screen, "#artifacts-listen-citation"))
        assert len(inspected) > count and get_ident() not in inspected
        count = len(inspected)
        await state.pilot.pause(0.65)
        assert len(inspected) == count


@pytest.mark.asyncio
async def test_reader_open_publishes_while_listening_eligibility_is_pending(
    monkeypatch,
):
    async with _watchlists(monkeypatch) as state:
        screen, pilot, db = state.screen, state.pilot, state.db
        screen.active_section = "items"
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        await _until(lambda: len(pane.items) >= 2)
        first, second = state.ids[:2]
        entered = {item: asyncio.Event() for item in (first, second)}
        release = {item: asyncio.Event() for item in (first, second)}
        completed = {item: asyncio.Event() for item in (first, second)}

        async def eligibility(item):
            key = item["item_id"]
            entered[key].set()
            try:
                await release[key].wait()
            except asyncio.CancelledError:
                # A late completion still needs the selection fence.
                await release[key].wait()
            completed[key].set()
            return ""

        monkeypatch.setattr(screen._listening, "prepare_reader_issue", eligibility)
        try:
            pane.select_item_by_id(f"local:watchlist_item:{first}")
            await _until(entered[first].is_set)
            assert (
                screen._selected_content_item is not None
            ), "Ordinary Open must publish before optional eligibility finishes"
            assert screen._selected_content_item["content"] == "Saved prose"
            await _until(lambda: db.get_item_status(first) == "reviewed")
            await _until(lambda: bool(screen.query("#watchlists-listen")))
            assert not _enabled(screen, "#watchlists-listen")

            pane.select_item_by_id(f"local:watchlist_item:{second}")
            await _until(entered[second].is_set)
            assert screen._selected_content_item["content"] == "Next saved prose"
            await _until(lambda: db.get_item_status(second) == "reviewed")
            release[first].set()
            await _until(completed[first].is_set)
            await pilot.pause()
            assert not _enabled(screen, "#watchlists-listen")

            release[second].set()
            await _until(lambda: _enabled(screen, "#watchlists-listen"))
            assert screen._selected_content_item["item_id"] == second
            assert state.service.calls == []
        finally:
            for ready in release.values():
                ready.set()


@pytest.mark.asyncio
async def test_manual_fixture_joins_complete_tts_shutdown(monkeypatch, tmp_path):
    from contextlib import nullcontext

    from Tests.UI import native_watchlist_article_listening as helper
    from tldw_chatbook import app, config

    calls = []

    async def close_listening():
        calls.append("listening")

    async def begin_shutdown():
        calls.append("tts registry only")

    async def finish_shutdown():
        calls.append("tts complete")
        raise RuntimeError("bounded complete shutdown failure")

    db = SimpleNamespace(
        db_path=tmp_path / "fixture.db", close=lambda: calls.append("database")
    )
    store = SimpleNamespace(
        session_lease=lambda root: nullcontext(
            SimpleNamespace(recover_predecessor=lambda: None),
        )
    )
    service = SimpleNamespace(store=store, close=close_listening)
    watchlists = SimpleNamespace(listening=service, subscriptions_db=db)
    monkeypatch.setattr(config, "get_cli_config_path", lambda: tmp_path / "config.toml")
    monkeypatch.setattr(config, "get_user_data_dir", lambda: tmp_path)
    monkeypatch.setattr(
        app,
        "TldwCli",
        lambda: SimpleNamespace(
            subscriptions_db=db,
            services=SimpleNamespace(watchlists=watchlists),
            tts_service=SimpleNamespace(
                close=begin_shutdown, wait_closed=finish_shutdown
            ),
        ),
    )
    with pytest.raises(RuntimeError, match="bounded complete shutdown failure"):
        await helper._prepare_private_profile(tmp_path, verify_only=True)
    assert calls == ["listening", "tts complete", "database"]
