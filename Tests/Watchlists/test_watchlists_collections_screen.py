"""Tests for the Watchlists collections screen action handlers."""

from contextlib import asynccontextmanager

import pytest
from unittest.mock import AsyncMock, Mock

from rich.text import Text
from textual.widgets import Button, Input, Static, TextArea

from Tests.UI.test_destination_shells import DestinationHarness, _static_text
from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.UI.Screens.watchlists_collections_screen import WatchlistsCollectionsScreen
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.UI.Watchlists_Modules.inspector_pane import (
    BreadcrumbScopeSelected,
    CheckNowRequested,
    InspectorPane,
    PreviewRequested,
)
from tldw_chatbook.UI.Watchlists_Modules.items_pane import ItemSelected
from tldw_chatbook.UI.Watchlists_Modules.opml_dialogs import (
    OpmlExportDialog,
    OpmlImportDialog,
    WatchlistNameDialog,
    WatchlistSourcePickerDialog,
)
from tldw_chatbook.UI.Watchlists_Modules.sources_pane import (
    ExportOpmlRequested,
    ImportOpmlRequested,
    SourceSelected,
)
from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import TreeScope, TreeScopeChanged


@pytest.fixture
def fake_controller():
    controller = AsyncMock()
    controller.preview_source = AsyncMock(
        return_value={"items": [{"title": "Post"}], "log_text": "ok"}
    )
    controller.check_now = AsyncMock(return_value={"run_id": "1"})
    controller.import_opml = AsyncMock(return_value={"created": 2})
    controller.export_opml = AsyncMock(return_value="<opml></opml>")
    controller.get_overview_data = AsyncMock(
        return_value={
            "total_sources": 0,
            "active_sources": 0,
            "sources_in_error": 0,
            "total_items": 0,
            "new_items": 0,
            "latest_run_status": "unavailable",
            "failed_runs": [],
            "active_alert_rules": 0,
        }
    )
    return controller


@asynccontextmanager
async def _open_screen(controller):
    app_instance = _build_test_app()
    host = DestinationHarness(app_instance, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert isinstance(screen, WatchlistsCollectionsScreen)
        screen._controller = controller
        yield screen, pilot


@pytest.mark.asyncio
async def test_preview_source_handler_calls_controller(fake_controller):
    async with _open_screen(fake_controller) as (screen, pilot):
        screen.post_message(PreviewRequested({"id": "source-1", "name": "Feed"}))
        await pilot.pause(0.2)

        fake_controller.preview_source.assert_awaited_once_with(
            runtime_backend="local", source_config={"id": "source-1", "name": "Feed"}
        )


@pytest.mark.asyncio
async def test_check_now_source_handler_calls_controller(fake_controller):
    async with _open_screen(fake_controller) as (screen, pilot):
        screen.post_message(CheckNowRequested({"id": "source-1", "name": "Feed"}))
        await pilot.pause(0.2)

        fake_controller.check_now.assert_awaited_once_with(
            runtime_backend="local", source_id="source-1"
        )


@pytest.mark.asyncio
async def test_import_opml_handler_calls_controller(fake_controller):
    async with _open_screen(fake_controller) as (screen, pilot):
        screen.post_message(ImportOpmlRequested())
        await pilot.pause(0.1)

        top_screen = screen.app.screen
        assert isinstance(top_screen, OpmlImportDialog)
        text_area = top_screen.query_one("#opml-import-text", TextArea)
        text_area.text = "<opml><outline text=\"A\" xmlUrl=\"http://a.com/feed\"/>"
        top_screen.query_one("#opml-import-confirm", Button).press()
        await pilot.pause(0.2)

        fake_controller.import_opml.assert_awaited_once_with(
            runtime_backend="local",
            xml_text="<opml><outline text=\"A\" xmlUrl=\"http://a.com/feed\"/>",
        )


@pytest.mark.asyncio
async def test_export_opml_handler_calls_controller(fake_controller):
    async with _open_screen(fake_controller) as (screen, pilot):
        screen.post_message(ExportOpmlRequested())
        await pilot.pause(0.2)

        fake_controller.export_opml.assert_awaited_once_with(runtime_backend="local")
        assert isinstance(screen.app.screen, OpmlExportDialog)


# --- Task 7: scope-driven Feeds region, with real seeded data -------------
#
# Tests/UI/test_watchlists_destination_shell.py's own scope tests run
# against DestinationHarness's empty subscriptions DB, so the strongest
# thing they can assert is "narrowing differs, or both sides were already
# empty." These seed real rows through the same `watchlist_bundle_service`
# the screen itself reaches (`_build_test_app()` wires it to an isolated
# temp-dir SQLite file -- see that fixture's `get_subscriptions_db_path`
# patch -- never the user's real database), so the comparisons here are
# exact rather than escape-hatched.


@pytest.mark.asyncio
async def test_scoped_source_rows_narrows_by_watchlist_and_unassigned():
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        service = app.watchlist_bundle_service
        db = service._db

        morning = service.create("Morning AI Brief")
        security = service.create("Security")
        a = db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")
        b = db.add_subscription(name="Krebs", type="rss", source="https://b.example/f")
        c = db.add_subscription(
            name="Loose Feed", type="rss", source="https://c.example/f"
        )
        service.add_source(morning["id"], a)
        service.add_source(security["id"], b)

        screen.post_message(TreeScopeChanged(TreeScope(kind="all")))
        await pilot.pause()
        assert {row["id"] for row in screen.scoped_source_rows()} == {a, b, c}

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=morning["id"]))
        )
        await pilot.pause()
        assert [row["id"] for row in screen.scoped_source_rows()] == [a]

        screen.post_message(TreeScopeChanged(TreeScope(kind="unassigned")))
        await pilot.pause()
        assert [row["id"] for row in screen.scoped_source_rows()] == [c]

        screen.post_message(
            TreeScopeChanged(
                TreeScope(kind="source", watchlist_id=security["id"], source_id=b)
            )
        )
        await pilot.pause()
        assert [row["id"] for row in screen.scoped_source_rows()] == [b]


@pytest.mark.asyncio
async def test_feeds_heading_names_the_scope_with_a_live_count():
    # Seeded *before* the screen mounts (unlike the narrowing test above,
    # which only needs id/type matches): the heading's watchlist-name lookup
    # resolves against `_tree_watchlists`, populated once by `_load_tree_data`
    # in `on_mount` -- the same in-memory-only, no-second-query lookup
    # `_resolve_breadcrumb_labels` already relies on. Seeding after mount
    # would leave that cache stale and fall back to "Watchlist {id}", which
    # is correct behaviour for a real race but not what this test measures.
    app = _build_test_app()
    watchlist = app.watchlist_bundle_service.create("Morning AI Brief")
    db = app.watchlist_bundle_service._db
    a = db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")
    b = db.add_subscription(name="Krebs", type="rss", source="https://b.example/f")
    app.watchlist_bundle_service.add_source(watchlist["id"], a)
    app.watchlist_bundle_service.add_source(watchlist["id"], b)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=watchlist["id"]))
        )
        await pilot.pause()

        heading = screen.query_one("#wl-feeds-scope-heading", Static)
        assert _static_text(heading) == "Feeds in Morning AI Brief (2)"


@pytest.mark.asyncio
async def test_feeds_source_row_escapes_an_untrusted_name():
    """A remote feed's own title reaches the Feeds row label unescaped
    markup has broken this exact screen before; pin it so a source named
    with Rich markup syntax renders as literal text, not parsed markup.
    """
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        service = app.watchlist_bundle_service
        db = service._db

        watchlist = service.create("Morning AI Brief")
        source_id = db.add_subscription(
            name="[bold red]Not Actually Bold[/bold red]",
            type="rss",
            source="https://a.example/f",
        )
        service.add_source(watchlist["id"], source_id)

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=watchlist["id"]))
        )
        await pilot.pause()

        row = screen.query_one(f"#wl-feeds-source-{source_id}", Static)
        row_text = _static_text(row)
        assert "[bold red]Not Actually Bold[/bold red]" in row_text


# --- Fix round 1, Finding 2: a pane-row click must not discard the tree scope


@pytest.mark.asyncio
async def test_selecting_a_pane_row_keeps_the_feeds_region_on_the_tree_scope():
    """Finding 2's exact reproduction: click a watchlist in the tree, then
    click a row in the Sources table to inspect it.

    Before this fix `_select_entity` reset `selected_scope` to "all", and
    since Task 7 made that same reactive drive the Feeds region, the heading
    silently jumped from `Feeds in Morning AI Brief (1)` back to
    `Feeds in All sources (2)` -- an interaction in one region discarding
    the user's navigation in another, with no selection highlight in the
    tree to fall back on.
    """
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        service = app.watchlist_bundle_service
        db = service._db

        morning = service.create("Morning AI Brief")
        a = db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")
        db.add_subscription(name="Loose", type="rss", source="https://c.example/f")
        service.add_source(morning["id"], a)
        screen._tree_watchlists = [{"id": morning["id"], "name": "Morning AI Brief"}]

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=morning["id"]))
        )
        await pilot.pause()
        assert [row["id"] for row in screen.scoped_source_rows()] == [a]
        assert (
            _static_text(screen.query_one("#wl-feeds-scope-heading", Static))
            == "Feeds in Morning AI Brief (1)"
        )

        screen.post_message(SourceSelected({"id": "source-1", "name": "Some Feed", "url": "https://x"}))
        await pilot.pause()

        assert screen.tree_scope == TreeScope(
            kind="watchlist", watchlist_id=morning["id"]
        ), "inspecting a pane row is not navigation; the tree scope must survive it"
        assert [row["id"] for row in screen.scoped_source_rows()] == [a]
        assert (
            _static_text(screen.query_one("#wl-feeds-scope-heading", Static))
            == "Feeds in Morning AI Brief (1)"
        )


@pytest.mark.asyncio
async def test_pane_row_selection_still_claims_no_inspector_ancestry():
    """The half of Task 5 fix round 2 that must NOT regress while Finding 2
    is fixed.

    A pane row carries no watchlist/source ancestry, so the Inspector must
    not put a breadcrumb above it. Clearing `_breadcrumb_labels` alone is
    *not* enough: `InspectorPane._scope_levels` derives an ancestor level
    from `scope` alone and falls back to a `Watchlist {id}` label when no
    label is supplied, so the crumb would still render (just anonymously).
    The Inspector's `scope` must itself be reset -- which is why the tree's
    own navigation state now lives in a separate `tree_scope` reactive
    rather than being read back off `selected_scope`.
    """
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        screen._tree_watchlists = [{"id": 1, "name": "Morning AI Brief"}]

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="source", watchlist_id=1, source_id=10))
        )
        await pilot.pause()
        screen.post_message(ItemSelected({"item_id": "item-1", "title": "RAG Eval"}))
        await pilot.pause()

        inspector = screen.query_one("#watchlists-entity-inspector", InspectorPane)
        assert screen.selected_scope == TreeScope(kind="all")
        assert inspector.scope == TreeScope(kind="all")
        assert inspector.breadcrumb_labels == []
        assert not inspector.query("#inspector-breadcrumb-0")
        assert screen.tree_scope == TreeScope(
            kind="source", watchlist_id=1, source_id=10
        ), "...while the tree itself has not moved"


# --- Fix round 1, Finding 1: staging follows the tree scope ----------------


@pytest.mark.asyncio
async def test_staged_console_payload_follows_the_tree_scope():
    """The Console handoff must send the scope the user navigated to.

    Before this fix the payload was built from `_local_watchlist_records` --
    `WatchlistScopeService.list_watch_items` over every local source,
    regardless of the tree selection -- which is also why the Feeds region
    printed the same sources twice. Selecting "Morning AI Brief" and then
    pressing Stage must stage Morning AI Brief.
    """
    app = _build_test_app()
    app.open_chat_with_handoff = Mock()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        service = app.watchlist_bundle_service
        db = service._db

        morning = service.create("Morning AI Brief")
        arxiv = db.add_subscription(
            name="ArXiv", type="rss", source="https://a.example/f"
        )
        db.add_subscription(name="Krebs", type="rss", source="https://b.example/f")
        service.add_source(morning["id"], arxiv)
        screen._tree_watchlists = [{"id": morning["id"], "name": "Morning AI Brief"}]

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=morning["id"]))
        )
        await pilot.pause()

        body = screen._snapshot_body()
        metadata = screen._snapshot_metadata()

    assert "Morning AI Brief" in body
    assert "ArXiv" in body
    assert "Krebs" not in body, "a source outside the scope must not be staged"
    assert metadata["scope_kind"] == "watchlist"
    assert metadata["scope_label"] == "Morning AI Brief"
    assert metadata["scope_watchlist_id"] == morning["id"]
    assert metadata["source_count"] == 1
    assert metadata["source_titles"] == ["ArXiv"]


@pytest.mark.asyncio
async def test_feeds_lists_each_source_once_under_the_all_scope():
    """Finding 1's headline symptom: with <= WC_LOCAL_PAGE_SIZE sources, the
    unscoped staging block printed every source a second time in the same
    box, in identical typography (`watchlist-feed-source-row` had no rule).
    """
    app = _build_test_app()
    db = app.watchlist_bundle_service._db
    db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")
    db.add_subscription(name="Krebs", type="rss", source="https://b.example/f")

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        # Seeded before mount deliberately: "all" is the default scope, so
        # re-posting it would not move the reactive and nothing would
        # rebuild. This is the resting state a user actually lands on.
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        for _ in range(20):
            await pilot.pause()
            if list(screen.query(".watchlist-feed-source-row")):
                break

        pane_text = "\n".join(
            _static_text(node) for node in screen.query("#watchlists-list-pane Static")
        )
        assert pane_text.count("ArXiv") == 1, pane_text
        assert pane_text.count("Krebs") == 1, pane_text
        assert "Local Watchlists snapshot: All sources (2 sources)" in pane_text


@pytest.mark.asyncio
async def test_feeds_heading_escapes_an_untrusted_source_name():
    """The `source` scope takes its heading label from `rows[0]["name"]` --
    a remote feed's own title. Only the row-level escaping was pinned.
    """
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        service = app.watchlist_bundle_service
        watchlist = service.create("Morning AI Brief")
        source_id = service._db.add_subscription(
            name="[bold red]Not Actually Bold[/bold red]",
            type="rss",
            source="https://a.example/f",
        )
        service.add_source(watchlist["id"], source_id)

        screen.post_message(
            TreeScopeChanged(
                TreeScope(
                    kind="source", watchlist_id=watchlist["id"], source_id=source_id
                )
            )
        )
        for _ in range(20):
            await pilot.pause()
            if list(screen.query(".watchlist-feed-source-row")):
                break

        heading = _static_text(screen.query_one("#wl-feeds-scope-heading", Static))
        assert heading == "Feeds in [bold red]Not Actually Bold[/bold red] (1)"
        summary = _static_text(screen.query_one("#wc-watchlists-summary", Static))
        assert "[bold red]Not Actually Bold[/bold red]" in summary


# --- task-876: the tree's own selection highlight --------------------------
#
# `WatchlistTree` never read `tree_scope`, so nothing in the rail showed
# which node the centre was scoped to. `_apply_tree_scope` is the single
# reconciliation point for BOTH a real tree click (`_on_tree_scope_changed`)
# and a breadcrumb promotion (`handle_breadcrumb_scope_selected`); these
# confirm the highlight follows either path, and survives the two rebuild
# paths (section switch, rail toggle) Phase C already had to fix once for
# `expanded`.


@pytest.mark.asyncio
async def test_breadcrumb_promotion_moves_the_tree_highlight_same_as_a_click():
    # Seeded *before* the screen mounts, like
    # `test_feeds_heading_names_the_scope_with_a_live_count` above: the
    # mounted `WatchlistTree` captures its own `_watchlists` once, from
    # whatever `_load_tree_data` populated `_tree_watchlists` with by the
    # time IT (not this test) last rebuilt the tree -- setting
    # `screen._tree_watchlists` after mount would not reach the
    # already-constructed tree instance's own copy.
    app = _build_test_app()
    morning = app.watchlist_bundle_service.create("Morning AI Brief")
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        for _ in range(20):
            await pilot.pause()
            if host.screen_stack[-1].query(f"#wl-tree-node-watchlist-{morning['id']}"):
                break
        screen = host.screen_stack[-1]

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=morning["id"]))
        )
        await pilot.pause()
        assert screen.query_one(
            f"#wl-tree-node-watchlist-{morning['id']}", Button
        ).has_class("is-active")

        # Promote a breadcrumb back to "all" -- a path that never touches
        # the tree widget directly (see `handle_breadcrumb_scope_selected`)
        # -- and confirm the SAME tree instance updates exactly as a real
        # click would.
        screen.post_message(BreadcrumbScopeSelected(TreeScope(kind="all")))
        await pilot.pause()
        assert screen.query_one("#wl-tree-node-all", Button).has_class("is-active")
        assert not screen.query_one(
            f"#wl-tree-node-watchlist-{morning['id']}", Button
        ).has_class("is-active")


@pytest.mark.asyncio
async def test_tree_highlight_survives_a_section_switch_and_a_rail_toggle():
    """Both a section switch (`watch_active_section`) and a rail toggle
    (`action_toggle_left_rail`) rebuild the whole workbench, constructing a
    brand new `WatchlistTree` -- the same class of bug Phase C already fixed
    once for `expanded`/`active_tag`. Do not assume the fix generalizes;
    test it (task-876, AC #3).
    """
    app = _build_test_app()
    morning = app.watchlist_bundle_service.create("Morning AI Brief")
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        for _ in range(20):
            await pilot.pause()
            if host.screen_stack[-1].query(f"#wl-tree-node-watchlist-{morning['id']}"):
                break
        screen = host.screen_stack[-1]

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=morning["id"]))
        )
        await pilot.pause()
        assert screen.query_one(
            f"#wl-tree-node-watchlist-{morning['id']}", Button
        ).has_class("is-active")

        screen.active_section = "sources"
        await pilot.pause()
        assert screen.query_one(
            f"#wl-tree-node-watchlist-{morning['id']}", Button
        ).has_class("is-active"), "the highlight must survive a section switch"

        screen.action_toggle_left_rail()
        await pilot.pause()
        assert not screen.query("#wl-tree"), "the rail should now be collapsed"

        screen.action_toggle_left_rail()
        await pilot.pause()
        assert screen.query_one(
            f"#wl-tree-node-watchlist-{morning['id']}", Button
        ).has_class("is-active"), "the highlight must survive a rail toggle"


@pytest.mark.asyncio
async def test_load_tree_data_failure_notifies_the_user():
    """A real database failure in `_load_tree_data` must not render
    identically to "you have zero watchlists" -- two empty tree roots and no
    message (task-876). Mirrors every sibling loader's own error-notify
    behaviour (`_load_sources`/`_load_runs`/`_load_notifications`, etc.).
    """
    app = _build_test_app()
    app.watchlist_bundle_service.list_watchlists = Mock(side_effect=RuntimeError("boom"))
    app.notify = Mock()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        for _ in range(20):
            await pilot.pause()
            if app.notify.called:
                break

        assert app.notify.called, "a tree-load failure must notify the user"
        _args, kwargs = app.notify.call_args
        assert kwargs.get("severity") == "error"
        screen = host.screen_stack[-1]
        assert screen.query_one("#wl-tree-node-all", Button)
        assert screen.query_one("#wl-tree-node-unassigned", Button)


# --- TASK-895: the tree's write verbs, end to end -------------------------
#
# Five `WatchlistBundleService` methods had no production caller: Phase C
# shipped the tree's read half, so a user could browse watchlists but never
# make one. These drive the real buttons, the real dialogs and the real
# service against `_build_test_app()`'s isolated temp-dir SQLite file (see
# that fixture's `get_subscriptions_db_path` patch -- never the user's own
# database), so they measure the wiring rather than a mock of it.


async def _wait_for_dialog(host, dialog_type, pilot, *, ticks: int = 60):
    """Return the modal `dialog_type` once the flow's worker has pushed it.

    The write flows `await push_screen_wait(...)`, so the dialog appears a
    few ticks after the button press rather than synchronously.
    """
    for _ in range(ticks):
        await pilot.pause()
        if isinstance(host.screen, dialog_type):
            return host.screen
    raise AssertionError(f"{dialog_type.__name__} never opened")


async def _wait_until(pilot, predicate, *, ticks: int = 80) -> bool:
    for _ in range(ticks):
        await pilot.pause()
        if predicate():
            return True
    return False


def _label_plain(widget) -> str:
    """The text a markup-rendering label actually paints.

    `Static`/`Label` parse markup, so a name carrying Rich syntax is only
    safe if it was escaped on the way in. Re-parsing the stored content
    here is what proves that: an unescaped `[bold]` disappears into a
    style, an escaped one survives as literal text.
    """
    renderable = widget.renderable
    raw = getattr(renderable, "plain", None)
    if raw is None:
        raw = str(renderable)
        return Text.from_markup(raw).plain
    return raw


@pytest.mark.asyncio
async def test_creating_a_watchlist_from_the_tree_shows_it_without_a_refresh():
    """AC #1. The rail must show the new watchlist on its own -- the whole
    point of wiring `create` is that the only watchlists that can exist stop
    being ones seeded outside the app.
    """
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        service = app.watchlist_bundle_service
        assert service.list_watchlists() == []

        screen.query_one("#wl-tree-new", Button).press()
        dialog = await _wait_for_dialog(host, WatchlistNameDialog, pilot)
        dialog.query_one("#watchlist-name-input", Input).value = "Morning AI Brief"
        dialog.query_one("#watchlist-name-submit", Button).press()

        assert await _wait_until(pilot, lambda: bool(service.list_watchlists()))
        rows = service.list_watchlists()
        assert [row["name"] for row in rows] == ["Morning AI Brief"]

        watchlist_id = rows[0]["id"]
        assert await _wait_until(
            pilot, lambda: bool(screen.query(f"#wl-tree-node-watchlist-{watchlist_id}"))
        ), "the new watchlist must appear in the rail with no manual refresh"


@pytest.mark.asyncio
async def test_an_empty_name_is_rejected_with_a_visible_reason():
    """AC #7. Not a silent no-op: the dialog stays open and says why."""
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]

        screen.query_one("#wl-tree-new", Button).press()
        dialog = await _wait_for_dialog(host, WatchlistNameDialog, pilot)
        dialog.query_one("#watchlist-name-input", Input).value = "   "
        dialog.query_one("#watchlist-name-submit", Button).press()
        await pilot.pause()
        await pilot.pause()

        assert isinstance(host.screen, WatchlistNameDialog), (
            "an invalid name must not dismiss the prompt"
        )
        error = _label_plain(dialog.query_one("#watchlist-name-error", Static))
        assert "cannot be empty" in error
        assert app.watchlist_bundle_service.list_watchlists() == []


@pytest.mark.asyncio
async def test_a_duplicate_name_is_rejected_and_the_reason_escapes_the_name():
    """AC #7, both halves at once.

    The duplicate is reported rather than silently suffixed to "X (2)" by
    `_unique_name` -- and because the reported name is user-authored free
    text, the reason must render it as literal characters. Unescaped remote
    and user titles have shipped as bugs on this screen before.
    """
    app = _build_test_app()
    app.watchlist_bundle_service.create("[bold red]Alpha[/bold red]")

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert await _wait_until(pilot, lambda: bool(screen._tree_watchlists))

        screen.query_one("#wl-tree-new", Button).press()
        dialog = await _wait_for_dialog(host, WatchlistNameDialog, pilot)
        dialog.query_one("#watchlist-name-input", Input).value = (
            "[bold red]alpha[/bold red]"
        )
        dialog.query_one("#watchlist-name-submit", Button).press()
        await pilot.pause()
        await pilot.pause()

        assert isinstance(host.screen, WatchlistNameDialog)
        error = _label_plain(dialog.query_one("#watchlist-name-error", Static))
        assert "already exists" in error
        assert "[bold red]alpha[/bold red]" in error, (
            "the rejected name must render as literal text, not as markup"
        )
        assert len(app.watchlist_bundle_service.list_watchlists()) == 1


@pytest.mark.asyncio
async def test_renaming_a_watchlist_updates_the_rail():
    """AC #2, rename half."""
    app = _build_test_app()
    watchlist = app.watchlist_bundle_service.create("Mroning AI Brief")

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert await _wait_until(pilot, lambda: bool(screen._tree_watchlists))

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=watchlist["id"]))
        )
        await pilot.pause()
        await pilot.pause()

        rename_button = screen.query_one("#wl-tree-rename", Button)
        assert not rename_button.disabled
        rename_button.press()

        dialog = await _wait_for_dialog(host, WatchlistNameDialog, pilot)
        assert dialog.query_one("#watchlist-name-input", Input).value == (
            "Mroning AI Brief"
        ), "the prompt should start from the current name"
        dialog.query_one("#watchlist-name-input", Input).value = "Morning AI Brief"
        dialog.query_one("#watchlist-name-submit", Button).press()

        service = app.watchlist_bundle_service
        assert await _wait_until(
            pilot,
            lambda: [row["name"] for row in service.list_watchlists()]
            == ["Morning AI Brief"],
        )
        assert await _wait_until(
            pilot,
            lambda: any(
                "Morning AI Brief" in str(button.label)
                for button in screen.query(Button)
                if (button.id or "").startswith("wl-tree-node-watchlist-")
            ),
        )
        # The rename must also reach the scope-derived copy, not just the
        # rail: `_tree_scope_label` and `_resolve_breadcrumb_labels` both
        # read `_tree_watchlists`, which a rename leaves stale until the
        # reload re-resolves it.
        assert (
            _static_text(screen.query_one("#wl-feeds-scope-heading", Static))
            == "Feeds in Morning AI Brief (0)"
        )
        assert screen._breadcrumb_labels == ["Morning AI Brief"]


@pytest.mark.asyncio
async def test_deleting_a_watchlist_says_what_happens_to_its_sources_first():
    """AC #2, delete half: the count and the destination are stated before
    the user commits, and the name is escaped on the way into the message.
    """
    app = _build_test_app()
    service = app.watchlist_bundle_service
    watchlist = service.create("[bold]Danger[/bold]")
    db = service._db
    for index in range(2):
        service.add_source(
            watchlist["id"],
            db.add_subscription(
                name=f"Feed {index}", type="rss", source=f"https://{index}.example/f"
            ),
        )

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert await _wait_until(pilot, lambda: bool(screen._tree_watchlists))

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=watchlist["id"]))
        )
        await pilot.pause()
        await pilot.pause()
        screen.query_one("#wl-tree-delete", Button).press()

        dialog = await _wait_for_dialog(host, ConfirmationDialog, pilot)
        message = Text.from_markup(dialog.message).plain
        assert "[bold]Danger[/bold]" in message, (
            "the watchlist name must reach the prompt as literal text"
        )
        assert "2 sources are not deleted" in message
        assert "Unassigned" in message

        dialog.query_one("#cancel-button", Button).press()
        assert await _wait_until(
            pilot, lambda: not isinstance(host.screen, ConfirmationDialog)
        )
        assert len(service.list_watchlists()) == 1, "Cancel must not delete anything"


@pytest.mark.asyncio
async def test_deleting_a_watchlist_never_orphans_its_sources_into_invisibility():
    """AC #3. Deleting cascades only the membership rows, so the sources
    survive -- but survival is worthless if nothing in the tree can reach
    them. They must land under the permanent Unassigned root, which is what
    that root exists for.
    """
    app = _build_test_app()
    service = app.watchlist_bundle_service
    watchlist = service.create("Morning AI Brief")
    db = service._db
    source_ids = [
        db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f"),
        db.add_subscription(name="Krebs", type="rss", source="https://b.example/f"),
    ]
    for source_id in source_ids:
        service.add_source(watchlist["id"], source_id)
    assert service.list_unassigned_source_rows() == []

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert await _wait_until(pilot, lambda: bool(screen._tree_watchlists))

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=watchlist["id"]))
        )
        await pilot.pause()
        await pilot.pause()
        screen.query_one("#wl-tree-delete", Button).press()

        dialog = await _wait_for_dialog(host, ConfirmationDialog, pilot)
        dialog.query_one("#confirm-button", Button).press()

        assert await _wait_until(pilot, lambda: service.list_watchlists() == [])
        assert {row["id"] for row in service.list_unassigned_source_rows()} == set(
            source_ids
        )
        # And the screen actually shows them: the scope moves to Unassigned,
        # whose rows are exactly the sources the deleted watchlist held.
        assert await _wait_until(
            pilot, lambda: screen.tree_scope == TreeScope(kind="unassigned")
        )
        assert {row["id"] for row in screen.scoped_source_rows()} == set(source_ids)
        assert screen.query("#wl-tree-node-unassigned")
        assert not screen.query(f"#wl-tree-node-watchlist-{watchlist['id']}")


@pytest.mark.asyncio
async def test_adding_a_source_to_a_watchlist_from_the_tree():
    """AC #4, add half. The picker offers only sources that are not already
    members, so adding one twice is not something the UI can even ask for.
    """
    app = _build_test_app()
    service = app.watchlist_bundle_service
    watchlist = service.create("Morning AI Brief")
    db = service._db
    member = db.add_subscription(
        name="Already In", type="rss", source="https://in.example/f"
    )
    candidate = db.add_subscription(
        name="ArXiv", type="rss", source="https://a.example/f"
    )
    service.add_source(watchlist["id"], member)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert await _wait_until(pilot, lambda: bool(screen._tree_watchlists))

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=watchlist["id"]))
        )
        await pilot.pause()
        await pilot.pause()
        screen.query_one("#wl-tree-add-source", Button).press()

        dialog = await _wait_for_dialog(host, WatchlistSourcePickerDialog, pilot)
        assert dialog.query(f"#wl-add-source-option-{candidate}")
        assert not dialog.query(f"#wl-add-source-option-{member}"), (
            "an existing member must not be offered again"
        )
        dialog.query_one(f"#wl-add-source-option-{candidate}", Button).press()

        assert await _wait_until(
            pilot, lambda: set(service.list_sources(watchlist["id"])) == {member, candidate}
        )
        assert service.list_unassigned_source_rows() == []


@pytest.mark.asyncio
async def test_removing_a_source_from_a_watchlist_keeps_the_source():
    """AC #4, remove half -- and the other side of AC #3: a removed source
    is still reachable, it just moves to Unassigned.
    """
    app = _build_test_app()
    service = app.watchlist_bundle_service
    watchlist = service.create("Morning AI Brief")
    db = service._db
    source_id = db.add_subscription(
        name="ArXiv", type="rss", source="https://a.example/f"
    )
    service.add_source(watchlist["id"], source_id)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert await _wait_until(pilot, lambda: bool(screen._tree_watchlists))

        screen.post_message(
            TreeScopeChanged(
                TreeScope(
                    kind="source", watchlist_id=watchlist["id"], source_id=source_id
                )
            )
        )
        await pilot.pause()
        await pilot.pause()
        remove_button = screen.query_one("#wl-tree-remove-source", Button)
        assert not remove_button.disabled
        remove_button.press()

        assert await _wait_until(
            pilot, lambda: service.list_sources(watchlist["id"]) == []
        )
        assert [row["id"] for row in service.list_unassigned_source_rows()] == [
            source_id
        ]
        # The scope fell back to the parent watchlist rather than sitting on
        # a source node that no longer exists.
        assert await _wait_until(
            pilot,
            lambda: screen.tree_scope
            == TreeScope(kind="watchlist", watchlist_id=watchlist["id"]),
        )


@pytest.mark.asyncio
async def test_the_server_backend_disables_all_five_verbs_with_a_stated_reason():
    """AC #5. Not cosmetic hiding: `SourceUpdateRequest` carries no
    `group_ids`, neither group request carries members, and all of them are
    `extra="forbid"` -- so there is no wire path at all, and the screen says
    so rather than offering an action that cannot be sent.
    """
    app = _build_test_app()
    app.watchlist_bundle_service.create("Morning AI Brief")

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert await _wait_until(pilot, lambda: bool(screen._tree_watchlists))
        watchlist_id = screen._tree_watchlists[0]["id"]
        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=watchlist_id))
        )
        await pilot.pause()
        await pilot.pause()
        # Locally, four of the five are live on a watchlist scope.
        assert not screen.query_one("#wl-tree-rename", Button).disabled

        screen.runtime_backend = "server"
        await pilot.pause()
        await pilot.pause()

        for action_id in (
            "#wl-tree-new",
            "#wl-tree-rename",
            "#wl-tree-delete",
            "#wl-tree-add-source",
            "#wl-tree-remove-source",
        ):
            button = screen.query_one(action_id, Button)
            assert button.disabled, f"{action_id} must be disabled on the server backend"
            assert "no wire path" in str(button.tooltip)

        note = screen.query_one("#wl-tree-actions-unavailable", Static)
        assert "Switch the backend to Local" in _label_plain(note)


@pytest.mark.asyncio
async def test_the_verbs_are_disabled_when_the_bundle_service_is_missing():
    """The same degrade-don't-crash contract every other caller of
    `_watchlist_bundle_service()` follows -- and the same disabled-with-a-
    reason treatment, rather than buttons that look live over a runtime that
    cannot service them.
    """
    app = _build_test_app()
    app.watchlist_bundle_service = None

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        for action_id in ("#wl-tree-new", "#wl-tree-rename", "#wl-tree-remove-source"):
            button = screen.query_one(action_id, Button)
            assert button.disabled
            assert "unavailable" in str(button.tooltip)


def test_every_watchlist_bundle_service_method_has_a_production_caller():
    """AC #6, enforced rather than asserted once by hand.

    Five of these methods were complete, tested, and reachable from nothing
    at all before this task. A future slice that quietly drops the last
    caller of one should fail here rather than be rediscovered as dead code
    with a green suite.

    Resolved through the AST rather than by grepping for `.create(`: a plain
    text scan matches `completions.create(` in `OCR_Backends` and
    `os.rename(` in `Chat_Functions`, so it would report a caller for
    `create` and `rename` even with every real call deleted -- verified by
    mutation. This instead follows the two ways the service is actually
    reached (`self._watchlist_bundle_service()` and the
    `watchlist_bundle_service` attribute on the app) plus any local bound to
    one of them, so `self._controller.list_sources(...)` -- a different
    object with a colliding method name, in the same file -- is not counted.
    """
    import ast
    import inspect
    import warnings
    from pathlib import Path

    from tldw_chatbook.Subscriptions.watchlist_bundle_service import (
        WatchlistBundleService,
    )

    class _BundleServiceCalls(ast.NodeVisitor):
        def __init__(self) -> None:
            self.aliases: set[str] = set()
            self.called: set[str] = set()

        def _is_service(self, node: ast.AST) -> bool:
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_watchlist_bundle_service"
            ):
                return True
            if isinstance(node, ast.Attribute) and node.attr == "watchlist_bundle_service":
                return True
            return isinstance(node, ast.Name) and node.id in self.aliases

        def visit_Assign(self, node: ast.Assign) -> None:
            if self._is_service(node.value):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.aliases.add(target.id)
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Attribute) and self._is_service(node.func.value):
                self.called.add(node.func.attr)
            self.generic_visit(node)

    service_file = Path(inspect.getfile(WatchlistBundleService)).resolve()
    package_root = service_file.parents[1]

    called: set[str] = set()
    # `ast.parse` re-emits each file's own SyntaxWarnings (stray escape
    # sequences in unrelated modules); they are pre-existing and not this
    # test's subject, so they are silenced rather than left to bury the
    # assertion message below in noise.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        for path in package_root.rglob("*.py"):
            if path.resolve() == service_file:
                continue
            visitor = _BundleServiceCalls()
            visitor.visit(ast.parse(path.read_text(encoding="utf-8")))
            called |= visitor.called

    public_methods = {
        name
        for name, member in vars(WatchlistBundleService).items()
        if not name.startswith("_")
        and callable(getattr(member, "__func__", member))
    }
    # Guard the guard: if the reflection above ever stops seeing the class's
    # own methods, the emptiness check below would pass vacuously.
    assert {"create", "rename", "delete", "add_source", "remove_source"} <= (
        public_methods
    )

    uncalled = sorted(public_methods - called)
    assert uncalled == [], (
        f"WatchlistBundleService methods with no production caller: {uncalled}"
    )


@pytest.mark.asyncio
async def test_a_failed_tree_write_start_does_not_wedge_later_writes():
    """Qodo #3 on PR #989: `_tree_write_active` could stick True forever.

    The flag was raised before `flow_factory()` ran, and is lowered only by
    `_run_tree_write`'s `finally`. If building the flow raised synchronously
    that worker never started, so the flag stayed up and every later
    create/rename/delete returned at the guard -- silently, for the life of
    the screen.
    """
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test() as pilot:
        await pilot.pause()
        screen = host.screen_stack[-1]

        def exploding_factory():
            raise RuntimeError("flow could not be built")

        screen._start_tree_write(exploding_factory)
        await pilot.pause()

        assert screen._tree_write_active is False, (
            "a write that never started must leave the guard down, or every "
            "later watchlist action is silently swallowed"
        )

        ran = []

        async def working_flow():
            ran.append(True)

        screen._start_tree_write(working_flow)
        for _ in range(20):
            await pilot.pause()
            if ran:
                break
        assert ran, "the next write must still be able to start"
