import pytest

pytestmark = pytest.mark.unit


def test_article_renders_title_source_and_body():
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import render_article

    out = str(render_article({
        "title": "Claude Opus 4.5 is now available",
        "source_name": "Anthropic News",
        "published_date": "2026-07-28",
        "content": "The model is available in the API today.",
        "content_kind": "article",
        "content_format": "text",
    }))

    assert "Claude Opus 4.5 is now available" in out
    assert "Anthropic News" in out
    assert "The model is available in the API today." in out


def test_article_with_no_body_explains_why():
    """`content` is NULL for every pre-existing item. Never render blank."""
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import render_article

    out = str(render_article({
        "title": "An item from before bodies were captured",
        "source_name": "Old Feed",
        "content": None,
        "content_kind": "article",
    }))

    assert "no body captured" in out.lower()
    assert "re-check" in out.lower()


def test_untrusted_body_markup_is_escaped():
    """Remote content reaches a Textual renderable; it must not be markup.

    NOTE: these assertions require the *escaped* (backslash-prefixed) form
    specifically, not `"...[bold red]..." in out or "\\[bold red]" in out`.
    That "or" is a tautology: `rich.markup.escape` only prepends a
    backslash before the bracket, so the unescaped substring is always
    contained inside the escaped one too, and the assertion would pass
    whether or not escaping actually ran. Verified empirically while
    implementing this test (mutation check: deleting the `escape_markup`
    call around the body left the original two-branch "or" assertions
    green). Requiring the backslash form is what actually goes red when
    the escaping is removed.
    """
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import render_article

    out = str(render_article({
        "title": "[bold red]not a style[/]",
        "source_name": "Hostile Feed",
        "content": "[link=evil]click[/link]",
        "content_kind": "article",
    }))

    assert "\\[bold red]" in out
    assert "\\[link=evil]" in out


@pytest.mark.asyncio
async def test_content_pane_shows_placeholder_with_no_item():
    """The pane must actually compose the placeholder `Static`, not merely
    start with `item is None`.

    The original version of this test asserted only `pane.item is None` --
    true the instant a `ContentPane` is constructed, and true whether or not
    `compose()` ever runs or produces the right widget. Mount it for real and
    read back the rendered text.
    """
    from textual.app import App
    from textual.widgets import Static

    from tldw_chatbook.UI.Watchlists_Modules.content_pane import ContentPane

    class _PaneHost(App):
        def compose(self):
            yield ContentPane()

    app = _PaneHost()
    async with app.run_test() as pilot:
        await pilot.pause()
        placeholder = app.query_one("#content-empty", Static)
        assert str(placeholder.renderable) == "Select an item to read it."


def test_change_renders_percent_type_and_diff_lines():
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import render_change

    out = str(render_change({
        "title": "anthropic.com/news",
        "source_name": "Anthropic",
        "content": "+ Opus 4.5 available\n- Opus 4.1 available",
        "content_kind": "change",
        "content_format": "diff",
        "change_percentage": 12.0,
        "change_type": "structural",
    }))

    assert "12" in out and "%" in out
    assert "structural" in out
    assert "+ Opus 4.5 available" in out
    assert "- Opus 4.1 available" in out


def test_dispatch_selects_the_renderer_by_kind():
    """The two kinds must not render through the same arm by accident."""
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import render_for

    change = str(render_for({
        "title": "site", "content": "+ x", "content_kind": "change",
        "change_percentage": 3.0, "change_type": "text",
    }))
    article = str(render_for({
        "title": "post", "content": "prose", "content_kind": "article",
    }))

    # A discriminator only the change arm emits.
    assert "3" in change and "%" in change
    assert "%" not in article


def test_unknown_kind_falls_back_to_article_without_raising():
    """An escaping exception in compose() exits the whole app.

    Asserting only `"odd" in out` does not pin the fallback to the article
    arm specifically: with no `change_percentage`/`change_type` on this
    item, `render_change`'s headline falls back to the bare word "changed"
    (no "%" at all) and its diff loop happily emits an unprefixed "x" line
    too, so a fallback-to-`render_change` default would satisfy that
    assertion just as well -- confirmed empirically while fixing this test:
    `"%" not in out` stays green either way for *this* input, since
    `render_change` only emits "%" when `change_percentage` is present.
    Assert the article-only word-count marker instead
    (`f"{len(body.split())} words"` from `render_article`'s meta line):
    `render_change` never emits "words" under any input, so this is the
    discriminator that actually goes red under the wrong default.
    """
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import render_for

    out = str(render_for({"title": "odd", "content": "x", "content_kind": "wat"}))
    assert "odd" in out
    assert "words" in out


def test_change_with_no_body_explains_why():
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import render_change

    out = str(render_change({
        "title": "site", "content": None, "content_kind": "change",
        "change_percentage": 5.0, "change_type": "text",
    }))
    assert "no body captured" in out.lower()


def test_diff_lines_with_hostile_markup_are_escaped():
    """Diff lines are remote content too; styling them must not mean
    interpreting them as markup.

    Same reasoning as `test_untrusted_body_markup_is_escaped` above: the
    assertion must require the *escaped* (backslash-prefixed) form, since the
    unescaped substring is always contained inside the escaped one and a
    looser assertion would pass either way.
    """
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import render_change

    out = str(render_change({
        "title": "site",
        "content": "+ [bold red]injected[/]",
        "content_kind": "change",
        "change_percentage": 1.0,
        "change_type": "text",
    }))

    assert "\\[bold red]" in out


def test_content_region_is_not_collapsed_by_default_now_it_has_a_reader():
    """Task 4: through Phase C, CONTENT held only a placeholder stub and
    started collapsed. Now it hosts the real reader, so a fresh screen must
    show it expanded like every other region.
    """
    from tldw_chatbook.UI.Watchlists_Modules.region_layout import Region
    from tldw_chatbook.UI.Screens.watchlists_collections_screen import (
        WatchlistsCollectionsScreen,
    )

    # `Reactive` does not expose its default value publicly in this Textual
    # version (`8.2.8`) -- only the private `_default` attribute holds it;
    # `.default` (as an earlier draft of this test used) does not exist and
    # raises `AttributeError` before ever reaching the real assertion.
    default = WatchlistsCollectionsScreen.region_layout._default
    layout = default() if callable(default) else default
    assert Region.CONTENT not in layout.collapsed


@pytest.mark.asyncio
async def test_selecting_an_item_renders_it_in_the_content_region():
    """Selection must reach the reader through the real screen wiring.

    Drives the real `ItemsPane` -> `ItemSelected` -> screen handler ->
    `ContentPane` path (not a direct call into `ContentPane`), following the
    harness the Phase C screen tests use (`test_watchlists_item_actions.py`).
    """
    from textual.widgets import Static

    from Tests.UI.test_destination_shells import DestinationHarness
    from Tests.UI.test_screen_navigation import _build_test_app
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import ContentPane
    from tldw_chatbook.UI.Watchlists_Modules.items_pane import ItemsPane

    item = {
        "id": 7,
        "title": "Claude Opus 4.5 is now available",
        "source_name": "Anthropic News",
        "content": "The model is available in the API today.",
        "content_kind": "article",
        "content_format": "text",
    }

    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        screen.active_section = "items"
        await pilot.pause(0.2)

        items_pane = screen.query_one("#watchlists-items-pane", ItemsPane)
        items_pane.items = [item]
        await pilot.pause(0.2)

        items_pane.select_item_by_id("7")
        await pilot.pause(0.3)

        content_pane = screen.query_one("#watchlists-content-pane", ContentPane)
        body = content_pane.query_one("#content-body", Static)
        rendered = str(body.renderable)
        assert "Claude Opus 4.5 is now available" in rendered
        assert "The model is available in the API today." in rendered

        # Selecting `None` (e.g. the row no longer matches anything) must
        # clear the pane back to its placeholder, not leave the stale body.
        items_pane.select_item_by_id("does-not-exist")
        await pilot.pause(0.3)

        assert content_pane.item is None
        empty_placeholder = content_pane.query_one("#content-empty", Static)
        assert str(empty_placeholder.renderable) == "Select an item to read it."


@pytest.mark.asyncio
async def test_content_region_is_gated_to_the_items_read_tab():
    """Fix round 1 (coordinator review): per the approved design spec
    ("### Tabs"), only Read uses the three-pane split -- Sources, Runs,
    Rules, and Artifacts take the full centre width, with no
    collection->feed->item relationship to show a reader for. This
    implementation's Items tab is that Read tab (the only section
    `ItemSelected` ever comes from), so CONTENT must occupy real space
    there and nowhere else, regardless of the user's stored collapse
    preference for it.
    """
    from Tests.UI.test_destination_shells import DestinationHarness
    from Tests.UI.test_screen_navigation import _build_test_app
    from tldw_chatbook.UI.Watchlists_Modules.region_layout import Region

    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]

        # Default section is "overview" -- not Read -- so CONTENT must be
        # gated to its collapsed header despite the un-gated default
        # (`region_layout`) being expanded.
        assert not screen.region_layout.is_collapsed(Region.CONTENT)
        assert screen.query("#wl-header-content")
        assert not screen.query("#wl-region-content")

        screen.active_section = "items"
        await pilot.pause(0.2)
        assert screen.query("#wl-region-content")
        assert not screen.query("#wl-header-content")

        screen.active_section = "sources"
        await pilot.pause(0.2)
        assert screen.query("#wl-header-content")
        assert not screen.query("#wl-region-content")


@pytest.mark.asyncio
async def test_content_region_gating_does_not_clobber_a_real_collapse_preference():
    """The tab gate is display-only: a user's REAL choice to collapse
    CONTENT (made on the Items tab) must survive a trip through a non-Read
    tab and back -- not get silently re-expanded just because the gate
    also forces it collapsed everywhere else.
    """
    from tldw_chatbook.UI.Watchlists_Modules.region_layout import Region

    from Tests.UI.test_destination_shells import DestinationHarness
    from Tests.UI.test_screen_navigation import _build_test_app

    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]

        screen.active_section = "items"
        await pilot.pause(0.2)
        assert screen.query("#wl-region-content"), "CONTENT should start expanded on Items"

        # The user deliberately collapses CONTENT while actually on Items.
        screen._apply_layout(screen.region_layout.toggle(Region.CONTENT))
        await pilot.pause(0.2)
        assert screen.region_layout.is_collapsed(Region.CONTENT)
        assert screen.query("#wl-header-content")

        screen.active_section = "sources"
        await pilot.pause(0.2)
        assert screen.query("#wl-header-content")

        screen.active_section = "items"
        await pilot.pause(0.2)
        assert screen.region_layout.is_collapsed(Region.CONTENT), (
            "the real preference must still read collapsed -- the gate must "
            "never have touched it"
        )
        assert screen.query("#wl-header-content"), (
            "returning to Items must restore the user's own collapse choice, "
            "not silently force CONTENT back open"
        )


# --- Task 6: `j` / `k` item navigation -------------------------------------


def test_j_and_k_are_bound_and_do_not_collide_with_any_ancestor_bindings():
    """Task 6 Step 1: the binding-conflict audit the brief requires before
    adding anything.

    `j`/`k` must be bound on this screen, and those exact keys must not
    already appear in `BaseAppScreen`'s BINDINGS, the app class's BINDINGS,
    or the built-in `DataTable`/`Tree` widgets whose own bindings would
    otherwise swallow the keypress while a table or tree has focus. Reads
    the class attributes directly -- tmux cannot even encode some keys and
    has produced false conclusions about this screen's bindings before, so
    it is not evidence either way.
    """
    from textual.widgets import DataTable, Tree

    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.Navigation.base_app_screen import BaseAppScreen
    from tldw_chatbook.UI.Screens.watchlists_collections_screen import (
        WatchlistsCollectionsScreen,
    )

    def _keys(bindings) -> set[str]:
        keys: set[str] = set()
        for entry in bindings:
            # The screen's own BINDINGS are plain (key, action, description)
            # tuples; Textual's built-in widgets and the app class use
            # `Binding` objects. Handle both rather than assume one shape.
            key = entry.key if hasattr(entry, "key") else entry[0]
            keys.update(key.split(","))
        return keys

    screen_keys = _keys(WatchlistsCollectionsScreen.BINDINGS)
    assert "j" in screen_keys, "j must be bound on WatchlistsCollectionsScreen"
    assert "k" in screen_keys, "k must be bound on WatchlistsCollectionsScreen"

    # `BaseAppScreen` defines no `BINDINGS` of its own, so this resolves
    # through the MRO to Textual's `Screen.BINDINGS` (tab/shift+tab/copy at
    # the time of writing) -- checking the resolved attribute, not assuming
    # BaseAppScreen is empty, is the point of the audit.
    ancestor_keys = _keys(BaseAppScreen.BINDINGS)
    ancestor_keys |= _keys(TldwCli.BINDINGS)
    ancestor_keys |= _keys(DataTable.BINDINGS)
    ancestor_keys |= _keys(Tree.BINDINGS)

    assert "j" not in ancestor_keys, (
        f"j collides with an ancestor/built-in binding: {sorted(ancestor_keys)}"
    )
    assert "k" not in ancestor_keys, (
        f"k collides with an ancestor/built-in binding: {sorted(ancestor_keys)}"
    )


def _seed_three_items(db):
    """Add one source with three "new" items, in a fixed, known order.

    Returns (source_id, [item_id, item_id, item_id]) in insertion order.
    Insertion order is NOT asserted to be the order the screen loads them in
    -- tests below read `screen._loaded_items` back and use ITS order as
    ground truth, since that is the exact list `j`/`k` walk.
    """
    from tldw_chatbook.Subscriptions.item_persist import persist_subscription_item

    source_id = db.add_subscription(
        name="Summit Route", type="rss", source="https://summitroute.com/blog/feed.xml"
    )
    item_ids = []
    with db.transaction() as conn:
        for index in range(3):
            item_id = persist_subscription_item(
                conn,
                source_id,
                {
                    "url": f"https://summitroute.com/blog/2024/nav-item-{index}/",
                    "title": f"Nav item {index}",
                    "content": f"body for nav item {index}",
                    "content_hash": f"hash-jk-{index}",
                    "status": "new",
                },
                run_id=None,
                now=f"2026-07-28T09:0{index}:00+00:00",
            )
            item_ids.append(item_id)
    return source_id, item_ids


async def _mount_items_screen(pilot, host, expected_count: int = 3):
    """Shared setup: switch to Items, wait for the seeded items to load."""
    from tldw_chatbook.UI.Watchlists_Modules.items_pane import ItemsPane

    await pilot.pause(0.2)
    screen = host.screen_stack[-1]
    screen.active_section = "items"
    await pilot.pause(0.3)

    pane = screen.query_one("#watchlists-items-pane", ItemsPane)
    for _ in range(40):
        await pilot.pause()
        if len(pane.items) >= expected_count:
            break
    assert len(pane.items) == expected_count, (
        "all seeded items must reach the Items pane before driving j/k"
    )
    return screen, pane


@pytest.mark.asyncio
async def test_j_and_k_move_to_the_next_and_previous_item_and_update_the_reader():
    """The core Task 6 behaviour: `j` moves forward, `k` moves back, and
    each move updates the reader -- not just some in-memory index -- and
    marks the newly-opened item read exactly as clicking it does (Task 5's
    `_mark_item_read_on_open`).
    """
    from Tests.UI.test_destination_shells import DestinationHarness
    from Tests.UI.test_screen_navigation import _build_test_app
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import ContentPane
    from textual.widgets import Static

    app = _build_test_app()
    db = app.local_watchlists_service._db()
    _source_id, item_ids = _seed_three_items(db)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen, pane = await _mount_items_screen(pilot, host)
        items = screen._loaded_items
        # `items[i]["id"]` is the NORMALIZED id (`local:watchlist_item:<row
        # id>`, see `normalize_watchlist_item`), not the raw DB row id
        # `_seed_three_items` returns -- compare counts, not the raw ids.
        assert len(items) == len(item_ids)

        content_pane = screen.query_one("#watchlists-content-pane", ContentPane)

        pane.select_item_by_id(str(items[0]["id"]))
        await pilot.pause(0.3)
        body = content_pane.query_one("#content-body", Static)
        assert items[0]["title"] in str(body.renderable)
        for _ in range(30):
            await pilot.pause()
            if db.get_new_items(status="reviewed", limit=10):
                break
        # `db.get_new_items` returns raw DB rows keyed by the raw row id
        # (`item_id` on the normalized dict), not the normalized `id` string
        # (`local:watchlist_item:<row id>`) `j`/`k` navigate by.
        assert {row["id"] for row in db.get_new_items(status="reviewed", limit=10)} == {
            items[0]["item_id"]
        }, "selecting the first item must mark it read, same as any other open"

        await pilot.press("j")
        await pilot.pause(0.3)
        assert screen._selected_content_item["id"] == items[1]["id"]
        body = content_pane.query_one("#content-body", Static)
        assert items[1]["title"] in str(body.renderable)
        for _ in range(30):
            await pilot.pause()
            if len(db.get_new_items(status="reviewed", limit=10)) >= 2:
                break
        assert {row["id"] for row in db.get_new_items(status="reviewed", limit=10)} == {
            items[0]["item_id"],
            items[1]["item_id"],
        }, "j must mark the item it moves to read, exactly as clicking would"

        await pilot.press("j")
        await pilot.pause(0.3)
        assert screen._selected_content_item["id"] == items[2]["id"]
        body = content_pane.query_one("#content-body", Static)
        assert items[2]["title"] in str(body.renderable)

        await pilot.press("k")
        await pilot.pause(0.3)
        assert screen._selected_content_item["id"] == items[1]["id"]
        body = content_pane.query_one("#content-body", Static)
        assert items[1]["title"] in str(body.renderable)


@pytest.mark.asyncio
async def test_j_and_k_do_not_raise_at_the_list_boundaries():
    """`k` at the first item and `j` at the last item must be a no-op, not
    an exception -- an exception escaping an event handler exits the whole
    application.
    """
    from Tests.UI.test_destination_shells import DestinationHarness
    from Tests.UI.test_screen_navigation import _build_test_app
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import ContentPane
    from textual.widgets import Static

    app = _build_test_app()
    db = app.local_watchlists_service._db()
    _seed_three_items(db)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen, pane = await _mount_items_screen(pilot, host)
        items = screen._loaded_items
        content_pane = screen.query_one("#watchlists-content-pane", ContentPane)

        first_item = items[0]
        pane.select_item_by_id(str(first_item["id"]))
        await pilot.pause(0.3)

        await pilot.press("k")
        await pilot.pause(0.3)
        assert screen._selected_content_item["id"] == first_item["id"], (
            "k at the first item must not move"
        )
        body = content_pane.query_one("#content-body", Static)
        assert first_item["title"] in str(body.renderable)

        last_item = items[-1]
        pane.select_item_by_id(str(last_item["id"]))
        await pilot.pause(0.3)

        await pilot.press("j")
        await pilot.pause(0.3)
        assert screen._selected_content_item["id"] == last_item["id"], (
            "j at the last item must not move"
        )
        body = content_pane.query_one("#content-body", Static)
        assert last_item["title"] in str(body.renderable)


@pytest.mark.asyncio
async def test_typing_j_in_the_search_input_does_not_navigate():
    """A user typing "j" into the items search box must get the character,
    not next-item navigation -- the failure a user would hit within a
    minute of real use if the focused-input guard were missing.
    """
    from textual.widgets import Input

    from Tests.UI.test_destination_shells import DestinationHarness
    from Tests.UI.test_screen_navigation import _build_test_app

    app = _build_test_app()
    db = app.local_watchlists_service._db()
    _source_id, item_ids = _seed_three_items(db)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen, pane = await _mount_items_screen(pilot, host)
        items = screen._loaded_items
        first_item = items[0]

        pane.select_item_by_id(str(first_item["id"]))
        await pilot.pause(0.3)
        assert screen._selected_content_item["id"] == first_item["id"]

        search_input = pane.query_one("#items-search-input", Input)
        search_input.focus()
        await pilot.pause(0.2)
        assert search_input.has_focus, "the search input must actually hold focus"

        await pilot.press("j")
        await pilot.pause(0.3)

        assert search_input.value == "j", (
            "the focused search box must receive the typed character"
        )
        assert screen._selected_content_item["id"] == first_item["id"], (
            "typing j into a focused text input must not navigate the reader"
        )


@pytest.mark.asyncio
async def test_action_next_item_is_a_noop_when_a_text_input_has_focus():
    """Isolates `_navigate_item`'s own focused-widget guard from `Input`'s
    key handling.

    `Input._on_key` already stops a printable key before it can ever reach
    this screen's BINDINGS resolution -- confirmed empirically: deleting the
    `isinstance(focused, (Input, TextArea))` check in `_navigate_item` does
    NOT turn `test_typing_j_in_the_search_input_does_not_navigate` red,
    because that test drives a real keypress, and the keypress never gets
    as far as `action_next_item` either way. Calling `action_next_item()`
    directly, bypassing the key-event pipeline entirely, is what actually
    isolates the guard: with it removed, this is the one test that goes
    red, because nothing else stops a direct call from navigating while
    focus sits on the search box.
    """
    from textual.widgets import Input

    from Tests.UI.test_destination_shells import DestinationHarness
    from Tests.UI.test_screen_navigation import _build_test_app

    app = _build_test_app()
    db = app.local_watchlists_service._db()
    _source_id, item_ids = _seed_three_items(db)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen, pane = await _mount_items_screen(pilot, host)
        items = screen._loaded_items
        first_item = items[0]

        pane.select_item_by_id(str(first_item["id"]))
        await pilot.pause(0.3)
        assert screen._selected_content_item["id"] == first_item["id"]

        search_input = pane.query_one("#items-search-input", Input)
        search_input.focus()
        await pilot.pause(0.2)
        assert search_input.has_focus, "the search input must actually hold focus"

        screen.action_next_item()
        await pilot.pause(0.2)

        assert screen._selected_content_item["id"] == first_item["id"], (
            "action_next_item() must be a no-op while a text input has "
            "focus, regardless of how it was invoked"
        )


@pytest.mark.asyncio
async def test_j_navigation_does_not_recompose_the_screen():
    """Task 6 must reuse the Task 5 fix, not reintroduce the reload.

    `_update_item_status`'s default refresh path ends in
    `_refresh_overview_data()`, and `overview_data` is
    `reactive({}, recompose=True)` on the screen -- a full rebuild that
    replaces the mounted `ItemsPane`/`ContentPane` wholesale and drops
    focus (Task 5's CRITICAL finding). `j`/`k` reuse `handle_item_selected`,
    which marks read via `_mark_item_read_on_open`'s `refresh=False` +
    `patch_item` path, so the SAME pane instances must survive two
    consecutive `j` presses.
    """
    from Tests.UI.test_destination_shells import DestinationHarness
    from Tests.UI.test_screen_navigation import _build_test_app
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import ContentPane

    app = _build_test_app()
    db = app.local_watchlists_service._db()
    _seed_three_items(db)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen, pane = await _mount_items_screen(pilot, host)
        items = screen._loaded_items
        content_pane_before = screen.query_one("#watchlists-content-pane", ContentPane)

        pane.select_item_by_id(str(items[0]["id"]))
        await pilot.pause(0.3)

        await pilot.press("j")
        await pilot.pause(0.3)
        await pilot.press("j")
        await pilot.pause(0.3)

        assert screen.query_one("#watchlists-items-pane", type(pane)) is pane, (
            "the ItemsPane instance must survive two j presses, not be "
            "rebuilt by a screen-level recompose"
        )
        assert (
            screen.query_one("#watchlists-content-pane", ContentPane)
            is content_pane_before
        ), "the ContentPane instance must survive two j presses too"
        assert screen._selected_content_item["id"] == items[2]["id"]
