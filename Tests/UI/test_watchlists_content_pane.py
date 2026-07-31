import pytest

pytestmark = pytest.mark.unit


def _render_to_console(renderable, *, width: int = 100) -> tuple[str, str]:
    """Render through a real `rich.console.Console` and return (plain, ansi).

    Whole-branch review: `str(Text)` is not evidence about what a user sees.
    It shows the characters but says nothing about which of them were
    *interpreted* -- and interpretation is the entire question when the body
    is remote text that happens to be bracket-shaped. Rendering through a
    Console and reading both the painted characters and the style codes is
    what actually distinguishes "rendered as text" from "parsed as markup".
    """
    from rich.console import Console

    console = Console(
        width=width, record=True, color_system="standard", force_terminal=True
    )
    console.print(renderable)
    return console.export_text(clear=False), console.export_text(styles=True)


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


def test_markup_shaped_body_is_rendered_as_characters_not_interpreted():
    """The property that actually matters: bracket-shaped remote text is
    painted, not parsed -- and ordinary bracket-shaped prose is not mangled.

    This replaces a pair of tests that asserted `"\\\\[bold red]" in out`,
    i.e. that `rich.markup.escape` had run. Escaping protected nothing on
    this path -- `Text.append` never parses markup and `Static(Text)` does
    not re-parse it, so there was no parser to defend against -- while
    `escape` prefixes a backslash on anything tag-shaped, so every markdown
    link `[docs](url)` and every `[sic]` in a real feed reached the reader
    with a stray backslash in it. Those tests pinned that corruption in
    place. What must actually hold is asserted here instead, through a real
    Console: the tag text arrives verbatim, and it styles nothing.
    """
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import render_article

    plain, ansi = _render_to_console(render_article({
        "title": "[bold red]not a style[/]",
        "source_name": "Hostile Feed",
        "content": "[link=evil]click[/link] then [docs](https://example.test)",
        "content_kind": "article",
        "content_format": "text",
    }))

    # The characters reach the screen exactly as the feed wrote them...
    assert "[bold red]not a style[/]" in plain
    assert "[link=evil]click[/link]" in plain
    # ...and nothing was interpreted: no red was applied by the `bold red`
    # tag, and the `link=` tag emitted no OSC-8 hyperlink.
    assert "\x1b[31m" not in ansi, "the [bold red] tag must not have styled anything"
    assert "\x1b]8;;" not in ansi, "the [link=...] tag must not have become a link"
    # And ordinary prose is not corrupted -- the regression the escaping
    # caused for every markdown link and every "[sic]" on the common path.
    assert "[docs](https://example.test)" in plain
    assert "\\[" not in plain


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
    """The two kinds must not render through the same arm by accident.

    Whole-branch review: this test used to pin only ONE arm. Its article
    assertion was `"%" not in article`, and `render_change` emits "%" only
    when `change_percentage` is present -- which the article fixture has no
    reason to carry -- so `_RENDERERS = {"article": render_change, "change":
    render_change}` passed the whole suite. Use the same "words"
    discriminator `test_unknown_kind_falls_back_to_article_without_raising`
    below already had to adopt for exactly this reason: it comes from
    `render_article`'s meta line and `render_change` never emits it under any
    input, so both arms are now pinned in both directions.
    """
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import render_for

    change = str(render_for({
        "title": "site", "content": "+ x", "content_kind": "change",
        "change_percentage": 3.0, "change_type": "text",
    }))
    article = str(render_for({
        "title": "post", "content": "prose", "content_kind": "article",
    }))

    # A discriminator only the change arm emits...
    assert "3" in change and "%" in change
    assert "%" not in article
    # ...and one only the article arm emits.
    assert "words" in article
    assert "words" not in change


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


def test_a_markdown_body_is_rendered_as_markdown_not_as_raw_source():
    """Whole-branch review (Minor): `content_format` had no consumer, so a
    body captured as markdown was shown to the reader as its source --
    literal `#` heading marks and `[text](url)` link syntax.
    """
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import render_article

    plain, _ansi = _render_to_console(render_article({
        "title": "Release notes",
        "source_name": "Anthropic News",
        "content": "# Heading\n\nSee [the docs](https://example.test) for *more*.",
        "content_kind": "article",
        "content_format": "markdown",
    }))

    assert "Heading" in plain
    assert "the docs" in plain
    assert "#" not in plain, "the heading marker must be consumed, not shown"
    assert "[the docs](https://example.test)" not in plain, (
        "link syntax must be consumed, not shown"
    )


def test_a_plain_text_body_is_never_run_through_the_markdown_renderer():
    """The other half of the same decision: a body that is NOT markdown must
    keep every character it was captured with, `#` and all.
    """
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import render_article

    plain, _ansi = _render_to_console(render_article({
        "title": "Plain",
        "source_name": "Feed",
        "content": "# not a heading, just a hash",
        "content_kind": "article",
        "content_format": "text",
    }))

    assert "# not a heading, just a hash" in plain


def test_change_headline_states_the_diff_summary():
    """Whole-branch review (Minor): `diff_summary` was carried all the way
    through normalization with no consumer at all. It is the engine's own
    one-line account of the change, which is what this headline is for.
    """
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import render_change

    out = str(render_change({
        "title": "anthropic.com/news",
        "content": "+ a\n- b",
        "content_kind": "change",
        "change_percentage": 12.0,
        "change_type": "structural",
        "diff_summary": "2 lines changed",
    }))

    assert "2 lines changed" in out


def test_change_with_no_body_explains_why():
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import render_change

    out = str(render_change({
        "title": "site", "content": None, "content_kind": "change",
        "change_percentage": 5.0, "change_type": "text",
    }))
    assert "no body captured" in out.lower()


def test_diff_lines_with_markup_shaped_text_keep_our_colour_and_gain_none():
    """Same property as the article case, for the one place this module does
    apply a style: the diff line must be green because it starts with `+`,
    and red must be entirely absent -- the `[bold red]` tag inside it styled
    nothing, because nothing parsed it.
    """
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import render_change

    plain, ansi = _render_to_console(render_change({
        "title": "site",
        "content": "+ [bold red]injected[/]",
        "content_kind": "change",
        "change_percentage": 1.0,
        "change_type": "text",
    }))

    assert "+ [bold red]injected[/]" in plain
    assert "\\[" not in plain
    assert "\x1b[32m" in ansi, "a `+` diff line is still coloured green by us"
    assert "\x1b[31m" not in ansi, "the [bold red] tag must not have styled anything"


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
    from Tests.UI.app_factory import _build_test_app
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
    from Tests.UI.app_factory import _build_test_app
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
    from Tests.UI.app_factory import _build_test_app

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
    from Tests.UI.app_factory import _build_test_app
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
    from Tests.UI.app_factory import _build_test_app
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
    from Tests.UI.app_factory import _build_test_app

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
async def test_navigate_item_is_a_noop_when_a_text_input_has_focus():
    """Pins `_navigate_item`'s own defensive branch, isolated from `Input`'s
    key handling.

    This is NOT what protects a real keypress today: `Input._on_key`
    already stops a printable key before it can ever reach this screen's
    BINDINGS resolution -- confirmed empirically:
    `test_typing_j_in_the_search_input_does_not_navigate` does not go red
    when the `isinstance` check in `_navigate_item` is deleted, because
    that test drives a real keypress and `Input` already stopped it first.
    The check is kept as defense-in-depth for any direct caller (and this
    repo has precedent for a bare-letter `priority=True` binding --
    `SearchRAGWindow`'s `f`/`focus_search` -- which would bypass `Input`'s
    protection entirely and make this guard load-bearing overnight). Calling
    `action_next_item()` directly, bypassing the key-event pipeline
    entirely, is what actually isolates the guard: with it removed, this is
    the one test that goes red, because nothing else stops a direct call
    from navigating while focus sits on the search box.
    """
    from textual.widgets import Input

    from Tests.UI.test_destination_shells import DestinationHarness
    from Tests.UI.app_factory import _build_test_app

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
async def test_j_skips_items_hidden_by_a_filter_and_does_not_mark_them_read():
    """Task 6 fix round 1, Important #1.

    `_navigate_item` must walk `ItemsPane.displayed_items()` -- the SAME
    filtered/searched sequence the table renders -- not the screen's raw
    unfiltered `_loaded_items`. Otherwise, with a search query active, `j`
    could open, and silently mark read, an item that is not on screen at
    all: marking read is destructive (it drops the item out of the unread
    bucket) and the user would have no way to know it happened for an item
    they never saw.
    """
    from Tests.UI.test_destination_shells import DestinationHarness
    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.Subscriptions.item_persist import persist_subscription_item
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import ContentPane
    from textual.widgets import Static

    app = _build_test_app()
    db = app.local_watchlists_service._db()
    source_id = db.add_subscription(
        name="Summit Route", type="rss", source="https://summitroute.com/blog/feed.xml"
    )
    with db.transaction() as conn:
        for index, title in enumerate(["Keep alpha", "Hide me", "Keep beta"]):
            persist_subscription_item(
                conn,
                source_id,
                {
                    "url": f"https://summitroute.com/blog/2024/filter-item-{index}/",
                    "title": title,
                    "content": f"body {index}",
                    "content_hash": f"hash-filter-{index}",
                    "status": "new",
                },
                run_id=None,
                now=f"2026-07-28T09:0{index}:00+00:00",
            )

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen, pane = await _mount_items_screen(pilot, host)
        content_pane = screen.query_one("#watchlists-content-pane", ContentPane)

        pane.search_query = "Keep"
        await pilot.pause(0.3)

        displayed = pane.displayed_items()
        assert {item["title"] for item in displayed} == {"Keep alpha", "Keep beta"}, (
            "the search filter must hide 'Hide me' from the displayed sequence"
        )
        assert len(displayed) == 2

        pane.select_and_reveal(displayed[0])
        await pilot.pause(0.3)
        assert screen._selected_content_item["id"] == displayed[0]["id"]

        await pilot.press("j")
        await pilot.pause(0.3)

        assert screen._selected_content_item["id"] == displayed[1]["id"], (
            "j must move to the next VISIBLE item, skipping the one the "
            "filter hides entirely"
        )
        body = content_pane.query_one("#content-body", Static)
        assert displayed[1]["title"] in str(body.renderable)

        # The filtered-out item must never have been opened, and therefore
        # never marked read.
        hidden_raw_id = next(
            item["item_id"] for item in screen._loaded_items if item["title"] == "Hide me"
        )
        reviewed_raw_ids = {row["id"] for row in db.get_new_items(status="reviewed", limit=10)}
        assert hidden_raw_id not in reviewed_raw_ids, (
            "j must never open -- and therefore never mark read -- an item "
            "hidden by the active filter"
        )


@pytest.mark.asyncio
async def test_j_keeps_the_reader_the_pane_selection_and_the_cursor_in_sync():
    """Task 6 fix round 1, Important #2.

    Before this fix, `_navigate_item` updated only the reader, never
    `ItemsPane.selected_item` or the table's cursor. Since `selected_item`
    is a plain `reactive` with no `always_update`, a later click on the row
    the reader had just left was silently swallowed: the reactive saw no
    change (it was already set to that same item from before `j`/`k` ever
    ran) and never re-posted `ItemSelected`. Asserts all three agree after
    `j`, and that a subsequent click on the row the reader left IS honoured.
    """
    from textual.widgets import DataTable, Static

    from Tests.UI.test_destination_shells import DestinationHarness
    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import ContentPane

    app = _build_test_app()
    db = app.local_watchlists_service._db()
    _seed_three_items(db)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen, pane = await _mount_items_screen(pilot, host)
        items = screen._loaded_items
        content_pane = screen.query_one("#watchlists-content-pane", ContentPane)
        table = pane.query_one("#items-table", DataTable)

        pane.select_and_reveal(items[0])
        await pilot.pause(0.3)
        assert table.cursor_row == 0

        await pilot.press("j")
        await pilot.pause(0.3)

        assert screen._selected_content_item["id"] == items[1]["id"], (
            "the reader must have moved to the second item"
        )
        assert pane.selected_item is not None
        assert pane.selected_item["id"] == items[1]["id"], (
            "ItemsPane.selected_item must follow the reader, not stay stuck "
            "on the previous item"
        )
        assert table.cursor_row == 1, (
            "the table's cursor must follow the reader too, so the "
            "highlighted row and the open item are never out of sync"
        )
        body = content_pane.query_one("#content-body", Static)
        assert items[1]["title"] in str(body.renderable)

        # Now select the row the reader just left (row 0) -- the exact
        # scenario Important #2 reported as silently swallowed.
        pane.select_item_by_id(str(items[0]["id"]))
        await pilot.pause(0.3)

        assert screen._selected_content_item["id"] == items[0]["id"], (
            "selecting the row the reader just left must be honoured, not "
            "swallowed by a stale selected_item value that never changed"
        )
        body = content_pane.query_one("#content-body", Static)
        assert items[0]["title"] in str(body.renderable)


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
    from Tests.UI.app_factory import _build_test_app
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


# --- Whole-branch review fixes ---------------------------------------------


@pytest.mark.asyncio
async def test_j_and_k_move_forward_and_back_under_a_status_filter():
    """CRITICAL. A seam between Task 5 and Task 6, present in neither alone.

    Reproduced live with the Items filter set to New and three items shown:
    click the middle one, press `j`, and the reader jumped BACKWARDS -- and
    `k` then did nothing for the rest of the session.

    Task 5's `patch_item["status"] = "reviewed"` mutates the very dict
    `ItemsPane.items` holds, so the item the user just opened stopped
    matching the New filter the instant it was opened. Task 6's
    `_navigate_item` then could not find the open item in the displayed list,
    left `index = -1`, and used it anyway: `j` computed `-1 + 1 = 0` and
    opened the FIRST item, `k` computed `-2` and silently no-opped -- for
    every subsequent press, since every newly-opened item vanished the same
    way.

    The unfiltered path already passed, which is why this shipped.
    """
    from Tests.UI.test_destination_shells import DestinationHarness
    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import ContentPane
    from textual.widgets import Static

    app = _build_test_app()
    db = app.local_watchlists_service._db()
    _seed_three_items(db)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen, pane = await _mount_items_screen(pilot, host)
        content_pane = screen.query_one("#watchlists-content-pane", ContentPane)

        pane.status_filter = "new"
        await pilot.pause(0.3)
        displayed = pane.displayed_items()
        assert len(displayed) == 3, "all three seeded items are still new"

        middle = displayed[1]
        pane.select_and_reveal(middle)
        await pilot.pause(0.5)
        assert screen._selected_content_item["id"] == middle["id"]

        # The open item must not disappear from its own list just because
        # opening it marked it read.
        assert [item["id"] for item in pane.displayed_items()] == [
            item["id"] for item in displayed
        ], "opening an item must not remove it from the displayed list"

        await pilot.press("j")
        await pilot.pause(0.5)
        assert screen._selected_content_item["id"] == displayed[2]["id"], (
            "j must move FORWARD to the next displayed item, not back to the first"
        )
        body = content_pane.query_one("#content-body", Static)
        assert displayed[2]["title"] in str(body.renderable)

        await pilot.press("k")
        await pilot.pause(0.5)
        assert screen._selected_content_item["id"] == displayed[1]["id"], (
            "k must still work after a j, not be dead for the session"
        )


@pytest.mark.asyncio
async def test_the_open_item_survives_a_rebuild_of_the_filtered_table():
    """The other half of the CRITICAL fix: the pin in `_filtered_items`.

    A recompose (changing the search box, reloading items) re-derives the
    rows from scratch. Without pinning the selection, the item the user is
    reading is dropped out of the table under them the moment its status no
    longer matches the active filter -- the reader shows an article that has
    no row.
    """
    from Tests.UI.test_destination_shells import DestinationHarness
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    db = app.local_watchlists_service._db()
    _seed_three_items(db)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen, pane = await _mount_items_screen(pilot, host)

        pane.status_filter = "new"
        await pilot.pause(0.3)
        open_item = pane.displayed_items()[0]

        pane.select_and_reveal(open_item)
        await pilot.pause(0.5)
        assert str(open_item.get("status")).lower() == "reviewed", (
            "opening the item must have marked it read -- the precondition"
        )

        # Force a genuine rebuild of the rows while it is still open.
        pane.search_query = "Nav item"
        await pilot.pause(0.4)

        assert open_item["id"] in {item["id"] for item in pane.displayed_items()}, (
            "the item the reader is showing must still have a row"
        )
        assert screen._selected_content_item["id"] == open_item["id"]


@pytest.mark.asyncio
async def test_k_with_nothing_open_goes_to_the_last_item_not_nowhere():
    """"The current item is not in the list" is its own case, not index -1.

    With nothing open, the old code computed `-1 + delta`, so `k` produced
    `-2`, failed the bounds check, and silently did nothing at all -- while
    `j` produced `0` and opened the first item, which looked fine and hid the
    real defect.
    """
    from Tests.UI.test_destination_shells import DestinationHarness
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    db = app.local_watchlists_service._db()
    _seed_three_items(db)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen, pane = await _mount_items_screen(pilot, host)
        assert screen._selected_content_item is None, "nothing open yet"

        await pilot.press("k")
        await pilot.pause(0.5)

        displayed = pane.displayed_items()
        assert screen._selected_content_item is not None, (
            "k with nothing open must open something, not silently no-op"
        )
        assert screen._selected_content_item["id"] == displayed[-1]["id"], (
            "with no current position, 'previous' means the last item"
        )


@pytest.mark.asyncio
async def test_opening_an_item_repaints_its_status_cell_in_the_table():
    """The Items table never showed what you had read.

    Rows are built once, in `ItemsPane.compose()`, and the mark-read-on-open
    path deliberately never recomposes (Task 5: a recompose destroys the live
    table). So `patch_item`'s in-place mutation was invisible: the Status
    column read "new" for every item the user had opened until they left the
    tab. Visible with no filter at all.
    """
    from textual.widgets import DataTable

    from Tests.UI.test_destination_shells import DestinationHarness
    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.UI.Watchlists_Modules.items_pane import ItemsPane

    app = _build_test_app()
    db = app.local_watchlists_service._db()
    _seed_three_items(db)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen, pane = await _mount_items_screen(pilot, host)
        items = screen._loaded_items
        table = pane.query_one("#items-table", DataTable)
        row_key = str(items[0]["id"])

        assert table.get_row(row_key)[2] == "new"

        pane.select_item_by_id(row_key)
        await pilot.pause(0.6)

        assert table.get_row(row_key)[2] == "reviewed", (
            "the Status cell must show what the user has actually read"
        )
        # And it must have got there WITHOUT a recompose (Task 5's CRITICAL).
        assert screen.query_one("#watchlists-items-pane", ItemsPane) is pane


@pytest.mark.asyncio
async def test_opening_an_item_does_not_cancel_unrelated_background_work():
    """`run_worker(exclusive=True)` with no `group=` lands in the default
    group, shared by ~25 call sites on this screen -- including the "Check
    now" fetch. Since Phase D the mark-read worker fires on every selection
    and every `j`/`k`, so reading an item cancelled a network fetch the user
    had just been toasted about.
    """
    import asyncio

    from textual.worker import WorkerState

    from Tests.UI.test_destination_shells import DestinationHarness
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    db = app.local_watchlists_service._db()
    _seed_three_items(db)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen, pane = await _mount_items_screen(pilot, host)
        items = screen._loaded_items

        async def _unrelated_long_fetch() -> None:
            await asyncio.sleep(3)

        # Exactly the shape of `_check_now_source`'s call site: exclusive,
        # default group.
        unrelated = screen.run_worker(_unrelated_long_fetch(), exclusive=True)
        await pilot.pause(0.1)
        assert unrelated.state is not WorkerState.CANCELLED

        pane.select_item_by_id(str(items[0]["id"]))
        await pilot.pause(0.6)

        assert unrelated.state is not WorkerState.CANCELLED, (
            "reading an item must not cancel unrelated in-flight work"
        )


@pytest.mark.asyncio
async def test_the_content_chevron_off_the_read_tab_neither_collapses_nor_persists():
    """The gate force-collapses CONTENT off the Read tab, which renders a
    real, focusable `▸ Content` button. Clicking it (or pressing `z` with it
    focused) ran the toggle against the REAL `region_layout` rather than the
    derived view, so nothing visibly changed, the user's genuine preference
    silently flipped to collapsed, and `"content"` was written to
    `[watchlists].collapsed_regions` on disk -- honoured forever, since the
    Phase D migration marker is already set. That permanently recreates the
    exact state the migration exists to repair, from a control that looked
    inert.
    """
    from Tests.UI.test_destination_shells import DestinationHarness
    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.UI.Watchlists_Modules.region_layout import Region

    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]

        screen.active_section = "sources"
        await pilot.pause(0.3)
        assert screen.query("#wl-header-content"), "the gate collapsed CONTENT"
        assert not screen.region_layout.is_collapsed(Region.CONTENT), (
            "the real preference is still expanded -- the precondition"
        )

        await pilot.click("#wl-header-content")
        await pilot.pause(0.3)

        assert not screen.region_layout.is_collapsed(Region.CONTENT), (
            "clicking the gated chevron must not flip the real preference"
        )
        assert Region.CONTENT not in (screen._last_persisted_collapsed or frozenset()), (
            "and it must never reach the persisted collapse set"
        )

        # `z` with CONTENT focused is the same action by another route.
        screen.focused_region = Region.CONTENT
        screen.action_toggle_region()
        await pilot.pause(0.3)
        assert not screen.region_layout.is_collapsed(Region.CONTENT)

        # And back on Read the reader is still there.
        screen.active_section = "items"
        await pilot.pause(0.3)
        assert screen.query("#wl-region-content")


@pytest.mark.asyncio
async def test_a_workbench_rebuild_keeps_the_items_filter_search_and_selection():
    """`_build_detail_pane` seeded only `.items`, unlike the sibling Sources,
    Runs and Notifications panes, which all re-seed their selection (and, for
    Sources, the whole create-form draft).

    So every workbench rebuild silently reset the user's view to "all items,
    nothing selected, empty search box". `region_layout` is
    `recompose=True`, so ANY collapse/expand -- `z`, `[`, `]`, a chevron
    click -- rebuilds every region and constructs a brand new `ItemsPane`;
    that is the deterministic trigger used here. The reported route was
    "Mark unread", whose `refresh=True` ends in `_refresh_overview_data()`
    setting `overview_data` (`reactive(recompose=True)`): same rebuild, but
    it only fires when the overview counts actually change value, which is
    why it is not what this test drives.
    """
    from Tests.UI.test_destination_shells import DestinationHarness
    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.UI.Watchlists_Modules.items_pane import ItemsPane

    app = _build_test_app()
    db = app.local_watchlists_service._db()
    _seed_three_items(db)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen, pane = await _mount_items_screen(pilot, host)

        pane.status_filter = "new"
        pane.search_query = "Nav item"
        await pilot.pause(0.4)
        open_item = pane.displayed_items()[0]
        pane.select_and_reveal(open_item)
        await pilot.pause(0.5)

        # A rail toggle -- nothing to do with Items at all.
        screen.action_toggle_left_rail()
        await pilot.pause(0.5)

        rebuilt = screen.query_one("#watchlists-items-pane", ItemsPane)
        assert rebuilt is not pane, (
            "the precondition: the rail toggle really did rebuild the pane"
        )
        assert rebuilt.status_filter == "new", "the status filter must survive"
        assert rebuilt.search_query == "Nav item", "the search box must survive"
        assert rebuilt.selected_item is not None, "the selection must survive"
        assert rebuilt.selected_item["id"] == open_item["id"]
        assert open_item["id"] in {item["id"] for item in rebuilt.displayed_items()}, (
            "and the open item must still have a row in the rebuilt table"
        )


@pytest.mark.asyncio
async def test_mark_unread_refuses_to_overwrite_an_item_ingested_by_the_real_gesture():
    """Data loss, driven end to end through the gestures a user actually makes.

    Re-review, Important: the first version of this test set the item up with
    `_update_item_status(..., patch_item=item)`, and `patch_item=` is passed
    by exactly ONE caller in the whole app -- `_mark_item_read_on_open`. Never
    by Ingest, never by Ignore. So it only exercised the branch where the
    reader's cached dict happens to be fresh, which the real flow never
    produces, and it certified a data-loss path as closed while the loss was
    still there.

    The real sequence, and what it used to do:

        select the item        -> reader opens it, marks it read
        Ingest (Inspector)     -> DB says `ingested`; the Items table
                                  correctly re-renders as `ingested`; but
                                  `ContentPane.item` / `_selected_content_item`
                                  still say `reviewed`, because Ingest passes
                                  no `patch_item=`
        press "Mark unread"    -> the guard reads the stale `reviewed`, does
                                  not fire, and destroys the ingest

    So this drives `IngestRequested(screen.selected_entity)` exactly as the
    Inspector button does, then presses the real `#content-mark-unread-button`,
    and asserts the DB is untouched. It also asserts the staleness itself, so
    a future change that happens to keep the dict fresh cannot make this test
    silently stop testing the thing it exists for.
    """
    from textual.widgets import Button

    from Tests.UI.test_destination_shells import DestinationHarness
    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import ContentPane
    from tldw_chatbook.UI.Watchlists_Modules.inspector_pane import IngestRequested

    app = _build_test_app()
    db = app.local_watchlists_service._db()
    _seed_three_items(db)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen, pane = await _mount_items_screen(pilot, host)
        item = screen._loaded_items[0]
        raw_id = item["item_id"]

        # 1. Open it in the reader, exactly as a click does.
        pane.select_item_by_id(str(item["id"]))
        await pilot.pause(0.5)
        assert screen.selected_entity is not None
        assert screen.selected_entity["id"] == item["id"]

        # 2. Ingest it through the Inspector's own message. NO `patch_item=`:
        #    that is the whole point.
        screen.post_message(IngestRequested(screen.selected_entity))
        for _ in range(40):
            await pilot.pause(0.05)
            if raw_id in {row["id"] for row in db.get_new_items(status="ingested", limit=10)}:
                break
        assert raw_id in {
            row["id"] for row in db.get_new_items(status="ingested", limit=10)
        }, "the precondition: the real Ingest gesture really did write `ingested`"

        # The staleness this fix exists for -- assert it, do not assume it.
        content_pane = screen.query_one("#watchlists-content-pane", ContentPane)
        assert str(content_pane.item.get("status")).lower() != "ingested", (
            "the reader's cached dict is stale after Ingest -- if this ever "
            "stops being true, this test is no longer covering the real bug"
        )

        # 3. Press the real button.
        screen.query_one("#content-mark-unread-button", Button).press()
        await pilot.pause(0.8)

        assert raw_id in {
            row["id"] for row in db.get_new_items(status="ingested", limit=10)
        }, "Mark unread must not overwrite an ingested item's status"
        assert raw_id not in {
            row["id"] for row in db.get_new_items(status="new", limit=10)
        }, "the ingested item must not have been pushed back to new"


@pytest.mark.asyncio
async def test_mark_unread_still_works_on_an_item_that_is_merely_read():
    """The other side of the guard: the normal case must not be refused.

    A re-read that refuses everything would pass the data-loss test above and
    break the feature.
    """
    from textual.widgets import Button

    from Tests.UI.test_destination_shells import DestinationHarness
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    db = app.local_watchlists_service._db()
    _seed_three_items(db)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen, pane = await _mount_items_screen(pilot, host)
        item = screen._loaded_items[0]
        raw_id = item["item_id"]

        pane.select_item_by_id(str(item["id"]))
        for _ in range(40):
            await pilot.pause(0.05)
            if raw_id in {row["id"] for row in db.get_new_items(status="reviewed", limit=10)}:
                break
        assert raw_id in {row["id"] for row in db.get_new_items(status="reviewed", limit=10)}

        screen.query_one("#content-mark-unread-button", Button).press()
        for _ in range(40):
            await pilot.pause(0.05)
            if raw_id in {row["id"] for row in db.get_new_items(status="new", limit=10)}:
                break

        assert raw_id in {
            row["id"] for row in db.get_new_items(status="new", limit=10)
        }, "Mark unread must still work on an item that is simply read"


@pytest.mark.asyncio
async def test_mark_unread_fails_closed_when_the_status_cannot_be_confirmed():
    """An unanswerable question must not resolve in favour of the destructive
    branch.

    Marking unread is a convenience the user can simply repeat; overwriting an
    ingest is not recoverable. So if the backend cannot be asked what the item
    currently is, the write is refused rather than attempted.

    Stubs `get_item_status`, the single-item read the guard now uses (PR #1091
    review, F1). It used to stub `list_items`, which the guard no longer calls
    at all -- so left as it was, this test would have gone on passing while
    testing nothing.
    """
    from textual.widgets import Button

    from Tests.UI.test_destination_shells import DestinationHarness
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    db = app.local_watchlists_service._db()
    _seed_three_items(db)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen, pane = await _mount_items_screen(pilot, host)
        item = screen._loaded_items[0]
        raw_id = item["item_id"]

        pane.select_item_by_id(str(item["id"]))
        for _ in range(40):
            await pilot.pause(0.05)
            if raw_id in {r["id"] for r in db.get_new_items(status="reviewed", limit=10)}:
                break
        assert raw_id in {r["id"] for r in db.get_new_items(status="reviewed", limit=10)}

        async def _unavailable(**_kwargs):
            raise RuntimeError("backend unavailable")

        screen._controller.get_item_status = _unavailable

        screen.query_one("#content-mark-unread-button", Button).press()
        await pilot.pause(0.8)

        assert raw_id in {
            r["id"] for r in db.get_new_items(status="reviewed", limit=10)
        }, "an unconfirmable status must be left exactly as it was"
        assert raw_id not in {
            r["id"] for r in db.get_new_items(status="new", limit=10)
        }, "the write must be refused, not attempted, when it cannot be checked"


@pytest.mark.asyncio
async def test_a_persisted_body_reaches_the_reader_end_to_end():
    """Task 1's entire reason for existing, finally pinned.

    Every other UI assertion on this branch checks only the TITLE, so
    hard-coding `"content": None` in the normalizer leaves them all green --
    the reader could render "no body captured" for every item, end to end,
    and only one unit test in `Tests/Subscriptions/` would notice. Assert the
    seeded BODY, through the real persist -> normalize -> load -> select ->
    render path.
    """
    from textual.widgets import Static

    from Tests.UI.test_destination_shells import DestinationHarness
    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import ContentPane

    app = _build_test_app()
    db = app.local_watchlists_service._db()
    _seed_three_items(db)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen, pane = await _mount_items_screen(pilot, host)
        item = screen._loaded_items[0]
        # `_seed_three_items` writes "body for nav item <n>" alongside
        # "Nav item <n>", so the body is derivable from the title.
        index = item["title"].rsplit(" ", 1)[-1]

        pane.select_item_by_id(str(item["id"]))
        await pilot.pause(0.5)

        content_pane = screen.query_one("#watchlists-content-pane", ContentPane)
        rendered, _ansi = _render_to_console(
            content_pane.query_one("#content-body", Static).renderable, width=160
        )

        assert item["title"] in rendered
        assert f"body for nav item {index}" in rendered, (
            "the persisted body must reach the reader, not just the title"
        )
        assert "no body captured" not in rendered


@pytest.mark.asyncio
async def test_the_mark_unread_button_is_compact():
    """A default `Button` is three rows tall (border, label, border), and
    CONTENT has about nine usable rows -- the same third of the pane the
    tooltip fix reclaimed, spent again on chrome.
    """
    from textual.app import App
    from textual.widgets import Button

    from tldw_chatbook.UI.Watchlists_Modules.content_pane import ContentPane

    class _PaneHost(App):
        def compose(self):
            pane = ContentPane()
            pane.item = {"title": "x", "content": "y", "content_kind": "article"}
            yield pane

    app = _PaneHost()
    async with app.run_test() as pilot:
        await pilot.pause()
        button = app.query_one("#content-mark-unread-button", Button)
        assert button.compact, "the reader's button must not cost three rows"


# --- PR #1091 review ---------------------------------------------------------


#: The page depth the unread guard's old lookup used
#: (`_ITEM_STATUS_LOOKUP_LIMIT`, deleted along with that lookup). Written as a
#: literal here on purpose: the seeding below has to be provably deeper than
#: the window that used to truncate the guard's answer, and importing a
#: constant that no longer exists would just make the test unrunnable.
_LEGACY_STATUS_LOOKUP_LIMIT = 500


@pytest.mark.asyncio
async def test_mark_unread_refuses_an_ingest_that_sits_beyond_a_lookup_page():
    """PR #1091 review, F1: the guard must not be defeated by a page size.

    The guard used to answer "is this item ingested?" by calling
    `list_items(status="ingested", limit=500)` once per blocking status and
    looking for the item in the result.
    `LocalWatchlistsService.list_items` slices to the requested window, so an
    ingested item outside the first page was simply not in the answer -- and
    the guard read that absence as "not ingested" and let `Mark unread`
    overwrite the ingest. Absence from a truncated page is not proof of
    absence.

    So this seeds one source with more ingested items than that window held,
    arranges for the target to be the OLDEST of them (`get_new_items` orders
    `created_at DESC`, so it lands past the end of the first page), and then
    drives the real gestures: open the item, Ingest it through the
    Inspector's own message, press the real `Mark unread` button. The DB must
    be untouched.
    """
    from textual.widgets import Button

    from Tests.UI.full_app_destination_context import (
        full_app_destination_context,
        wait_for_selector,
    )
    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.Subscriptions.item_persist import persist_subscription_item
    from tldw_chatbook.UI.Watchlists_Modules.inspector_pane import IngestRequested

    app = _build_test_app()
    db = app.local_watchlists_service._db()
    source_id = db.add_subscription(
        name="Busy Feed", type="rss", source="https://busy.test/feed.xml"
    )

    filler_ids = []
    with db.transaction() as conn:
        # The target is seeded FIRST and dated oldest, so once it is ingested
        # it sorts last in the `created_at DESC` listing the old lookup paged
        # through. It stays "new" for now -- that is the only status the Items
        # pane lists, and the test has to reach it through the real UI.
        target_id = persist_subscription_item(
            conn,
            source_id,
            {
                "url": "https://busy.test/the-one-that-matters/",
                "title": "The one that matters",
                "content": "body of the item that must survive",
                "content_hash": "hash-beyond-page-target",
            },
            run_id=None,
            now="2020-01-01T00:00:00.0000+00:00",
        )
        # Zero-padded counter in the timestamp: `created_at` is TEXT, so
        # ORDER BY compares lexicographically and every filler must sort
        # above the target deterministically, not by insertion luck.
        for index in range(_LEGACY_STATUS_LOOKUP_LIMIT + 20):
            filler_ids.append(
                persist_subscription_item(
                    conn,
                    source_id,
                    {
                        "url": f"https://busy.test/filler-{index}/",
                        "title": f"Filler {index}",
                        "content_hash": f"hash-beyond-page-{index}",
                    },
                    run_id=None,
                    now=f"2026-07-28T09:00:00.{index:04d}+00:00",
                )
            )
    # `persist_subscription_item` always writes "new" on insert, so the
    # fillers are moved to the blocking status separately.
    db.bulk_update_items(filler_ids, "ingested")
    assert (
        len(db.get_new_items(subscription_id=source_id, status="ingested", limit=10_000))
        > _LEGACY_STATUS_LOOKUP_LIMIT
    ), "the fixture must be deeper than the page the old lookup could see"

    host = full_app_destination_context(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen, pane = await _mount_items_screen(pilot, host, expected_count=1)
        item = screen._loaded_items[0]
        assert item["item_id"] == target_id, "only the target is still unread"

        pane.select_item_by_id(str(item["id"]))
        await pilot.pause(0.5)
        assert screen.selected_entity is not None

        screen.post_message(IngestRequested(screen.selected_entity))
        for _ in range(40):
            await pilot.pause(0.05)
            if db.get_item_status(target_id) == "ingested":
                break
        assert db.get_item_status(target_id) == "ingested", (
            "the precondition: the real Ingest gesture wrote `ingested`"
        )
        # And it really is out of reach of a single 500-row page.
        page = db.get_new_items(
            subscription_id=source_id,
            status="ingested",
            limit=_LEGACY_STATUS_LOOKUP_LIMIT,
        )
        assert target_id not in {row["id"] for row in page}, (
            "the target must fall outside the page the old lookup read, or "
            "this test proves nothing about the truncation bug"
        )

        await wait_for_selector(
            screen,
            pilot,
            "#content-mark-unread-button",
            timeout=4.0,
        )
        screen.query_one("#content-mark-unread-button", Button).press()
        await pilot.pause(0.8)

    assert db.get_item_status(target_id) == "ingested", (
        "Mark unread must not overwrite an ingest just because the item sits "
        "beyond the first page of a status listing"
    )


@pytest.mark.asyncio
async def test_soloing_content_then_leaving_read_leaves_a_centre_region_expanded():
    """PR #1091 review, F2: the workbench must never render an empty centre.

    Soloing CONTENT sets `collapsed` to `{FEEDS, ITEMS}`. Off the Read tab
    `_visible_region_layout` force-collapses CONTENT as well, and it used to
    add that to the SOLO view -- collapsing all three centre regions, so the
    workbench mounted three header buttons and nothing else. Deriving from
    the pre-solo baseline instead leaves the user's real layout showing, and
    their solo is still there when they come back to Read.
    """
    from Tests.UI.test_destination_shells import DestinationHarness
    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.UI.Watchlists_Modules.region_layout import (
        CENTRE_REGIONS,
        Region,
    )

    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]

        screen.active_section = "items"
        await pilot.pause(0.3)
        screen.focused_region = Region.CONTENT
        screen.action_solo_region()
        await pilot.pause(0.3)
        assert screen.region_layout.solo_region is Region.CONTENT, (
            "the precondition: CONTENT really is soloed"
        )
        assert screen.query("#wl-region-content")

        screen.active_section = "sources"
        await pilot.pause(0.3)

        rendered = screen._visible_region_layout()
        expanded = [r for r in CENTRE_REGIONS if not rendered.is_collapsed(r)]
        assert expanded, (
            "leaving Read with CONTENT soloed collapsed every centre region: "
            f"rendered layout was {sorted(r.value for r in rendered.collapsed)}"
        )
        # Not just the derived value -- what actually mounted.
        assert screen.query("#wl-region-feeds") or screen.query("#wl-region-items"), (
            "at least one centre region must be mounted expanded, not three "
            "header buttons over an empty centre"
        )

        screen.active_section = "items"
        await pilot.pause(0.3)
        assert screen.region_layout.solo_region is Region.CONTENT, (
            "the gate is display-only: the user's solo must survive the trip"
        )
        assert screen.query("#wl-region-content")
        assert not screen.query("#wl-region-feeds"), "and it must still be a solo"


@pytest.mark.asyncio
async def test_solo_on_content_off_the_read_tab_is_refused():
    """PR #1091 review, F2 (second half) / TASK-1344 AC#2.

    The collapsed `▸ Content` header is focusable on every tab, so `Z` could
    still solo a region the user cannot see -- collapsing FEEDS and ITEMS
    around it and leaving nothing on screen. The chevron and `z` are already
    refused there; `Z` must be too.
    """
    from Tests.UI.test_destination_shells import DestinationHarness
    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.UI.Watchlists_Modules.region_layout import Region

    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]

        screen.active_section = "sources"
        await pilot.pause(0.3)
        before = screen.region_layout

        screen.focused_region = Region.CONTENT
        screen.action_solo_region()
        await pilot.pause(0.3)

        assert screen.region_layout is before or screen.region_layout == before, (
            "solo on the gated CONTENT region must not touch the real layout"
        )
        assert screen.region_layout.solo_region is None
        assert screen.query("#wl-region-feeds") or screen.query("#wl-region-items"), (
            "and the centre must still have something in it"
        )


def test_a_hostile_markdown_body_never_emits_a_terminal_hyperlink():
    """PR #1091 review, F3: `Markdown` is a parser, so defend at it.

    `rich.markdown.Markdown` defaults to `hyperlinks=True`, which turns
    `[label](url)` from a remote feed into a real OSC-8 terminal hyperlink:
    an attacker-chosen label over a destination the reader cannot see. The
    reader passes `hyperlinks=False`, so the label renders and the URL
    renders beside it as ordinary visible text.

    Asserts through a real Console, since the question is what was
    *interpreted*, not what characters exist. Also pins that the markdown
    branch was actually taken -- otherwise a regression that stopped
    rendering markdown at all would leave this test green for the wrong
    reason.
    """
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import render_article

    plain, ansi = _render_to_console(
        render_article(
            {
                "title": "Fresh drop",
                "source_name": "Hostile Feed",
                "content": (
                    "[Anthropic docs](https://evil.test/steal)\n\n"
                    "[click](javascript:alert)\n\n"
                    '<a href="https://evil.test/raw">raw html</a>\n\n'
                    "<script>alert(1)</script>\n\n"
                    "<https://evil.test/autolink>"
                ),
                "content_kind": "article",
                "content_format": "markdown",
            }
        ),
        width=200,
    )

    assert "\x1b]8;" not in ansi, (
        "no markdown link may become a real terminal hyperlink -- the label "
        "is attacker-chosen and would hide its destination"
    )
    # The markdown branch really did run: a plain-text render would still
    # show the raw link syntax.
    assert "[Anthropic docs](" not in plain, "the body must have been parsed as markdown"
    # And the destination is disclosed rather than hidden behind the label.
    assert "https://evil.test/steal" in plain
    assert "https://evil.test/autolink" in plain
