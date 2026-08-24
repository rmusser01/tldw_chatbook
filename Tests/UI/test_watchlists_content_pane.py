from unittest.mock import Mock

import pytest

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp

pytestmark = pytest.mark.unit


def _render_to_console(renderable, *, width: int = 100) -> tuple[str, str]:
    """Render through a real `rich.console.Console` and return (plain, ansi).

    Whole-branch review: `str(Text)` is not evidence about what a user sees.
    It shows the characters but says nothing about which of them were
    *interpreted* -- and interpretation is the entire question when the body
    is remote text that happens to be bracket-shaped. Rendering through a
    Console and reading both the painted characters and the style codes is
    what actually distinguishes "rendered as text" from "parsed as markup".

    `file=io.StringIO()` keeps this rendering off real stdout -- without it,
    `force_terminal=True` makes `console.print` write the rendered article to
    the actual test-run stdout on every call (task-1347), which is cosmetic
    noise but noise nonetheless. `record=True` still captures everything
    printed to that buffer, so `export_text()` is unaffected.
    """
    import io

    from rich.console import Console

    console = Console(
        width=width,
        record=True,
        color_system="standard",
        force_terminal=True,
        file=io.StringIO(),
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
    from textual.widgets import Static

    from tldw_chatbook.UI.Watchlists_Modules.content_pane import ContentPane

    class _PaneHost(ConsolidatedCSSApp):
        def compose(self):
            yield ContentPane()

    app = _PaneHost()
    async with app.run_test() as pilot:
        await pilot.pause()
        placeholder = app.query_one("#content-empty", Static)
        assert str(placeholder.renderable) == "Select a feed item to display it here."
        assert not app.query("#content-body-scroll"), (
            "the empty-state path stays a direct placeholder rather than an empty scroller"
        )


@pytest.mark.asyncio
async def test_open_content_wraps_only_the_body_in_a_vertical_scroll():
    from textual.containers import HorizontalScroll, VerticalScroll
    from textual.widgets import Static

    from tldw_chatbook.UI.Watchlists_Modules.content_pane import ContentPane

    class _PaneHost(ConsolidatedCSSApp):
        def compose(self):
            pane = ContentPane()
            pane.item = {
                "title": "Scrollable article",
                "source_name": "Feed",
                "content": "first\n\nlast",
                "content_kind": "article",
                "content_format": "text",
            }
            yield pane

    app = _PaneHost()
    async with app.run_test() as pilot:
        await pilot.pause()
        pane = app.query_one(ContentPane)
        actions = app.query_one("#content-actions", HorizontalScroll)
        body_scroll = app.query_one("#content-body-scroll", VerticalScroll)
        body = app.query_one("#content-body", Static)
        footer = app.query_one("#content-footer")

        assert list(pane.children) == [actions, body_scroll, footer]
        assert list(body_scroll.children) == [body]


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


def test_content_is_not_a_preferred_collapsible_region():
    """The permanent Reader is absent from preferred side-pane state."""
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
    from tldw_chatbook.UI.Watchlists_Modules.article_list import ArticleListPane

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

        items_pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
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
        assert str(empty_placeholder.renderable) == (
            "Select a feed item to display it here."
        )


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

    "Nowhere else" means CONTENT has no DOM presence at all off Read.
    """
    from Tests.UI.test_destination_shells import DestinationHarness
    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.UI.Watchlists_Modules.region_layout import Region

    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]

        # Default section is Read ("items") since task-2513, so CONTENT is
        # mounted at rest...
        assert not screen.region_layout.is_collapsed(Region.CONTENT)
        assert screen.query("#wl-region-content")

        # ...and off Read it must be unmounted despite the un-gated default
        # (`region_layout`) being expanded.
        screen.active_section = "overview"
        await pilot.pause(0.2)
        assert not screen.query("#wl-region-content")

        screen.active_section = "items"
        await pilot.pause(0.2)
        assert screen.query("#wl-region-content")

        screen.active_section = "sources"
        await pilot.pause(0.2)
        assert not screen.query("#wl-region-content")


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
    from tldw_chatbook.UI.Watchlists_Modules.article_list import ArticleListPane

    await pilot.pause(0.2)
    screen = host.screen_stack[-1]
    screen.active_section = "items"
    await pilot.pause(0.3)

    pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
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
    seeded_ids: dict[str, int] = {}
    with db.transaction() as conn:
        for index, title in enumerate(["Keep alpha", "Hide me", "Keep beta"]):
            seeded_ids[title] = persist_subscription_item(
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
        #
        # The id comes from the seed, not from `screen._loaded_items`
        # (task-15463). A non-blank search is part of the QUERY, not a
        # client-side filter -- `_load_items` re-reads with `search="Keep"`
        # and rewrites `_loaded_items` with just the matches -- so reading
        # the hidden item's id out of screen state only ever worked while
        # that reload had not landed within the pause above. Caching the
        # `SubscriptionsDB` instance made the reload land in time and the
        # lookup raised `StopIteration`. The seeded id is the same row, with
        # no dependence on when a background load finishes, and it keeps the
        # assertion honest in the failure case too: an id read back from the
        # `new` bucket would vanish exactly when the bug being guarded
        # against occurred.
        hidden_raw_id = seeded_ids["Hide me"]
        reviewed_raw_ids = {row["id"] for row in db.get_new_items(status="reviewed", limit=10)}
        assert hidden_raw_id not in reviewed_raw_ids, (
            "j must never open -- and therefore never mark read -- an item "
            "hidden by the active filter"
        )
        # Positive half, so a wrong/absent id cannot make the check above
        # pass vacuously: the row is real and still sitting unread.
        unread_raw_ids = {row["id"] for row in db.get_new_items(status="new", limit=10)}
        assert hidden_raw_id in unread_raw_ids, (
            "the hidden item must still be in the unread bucket -- if this "
            "id does not name a real, still-new row, the assertion above "
            "proves nothing"
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
    from textual.widgets import ListView, Static

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
        # TASK-3072 note: the three items share one day bucket, so the
        # ListView's child 0 is the date-group header and the rows sit at
        # indices 1-3 -- every old `cursor_row` assertion shifts by one.
        list_view = pane.query_one("#items-table", ListView)

        pane.select_and_reveal(items[0])
        await pilot.pause(0.3)
        assert list_view.index == 1

        await pilot.press("j")
        await pilot.pause(0.3)

        assert screen._selected_content_item["id"] == items[1]["id"], (
            "the reader must have moved to the second item"
        )
        assert pane.selected_item is not None
        assert pane.selected_item["id"] == items[1]["id"], (
            "ArticleListPane.selected_item must follow the reader, not stay "
            "stuck on the previous item"
        )
        assert list_view.index == 2, (
            "the list's cursor must follow the reader too, so the "
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

        pane.status_filter = "unread"
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
async def test_the_open_item_survives_a_same_page_rebuild():
    """A same-page rebuild pins the open item in `_filtered_items`.

    Reapplying the committed page re-derives its rows from copied item dicts.
    The selected id must remain pinned when its status no longer matches the
    active filter. Query-context changes intentionally invalidate this pin and
    are covered by the pagination provenance tests.
    """
    from Tests.UI.test_destination_shells import DestinationHarness
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    db = app.local_watchlists_service._db()
    _seed_three_items(db)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen, pane = await _mount_items_screen(pilot, host)

        pane.status_filter = "unread"
        await pilot.pause(0.3)
        open_item = pane.displayed_items()[0]

        pane.select_and_reveal(open_item)
        await pilot.pause(0.5)
        assert str(open_item.get("status")).lower() == "reviewed", (
            "opening the item must have marked it read -- the precondition"
        )

        copied_page = [dict(item) for item in pane.items]
        await pane.apply_page_items(copied_page)

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
async def test_opening_an_item_repaints_its_row_in_the_list():
    """The Items list never showed what you had read.

    Rows are built once, in `ArticleListPane.compose()`, and the
    mark-read-on-open path deliberately never recomposes (Task 5: a recompose
    destroys the live list). So `patch_item`'s in-place mutation was
    invisible: the row stayed bold with its unread dot for every item the
    user had opened until they left the tab. Visible with no filter at all.

    TASK-3072 note: what ItemsPane's Status column said in words, the reader
    row says in shape -- unread is a leading dot and bold title, read is
    neither. `_repaint_row` rebuilds the row's `Text` in place from the
    patched dict, which is what the assertions below read back.
    """
    from textual.widgets import ListView

    from Tests.UI.test_destination_shells import DestinationHarness
    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.UI.Watchlists_Modules.article_list import (
        ArticleListPane,
        _ArticleRow,
    )

    app = _build_test_app()
    db = app.local_watchlists_service._db()
    _seed_three_items(db)

    def _row_text(pane, item_id) -> str:
        list_view = pane.query_one("#items-table", ListView)
        for node in list_view.children:
            if isinstance(node, _ArticleRow) and node.item_id_key == str(item_id):
                # task-15776: the row renders itself -- no inner Static.
                return node.render().plain
        raise AssertionError(f"no rendered row for {item_id!r}")

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen, pane = await _mount_items_screen(pilot, host)
        items = screen._loaded_items
        item_id = str(items[0]["id"])

        assert _row_text(pane, item_id).startswith("● "), (
            "precondition: an unread row leads with the unread dot"
        )

        pane.select_item_by_id(item_id)
        await pilot.pause(0.6)

        assert not _row_text(pane, item_id).startswith("● "), (
            "the row must stop showing unread the moment the user has read it"
        )
        # And it must have got there WITHOUT a recompose (Task 5's CRITICAL).
        assert screen.query_one("#watchlists-items-pane", ArticleListPane) is pane


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
async def test_article_focus_is_refused_off_read_without_changing_layout():
    """The permanent Reader exists only on Read, and its focus mode is local."""
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
        assert not screen.query("#wl-region-content")
        preferred_before = screen.region_layout
        effective_before = screen._effective_region_layout
        screen.notify = Mock()
        screen.action_article_focus()
        await pilot.pause(0.3)
        assert screen.region_layout == preferred_before
        assert screen._effective_region_layout == effective_before
        assert screen._article_focus_active is False
        screen.notify.assert_called_once()

        screen.active_section = "items"
        await pilot.pause(0.3)
        assert screen.query("#wl-region-content")
        assert not screen._effective_region_layout.is_collapsed(Region.CONTENT)


@pytest.mark.asyncio
async def test_a_workbench_rebuild_keeps_the_items_filter_search_and_selection():
    """`_build_detail_pane` seeded only `.items`, unlike the sibling Sources,
    Runs and Notifications panes, which all re-seed their selection (and, for
    Sources, the whole create-form draft).

    So every workbench rebuild silently reset the user's view to "all items,
    nothing selected, empty search box". The reported route was
    "Mark unread", whose `refresh=True` ends in `_refresh_overview_data()`
    setting `overview_data` (`reactive(recompose=True)`): same rebuild, but
    it only fires when the overview counts actually change value, which is
    why it is not what this test drives.

    Collapsing and re-expanding Feed Items through its permanent grip really
    remounts the pane, so it exercises the same `_build_detail_pane` seeding.
    """
    from textual.widgets import Button

    from Tests.UI.test_destination_shells import DestinationHarness
    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.UI.Watchlists_Modules.article_list import ArticleListPane
    from tldw_chatbook.UI.Watchlists_Modules.region_layout import Region

    app = _build_test_app()
    db = app.local_watchlists_service._db()
    _seed_three_items(db)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen, pane = await _mount_items_screen(pilot, host)

        pane.status_filter = "unread"
        pane.search_query = "Nav item"
        await pilot.pause(0.4)
        open_item = pane.displayed_items()[0]
        pane.select_and_reveal(open_item)
        await pilot.pause(0.5)

        screen.query_one("#wl-grip-items", Button).press()
        await pilot.pause(0.5)
        assert not screen.query("#watchlists-items-pane")
        assert screen.region_layout.is_collapsed(Region.ITEMS)
        assert screen._effective_region_layout.is_collapsed(Region.ITEMS)

        screen.query_one("#wl-grip-items", Button).press()
        await pilot.pause(0.5)

        rebuilt = screen.query_one("#watchlists-items-pane", ArticleListPane)
        assert rebuilt is not pane, (
            "the precondition: collapsing and re-expanding ITEMS really did "
            "rebuild the pane"
        )
        assert rebuilt.status_filter == "unread", "the status filter must survive"
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
    from Tests.UI.full_app_destination_context import wait_for_selector
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

        # Ingest's `refresh=True` tail sets `overview_data`
        # (`reactive({}, recompose=True)`), which unmounts and remounts the
        # whole screen; querying immediately after the DB write races that
        # recompose. Wait for the pane to resettle before querying it -- the
        # sibling test at :1631 does the same for the identical reason.
        await wait_for_selector(
            screen, pilot, "#watchlists-content-pane", timeout=4.0
        )

        # The staleness this fix exists for -- assert it, do not assume it.
        content_pane = screen.query_one("#watchlists-content-pane", ContentPane)
        assert str(content_pane.item.get("status")).lower() != "ingested", (
            "the reader's cached dict is stale after Ingest -- if this ever "
            "stops being true, this test is no longer covering the real bug"
        )

        # 3. Press the real button.
        await wait_for_selector(
            screen, pilot, "#content-mark-unread-button", timeout=4.0
        )
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
    """Reader actions stay in the established one-row chrome budget."""
    from textual.widgets import Button

    from tldw_chatbook.UI.Watchlists_Modules.content_pane import ContentPane

    class _PaneHost(ConsolidatedCSSApp):
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

    with db.transaction() as conn:
        # The target is seeded FIRST and dated oldest, so once it is ingested
        # it sorts last in the `created_at DESC` listing the old lookup paged
        # through.
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

    def _seed_the_page_deep_fillers() -> None:
        """Bury the target under more ingested items than the old page held.

        TASK-2301 moved this out of the initial fixture and behind the
        target's own Ingest. It used to run up front, because the Items pane
        could only ever list `new` items, so 520 `ingested` fillers were
        invisible to it and the still-`new` target was conveniently the only
        row on screen. The pane now lists EVERY status, newest first, and the
        target is the oldest row in the database by five years -- so seeding
        the fillers first would bury it 520 rows deep in the pane too, and
        the test could no longer reach it through a real gesture at all.

        Seeding them here is also the truer reproduction: the page fills up
        BETWEEN the ingest and the `Mark unread` press, which is exactly the
        situation the guard exists for -- a busy feed moving the item out of
        any page a listing-based lookup could see, while the user is still
        looking at it.
        """
        filler_ids = []
        with db.transaction() as conn:
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
            len(
                db.get_new_items(
                    subscription_id=source_id, status="ingested", limit=10_000
                )
            )
            > _LEGACY_STATUS_LOOKUP_LIMIT
        ), "the fixture must be deeper than the page the old lookup could see"

    host = full_app_destination_context(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen, pane = await _mount_items_screen(pilot, host, expected_count=1)
        item = screen._loaded_items[0]
        assert item["item_id"] == target_id, "the target is the only item so far"

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

        _seed_the_page_deep_fillers()

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
async def test_article_focus_is_transient_and_the_reader_stays_permanent():
    """Article Focus hides side panes without changing preferred state."""
    from Tests.UI.test_destination_shells import DestinationHarness
    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.UI.Watchlists_Modules.region_layout import Region

    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]

        screen.active_section = "items"
        await pilot.pause(0.3)
        preferred_before = screen.region_layout
        screen.action_article_focus()
        await pilot.pause(0.3)
        assert screen._article_focus_active is True
        assert screen.region_layout == preferred_before
        assert screen.query("#wl-region-content")
        assert not screen.query("#wl-region-items")
        for side in (Region.LEFT_RAIL, Region.ITEMS, Region.RIGHT_RAIL):
            assert screen._effective_region_layout.is_collapsed(side)
            assert screen.query(f"#wl-grip-{side.value}")

        screen.active_section = "sources"
        await pilot.pause(0.3)
        assert screen._article_focus_active is False
        assert screen.region_layout == preferred_before
        assert screen.query("#wl-region-items")
        assert not screen.query("#wl-region-content")

        screen.active_section = "items"
        await pilot.pause(0.3)
        assert screen.query("#wl-region-content")
        assert screen.query("#wl-region-items")


@pytest.mark.asyncio
async def test_article_focus_off_read_does_not_change_preferred_or_effective_state():
    """Management tabs reject Article Focus and retain their canvas."""
    from Tests.UI.test_destination_shells import DestinationHarness
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]

        screen.active_section = "sources"
        await pilot.pause(0.3)
        preferred_before = screen.region_layout
        effective_before = screen._effective_region_layout
        screen.notify = Mock()
        screen.action_article_focus()
        await pilot.pause(0.3)

        assert screen.region_layout == preferred_before
        assert screen._effective_region_layout == effective_before
        assert screen._article_focus_active is False
        assert screen.query("#wl-region-items")
        screen.notify.assert_called_once()


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


def test_a_raw_control_byte_in_a_markdown_body_is_stripped_before_rendering():
    """Batch-4 review, I4. Every existing hostile-markdown test (including
    the one directly above) attacks the `Markdown(hyperlinks=False)` OSC-8
    suppression through `[label](url)` LINK SYNTAX -- none embeds a raw
    control byte directly in the markdown source, so nothing had proven the
    `strip_control_characters(raw_body)` call in `render_article`'s markdown
    branch (`_is_markdown(item)` true) does anything at all. Mutation-
    verified by the review: replacing that branch with unstripped
    `str(raw_body or "")` left the whole content-pane suite green.

    A raw OSC-8 sequence has no CommonMark significance -- to the parser it
    is ordinary inline text, so `Markdown` would hand it straight through to
    the terminal exactly as written unless stripped first.
    """
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import render_article

    payload = "before \x1b]8;;http://evil.test\x07label\x1b]8;;\x07 after"
    plain, ansi = _render_to_console(
        render_article(
            {
                "title": "Raw control byte in markdown",
                "source_name": "Hostile Feed",
                "content": payload,
                "content_kind": "article",
                "content_format": "markdown",
            }
        ),
        width=200,
    )
    assert "\x1b" not in plain, "the raw ESC byte must not reach the rendered text"
    assert "\x1b]8;;" not in ansi, (
        "no OSC-8 hyperlink may be manufactured from a raw byte sequence "
        "embedded directly in the markdown source"
    )
    assert "before" in plain and "label" in plain and "after" in plain


# --- TASK-1494: the reader's `[full page]`/`[previous snapshot]` affordances -


@pytest.mark.parametrize("content_kind", ["change", "article"])
@pytest.mark.asyncio
async def test_only_change_items_get_snapshot_affordances_in_inspector(content_kind):
    """Stored-page actions belong to change items in the Inspector only."""
    from textual.widgets import Button

    from tldw_chatbook.UI.Watchlists_Modules.inspector_pane import InspectorPane

    class _PaneHost(ConsolidatedCSSApp):
        def compose(self):
            pane = InspectorPane()
            pane.selected_entity = {
                "entity_kind": "watchlist_item",
                "item_id": 7,
                "title": "Selected item",
                "content_kind": content_kind,
            }
            yield pane

    app = _PaneHost()
    async with app.run_test() as pilot:
        await pilot.pause()
        full_page = app.query("#inspector-full-page-button")
        previous = app.query("#inspector-previous-snapshot-button")
        assert bool(full_page) is (content_kind == "change")
        assert bool(previous) is (content_kind == "change")
        if content_kind == "change":
            assert app.query_one("#inspector-full-page-button", Button).compact
            assert app.query_one("#inspector-previous-snapshot-button", Button).compact


@pytest.mark.parametrize(
    ("button_id", "which"),
    [
        ("inspector-full-page-button", "full_page"),
        ("inspector-previous-snapshot-button", "previous"),
    ],
)
@pytest.mark.asyncio
async def test_inspector_snapshot_buttons_post_existing_request(button_id, which):
    """Both Inspector buttons post the shared screen request with the item."""
    from textual.widgets import Button

    from tldw_chatbook.UI.Watchlists_Modules.inspector_pane import (
        InspectorPane,
        ViewSnapshotRequested,
    )

    captured: list[ViewSnapshotRequested] = []
    item = {
        "entity_kind": "watchlist_item",
        "item_id": 7,
        "title": "anthropic.com/news changed",
        "content_kind": "change",
        "source_id": 7,
        "url": "https://anthropic.com/news",
    }

    class _PaneHost(ConsolidatedCSSApp):
        def compose(self):
            pane = InspectorPane()
            pane.selected_entity = item
            yield pane

        def on_view_snapshot_requested(self, event: ViewSnapshotRequested) -> None:
            captured.append(event)

    app = _PaneHost()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one(f"#{button_id}", Button).press()
        await pilot.pause()

    assert len(captured) == 1
    assert captured[0].which == which
    assert captured[0].item is item


def _seed_change_item_with_snapshots(db, *, snapshot_rows):
    """One subscription with one `change`-kind item, plus `url_snapshots` rows.

    Args:
        db: A real `SubscriptionsDB` (the app fixture's own, via
            `app.local_watchlists_service._db()`).
        snapshot_rows: A sequence of `(content_hash, extracted_content,
            created_at)` tuples to insert into `url_snapshots` for the same
            (subscription, url) the item carries. Inserted directly by SQL
            -- `_store_snapshot` (the production writer) lives in
            `monitoring_engine.py` and is not exercised here, the same
            choice `Tests/DB/test_subscriptions_db.py`'s own
            `get_url_snapshots` tests make.

    Returns:
        `(source_id, url)`.
    """
    from tldw_chatbook.Subscriptions.item_persist import persist_subscription_item

    url = "https://anthropic.com/news"
    source_id = db.add_subscription(
        name="Anthropic", type="url", source=url
    )
    with db.transaction() as conn:
        persist_subscription_item(
            conn,
            source_id,
            {
                "url": url,
                "title": "anthropic.com/news changed",
                "content": "+ Opus 4.5 available\n- Opus 4.1 available",
                "content_kind": "change",
                "content_format": "diff",
                "content_hash": "hash-change-item",
                "change_percentage": 12.0,
                "change_type": "content",
            },
            run_id=None,
            now="2026-07-28T09:00:00+00:00",
        )
    with db.transaction() as conn:
        for content_hash, extracted_content, created_at in snapshot_rows:
            conn.execute(
                """
                INSERT INTO url_snapshots
                    (subscription_id, url, content_hash, extracted_content, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (source_id, url, content_hash, extracted_content, created_at),
            )
    return source_id, url


async def _select_first_item_with_inspector(pilot, screen, pane):
    """Open the right rail and select the first loaded item."""
    from tldw_chatbook.UI.Watchlists_Modules.inspector_pane import InspectorPane

    if not screen.query("#watchlists-entity-inspector"):
        screen.action_toggle_right_rail()
        await pilot.pause(0.2)
    item = screen._loaded_items[0]
    pane.select_item_by_id(str(item["id"]))
    await pilot.pause(0.3)
    return screen.query_one("#watchlists-entity-inspector", InspectorPane)


@pytest.mark.asyncio
async def test_full_page_button_opens_the_newest_snapshot_in_a_modal():
    """The screen's handler must resolve `"full_page"` to the newest
    `url_snapshots` row and push `SnapshotViewModal` with it -- driven
    through a real button press, not a direct method call.
    """
    from textual.widgets import Button, Static

    from Tests.UI.test_destination_shells import DestinationHarness
    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.UI.Watchlists_Modules.inspector_pane import InspectorPane
    from tldw_chatbook.UI.Watchlists_Modules.snapshot_view_modal import SnapshotViewModal

    app = _build_test_app()
    db = app.local_watchlists_service._db()
    _seed_change_item_with_snapshots(
        db,
        snapshot_rows=[
            ("hash-older", "the page as it was before", "2026-07-27T09:00:00"),
            ("hash-newest", "the page as it is now", "2026-07-28T09:00:00"),
        ],
    )

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen, pane = await _mount_items_screen(pilot, host, expected_count=1)
        inspector = await _select_first_item_with_inspector(pilot, screen, pane)
        assert isinstance(inspector, InspectorPane)
        inspector.query_one("#inspector-full-page-button", Button).press()

        modal = None
        for _ in range(60):
            await pilot.pause(0.05)
            if isinstance(host.screen_stack[-1], SnapshotViewModal):
                modal = host.screen_stack[-1]
                break
        assert modal is not None, "the full-page button must push the snapshot modal"

        body = str(modal.query_one("#svm-body", Static).renderable)
        assert "the page as it is now" in body
        assert "the page as it was before" not in body, (
            "full page must be the NEWEST snapshot, not the previous one"
        )
        header = str(modal.query_one("#svm-header", Static).renderable)
        assert "https://anthropic.com/news" in header

        modal.query_one("#svm-close", Button).press()
        await pilot.pause(0.3)
        assert not isinstance(host.screen_stack[-1], SnapshotViewModal)


@pytest.mark.asyncio
async def test_previous_snapshot_button_opens_the_second_newest_snapshot():
    """`"previous"` must resolve to the SECOND-newest row, not the newest --
    the whole reason `_SNAPSHOTS_KEPT_PER_URL` keeps three rows instead of
    one.
    """
    from textual.widgets import Button, Static

    from Tests.UI.test_destination_shells import DestinationHarness
    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.UI.Watchlists_Modules.inspector_pane import InspectorPane
    from tldw_chatbook.UI.Watchlists_Modules.snapshot_view_modal import SnapshotViewModal

    app = _build_test_app()
    db = app.local_watchlists_service._db()
    _seed_change_item_with_snapshots(
        db,
        snapshot_rows=[
            ("hash-oldest", "oldest of three", "2026-07-26T09:00:00"),
            ("hash-previous", "the previous page", "2026-07-27T09:00:00"),
            ("hash-newest", "the newest page", "2026-07-28T09:00:00"),
        ],
    )

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen, pane = await _mount_items_screen(pilot, host, expected_count=1)
        inspector = await _select_first_item_with_inspector(pilot, screen, pane)
        assert isinstance(inspector, InspectorPane)
        inspector.query_one("#inspector-previous-snapshot-button", Button).press()

        modal = None
        for _ in range(60):
            await pilot.pause(0.05)
            if isinstance(host.screen_stack[-1], SnapshotViewModal):
                modal = host.screen_stack[-1]
                break
        assert modal is not None, "the previous-snapshot button must push the modal"

        body = str(modal.query_one("#svm-body", Static).renderable)
        assert "the previous page" in body
        assert "the newest page" not in body
        assert "oldest of three" not in body

        modal.query_one("#svm-close", Button).press()
        await pilot.pause(0.3)


@pytest.mark.asyncio
async def test_previous_snapshot_with_only_one_stored_degrades_to_an_honest_toast():
    """AC#2: no previous snapshot exists yet (only one check has ever run)
    must degrade to an honest toast, never an empty modal and never a
    silent no-op.
    """
    from unittest.mock import Mock

    from textual.widgets import Button

    from Tests.UI.test_destination_shells import DestinationHarness
    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.UI.Watchlists_Modules.inspector_pane import InspectorPane
    from tldw_chatbook.UI.Watchlists_Modules.snapshot_view_modal import SnapshotViewModal

    app = _build_test_app()
    db = app.local_watchlists_service._db()
    _seed_change_item_with_snapshots(
        db,
        snapshot_rows=[
            ("hash-only", "the only snapshot so far", "2026-07-28T09:00:00"),
        ],
    )
    app.notify = Mock()

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen, pane = await _mount_items_screen(pilot, host, expected_count=1)
        inspector = await _select_first_item_with_inspector(pilot, screen, pane)
        assert isinstance(inspector, InspectorPane)
        inspector.query_one("#inspector-previous-snapshot-button", Button).press()

        for _ in range(60):
            await pilot.pause(0.05)
            if app.notify.called:
                break

        assert app.notify.called, "the absence must be reported, not silent"
        args, kwargs = app.notify.call_args
        assert "no previous snapshot" in str(args[0]).lower()
        assert kwargs.get("severity") == "warning"
        assert kwargs.get("markup") is False, (
            "this item's own url/title are not this app's text to interpret "
            "as markup"
        )
        assert not isinstance(host.screen_stack[-1], SnapshotViewModal), (
            "AC#2: an absent snapshot must never open an empty modal"
        )


@pytest.mark.asyncio
async def test_snapshot_modal_renders_remote_markup_as_literal_text():
    """AC#3: `extracted_content` is scraped from a page this app does not
    control. A markup-shaped fragment in it must paint as literal
    characters -- never be interpreted as a style, and never become a live
    hyperlink -- the same property `test_markup_shaped_body_is_rendered_
    as_characters_not_interpreted` pins for the article renderer.
    """
    from textual.widgets import Button, Static

    from Tests.UI.test_destination_shells import DestinationHarness
    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.UI.Watchlists_Modules.inspector_pane import InspectorPane
    from tldw_chatbook.UI.Watchlists_Modules.snapshot_view_modal import SnapshotViewModal

    app = _build_test_app()
    db = app.local_watchlists_service._db()
    _seed_change_item_with_snapshots(
        db,
        snapshot_rows=[
            (
                "hash-hostile",
                "before [bold red]INJECTED[/] and [link=evil]click[/link] after",
                "2026-07-28T09:00:00",
            ),
        ],
    )

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen, pane = await _mount_items_screen(pilot, host, expected_count=1)
        inspector = await _select_first_item_with_inspector(pilot, screen, pane)
        assert isinstance(inspector, InspectorPane)
        inspector.query_one("#inspector-full-page-button", Button).press()

        modal = None
        for _ in range(60):
            await pilot.pause(0.05)
            if isinstance(host.screen_stack[-1], SnapshotViewModal):
                modal = host.screen_stack[-1]
                break
        assert modal is not None

        renderable = modal.query_one("#svm-body", Static).renderable
        plain, ansi = _render_to_console(renderable, width=160)

        assert "[bold red]INJECTED[/]" in plain, (
            "the tag text must reach the screen verbatim, characters intact"
        )
        assert "[link=evil]click[/link]" in plain
        assert "\x1b[31m" not in ansi, "the [bold red] tag must not have styled anything"
        assert "\x1b]8;;" not in ansi, "the [link=...] tag must not have become a hyperlink"

        modal.query_one("#svm-close", Button).press()
        await pilot.pause(0.3)


# --- TASK-2307: HTML feed bodies render as readable text -------------------


def test_an_html_body_renders_as_readable_prose_with_the_link_visible_as_text():
    """The UAT finding, at the unit level: a feed's `<p>`/`<a href>` body
    used to show up on screen exactly as written, one long unreadable line.
    `readable_body_text` converts it before `render_article` ever appends it,
    so the tags themselves must be gone and the link's destination must
    still be legible -- as ordinary visible text, not a hidden hyperlink.
    """
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import render_article

    out = str(render_article({
        "title": "Claude Opus 4.5 is now available",
        "source_name": "Anthropic News",
        "content": (
            "<p>Article URL: <a href=\"https://example.test/opus\">"
            "read more</a></p><p>It is <strong>fast</strong>.</p>"
        ),
        "content_kind": "article",
        "content_format": "text",
    }))

    assert "<p>" not in out and "</p>" not in out, "block tags must be gone"
    assert "<a href" not in out, "the raw anchor tag must be gone"
    assert "<strong>" not in out
    assert "read more" in out, "the link's label must survive"
    assert "https://example.test/opus" in out, (
        "the destination must be legible as text -- a terminal reader who "
        "cannot see the address cannot judge it"
    )
    assert "It is fast." in out


def test_html_derived_prose_is_still_inert_when_actually_rendered():
    """The escaping-terminal rule holds at the NEW final render step too
    (AC#1): the HTML converter must not turn a feed body into something a
    real Rich console would interpret, and a raw control byte the HTML
    parser passes through untouched must not reach the terminal either.
    """
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import render_article

    hostile = (
        "<p>[bold red]not a style[/] and "
        "<a href=\"javascript:alert(1)\">click</a></p>"
        "<p>\x1b]8;;http://evil.test\x07label\x1b]8;;\x07 tail</p>"
    )
    plain, ansi = _render_to_console(render_article({
        "title": "Hostile",
        "source_name": "Hostile Feed",
        "content": hostile,
        "content_kind": "article",
        "content_format": "text",
    }))

    assert "[bold red]not a style[/]" in plain, "bracket text must survive as characters"
    assert "\x1b[31m" not in ansi, "the [bold red] tag must not have styled anything"
    assert "\x1b]8;;" not in ansi, "no OSC-8 hyperlink may reach the terminal"
    assert "\x1b" not in plain, "the raw ESC byte must not have survived at all"
    assert "label" in plain and "tail" in plain, "the surrounding text must still render"


def test_a_non_html_body_is_left_alone_apart_from_control_bytes():
    """The other half of `readable_body_text`'s dispatch: a plain-text feed
    body (most of them) must not be run through the HTML converter at all,
    and must survive byte-for-byte apart from characters that cannot
    legally reach the terminal.
    """
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import render_article

    out = str(render_article({
        "title": "Plain text feed",
        "source_name": "Feed",
        "content": "1 < 2 and 2 < 3, plainly written, no markup at all.",
        "content_kind": "article",
        "content_format": "text",
    }))

    assert "1 < 2 and 2 < 3, plainly written, no markup at all." in out


@pytest.mark.asyncio
async def test_selecting_an_html_item_shows_readable_prose_in_the_mounted_reader():
    """End to end through the real screen wiring (not a direct renderer
    call): the same `<p>`/`<a href>` shape the UAT found, opened the way a
    user actually opens it, must not show a single literal tag on screen.
    """
    from textual.widgets import Static

    from Tests.UI.test_destination_shells import DestinationHarness
    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import ContentPane
    from tldw_chatbook.UI.Watchlists_Modules.article_list import ArticleListPane

    item = {
        "id": 9,
        "title": "Site update",
        "source_name": "Some Blog",
        "content": '<p>Article URL: <a href="https://example.test/post">here</a></p>',
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

        items_pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        items_pane.items = [item]
        await pilot.pause(0.2)
        items_pane.select_item_by_id("9")
        await pilot.pause(0.3)

        content_pane = screen.query_one("#watchlists-content-pane", ContentPane)
        body = content_pane.query_one("#content-body", Static)
        rendered = str(body.renderable)

        assert "<p>" not in rendered and "<a href=" not in rendered
        assert "here" in rendered
        assert "https://example.test/post" in rendered


# --- TASK-3072 plan task 7: the reader's Star button ---------------------------


@pytest.mark.asyncio
async def test_the_star_button_reflects_the_open_items_state():
    """The button is seeded from the open item's `is_flagged`: starred items
    read "★ Starred", unstarred read "☆ Star" -- the same vocabulary the
    rail's Starred root and the list row's glyph speak."""
    from textual.widgets import Button

    from tldw_chatbook.UI.Watchlists_Modules.content_pane import ContentPane

    class _PaneHost(ConsolidatedCSSApp):
        def compose(self):
            pane = ContentPane()
            pane.item = {
                "title": "x",
                "content": "y",
                "content_kind": "article",
                "is_flagged": True,
            }
            yield pane

    app = _PaneHost()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert str(app.query_one("#content-star-button", Button).label) == "★ Starred"


@pytest.mark.asyncio
async def test_the_star_button_posts_the_toggle_without_an_optimistic_flip():
    """Pressing Star posts `StarToggleRequested` with the full item (the same
    message the screen's `s` handler serves) and does NOT flip its own
    label: the write is async and can fail, so the flip lands on the
    screen's success path -- the label can never lie about a failed write."""
    from textual.widgets import Button

    from tldw_chatbook.UI.Watchlists_Modules.content_pane import (
        ContentPane,
        StarToggleRequested,
    )

    class _PaneHost(ConsolidatedCSSApp):
        def __init__(self) -> None:
            super().__init__()
            self.captured: list[dict] = []

        def compose(self):
            pane = ContentPane()
            pane.item = {"title": "x", "content": "y", "content_kind": "article"}
            yield pane

        def on_star_toggle_requested(self, message: StarToggleRequested) -> None:
            self.captured.append(message.item)

    app = _PaneHost()
    async with app.run_test() as pilot:
        await pilot.pause()
        button = app.query_one("#content-star-button", Button)
        assert str(button.label) == "☆ Star", (
            "an unstarred item must offer the star, not hide the state"
        )

        button.press()
        await pilot.pause()

        assert [item.get("title") for item in app.captured] == ["x"], (
            "pressing the button must post StarToggleRequested exactly once, "
            "carrying the open item"
        )
        assert str(button.label) == "☆ Star", (
            "no optimistic flip: the screen flips the label on write success"
        )


# --- TASK-3072 plan task 8: the reader's remaining action-row verbs ------------


def _action_row_app(item: dict):
    """Host Reader and Inspector while capturing their shared action messages."""
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import (
        ContentPane,
        OpenInBrowserRequested,
    )
    from tldw_chatbook.UI.Watchlists_Modules.inspector_pane import (
        IngestRequested,
        InspectorPane,
        ToggleBriefingQueueRequested,
    )

    class _PaneHost(ConsolidatedCSSApp):
        def __init__(self) -> None:
            super().__init__()
            self.opened: list[dict] = []
            self.ingested: list[dict] = []
            self.queue_toggles: list[tuple] = []

        def compose(self):
            pane = ContentPane()
            pane.item = item
            yield pane
            inspector = InspectorPane()
            inspector.selected_entity = {"entity_kind": "watchlist_item", **item}
            yield inspector

        def on_open_in_browser_requested(self, message: OpenInBrowserRequested) -> None:
            self.opened.append(message.item)

        def on_ingest_requested(self, message: IngestRequested) -> None:
            self.ingested.append(message.entity)

        def on_toggle_briefing_queue_requested(
            self, message: ToggleBriefingQueueRequested
        ) -> None:
            self.queue_toggles.append((message.item_id, message.queued))

    return _PaneHost()


@pytest.mark.asyncio
async def test_reader_offers_open_while_inspector_offers_ingest_and_queue():
    """Core browser action stays in Reader; advanced actions live in Inspector."""
    from textual.widgets import Button

    app = _action_row_app(
        {"title": "x", "content": "y", "content_kind": "article", "item_id": 7}
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        assert str(app.query_one("#content-open-button", Button).label) == "Open"
        assert str(app.query_one("#inspector-ingest-button", Button).label) == "Ingest"
        assert str(
            app.query_one("#inspector-queue-briefing-button", Button).label
        ) == "Queue for briefing"


@pytest.mark.asyncio
async def test_the_inspector_queue_button_label_reflects_queued_state():
    from textual.widgets import Button

    app = _action_row_app(
        {
            "title": "x",
            "content": "y",
            "content_kind": "article",
            "item_id": 7,
            "queued_for_briefing": True,
        }
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        assert str(
            app.query_one("#inspector-queue-briefing-button", Button).label
        ) == "Unqueue from briefing"


@pytest.mark.asyncio
async def test_reader_and_inspector_action_buttons_post_shared_messages():
    """The two surfaces keep using the screen's existing message handlers."""
    from textual.widgets import Button

    app = _action_row_app(
        {"title": "x", "content": "y", "content_kind": "article", "item_id": 7}
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#content-open-button", Button).press()
        app.query_one("#inspector-ingest-button", Button).press()
        app.query_one("#inspector-queue-briefing-button", Button).press()
        await pilot.pause()

        assert [item.get("title") for item in app.opened] == ["x"]
        assert [entity.get("title") for entity in app.ingested] == ["x"]
        assert app.queue_toggles == [(7, True)], (
            "Queue on an unqueued item must request the flip TO queued, "
            "carrying the raw item id exactly as the Inspector does"
        )


# --- TASK-3072 plan task 9: the reader's position footer -----------------------


@pytest.mark.asyncio
async def test_the_position_footer_renders_and_updates_in_place():
    """The footer shows the screen-pushed `position` string; updating the
    reactive re-renders the one Static in place -- never a reader recompose."""
    from textual.widgets import Static

    from tldw_chatbook.UI.Watchlists_Modules.content_pane import ContentPane

    class _PaneHost(ConsolidatedCSSApp):
        def compose(self):
            pane = ContentPane()
            pane.item = {"title": "x", "content": "y", "content_kind": "article"}
            pane.position = "2 of 9"
            yield pane

    app = _PaneHost()
    async with app.run_test() as pilot:
        await pilot.pause()
        position = app.query_one("#content-position", Static)
        assert str(position.renderable) == "2 of 9"

        app.query_one(ContentPane).position = "3 of 9"
        await pilot.pause()
        assert str(position.renderable) == "3 of 9", (
            "the same Static must update in place (a recompose would replace it)"
        )


@pytest.mark.asyncio
async def test_the_next_unread_footer_button_posts_the_panes_message():
    """The footer's Next Unread control posts the pane's existing
    `NextUnreadRequested` -- the same message `space` already raises, so one
    screen handler serves both."""
    from textual.widgets import Button

    from tldw_chatbook.UI.Watchlists_Modules.content_pane import ContentPane
    from tldw_chatbook.UI.Watchlists_Modules.items_pane import NextUnreadRequested

    class _PaneHost(ConsolidatedCSSApp):
        def __init__(self) -> None:
            super().__init__()
            self.captured: list[NextUnreadRequested] = []

        def compose(self):
            pane = ContentPane()
            pane.item = {"title": "x", "content": "y", "content_kind": "article"}
            yield pane

        def on_next_unread_requested(self, message: NextUnreadRequested) -> None:
            self.captured.append(message)

    app = _PaneHost()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#content-next-unread-button", Button).press()
        await pilot.pause()
        assert len(app.captured) == 1


@pytest.mark.asyncio
async def test_the_empty_reader_has_no_position_footer():
    """With nothing open the footer is absent entirely -- not "0 of 0"."""
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import ContentPane

    class _PaneHost(ConsolidatedCSSApp):
        def compose(self):
            yield ContentPane()

    app = _PaneHost()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert not app.query("#content-position")
        assert not app.query("#content-next-unread-button")
