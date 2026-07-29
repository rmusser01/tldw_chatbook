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
