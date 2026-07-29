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


def test_content_pane_shows_placeholder_with_no_item():
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import ContentPane

    pane = ContentPane()
    assert pane.item is None
