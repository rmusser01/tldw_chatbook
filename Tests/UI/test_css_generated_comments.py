"""Generated bundles omit source prose without changing the CSS token stream."""

from pathlib import Path

import pytest
from textual.css.parse import parse_selectors
from textual.css.tokenize import tokenize

from tldw_chatbook.css import build_css, widget_css


def normalized_tokens(css):
    """Keep whitespace boundaries, coalescing only adjacent whitespace tokens."""
    result = []
    for token in tokenize(css, ("test", "")):
        if token.name in {"comment_line", "comment_start", "comment_end"}:
            continue
        item = (token.name, " " if token.name == "whitespace" else token.value)
        if item == ("whitespace", " ") and (not result or result[-1] == item):
            continue
        result.append(item)
    if result and result[-1] == ("whitespace", " "):
        result.pop()
    return result


def test_build_omits_comments_preserves_strings_tokens_and_source(
    tmp_path, monkeypatch
):
    source = """/* Defect rationale belongs in the authoring source. */
$x: 2;
Widget /* keep selector separation */ .child {
    width: $x; /* inline rationale */
    border-title: "/* literal, not a comment */";
}
"""
    path = tmp_path / "test.tcss"
    path.write_text(source)
    output = tmp_path / "bundle.tcss"
    monkeypatch.setattr(build_css, "CSS_MODULES", ["test.tcss"])
    build_css.build_css(tmp_path, output)
    result = output.read_text()
    assert "Defect rationale" not in result
    assert "/* ===== MODULE: test.tcss ===== */" in result
    assert '"/* literal, not a comment */"' in result
    assert path.read_text() == source
    assert normalized_tokens(result) == normalized_tokens(source)


@pytest.mark.parametrize(
    "selector",
    [
        "Button/* note */.active",
        "Button /* note */.active",
        "Button/* note */ .active",
        "Button /* note */ > .active",
    ],
)
def test_comment_removal_preserves_parsed_selector_structure(selector):
    source = selector + " { width: 1; }"
    result = build_css._without_source_comments(source)
    original = parse_selectors(selector)
    rewritten = parse_selectors(result.split("{", 1)[0])
    assert [str(group) for group in rewritten] == [str(group) for group in original]


def test_comment_removal_keeps_quotes_and_surrounding_whitespace():
    source = 'Button/*note*/.active { border-title: "x /* not prose */  "; }'
    result = build_css._without_source_comments(source)
    assert result == 'Button.active { border-title: "x /* not prose */  "; }'


def test_widget_streams_omit_source_prose_but_keep_provenance_and_tokens():
    source = """/* Widget rationale stays in Python. */
Sample/* self compound */.active { height: auto; }
Sample /* descendant boundary */ .child {
    border-title: "/* literal marker */  "; /* do not publish this prose */
}
.scoped { padding: 0 1; }
"""
    block = widget_css.BundledBlock("sample.py", "Sample", 1, source)
    expected = widget_css.split_scoped_css(source, "Sample", scope_every_selector=True)
    actual = widget_css.render_stylesheets([block], "test", scope_every_selector=True)
    for before, after in zip(expected, actual, strict=True):
        assert "Widget rationale" not in after
        assert "do not publish this prose" not in after
        assert "/* ===== WIDGET: Sample (sample.py) ===== */" in after
        assert normalized_tokens(before) == normalized_tokens(after)
    assert block.css == source


@pytest.mark.parametrize("relative_path", build_css.CSS_MODULES)
def test_real_source_css_keeps_token_and_selector_whitespace(relative_path):
    source = (Path(build_css.__file__).parent / relative_path).read_text()
    result = build_css._without_source_comments(source)
    assert normalized_tokens(result) == normalized_tokens(source)


@pytest.mark.parametrize(
    "block",
    widget_css.iter_blocks(Path(widget_css.__file__).parents[1], widget_css.WIDGET_ATTR)
    + widget_css.iter_blocks(
        Path(widget_css.__file__).parents[1], widget_css.SCREEN_ATTR
    ),
    ids=lambda block: f"{block.module}::{block.class_name}",
)
def test_real_bundled_block_comment_removal_preserves_tokens(block):
    source = widget_css.isolate_local_variables(block.css, scope=block.class_name)
    for stream in widget_css.split_scoped_css(
        source, block.class_name, scope_every_selector=True
    ):
        assert normalized_tokens(
            widget_css.without_source_comments(stream)
        ) == normalized_tokens(stream)
