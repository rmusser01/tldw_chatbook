"""Generated bundles omit source prose without changing the CSS token stream."""

from pathlib import Path

import pytest
from textual.css.parse import parse_selectors
from textual.css.tokenize import tokenize

from tldw_chatbook.css import build_css


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


@pytest.mark.parametrize("relative_path", build_css.CSS_MODULES)
def test_real_source_css_keeps_token_and_selector_whitespace(relative_path):
    source = (Path(build_css.__file__).parent / relative_path).read_text()
    result = build_css._without_source_comments(source)
    assert normalized_tokens(result) == normalized_tokens(source)
