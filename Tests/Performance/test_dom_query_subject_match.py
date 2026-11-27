"""Widget searches must answer exactly as upstream, without ancestor lists.

``textual.css.match.match`` builds ``node.css_path_nodes`` -- every ancestor,
root first -- for each node a search tests, then walks that list with the
selector. A selector with no ancestry (``#id``, ``.class``, ``Type``, a
compound of those) can only succeed on the node itself, so
``tldw_chatbook/Utils/textual_css_fastpath.py`` runs those checks on the node
alone and delegates every other shape to upstream.

Measured on the Console (558 widgets) in one approved-edit turn, 2026-11-26
(TASK-40608): 312 searches, 0.46 s of the UI thread, among them one ``#id``
search of the whole screen at 8 ms.

Three things have to stay true, and each has a test here:

1. **Same answers.** Every node of a live tree is matched both ways against
   selectors of every shape, and whole ``query`` / ``query_one`` results are
   compared.
2. **The cost is gone, and only where it is safe.** A search with no ancestry
   reads no ancestor list; one with ancestry still delegates.
3. **Upstream still has the shape this relies on.** A Textual upgrade that
   changes ``match`` or how ``DOMQuery`` calls it fails here.
"""

from __future__ import annotations

import inspect
from unittest.mock import patch

import pytest
from textual import dom
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css import match as upstream_module
from textual.css import query as css_query
from textual.css.match import _check_selectors as upstream_check
from textual.css.parse import parse_selectors
from textual.dom import DOMNode
from textual.widgets import Button, Input, Static

from tldw_chatbook.Utils import textual_css_fastpath as fast
from tldw_chatbook.Utils.textual_css_fastpath import install_stylesheet_fastpath

#: Subject-only shapes: the node alone decides.
SUBJECT_ONLY = [
    "*",
    "Button",
    "#target",
    ".hot",
    "Button.hot",
    "Button#target.hot",
    "Button:disabled",
    "Button.hot:enabled",
    "*:focus",
    "Static.missing",
    "Button, Static",
    "#target, .cold",
]
#: Shapes with ancestry: upstream decides.
WITH_ANCESTRY = [
    "Vertical Button",
    "Vertical > Button",
    "#outer .hot",
    "#outer > Vertical Button.hot",
    "Horizontal Static",
    ".hot Button",
    "Vertical.hot > *",
    "#outer Button:disabled",
]
#: Lists mixing both kinds.
MIXED = ["Vertical > Button, #target", ".cold, Horizontal Static", "Input, #outer .hot"]


def upstream_match(selector_sets, node) -> bool:
    """Upstream's ``match``, spelled out so the patched name is never used."""
    return any(
        upstream_check(selector_set.selectors, node.css_path_nodes)
        for selector_set in selector_sets
    )


class _Tree(App):
    def compose(self) -> ComposeResult:
        with Vertical(id="outer", classes="hot"):
            with Vertical(classes="cold"):
                yield Button("a", id="target", classes="hot")
                yield Button("b", classes="hot", disabled=True)
                yield Static("c", classes="hot cold")
            with Horizontal(classes="hot"):
                yield Static("d")
                yield Button("e")
                yield Input(id="field")
            with Vertical():
                with Horizontal():
                    yield Button("f", classes="hot cold")
        yield Static("g", id="outside", classes="hot")


def _reads_of_ancestor_lists():
    """Count reads of ``css_path_nodes`` while the context is open."""
    original = DOMNode.css_path_nodes
    reads = []

    def counted(node):
        reads.append(node)
        return original.fget(node)

    return patch.object(DOMNode, "css_path_nodes", property(counted)), reads


def test_upstream_match_still_has_the_shape_the_fast_path_assumes() -> None:
    """A Textual upgrade that changes the matcher must fail here, not drift."""
    source = inspect.getsource(upstream_module.match)
    assert "_check_selectors(selector_set.selectors, node.css_path_nodes)" in source
    assert "for selector_set in selector_sets" in source
    nodes = inspect.getsource(css_query.DOMQuery.nodes.fget)
    assert "match(selector_set, node) for selector_set in self._filters" in nodes
    assert "match(selector_set, node) for selector_set in self._excludes" in nodes
    assert "if not match(selector_set, node):" in inspect.getsource(dom.DOMNode.query_one)


def test_install_binds_the_searches_and_is_idempotent() -> None:
    install_stylesheet_fastpath()
    assert css_query.match is fast._match_on_subject
    assert dom.match is fast._match_on_subject
    assert install_stylesheet_fastpath() is False
    assert css_query.match is fast._match_on_subject
    # Upstream's own name stays upstream: only the two search modules change.
    assert upstream_module.match.__module__ == "textual.css.match"


@pytest.mark.asyncio
async def test_every_node_matches_exactly_as_upstream() -> None:
    install_stylesheet_fastpath()
    app = _Tree()
    async with app.run_test() as pilot:
        app.screen.query_one("#target", Button).focus()
        await pilot.pause()
        nodes = [app.screen, *app.screen.walk_children()]
        assert len(nodes) >= 13
        outcomes = set()
        for selector in SUBJECT_ONLY + WITH_ANCESTRY + MIXED:
            selector_sets = parse_selectors(selector)
            for node in nodes:
                expected = upstream_match(selector_sets, node)
                assert fast._match_on_subject(selector_sets, node) is expected, (
                    selector,
                    node,
                )
                outcomes.add((selector, expected))
        # Every selector but the deliberately absent one both matched and missed.
        for selector in SUBJECT_ONLY + WITH_ANCESTRY + MIXED:
            if selector not in ("*", "Static.missing"):
                assert (selector, True) in outcomes, selector
                assert (selector, False) in outcomes, selector


@pytest.mark.asyncio
async def test_whole_searches_return_the_same_widgets_in_the_same_order() -> None:
    install_stylesheet_fastpath()
    app = _Tree()
    async with app.run_test():
        screen = app.screen
        for selector in SUBJECT_ONLY + WITH_ANCESTRY + MIXED:
            found = list(screen.query(selector))
            excluded = list(screen.query("*").exclude(selector))
            with patch.object(css_query, "match", upstream_match):
                assert list(screen.query(selector)) == found, selector
                assert list(screen.query("*").exclude(selector)) == excluded, selector
        for selector in ("Button.hot", "Vertical > Button", "Horizontal Static"):
            first = screen.query_one(selector)
            with patch.object(dom, "match", upstream_match):
                # A different expected type is a different cache entry.
                assert screen.query_one(selector, type(first)) is first


@pytest.mark.asyncio
async def test_a_search_with_no_ancestry_reads_no_ancestor_list() -> None:
    """The cost this exists to remove; upstream reads one list per tested node."""
    install_stylesheet_fastpath()
    app = _Tree()
    async with app.run_test():
        screen = app.screen
        widgets = len(screen.walk_children())
        for selector in SUBJECT_ONLY:
            counting, reads = _reads_of_ancestor_lists()
            with counting:
                list(screen.query(selector))
            assert reads == [], selector
        counting, reads = _reads_of_ancestor_lists()
        with counting, patch.object(css_query, "match", upstream_match):
            list(screen.query("#target"))
        assert len(reads) == widgets  # the control: upstream, one per widget


@pytest.mark.asyncio
async def test_a_search_with_ancestry_delegates_to_upstream() -> None:
    install_stylesheet_fastpath()
    app = _Tree()
    async with app.run_test():
        screen = app.screen
        widgets = len(screen.walk_children())
        for selector in WITH_ANCESTRY:
            with patch.object(
                fast, "_upstream_check_selectors", wraps=upstream_check
            ) as original:
                list(screen.query(selector))
            assert original.call_count == widgets, selector


def test_an_unsupported_compound_shape_stays_upstream() -> None:
    """A compound whose parts do not all stay on one node is not subject-only."""
    parent = Vertical(classes="hot")
    button = Button(classes="hot")
    button._attach(parent)
    selector_sets = parse_selectors("Button.hot")
    selectors = selector_sets[0].selectors
    selectors[0].advance = 1  # Not the parsed shape: upstream must decide.
    try:
        with patch.object(
            fast, "_upstream_check_selectors", wraps=upstream_check
        ) as original:
            assert fast._match_on_subject(selector_sets, button) == upstream_match(
                selector_sets, button
            )
        original.assert_called_once()
    finally:
        selectors[0].advance = 0  # ``parse_selectors`` caches what it returned.
