"""Bound candidate selection without caching resolved styles or DOM nodes."""

from types import SimpleNamespace

from textual.css.model import RuleSet
from textual.css.stylesheet import Stylesheet

from tldw_chatbook.Utils import textual_css_fastpath as fastpath


def _sheet():
    sheet = Stylesheet()
    sheet.add_source(
        "Button { color: red; } .active Button { color: blue; } "
        "#other Button { color: green; } Button:disabled { opacity: 50%; }"
    )
    sheet.parse()
    return sheet


def _node(*ancestors, names=None):
    return SimpleNamespace(
        _selector_names=names or {"Button", "*"}, ancestors=list(ancestors)
    )


def test_repeated_candidate_selection_does_not_rehash_rules(monkeypatch):
    sheet = _sheet()
    node = _node()
    hashes = []
    original = RuleSet.__hash__

    def counted(rule):
        hashes.append(rule)
        return original(rule)

    monkeypatch.setattr(RuleSet, "__hash__", counted)
    first = fastpath._ordered_candidates(sheet, node)
    assert hashes  # Counter observes real selection work.
    hashes.clear()
    assert fastpath._ordered_candidates(sheet, _node()) == first
    assert hashes == []


def test_candidates_follow_ancestry_subject_and_stylesheet_changes():
    sheet = _sheet()
    parent = SimpleNamespace(id=None, classes=set())
    node = _node(parent)
    plain = fastpath._ordered_candidates(sheet, node)
    parent.classes.add("active")
    active = fastpath._ordered_candidates(sheet, node)
    assert len(active) == len(plain) + 1
    parent.classes.clear()
    assert fastpath._ordered_candidates(sheet, node) == plain
    parent.id = "other"
    assert len(fastpath._ordered_candidates(sheet, node)) == len(plain) + 1
    node.ancestors.clear()  # Reparenting is observed without an invalidation hook.
    assert fastpath._ordered_candidates(sheet, node) == plain
    node._selector_names = {"Static", "*"}
    assert fastpath._ordered_candidates(sheet, node) == []
    node._selector_names = {"Button", "*"}
    sheet.add_source("Button { background: yellow; }", read_from=("added", ""))
    assert len(fastpath._ordered_candidates(sheet, node)) == len(plain) + 1
    # Replacing a source also invalidates the candidate cache, not just appending.
    sheet.add_source("Static { background: blue; }", read_from=("added", ""))
    assert len(fastpath._ordered_candidates(sheet, node)) == len(plain)


def test_candidate_cache_is_bounded_and_does_not_retain_nodes():
    import gc
    import weakref

    class Node:
        ancestors = ()

    sheet = _sheet()
    node = Node()
    node._selector_names = {"Button", "*"}
    ref = weakref.ref(node)
    fastpath._ordered_candidates(sheet, node)
    del node
    gc.collect()
    assert ref() is None
    for index in range(2048):
        fastpath._ordered_candidates(sheet, _node(names={"Button", f"#id-{index}"}))
    cache = getattr(sheet, fastpath._CACHE_ATTR)[3]
    assert len(cache) <= 1024


async def test_cached_candidates_still_match_live_pseudo_classes():
    from textual.app import App, ComposeResult
    from textual.widgets import Button, Input

    class Demo(App):
        CSS = (
            "Button { color: $primary; } Button:disabled { color: blue; } "
            "Button:first-child { background: red; } "
            "Input > .input--cursor { color: red; } "
            ".accent Input > .input--cursor { color: blue; }"
        )

        def compose(self) -> ComposeResult:
            yield Button("Example")
            yield Button("Sibling")
            yield Input("Text")

    fastpath.install_stylesheet_fastpath()
    async with Demo().run_test() as pilot:
        button = pilot.app.query_one(Button)
        sheet = pilot.app.stylesheet
        patched = Stylesheet.apply
        upstream = patched.__wrapped__

        def snapshot(node):
            return (
                dict(node.styles.base.get_rules()),
                {
                    key: value.get_rules()
                    for key, value in node._component_styles.items()
                },
            )

        def check(node):
            # Recursive virtual component applications must also use upstream.
            try:
                Stylesheet.apply = upstream
                upstream(sheet, node)
                expected = snapshot(node)
            finally:
                Stylesheet.apply = patched
            patched(sheet, node)
            assert snapshot(node) == expected

        for disabled in (False, True, False):
            button.disabled = disabled
            await pilot.pause()
            check(button)
            check(pilot.app.query_one(Input))
        pilot.app.theme = "textual-light"
        pilot.app.screen.add_class("accent")
        await pilot.pause()
        await button.remove()
        await pilot.app.screen.mount(button)  # Now last, with the same selector names.
        await pilot.pause()
        check(button)
        check(pilot.app.query_one(Input))
