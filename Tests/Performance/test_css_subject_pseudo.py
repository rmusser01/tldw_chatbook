"""Live subject rejection must preserve upstream matching and invalidation."""

import inspect
from unittest.mock import patch

import pytest

from textual.containers import Vertical
from textual.css import stylesheet
from textual.css.match import _check_selectors as upstream_match
from textual.css.model import CombinatorType, Selector
from textual.css.stylesheet import Stylesheet
from textual.widgets import Button

from tldw_chatbook.Utils.textual_css_fastpath import install_stylesheet_fastpath


@pytest.mark.parametrize("rule", ["Button", "Button.hot", "Button:enabled.hot"])
def test_subject_only_rule_checks_each_selector_on_subject_once(rule):
    """A compound with no ancestry cannot gain a match from scanning parents."""
    install_stylesheet_fastpath()
    parent = Vertical()
    button = Button(classes="hot")
    button._attach(parent)
    sheet = Stylesheet()
    sheet.add_source(f"{rule} {{ color: red; }}")
    selectors = sheet.rules[0].selector_set[0].selectors
    checked = []
    original = Selector.check

    def check(selector, node):
        checked.append(node)
        return original(selector, node)

    assert upstream_match(selectors, button.css_path_nodes)
    with patch.object(Selector, "check", check):
        assert stylesheet._check_selectors(selectors, button.css_path_nodes)
    assert checked == [button] * len(selectors)


def test_uncertain_advance_and_ancestry_keep_original_matcher():
    from tldw_chatbook.Utils import textual_css_fastpath as fast

    parent = Vertical(classes="hot")
    button = Button(classes="hot")
    button._attach(parent)
    sheet = Stylesheet()
    sheet.add_source("Button.hot { color: red; } Vertical > Button { color: blue; }")
    compound, ancestry = [rule.selector_set[0].selectors for rule in sheet.rules]
    compound[0].advance = 1  # Unsupported/custom shape must stay upstream.
    for selectors in (compound, ancestry):
        with patch.object(
            fast, "_upstream_check_selectors", wraps=upstream_match
        ) as original:
            assert fast._check_subject_pseudo_classes(
                selectors, button.css_path_nodes
            ) == upstream_match(selectors, button.css_path_nodes)
        original.assert_called_once()


@pytest.mark.parametrize("start", [CombinatorType.CHILD, CombinatorType.SAME])
@pytest.mark.parametrize("rule", ["Button", "Button.hot"])
def test_non_descendant_start_keeps_original_matcher(start, rule):
    from tldw_chatbook.Utils import textual_css_fastpath as fast

    parent = Vertical()
    button = Button(classes="hot")
    button._attach(parent)
    sheet = Stylesheet()
    sheet.add_source(f"{rule} {{ color: red; }}")
    selectors = sheet.rules[0].selector_set[0].selectors
    selectors[0].combinator = start
    assert not upstream_match(selectors, button.css_path_nodes)
    assert not fast._check_subject_pseudo_classes(selectors, button.css_path_nodes)


def test_upstream_success_requires_final_selector_on_subject():
    """A Textual upgrade must not silently invalidate the necessary condition."""
    source = " ".join(inspect.getsource(upstream_match).split())
    assert "node = css_path_nodes[-1]" in source
    success = (
        "if selector.check(path_node): "
        "if path_node is node and selector_index == selector_count - 1: return True"
    )
    assert source.count(success) == 2
    assert source.count("return True") == 2
    assert source.count("selector_index + 1, node_index + selector.advance") == 2


def test_false_subject_pseudo_does_not_scan_ancestors():
    install_stylesheet_fastpath()
    parent = Vertical()
    button = Button()
    button._attach(parent)
    sheet = Stylesheet()
    sheet.add_source("Vertical *:disabled:can-focus { color: red; }")
    selectors = sheet.rules[0].selector_set[0].selectors
    checked = []
    original = Selector.check

    def check(selector, node):
        checked.append(node)
        return original(selector, node)

    with patch.object(Selector, "check", check):
        assert not stylesheet._check_selectors(selectors, button.css_path_nodes)
    assert checked == [button]
    button.disabled = True
    assert stylesheet._check_selectors(selectors, button.css_path_nodes)
    assert upstream_match(selectors, button.css_path_nodes)


async def test_live_subject_states_keep_styles_components_and_refresh_flags():
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal
    from textual.widgets import Input

    class Demo(App):
        CSS = """
        *:focus { color: red; }
        Vertical *:disabled:can-focus { color: yellow; }
        Vertical:focus-within > Horizontal Button:hover { background: blue; }
        Button.hot:disabled, Button:focus { border: solid green; }
        Button:disabled.hot { color: magenta; }
        Button:enabled.hot.-style-default { background: yellow; }
        Horizontal > Button:first-child { background: green; }
        Input > .input--cursor { color: red; }
        """

        def compose(self) -> ComposeResult:
            with Vertical():
                with Horizontal():
                    yield Button("One", classes="hot")
                    yield Button("Two")
                    yield Input("Text")

    install_stylesheet_fastpath()
    optimized = stylesheet._check_selectors
    assert optimized is not upstream_match
    async with Demo().run_test() as pilot:
        sheet = pilot.app.stylesheet
        one, two = list(pilot.app.query(Button))

        def snapshot():
            return [
                (
                    dict(node.styles.base.get_rules()),
                    {
                        key: value.get_rules()
                        for key, value in node._component_styles.items()
                    },
                    node._has_hover_style,
                    node._has_focus_within,
                    node._has_order_style,
                    node._has_odd_or_even,
                )
                for node in pilot.app.screen.walk_children(with_self=True)
            ]

        def compare():
            with patch.object(stylesheet, "_check_selectors", upstream_match):
                sheet.update(pilot.app.screen)
                expected = snapshot()
            sheet.update(pilot.app.screen)
            assert snapshot() == expected
            for node in pilot.app.screen.walk_children(with_self=True):
                for rule in sheet.rules:
                    for alternative in rule.selector_set:
                        selectors = alternative.selectors
                        assert optimized(
                            selectors, node.css_path_nodes
                        ) == upstream_match(selectors, node.css_path_nodes)

        compare()
        one.disabled = True
        two.focus()
        await pilot.hover(two)
        await pilot.pause()
        compare()
        one.disabled = False
        one.add_class("-textual-system")
        one.focus()
        await one.remove()
        await pilot.app.query_one(Horizontal).mount(one)
        pilot.app.theme = "textual-light"
        await pilot.pause()
        compare()
