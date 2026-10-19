"""Reject impossible selectors without caching their dynamic matching result."""

from textual.containers import Horizontal, Vertical
from textual.css.stylesheet import Stylesheet
from textual.widgets import Button

from tldw_chatbook.Utils.textual_css_fastpath import _ordered_candidates


def _rules(css, node):
    sheet = Stylesheet()
    sheet.add_source(css)
    sheet.parse()
    return sheet, _ordered_candidates(sheet, node)


def test_absent_type_and_unrelated_alternative_do_not_reach_matcher():
    parent = Vertical()
    button = Button()
    button._attach(parent)
    _, rules = _rules(
        "MissingScreen Button { color: red; } "
        "#unrelated, MissingScreen Button { color: blue; } "
        "Vertical Button { color: green; }",
        button,
    )
    assert [rule.selectors for rule in rules] == ["Vertical Button"]


def test_every_static_requirement_is_needed_but_compounds_stay_on_subject():
    parent = Vertical(classes="outer")
    button = Button(id="target", classes="hot")
    button._attach(parent)
    sheet, rules = _rules(
        ".outer .missing Button { color: red; } "
        "Vertical.missing Button { color: red; } "
        "Static.hot { color: red; } "
        "#target.hot { color: blue; } "
        "Vertical Button.hot { color: green; }",
        button,
    )
    assert [rule.selectors for rule in rules] == ["#target.hot", "Vertical Button.hot"]
    parent.add_class("missing", update=False)
    assert len(_ordered_candidates(sheet, button)) == 4
    parent.remove_class("missing", update=False)
    assert _ordered_candidates(sheet, button) == rules
    button.remove_class("hot", update=False)
    assert _ordered_candidates(sheet, button) == []


def test_inherited_types_and_reparenting_are_observed():
    class SpecialVertical(Vertical):
        pass

    class SpecialButton(Button):
        pass

    parent = SpecialVertical()
    button = SpecialButton(classes="hot")
    button._attach(parent)
    sheet, rules = _rules("Vertical Button.hot { color: red; }", button)
    assert len(rules) == 1
    assert list(Stylesheet._check_rule(rules[0], button.css_path_nodes))
    button._attach(Horizontal())
    assert _ordered_candidates(sheet, button) == []
    button._attach(parent)
    assert _ordered_candidates(sheet, button) == rules


def test_disabled_css_inheritance_does_not_add_python_base_types():
    class IsolatedVertical(Vertical, inherit_css=False):
        pass

    button = Button(classes="hot")
    button._attach(IsolatedVertical())
    sheet, rules = _rules("Vertical Button.hot { color: red; }", button)
    assert not list(Stylesheet._check_rule(sheet.rules[0], button.css_path_nodes))
    assert rules == []


async def test_static_rejection_preserves_dynamic_matching_and_components():
    from textual.app import App, ComposeResult
    from textual.widgets import Input

    from tldw_chatbook.Utils.textual_css_fastpath import install_stylesheet_fastpath

    class Demo(App):
        CSS = """
        Vertical.outer Horizontal.inner Button.hot { color: red; }
        Vertical.inner Horizontal.outer Button.hot { color: blue; }
        #absent, MissingScreen Button { background: red; }
        Button:disabled, Horizontal.inner Button:focus { color: yellow; }
        Horizontal > Button:first-child { background: green; }
        Vertical.outer Input > .input--cursor { color: red; }
        Vertical.inner Input > .input--cursor { color: blue; }
        """

        def compose(self) -> ComposeResult:
            with Vertical(classes="outer"):
                with Horizontal(classes="inner"):
                    yield Button("One", classes="hot")
                    yield Button("Two", classes="hot")
                    yield Input("Text")

    install_stylesheet_fastpath()
    async with Demo().run_test() as pilot:
        sheet = pilot.app.stylesheet
        patched = Stylesheet.apply
        upstream = patched.__wrapped__
        vertical = pilot.app.query_one(Vertical)
        horizontal = pilot.app.query_one(Horizontal)
        button = pilot.app.query_one(Button)

        def snapshot():
            return [
                (
                    dict(node.styles.base.get_rules()),
                    {
                        key: value.get_rules()
                        for key, value in node._component_styles.items()
                    },
                )
                for node in pilot.app.screen.walk_children(with_self=True)
            ]

        def check():
            try:
                Stylesheet.apply = upstream
                sheet.update(pilot.app.screen)
                expected = snapshot()
            finally:
                Stylesheet.apply = patched
            sheet.update(pilot.app.screen)
            assert snapshot() == expected

        check()
        # Same union of ancestor names, different required order: cached candidates
        # must still delegate the full match, including child and same-node tests.
        vertical.set_classes("inner")
        horizontal.set_classes("outer")
        button.disabled = True
        await pilot.pause()
        check()
        vertical.set_classes("outer")
        horizontal.set_classes("inner")
        button.disabled = False
        await button.remove()
        await horizontal.mount(button)
        pilot.app.theme = "textual-light"
        await pilot.pause()
        check()
        sheet.add_source("Button.hot { color: purple; }", read_from=("dynamic", ""))
        check()
        sheet.add_source("Button.hot { color: orange; }", read_from=("dynamic", ""))
        check()
