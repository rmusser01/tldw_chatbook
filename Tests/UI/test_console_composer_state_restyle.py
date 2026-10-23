"""TASK-33620.15.8: the composer's state classes restyle only what they style.

A Send's acceptance clears the composer inside run start's post-commit step
(``ConsoleRuntime`` -> ``_project_console_received_input`` ->
``ConsoleComposerBar.load_draft("")``). On origin/dev 5b5920b8e2 that clear was
the longest part of run start's longest UI-loop callback: removing
``console-composer-has-draft`` made Textual restyle the bar's whole subtree
(26 nodes, about 10 of the step's 19 ms live), and the Send button's
readiness flip restyled the button once per state class. These tests pin the
smaller restyles, and that the styles they leave are the ones a full restyle
computes.
"""

from __future__ import annotations

import pytest
from textual.css.model import CombinatorType, SelectorType
from textual.widgets import Button, Static

from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar

pytestmark = pytest.mark.bootstrap_profile


def _spy_restyles(host, monkeypatch) -> list:
    """Record every node the app's stylesheet styles from now on."""
    applied: list = []
    original = host.stylesheet.apply

    def apply(node, *args, **kwargs):
        applied.append(node)
        return original(node, *args, **kwargs)

    monkeypatch.setattr(host.stylesheet, "apply", apply)
    return applied


def _composer_styles(composer: ConsoleComposerBar) -> dict[str, dict]:
    """Each composer node's stylesheet-computed rules, keyed by a stable path."""
    snapshot: dict[str, dict] = {}
    for index, node in enumerate(composer.walk_children(with_self=True)):
        key = f"{index}:{type(node).__name__}#{node.id}"
        snapshot[key] = dict(node._css_styles.get_rules())
    return snapshot


def _ready_host() -> ConsoleHarness:
    """A Console whose Send is ready, as it is when a real Send is accepted."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    return ConsoleHarness(app)


async def _mounted_composer(host, pilot) -> ConsoleComposerBar:
    console = host.screen_stack[-1]
    await _wait_for_selector(console, pilot, "#console-native-composer")
    return console.query_one("#console-native-composer", ConsoleComposerBar)


@pytest.mark.asyncio
async def test_clearing_the_draft_does_not_restyle_unrelated_composer_controls(
    monkeypatch,
):
    """The Send-acceptance clear restyles the bar and its draft, not the row.

    Red on origin/dev: the has-draft class change restyled all 26 composer
    nodes (dictation, menu, collapse and the rest), because Textual restyles
    a node's whole subtree whenever one of its classes changes.
    """
    host = _ready_host()
    async with host.run_test(size=(160, 45)) as pilot:
        composer = await _mounted_composer(host, pilot)
        composer.load_draft("Reply with the single word ok")
        await pilot.pause(0.1)
        assert composer.has_class("console-composer-has-draft")
        applied = _spy_restyles(host, monkeypatch)

        composer.load_draft("")

        assert not composer.has_class("console-composer-has-draft")
        assert composer.draft_text() == ""
        expected = {
            "console-native-composer",
            "console-command-visible-text",
            # Its readiness and the disabled-reason strip are styled by
            # their own classes and states, which the clear does change.
            "console-send-message",
            "console-send-disabled-reason",
        }
        restyled = {
            node.id or type(node).__name__
            for node in applied
            if node is composer or composer in node.ancestors
        }
        unrelated = sorted(restyled - expected)
        assert not unrelated, (
            "Clearing the draft restyled composer nodes no state class styles: "
            f"{unrelated}"
        )


@pytest.mark.asyncio
async def test_a_send_readiness_flip_restyles_the_send_button_once_for_its_classes(
    monkeypatch,
):
    """Six readiness classes change together and cost one restyle, not six.

    Red on origin/dev: each ``set_class`` restyled the button on its own, so a
    ready -> empty flip restyled it 8 times (5 class changes, plus Textual's
    own restyles for ``disabled`` and the two ``variant`` classes).
    """
    host = _ready_host()
    async with host.run_test(size=(160, 45)) as pilot:
        composer = await _mounted_composer(host, pilot)
        send = composer.query_one("#console-send-message", Button)
        composer.load_draft("Reply with the single word ok")
        composer.sync_action_state(
            has_draft=True, run_active=False, can_save_chatbook=False
        )
        assert send.has_class("console-send-ready")
        applied = _spy_restyles(host, monkeypatch)

        composer.sync_action_state(
            has_draft=False, run_active=False, can_save_chatbook=False
        )

        assert send.disabled
        assert send.classes >= {
            "console-action-subdued",
            "console-action-disabled",
            "console-send-inactive",
        }
        assert not send.classes & {"console-action-primary", "console-send-ready"}
        send_restyles = sum(1 for node in applied if node is send)
        # One for the six state classes, one for ``disabled``, two for the
        # ``variant`` watcher's remove/add.
        assert send_restyles <= 4, (
            f"The Send button was restyled {send_restyles} times for one "
            "readiness flip."
        )


@pytest.mark.asyncio
async def test_targeted_state_restyles_leave_the_styles_a_full_restyle_computes():
    """Guard: skipping the rest of the subtree changes no computed style.

    Passes on origin/dev by construction (it restyled everything); it fails
    once a rule keyed on a state class styles a node the targeted restyle
    skips.
    """
    host = _ready_host()
    async with host.run_test(size=(160, 45)) as pilot:
        composer = await _mounted_composer(host, pilot)
        visible = composer.query_one("#console-command-visible-text", Static)

        async def settle_and_compare(label: str) -> None:
            targeted = _composer_styles(composer)
            host.update_styles(composer)
            assert _composer_styles(composer) == targeted, label

        composer.load_draft("Reply with the single word ok")
        await settle_and_compare("draft added")
        composer.load_draft("")
        await settle_and_compare("draft cleared")
        visible.focus()
        await pilot.pause(0.1)
        assert composer.has_class("console-composer-focused")
        await settle_and_compare("focused")
        composer.load_draft("typed while focused")
        await settle_and_compare("draft added while focused")
        composer.blur()
        visible.blur()
        await pilot.pause(0.1)
        assert not composer.has_class("console-composer-focused")
        await settle_and_compare("blurred")


def _target_compound(selectors: list) -> list:
    """The trailing compound selector (the part a rule actually styles)."""
    start = len(selectors) - 1
    while start > 0 and selectors[start].combinator == CombinatorType.SAME:
        start -= 1
    return selectors[start:]


@pytest.mark.asyncio
async def test_rules_keyed_on_state_classes_style_only_the_declared_nodes():
    """Guard: every rule keyed on a state class targets a node it restyles.

    Covers every rule the app loads, including composer states the mounted
    guard above does not reach. Fails with the rule's selector when CSS
    starts styling another descendant from one of these classes.
    """
    host = _ready_host()
    async with host.run_test(size=(160, 45)) as pilot:
        await _mounted_composer(host, pilot)
        classes = set(ConsoleComposerBar.INTERACTION_CLASSES)
        allowed_ids = {"console-native-composer"} | {
            selector.lstrip("#")
            for selector in ConsoleComposerBar.INTERACTION_STYLED_DESCENDANTS
        }
        offenders: list[str] = []
        keyed = 0
        for rule in host.stylesheet.rules:
            for selector_set in rule.selector_set:
                selectors = selector_set.selectors
                if not any(
                    s.type == SelectorType.CLASS and s.name in classes
                    for s in selectors
                ):
                    continue
                keyed += 1
                target = _target_compound(selectors)
                ids = {s.name for s in target if s.type == SelectorType.ID}
                carries_class = any(
                    s.type == SelectorType.CLASS and s.name in classes for s in target
                )
                if not (ids & allowed_ids) and not carries_class:
                    offenders.append(
                        "".join(
                            {
                                CombinatorType.SAME: "",
                                CombinatorType.CHILD: " > ",
                            }.get(s.combinator, " ")
                            + s.css
                            for s in selectors
                        ).strip()
                    )
        assert keyed, "No rule is keyed on the composer's state classes."
        assert not offenders, (
            "These rules style a node the composer's targeted state restyle "
            "skips; add its selector to INTERACTION_STYLED_DESCENDANTS: "
            f"{offenders}"
        )
