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

import asyncio
from pathlib import Path

import pytest
from textual.css.model import CombinatorType, SelectorType
from textual.widgets import Button

import tldw_chatbook
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar

pytestmark = pytest.mark.bootstrap_profile

#: The built sheets (the app bundle and each lazy screen sheet).
_CSS_DIR = Path(tldw_chatbook.__file__).resolve().parent / "css"


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


async def _wait_until(pilot, condition, what: str, timeout: float = 10.0) -> None:
    """Pump the app until ``condition()`` holds; fail after ``timeout`` seconds."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        if loop.time() >= deadline:
            pytest.fail(f"Timed out after {timeout:.0f} s waiting for {what}")
        await pilot.pause(0.02)


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
        console = host.screen_stack[-1]
        focused = "console-composer-focused"

        async def settle_and_compare(label: str) -> None:
            targeted = _composer_styles(composer)
            host.update_styles(composer)
            assert _composer_styles(composer) == targeted, label

        def blurred() -> bool:
            # The Console's activation focus restore (after refresh, and
            # again on a 0.2 s timer) can hand focus back to the bar after a
            # blur, so blur again until the bar stays unfocused.
            if console.focused is composer:
                composer.blur()
            return not composer.has_class(focused)

        # Each wait ends with its state in place; the loads and comparisons
        # after it never yield to the loop, so no restore lands between them.
        await _wait_until(pilot, blurred, "the composer to lose focus")
        composer.load_draft("Reply with the single word ok")
        await settle_and_compare("draft added")
        composer.load_draft("")
        await settle_and_compare("draft cleared")
        # The bar is the focus target: its visible draft has can_focus off.
        composer.focus()
        await _wait_until(pilot, lambda: composer.has_class(focused), "focus")
        await settle_and_compare("focused")
        composer.load_draft("typed while focused")
        await settle_and_compare("draft added while focused")
        await _wait_until(pilot, blurred, "the composer to lose focus")
        await settle_and_compare("blurred with a draft")


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
    starts styling another descendant from one of these classes, and with
    the sheet's name when a built sheet that mentions them is not loaded
    here (a lazy screen sheet this harness stopped loading).
    """
    host = _ready_host()
    async with host.run_test(size=(160, 45)) as pilot:
        await _mounted_composer(host, pilot)
        classes = set(ConsoleComposerBar.INTERACTION_CLASSES)
        loaded = {Path(path).name for path, _ in host.stylesheet.source}
        unchecked = sorted(
            sheet.name
            for sheet in _CSS_DIR.glob("*.tcss")
            if sheet.name not in loaded
            and any(name in sheet.read_text(encoding="utf-8") for name in classes)
        )
        assert not unchecked, (
            "These built sheets mention the composer's state classes but this "
            f"guard does not load them: {unchecked}"
        )
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


@pytest.mark.asyncio
async def test_the_acceptance_projection_clears_now_and_restyles_send_after(
    monkeypatch,
):
    """The accepted clear lands in the step; Send's readiness restyles after it.

    Red on origin/dev: projecting the accepted (empty) draft restyled the Send
    button inside the projection call, which runs in run start's post-commit
    step, the step's longest piece once the bar's own restyle was targeted.
    """
    host = _ready_host()
    async with host.run_test(size=(160, 45)) as pilot:
        composer = await _mounted_composer(host, pilot)
        console = host.screen_stack[-1]
        send = composer.query_one("#console-send-message", Button)
        composer.load_draft("Reply with the single word ok")
        await pilot.pause(0.1)
        assert send.has_class("console-send-ready") and not send.disabled
        session_id = console._console_visible_draft_session_id
        assert session_id is not None
        store = console._console_chat_store
        # The screen's own action-row sync (its config-sync cadence) can land
        # in the pause below and flip Send by itself, which would hide a
        # projection that never synced the row at all; keep it out, so Send
        # turns inactive only through the composer's own deferred sync.
        monkeypatch.setattr(
            console, "_sync_console_composer_action_state", lambda **_: None
        )
        own_syncs: list[str] = []
        sync_current = composer._sync_current_action_state

        def record_own_sync() -> None:
            own_syncs.append(composer.draft_text())
            sync_current()

        monkeypatch.setattr(composer, "_sync_current_action_state", record_own_sync)
        applied = _spy_restyles(host, monkeypatch)

        # Run the durable acceptance's real input CAS, as run start does just
        # before it projects. TASK-37654: faking only the read left the store
        # holding the old draft, so a Console sync tick's write-back of the
        # cleared draft bumped the store revision past the projected one and
        # the next tick reloaded the composer: a second sync, by timing.
        assert store.commit_session_input_draft(
            store.session_input_snapshot(session_id)
        )
        console._project_console_received_input(session_id)

        assert composer.draft_text() == ""
        assert not composer.has_class("console-composer-has-draft")
        assert not any(
            node is send for node in applied
        ), "Send was restyled inside the acceptance projection"
        assert own_syncs == [], "The action row synced inside the projection"
        # Two Console sync ticks' draft reconcile (write-back, then revision
        # check) can land before the deferred sync; neither may reload the row.
        console._session._sync_console_session_draft()
        console._session._sync_console_session_draft()
        assert own_syncs == [], f"A Console sync tick resynced the row: {own_syncs!r}"
        # call_later captured record_own_sync during the projection. Restore
        # the method before yielding so a background session-draft refresh
        # cannot be counted as another execution of that deferred callback.
        monkeypatch.setattr(composer, "_sync_current_action_state", sync_current)
        await pilot.pause()
        assert own_syncs == [""], (
            "The projection should sync the action row exactly once, after "
            f"its step, from the cleared draft: {own_syncs!r}"
        )
        assert send.disabled
        assert send.has_class("console-send-inactive")
        assert not send.has_class("console-send-ready")


@pytest.mark.asyncio
async def test_a_focus_change_restyles_the_bar_and_its_draft_text_only(monkeypatch):
    """TASK-36511.2: a focus change restyles the bar and its draft text, not its subtree.

    Textual's ``Widget.watch_has_focus`` restyles the focused widget's whole
    subtree. On origin/dev bb80905eb3 each focus move onto or off the bar
    restyled its 27 nodes: 8-13 ms of CPU in the compact gates' attach, up
    to 98 ms of wall at load 17-34, and the 135 ms gate failure TASK-36511
    recorded.
    """
    host = _ready_host()
    async with host.run_test(size=(160, 45)) as pilot:
        composer = await _mounted_composer(host, pilot)
        screen = host.screen_stack[-1]
        await pilot.pause()
        outside = next(
            widget
            for widget in screen.focus_chain
            if widget is not composer and composer not in widget.ancestors
        )
        draft_text = composer.query_one("#console-command-visible-text")
        others = {
            id(node) for node in composer.walk_children() if node is not draft_text
        }
        assert len(others) > 10

        def moved(focused: bool):
            # The Console's activation focus restore can hand focus back to
            # the bar after it leaves, so move it again until it stays.
            def check() -> bool:
                if composer.has_focus is not focused:
                    screen.set_focus(composer if focused else outside)
                return composer.has_focus is focused

            return check

        await _wait_until(pilot, moved(True), "the bar to take focus")
        applied = _spy_restyles(host, monkeypatch)
        for focused in (False, True, False):
            applied.clear()
            await _wait_until(pilot, moved(focused), f"focus={focused}")
            restyled = [
                f"{type(n).__name__}#{n.id}" for n in applied if id(n) in others
            ]
            assert not restyled, (
                f"a focus change restyled {len(restyled)} composer descendants "
                f"no focus rule styles: {sorted(set(restyled))}"
            )
            assert any(node is composer for node in applied)
            targeted = _composer_styles(composer)
            host.update_styles(composer)
            assert _composer_styles(composer) == targeted, f"focused={focused}"


def _compounds(selectors: list) -> list[list]:
    """Split a selector into its compound selectors, left to right."""
    compounds: list[list] = [[]]
    for part in selectors:
        if compounds[-1] and part.combinator != CombinatorType.SAME:
            compounds.append([])
        compounds[-1].append(part)
    return compounds


def _could_match(parts: list, node, classes: set[str]) -> bool:
    """Whether a compound's type, id and class parts can all match ``node``."""
    for part in parts:
        if part.type == SelectorType.TYPE and part.name not in node._css_type_names:
            return False
        if part.type == SelectorType.ID and part.name != node.id:
            return False
        if part.type == SelectorType.CLASS and part.name not in classes:
            return False
    return True


#: Pseudo-classes a focus change on the bar flips on the bar itself.
_BAR_FOCUS_PSEUDO = frozenset({"focus", "focus-within", "blur"})


@pytest.mark.asyncio
async def test_rules_keyed_on_the_bars_focus_style_only_the_declared_nodes():
    """Guard for the scoped focus restyle above (TASK-36511.2).

    The bar's focus watcher restyles only the bar and
    ``INTERACTION_STYLED_DESCENDANTS``. That is exact only while:

    * every rule keyed on the bar's ``:focus``, ``:focus-within`` or
      ``:blur`` above its subject styles one of those nodes. The bar is
      matched with every class it can carry: its own, its state classes,
      and any class a rule pairs with its id or type;
    * no rule keyed on an ancestor's ``:focus-within`` styles another
      composer descendant (Textual's own focus-within pass restyles only
      the rule's subject when it is an ancestor of the focused widget);
    * a focus rule on the bar itself sets no property its descendants read
      from it.
    """
    from Tests.UI.test_console_projection_restyles import _ANCESTOR_READ_RULES

    host = _ready_host()
    async with host.run_test(size=(160, 45)) as pilot:
        composer = await _mounted_composer(host, pilot)
        rules = list(host.stylesheet.rules)
        bar_classes = set(composer.classes) | set(
            ConsoleComposerBar.INTERACTION_CLASSES
        )
        for rule in rules:
            for selector_set in rule.selector_set:
                for parts in _compounds(selector_set.selectors):
                    if any(
                        (part.type == SelectorType.ID and part.name == composer.id)
                        or (
                            part.type == SelectorType.TYPE
                            and part.name == "ConsoleComposerBar"
                        )
                        for part in parts
                    ):
                        bar_classes |= {
                            p.name for p in parts if p.type == SelectorType.CLASS
                        }
        allowed = {
            selector.lstrip("#")
            for selector in ConsoleComposerBar.INTERACTION_STYLED_DESCENDANTS
        }
        others = [n for n in composer.walk_children() if n.id not in allowed]
        ancestors = list(composer.ancestors)
        keyed, offenders, inherited = 0, [], []
        for rule in rules:
            declared = set(rule.styles.get_rules()) & _ANCESTOR_READ_RULES
            for selector_set in rule.selector_set:
                compounds = _compounds(selector_set.selectors)
                subject = compounds[-1]
                text = " ".join(part.css for part in selector_set.selectors)
                for parts in compounds:
                    pseudo = {name for part in parts for name in part.pseudo_classes}
                    on_bar = bool(pseudo & _BAR_FOCUS_PSEUDO) and _could_match(
                        parts, composer, bar_classes
                    )
                    on_ancestor = (
                        parts is not subject
                        and "focus-within" in pseudo
                        and any(
                            _could_match(parts, node, set(node.classes))
                            for node in ancestors
                        )
                    )
                    if not (on_bar or on_ancestor):
                        continue
                    keyed += 1
                    if parts is subject:
                        if declared:
                            inherited.append(
                                f"{text} {{{', '.join(sorted(declared))}}}"
                            )
                    elif on_bar:
                        ids = {s.name for s in subject if s.type == SelectorType.ID}
                        if not ids & allowed:
                            offenders.append(text)
                    elif any(_could_match(subject, n, set(n.classes)) for n in others):
                        offenders.append(text)
        assert keyed, "No rule keys on the composer's focus."
        assert not offenders, (
            "These rules style a composer node from its focus that the scoped "
            f"focus restyle skips; add it to INTERACTION_STYLED_DESCENDANTS: {offenders}"
        )
        assert not inherited, (
            "These composer focus rules set a property its descendants read "
            f"from it: {inherited}"
        )
