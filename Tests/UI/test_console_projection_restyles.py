"""TASK-36571: a run's projection pays one restyle per changed widget, not one per class.

Measured live on origin/dev 7536ac3b87 (P71, 2026-10-30; Anthropic haiku via the
review proxy, 160x45): the Console's run projection spent 14-16 ms of every
send window (custody to the provider request) in ``App.update_styles``, about
22 restyles per window. Textual restyles a node's whole subtree once for every
class it changes, so one state flip paid several restyles:

* the Workbench header's status flip (``ready`` -> ``running``) removed one
  status class and added another: two restyles of the header's whole subtree
  (the speech controls live inside it), 4.4 ms of the window's first
  projection;
* the composer's Stop button paid four for one availability flip (the
  ``variant`` watcher's remove and add, then its ``set_class`` calls), and
  Send three for one readiness flip (two for ``variant``, one for its classes);
* the hidden Workbench command strip's buttons, the Agent status line and the
  Send reason strip each paid one restyle per changed class.

These tests pin one restyle per state flip (plus Textual's own for a
``disabled`` change), that the header's status flip restyles the header alone,
and that the targeted restyles leave the styles a full restyle computes.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import replace
from pathlib import Path
from typing import get_args

import pytest
from textual.app import App, ComposeResult
from textual.css.model import CombinatorType, SelectorType
from textual.css.styles import RulesMap
from textual.css.stylesheet import Stylesheet
from textual.screen import Screen
from textual.widgets import Button, Static

import tldw_chatbook
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from Tests.UI.test_console_composer_reason_restyles import ComposerHarness
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.UI.Console_Modules.agent import apply_console_agent_status_state
from tldw_chatbook.UI.Console_Modules.frame import (
    ANCESTOR_READ_RULES,
    apply_scoped_breakpoints,
)
from tldw_chatbook.UI.Navigation.base_app_screen import BaseAppScreen
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Workbench.workbench_state import (
    WorkbenchAction,
    WorkbenchStatus,
)
from tldw_chatbook.UI.Workbench.workbench_widgets import CommandStrip, DestinationHeader
from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar

pytestmark = pytest.mark.bootstrap_profile

#: The built sheets (the app bundle and each lazy screen sheet).
_CSS_DIR = Path(tldw_chatbook.__file__).resolve().parent / "css"
#: Every class the Workbench status sync can set.
_STATUS_CLASSES = frozenset(f"status-{status}" for status in get_args(WorkbenchStatus))
#: One source with the scoped restyle (re-exported for the composer guard).
_ANCESTOR_READ_RULES = ANCESTOR_READ_RULES


def _spy_restyles(host, monkeypatch) -> list:
    """Record every node the app's stylesheet styles from now on."""
    applied: list = []
    original = host.stylesheet.apply

    def apply(node, *args, **kwargs):
        applied.append(node)
        return original(node, *args, **kwargs)

    monkeypatch.setattr(host.stylesheet, "apply", apply)
    return applied


def _styles(root) -> dict[str, dict]:
    """Each node's stylesheet-computed rules under ``root``, by a stable path."""
    return {
        f"{index}:{type(node).__name__}#{node.id}": dict(node._css_styles.get_rules())
        for index, node in enumerate(root.walk_children(with_self=True))
    }


def _ready_host() -> ConsoleHarness:
    app = _build_test_app()
    _configure_native_ready_console(app)
    return ConsoleHarness(app)


async def _mounted(host, pilot, selector: str):
    console = host.screen_stack[-1]
    await _wait_for_selector(console, pilot, selector)
    return console.query_one(selector)


def _other_status(status: str) -> str:
    return "running" if status != "running" else "ready"


@pytest.mark.asyncio
async def test_a_header_status_flip_restyles_only_the_header_once(monkeypatch):
    """A run start's status flip restyles the header node, once.

    Red on origin/dev: ``_sync_status_classes`` removed the old status class
    and added the new one, each restyling the header's whole subtree (the
    title, subtitle, status chip and the speech controls inside it).
    """
    host = _ready_host()
    async with host.run_test(size=(160, 45)) as pilot:
        header = await _mounted(host, pilot, "#console-workbench-header")
        assert isinstance(header, DestinationHeader)
        state = header.state
        flipped = replace(state, status=_other_status(state.status))
        descendants = list(header.walk_children())
        assert descendants, "the header has no children to leave alone"
        applied = _spy_restyles(host, monkeypatch)

        header.sync_state(flipped)

        assert header.has_class(f"status-{flipped.status}")
        assert not header.has_class(f"status-{state.status}")
        header_restyles = sum(1 for node in applied if node is header)
        restyled_children = sorted(
            {
                f"{type(node).__name__}#{node.id}"
                for node in applied
                if node in descendants
            }
        )
        assert (
            header_restyles == 1
        ), f"The header was restyled {header_restyles} times for one status flip."
        assert not restyled_children, (
            "A status flip restyled header descendants no status rule styles: "
            f"{restyled_children}"
        )


@pytest.mark.asyncio
async def test_a_header_status_flip_leaves_the_styles_a_full_restyle_computes():
    """Guard: restyling the header alone changes no computed style below it."""
    host = _ready_host()
    async with host.run_test(size=(160, 45)) as pilot:
        header = await _mounted(host, pilot, "#console-workbench-header")
        for _ in range(2):
            state = header.state
            header.sync_state(replace(state, status=_other_status(state.status)))
            targeted = _styles(header)
            host.update_styles(header)
            assert _styles(header) == targeted


def _target_compound(selectors: list) -> list:
    """The trailing compound selector (the part a rule actually styles)."""
    start = len(selectors) - 1
    while start > 0 and selectors[start].combinator == CombinatorType.SAME:
        start -= 1
    return selectors[start:]


def _ancestor_keyed(rules, classes: set[str]) -> tuple[int, list[str], list[str]]:
    """Count rules keyed on ``classes``; list the ones a node-only restyle misses.

    Returns:
        The number of keyed selectors, those keyed above their target (they
        style a descendant), and those whose rule sets a property the
        target's descendants read from it (``_ANCESTOR_READ_RULES``).
    """
    keyed = 0
    offenders: list[str] = []
    inherited: list[str] = []
    for rule in rules:
        declared = set(rule.styles.get_rules()) & _ANCESTOR_READ_RULES
        for selector_set in rule.selector_set:
            selectors = selector_set.selectors
            hits = [
                index
                for index, part in enumerate(selectors)
                if part.type == SelectorType.CLASS and part.name in classes
            ]
            if not hits:
                continue
            keyed += 1
            text = " ".join(part.css for part in selectors)
            first_target = len(selectors) - len(_target_compound(selectors))
            if any(index < first_target for index in hits):
                offenders.append(text)
            if declared:
                inherited.append(f"{text} {{{', '.join(sorted(declared))}}}")
    return keyed, offenders, inherited


@pytest.mark.asyncio
async def test_no_rule_styles_a_descendant_from_a_workbench_status_class():
    """Guard: status classes style only the node that carries them.

    That is what lets a status flip restyle the class-carrying node alone.
    Scans every rule the mounted app loads (widget DEFAULT_CSS included) and
    every built sheet, parsed on its own, so a lazy screen sheet the harness
    does not load is still covered. Fails with the selector when a rule
    starts keying a descendant on a status class, or when a status-keyed
    rule sets a property the node's descendants read from it (colour,
    background, text style, opacity, tint, link styles): the descendants
    are not restyled, so their cached styles would keep the old value.
    Today every status rule sets ``border-title-color`` alone, which only
    the node itself paints.
    """
    assert _ANCESTOR_READ_RULES <= set(RulesMap.__annotations__)
    classes = set(_STATUS_CLASSES)
    host = _ready_host()
    async with host.run_test(size=(160, 45)) as pilot:
        await _mounted(host, pilot, "#console-workbench-header")
        keyed, offenders, inherited = _ancestor_keyed(host.stylesheet.rules, classes)
        variables = dict(host.stylesheet._variables)
    sheets = sorted(_CSS_DIR.glob("*.tcss"))
    assert sheets, f"No built sheets under {_CSS_DIR}"
    for sheet in sheets:
        text = sheet.read_text(encoding="utf-8")
        if not any(f".{name}" in text for name in classes):
            continue
        stylesheet = Stylesheet(variables=variables)
        stylesheet.add_source(text, read_from=(str(sheet), ""))
        stylesheet.parse()
        sheet_keyed, sheet_offenders, sheet_inherited = _ancestor_keyed(
            stylesheet.rules, classes
        )
        keyed += sheet_keyed
        offenders += [f"{sheet.name}: {selector}" for selector in sheet_offenders]
        inherited += [f"{sheet.name}: {selector}" for selector in sheet_inherited]
    assert keyed, "No rule is keyed on a Workbench status class."
    assert not offenders, (
        "These rules style a descendant from a Workbench status class, which a "
        f"status flip's targeted restyle skips: {offenders}"
    )
    assert not inherited, (
        "These Workbench status rules set a property the node's descendants "
        "read from it; a status flip's targeted restyle leaves the descendants' "
        f"cached styles stale: {inherited}"
    )


@pytest.mark.asyncio
async def test_a_stop_availability_flip_restyles_the_stop_button_once(monkeypatch):
    """Stop's variant and its three state classes cost one restyle.

    Red on origin/dev: the ``variant`` watcher's remove and add, then the
    ``set_class`` calls, restyled Stop four times when a run started.
    """
    host = _ready_host()
    async with host.run_test(size=(160, 45)) as pilot:
        composer = await _mounted(host, pilot, "#console-native-composer")
        assert isinstance(composer, ConsoleComposerBar)
        stop = composer.query_one("#console-stop-generation", Button)
        composer.sync_action_state(
            has_draft=False, run_active=False, can_save_chatbook=False
        )
        assert stop.variant == "default" and stop.has_class("console-stop-idle")
        applied = _spy_restyles(host, monkeypatch)

        composer.sync_action_state(
            has_draft=False, run_active=True, can_save_chatbook=False
        )

        assert stop.variant == "warning"
        assert stop.has_class("-warning") and not stop.has_class("-default")
        assert stop.has_class("console-stop-active")
        assert not stop.classes & {"console-stop-idle", "console-action-disabled"}
        stop_restyles = sum(1 for node in applied if node is stop)
        assert (
            stop_restyles == 1
        ), f"Stop was restyled {stop_restyles} times for one availability flip."
        applied.clear()

        composer.sync_action_state(
            has_draft=False, run_active=False, can_save_chatbook=False
        )

        assert stop.variant == "default"
        assert stop.has_class("-default") and not stop.has_class("-warning")
        assert stop.classes >= {"console-stop-idle", "console-action-disabled"}
        assert sum(1 for node in applied if node is stop) == 1


@pytest.mark.asyncio
async def test_a_send_readiness_flip_restyles_send_once_beside_disabled(monkeypatch):
    """Send's variant joins its readiness classes in one restyle.

    Red on origin/dev: a ready -> empty flip restyled Send four times (its
    ``disabled`` change, the ``variant`` watcher's remove and add, and the
    readiness classes). ``disabled`` keeps Textual's own restyle: its watcher
    also blurs and un-hovers the button.
    """
    host = _ready_host()
    async with host.run_test(size=(160, 45)) as pilot:
        composer = await _mounted(host, pilot, "#console-native-composer")
        send = composer.query_one("#console-send-message", Button)
        composer.load_draft("Reply with the single word ok")
        composer.sync_action_state(
            has_draft=True, run_active=False, can_save_chatbook=False
        )
        assert send.variant == "primary" and not send.disabled
        applied = _spy_restyles(host, monkeypatch)

        composer.sync_action_state(
            has_draft=False, run_active=False, can_save_chatbook=False
        )

        assert send.disabled and send.variant == "default"
        assert send.has_class("-default") and not send.has_class("-primary")
        assert send.classes >= {"console-action-subdued", "console-send-inactive"}
        send_restyles = sum(1 for node in applied if node is send)
        assert (
            send_restyles <= 2
        ), f"Send was restyled {send_restyles} times for one readiness flip."


@pytest.mark.asyncio
async def test_composer_button_flips_leave_the_styles_a_full_restyle_computes():
    """Guard: the batched button restyles compute what Textual's own path does."""
    host = _ready_host()
    async with host.run_test(size=(160, 45)) as pilot:
        composer = await _mounted(host, pilot, "#console-native-composer")
        buttons = [
            composer.query_one("#console-send-message", Button),
            composer.query_one("#console-stop-generation", Button),
        ]
        for has_draft, run_active in ((True, False), (False, True), (False, False)):
            composer.load_draft("draft" if has_draft else "")
            composer.sync_action_state(
                has_draft=has_draft, run_active=run_active, can_save_chatbook=False
            )
            for button in buttons:
                targeted = _styles(button)
                host.update_styles(button)
                assert _styles(button) == targeted, (button.id, has_draft, run_active)


@pytest.mark.asyncio
async def test_a_send_reason_muted_and_size_flip_restyles_the_strip_once(monkeypatch):
    """The reason strip's muted style joins its size-class swap: one restyle.

    Red on origin/dev: ``set_class`` for the muted style restyled the strip,
    then the size-class replacement restyled it again, so going from hidden
    idle guidance to a visible blocker (and back) paid two restyles.
    """
    idle = "console-send-disabled-reason-idle"
    app = ComposerHarness()
    async with app.run_test(size=(160, 20)) as pilot:
        composer = app.composer
        strip = composer.query_one("#console-send-disabled-reason", Static)
        composer._sync_send_disabled_reason("", muted=True)
        await pilot.pause()
        assert strip.has_class(idle) and strip.classes >= {"w-0", "h-0"}
        applied = _spy_restyles(app, monkeypatch)

        composer._sync_send_disabled_reason("Blocked", muted=False)

        assert not strip.has_class(idle)
        assert strip.classes >= {"w-auto", "h-1"}
        restyles = sum(1 for node in applied if node is strip)
        assert (
            restyles == 1
        ), f"The reason strip was restyled {restyles} times for one muted and size flip."
        await pilot.pause()
        assert strip.display and strip.region.height == 1
        applied.clear()

        composer._sync_send_disabled_reason("", muted=True)

        assert strip.has_class(idle) and strip.classes >= {"w-0", "h-0"}
        assert sum(1 for node in applied if node is strip) == 1


class _Harness(App):
    """A bare app: one Static status line and one command strip."""

    def compose(self) -> ComposeResult:
        yield Static("", id="status-line")
        yield CommandStrip(
            (WorkbenchAction("send", "Send", disabled=True),),
            id="strip",
        )


@pytest.mark.asyncio
async def test_an_agent_status_change_restyles_the_status_line_once(monkeypatch):
    """Red on origin/dev: removing the old state class and adding the new one
    restyled the Agent status line twice."""
    app = _Harness()
    async with app.run_test() as pilot:
        await pilot.pause()
        line = app.query_one("#status-line", Static)
        apply_console_agent_status_state(line, "Agent: running · 2 steps")
        assert line.has_class("console-agent-section-status-running")
        applied = _spy_restyles(app, monkeypatch)

        apply_console_agent_status_state(line, "Agent: done · 2 steps")

        assert line.has_class("console-agent-section-status-done")
        assert not line.has_class("console-agent-section-status-running")
        assert sum(1 for node in applied if node is line) == 1


@pytest.mark.asyncio
async def test_a_command_strip_action_flip_restyles_its_button_once_beside_disabled(
    monkeypatch,
):
    """Red on origin/dev: ``is-primary`` and ``is-disabled`` restyled the
    button once each, beside Textual's own restyle for ``disabled``."""
    app = _Harness()
    async with app.run_test() as pilot:
        await pilot.pause()
        strip = app.query_one("#strip", CommandStrip)
        button = strip.query_one(Button)
        assert button.disabled and button.has_class("is-disabled")
        applied = _spy_restyles(app, monkeypatch)

        strip.sync_actions((WorkbenchAction("send", "Send", primary=True),))

        assert not button.disabled
        assert button.has_class("is-primary") and not button.has_class("is-disabled")
        assert sum(1 for node in applied if node is button) <= 2


def test_the_variant_watcher_the_batched_restyle_mirrors_is_textuals():
    """Guard: the composer's batch replays exactly this watcher's two classes.

    A Textual upgrade that adds any watcher behavior fails here rather than
    silently omitting that behavior from Stop and Send's batched path.
    """
    watcher = ast.parse(textwrap.dedent(inspect.getsource(Button.watch_variant)))
    expected = ast.parse(
        'self.remove_class(f"-{old_variant}")\nself.add_class(f"-{variant}")'
    )
    actual = ast.Module(body=watcher.body[0].body, type_ignores=[])
    assert ast.dump(actual, include_attributes=False) == ast.dump(
        expected, include_attributes=False
    )


def _height_classes(rules) -> set[str]:
    """Every ``h-*`` utility class a rule's selector names."""
    return {
        part.name
        for rule in rules
        for selector_set in rule.selector_set
        for part in selector_set.selectors
        if part.type == SelectorType.CLASS and part.name.startswith("h-")
    }


@pytest.mark.asyncio
async def test_an_allocation_pass_restyles_only_the_sections_it_reclasses(monkeypatch):
    """TASK-36511.1: the left rail's prepare pass restyles each section, not its subtree.

    The attach's first allocation pass adds ``h-auto`` to five bounded
    sections; on origin/dev each add restyled the section's whole subtree,
    about 110 nodes and 20-30 ms of CPU in the compact gates' batch. A
    pass with nothing to reclass restyles nothing.
    """
    host = _ready_host()
    async with host.run_test(size=(160, 45)) as pilot:
        rail = await _mounted(host, pilot, "#console-left-rail")
        await pilot.pause()
        sections = [
            section
            for section in rail.query(".console-bounded-section")
            if section.native_scroll_owner is None and section.has_class("h-auto")
        ]
        assert len(sections) >= 3, "too few auto-height sections to reclass"
        for section in sections:  # The state a just-mounted section starts in.
            section.remove_class("h-auto", update=False)
        descendants = {id(node) for s in sections for node in s.walk_children()}
        applied = _spy_restyles(host, monkeypatch)

        rail._prepare_allocation_reconcile()

        assert all(section.has_class("h-auto") for section in sections)
        restyled = [node for node in applied if id(node) in descendants]
        assert not restyled, (
            f"the prepare pass restyled {len(restyled)} section descendants "
            f"for {len(sections)} class changes"
        )
        assert sorted(map(id, applied)) == sorted(map(id, sections))
        for section in sections:
            targeted = _styles(section)
            host.update_styles(section)
            assert _styles(section) == targeted
        applied.clear()
        rail._prepare_allocation_reconcile()
        assert not applied, "an unchanged prepare pass restyled nodes"


@pytest.mark.asyncio
async def test_an_unchanged_inspector_disclosure_restyles_nothing(monkeypatch):
    """TASK-36511.1: ``ConsoleInspectorMore._apply_open`` restyles only on a change.

    On origin/dev it removed its ``h-*`` class and added the same one back,
    restyling its subtree twice (44 nodes) in every attach's allocation batch.
    """
    from tldw_chatbook.Widgets.Console.console_run_inspector import (
        ConsoleInspectorMore,
    )

    host = _ready_host()
    async with host.run_test(size=(235, 52)) as pilot:
        more = await _mounted(host, pilot, "#console-inspector-more")
        assert isinstance(more, ConsoleInspectorMore)
        await pilot.pause()
        descendants = {id(node) for node in more.walk_children()}
        assert descendants
        applied = _spy_restyles(host, monkeypatch)

        more._apply_open()
        assert not applied, f"an unchanged disclosure restyled {len(applied)} nodes"

        for _ in range(2):
            more.set_open(not more.open)
            assert more.has_class("h-auto" if more.open else "h-2")
            assert len([c for c in more.classes if c.startswith("h-")]) == 1
            assert not [node for node in applied if id(node) in descendants]
            assert [node for node in applied if node is more] == [more]
            targeted = _styles(more)
            host.update_styles(more)
            assert _styles(more) == targeted
            applied.clear()


@pytest.mark.asyncio
async def test_no_rule_styles_a_descendant_from_a_height_utility_class():
    """Guard for the two node-only restyles above (TASK-36511.1).

    Swapping a node's ``h-*`` utility class restyles that node alone, which
    is exact only while no rule keys another node on an ``h-*`` class and no
    ``h-*`` rule sets a property descendants read from the node. Today each
    is a bare ``.h-<n> { height: ... }``.
    """
    host = _ready_host()
    async with host.run_test(size=(235, 52)) as pilot:
        await _mounted(host, pilot, "#console-inspector-more")
        rules = list(host.stylesheet.rules)
        variables = dict(host.stylesheet._variables)
    classes = _height_classes(rules)
    assert {"h-auto", "h-2"} <= classes
    keyed, offenders, inherited = _ancestor_keyed(rules, classes)
    for sheet in sorted(_CSS_DIR.glob("*.tcss")):
        text = sheet.read_text(encoding="utf-8")
        if ".h-" not in text:
            continue
        stylesheet = Stylesheet(variables=variables)
        stylesheet.add_source(text, read_from=(str(sheet), ""))
        stylesheet.parse()
        sheet_classes = _height_classes(stylesheet.rules)
        sheet_keyed, sheet_offenders, sheet_inherited = _ancestor_keyed(
            stylesheet.rules, sheet_classes
        )
        keyed += sheet_keyed
        offenders += [f"{sheet.name}: {selector}" for selector in sheet_offenders]
        inherited += [f"{sheet.name}: {selector}" for selector in sheet_inherited]
    assert keyed
    assert not offenders, f"rules key a descendant on an h-* class: {offenders}"
    assert not inherited, f"h-* rules set an ancestor-read property: {inherited}"


class _AwayScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Static("away")


#: An App class only these tests set, and the one rule keyed on it.
_PROBE_CLASS = "-resume-probe"
_PROBE_CSS = f"App.{_PROBE_CLASS} #console-shell {{ tint: red 20%; }}"


class _SwitchHarness(ConsolidatedCSSApp):
    """The ready Console, installed: a switch away suspends it, a switch back resumes it."""

    CSS_PATH = ConsoleHarness.CSS_PATH  # The app's own sheets, as the gate harness loads them.

    def __init__(self, app_instance):
        super().__init__()
        self.app_instance = app_instance

    async def on_mount(self) -> None:
        self.app_instance._ui_ready = True
        self.stylesheet.add_source(_PROBE_CSS, read_from=(__file__, "_PROBE_CSS"))
        self.install_screen(ChatScreen(self.app_instance), "console")
        self.install_screen(_AwayScreen(), "away")
        await self.push_screen("console")


def _switch_host() -> _SwitchHarness:
    app = _build_test_app()
    _configure_native_ready_console(app)
    return _SwitchHarness(app)


async def _switched_host(pilot, host):
    console = await _mounted(host, pilot, "#console-native-composer")
    console = console.screen
    await pilot.pause(1.0)
    return console


def _tinted(console) -> bool:
    return console.query_one("#console-shell").styles.tint.a > 0


def _assert_fully_styled(host, console, message: str) -> None:
    """Every Console node holds the styles a full restyle computes now."""
    shown = _styles(console)
    host.update_styles(console)
    assert shown == _styles(console), message


async def _return_to(host, pilot, console, away=None) -> tuple[list, int]:
    """Switch away (running ``away`` there), then back; the Console nodes restyled."""
    await host.switch_screen("away")
    await pilot.pause()
    if away is not None:
        await away()
    applied = []
    original = host.stylesheet.apply

    def apply(node, *args, **kwargs):
        applied.append(node)
        return original(node, *args, **kwargs)

    host.stylesheet.apply = apply
    try:
        stack_updates = console.stack_updates
        await host.switch_screen("console")
        await pilot.pause()
    finally:
        del host.stylesheet.apply
    nodes = {id(node) for node in console.walk_children(with_self=True)}
    return [node for node in applied if id(node) in nodes], stack_updates


@pytest.mark.asyncio
async def test_a_return_restyles_the_console_only_when_a_style_input_moved():
    """TASK-38950: a return keeps the Console's styles unless something restyles them.

    Textual's ``Screen._on_screen_resume`` restyles every node whenever its
    ``ScreenResume`` asks, and ``switch_screen`` always asks. On the base each
    return restyled all 555 of this harness's Console nodes (546 in the send
    harness). With no rule, App class or App pseudo-class changed while it was
    away, the Console now clears that flag before Textual's handler runs. That
    handler still does the rest of its work: the stack update count moves, and
    a size change made while away lays the Console out at the new size. A rule
    added or a theme changed while away reparses the rules, so that return
    restyles in full and shows it (red under a variant that always skips, and
    the added rule under one that drops the rules check).
    """
    host = _switch_host()
    async with host.run_test(size=(160, 45)) as pilot:
        console = await _switched_host(pilot, host)
        # A widget type first mounted after the Console's first resume adds its
        # DEFAULT_CSS, and the reparse makes a new rules list, so the first
        # return may restyle in full (under load it did, 2 runs in 3). The
        # second return is the steady state.
        await _return_to(host, pilot, console)
        restyled, stack_updates = await _return_to(host, pilot, console)
        assert host.screen is console
        assert console.stack_updates > stack_updates, "the resume handler did not run"
        total = sum(1 for _ in console.walk_children(with_self=True))
        assert len(restyled) * 10 < total, (
            f"a return with no style change restyled {len(restyled)} of {total} nodes"
        )
        styles = _styles(console)
        host.update_styles(console)
        assert _styles(console) == styles, "the skipped restyle left a stale style"

        async def resize():
            await pilot.resize_terminal(150, 40)
            await pilot.pause()

        await _return_to(host, pilot, console, resize)
        await pilot.pause()
        assert console.size == host.size == (150, 40), "the return kept the old layout"

        async def add_sheet():  # Rules alone: no theme, no App class moves.
            host.stylesheet.add_source(
                "#console-shell { tint: blue 30%; }", read_from=(__file__, "add_sheet")
            )
            await pilot.pause()

        await _return_to(host, pilot, console, add_sheet)
        assert _tinted(console), "a rule added while away did not show on return"
        _assert_fully_styled(host, console, "a rule added while away left a stale style")

        before = _styles(console)
        theme = "textual-light" if host.theme != "textual-light" else "textual-dark"

        async def change_theme():
            host.theme = theme
            await pilot.pause()

        restyled, _ = await _return_to(host, pilot, console, change_theme)
        shown = _styles(console)
        host.update_styles(console)
        assert _styles(console) != before, "the theme change styles nothing in the Console"
        assert shown == _styles(console), (
            "the Console came back with styles from before the theme change"
        )
        assert restyled, "the return after a theme change restyled nothing"


def _breakpoint_styled(root, marker: str = ".shell-header-") -> set[int]:
    """Ids of the nodes under ``root`` a rule whose selector holds ``marker`` styles.

    A target's subtree counts too when the rule sets a property descendants
    read from their ancestors (TASK-39132).
    """
    names: set[str] = set()
    deep: set[str] = set()
    for rule in root.app.stylesheet.rules:
        if marker in rule.selectors:
            names |= rule.selector_names
            if _ANCESTOR_READ_RULES & rule.styles.get_rules().keys():
                deep |= rule.selector_names
    styled: set[int] = set()
    for target in root.walk_children(with_self=True):
        if target._selector_names & deep:
            styled |= {id(node) for node in target.walk_children(with_self=True)}
        elif target._selector_names & names:
            styled.add(id(target))
    return styled


@pytest.mark.asyncio
async def test_a_header_breakpoint_flip_restyles_only_what_its_rules_style(monkeypatch):
    """TASK-38950: a shell-header breakpoint flip restyles the nodes its rules style.

    Textual's ``Screen._on_resize`` flips ``shell-header-compact`` /
    ``shell-header-normal`` with ``update_classes``, a whole-screen restyle:
    all 555 Console nodes on the base, on the first layout and at every height
    change across 24/25 rows. Only the header and its subtitle have rules keyed
    on those classes. The Resize is dispatched straight to the screen, so the
    count holds only what its handlers restyle; both heights are under 34 rows,
    so the shell's own ``-console-compact`` class does not flip with it.
    """
    from textual import events
    from textual.geometry import Size

    host = _ready_host()
    async with host.run_test(size=(160, 30)) as pilot:
        await _mounted(host, pilot, "#console-workbench-header")
        console = host.screen
        await pilot.pause(0.5)
        assert console.has_class("shell-header-normal")
        assert console.query_one("#console-shell").has_class("-console-compact")
        styled = _breakpoint_styled(console)
        total = sum(1 for _ in console.walk_children(with_self=True))
        assert styled and len(styled) * 4 < total, (len(styled), total)
        applied = _spy_restyles(host, monkeypatch)
        await console._on_message(events.Resize(Size(160, 24), Size(160, 24)))
        assert console.has_class("shell-header-compact")
        assert not console.has_class("shell-header-normal")
        nodes = {id(node) for node in console.walk_children(with_self=True)}
        restyled = {id(node) for node in applied if id(node) in nodes}
        assert restyled == styled, (
            f"the flip restyled {len(restyled)} of {total} nodes; its rules style {len(styled)}"
        )
        flipped = _styles(console)
        host.update_styles(console)
        assert _styles(console) == flipped, "the scoped restyle left a stale style"

        await pilot.resize_terminal(160, 30)  # A real resize back across 24/25 rows.
        await pilot.pause()
        assert console.has_class("shell-header-normal")
        assert not console.has_class("shell-header-compact")

        # The rule scan is cached per parsed sheet and class set; each input
        # changing alone must rebuild it. A new rule naming a breakpoint class:
        host.stylesheet.add_source(
            ".shell-header-compact #console-transcript-region { tint: blue 30%; }\n"
            ".r11-breakpoint-probe #console-native-composer { tint: red 30%; }",
            read_from=(__file__, "breakpoint cache"),
        )
        apply_scoped_breakpoints(console, BaseAppScreen.VERTICAL_BREAKPOINTS, 24)
        transcript = console.query_one("#console-transcript-region")
        assert transcript.styles.tint.a > 0, "a rule added to the sheet was not scanned"
        assert console.query_one("#console-native-composer") not in (
            transcript.walk_children(with_self=True)
        )
        # Another breakpoint class over the same sheet:
        apply_scoped_breakpoints(console, [(0, "r11-breakpoint-probe")], 24)
        composer = console.query_one("#console-native-composer")
        assert composer.styles.tint.a > 0, "a new breakpoint class was not scanned"
        flipped = _styles(console)
        host.update_styles(console)
        assert _styles(console) == flipped, "a cached scan left a stale style"


@pytest.mark.asyncio
async def test_a_breakpoint_rule_descendants_read_refreshes_their_styles():
    """TASK-39132 review: a breakpoint rule setting a property descendants read
    from their ancestors restyles the target's subtree, so each descendant's
    cached visual style is the one a fresh computation gives. Red when the
    helper never walks a subtree (its ``deep`` set always empty)."""
    from textual.widget import Widget

    host = _ready_host()
    async with host.run_test(size=(160, 45)) as pilot:
        await _mounted(host, pilot, "#console-workbench-header")
        console = host.screen
        await pilot.pause(0.5)
        shell = console.query_one("#console-shell")
        nodes = [node for node in shell.walk_children() if isinstance(node, Widget)]
        for node in nodes:
            node.visual_style  # Cache each against the old ancestry.
        host.stylesheet.add_source(
            ".r12-deep-probe #console-shell { text-style: italic; }",
            read_from=(__file__, "deep probe"),
        )
        apply_scoped_breakpoints(console, [(0, "r12-deep-probe")], 45)
        shown = {id(node): node.visual_style for node in nodes}
        for node in nodes:
            node._visual_style = None
        fresh = {id(node): node.visual_style for node in nodes}
        assert any(style.italic for style in fresh.values()), (
            "the probe rule italicized no descendant"
        )
        stale = [
            f"{type(node).__name__}#{node.id}"
            for node in nodes
            if shown[id(node)] != fresh[id(node)]
        ]
        assert not stale, f"{len(stale)} descendants kept a stale style: {stale[:5]}"


@pytest.mark.asyncio
async def test_a_compact_flip_restyles_only_what_its_rules_style(monkeypatch):
    """TASK-39132: crossing 34/35 rows restyles the nodes the compact rules style.

    ``_adapt_console_shell_to_height`` flipped ``-console-compact`` on
    ``#console-shell`` with ``set_class``, which restyles the shell's whole
    subtree: 528 nodes at 160x40 on the base, at every crossing. Its rules
    set padding, margin and border on six nodes, none of which their
    descendants read, so only those restyle. Heights that stay on one side
    restyle nothing, and the class is still set for readers such as
    ``ChatApprovalCard``. The Resize is dispatched straight to the screen,
    as in the header test above; every height is over 24 rows.
    """
    from textual import events
    from textual.geometry import Size

    host = _ready_host()
    async with host.run_test(size=(160, 40)) as pilot:
        await _mounted(host, pilot, "#console-workbench-header")
        console = host.screen
        await pilot.pause(0.5)
        shell = console.query_one("#console-shell")
        assert not shell.has_class("-console-compact")
        styled = _breakpoint_styled(shell, ".-console-compact")
        total = sum(1 for _ in shell.walk_children(with_self=True))
        assert styled and len(styled) * 20 < total, (len(styled), total)
        applied = _spy_restyles(host, monkeypatch)
        for height, crosses in ((38, False), (34, True), (30, False), (36, True)):
            classes = {id(node): set(node.classes) for node in shell.walk_children()}
            applied.clear()
            await console._on_message(
                events.Resize(Size(160, height), Size(160, height))
            )
            assert shell.has_class("-console-compact") is (height < 35)
            nodes = {id(node) for node in shell.walk_children(with_self=True)}
            restyled = {id(node) for node in applied if id(node) in nodes}
            # The approval card, the one allowed reader, answers the flip with
            # its own class (``approval-compact``, its buttons' compact) and
            # restyles its own subtree; the shell's flip itself is never excused.
            reaction = {
                id(node)
                for reader in shell.walk_children()
                if classes.get(id(reader), set(reader.classes)) != set(reader.classes)
                for node in reader.walk_children(with_self=True)
            }
            assert crosses or not reaction, f"40->{height} moved a reader's classes"
            cards = {
                id(node)
                for card in console.query(ChatApprovalCard)
                for node in card.walk_children(with_self=True)
            }
            assert reaction <= cards, (
                f"40->{height}: {len(reaction - cards)} reaction nodes outside the approval card"
            )
            assert restyled == ((styled if crosses else set()) | reaction), (
                f"40->{height}: restyled {len(restyled)} of {total} shell nodes; "
                f"the compact rules style {len(styled)} and readers' "
                f"reactions {len(reaction)}"
            )
            flipped = _styles(console)
            host.update_styles(console)
            assert _styles(console) == flipped, (
                f"the compact flip to {height} rows left a stale style"
            )


@pytest.mark.asyncio
async def test_an_app_class_that_moved_while_the_console_was_covered_shows_on_return():
    """TASK-38950: an App class change the Console missed restyles it on return.

    App-level class changes restyle only the screens on top. Two ways the
    Console misses one:

    * set while the Console is current, cleared while it is switched away:
      the App classes on return match the last full restyle but not the ones
      the Console left with (red when the suspend check is dropped);
    * set while current, cleared under a pushed screen that then pops: Textual
      posts every pop's ``ScreenResume`` with ``refresh_styles=False`` and
      restyles nothing, so the next return must not skip either.

    A pop with no App change keeps the skip.
    """
    host = _switch_host()
    async with host.run_test(size=(160, 45)) as pilot:
        console = await _switched_host(pilot, host)
        await _return_to(host, pilot, console)  # Settle the sheet (see above).

        # A pop with nothing moved keeps the skip (a dropped record on every
        # pop would bring the full restyle back after each dialog).
        await host.push_screen(_AwayScreen())
        await pilot.pause()
        await host.pop_screen()
        await pilot.pause()
        restyled, _ = await _return_to(host, pilot, console)
        total = sum(1 for _ in console.walk_children(with_self=True))
        assert len(restyled) * 10 < total, (
            f"a pop with no App change cost the next return {len(restyled)} of {total} nodes"
        )

        host.add_class(_PROBE_CLASS)
        await pilot.pause()
        assert _tinted(console), "the probe rule styles nothing in the Console"

        async def clear_probe():
            host.remove_class(_PROBE_CLASS)
            await pilot.pause()

        await _return_to(host, pilot, console, clear_probe)
        assert not _tinted(console), "an App class cleared while away still styles the Console"
        _assert_fully_styled(host, console, "a cleared App class left a stale style")

        host.add_class(_PROBE_CLASS)
        await pilot.pause()
        assert _tinted(console)
        await host.push_screen(_AwayScreen())
        await pilot.pause()
        await clear_probe()
        await host.pop_screen()
        await pilot.pause()
        await _return_to(host, pilot, console)
        assert not _tinted(console), (
            "an App class cleared under a popped screen still styles the Console"
        )
        _assert_fully_styled(host, console, "a popped screen left a stale style")
