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
* the composer's Stop button paid five for one availability flip (the
  ``variant`` watcher's remove and add, then three ``set_class`` calls), and
  Send three for one readiness flip (two for ``variant``, one for its classes);
* the hidden Workbench command strip's buttons, the Agent status line and the
  Send reason strip each paid one restyle per changed class.

These tests pin one restyle per state flip (plus Textual's own for a
``disabled`` change), that the header's status flip restyles the header alone,
and that the targeted restyles leave the styles a full restyle computes.
"""

from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path
from typing import get_args

import pytest
from textual.app import App, ComposeResult
from textual.css.model import CombinatorType, SelectorType
from textual.css.stylesheet import Stylesheet
from textual.widgets import Button, Static

import tldw_chatbook
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.UI.Console_Modules.agent import apply_console_agent_status_state
from tldw_chatbook.UI.Workbench.workbench_state import (
    WorkbenchAction,
    WorkbenchStatus,
)
from tldw_chatbook.UI.Workbench.workbench_widgets import CommandStrip, DestinationHeader
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar

pytestmark = pytest.mark.bootstrap_profile

#: The built sheets (the app bundle and each lazy screen sheet).
_CSS_DIR = Path(tldw_chatbook.__file__).resolve().parent / "css"
#: Every class the Workbench status sync can set.
_STATUS_CLASSES = frozenset(f"status-{status}" for status in get_args(WorkbenchStatus))


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
            {f"{type(node).__name__}#{node.id}" for node in applied if node in descendants}
        )
        assert header_restyles == 1, (
            f"The header was restyled {header_restyles} times for one status flip."
        )
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


def _ancestor_keyed(rules, classes: set[str]) -> tuple[int, list[str]]:
    """Count rules keyed on ``classes``; list those keyed above their target."""
    keyed = 0
    offenders: list[str] = []
    for rule in rules:
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
            first_target = len(selectors) - len(_target_compound(selectors))
            if any(index < first_target for index in hits):
                offenders.append(" ".join(part.css for part in selectors))
    return keyed, offenders


@pytest.mark.asyncio
async def test_no_rule_styles_a_descendant_from_a_workbench_status_class():
    """Guard: status classes style only the node that carries them.

    That is what lets a status flip restyle the class-carrying node alone.
    Scans every rule the mounted app loads (widget DEFAULT_CSS included) and
    every built sheet, parsed on its own, so a lazy screen sheet the harness
    does not load is still covered. Fails with the selector when a rule
    starts keying a descendant on a status class.
    """
    classes = set(_STATUS_CLASSES)
    host = _ready_host()
    async with host.run_test(size=(160, 45)) as pilot:
        await _mounted(host, pilot, "#console-workbench-header")
        keyed, offenders = _ancestor_keyed(host.stylesheet.rules, classes)
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
        sheet_keyed, sheet_offenders = _ancestor_keyed(stylesheet.rules, classes)
        keyed += sheet_keyed
        offenders += [f"{sheet.name}: {selector}" for selector in sheet_offenders]
    assert keyed, "No rule is keyed on a Workbench status class."
    assert not offenders, (
        "These rules style a descendant from a Workbench status class, which a "
        f"status flip's targeted restyle skips: {offenders}"
    )


@pytest.mark.asyncio
async def test_a_stop_availability_flip_restyles_the_stop_button_once(monkeypatch):
    """Stop's variant and its three state classes cost one restyle.

    Red on origin/dev: the ``variant`` watcher's remove and add, then three
    ``set_class`` calls, restyled Stop five times when a run started.
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
        assert stop_restyles == 1, (
            f"Stop was restyled {stop_restyles} times for one availability flip."
        )
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
        assert send_restyles <= 2, (
            f"Send was restyled {send_restyles} times for one readiness flip."
        )


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

    A Textual upgrade that changes what ``Button.variant`` does to classes
    fails here instead of leaving Stop and Send with stale variant classes.
    """
    source = inspect.getsource(Button.watch_variant)
    assert 'self.remove_class(f"-{old_variant}")' in source
    assert 'self.add_class(f"-{variant}")' in source
    assert source.count("_class(") == 2
