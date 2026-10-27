"""Shared Pilot harness for the first-run Provider step tests (TASK-34100.6).

The helpers mirror ``test_first_run_provider_catalog.py``'s private ones: a
real ``FirstRunSetupWizard`` pushed onto a host app that loads the real app
stylesheet, a ``MagicMock`` app instance holding the loaded settings and no
catalog scope service, so picking a provider never reaches the network.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable
from unittest.mock import MagicMock

from textual.app import App, ComposeResult
from textual.widgets import Input, Static

from tldw_chatbook.UI.Wizards.first_run_setup_state import (
    SETUP_DRAFT_VERSION,
    STEP_PROVIDER,
    STEP_WELCOME,
    TRACK_QUICK,
    SetupDraft,
)
from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
    FirstRunSetupWizard,
    ProviderChoiceList,
    ProviderStep,
    SetupWizardContainer,
)

SIZE = (160, 52)


class WizardHost(App):
    """Host app with the real stylesheet (nav geometry depends on it)."""

    CSS_PATH = str(
        Path(__file__).resolve().parents[2] / "tldw_chatbook/css/tldw_cli_modular.tcss"
    )

    def __init__(self, wizard: FirstRunSetupWizard):
        super().__init__()
        self._wizard = wizard
        self.wizard_results: list[object] = []

    def compose(self) -> ComposeResult:
        yield from ()

    async def on_mount(self) -> None:
        self.push_screen(self._wizard, self.wizard_results.append)


def app_instance(mutate: Callable[[dict], None] | None = None) -> MagicMock:
    """A stand-in TldwCli holding the real loaded settings."""
    from tldw_chatbook.config import load_settings

    instance = MagicMock()
    config = copy.deepcopy(load_settings())
    if mutate is not None:
        mutate(config)
    instance.app_config = config
    instance.llm_provider_catalog_scope_service = None
    return instance


def resumed_on_provider(
    mutate: Callable[[dict], None] | None = None, **kwargs: Any
) -> FirstRunSetupWizard:
    """A wizard resumed from "Continue setup?" onto Quick setup ▸ Provider."""
    draft = SetupDraft(
        version=SETUP_DRAFT_VERSION,
        track=TRACK_QUICK,
        active_step_id=STEP_PROVIDER,
        values={STEP_WELCOME: {"track": TRACK_QUICK}},
    )
    return FirstRunSetupWizard(app_instance(mutate), resume_draft=draft, **kwargs)


def current_step(wizard: FirstRunSetupWizard):
    container = wizard.query_one(SetupWizardContainer)
    return container, container.steps[container.current_step]


def step_id(wizard: FirstRunSetupWizard) -> str:
    _container, step = current_step(wizard)
    return step.config.id if step.config is not None else ""


async def wait_for_step(pilot, wizard, wanted: str, *, tries: int = 150) -> None:
    for _ in range(tries):
        _container, step = current_step(wizard)
        if step.config is not None and step.config.id == wanted and step.display:
            await pilot.pause()
            return
        await pilot.pause(0.02)
    raise AssertionError(
        f"the wizard never showed {wanted!r}; it is on {step_id(wizard)!r} "
        f"(pinned: {pinned_error(wizard)!r})"
    )


async def wait_until(pilot, predicate: Callable[[], bool], *, tries: int = 150):
    for _ in range(tries):
        if predicate():
            return
        await pilot.pause(0.02)
    raise AssertionError("condition never held")


def provider_list(step: ProviderStep) -> ProviderChoiceList:
    return step.query_one("#setup-provider-choice", ProviderChoiceList)


def key_input(step: ProviderStep) -> Input:
    return step.query_one("#setup-provider-api-key", Input)


def static_text(node, selector: str) -> str:
    return str(node.query_one(selector, Static).render())


async def pick_provider(pilot, step: ProviderStep, provider: str) -> None:
    """Pick ``provider`` the way a keyboard user does: highlight, then Enter."""
    choices = provider_list(step)
    choices.focus()
    await pilot.pause()
    index = choices.get_option_index(f"provider-{provider}")
    choices.highlighted = index
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()
    assert step.selected_provider_key == provider


async def click_provider_row(pilot, step: ProviderStep, option_id: str) -> None:
    """Click the list row ``option_id`` the way a mouse user does (one click).

    OptionList's own click handler highlights and selects in one go, so the
    highlight's browse and the pick land together -- unlike ``pick_provider``,
    whose highlight settles a frame before Enter.

    Waits for scroll animations first: Textual centres a focused widget that
    doesn't fit with an animated scroll. A row the step's viewport clips is
    scrolled into view first, as a user would before clicking it (a click
    there lands on the step's padding, not the row).
    """
    await pilot.wait_for_scheduled_animations()
    choices = provider_list(step)
    index = choices.get_option_index(option_id)
    line = choices._index_to_line[index]
    if not choices.scroll_offset.y <= line < choices.scroll_offset.y + choices.scrollable_content_region.height:
        choices.scroll_to(y=line, animate=False, immediate=True)
        await pilot.pause()
    y = choices.gutter.top + line - choices.scroll_offset.y
    visible = choices.screen.find_widget(choices).visible_region
    row = choices.region.y + y
    if row >= visible.bottom:
        step.scroll_to(y=step.scroll_y + row - visible.bottom + 1, animate=False)
        await pilot.pause()
    elif row < visible.y:
        step.scroll_to(y=step.scroll_y - (visible.y - row), animate=False)
        await pilot.pause()
    await pilot.click(choices, offset=(choices.gutter.left + 2, y))
    await pilot.pause()


def pinned_error(wizard: FirstRunSetupWizard) -> str:
    strip = wizard.query_one("#setup-step-error-pinned", Static)
    if strip.has_class("hidden"):
        return ""
    return str(strip.render())


def tracker_states(wizard: FirstRunSetupWizard) -> dict[str, str]:
    container = wizard.query_one(SetupWizardContainer)
    from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import SetupWizardProgress

    progress = container.query_one(".wizard-progress", SetupWizardProgress)
    return {item.step_id: item.state for item in progress.items}
