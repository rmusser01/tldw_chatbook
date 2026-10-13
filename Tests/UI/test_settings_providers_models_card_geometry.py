"""Painted-geometry net for the Providers & Models card (TASK-33007.1).

The screen-decomposition spec's Six Migration Rules (rule 3,
``Docs/superpowers/specs/2026-08-02-screen-decomposition-design.md``) require
every extraction to carry a test that the moved region's controls are
hit-testable -- ``get_widget_at(*control.region.center)`` resolves to the
control -- at 160x45 and 235x52, written against the code *before* the move.

The control set is discovered, not listed: every interactive widget the card
paints (with both of its disclosures opened) is scrolled to the top of the
detail pane and hit-tested. Later Phase 7 tasks reorder the card; a
discovered set keeps this net valid without pinning today's ids, and the
floor keeps it from passing vacuously.
"""

from __future__ import annotations

import pytest
from textual.errors import NoWidget
from textual.widget import Widget
from textual.widgets import (
    Button,
    Checkbox,
    Collapsible,
    Input,
    OptionList,
    Select,
    SelectionList,
)
from textual.widgets._collapsible import CollapsibleTitle

from Tests.private_profile import private_profile_test
from Tests.UI.test_destination_shells import _active_destination_screen
from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_category_sweep import (
    _click_settings_category,
    _settle_settings,
)
from Tests.UI.test_settings_narrow_layout import _SettingsCssHarness

_CARD_SELECTOR = "#settings-providers-models-card"
_CONTROL_TYPES = (
    Button,
    Checkbox,
    CollapsibleTitle,
    Input,
    OptionList,
    Select,
    SelectionList,
)
# The card paints well over this many controls at rest for the default
# (OpenAI) provider; a floor stops the discovered set passing vacuously.
_MIN_CONTROLS = 30
# A late Settings refresh can land between the scroll and the hit-test; the
# scroll is re-issued until the control's centre is on screen.
_SCROLL_SETTLE_ATTEMPTS = 5


def _painted(widget: Widget, card: Widget) -> bool:
    """Return True when the widget and every ancestor up to the card display.

    Args:
        widget: A widget inside the card.
        card: The card container that bounds the ancestor walk.

    Returns:
        False when the widget, or any ancestor below the card, is hidden.
    """
    node: Widget | None = widget
    while node is not None and node is not card:
        if not node.display or not node.visible:
            return False
        node = node.parent  # type: ignore[assignment]
    return node is card


def _card_controls(card: Widget) -> list[Widget]:
    """Return the card's painted interactive widgets, outermost first.

    A Select's own popup OptionList is excluded: it is the Select's
    internals, hit-tested through the Select itself.

    Args:
        card: The Providers & Models card container.

    Returns:
        Every painted control in DOM order.
    """
    controls: list[Widget] = []
    for widget in card.query("*"):
        if not isinstance(widget, _CONTROL_TYPES):
            continue
        if any(isinstance(ancestor, Select) for ancestor in widget.ancestors):
            continue
        if _painted(widget, card):
            controls.append(widget)
    return controls


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(160, 45), (235, 52)])
@private_profile_test
async def test_every_providers_models_card_control_is_hit_testable(request, size):
    app = _build_test_app()
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=size) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "providers-models")
        screen = _active_destination_screen(host)
        card = screen.query_one(_CARD_SELECTOR)
        for disclosure in card.query(Collapsible):
            disclosure.collapsed = False
        await pilot.pause()
        # TASK-33007.5, extended on purpose: an expanded Collapsible scrolls
        # itself into view with an animation (Textual's _watch_collapsed); the
        # nested Sampling disclosure made that scroll land after the first
        # control's scroll_visible, so the pane is let settle first.
        await pilot.wait_for_scheduled_animations()

        controls = _card_controls(card)
        assert len(controls) >= _MIN_CONTROLS, (
            f"only {len(controls)} painted controls found in the card at {size}"
        )
        misses = []
        for control in controls:
            for _attempt in range(_SCROLL_SETTLE_ATTEMPTS):
                control.scroll_visible(animate=False, top=True, immediate=True)
                await pilot.pause()
                if host.screen.region.contains_point(control.region.center):
                    break
            region = control.region
            try:
                hit, _ = host.get_widget_at(*region.center)
            except NoWidget:
                misses.append(f"{type(control).__name__}#{control.id} at {region}")
                continue
            if hit is not control and control not in hit.ancestors:
                misses.append(
                    f"{type(control).__name__}#{control.id} at {region} "
                    f"-> {type(hit).__name__}#{hit.id}"
                )
        assert not misses, f"controls not hit-testable at {size}: {misses}"


def _blank_rows_below(box: Widget) -> int:
    """Return the rows a box paints under its last displayed child."""
    shown = [child for child in box.children if child.display]
    return box.region.bottom - max(child.region.bottom for child in shown)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(211, 44), (235, 52)])
@pytest.mark.parametrize("box_id", ["return-continuation", "navigation-conflict"])
@private_profile_test
async def test_a_routed_pages_return_box_is_as_tall_as_its_rows(
    request, monkeypatch, box_id, size
):
    """Capture fix 3, finding 3 (Phase 8's capture): on a provider page a
    Chat settings link opened, the return box under Default model was a
    plain Vertical whose 1fr height took the card's leftover rows: three
    one-row children in eight rows, five blank under Stay in Settings. Both
    return boxes are as tall as what they hold, on a page that is not the
    default provider's."""
    from textual.widgets import Input

    from Tests.UI.test_settings_configuration_hub import (
        _capture_provider_settings_mutations,
        _stage_conversation_settings_return_intent,
    )
    from tldw_chatbook.UI.Screens.settings_screen import (
        SettingsCategoryId,
        SettingsDraft,
    )

    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-5"}
    app.app_config["api_settings"] = {"openai": {}, "anthropic": {}}
    _intent, target = _stage_conversation_settings_return_intent(
        app, provider="anthropic"
    )
    _capture_provider_settings_mutations(monkeypatch)
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=size) as pilot:
        await _settle_settings(pilot)
        await host.workers.wait_for_complete()
        await pilot.pause()
        screen = _active_destination_screen(host)
        if box_id == "navigation-conflict":
            draft = SettingsDraft(category=SettingsCategoryId.PROVIDERS_MODELS)
            draft.set_value("endpoint", "", "https://draft.example/v1")
            screen._settings_drafts[SettingsCategoryId.PROVIDERS_MODELS] = draft
        screen.apply_navigation_context(target.to_context())
        await pilot.pause()
        if box_id == "return-continuation":
            key = screen.query_one("#settings-provider-api-key", Input)
            key.value = "DUMMY-ANTHROPIC-RETURN-KEY"
            await pilot.pause()
            screen.action_settings_save_category(allow_text_entry_focus=True)
        await pilot.pause()
        await pilot.pause()

        box = screen.query_one(f"#settings-provider-{box_id}")
        assert box.display
        assert _blank_rows_below(box) == 0, box.region
