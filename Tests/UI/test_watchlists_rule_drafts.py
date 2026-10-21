"""Raw Rules drafts survive list publication and section reconstruction."""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, DataTable, Input, Select, Static, Switch

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_watchlists_destination_shell import DestinationHarness
from tldw_chatbook.UI.Watchlists_Modules.rules_pane import RulesPane, SaveRuleRequested

RULE = {
    "id": "local:watchlist_alert_rule:1",
    "rule_id": 1,
    "name": "Original",
    "condition_type": "no_items",
    "condition_value": {"threshold": 3},
    "severity": "warning",
    "enabled": True,
}
OTHER = dict(RULE, id="local:watchlist_alert_rule:2", rule_id=2, name="Other")


class RuleDraftHarness(App):
    def __init__(self) -> None:
        super().__init__()
        self.saved = []

    def compose(self) -> ComposeResult:
        pane = RulesPane()
        pane.set_reactive(RulesPane.rules, [RULE, OTHER])
        yield pane

    def on_save_rule_requested(self, event: SaveRuleRequested) -> None:
        self.saved.append(event.payload)


async def _open(pane, pilot, mode):
    if mode == "Edit":
        pane.edit_rule(RULE)
    else:
        pane.query_one("#rules-new-button", Button).press()
    await pilot.pause()
    assert pane.query_one("#rules-create-name", Input).is_mounted


async def _type_draft(pane, pilot):
    name = pane.query_one("#rules-create-name", Input)
    name.focus()
    await pilot.pause()
    await pilot.press("home", "shift+end", "backspace", *"Unsaved rule")
    condition = pane.query_one("#rules-create-condition", Select)
    condition.focus()
    await pilot.pause()
    await pilot.press("enter", "home", "down", "enter")
    severity = pane.query_one("#rules-create-severity", Select)
    severity.focus()
    await pilot.pause()
    await pilot.press("enter", "end", "enter")
    enabled = pane.query_one("#rules-create-enabled", Switch)
    enabled.focus()
    await pilot.pause()
    await pilot.press("space")
    threshold = pane.query_one("#rules-create-threshold", Input)
    threshold.focus()
    await pilot.pause()
    await pilot.press("home", "shift+end", "backspace", *"  invalid-0.  ")
    # Establish actual settled focus and a nonempty selection before publication.
    await pilot.press("home", "right", "shift+right", "shift+right")
    await pilot.pause()
    assert pane.app.focused is threshold
    assert threshold.selection.start != threshold.selection.end
    _assert_draft(pane)
    return threshold


def _assert_draft(pane):
    assert pane.query_one("#rules-create-name", Input).value == "Unsaved rule"
    assert pane.query_one("#rules-create-condition", Select).value == "error_rate_above"
    assert pane.query_one("#rules-create-threshold", Input).value == "  invalid-0.  "
    assert pane.query_one("#rules-create-severity", Select).value == "critical"
    assert pane.query_one("#rules-create-enabled", Switch).value is False
    assert "50%" in str(
        pane.query_one("#rules-create-threshold-help", Static).renderable
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["New", "Edit"])
@pytest.mark.parametrize("size", [(90, 32), (160, 48)], ids=["compact", "wide"])
@private_profile_test
async def test_list_refresh_keeps_raw_rule_draft_and_focus(request, mode, size):
    app = RuleDraftHarness()
    async with app.run_test(size=size) as pilot:
        pane = app.query_one(RulesPane)
        await _open(pane, pilot, mode)
        control = await _type_draft(pane, pilot)
        cursor, selection = control.cursor_position, control.selection
        # Browsing another row must never change the original editing target.
        pane.select_rule_by_id(OTHER["id"])
        pane.rules = [OTHER, dict(RULE, name="Server changed")]
        await pilot.pause()
        _assert_draft(pane)
        assert pane.query_one("#rules-create-threshold", Input) is control
        assert app.focused is control
        assert (control.cursor_position, control.selection) == (cursor, selection)
        assert pane.query_one("#rules-table", DataTable).row_count == 2
        pane.query_one("#rules-create-submit", Button).press()
        await pilot.pause()
        assert app.saved[-1].get("id") == (RULE["id"] if mode == "Edit" else None)


async def _settle_screen(host, pilot):
    await pilot.pause()
    await host.workers.wait_for_complete()
    await pilot.pause()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["New", "Edit"])
@private_profile_test
async def test_section_return_restores_rule_draft_without_saving(request, mode):
    app = _build_test_app()
    service = app.local_watchlists_service
    await service.create_alert_rule(name="Original", condition_type="no_items")
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = host.context_screen
        await _settle_screen(host, pilot)
        screen.query_one("#wl-tab-rules", Button).focus()
        await pilot.pause()
        await pilot.press("enter")
        await _settle_screen(host, pilot)
        pane = screen.query_one(RulesPane)
        original = pane.rules[0]
        if mode == "Edit":
            pane.edit_rule(original)
            await pilot.pause()
        else:
            await _open(pane, pilot, mode)
        await _type_draft(pane, pilot)
        screen.query_one("#wl-tab-runs", Button).focus()
        await pilot.pause()
        await pilot.press("enter")
        await _settle_screen(host, pilot)
        assert screen.active_section == "runs"
        assert not screen.query(RulesPane)
        screen.query_one("#wl-tab-rules", Button).focus()
        await pilot.pause()
        await pilot.press("enter")
        await _settle_screen(host, pilot)
        restored = screen.query_one(RulesPane)
        assert restored is not pane
        _assert_draft(restored)
        assert restored._editing_rule_id == (original["id"] if mode == "Edit" else None)
        assert (await service.list_alert_rules())[0]["name"] == "Original"
        # Cancel retires the retained draft; a new form starts clean after return.
        restored.query_one("#rules-create-cancel", Button).press()
        await _settle_screen(host, pilot)
        screen.query_one("#wl-tab-runs", Button).press()
        await _settle_screen(host, pilot)
        screen.query_one("#wl-tab-rules", Button).press()
        await _settle_screen(host, pilot)
        assert not screen.query("#rules-create-form")
        await _open(screen.query_one(RulesPane), pilot, "New")
        assert screen.query_one("#rules-create-name", Input).value == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["New", "Edit"])
@private_profile_test
async def test_validation_keeps_draft_and_save_retires_it(request, mode):
    app = RuleDraftHarness()
    async with app.run_test(size=(160, 48)) as pilot:
        pane = app.query_one(RulesPane)
        await _open(pane, pilot, mode)
        await _type_draft(pane, pilot)
        name = pane.query_one("#rules-create-name", Input)
        name.focus()
        await pilot.pause()
        await pilot.press("home", "shift+end", "backspace")
        pane.query_one("#rules-create-submit", Button).press()
        await pilot.pause()
        assert pane.show_rule_form
        assert not app.saved
        pane.rules = [OTHER]
        await pilot.pause()
        assert pane.query_one("#rules-create-name", Input).value == ""
        assert (
            pane.query_one("#rules-create-threshold", Input).value == "  invalid-0.  "
        )
        name = pane.query_one("#rules-create-name", Input)
        name.focus()
        await pilot.pause()
        await pilot.press(*"Saved name")
        pane.query_one("#rules-create-submit", Button).press()
        await pilot.pause()
        assert len(app.saved) == 1
        assert not pane.show_rule_form
        await _open(pane, pilot, "New")
        assert pane.query_one("#rules-create-name", Input).value == ""
        assert pane.query_one("#rules-create-threshold", Input).value == ""
