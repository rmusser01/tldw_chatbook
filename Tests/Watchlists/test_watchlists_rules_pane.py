"""Tests for the Watchlists alert rules pane."""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, DataTable, Input, Select, Switch

from tldw_chatbook.UI.Watchlists_Modules.rules_pane import (
    RefreshRulesRequested,
    RuleFormVisibilityChanged,
    RuleSelected,
    RulesPane,
    SaveRuleRequested,
)


class RulesPaneHarness(App):
    def __init__(self):
        super().__init__()
        self.captured_messages = []

    def compose(self) -> ComposeResult:
        yield RulesPane()

    def on_rule_selected(self, message: RuleSelected) -> None:
        self.captured_messages.append(("rule_selected", message.rule))

    def on_refresh_rules_requested(self, message: RefreshRulesRequested) -> None:
        self.captured_messages.append(("refresh_rules_requested", None))

    def on_save_rule_requested(self, message: SaveRuleRequested) -> None:
        self.captured_messages.append(("save_rule_requested", message.payload))

    def on_rule_form_visibility_changed(
        self, message: RuleFormVisibilityChanged
    ) -> None:
        self.captured_messages.append(
            ("rule_form_visibility_changed", message.is_open, message.editing_rule)
        )


@pytest.fixture
def sample_rules():
    return [
        {
            "id": "local:watchlist_alert_rule:1",
            "rule_id": 1,
            "name": "No items alert",
            "condition_type": "no_items",
            "severity": "warning",
            "enabled": True,
        },
        {
            "id": "local:watchlist_alert_rule:2",
            "rule_id": 2,
            "name": "Run failed alert",
            "condition_type": "run_failed",
            "severity": "critical",
            "enabled": False,
        },
    ]


@pytest.mark.asyncio
async def test_rules_pane_renders_table_and_toolbar():
    app = RulesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(RulesPane)
        assert pane.query_one("#rules-refresh-button", Button)
        assert pane.query_one("#rules-new-button", Button)
        assert pane.query_one("#rules-table", DataTable)


@pytest.mark.asyncio
async def test_rules_pane_populates_table(sample_rules):
    app = RulesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(RulesPane)
        pane.rules = sample_rules
        await pilot.pause()

        table = pane.query_one("#rules-table", DataTable)
        assert table.row_count == 2


@pytest.mark.asyncio
async def test_rules_pane_refresh_posts_request():
    app = RulesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(RulesPane)
        pane.query_one("#rules-refresh-button", Button).press()
        await pilot.pause()

        assert app.captured_messages == [("refresh_rules_requested", None)]


@pytest.mark.asyncio
async def test_rules_pane_selects_rule_and_posts_message(sample_rules):
    app = RulesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(RulesPane)
        pane.rules = sample_rules
        await pilot.pause()

        pane.select_rule_by_id("local:watchlist_alert_rule:1")
        await pilot.pause()

        assert pane.selected_rule == sample_rules[0]
        assert app.captured_messages == [("rule_selected", sample_rules[0])]


@pytest.mark.asyncio
async def test_rules_pane_new_rule_form_posts_request():
    app = RulesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(RulesPane)
        pane.query_one("#rules-new-button", Button).press()
        await pilot.pause()

        assert pane.query_one("#rules-create-form")
        pane.query_one("#rules-create-name", Input).value = "High error rate"
        pane.query_one("#rules-create-condition", Select).value = "error_rate_above"
        pane.query_one("#rules-create-threshold", Input).value = "0.5"
        pane.query_one("#rules-create-severity", Select).value = "critical"
        pane.query_one("#rules-create-enabled", Switch).value = True
        pane.query_one("#rules-create-submit", Button).press()
        await pilot.pause()

        assert not pane.query("#rules-create-form")
        # Fix round 2, Finding 4: the pane now also posts
        # RuleFormVisibilityChanged when the form opens (New Rule button) and
        # closes (Submit) -- legitimate additional messages, not a regression
        # -- so this filters for the save request specifically rather than
        # asserting the total message count.
        save_requests = [
            message
            for message in app.captured_messages
            if message[0] == "save_rule_requested"
        ]
        assert len(save_requests) == 1
        kind, payload = save_requests[0]
        assert kind == "save_rule_requested"
        assert payload["name"] == "High error rate"
        assert payload["condition_type"] == "error_rate_above"
        assert payload["condition_value"] == {"threshold": 0.5}
        assert payload["severity"] == "critical"
        assert payload["enabled"] is True


# --- Fix round 2, Finding 4: `RuleFormVisibilityChanged` lets the owning
# screen mirror the edit-form state so an in-progress edit survives an
# unrelated workbench rebuild, the same treatment
# CreateFormVisibilityChanged already gives the Sources create form. These
# pin the message's payload shape; the end-to-end "survives a rail toggle"
# behavior is covered at the screen level in
# Tests/UI/test_watchlists_destination_shell.py.


@pytest.mark.asyncio
async def test_rules_pane_edit_rule_posts_form_visibility_with_the_rule(sample_rules):
    app = RulesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(RulesPane)
        pane.rules = sample_rules
        await pilot.pause()

        pane.edit_rule(sample_rules[1])
        await pilot.pause()

        assert (
            "rule_form_visibility_changed",
            True,
            sample_rules[1],
        ) in app.captured_messages


@pytest.mark.asyncio
async def test_rules_pane_new_rule_button_posts_form_visibility_without_a_rule():
    app = RulesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(RulesPane)
        pane.query_one("#rules-new-button", Button).press()
        await pilot.pause()

        assert ("rule_form_visibility_changed", True, None) in app.captured_messages


@pytest.mark.asyncio
async def test_rules_pane_cancel_posts_form_closed(sample_rules):
    app = RulesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(RulesPane)
        pane.rules = sample_rules
        await pilot.pause()

        pane.edit_rule(sample_rules[0])
        await pilot.pause()
        pane.query_one("#rules-create-cancel", Button).press()
        await pilot.pause()

        visibility_events = [
            message
            for message in app.captured_messages
            if message[0] == "rule_form_visibility_changed"
        ]
        assert visibility_events[-1][1] is False, (
            "Cancel should post a form-closed event as the LAST visibility change"
        )
