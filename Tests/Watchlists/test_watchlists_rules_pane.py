"""Tests for the Watchlists alert rules pane."""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, DataTable, Input, Select, Switch

from tldw_chatbook.UI.Watchlists_Modules.rules_pane import (
    RefreshRulesRequested,
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
        assert len(app.captured_messages) == 1
        kind, payload = app.captured_messages[0]
        assert kind == "save_rule_requested"
        assert payload["name"] == "High error rate"
        assert payload["condition_type"] == "error_rate_above"
        assert payload["condition_value"] == {"threshold": 0.5}
        assert payload["severity"] == "critical"
        assert payload["enabled"] is True
