"""Alert rules pane for the watchlists screen."""

from __future__ import annotations

from typing import Any

from textual.containers import Grid, Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, DataTable, Input, Select, Static, Switch


class RuleSelected(Message):
    """Posted when the user selects an alert rule in the rules table."""

    def __init__(self, rule: dict[str, Any] | None) -> None:
        self.rule = rule
        super().__init__()


class RefreshRulesRequested(Message):
    """Posted when the user requests a refresh of the alert rules list."""


class SaveRuleRequested(Message):
    """Posted when the user submits the alert rule form."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        super().__init__()


class RulesPane(Vertical):
    """Alert rule list and editor for watchlists."""

    rules = reactive[list[dict[str, Any]]]([], recompose=True)
    selected_rule = reactive[dict[str, Any] | None](None)
    show_rule_form = reactive(False, recompose=True)
    runtime_backend = reactive("local", recompose=True)

    _CONDITION_OPTIONS = [
        ("No items", "no_items"),
        ("Error rate above", "error_rate_above"),
        ("Items below", "items_below"),
        ("Items above", "items_above"),
        ("Run failed", "run_failed"),
    ]

    _SEVERITY_OPTIONS = [
        ("Info", "info"),
        ("Warning", "warning"),
        ("Critical", "critical"),
    ]

    def compose(self):
        with Horizontal(id="rules-toolbar", classes="destination-filter-strip"):
            yield Button("Refresh", id="rules-refresh-button", variant="primary")
            yield Button("New Rule", id="rules-new-button", variant="primary")

        if self.show_rule_form:
            with Grid(id="rules-create-form"):
                yield Input(placeholder="Name", id="rules-create-name")
                yield Select(
                    self._CONDITION_OPTIONS,
                    value="no_items",
                    id="rules-create-condition",
                    allow_blank=False,
                )
                yield Input(placeholder="Threshold", id="rules-create-threshold")
                yield Select(
                    self._SEVERITY_OPTIONS,
                    value="warning",
                    id="rules-create-severity",
                    allow_blank=False,
                )
                yield Horizontal(
                    Static("Enabled"),
                    Switch(value=True, id="rules-create-enabled"),
                    classes="rules-create-enabled-row",
                )
                yield Button("Save", id="rules-create-submit", variant="success")
                yield Button("Cancel", id="rules-create-cancel", variant="default")

        table = DataTable(id="rules-table")
        table.add_columns("Name", "Condition", "Severity", "Enabled")
        for rule in self.rules:
            table.add_row(
                str(rule.get("name") or "Untitled"),
                str(rule.get("condition_type") or "-"),
                str(rule.get("severity") or "-"),
                "Yes" if rule.get("enabled") else "No",
                key=str(rule.get("id") or id(rule)),
            )
        yield table

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        self.select_rule_by_id(str(event.row_key.value))

    def on_data_table_cell_selected(self, event: DataTable.CellSelected) -> None:
        event.stop()
        self.select_rule_by_id(str(event.cell_key.row_key.value))

    def select_rule_by_id(self, rule_id: str) -> None:
        """Select the rule with the given id and notify listeners."""
        rule = None
        for candidate in self.rules:
            if str(candidate.get("id") or "") == rule_id:
                rule = candidate
                break
        self.selected_rule = rule

    def watch_selected_rule(self, rule: dict[str, Any] | None) -> None:
        if self.is_mounted:
            self.post_message(RuleSelected(rule))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = str(event.button.id)
        if button_id == "rules-new-button":
            self.show_rule_form = True
        elif button_id == "rules-create-cancel":
            self.show_rule_form = False
        elif button_id == "rules-create-submit":
            self._submit_rule_form()
        elif button_id == "rules-refresh-button":
            self.post_message(RefreshRulesRequested())
        event.stop()

    def _submit_rule_form(self) -> None:
        name = self.query_one("#rules-create-name", Input).value.strip()
        condition_type = str(self.query_one("#rules-create-condition", Select).value or "no_items")
        threshold_text = self.query_one("#rules-create-threshold", Input).value.strip()
        severity = str(self.query_one("#rules-create-severity", Select).value or "warning")
        enabled = self.query_one("#rules-create-enabled", Switch).value
        condition_value: dict[str, Any] = {}
        if threshold_text:
            try:
                condition_value["threshold"] = float(threshold_text)
            except ValueError:
                condition_value["threshold"] = threshold_text
        self.post_message(
            SaveRuleRequested(
                {
                    "name": name,
                    "condition_type": condition_type,
                    "condition_value": condition_value,
                    "severity": severity,
                    "enabled": enabled,
                }
            )
        )
        self.show_rule_form = False
