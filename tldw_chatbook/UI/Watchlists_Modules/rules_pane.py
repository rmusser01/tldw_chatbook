"""Alert rules pane for the watchlists screen."""

from __future__ import annotations

from typing import Any

from textual.containers import Grid, Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, DataTable, Input, Select, Static, Switch

from ...Widgets.prune_safe_select import PruneSafeSelect
from ...Widgets.recompose_capture_guard import RecomposeCaptureGuard
from .table_selection import highlight_is_user_driven


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


class EditRuleRequested(Message):
    """Posted when the user requests editing an alert rule."""

    def __init__(self, rule: dict[str, Any]) -> None:
        self.rule = rule
        super().__init__()


class RuleFormVisibilityChanged(Message):
    """Posted whenever the rule form opens or closes, and which rule (if any)
    it is editing.

    `RulesPane` lives inside a `WatchlistsWorkbench` region, and that region
    is swapped for a freshly built one whenever it collapses or expands, or
    whenever the section switches — each of which constructs a brand new
    `RulesPane`. (Until task-15461 the trigger was wider still: `region_
    layout` was `recompose=True`, so `[` on the left rail — a region
    unrelated to Rules — rebuilt this pane too.) Without this message the
    screen has no way to know an edit was in progress, so an open edit form
    would be silently destroyed on the next such rebuild — the same failure
    `CreateFormVisibilityChanged` in
    sources_pane.py already fixes for the Sources create form. The owning
    screen mirrors this into its own state and re-seeds it into the
    freshly-constructed pane via `RulesPane.edit_rule`.
    """

    def __init__(self, is_open: bool, editing_rule: dict[str, Any] | None) -> None:
        self.is_open = is_open
        self.editing_rule = editing_rule
        super().__init__()


class RulesPane(RecomposeCaptureGuard, Vertical):
    """Alert rule list and editor for watchlists."""

    rules = reactive[list[dict[str, Any]]](list, recompose=True)
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

    #: TASK-2310: the Threshold field's unit/meaning depends entirely on the
    #: selected condition -- `LocalWatchlistsService._evaluate_condition`
    #: (`Subscriptions/local_watchlists_service.py`) reads it as a FRACTION
    #: for `error_rate_above` (compared directly against an 0-1 error rate,
    #: formatted as a percentage in the resulting alert) and as an item
    #: COUNT for `items_below`/`items_above`; `no_items`/`run_failed` never
    #: read it at all. UAT: the field had a bare "Threshold" placeholder and
    #: no unit anywhere, so "50" (a very reasonable guess for "50%") would
    #: silently mean "an error rate of 5000%" -- i.e. never fires.
    _THRESHOLD_GUIDANCE: dict[str, tuple[str, str]] = {
        "no_items": (
            "Not used",
            "Not used for this condition -- it fires whenever a run yields "
            "zero items.",
        ),
        "error_rate_above": (
            "0.5 = 50%",
            "Fraction of failed items, 0-1 (e.g. 0.5 for a 50% error rate).",
        ),
        "items_below": ("e.g. 5", "Item count for the run (whole number)."),
        "items_above": ("e.g. 1000", "Item count for the run (whole number)."),
        "run_failed": (
            "Not used",
            "Not used for this condition -- it fires whenever a run's "
            "status is failed.",
        ),
    }
    _DEFAULT_THRESHOLD_GUIDANCE = ("Threshold", "")

    @classmethod
    def _threshold_guidance(cls, condition_type: str) -> tuple[str, str]:
        """The (placeholder, help line) pair for a condition's Threshold field."""
        return cls._THRESHOLD_GUIDANCE.get(
            condition_type, cls._DEFAULT_THRESHOLD_GUIDANCE
        )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._editing_rule_id: str | None = None

    def compose(self):
        with Horizontal(id="rules-toolbar", classes="destination-filter-strip"):
            yield Button("Refresh", id="rules-refresh-button", variant="primary")
            yield Button("New Rule", id="rules-new-button", variant="primary")

        if self.show_rule_form:
            rule = self.selected_rule if self._editing_rule_id else None
            condition_type = (
                str(rule.get("condition_type") or "no_items") if rule else "no_items"
            )
            with Grid(id="rules-create-form"):
                yield Input(
                    placeholder="Name",
                    id="rules-create-name",
                    value=str(rule.get("name") or "") if rule else "",
                )
                yield Static("Condition", classes="watchlists-inline-select-label")
                yield PruneSafeSelect(
                    self._CONDITION_OPTIONS,
                    value=condition_type,
                    id="rules-create-condition",
                    allow_blank=False,
                )
                threshold_value = ""
                if rule:
                    condition_value = rule.get("condition_value") or {}
                    if isinstance(condition_value, dict):
                        threshold_value = str(condition_value.get("threshold", ""))
                threshold_placeholder, threshold_help = self._threshold_guidance(
                    condition_type
                )
                yield Input(
                    placeholder=threshold_placeholder,
                    id="rules-create-threshold",
                    value=threshold_value,
                )
                yield Static(
                    threshold_help,
                    id="rules-create-threshold-help",
                    classes="rules-create-threshold-help",
                )
                yield Static("Severity", classes="watchlists-inline-select-label")
                yield PruneSafeSelect(
                    self._SEVERITY_OPTIONS,
                    value=str(rule.get("severity") or "warning") if rule else "warning",
                    id="rules-create-severity",
                    allow_blank=False,
                )
                yield Horizontal(
                    Static("Enabled"),
                    Switch(
                        value=bool(rule.get("enabled", True)) if rule else True,
                        id="rules-create-enabled",
                    ),
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
        # TASK-2313, AC#4: one line of guidance for the bare-table state,
        # matching Runs/Notifications' identical fix.
        if not self.rules:
            yield Static(
                "No alert rules yet. Press New Rule to watch for a "
                "condition like a run failing or items dropping off.",
                id="rules-empty-state",
                classes="watchlists-hint-line",
            )

    def on_select_changed(self, event: Select.Changed) -> None:
        """TASK-2310: repaint the Threshold guidance when Condition changes.

        In place, not a recompose -- the form is mid-edit (the Name Input
        may already carry typed text) and a recompose would rebuild every
        field from scratch, discarding it.
        """
        if event.select.id != "rules-create-condition":
            return
        event.stop()
        try:
            threshold_input = self.query_one("#rules-create-threshold", Input)
            threshold_help = self.query_one("#rules-create-threshold-help", Static)
        except Exception:
            return
        placeholder, help_text = self._threshold_guidance(str(event.value))
        threshold_input.placeholder = placeholder
        threshold_help.update(help_text)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        self.select_rule_by_id(str(event.row_key.value))

    def on_data_table_cell_selected(self, event: DataTable.CellSelected) -> None:
        event.stop()
        self.select_rule_by_id(str(event.cell_key.row_key.value))

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Select on cursor movement, which is what a mouse click produces.

        TASK-1105, matching `SourcesPane`. `RowSelected`/`CellSelected` fire on
        *activation* -- Enter, or a second click on an already-current cell --
        so a single click on any row but the current one moved the cursor and
        selected nothing.
        """
        event.stop()
        if not highlight_is_user_driven(event):
            return
        if event.row_key is not None and event.row_key.value is not None:
            self.select_rule_by_id(str(event.row_key.value))

    def on_data_table_cell_highlighted(self, event: DataTable.CellHighlighted) -> None:
        """Same, for a table whose cursor is cell-shaped rather than row-shaped."""
        event.stop()
        if not highlight_is_user_driven(event):
            return
        row_key = getattr(event.cell_key, "row_key", None)
        if row_key is not None and row_key.value is not None:
            self.select_rule_by_id(str(row_key.value))

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

    def watch_show_rule_form(self, is_open: bool) -> None:
        """Tell the owning screen the rule form opened or closed, and on what.

        Mirrors `show_rule_form` (plus which rule, if any, is being edited)
        into a `RuleFormVisibilityChanged` message so the screen can persist
        it across a workbench rebuild — see that message's docstring for why
        this pane cannot just rely on its own reactives surviving a
        recompose.

        Args:
            is_open: The form's new visibility.
        """
        if self.is_mounted:
            editing_rule = self.selected_rule if self._editing_rule_id else None
            self.post_message(RuleFormVisibilityChanged(is_open, editing_rule))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = str(event.button.id)
        if button_id == "rules-new-button":
            self._editing_rule_id = None
            self.selected_rule = None
            self.show_rule_form = True
        elif button_id == "rules-create-cancel":
            self.show_rule_form = False
            self._editing_rule_id = None
        elif button_id == "rules-create-submit":
            self._submit_rule_form()
        elif button_id == "rules-refresh-button":
            self.post_message(RefreshRulesRequested())
        event.stop()

    def _submit_rule_form(self) -> None:
        name = self.query_one("#rules-create-name", Input).value.strip()
        if not name:
            self.app.notify("Rule name is required.", severity="error")
            return
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
        payload: dict[str, Any] = {
            "name": name,
            "condition_type": condition_type,
            "condition_value": condition_value,
            "severity": severity,
            "enabled": enabled,
        }
        if self._editing_rule_id:
            payload["id"] = self._editing_rule_id
        self.post_message(SaveRuleRequested(payload))
        self.show_rule_form = False
        self._editing_rule_id = None

    def edit_rule(self, rule: dict[str, Any]) -> None:
        """Open the rule form pre-filled for editing.

        Two routes on purpose (task-15778). Mounted (the interactive
        `EditRuleRequested` path), the plain assignments are the point:
        `show_rule_form`'s recompose renders the form and
        `watch_selected_rule` tells the screen. Unmounted (the
        `_build_detail_pane` factory re-seeding an in-progress edit across
        a region rebuild), both watchers are `is_mounted`-gated no-ops and
        `compose()` reads the same reactives, so the plain assignments
        bought nothing there but a queued `_check_recompose` that tore the
        freshly mounted pane straight back down -- the pre-mount seeding
        recompose that task removes.

        Args:
            rule: The rule row to pre-fill the form with.
        """
        self._editing_rule_id = str(rule.get("id") or "")
        if self.is_mounted:
            self.selected_rule = rule
            self.show_rule_form = True
            return
        self.set_reactive(RulesPane.selected_rule, rule)
        self.set_reactive(RulesPane.show_rule_form, True)
