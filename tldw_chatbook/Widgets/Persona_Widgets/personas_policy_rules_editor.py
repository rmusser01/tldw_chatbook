"""Personas workbench policy-rules editor (workspace-assistant-defaults).

A list + mini-form CRUD editor over a persona's ``policy_rules`` — the
narrowing-only tool policy carried by local persona records (ADR-079).
Follows the ``settings_agents_panel`` list/form idiom, but owns no storage:
every mutation posts ``PersonaPolicyRulesChanged`` and the screen persists
through the persona service's existing update flow.

Deny-by-default contract (controller ruling, Task 7 review): the moment the
rule set contains at least one ALLOW rule, a visible warning Static explains
that every tool of that kind not named as allowed is un-advertised —
including runtime/builtin tools such as ``spawn_subagent``. The warning is
recomputed on every rules change (load, save, delete).
"""

from __future__ import annotations

from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Checkbox, Input, ListItem, ListView, Static

from tldw_chatbook.Character_Chat.local_character_persona_service import (
    normalize_policy_rules,
)

#: The one ruled warning copy. Names spawn_subagent explicitly because it is
#: the runtime/builtin tool users most often lose without noticing.
DENY_BY_DEFAULT_WARNING = (
    "Allow rules active: tools of this kind not listed as allowed are "
    "un-advertised (including spawn_subagent and other runtime/builtin "
    "tools — name them in a rule to re-admit)."
)

_VALID_KINDS = ("mcp_tool", "skill")


class PersonaPolicyRulesChanged(Message):
    """Posted on every committed rules mutation; carries the full rule list."""

    def __init__(self, rules: list[dict[str, Any]]) -> None:
        super().__init__()
        self.rules = list(rules)


class PersonasPolicyRulesEditor(Vertical):
    """List + mini-form editor over one persona's policy_rules list."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("id", "personas-policy-rules-editor")
        super().__init__(**kwargs)
        self._rules: list[dict[str, Any]] = []
        self._selected_index: int | None = None

    def compose(self) -> ComposeResult:
        yield Static(
            "Tool policy rules (narrowing-only)", classes="destination-section"
        )
        warning = Static(
            DENY_BY_DEFAULT_WARNING,
            id="personas-policy-warning",
            markup=False,
        )
        warning.display = False
        yield warning
        yield ListView(id="personas-policy-rules-list")
        with Vertical():
            with Horizontal(classes="ds-field-row"):
                yield Static("Kind", classes="settings-input-label")
                yield Input(
                    placeholder="mcp_tool or skill",
                    id="personas-policy-kind",
                )
            with Horizontal(classes="ds-field-row"):
                yield Static("Name", classes="settings-input-label")
                yield Input(
                    placeholder="exact tool/skill name",
                    id="personas-policy-name",
                )
            with Horizontal(classes="ds-field-row"):
                yield Static("Allowed", classes="settings-input-label")
                yield Checkbox(value=True, id="personas-policy-allowed")
            with Horizontal(classes="ds-field-row"):
                yield Static("Require confirmation", classes="settings-input-label")
                yield Checkbox(value=False, id="personas-policy-confirm")
            with Horizontal(classes="ds-field-row"):
                yield Static("Max calls/turn", classes="settings-input-label")
                yield Input(
                    placeholder="blank = no cap",
                    id="personas-policy-caps",
                )
            with Horizontal(classes="ds-field-row"):
                yield Button("New", id="personas-policy-new")
                yield Button("Save", variant="primary", id="personas-policy-save")
                yield Button("Delete", variant="error", id="personas-policy-delete")
        yield Static("", id="personas-policy-status", markup=False)

    # -- data push ---------------------------------------------------------

    def show_rules(self, rules: list[dict[str, Any]] | None) -> None:
        """Replace the edited rule list (normalized) and reset the form."""
        self._rules = normalize_policy_rules(rules)
        self._selected_index = None
        self._reload_list()
        self._clear_form()
        self._sync_warning()
        self._set_status("")

    def clear_rules(self) -> None:
        """Clear the editor (no selection / non-persona kinds)."""
        self.show_rules(None)

    # -- internals ----------------------------------------------------------

    def _reload_list(self) -> None:
        try:
            list_view = self.query_one("#personas-policy-rules-list", ListView)
        except Exception:
            return
        list_view.clear()
        for rule in self._rules:
            verb = "allow" if rule.get("allowed", True) else "deny"
            extras: list[str] = []
            if rule.get("require_confirmation"):
                extras.append("confirm")
            if rule.get("max_calls_per_turn") is not None:
                extras.append(f"cap {rule['max_calls_per_turn']}")
            suffix = f" ({', '.join(extras)})" if extras else ""
            list_view.append(
                ListItem(
                    Static(
                        f"{rule.get('rule_kind')}: {rule.get('rule_name')} "
                        f"→ {verb}{suffix}",
                        markup=False,
                    )
                )
            )

    def _clear_form(self) -> None:
        self.query_one("#personas-policy-kind", Input).value = ""
        self.query_one("#personas-policy-name", Input).value = ""
        self.query_one("#personas-policy-allowed", Checkbox).value = True
        self.query_one("#personas-policy-confirm", Checkbox).value = False
        self.query_one("#personas-policy-caps", Input).value = ""

    def _set_status(self, text: str) -> None:
        try:
            self.query_one("#personas-policy-status", Static).update(text)
        except Exception:
            pass

    def _sync_warning(self) -> None:
        """Recompute the deny-by-default warning on every rules change."""
        try:
            warning = self.query_one("#personas-policy-warning", Static)
        except Exception:
            return
        has_allow = any(rule.get("allowed", True) for rule in self._rules)
        warning.display = has_allow

    def _validate_form(self) -> dict[str, Any] | None:
        kind = self.query_one("#personas-policy-kind", Input).value.strip()
        name = self.query_one("#personas-policy-name", Input).value.strip()
        if kind not in _VALID_KINDS:
            self._set_status(
                f"Invalid kind {kind!r}: must be mcp_tool or skill."
            )
            return None
        if not name:
            self._set_status("Rule name is required.")
            return None
        caps_raw = self.query_one("#personas-policy-caps", Input).value.strip()
        caps: int | None = None
        if caps_raw:
            try:
                caps = int(caps_raw)
            except ValueError:
                caps = -1
            if caps < 1:
                self._set_status("Max calls/turn must be an integer ≥ 1 or blank.")
                return None
        return {
            "rule_kind": kind,
            "rule_name": name,
            "allowed": self.query_one("#personas-policy-allowed", Checkbox).value,
            "require_confirmation": self.query_one(
                "#personas-policy-confirm", Checkbox
            ).value,
            "max_calls_per_turn": caps,
        }

    # -- events --------------------------------------------------------------

    @on(ListView.Selected, "#personas-policy-rules-list")
    def _rule_selected(self, event: ListView.Selected) -> None:
        try:
            index = event.list_view.children.index(event.item)
        except ValueError:
            return
        if not 0 <= index < len(self._rules):
            return
        self._selected_index = index
        rule = self._rules[index]
        self.query_one("#personas-policy-kind", Input).value = str(
            rule.get("rule_kind") or ""
        )
        self.query_one("#personas-policy-name", Input).value = str(
            rule.get("rule_name") or ""
        )
        self.query_one("#personas-policy-allowed", Checkbox).value = bool(
            rule.get("allowed", True)
        )
        self.query_one("#personas-policy-confirm", Checkbox).value = bool(
            rule.get("require_confirmation")
        )
        caps = rule.get("max_calls_per_turn")
        self.query_one("#personas-policy-caps", Input).value = (
            "" if caps is None else str(caps)
        )
        self._set_status(f"Editing rule {rule.get('rule_name')!r}.")

    @on(Button.Pressed, "#personas-policy-new")
    def _new_pressed(self) -> None:
        self._selected_index = None
        self._clear_form()
        self._set_status("")

    @on(Button.Pressed, "#personas-policy-save")
    def _save_pressed(self) -> None:
        rule = self._validate_form()
        if rule is None:
            return
        if self._selected_index is not None and 0 <= self._selected_index < len(
            self._rules
        ):
            self._rules[self._selected_index] = rule
        else:
            self._rules.append(rule)
            self._selected_index = len(self._rules) - 1
        self._reload_list()
        self._sync_warning()
        self._set_status(
            f"Saved rule {rule['rule_name']!r} — persona save persists it."
        )
        self.post_message(PersonaPolicyRulesChanged(list(self._rules)))

    @on(Button.Pressed, "#personas-policy-delete")
    def _delete_pressed(self) -> None:
        if self._selected_index is None or not 0 <= self._selected_index < len(
            self._rules
        ):
            self._set_status("Select a rule to delete.")
            return
        removed = self._rules.pop(self._selected_index)
        self._selected_index = None
        self._reload_list()
        self._sync_warning()
        self._clear_form()
        self._set_status(f"Deleted rule {removed.get('rule_name')!r}.")
        self.post_message(PersonaPolicyRulesChanged(list(self._rules)))
