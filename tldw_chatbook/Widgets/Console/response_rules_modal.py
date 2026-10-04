"""One explicit-scope editor for private, non-executable response rules."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import TYPE_CHECKING

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    Input,
    Label,
    OptionList,
    Select,
    Static,
    TextArea,
)

from tldw_chatbook.Chat.response_rules.models import (
    RuleBinding,
    RuleCandidate,
    RuleLearningResult,
    RuleScope,
)
from tldw_chatbook.UI.Console_Modules.response_rules import reason_copy
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin

if TYPE_CHECKING:
    from tldw_chatbook.Chat.response_rules.runtime import ResponseRuleRuntime
    from tldw_chatbook.Chat.response_rules.store import ResponseRuleStore


class ResponseRulesModal(SafeModalDismissMixin, ModalScreen[None]):
    """Testing creates an inactive revision; only reviewed Save changes a pin."""

    SAFE_MODAL_CONTENT = "#response-rules-dialog"
    BINDINGS = (("escape", "request_safe_cancel", "Close"),)

    def __init__(
        self, scope: RuleScope, store: ResponseRuleStore, runtime: ResponseRuleRuntime
    ) -> None:
        super().__init__()
        self.scope, self.store, self.runtime = scope, store, runtime
        self.session_id = runtime.controller.store.active_session_id
        self._scopes = (
            runtime.scopes(self.session_id)
            if self.session_id
            else (None, None, RuleScope("global", runtime.profile_id))
        )
        self.tested: RuleLearningResult | None = None
        self._rows = []
        self._selected = None
        self._binding = None
        self._expected = 0
        self._busy = False
        self._dirty = False
        self._loaded_digest = None

    def compose(self) -> ComposeResult:
        with Vertical(id="response-rules-dialog", classes="ds-panel"):
            yield Static("Response rules", classes="dialog-title")
            yield Static(
                "Local to this device. Examples test a rule; they do not guarantee every future answer.",
                id="rr-notice",
                markup=False,
            )
            yield Select(
                [
                    (label, str(index))
                    for index, label in enumerate(
                        ("Current Chat", "Workspace", "Global · this profile")
                    )
                    if self._scopes[index] is not None
                ],
                value=str(self._scopes.index(self.scope)),
                allow_blank=False,
                id="rr-scope",
                classes="form-select",
            )
            with VerticalScroll(id="rr-body"):
                yield OptionList(id="rr-rules")
                yield Static("", id="rr-details", markup=False)
                yield Label("Title", classes="form-label")
                yield Input(id="rr-title", classes="form-input")
                yield Label("Applies to", classes="form-label")
                yield Select(
                    [
                        ("Every answer", "always"),
                        ("When these conditions apply", "semantic"),
                    ],
                    value="always",
                    allow_blank=False,
                    id="rr-applicability",
                    classes="form-select",
                )
                yield Input(
                    placeholder="Applicability conditions",
                    id="rr-app-criteria",
                    classes="form-input",
                )
                yield Label("Check", classes="form-label")
                yield Select(
                    [
                        ("Must include text", "include"),
                        ("Must exclude text", "exclude"),
                        ("Required headings", "headings"),
                        ("Judge these criteria", "semantic"),
                    ],
                    value="include",
                    allow_blank=False,
                    id="rr-detector",
                    classes="form-select",
                )
                yield TextArea(id="rr-criteria", classes="form-textarea")
                yield Static(
                    "One literal or heading per line; semantic checks use the whole criteria text.",
                    markup=False,
                )
                yield Checkbox("Match exact letter case", id="rr-case")
                yield Label("Correction guidance", classes="form-label")
                yield TextArea(id="rr-feedback", classes="form-textarea")
                yield Label(
                    "Replacement example (only when needed)", classes="form-label"
                )
                messages = (
                    self.runtime.controller.store.read_only_messages_for_session(
                        self.session_id
                    )
                    if self.session_id
                    else ()
                )
                yield Select(
                    [
                        ("Completed answer " + str(i + 1), m.id)
                        for i, m in enumerate(messages)
                        if m.role.value == "assistant"
                        and m.status == "complete"
                        and m.content
                        and not m.generation_metadata
                    ],
                    prompt="Use original example",
                    id="rr-example",
                    classes="form-select",
                )
                yield Label("Promote the reviewed revision to", classes="form-label")
                yield Select(
                    [
                        (label, str(i))
                        for i, label in ((1, "Workspace"), (2, "Global · this profile"))
                        if self._scopes[i] is not None
                    ],
                    value="2",
                    allow_blank=False,
                    id="rr-promote-scope",
                    classes="form-select",
                )
                yield Static("", id="rr-examples", markup=False)
            yield Static("", id="rr-status", markup=False)
            with Horizontal(classes="rr-actions dialog-buttons"):
                yield Button("Disable here", id="rr-disable")
                yield Button("Exclude here", id="rr-exclude")
                yield Button("Delete pin", id="rr-delete")
                yield Button("Promote…", id="rr-promote")
            with Horizontal(classes="rr-actions dialog-buttons"):
                yield Button("Test", id="rr-test")
                yield Button("Save", id="rr-save", variant="primary", disabled=True)
                yield Button("Close", id="rr-close")

    async def on_mount(self) -> None:
        super().on_mount()
        await self._reload()

    def _current_scope(self) -> bool:
        try:
            return bool(
                self.session_id
                and self.runtime.controller.store.active_session_id == self.session_id
                and self.runtime.scopes(self.session_id) == self._scopes
            )
        except (KeyError, StopIteration):
            return False

    def _status(self, text: str) -> None:
        if self.is_mounted:
            self.query_one("#rr-status", Static).update(text)

    async def _reload(self) -> None:
        bindings = []
        for scope in self._scopes[self._scopes.index(self.scope) :]:
            if scope is not None:
                bindings.extend(
                    await asyncio.to_thread(self.store.list_bindings, scope)
                )
        drafts = await asyncio.to_thread(self.store.list_drafts, self.scope)
        self._rows = [(b, None) for b in bindings if b.scope == self.scope]
        local = {b.rule_id for b, _ in self._rows}
        for binding in bindings:
            if binding.scope != self.scope and binding.rule_id not in local:
                self._rows.append((binding, None))
                local.add(binding.rule_id)
        self._rows += [(None, d) for d in drafts]
        options = self.query_one("#rr-rules", OptionList)
        options.clear_options()
        for binding, draft in self._rows:
            if binding is not None:
                label = f"{binding.state} · {binding.scope.kind} · revision {binding.revision}"
                try:
                    label = (
                        self.store.get_revision(
                            binding.rule_id, binding.revision
                        ).candidate.title
                        + " · "
                        + label
                    )
                except KeyError:
                    label = "Missing revision · " + label
            else:
                label = "Inactive draft · " + (
                    draft.rule.candidate.title
                    if draft.rule
                    else reason_copy(draft.reason)
                )
            from rich.text import Text

            options.add_option(Text(label))
        if self._rows:
            options.highlighted = 0
            self._load_row(0)
        else:
            self._selected = None
            self.query_one("#rr-details", Static).update(
                "No rules in this scope. Use /omfg <problem> after a completed answer."
            )
            self._buttons()

    def _load_row(self, index: int) -> None:
        binding, draft = self._rows[index]
        self._binding = (
            binding if binding is not None and binding.scope == self.scope else None
        )
        self._expected = self._binding.binding_revision if self._binding else 0
        try:
            rule = (
                self.store.get_revision(binding.rule_id, binding.revision)
                if binding is not None and binding.revision is not None
                else (draft.rule if draft else None)
            )
        except KeyError:
            rule = None
        self._selected, self.tested = rule, None
        if rule is None:
            self.query_one("#rr-details", Static).update(
                "Excluded here."
                if binding is not None and binding.state == "excluded"
                else reason_copy(
                    draft.reason if draft else "original_evidence_unavailable"
                )
            )
            self._buttons()
            return
        candidate = rule.candidate
        if self._binding is None:
            self._binding = next(
                (
                    b
                    for b in self.store.list_bindings(self.scope)
                    if b.rule_id == rule.rule_id
                ),
                None,
            )
            self._expected = self._binding.binding_revision if self._binding else 0
        self._loaded_digest = candidate.candidate_digest()
        self.query_one("#rr-title", Input).value = candidate.title
        self.query_one("#rr-applicability", Select).value = candidate.applicability.kind
        self.query_one("#rr-app-criteria", Input).value = (
            candidate.applicability.criteria or ""
        )
        self.query_one("#rr-detector", Select).value = candidate.detector.kind
        self.query_one("#rr-criteria", TextArea).load_text(
            candidate.detector.criteria or "\n".join(candidate.detector.literals)
        )
        self.query_one("#rr-case", Checkbox).value = candidate.detector.case_sensitive
        self.query_one("#rr-feedback", TextArea).load_text(candidate.feedback)
        inherited = binding is not None and binding.scope != self.scope
        self.query_one("#rr-details", Static).update(
            f"{'Inherited' if inherited else 'Local'} · revision {rule.revision}. Edits create an inactive revision until Test and Save."
        )
        if draft is None:
            # Only this selected scope exposes private calibration examples.
            # A promoted definition does not expose its source Chat's fixtures.
            draft = next(
                (
                    d
                    for _, d in self._rows
                    if d is not None
                    and d.rule is not None
                    and (d.rule.rule_id, d.rule.revision)
                    == (rule.rule_id, rule.revision)
                ),
                None,
            )
        self._show_examples(draft)
        self._buttons()

    def _candidate(self) -> RuleCandidate:
        app = str(self.query_one("#rr-applicability", Select).value)
        detector = str(self.query_one("#rr-detector", Select).value)
        criteria = self.query_one("#rr-criteria", TextArea).text
        return RuleCandidate.model_validate(
            {
                "title": self.query_one("#rr-title", Input).value,
                "applicability": {
                    "kind": app,
                    **(
                        {"criteria": self.query_one("#rr-app-criteria", Input).value}
                        if app == "semantic"
                        else {}
                    ),
                },
                "detector": {
                    "kind": detector,
                    "case_sensitive": self.query_one("#rr-case", Checkbox).value,
                    **(
                        {"criteria": criteria}
                        if detector == "semantic"
                        else {
                            "literals": tuple(
                                line for line in criteria.splitlines() if line.strip()
                            )
                        }
                    ),
                },
                "feedback": self.query_one("#rr-feedback", TextArea).text,
            }
        )

    def _show_examples(self, result: RuleLearningResult | None) -> None:
        lines = []
        validation = (
            result.validation
            if result is not None
            else (
                self.store.get_validation(
                    self._selected.rule_id, self._selected.revision
                )
                if self._selected is not None
                else None
            )
        )
        if validation is not None:
            for case in validation.case_results:
                lines.append(
                    f"{case.case_type.replace('_', ' ')}: {case.check.verdict or 'unavailable'}"
                )
                inputs = (
                    result.fixtures.get(case.case_id) if result is not None else None
                )
                if inputs is not None:
                    lines.extend(
                        (
                            "Request: " + inputs.request_text,
                            "Answer: " + inputs.response_text,
                            "Evidence: "
                            + (
                                "complete" if inputs.evidence_complete else "incomplete"
                            ),
                        )
                    )
        if result is not None and result.reason not in {"tested", "validation_reused"}:
            lines.append(reason_copy(result.reason))
        self.query_one("#rr-examples", Static).update(
            "\n".join(lines)
            or "Recorded examples are unavailable. Choose a replacement completed answer and Test."
        )

    def _buttons(self) -> None:
        # Owned workers can finish while Textual removes the dialog's children.
        # The durable mutation remains valid; a removed view needs no repaint.
        if not self.query("#rr-test"):
            return
        for name in ("test", "disable", "exclude", "promote"):
            self.query_one("#rr-" + name, Button).disabled = (
                self._busy or self._selected is None
            )
        self.query_one("#rr-delete", Button).disabled = (
            self._busy or self._binding is None
        )
        self.query_one("#rr-save", Button).disabled = (
            self._busy
            or self.tested is None
            or self.tested.reason not in {"tested", "validation_reused"}
        )
        self.query_one("#rr-disable", Button).label = (
            "Enable here"
            if self._binding is not None and self._binding.state != "enabled"
            else "Disable here"
        )

    @on(OptionList.OptionSelected, "#rr-rules")
    def select_rule(self, event: OptionList.OptionSelected) -> None:
        if self._has_unsaved():
            self._status(
                "Your unsaved edits are retained. Save them or close and discard before choosing another rule."
            )
            return
        if not self._busy:
            self._load_row(event.option_index)

    def _has_unsaved(self) -> bool:
        if self._selected is None:
            return False
        try:
            return self._candidate().candidate_digest() != self._loaded_digest
        except ValueError:
            return True

    @on(Select.Changed, "#rr-scope")
    async def change_scope(self, event: Select.Changed) -> None:
        if not self.is_mounted or event.value is Select.NULL:
            return
        scope = self._scopes[int(str(event.value))]
        if scope != self.scope and not self._busy:
            if self._has_unsaved():
                self.query_one("#rr-scope", Select).value = str(
                    self._scopes.index(self.scope)
                )
                self._status(
                    "Your unsaved edits are retained. Save them or close and discard before changing scope."
                )
                return
            self.scope = scope
            self.tested = None
            await self._reload()

    @on(Button.Pressed)
    def action_pressed(self, event: Button.Pressed) -> None:
        if event.button.id and event.button.id.startswith("rr-"):
            event.stop()
            self.run_worker(
                self._action(event.button.id[3:]),
                group="response-rule-editor",
                exclusive=False,
            )

    async def _action(self, action: str) -> None:
        if action == "close":
            await self.request_safe_cancel(source="button")
            return
        if self._busy:
            return
        if not self._current_scope():
            self._status(
                "The Chat or Workspace changed. Close this dialog and reopen it; your edits are retained here."
            )
            return
        rule = self._selected
        if rule is None:
            return
        if action == "promote":
            destination = self._scopes[
                int(str(self.query_one("#rr-promote-scope", Select).value))
            ]
            existing = next(
                (
                    b
                    for b in self.store.list_bindings(destination)
                    if b.rule_id == rule.rule_id
                ),
                None,
            )
            expected = existing.binding_revision if existing else 0

            async def commit() -> None:
                if not self._current_scope():
                    self._status("Scope changed. Promotion was cancelled.")
                    return
                try:
                    await asyncio.to_thread(
                        self.store.promote,
                        rule.rule_id,
                        rule.revision,
                        destination,
                        expected_binding_revision=expected,
                    )
                    self._status(
                        "Reviewed revision promoted. Private examples remain in their original scope."
                    )
                except Exception:
                    self._status(
                        "Couldn't promote: the binding changed or saving failed. Review and try again."
                    )

            await self.app.push_screen(
                ConfirmationDialog(
                    title="Promote response rule",
                    message=f"Promote revision {rule.revision} to {destination.kind}?\n\n{rule.candidate.title}\nApplies: {rule.candidate.applicability.kind}\n{rule.candidate.applicability.criteria or ''}\nCheck: {rule.candidate.detector.kind}\n{rule.candidate.detector.criteria or chr(10).join(rule.candidate.detector.literals)}\nCorrection: {rule.candidate.feedback}\n\nThis changes future checks in that scope. Private source text and examples are not copied.",
                    confirm_label="Promote revision",
                    cancel_label="Cancel",
                    confirm_callback=commit,
                )
            )
            return
        self._busy = True
        self._buttons()
        try:
            if action == "test":
                candidate = self._candidate()
                self.tested = None
                self._status("Testing rule…")
                example = self.query_one("#rr-example", Select).value
                self.tested = await self.runtime.test_edit(
                    self.session_id,
                    self.scope,
                    rule.rule_id,
                    rule.revision,
                    candidate,
                    example_message_id=None if example is Select.NULL else str(example),
                )
                self._show_examples(self.tested)
                self._status(reason_copy(self.tested.reason))
            elif action == "save":
                result = self.tested
                if (
                    result is None
                    or result.rule is None
                    or result.validation is None
                    or self._candidate().candidate_digest()
                    != result.rule.candidate.candidate_digest()
                ):
                    self._status(
                        "The definition changed after testing. Test again before Save."
                    )
                    return
                await self.runtime.activate_tested(
                    result, self.scope, expected_binding_revision=self._expected
                )
                self._expected += 1
                self._binding = next(
                    b
                    for b in self.store.list_bindings(self.scope)
                    if b.rule_id == rule.rule_id
                )
                self._loaded_digest = result.rule.candidate.candidate_digest()
                self._status("Rule active · tested against examples.")
            elif action in {"disable", "exclude"}:
                state = (
                    (
                        "enabled"
                        if self._binding is not None
                        and self._binding.state != "enabled"
                        else "disabled"
                    )
                    if action == "disable"
                    else "excluded"
                )
                await asyncio.to_thread(
                    self.store.set_binding,
                    RuleBinding(
                        self.scope,
                        rule.rule_id,
                        rule.revision,
                        state,
                        self._expected + 1,
                    ),
                    expected_binding_revision=self._expected,
                )
                await self._reload()
                self._status(
                    "Enabled here."
                    if state == "enabled"
                    else (
                        "Disabled here."
                        if state == "disabled"
                        else "Excluded here; broader bindings remain unchanged."
                    )
                )
            elif action == "delete":
                await asyncio.to_thread(
                    self.store.delete_binding,
                    self.scope,
                    rule.rule_id,
                    expected_binding_revision=self._expected,
                )
                await self._reload()
                self._status("Local pin deleted. Inherited rules may apply again.")
        except Exception:
            self._status(
                "Couldn't save or test: the definition is invalid, its source/binding changed, or storage is unavailable. Your edits are retained; review and try again."
            )
        finally:
            self._busy = False
            if self.is_mounted:
                self._buttons()

    async def _perform_safe_cancel(self, *, source: str) -> None:
        if self._busy:
            self.runtime.cancel(self.session_id, "cancelled")
            self._status("Testing stopped. Your edits are retained.")
            return
        changed = self._has_unsaved()
        if changed:

            async def discard() -> None:
                self.app.call_after_refresh(self.dismiss_safe_once, None)

            await self.app.push_screen(
                ConfirmationDialog(
                    title="Keep editing?",
                    message="Closing discards the unsaved editor text. Tested inactive drafts stay in this scope.",
                    confirm_label="Discard edits",
                    cancel_label="Keep editing",
                    confirm_callback=discard,
                )
            )
            return
        self.dismiss_safe_once(None)
