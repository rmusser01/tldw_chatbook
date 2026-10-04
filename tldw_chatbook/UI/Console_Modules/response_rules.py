"""Thin Console adapters for app-owned response rules."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from tldw_chatbook.Chat.response_rules.models import RuleRuntimeState, RuleScope

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_command_grammar import CommandParse
    from tldw_chatbook.Chat.response_rules.runtime import ResponseRuleRuntime


REASON_COPY = {
    "too_many_effective_rules": "This change would exceed 16 active rules for a Chat. Disable or exclude a rule, then try again.",
    "active_run": "Finish or stop the current run before learning a rule.",
    "no_eligible_response": "Choose a completed text answer before learning a rule.",
    "original_evidence_unavailable": "The original example is unavailable. Choose a completed replacement answer and Test again.",
    "source_changed": "The answer or rule changed. Review it and Test again.",
    "helper_unavailable": "Couldn't verify: the rule helper is unavailable. Try again when capacity is free.",
    "rules_unavailable": "Couldn't verify: active rules are unavailable or exceed the check limit. Open /rules to review them.",
    "learning_timeout": "Couldn't verify within the learning time limit. Your draft is retained.",
    "calibration_failed": "Couldn't verify this rule against the examples. It remains inactive.",
    "feedback_out_of_scope": "The correction adds work outside the original request. Narrow it and Test again.",
    "complaint_too_large": "The problem description is too large (maximum 8 KiB). Shorten it and try again.",
    "cancelled": "Testing stopped. Your edits are retained.",
    "stopped": "Testing stopped. Your edits are retained.",
    "tested": "Tested against examples. Review the definition, then Save to activate.",
    "validation_reused": "Unchanged checks reuse the recorded example test. Review the guidance, then Save.",
    "initial_repair_unavailable": "Rule active for this Chat. The current answer couldn't be repaired; try Continue when the agent is ready.",
    "initial_repair_cancelled": "Rule active for this Chat. Repair stopped; the completed work is retained.",
}


def reason_copy(reason: str) -> str:
    """Use body-free, readable refusal copy instead of diagnostic payloads."""
    return REASON_COPY.get(
        reason, "Couldn't verify. Your draft is retained; review it and try again."
    )


def rule_status(state: RuleRuntimeState) -> str:
    """Project lifecycle and honest mixed verdicts into the existing run chip."""
    phases = {
        "drafting": "Drafting rule",
        "testing": "Testing rule",
        "checking": "Checking rules",
        "repairing": "Repairing rule violation",
    }
    if state.phase in phases:
        return phases[state.phase]
    if state.reason == "correction_limit":
        return "Correction limit reached"
    if state.reason == "rules_unavailable":
        return reason_copy(state.reason)
    if state.assessment is not None:
        text = {
            "pass": "Passed active rules",
            "violation": "Rule violation",
            "couldnt_verify": "Couldn't verify",
            "no_applicable_rules": "No applicable rules",
        }[state.assessment.outcome]
        if state.assessment.outcome != "couldnt_verify" and any(
            c.verdict == "couldnt_verify" or c.applicability == "unknown"
            for c in state.assessment.checks
        ):
            text += " · some checks unavailable"
        return text
    if state.learning is not None:
        return (
            "Tested against examples"
            if state.learning.reason in {"tested", "validation_reused", "activated"}
            or state.learning.state == "active"
            else reason_copy(state.learning.reason)
        )
    return ""


class ConsoleResponseRulesUI:
    """Named late-bound accessors keep disposable views outside domain ownership."""

    def __init__(
        self,
        *,
        runtime_accessor: Callable[[], ResponseRuleRuntime | None],
        session_accessor: Callable[[], str | None],
        composer_accessor: Callable[[], Any],
        show_manager: Callable[[Any], Awaitable[None]],
        notify: Callable[[str], None],
    ) -> None:
        self._runtime, self._session, self._composer = (
            runtime_accessor,
            session_accessor,
            composer_accessor,
        )
        self._show, self._notify = show_manager, notify

    async def learn(self, parse: CommandParse) -> None:
        """Learn from the current eligible answer while preserving refused drafts."""
        if not parse.args.strip():
            self._notify("Usage: /omfg <what was wrong with the answer>")
            return
        runtime, session_id = self._runtime(), self._session()
        if runtime is None or session_id is None:
            self._notify(reason_copy("no_eligible_response"))
            return
        composer = self._composer()
        captured = composer.capture_draft_for_send() if composer is not None else None
        try:
            result = await runtime.learn(session_id, parse.args.strip())
        except Exception:
            self._notify("Couldn't save the rule. Your draft is retained; try again.")
            return
        if result.state != "active":
            self._notify(reason_copy(result.reason))
            return
        current = self._composer()
        if (
            current is not None
            and self._session() == session_id
            and captured is not None
            and current.capture_draft_for_send() == captured
        ):
            current.load_draft("")
        self._notify(
            reason_copy(result.reason)
            if result.reason.startswith("initial_repair_")
            else "Rule active for this Chat - tested against examples."
        )

    async def manage(self, parse: CommandParse) -> None:
        runtime, session_id = self._runtime(), self._session()
        if runtime is None or session_id is None:
            self._notify("Open a Chat to manage its response rules.")
            return
        names = {"": 0, "chat": 0, "workspace": 1, "global": 2}
        choice = parse.args.strip().lower()
        if choice not in names:
            self._notify("Usage: /rules [chat|workspace|global]")
            return
        scope = runtime.scopes(session_id)[names[choice]]
        if scope is None:
            self._notify("This Chat has no named Workspace. Use Chat or global scope.")
            return
        await self.open_manager(scope)

    async def open_manager(self, scope: RuleScope) -> None:
        """Open the same manager from composer or scoped settings."""
        from tldw_chatbook.Widgets.Console.response_rules_modal import (
            ResponseRulesModal,
        )

        runtime = self._runtime()
        if runtime is None:
            self._notify("Response rules require a local profile.")
            return
        await self._show(ResponseRulesModal(scope, runtime.store, runtime))


async def open_profile_rules(screen: Any) -> None:
    """Canonical F9 entry, using the current profile's existing Console owner."""
    from tldw_chatbook.Widgets.Console.response_rules_modal import ResponseRulesModal

    from tldw_chatbook.Chat.console_runtime import ensure_console_runtime

    owner = ensure_console_runtime(screen.app_instance)
    rules = await asyncio.to_thread(owner.ensure_response_rules)
    if rules is None:
        screen.notify("Response rules require a local profile.")
        return
    await screen.app.push_screen(
        ResponseRulesModal(RuleScope("global", rules.profile_id), rules.store, rules)
    )
