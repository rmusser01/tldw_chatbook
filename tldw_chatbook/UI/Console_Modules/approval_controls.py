"""Staged approval choices; this collaborator never authorizes tool execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tldw_chatbook.Chat.approval_presentation import ApprovalBatchView


class ApprovalDraft:
    """Own choices and deliberate review for one captured presentation."""

    def __init__(self, view: ApprovalBatchView) -> None:
        self.view = view
        self._rows = {row.verdict_key: row for row in view.rows}
        self._choices = {
            row.verdict_key: (
                "deny"
                if row.requires_review
                else "approve_once"
                if "approve_once" in row.legal_decisions
                else row.legal_decisions[0]
                if row.legal_decisions
                else "deny"
            )
            for row in view.rows
        }
        self._reviewed: set[str] = set()

    def stage(self, verdict_key: str, decision: str, *, deliberate: bool) -> bool:
        """Stage a legal choice; same-value deliberate gestures count as review."""
        row = self._rows.get(verdict_key)
        if row is None or decision not in row.legal_decisions:
            return False
        self._choices[verdict_key] = decision
        if deliberate:
            self._reviewed.add(verdict_key)
        return True

    def can_apply(self) -> bool:
        """Return whether every required review has an explicit gesture."""
        return bool(self._rows) and all(
            self._choices[key] in row.legal_decisions
            and (not row.requires_review or key in self._reviewed)
            for key, row in self._rows.items()
        )

    def submit_map(self) -> dict[str, str]:
        """Return the complete map only when all choices are reviewable."""
        return dict(self._choices) if self.can_apply() else {}

    def summary(self, *, compact: bool = False) -> str:
        """Describe complete captured counts and selected scopes."""
        from dataclasses import replace
        from tldw_chatbook.Chat.approval_presentation import scope_copy

        allowed = sum(
            row.call_count
            for row in self.view.rows
            if self._choices[row.verdict_key] != "deny"
        )
        denied = self.view.call_count - allowed
        if compact:
            labels = {
                "approve_once": "Allow once",
                "approve_session": "Until Chatbook exits",
                "allow_matching": "Remember these inputs",
                "always_allow": "Always allow this tool",
                "deny": "Deny",
            }
            counts: dict[str, int] = {}
            profiles = set()
            for row in self.view.rows:
                decision = self._choices[row.verdict_key]
                counts[decision] = counts.get(decision, 0) + row.call_count
                if decision not in {"approve_once", "deny"}:
                    profiles.add(row.authority.profile_label)
            scopes = "; ".join(
                f"{labels[decision]}: {count}" for decision, count in counts.items()
            )
            profile = (
                f" Profile: {next(iter(profiles))}."
                if len(profiles) == 1
                else " Profiles: review the tool rows."
                if profiles
                else ""
            )
            return f"Allow {allowed} · Deny {denied}. {scopes}.{profile}"
        groups = {}
        for row in self.view.rows:
            decision = self._choices[row.verdict_key]
            key = (decision, row.authority, row.action_label)
            if key in groups:
                groups[key] = replace(
                    groups[key], call_count=groups[key].call_count + row.call_count
                )
            else:
                groups[key] = row
        scopes = " ".join(
            f"{row.action_label}: {scope_copy(row, decision)}"
            for (decision, _, _), row in groups.items()
        )
        return f"Allow {allowed} · Deny {denied}. {scopes}"
