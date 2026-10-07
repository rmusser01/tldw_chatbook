"""Monotonic session-only approval facts, independent of permission state."""

from __future__ import annotations

from dataclasses import dataclass, replace
from threading import RLock

from tldw_chatbook.Agents.approval_observation import (
    ApprovalObservation,
    ApprovalObservationIdentity,
    SCOPES,
)
from .approval_presentation import ApprovalBatchView


@dataclass(frozen=True)
class ApprovalFeedback:
    identity: ApprovalObservationIdentity
    sequence: int = 0
    decision_state: str = "pending"
    selected_scope: str | None = None
    grant_state: str = "unknown"
    applied_scope: str | None = None
    execution_state: str = ""
    terminal_outcome: str = ""
    confirmed_backend_start: bool = False


class ApprovalFeedbackStore:
    """Reduce owner facts under a short lock; no callbacks or UI waits here."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._facts: dict[ApprovalObservationIdentity, ApprovalFeedback] = {}
        self._latest: dict[tuple[str, str], ApprovalObservationIdentity] = {}
        self._counts: dict[ApprovalObservationIdentity, int] = {}
        self._aliases: dict[tuple[str, str], tuple[str, ...]] = {}
        self._projection_calls: dict[ApprovalObservationIdentity, set[str]] = {}
        self._sequence = 0

    def bind_round(self, view: ApprovalBatchView) -> None:
        with self._lock:
            for row in view.rows:
                identity = ApprovalObservationIdentity(
                    view.session_id,
                    view.run_id,
                    view.round_id,
                    view.revision,
                    row.verdict_key,
                )
                key = (view.run_id, row.verdict_key)
                old = self._latest.get(key)
                if old == identity:
                    continue
                if old is not None:
                    self._facts.pop(old, None)
                    self._counts.pop(old, None)
                    self._projection_calls.pop(old, None)
                self._latest[key] = identity
                self._facts[identity] = ApprovalFeedback(identity)
                self._counts[identity] = row.call_count

    def register_aliases(
        self,
        run_id: str,
        aliases: dict[str, tuple[str, ...]],
        *,
        legacy_keys: frozenset[str] = frozenset(),
    ) -> None:
        with self._lock:
            for name, keys in aliases.items():
                self._aliases[run_id, name] = (
                    tuple(keys) if len(keys) == 1 and keys[0] in legacy_keys else ()
                )

    def context_for_call(
        self, run_id: str, call_key: str, *, fallback_tool_name: str = ""
    ) -> ApprovalObservationIdentity | None:
        with self._lock:
            exact = self._latest.get((run_id, call_key)) if call_key else None
            if exact is not None:
                return exact
            keys = self._aliases.get((run_id, fallback_tool_name), ())
            identity = self._latest.get((run_id, keys[0])) if len(keys) == 1 else None
            if identity is not None and call_key:
                self._projection_calls.setdefault(identity, set()).add(call_key)
            return identity

    def projection_call_keys(
        self, identity: ApprovalObservationIdentity
    ) -> tuple[str, ...]:
        """Return only actual invocation members observed for this verdict group."""
        with self._lock:
            return tuple(self._projection_calls.get(identity, ())) or (
                identity.call_key,
            )

    def select_decisions(self, round_id: str, decisions: dict[str, str]) -> None:
        """Record displayed choices without deciding whether execution is permitted."""
        with self._lock:
            for identity, fact in tuple(self._facts.items()):
                scope = decisions.get(identity.call_key)
                if identity.round_id == round_id and scope in SCOPES:
                    self._facts[identity] = replace(fact, selected_scope=scope)

    def publish(self, observation: ApprovalObservation) -> bool:
        identity = observation.identity
        with self._lock:
            fact = self._facts.get(identity)
            if (
                fact is None
                or self._latest.get((identity.run_id, identity.call_key)) != identity
            ):
                return False
            fields = {}
            if (
                observation.kind == "backend_started"
                and not fact.confirmed_backend_start
                and self._counts[identity] == 1
            ):
                fields["confirmed_backend_start"] = True
            if observation.kind == "received" and fact.decision_state == "pending":
                fields["decision_state"] = "received"
            elif observation.kind == "settled" and fact.decision_state in {
                "pending",
                "received",
            }:
                fields["decision_state"] = observation.outcome
            elif observation.kind == "grant" and fact.grant_state == "unknown":
                # Repeated writers cannot prove several distinct exact-input rules.
                if (
                    self._counts[identity] > 1
                    and observation.actual_scope == "allow_matching"
                ):
                    return False
                fields.update(
                    grant_state=observation.outcome,
                    applied_scope=observation.actual_scope
                    if observation.outcome == "applied"
                    else None,
                )
            else:
                order = {
                    "": 0,
                    "dispatch_started": 1,
                    "backend_started": 2,
                    "tool_completed": 3,
                    "model_wait": 4,
                }
                if (
                    observation.kind in order
                    and order[observation.kind] > order[fact.execution_state]
                ):
                    if self._counts[identity] > 1 and observation.kind in {
                        "backend_started",
                        "tool_completed",
                        "model_wait",
                    }:
                        return False
                    fields["execution_state"] = observation.kind
                    if observation.kind == "tool_completed":
                        fields["terminal_outcome"] = observation.outcome
            if not fields:
                return False
            self._sequence += 1
            self._facts[identity] = replace(fact, sequence=self._sequence, **fields)
            return True

    def snapshot(self, session_id: str, run_id: str) -> tuple[ApprovalFeedback, ...]:
        with self._lock:
            return tuple(
                fact
                for identity, fact in self._facts.items()
                if identity.session_id == session_id and identity.run_id == run_id
            )

    def retire_run(self, run_id: str) -> None:
        with self._lock:
            for identity in tuple(self._facts):
                if identity.run_id == run_id:
                    self._facts.pop(identity)
                    self._counts.pop(identity, None)
                    self._projection_calls.pop(identity, None)
            self._latest = {
                key: value for key, value in self._latest.items() if key[0] != run_id
            }
            self._aliases = {
                key: value for key, value in self._aliases.items() if key[0] != run_id
            }


def format_approval_feedback(feedback: ApprovalFeedback) -> str:
    """Describe observed facts; an accepted choice is not proof a tool ran."""
    if feedback.decision_state in {"timeout", "cancelled", "revoked"}:
        return {
            "timeout": "Approval timed out",
            "cancelled": "Approval cancelled",
            "revoked": "Approval revoked",
        }[feedback.decision_state]
    execution = {
        "dispatch_started": "Starting",
        "backend_started": "Running",
        "tool_completed": "Tool completed",
        "model_wait": "Waiting for model",
    }.get(feedback.execution_state, "")
    if feedback.execution_state == "tool_completed":
        execution = {
            "success": "Tool completed",
            "failure": "Tool failed",
            "blocked": "Tool blocked",
            "timeout": "Tool timed out",
            "cancelled": "Tool stopped",
        }.get(feedback.terminal_outcome, "Tool finished; outcome unknown")
    permitted = (
        feedback.terminal_outcome == "success" or feedback.confirmed_backend_start
    )
    grant = ""
    if feedback.grant_state in {"failed", "not_applied"}:
        grant = (
            "Allowed this call; permission was not remembered"
            if permitted
            else "Permission was not remembered"
        )
    elif feedback.grant_state == "applied":
        grant = {
            "approve_session": "Permission applied: Until Chatbook exits",
            "raw_shell_session": "Permission applied: Until Chatbook exits or Disarm",
            "always_allow": "Permission remembered for this tool",
            "allow_matching": "Permission remembered for these exact inputs",
        }.get(feedback.applied_scope, "")
    state = {
        "pending": "",
        "received": "Decision received",
        "accepted": "Decision accepted",
    }.get(feedback.decision_state, "")
    if feedback.selected_scope == "deny" and feedback.decision_state == "accepted":
        state = "Denied"
    return " · ".join(part for part in (grant, execution or state) if part)
