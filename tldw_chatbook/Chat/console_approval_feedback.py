"""Monotonic session-only approval facts, independent of permission state."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from threading import RLock
from typing import TYPE_CHECKING, Any

from tldw_chatbook.Agents.approval_observation import (
    SCOPES,
    ApprovalObservation,
    ApprovalObservationIdentity,
)

from .approval_presentation import ApprovalBatchView

if TYPE_CHECKING:
    from tldw_chatbook.Agents.approval_observation import ApprovalObservationContext
    from tldw_chatbook.Agents.mcp_tool_provider import MCPPendingCall


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
            elif observation.kind == "grant" and (
                fact.grant_state == "unknown"
                or (
                    observation.outcome == "applied"
                    and fact.grant_state in {"failed", "not_applied"}
                )
            ):
                # A later actual writer can succeed after a prior remembering failure.
                # Failed repeat writes do not undo an already observed applied grant.
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


def format_approval_feedback(
    feedback: ApprovalFeedback, *, tool_status: str = ""
) -> str:
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
    if tool_status and tool_status not in {
        "queued",
        "awaiting_approval",
        "starting",
        "running",
        "live",
    }:
        # Shared group facts cannot establish every member's execution stage.
        # The individual terminal header already supplies its actual outcome.
        execution = ""
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


def bind_round_observations(
    pending: Sequence[MCPPendingCall],
    view: ApprovalBatchView,
    *,
    read_feedback: Callable[[], ApprovalFeedbackStore],
    read_publish: Callable[[], Callable[[ApprovalObservation], None]],
) -> tuple[
    dict[str, ApprovalObservationContext], dict[str, tuple[str, ...]], frozenset[str]
]:
    """Bind display attribution without changing the owner's pending round."""
    from tldw_chatbook.Agents.approval_observation import ApprovalObservationContext

    aliases = {}
    legacy_keys = frozenset()
    observation_contexts = {}
    try:
        legacy_keys = frozenset(
            call.call_id or call.llm_name
            for call in pending
            if not call.call_id or getattr(call, "legacy_observation_key", False)
        )
        feedback = read_feedback()
        feedback.bind_round(view)
        aliases: dict[str, tuple[str, ...]] = {}
        for call in pending:
            key = call.call_id or call.llm_name
            for name in (call.llm_name, call.tool_name):
                aliases[name] = tuple(dict.fromkeys((*aliases.get(name, ()), key)))
        feedback.register_aliases(view.run_id, aliases, legacy_keys=legacy_keys)
        observation_contexts = {
            row.verdict_key: ApprovalObservationContext(
                feedback.context_for_call(view.run_id, row.verdict_key),
                read_publish(),
            )
            for row in view.rows
        }
    except Exception:  # noqa: BLE001 - observers cannot affect authority
        observation_contexts = {}  # Optional display binding cannot stop review.
    return observation_contexts, aliases, legacy_keys


def settle_round_observations(
    round_id: str,
    decisions: dict[str, str],
    contexts: Mapping[str, ApprovalObservationContext],
    outcome: str,
    *,
    read_feedback: Callable[[], ApprovalFeedbackStore],
) -> None:
    """Observe the final owner snapshot after its cancellation/deadline arbitration."""
    from tldw_chatbook.Agents.approval_observation import ApprovalObservation

    try:
        feedback = read_feedback()
        feedback.select_decisions(round_id, decisions)
        for context in contexts.values():
            feedback.publish(ApprovalObservation(context.identity, "settled", outcome))
    except Exception:  # noqa: BLE001, S110 - observers cannot affect authority
        pass  # Final authority is already settled; feedback is optional.


def receive_round_observations(
    round_id: str,
    decisions: dict[str, str],
    round_state: Mapping[str, Any],
    *,
    read_feedback: Callable[[], ApprovalFeedbackStore | None],
    read_notify: Callable[[], Callable[[str, str], None]],
) -> None:
    """Observe receipt only after the host has released its existing Event."""
    from tldw_chatbook.Agents.approval_observation import ApprovalObservation

    try:
        feedback = read_feedback()
        if feedback is not None:
            feedback.select_decisions(round_id, decisions or {})
            for context in round_state.get("observation_contexts", {}).values():
                feedback.publish(
                    ApprovalObservation(context.identity, "received", "received")
                )
            read_notify()(
                round_state.get("session_id", ""), round_state.get("run_id", "")
            )
    except Exception:  # noqa: BLE001, S110 - observers cannot affect authority
        pass  # The worker Event is already released; never fail the receipt.


def publish_approval_observation(
    observation: ApprovalObservation,
    *,
    feedback: ApprovalFeedbackStore,
    project: Callable[[str, str], None] | None,
    notify: Callable[[str, str], None],
) -> None:
    """Project a store fact without allowing a missing display to suppress listeners."""
    feedback.publish(observation)
    if callable(project):
        try:
            project(observation.identity.session_id, observation.identity.run_id)
        except Exception:  # noqa: BLE001, S110 - observers cannot affect authority
            pass  # Display projection cannot suppress other observers.
    notify(observation.identity.session_id, observation.identity.run_id)


def controller_feedback_store(controller: Any) -> ApprovalFeedbackStore:
    """Construct one app-owned fact store on first approval use, never on attach."""
    if controller._approval_feedback is None:
        with controller._approval_feedback_lock:
            if controller._approval_feedback is None:
                from .console_approval_feedback import ApprovalFeedbackStore

                controller._approval_feedback = ApprovalFeedbackStore()
                if controller._agent_bridge is not None:
                    controller._agent_bridge.approval_feedback_store = (
                        controller._approval_feedback
                    )
    return controller._approval_feedback


def set_controller_feedback_store(
    controller: Any, value: ApprovalFeedbackStore | None
) -> None:
    controller._approval_feedback = value
    bridge = getattr(controller, "_agent_bridge", None)
    if bridge is not None:
        bridge.approval_feedback_store = value


def notify_controller_approval_feedback(
    controller: Any, session_id: str, run_id: str
) -> None:
    callback = getattr(controller, "approval_feedback_changed", None)
    if callable(callback):
        try:
            callback(session_id, run_id)
        except Exception:  # noqa: BLE001, S110 - optional display notification
            pass


def publish_controller_approval_observation(
    controller: Any, observation: ApprovalObservation
) -> None:
    from tldw_chatbook.Chat.console_approval_feedback import (
        publish_approval_observation,
    )

    publish_approval_observation(
        observation,
        feedback=controller.approval_feedback,
        project=getattr(
            getattr(controller, "_agent_bridge", None),
            "project_approval_feedback",
            None,
        ),
        notify=controller._notify_approval_feedback,
    )
