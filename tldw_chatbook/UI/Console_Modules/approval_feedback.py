"""Late-bound feedback projection without retaining approval DOM."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from textual.message import Message

from tldw_chatbook.Chat.console_approval_feedback import (
    ApprovalFeedback,
    format_approval_feedback,
)

if TYPE_CHECKING:
    from tldw_chatbook.Agents.approval_observation import (
        ApprovalObservation,
        ApprovalObservationContext,
    )
    from tldw_chatbook.Agents.mcp_tool_provider import MCPPendingCall
    from tldw_chatbook.Chat.approval_presentation import ApprovalBatchView
    from tldw_chatbook.Chat.console_approval_feedback import ApprovalFeedbackStore


class ApprovalFeedbackChanged(Message):
    def __init__(self, session_id: str, run_id: str) -> None:
        super().__init__()
        self.session_id = session_id
        self.run_id = run_id


class ApprovalFeedbackController:
    def __init__(
        self,
        *,
        read_snapshot: Callable[[str, str], tuple[ApprovalFeedback, ...]],
        active_session: Callable[[], str],
        card_identity: Callable[[], tuple[str, int] | None],
        paint_card: Callable[[str], None],
        refresh_status: Callable[[], None],
    ) -> None:
        self._read_snapshot = read_snapshot
        self._active_session = active_session
        self._card_identity = card_identity
        self._paint_card = paint_card
        self._refresh_status = refresh_status

    def refresh(self, session_id: str, run_id: str) -> None:
        if self._active_session() != session_id:
            return
        identity = self._card_identity()
        facts = self._read_snapshot(session_id, run_id)
        matching = tuple(
            fact
            for fact in facts
            if identity == (fact.identity.round_id, fact.identity.revision)
        )
        if matching:
            text = "\n".join(
                dict.fromkeys(format_approval_feedback(fact) for fact in matching)
            )
            if text:
                self._paint_card(text)
        self._refresh_status()


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

    feedback = read_feedback()
    try:
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
