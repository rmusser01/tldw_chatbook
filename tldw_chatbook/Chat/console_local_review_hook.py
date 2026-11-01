"""One-run approval review hook for local tools."""

from __future__ import annotations

from typing import Any, Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .console_chat_controller import (
        LocalToolProvider,
        MCPPendingCall,
        ToolCall,
        ToolReviewValue,
    )


def build_local_review_hook(
    provider: "LocalToolProvider",
    request_approvals: Callable[[list["MCPPendingCall"]], dict[str, str]],
    *,
    read_global_ToolReviewValue: Callable[[], Any],
    read_global_USER_DENIED_REFUSAL: Callable[[], Any],
    read_global__APPROVAL_SCOPE_RANK: Callable[[], Any],
    read_global__APPROVING_DECISIONS: Callable[[], Any],
    read_global__review_decision: Callable[[], Any],
    read_global__sibling_approval_refusals: Callable[[], Any],
    read_global__stamp_answer_provenance: Callable[[], Any],
    read_global_approval_was_unanswered: Callable[[], Any],
) -> Callable[[list["ToolCall"], str], dict[str, ToolReviewValue]]:
    """Build this run's review_tool_calls hook for the local provider.

    Identical discipline to build_mcp_review_hook (see its docstring for
    the full rationale -- every binding point applies unchanged here):
    clear-first stamps at entry (I3: a raising approval round trip must
    never leave a stale prior-turn stamp live for the fail-open runtime
    to hand to `invoke()`), and exactly ONE approval round trip per batch.
    Name-keyed stamps keep the widest approved scope because the provider
    gate is tool-scoped; per-call refusal verdicts stop only the denied
    sibling before dispatch. Calls the provider doesn't own resolve
    `None` from `pending_gate_for` and never enter the batch.

    Args:
        provider: This run's already-composed `LocalToolProvider` (built
            by `_compose_local_provider` on the main loop before the
            run's worker thread starts).
        request_approvals: The bound `ConsoleChatController.
            request_mcp_approvals` method for THIS run -- the same
            approval-card bridge the MCP hook uses; it consumes
            `MCPPendingCall` payloads regardless of origin.

    Returns:
        A `review_tool_calls`-shaped callable suitable for `LoopDeps`/
        `AgentService(review_tool_calls=...)`.
    """

    def review_tool_calls(
        calls: list["ToolCall"], run_id: str
    ) -> dict[str, read_global_ToolReviewValue()]:
        # I3: clear THIS turn's stamps FIRST -- see build_mcp_review_hook.
        # PR2a Task 5: scoped to `run_id`, so the clear cannot reach a
        # concurrent sibling run's live verdicts.
        provider.apply_batch_decisions(run_id, {})
        pending: list["MCPPendingCall"] = []
        for call in calls:
            gate = provider.pending_gate_for(
                call.name,
                call.args,
                str(getattr(call, "call_id", "") or ""),
                # Qodo review #10: the local owner must receive the call's
                # advisory rationale like the MCP and builtin owners do, or
                # every local approval row renders without model context.
                rationale=str(getattr(call, "rationale", "") or ""),
                run_id=run_id,
            )
            if gate is not None:
                pending.append(gate)
        if not pending:
            return {}
        decisions = request_approvals(pending)

        def _decision_for(row: "MCPPendingCall") -> str | None:
            key = str(getattr(row, "call_id", "") or "")
            if key and key in decisions:
                return decisions[key]
            return decisions.get(row.llm_name)

        ordinary_pending = [
            row
            for row in pending
            if not (
                row.llm_name == "fs_search_index"
                and row.options == ("approve_once", "deny")
            )
        ]
        approvals: dict[str, str] = {}
        denied: set[str] = set()
        for row in ordinary_pending:
            decision = _decision_for(row)
            if decision is None:
                continue
            if decision == "deny":
                denied.add(row.llm_name)
                continue
            current = approvals.get(row.llm_name)
            if current is None or read_global__APPROVAL_SCOPE_RANK().get(
                decision, 0
            ) > read_global__APPROVAL_SCOPE_RANK().get(current, 0):
                approvals[row.llm_name] = decision
        stamps = dict(approvals)
        for name in denied:
            stamps.setdefault(name, "deny")
        provider.apply_batch_decisions(
            run_id,
            read_global__stamp_answer_provenance()(stamps, ordinary_pending, decisions),
        )
        provider.apply_index_read_decisions(run_id, pending, decisions)
        reviewed_call_ids = {row.call_id for row in pending if row.call_id}
        provider.apply_promotion_decisions(
            run_id,
            [call for call in calls if call.call_id in reviewed_call_ids],
            decisions,
        )

        # task-32280 fix round (Critical): mirrors the MCP hook's own
        # record_user_denial call a few hundred lines up. `run_agent_loop`
        # turns any non-"proceed" verdict straight into the call's result
        # and skips dispatch entirely, so `LocalToolProvider.invoke_detailed`
        # -- the only thing that otherwise records a local refusal -- never
        # runs for a hook-level denied call. Record at the point the denial
        # becomes final, through the provider's own audit seam.
        verdicts: dict[str, read_global_ToolReviewValue()] = {
            row.llm_name: "proceed" for row in pending
        }
        for row in pending:
            if _decision_for(row) != "deny":
                continue
            # R23: as in the MCP hook -- a deny the user never chose (Stop
            # mid-card) is already logged as `denied-unresolved`; only the
            # REFUSAL below applies to it, not a second audit row.
            if not read_global_approval_was_unanswered()(row, decisions):
                provider.record_user_denial(row.llm_name)
            key = str(getattr(row, "call_id", "") or "") or row.llm_name
            verdicts[key] = read_global__review_decision()(
                row,
                decisions,
                read_global_USER_DENIED_REFUSAL().format(name=row.llm_name),
            )
        verdicts.update(
            read_global__sibling_approval_refusals()(
                pending,
                _decision_for,
                decisions,
                lambda _row: read_global__APPROVING_DECISIONS(),
                lambda row, timed_out: provider.record_hook_refusal(
                    row.llm_name, timed_out=timed_out
                ),
            )
        )
        # Preserve settled name-wide refusal fallback before adding facts.
        verdicts.update(
            {
                row.call_id or row.llm_name: read_global__review_decision()(
                    row, decisions, "proceed"
                )
                for row in pending
                if verdicts.get(row.call_id or row.llm_name, verdicts[row.llm_name])
                == "proceed"
            }
        )
        return verdicts

    return review_tool_calls
