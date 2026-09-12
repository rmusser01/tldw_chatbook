# ADR-067: Indefinite human approval waits with a pausable per-call clock

Status: Accepted
Date: 2026-08-15
Related Task: [TASK-16789](../tasks/task-16789%20-%20Console-approval-prompts-wait-indefinitely-by-default.md)
Supersedes: the `approval_timeout < max_tool_call_seconds` invariant as stated in
`_DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS`'s comment (Chat/console_chat_controller.py)
and `RunBudget.max_tool_call_seconds`'s comment (Agents/agent_models.py). Those
comments are updated by the task; the historical specs/plans that mention 120s
defaults remain reference-only.

## Decision

Blocking human prompts — the tool approval card, the skill install confirm, and
the skill script confirm — no longer auto-deny by default. They stay armed until
the user answers or the run is stopped/cancelled. The auto-deny ceiling remains
available: a configured positive timeout still fails undecided calls closed
(`"timeout"` verdicts), and `0` (the new default everywhere) means "no deadline".

To make an unbounded human wait safe, the per-tool-call wall clock
(`AgentService._call_with_timeout`, ceiling `RunBudget.max_tool_call_seconds`,
300s at defaults) now **pauses while a human decision is pending for the run**.
A new dependency-light registry, `Agents/human_input_wait.py`, exposes
`use_human_input_wait(run_id)` (thread-safe, set/cleared around each wait loop)
and `human_input_wait_active(run_id)`. `_make_invoke_tool` passes the predicate
into `_call_with_timeout` as `pauses_deadline`; while it polls true, the wrapper
re-arms its deadline so wall-clock time counts only actual tool *execution*.

The superseded invariant (`approval_timeout < max_tool_call_seconds`) existed
because the invoke-path approval wait runs inside the per-call wrapper: an
approval timeout at or above the wrapper ceiling would let the wrapper fire
first, report the call failed, abandon the waiting thread — and a late approval
would then execute the tool for real (double execution). Pausing the clock
removes the race at the root: the wrapper cannot expire while the human wait it
is hosting is still live. Cancellation uses `revoke_approval_rounds_for_run`; the TASK-13215 clarification
below closes late admission and serializes final decisions. A genuinely hung
tool still times out after `seconds` of execution time.

`[mcp] approval_timeout_seconds` keeps its key and semantics for positive
values; `<= 0` now means "wait indefinitely" (previously such values armed a
deadline in the past, i.e. an immediate auto-deny). The card's countdown copy
(`format_approval_deadline`) already renders nothing for 0/None.

## Alternatives considered

- **Just remove the 120s ceiling.** Rejected: reopens the documented
  abandoned-thread double-execution window — the 300s wrapper would fire under
  an infinite wait, and a late approval would act for real after the runtime
  reported failure.
- **Raise both ceilings (approval timeout and `max_tool_call_seconds`).** A
  large finite pair still bounds "go to dinner" badly (hours-long dinners,
  honest mistakes) and weakens the hung-tool guarantee for every tool. Rejected.
- **Move the approval wait outside `_call_with_timeout`.** Architectural
  rework of the provider/gate seams (task-327 put the fallback approval inside
  `invoke` deliberately). The pause achieves the same effect additively.
- **Keep 120s as the default, opt-in to indefinite via config.** Rejected by
  the product decision this ADR records: prompts should not expire on users
  who step away; auto-deny stays available for those who want it.

## Consequences

- A pending approval no longer expires; a session can sit in NEEDS_APPROVAL
  indefinitely and revive when the user returns. Stopping the run and owning-round teardown still deny/clear promptly.
  Navigation retains live rounds for remounting, as clarified by ADR-094.
- Per-call wall-clock is now "execution time, excluding human-decision waits".
  A tool that arms a human wait and then hangs *after* the decision still dies
  at the full ceiling (the deadline re-arms while paused, it does not vanish).
- `UnifiedMCPControlPlaneService.approval_timeout_seconds` (no product callers;
  the controller reads the config key directly) still returns 120.0 as its own
  default; if a consumer ever appears it must adopt the same `<= 0` semantics.
- Tests that rely on auto-deny inject positive seam values (e.g. 0.05) and are
  unaffected by the default flip.

## TASK-13215 clarification: revocation lifetime and decision commitment

The interrupt host retains an in-memory set of revoked run IDs for each swept
kind for its entire lifetime. Revocation remembers the owner even when no round
is armed; the return count still reports only rounds actually swept. Both early
controller registration and host registration consult that fence under the same
lock, so a delayed MCP/local approval or skill-script fallback fails closed
without publishing a card, badge, or retained payload. Primary-only kinds using
`check_revoked=False` retain their existing contract. Empty owners are not swept;
a revocable unowned admission emits one content-free warning and preserves the
legacy direct-call behavior.

The complete tool-approval verdict snapshot commits under the same lock as
revocation, with a fresh revoked-state check. If revocation wins first, every
verdict is unresolved deny. If a complete approval snapshot wins first,
revocation cannot retroactively retract that decision or a tool side effect.
A batch cannot contain partial approvals caused by a sweep between row reads.
Audit and UI callbacks run outside the non-reentrant lock. Payload cleanup
continues to remove exact round IDs, preserving sibling payloads and remounts.

Memory grows with distinct revoked runs per kind. No TTL, eviction, navigation,
session closure, or logical-run completion clears these fences: an abandoned
provider daemon can still reach a fallback after logical completion. Safe
reclamation requires proof that every physical invocation has drained and is
outside this fix; host disposal releases its sets. This restores the existing
cancellation contract without adding a provider dispatch authority boundary.
