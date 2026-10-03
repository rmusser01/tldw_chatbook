# ADR-211: Console chat destinations and bounded agent starts

Date: 2026-10-02
Status: Accepted after written-spec review on 2026-10-02
Task: [TASK-33802](../tasks/task-33802%20-%20Design-Console-chat-destinations-and-bounded-agent-starts.md)
Spec: [Console chat destinations and bounded agent starts](../../Docs/superpowers/specs/2026-10-02-console-chat-destinations-and-starts-design.md)
Extends: [ADR-150 for agent chat creation](150-agent-chat-fork-and-spawn.md)
Amends for automatic chat starts: [ADR-134](134-fleet-admission-and-automatic-work-budgets.md), [ADR-135](135-fleet-completion-delivery-and-crash-recovery.md)
Follows: [ADR-006](006-provider-aware-generation-settings.md), [ADR-029](029-local-private-data-boundary.md), [ADR-069](069-console-project-instruction-local-state-and-preflight.md), [ADR-079](079-workspace-assistant-defaults.md), [ADR-082](082-console-per-chat-private-scratch-space.md)

## Context

`new_chat` already creates a confirmed workspace chat with an opening draft.
The user approved choosing the source workspace or casual scope, draft or
immediate execution, destination assistant defaults, explicit standing-prompt
overrides, and session-scoped approval remembered separately by mode.

Immediate execution changes the authority and storage boundary. Ordinary sends
mint a user-work allowance; reusing that path for model-origin launches would
replenish budgets. Existing work chains and database guards require each run to
belong to one conversation. Current creation also restores source assistant
identity, uses an assistant-message ID in its `run_id` payload, and consumes saved
handoff metadata on first activation. These mechanisms do not satisfy the approved
destination-default and recovery behavior by themselves.

## Decision

1. Extend the existing primary-agent `new_chat` runtime tool with
   `destination=same_workspace|casual` and `mode=draft|start`, defaulting to
   workspace draft creation. Keep `fork_chat` under its existing contract.
   Resolve the source session's scope explicitly. Casual chats have global
   scope, no workspace membership, and fresh scratch and project control state.
2. Resolve and persist the destination's canonical assistant and generation
   defaults at creation. Explicit `instructions` retain ordinary custom standing
   prompt semantics with a plain/custom identity; disclose the override on the
   approval card. Fresh chats inherit no source identity, prompt, bindings,
   staged inputs, or grants.
3. Scope both approval caches to requesting session incarnation, tool, exact
   target scope/workspace, and mode. Remembered grants permit later bodies in
   that scope under ADR-150's existing policy, but cannot authorize another
   mode/destination or renew a budget. Use trusted run context for lineage,
   separately from the source assistant-message identity.
4. Preserve one-conversation work-chain and run ownership. Add an immutable,
   direct allowance-root reference for automatically created target chains.
   Reservations stay attributable to their own chain, while admission counts
   aggregate across the canonical root and its registered descendants. All
   members share finite counters, pause/review state, and the original deadline.
   Admission, settlement, clock/refusal updates, and recovery mutate the root;
   descendant uncertainty or overage must block sibling admission.
   Only explicit manual work establishes a new allowance. Source run lineage
   belongs to a launch record, not a cross-conversation run-parent link.
5. Introduce a typed durable chat-start attempt and guarded machine-origin
   submission. The attempt owns the exact source run/chain, target conversation,
   session incarnation, request revision, runtime owner, and generation reservation.
   Accept it once after normal preflight; keep its input visibly agent-authored.
   A pre-acceptance human decision or paused preparation leaves a draft and a
   reason rather than a pending automatic continuation.
   Opening text cannot execute composer slash commands or expand `@` references.
   Chat starts and fleet wakes share automatic-primary capacity and manual reserve.
6. Save the conversation and draft before launch preparation. New handoff drafts
   survive activation and persist edits/clears until accepted consumption or
   explicit discard. Version-2 handoffs distinguish this lifecycle from all old
   unversioned handoffs. Use revision fencing for target interaction and cleanup.
   Manual Send withdraws a still-prepared start before ordinary busy guards.
   After acceptance, target execution is independent of source navigation,
   stopping, and closure, within the shared allowance and ordinary stop controls.
7. Treat conversation storage and AgentRunsDB as separate commits. A durable
   attempt fence and saved request/checkpoint must succeed before dispatch.
   The AgentRunsDB acceptance transition is the source-ownership cutoff; report
   `started` only after both durable fences. Bind the conversation receipt and
   atomic draft consumption to the exact attempt ID so callbacks cannot resend.
   Proven pre-acceptance refusal refunds only its uncommitted reservation;
   accepted/uncertain work remains charged. Restart recovery revokes stale owners
   and requires review without automatically resending the opening request.
   Return truthful creation and launch outcomes; a blocked start keeps its draft
   and has no automatic retry queue.

## Alternatives

| Alternative | Reason not chosen |
| --- | --- |
| Separate `start_chat` tool | Duplicates destination, creation, and approval contracts for two modes of the same operation. |
| Submit launches as ordinary manual sends | Grants model-origin work a fresh allowance and user-origin behavior. |
| Attach another conversation directly to the source chain | Widens the existing conversation-ownership boundary throughout run and wake storage. A shared allowance reference preserves that boundary. |
| Create an independent allowance for each new chat | Recursive starts can reset finite limits despite session approval. |
| Keep source assistant identity | Conflicts with the user's destination-default choice and durable reopening. |
| Consume drafts on first activation | Viewing a blocked start would remove its durable recovery input. |
| Queue/replay blocked or interrupted starts automatically | Adds scheduling semantics beyond one immediate attempt and risks replaying uncertain effects. |

## Consequences

AgentRunsDB requires a versioned migration, shared-allowance admission/accounting,
and durable chat-start attempts. Existing root chains keep their identity and
allowance. Wakes in a launched conversation keep that conversation's local chain
while spending the original root's shared budget. Manual submissions continue to
create independent user work without reparenting older survivors.

Creation must resolve settings and identity through the normal destination path,
and new handoff metadata needs a versioned draft lifecycle. Runtime ownership
must survive view detachment; UI completion is an observer. Existing permission,
hook, project-instruction, capture, and physical-worker gates remain required.

The implementation has two bounded slices: budget/attempt groundwork followed by
tool and Console integration. No dependency or general chat-management framework
is introduced. This record changes the relevant policies through a new ADR;
accepted historical ADR bodies remain intact. Written-spec review is complete;
application implementation follows the subsequent implementation plan.

## Current-dev compatibility clarification (2026-10-03)

Preserve current-dev optional provider/model/preset routing on prepared new_chat: resolve from fresh destination generation defaults, retain ADR-147 override enablement/allowlist/final-provider guards and enabled routed preset parameters, capture/disclose before approval, persist the exact resolved snapshot, and retain prepared source/destination/runtime currentness. Never fill routing from source-session settings; chat creation ignores subagent default routing. The conditional schema continues to offer these arguments under its existing gates; the five base arguments above remain unchanged. Preset routing overrides provider/model and configured generation parameters only; it does not copy source persona, prompt, bindings, or grants. Explicit instructions and destination assistant identity follow the creation rules above. This preserves the existing [ADR-147 routing contract](147-agent-provider-routing.md) within the new preparation and approval boundary.
