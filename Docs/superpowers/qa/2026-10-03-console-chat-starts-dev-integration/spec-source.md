# Console chat destinations and bounded agent starts

Date: 2026-10-02
Status: Approved after written-spec review on 2026-10-02
Task: [TASK-33802](../../../backlog/tasks/task-33802%20-%20Design-Console-chat-destinations-and-bounded-agent-starts.md)
Decision: [ADR-211](../../../backlog/decisions/211-console-chat-destinations-and-bounded-starts.md)

```text
ADR required: yes
ADR path: backlog/decisions/211-console-chat-destinations-and-bounded-starts.md
Reason: Extends the agent-facing creation contract and adds shared automatic budgets, durable machine-origin launch authority, and draft recovery semantics.
```

## Purpose and scope

Extend the primary Console agent's existing `new_chat` runtime tool so it can
create a fresh chat in its own workspace or casual scope, either preparing an
opening draft or immediately starting a background agent turn. Reuse the existing
creation, assistant-default, submission, permission, and automatic-work owners.

This extends [ADR-150 for agent chat creation](../../../backlog/decisions/150-agent-chat-fork-and-spawn.md).
`fork_chat` retains its existing history-copy, destination, draft, and approval
contract. Sub-agent access remains the separate, existing
[TASK-32531](../../../backlog/tasks/task-32531%20-%20Extend-fork_chat-new_chat-tools-to-sub-agents.md).
The new tool provides no arbitrary-workspace targeting, history inheritance,
cross-chat messaging, completion reporting to the source, or launch queue.

## Tool contract

| Argument | Meaning | Default |
| --- | --- | --- |
| `title` | Existing bounded chat title | Existing `New Chat` fallback |
| `opening_prompt` | Initial draft or first agent request | Empty |
| `instructions` | Explicit standing system-prompt override | Destination assistant defaults |
| `destination` | `same_workspace` or `casual` | `same_workspace` |
| `mode` | `draft` or `start` | `draft` |

Retain the existing 120-character title cap and 20,000-character per-field
prompt/instructions caps. Validate argument types, enum values, and payload caps
before approval or mutation. Whitespace-only `instructions` mean no override.
`start` requires a nonempty opening prompt after a whitespace check; invalid
requests create no chat. Omitted new arguments preserve workspace draft behavior.
The new destination and mode arguments belong only to `new_chat`.

| Destination and mode | Outcome |
| --- | --- |
| Same workspace, draft | Saved chat in the source chat's scope, with a composer draft |
| Casual, draft | Saved global chat, with a composer draft |
| Same workspace, start | Saved chat in the source chat's scope, followed by one background start attempt |
| Casual, start | Saved global chat, followed by one background start attempt |

Both modes keep the user's current chat, active workspace, composer, and focus
intact. The tool returns at creation or first-turn acceptance, not when the new
agent finishes. After acceptance, the new chat runs independently through normal
Console controls. Closing or stopping the source chat then does not cancel it.

## Destination, assistant, and authority

Resolve `same_workspace` from the requesting session, never the foreground tab
or registry's current workspace. A globally scoped source remains global. A
source in the built-in Default workspace keeps that existing workspace scope.
`casual` is explicitly `scope_type=global`, `workspace_id=NULL`, with the existing
Console global sentinel used only in session state. It creates no workspace or
workspace membership and appears in Chats.

Capture the resolved destination and canonical creation settings before approval.
Use the same assistant resolver and generation defaults as ordinary creation in
that destination. Persist the selected assistant identity, prompt, and supported
durable generation snapshot together with the fresh conversation, and restore
that selection consistently. Later default edits follow ordinary session
independence; they do not silently change an already-created chat. Unavailable
workspace defaults use the existing plain fallback and visible notice.

Explicit nonblank `instructions` retain their standing-prompt semantics. Apply
the ordinary custom-prompt creation rule: the destination's generation defaults
remain the base, while the assistant is plain/custom rather than incorrectly
labeled as the workspace Persona. The approval preview names this override and
shows its complete body. Omitted instructions use the destination assistant.
Source persona, character, memory mode, custom settings, and system prompt are
not inputs to fresh-chat creation.

Each chat gets its own scratch and normal fresh project-instruction control
state, following [ADR-082](../../../backlog/decisions/082-console-per-chat-private-scratch-space.md)
and [ADR-069](../../../backlog/decisions/069-console-project-instruction-local-state-and-preflight.md).
Copy no scratch files, selected folder bindings, automatic instruction bodies,
attachments, staged evidence, one-shot prefill, or approval grants. Workspace
membership alone does not select a filesystem binding. Resolve current
destination tool permissions and all existing narrowing rules at run admission.
Recheck destination availability and source execution ownership before mutation
and again before launch; a removed or archived destination cannot be retargeted.

## Approval

Reuse the existing Allow / Allow for this session / Deny card and parked-round
behavior. Show destination, title, assistant/model selection, draft versus
immediate start, the complete opening prompt, and any instructions override.
Describe a remembered grant as allowing later requests in that same mode and
destination, including their supplied prompt bodies. Preserve ADR-150's accepted
session-remember policy; later bodies do not receive a fresh card under a matching
grant, and the resulting system prompt stays inspectable in the created chat.

For `new_chat`, both the bridge's run-local memo and the controller's session
grant use the same scope: requesting session incarnation, tool, resolved target
scope/workspace identity, and mode. Draft permission never authorizes start;
workspace permission never authorizes casual creation. Fresh chats, session close,
and app restart begin without these grants. A valid remembered grant permits only
that action; it grants no tool access or additional budget.

Every displayed round retains its exact request ID and cancellation binding.
An unanswered required confirmation, unavailable approval surface, cancellation,
or stale decision refuses creation. Previously valid session grants need no new
card, but their source-session/run ownership must still be live. Keep the existing
denial ceiling per requesting run and tool across modes and destinations so
changing arguments cannot evade a refusal.

Read the true agent run from trusted runtime context at each invocation. The
existing bridge payload calls an assistant-message ID `run_id`; keep message and
run identities distinct, and never use that legacy field as budget authority.
Neither source lineage nor launch authority is accepted from model arguments.

## Creation and draft custody

Create the conversation durably with its destination, assistant settings, and
opening draft, then restore a non-activated session through the shared runtime.
View hooks observe completion and refresh the chat list; they do not own launch
authority. Navigation or a detached Console screen cannot redirect accepted work.

For new `new_chat` handoffs, first activation does not clear the saved draft.
Persist the current handoff draft while it remains pending, including user edits
and explicit clearing, with revision checks so an old write cannot resurrect
earlier text. Consume it on successful submission acceptance or explicit discard.
An empty draft must not rehydrate the original prompt. Write new `new_chat`
handoffs with metadata version 2 and an explicit pending/consumed state and draft
revision. Existing unversioned handoffs, including old `new_chat` and `fork_chat`
records, retain their version-1 activation-clearing contract. Unknown versions
require review and never authorize a launch.

An immediate start submits the exact approved opening prompt against the new
session's captured draft/context revision. User edits, manual submission, closure,
or another accepted turn before acceptance invalidate that start. Preserve the
user's current draft and return a useful reason. Clear only the consumed revision;
an edit made later must survive completion callbacks. Manual Send must be
available while only an unaccepted chat start is preparing: arbitrate that intent
before the screen, queue dispatcher, and controller busy guards, withdraw the
prepared attempt, and let the manual send proceed through its normal validation.
After the durable acceptance cutoff, ordinary busy/stop controls apply. Losing
preparation callbacks cannot overwrite the manual send's draft or run state.

Conversation storage and AgentRunsDB are separate commits. Do not claim one
transaction across them. Save the chat before preparing a launch. If preparation
fails or the process ends in that gap, the conversation remains a recoverable
draft and receives no automatic launch on restart.

## Background start and shared budget

Add a distinct machine-origin submission, `agent_chat_start`, to the existing
controller pipeline. Only an app-owned, exact-target launch authorization can
invoke it. Persist the opening request in the normal request-role transcript row
with `origin=agent_chat_start` and source provenance, render it as an agent
handoff, and exclude it from trusted user-profile mutation authority. Provider
adapters encode it in their normal user-role wire format; this grants no user authority.
Treat slash-command and `@` prefixes as literal request text, and borrow no
foreground staged inputs. Apply the destination's normal readiness, trace,
instruction preflight, hook consent, permission, and configured retrieval checks.
If the destination agent runtime is disabled or unsupported, keep the draft and
report that an agent start was unavailable. Before acceptance, any preflight
requiring a new human decision or entering a paused preparation returns
`not_started` with the required action and retains the draft. Do not leave the
source tool waiting for a second target-chat dialog or schedule a continuation.
Tool approvals arising after acceptance use the target's ordinary approval flow.

Audit every origin-dependent submission branch explicitly. This origin has a
durable request row and destination-configured capture/retrieval, while it skips
composer command parsing, reference expansion, staged inputs, prompt history,
and trusted user-profile authority. Neither the manual-origin defaults nor all
of the wake-origin shortcuts implement that combination.

Retain the current one-conversation ownership of each work chain and each run.
Add an immutable `allowance_root_chain_id` reference for automatic descendant
chains. A null reference denotes the chain's own allowance. A nonnull reference
must point directly to a root whose reference is null; enforce this on persisted
writes so cycles and later reparenting cannot acquire authority. Existing and
manual chains use their own allowance; an admitted chat
start creates a target-conversation chain referring directly to the source's
canonical allowance root. A descendant cannot point to another descendant,
change its reference, or mint an allowance. Run parent links remain within their
conversation; the launch attempt records the source run separately.

Keep reservations attributed to their own chain and aggregate allowance counters
across the root and its registered descendants. All allowance mutations resolve
the canonical root, including admission, start/deadline stamps, refusal rollback,
pause, token settlement, and startup recovery. Keep reservation and run ownership
on their local chains. Unknown usage or an overage in any member updates the root
so sibling admissions cannot continue through a healthy-looking local chain.
All members observe the root's snapshotted limits, pause/review status, elapsed
deadline, and uncertain accounting. Root time checks remain monotonic across
members and retain the existing protection against backward clock changes.
Generation, provider-call, token, child-launch, and subsequent wake charges share
that balance. Each accepted chat start consumes one automatic generation;
draft-only creation consumes none. No approval, navigation, setting change, or
restart replenishes it. A later explicit manual Send creates ordinary new user
work, without reassigning earlier attempts or survivors.

Use a typed durable chat-start attempt in AgentRunsDB, distinct from survivor
wake claims. Its immutable identity includes source run/chain, target
conversation/session incarnation, current runtime owner, approved request
revision/fingerprint, target chain, and generation reservation. Store bodies in
private conversation storage, not budget tables or generic diagnostics. The
controller receives an opaque authorization; the model cannot fabricate one.
Extend automatic-work context checking to accepted chat-start attempts without
fabricating a completed child or a wake delivery.

Start attempts and fleet wakes share the existing automatic-primary admission
limit and manual capacity reserve. They must not each receive separate copies of
those slots. Apply existing physical worker ownership and global child/tool limits
from [ADR-134](../../../backlog/decisions/134-fleet-admission-and-automatic-work-budgets.md)
and [ADR-135](../../../backlog/decisions/135-fleet-completion-delivery-and-crash-recovery.md).
Capacity refusal leaves a draft; this feature adds no waiting/retry timer.

## Acceptance, outcomes, and recovery

Use this launch lifecycle:

1. Save the new conversation and draft; prepare the exact-target attempt and
   reserve a generation from the shared allowance.
2. Complete preflight, then recheck source ownership, target incarnation/revision,
   current policies, budget, and shared capacity at the acceptance boundary.
3. Commit acceptance of the prepared attempt once in AgentRunsDB. This atomic
   transition is the ownership cutoff: source cancellation and a competing
   manual send can withdraw only a still-prepared attempt. Once the transition
   wins, source stopping cannot revoke the target's ownership. Separately
   persist the machine request and dispatch checkpoint in conversation storage,
   consuming only its draft revision. Dispatch and notify the source tool of
   confirmed Console acceptance only after the required durable fences succeed;
   continue the target turn under ordinary runtime ownership.
4. Settle usage and the attempt on completion. Release physical capacity only
   after the owned execution has settled, including delayed worker cleanup.

Duplicate callbacks cannot consume an attempt or submit its opening request twice.
A proven pre-acceptance refusal releases only its uncommitted reservation.
Accepted or uncertain work retains its charge. If a durable fence or transcript
write fails during acceptance, dispatch nothing further and require review;
neither database implies the other commit succeeded. Target Stop or closure after
the ownership cutoff uses ordinary target cancellation and retains the accepted
charge. `started` is reported only after both durable stores confirm acceptance,
not merely after the AgentRunsDB transition. Restart recovery fences old
owners and never automatically resends an interrupted opening request. Existing
Retry/recovery controls require an explicit user action for another dispatch.

Recovery distinguishes these persisted states without guessing across stores:

| AgentRunsDB / conversation checkpoint | Recovery |
| --- | --- |
| No attempt; saved draft | Keep the draft; perform no automatic start |
| Prepared under an old owner | Fence the owner, retain unresolved charges for review, and preserve the current draft |
| Accepted; request/checkpoint missing | Require review, retain the charge and available draft, and dispatch nothing |
| Accepted; saved request/checkpoint present | Use ordinary interrupted-turn recovery; do not resend automatically or rehydrate consumed text |

The saved request and checkpoint carry the exact attempt ID. Their durable commit
consumes the matching draft revision atomically in conversation storage. Repeated
acceptance callbacks read that receipt rather than append another request. A
missing or mismatched receipt never authorizes dispatch.

On known durable creation, return a successful creation result containing the
conversation ID, destination, mode, and one truthful `launch_status`:

| Status | Meaning |
| --- | --- |
| `draft` | Draft-mode creation completed |
| `not_started` | Creation succeeded; pre-acceptance start was refused, with a reason |
| `started` | The first turn reached confirmed Console acceptance; completion is pending |
| `review_required` | Launch acceptance or execution became uncertain and needs review |

Creation success is not converted into a generic failure that encourages the
model to create another chat. The tool description tells it not to repeat creation
to recover a blocked or uncertain start. Pre-creation validation, denial, and
known persistence refusal return an ordinary tool error. After acceptance, provider
errors remain in the target's saved transcript and normal recovery controls.

The approval card, toast, tool result, chat row, and run activity distinguish
created drafts, starts, and blocked/review outcomes. Reuse existing status and
authority classes under the [design language](../../../backlog/docs/design-language.md).
Opening prompts and system-prompt bodies remain out of metadata-only badges and
generic diagnostics. No new settings surface, visual theme, or keybinding is needed.

## Implementation boundaries and verification

The budget/attempt foundation is one implementation slice; tool/schema, creation,
draft custody, approval, and Console integration form the next. Create concrete
atomic implementation tasks only after written-spec approval. AgentRunsDB needs
a versioned migration for allowance references and launch attempts. The
conversation database also needs a migration: existing dispatch checkpoints
accept only manual/queued origins, and the new machine-origin receipt must retain
its identity through recovery. Select both next versions from the actual
implementation branch and preserve existing rows, constraints, and local
ownership. Existing roots retain their current limits and ownership. Conversation
settings and handoff revisions use the existing persistence owners and local
metadata contracts.

Focused automated evidence must cover:

- All four destination/mode combinations, omitted arguments, strict validation,
  and unchanged fork behavior; real SQLite verifies global scope and membership.
- Destination assistant/settings consistency across creation and restart, plain
  fallback, custom instructions, and separation from source persona and bindings.
- Both approval caches, mode/destination separation, stale clicks, denial ceilings,
  remembered invocations, true run provenance, close, and restart.
- Shared counters/deadlines across recursive chat starts, children, and wakes;
  concurrent reservation bounds, no new allowance, migration, and old-root behavior;
  late descendant uncertainty/overage pauses sibling admission through the root.
- Exact-target submissions during navigation, target edits/manual sends/closure,
  source cancellation on each side of the cutoff, manual-send priority while
  preparing, combined background capacity, and ownership replacement.
- Draft edit/clear durability, blocked starts, duplicate callbacks, write failures
  between the two databases, paused preflight without a second dialog, and
  crash/restart at each acceptance boundary.
- Mounted approval-card and runtime-hook wiring; the runtime ownership slot-set
  check catches hooks that otherwise appear wired but do nothing.

Before implementation closure, exercise the real Console with a real configured
provider: create workspace and casual drafts, start each destination, keep focus
and the original composer intact, reopen after restart, and reproduce a blocked
start. Run targeted suites and relevant lint/format/governance checks; obtain user
opt-in before a full test sweep. This design document does not claim implementation
or runtime verification.
