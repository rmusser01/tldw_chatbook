# Expanded Console hook runtime

Date: 2026-09-15
Status: Draft for written-spec review; expanded capability scope approved.
Task: [TASK-32645](../../../backlog/tasks/task-32645%20-%20Design-managed-plugins-and-expanded-hook-runtime.md)
Decision: [ADR-163](../../../backlog/decisions/163-expanded-console-hook-runtime.md)
Companion: [Managed plugins](2026-09-15-managed-plugins-design.md)

## 1. Purpose and preserved authority

Extend Chatbook's shared Console hook engine so users and managed plugins can
observe lifecycle events, refuse operations, contribute context, propose tool
inputs, constrain child agents and request bounded continuations.

The hook engine consumes normalized owned definitions. It does not discover
repositories, install packages, manage marketplaces or grant tool permissions.
Plugin admission supplies an immutable definition set and a current-authority
validator. User-configured hooks remain supported independently of plugins.

This extends [ADR-148](../../../backlog/decisions/148-console-run-hooks.md).
Its permission floor remains: hooks can narrow or propose, never bypass
Chatbook's permission store, tool registration, workspace bindings or existing
policy floors. External processes run with the user's privileges. This engine
is not an OS sandbox.

New capabilities:

- PostToolUseFailure, SubagentStart, SessionStart, SessionEnd,
  PreCompact, PostCompact and Interrupt.
- Structured matchers and bounded attributable context.
- Revalidated tool-input transformations.
- MCP-backed hooks through ordinary tool authorization.
- Scheduler-owned Stop continuations with finite chain budgets.

All are in scope. Editor Tab events, prompt/model-evaluated hook handlers,
arbitrary Python callback imports and permission-bypass results are excluded.

## 2. Definition and result contracts

### 2.1 Configuration versions

Existing user config [hooks] + [[hooks.hook]] remains the legacy contract:
six events, argv commands, current matching, first-deny behavior and current
UserPromptSubmit output semantics. Do not reinterpret legacy stdout as v2 JSON.
The existing master enabled switch applies to both versions.

New user declarations use [[hooks.handler]], whose fields match a v2 handler.
Native plugin files use this closed versioned envelope:

~~~json
{
  "version": 2,
  "hooks": [
    {
      "id": "protect-writes",
      "event": "PreToolUse",
      "type": "command",
      "argv": ["python3", "${PLUGIN_ROOT}/hooks/protect_writes.py"],
      "effects": ["deny"],
      "match": {"operation": ["file.write", "file.edit"]},
      "timeout_seconds": 10
    }
  ]
}
~~~

The sample assumes the referenced package script exists. It is a definition,
not an instruction to run the script while inspecting this document.

Common fields:

| Field | Contract |
| --- | --- |
| id | Unique stable local identifier, 1–128 ASCII letters/digits/underscore/hyphen. |
| event | One event from section 3; case-sensitive after dialect normalization. |
| type | command or mcp_tool. |
| effects | Explicit subset permitted for the event; empty means observation only. |
| match | Optional structured matcher; absent matches every occurrence of that event. |
| timeout_seconds | Finite positive execution timeout within host limits. |
| input | MCP-only JSON argument template, default empty object. |

Allowed effect names are deny, context, updated_input, child_limits,
continuation and stop_continuations. Command-specific fields are argv, env and
cwd; MCP-specific fields are server, tool and input. Type-inappropriate fields
are invalid. id, event, type and effects are required. match and
timeout_seconds are optional; input is optional for MCP and forbidden for
commands.

Command handlers require a nonempty argv array. Optional env contains literal
or declared variable values; cwd is a contained package path for plugin hooks,
or an explicit user-selected path for user hooks. Native plugin default cwd is
the package root. Vendor workspace-cwd behavior requires a reviewed adaptation
to the run's existing scratch/selected binding; hooks do not add bindings.
No implicit shell is introduced. A user-authored explicit shell executable
is ordinary privileged command execution and must be presented as such.

MCP handlers require a server component/connection reference and tool reference.
Resolve to exact owned tool identity/definition at admission; require a connected
eligible server. Do not start a disconnected server automatically to satisfy a
hook. Authentication/setup happens through explicit connection flow.

Reject duplicate IDs, unknown fields, unsupported versions, illegal effect/event
combinations and malformed required handlers. Invalid required guards block
their dependent capabilities. An observation-only hook failure is diagnostic,
not authority to disable an unrelated component.

### 2.2 Event payload

Use a versioned JSON envelope with:

- event_id, event, protocol_version and timestamp.
- runtime_session_id, run_id, parent_run_id, turn_id and workspace identity
  where applicable.
- initiator (manual, scheduled, continuation, child or host_cleanup).
- owner installation/component ID for plugin handlers; host-assigned origin.
- Event-specific structured data and a causal chain ID/depth.

Tool data includes original and final/candidate arguments, stable tool identity,
provider, canonical operation, definition hash, dispatch/result status and
reason. Never infer canonical operation from a vendor's display name.
PreToolUse sees exact arguments; oversized input is refused whole, not
truncated into a different decision. Unknown properties remain data only.

Command stdin receives only the documented event fields. MCP handlers receive
only their explicit input mapping, not the entire event by default. A template
value consisting solely of ${field.path} preserves its JSON type; embedded
references render bounded scalar text. Missing paths and non-scalar embedded
values fail validation. Expansion is one pass with no evaluation, environment
lookup or recursive command substitution. Vendor spellings are adapter-owned.

### 2.3 V2 results

Command stdout is one UTF-8 JSON object. Empty stdout on success means no
effect. stderr is diagnostic only. Host validation accepts only:

~~~json
{
  "version": 2,
  "decision": "pass",
  "context": [{"text": "Use the repository review checklist.", "lifetime": "turn"}]
}
~~~

Additional result fields, when declared in effects and legal for the event:

- updated_input: complete replacement tool-argument object, not a partial patch.
- child_limits: explicit tool-ID subset and/or lower existing child budget caps.
- continuation: one bounded message requesting the next scheduler turn.
- stop_continuations: true vetoes hook-created continuation for this settlement.
- reason: bounded human-readable explanation.

decision is pass or deny; there is no allow-bypass value. Context origin,
authority, budget ownership and run IDs are assigned by Chatbook, never by
hook output. Output effects not declared in the reviewed definition are
invalid; they do not silently become new capabilities.

Unknown fields or malformed results follow the event's failure policy.
Nonzero exit, output overflow and invalid JSON are errors; exit code 2 is
not a new v2 protocol shortcut. Legacy/vendor exit meanings are translated
by the selected adapter before v2 result validation.

## 3. Event timing and effects

| Event | Exact boundary | Allowed v2 effects | Failure behavior |
| --- | --- | --- | --- |
| SessionStart | Before first run of a live hook-runtime session | context with runtime lifetime; deny initialization | Required initialization blocks dependent capability admission; optional context failure is visible. |
| UserPromptSubmit | Explicit user submission after base input validation, before context preparation/admission | deny; turn context | Explicit denial blocks; other failures remain fail-open as in ADR-148. Required material cannot be silently omitted from an admitted dependent capability. |
| PreToolUse | Candidate prepared, before normal approval and dispatch | updated_input; deny; turn context | Fail closed for selected transformation/guard failures. |
| ApprovalRequested | A normal permission request is created | observation only | Diagnostic only; cannot approve, modify or answer for the user. |
| PostToolUse | Dispatched call settled, including ordinary execution-error results | turn context; observation | Cannot rewrite the result or undo effects; failure is diagnostic. |
| PostToolUseFailure | After PostToolUse when a dispatched call reports execution failure | turn context; observation | Diagnostic only; denied/not-dispatched calls do not emit it. |
| SubagentStart | Child draft has inherited restrictions, before child admission | deny; child_limits; child-turn context | Fail closed for selected guards/restrictions. |
| SubagentStop | Child result has settled | observation; context for the parent's next model step if still active | No restart or child-output rewriting; otherwise discard late context visibly. |
| PreCompact | Actual compaction candidate prepared, before compactor runs | context supplied to compaction input | Required context failure aborts compaction; never silently drops history. |
| PostCompact | New compacted context successfully committed | turn context for subsequent model input | Diagnostic only; no rollback or editing of committed summary. |
| Stop | One accepted root turn settles normally, before scheduler decides next action | continuation; stop_continuations; observation | Failure ends hook continuation; no retry loop. |
| Interrupt | After user cancellation seals admission, during bounded cleanup | observation only | Cannot veto, extend or reverse cancellation. |
| SessionEnd | Graceful disposal/replacement of a live hook-runtime session | observation only | Cannot veto disposal; bounded best effort. |

PostToolUse retains existing behavior for unsuccessful dispatched tool results;
PostToolUseFailure adds the explicit failure channel. Native authors subscribing
to both intentionally receive both. An adapter whose source post-event is
success-only filters accordingly. Denied calls and calls cancelled before
dispatch produce neither event. Post-dispatch cancellation emits failure only
when an execution error is known; uncertain remote completion is recorded as
uncertain, never fabricated as a server failure.

UserPromptSubmit remains manual-origin only. Scheduled work and Stop
continuations do not masquerade as fresh user submission. They still pass
normal run admission and tool guards. Subagents do not emit root Stop or
root session events; they use child lifecycle events.

### 3.1 Live session boundaries

A hook-runtime session is a host-owned live execution context for a specific
conversation, workspace binding and immutable active hook set. It starts lazily
when the first run is admitted and ends on explicit conversation runtime
closure, process shutdown or an idle context/hook-set replacement.

Changing tab focus does not create or end it. Durable conversation history is
not itself a live session. Reopening archived history does not run hooks until
new execution is admitted. A new process creates a fresh live session after
recovery; there is no replay of missed SessionEnd events.

Enabling/updating a plugin applies to the next run. At the idle boundary,
replace the hook-runtime session and deliver SessionStart for the new set,
with reason configuration_changed. Initial creation uses startup; a new
runtime for resumed execution uses resume. These are explicit Chatbook
semantics; vendor sessions can have different lifetimes.

During replacement, still-authorized handlers may receive the old SessionEnd.
Revoked/removed plugin handlers never run as cleanup. Required initialization
must complete before the dependent component is advertised; do not fake
initialization by marking an already-open runtime as started.

Compaction is the real context operation, not a tab action or every token
estimate. Runtime context contributions are retained as separately owned
blocks and reassembled within budgets after compaction; do not duplicate them
into both the summary and the active context lane.

## 4. Matching and deterministic effects

Structured match keys are tool_id, provider, operation and reason. Values are
nonempty arrays of exact values or bounded case-sensitive glob patterns.
Keys combine with AND; values within a key combine with OR. Empty/unknown
keys or unavailable fields are invalid for that event. No shell expressions,
LLM matching or unbounded regular expressions.

The runtime supplies canonical operations for actual tool providers, such as
file.read, file.write, file.edit, shell.execute and mcp.call. Missing operation
classification is not a license to guess from the tool name. Guard mappings
that need unavailable classification remain unsupported.

Stable order for effectful hooks is user configuration order followed by
installation_id and local hook ID. Observation-only handlers can execute
concurrently in the separate bounded notification pool. Their completion order
does not affect decisions or context order.

For PreToolUse:

1. Apply existing unconditional host restrictions and schema validation.
2. Run declared transformers sequentially against the current candidate.
   Revalidate each replacement; any denial/invalid result blocks the call.
3. Freeze final arguments. Run all matching deny-only guards, including
   legacy guards, against those final arguments. A denial dominates all
   context or pass results. No further transformation runs afterward.
4. Compute the final definition/argument identity and run the full existing
   permission/profile/workspace review chain.
5. Immediately before dispatch, recheck that approval, arguments, definitions,
   policy and owner generations still match.

Legacy guard-only batches retain their documented concurrent first-deny
behavior. Without new transformers their input is unchanged. Within a mixed
pipeline, guards inspect the final candidate. Repeated transformation cycles
are forbidden; there is one finite pass. A transformer may deny on original
input, and original arguments remain available for attributed review.

If a preauthorized invocation's arguments change, its prior authorization
must be re-evaluated under the same host authority contract. Interactive review
is required when that contract cannot cover the new exact call; do not route
every unchanged preauthorized call through a new prompt.

Subagent limits intersect inherited tool sets and lower budgets. Empty
tool subsets mean no tools. They cannot change provider, workspace bindings,
persona authority, approval mode or raise any cap.

## 5. Context ownership

Context is untrusted attributed instruction material below host policy.
Allowed lifetimes:

- turn: next eligible model input(s) in the owning accepted turn.
- runtime: only SessionStart contributions, scoped to that live hook session.

Child-start context belongs to the child's first turn. A settled child's
contribution to its active parent is queued for the next input; never mutate
a prompt already in flight. Post-event context that arrives after its owner
settles is discarded with a bounded diagnostic, not applied to another run.

The host stages effects, then commits them at the event boundary only if the
event/owner is still current and no controlling denial occurred. Parallel
observations cannot append free-floating context after revocation.
Reject whole oversized blocks; never truncate decision JSON or required
instruction text. Automatic context bodies stay out of metadata surfaces and
ordinary logs; explicit review uses the existing context/privacy conventions.

## 6. MCP-backed hooks and recursion

MCP hook invocations use the same tool identity, schema validation, profile
floors, approvals, cancellation and audit as other invocations. Being a hook
does not confer tool access. The parent event holds no permission-store lock
or worker slot while awaiting user approval.

Propagate a host-owned causal chain containing visited hook IDs and exact
tool definition IDs. Guard recursion rules:

- A handler already in the causal chain cannot invoke itself again.
- Required guards are never silently skipped to break a cycle.
- If a required guard needs the very tool it is currently guarding, fail
  the invocation with a dependency-cycle diagnostic.
- At the recursion limit, fail the required parent guard; drop optional
  notifications with a diagnostic.

The hook invocation itself still passes host restrictions and other eligible
guards. Do not implement recursion prevention by exempting all hook-origin
calls from PreToolUse or the permission store. Matchers should allow authors
to target their intended tools instead of accidentally guarding the scanner.

An ApprovalRequested MCP observer must not open nested interactive approval
while handling an existing permission request. If ordinary permission would
require a new prompt, skip that optional observer and report the unresolved
permission; never auto-approve it. This event is observation-only.

Interrupt/SessionEnd MCP notifications likewise cannot prompt or connect a
server during shutdown. They execute only if existing ordinary authority
permits immediate dispatch on an already-connected server, within the cleanup
deadline. Otherwise report an omitted notification. This is an explicit
adaptation for teardown, not a permission bypass.

## 7. Stop continuations

Stop observes normal root-turn settlement. A continuation is a proposal to the
existing scheduler with the same conversation/workspace and a stable parent
turn/event identity. It is never a direct recursive model call.

Combine proposals in stable order into at most one next turn. Preserve their
separate origins. If combined input exceeds its budget, reject the continuation
whole. Any stop_continuations result vetoes hook-created continuation at that
settlement. A user stop, revoked owner, pending plugin update drain, exhausted
run budget or closed session also prevents admission.

The next turn uses normal scheduling capacity, provider budgets, current
authority and tool review. It carries initiator=continuation and does not fire
UserPromptSubmit. Foreground queued user work takes priority. If such work is
present at settlement, discard the automatic proposal visibly rather than
holding stale follow-up behind it.

Deduplicate admission using parent turn ID + Stop event ID. Restart/crash must
not resubmit an already-admitted continuation. An uncertain dispatch is handled
by the scheduler's existing recovery contract, without automatic replay.
Stop errors do not retry themselves, and a hook cannot reset its own chain
budget by changing text or requesting another child.

## 8. Cancellation, revocation and cleanup

Capture owner generations on admission; validate before command launch, MCP
dispatch, result acceptance and scheduling. Disable/uninstall seals admission
durably before cleanup and suppresses that plugin's pending events, including
Stop, Interrupt and SessionEnd. Cancellation cannot turn a revoked hook back on.

Host cleanup owns process termination, waiter settlement and final resource
release. A cancelled waiter must not abandon cleanup or release a lease before
the owned child has settled. Queue admission is bounded and closes before
shutdown drains. Preserve the caller's cancellation/error when cleanup also
fails; record unresolved resources separately.

Track pending launch ownership before a process becomes publicly usable.
Kill and reap the owned process group/tree on timeout or cancellation using
platform-qualified mechanisms. Deliberately escaped descendants remain
outside the guarantee. An unclean owner shutdown follows the companion plugin
spec's surviving-child recovery rule; lock acquisition does not establish
that the previous hook/MCP processes stopped.

User stop seals new work first. Interrupt gets a bounded observation window
afterward and cannot extend it. SessionEnd is best effort on graceful closure;
crashes do not replay it. Remote cancellation cannot establish that side
effects were undone; uncertain completion is preserved and never auto-retried.

## 9. Budgets and failure policy

V2 uses the following host defaults and hard ceilings. Legacy configurations
retain ADR-148's existing timeout/output contract; they are not silently
reinterpreted or clamped by v2. Package adapters exceeding v2 bounds require a
visible adaptation or remain unsupported.

| Resource | Default / ceiling | Behavior |
| --- | --- | --- |
| Command or MCP execution | 10 s / 60 s per handler | Timeout follows controlling event policy. |
| Effectful event execution | 60 s total active execution | Stop launching further handlers; fail controlling required effects. |
| Effectful event wall time | 180 s total including queue and all approval waits | Settle the required event as failed; no repeated wait can extend the deadline. |
| Interactive MCP approval wait | 120 s separate wall-time ceiling | Cancel pending hook call; deny required parent guard, otherwise diagnose omission. |
| Interrupt/SessionEnd | 1 s / 3 s total wall time per event | No new approval, connection or continuation; terminate outstanding owned work. |
| Post-kill reap | 5 s maximum | Preserve cleanup-pending ownership if unresolved. |
| Input envelope | 1 MiB UTF-8, depth 32, 16,384 JSON nodes | Required guard rejects the parent operation whole; optional observation drops with diagnostic. |
| Structured stdout / MCP result | 16 KiB UTF-8 | Bound during capture; invalid overflow fails the handler. |
| stderr retained | 4 KiB UTF-8 | Truncate diagnostic capture with marker; never parse as effects. |
| Context | 4 KiB/block; 16 KiB/event; companion 32 KiB/send aggregate | Reject blocks/effect batch whole when required; no silent constraint truncation. |
| Concurrent effectful executions | 4/runtime | Bounded FIFO admission; cancellation releases reservations correctly. |
| Observation queue | 64 events, 4 active workers/runtime | Drop newest optional observation with a count; guards never enter this lossy queue. |
| Definitions | 64 hooks/installation; 256 active/runtime | Refuse activation of the overflowing set; no arbitrary tail omission. |
| Matcher patterns | 32/handler; 256 characters/pattern | Reject invalid definition. |
| Causal depth | 4 nested hook/tool levels | Cycle/limit failure as section 6; no guard bypass. |
| Stop continuations | 3 turns and 120 s wall time per chain | End chain; tighter inherited budgets always win. |
| Continuation message | 4 KiB; 8 KiB combined/settlement | Reject oversized proposal/combined turn. |
| Context/decision reason display | 1,000 characters for reason | Sanitize diagnostic view; decision remains structured. |

Active execution timeout excludes approved interactive waiting but the
separate wall-time ceiling remains enforced. Never hold a scarce execution
slot while waiting for the user or for nested hook tool dependencies. Use
structured async orchestration; nested callbacks cannot block all available
workers while waiting for work queued behind themselves.

If an event requires initialization/context that cannot be supplied, the
dependent capability remains unready. UserPromptSubmit failure remains
fail-open for unrelated work; that does not authorize omitting a required
capability constraint. Observation errors cannot rewrite settled results.

New v2 diagnostics retain identifiers, event, owner, duration, status and
bounded sanitized reasons. Do not put stdout, exact tool arguments, prompts,
environment or credential values in ordinary logs. Explicit private debug
capture must use existing capture policy and redaction. Legacy logging
behavior remains documented separately; importing a plugin never opts it
into legacy raw-output logging.

## 10. Vendor adapters

Adapters translate a named, version-pinned dialect into these contracts.
They must test event timing, matchers, payload, output, execution type,
working directory, variable expansion and timeout behavior together.
A matching event name alone is insufficient.

Codex and Cursor offer their own hook schemas and lifecycle meanings:
[Codex hooks](https://learn.chatgpt.com/docs/hooks),
[Cursor hooks](https://cursor.com/docs/hooks).
Chatbook intentionally keeps its permission authority and live-runtime session
definition. Vendor permission-allow results never become bypasses here.

Initial mapping targets:

| Source event family | Target | Qualification condition |
| --- | --- | --- |
| Codex PreToolUse / Cursor preToolUse | PreToolUse | Exact argument/decision mapping; transformed inputs revalidated. |
| Codex PermissionRequest | ApprovalRequested | Observation-only adaptation; permission mutation unsupported. |
| Vendor prompt submission | UserPromptSubmit | Manual-origin and explicit refusal semantics preserved or shown as stricter adaptation. |
| Vendor post-tool/failure | PostToolUse / PostToolUseFailure | Success/error/dispatch distinctions explicitly matched. |
| Vendor subagent lifecycle | SubagentStart / SubagentStop | Native child boundary and narrowing restrictions available. |
| Vendor session/compaction | SessionStart/End, PreCompact/PostCompact | Show Chatbook session lifetime and actual compaction semantics. |
| Vendor stop/interrupt | Stop / Interrupt | Bounded scheduler continuation and cancellation precedence maintained. |
| Cursor shell/MCP/file-specific events | Canonical tool operation matcher | Provider classification and required payload are actually available. |
| Editor Tab, thought-stream or unavailable vendor events | Unsupported | Never invent an equivalent or silently discard a required guard. |

Source command strings are not passed wholesale to a shell. A supported
adapter can recognize a simple platform-specific executable/argv form and
show the resolved argv for review. Shell operators, substitutions, ambiguous
quoting and unsupported regex matchers remain unsupported until the user
provides an explicit native argv/matcher adaptation. Do not use a POSIX
tokenizer to claim Windows command-string equivalence.

Foreign regex matchers are not treated as globs. Support only an enumerated,
tested subset with equivalent bounded behavior; otherwise require explicit
replacement. Foreign hooks requiring bypass permissions, unavailable
payloads or unpreservable timing block their declared dependent behavior.

## 11. Verification and integration

1. Preserve legacy six-event config, default behavior and actual synchronous/
   asynchronous entry paths. Pair denial assertions with a successful control.
2. Exercise every event at its real Console/agent/scheduler boundary, including
   durable and preauthorized invocation paths; keep existing approval exemptions
   valid for unchanged calls.
3. Prove transformations cannot retain approvals for different arguments,
   rename a tool, widen child budgets or strip required restrictions.
4. Prove context attribution/lifetime and whole-block overflow behavior before
   and after compaction, session replacement, late completion and revocation.
5. Use controlled command processes and stdio/Streamable HTTP servers for
   output limits, auth/approval waits, malformed results and disconnects.
6. Exercise recursive MCP guard cycles, worker starvation, nested approval
   observers, missing connections and timeout/cancellation during cleanup.
7. Prove Stop deduplication, chain budgets, foreground priority and no replay
   after ambiguous scheduler dispatch; user stop and plugin drain always win.
8. Kill the real owner during launch and active work; observe surviving
   children and independent-process recovery on Linux/macOS/Windows.
9. Check exact input delivery separately from safe display/log projections
   using credential/prompt sentinels and bounded adversarial input.
10. Qualify version-pinned vendor fixtures with independent expected behavior,
    including documented adaptations and unsupported required guards.

Initial implementation targets include Agents/run_hooks.py, existing agent
dispatch/child lifecycle seams, Console runtime/submit/interrupt owners,
context compaction and scheduler admission. Their exact changes belong to
atomic implementation plans, not a generic event-bus rewrite.
Use existing Tests/Agents/test_run_hooks.py and Console hook regressions as
compatibility controls; add focused suites for new contracts and runtime
integration. Full test suite remains opt-in.

## 12. Delivery and ADR check

Implement native event/result/ownership contracts first; qualify direct MCP
transport and permission dispatch next; add MCP-backed handlers after those
dependencies. Plugin adapters supply owned definitions through the same
runtime. Every slice carries targeted tests and documentation.

ADR required: yes
ADR path: backlog/decisions/163-expanded-console-hook-runtime.md
Reason: New lifecycle events, structured effects, scheduling, recursion,
privacy and permission-sensitive runtime interfaces.

This spec and the plugin spec remain subject to written-spec review.
They define intended behavior and acceptance evidence; no implementation or
runtime test success is asserted here.
