# Expanded Console hook runtime

Date: 2026-09-15
Status: Approved for implementation planning after written-spec review and amendments.
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
The existing master enabled switch applies to both versions. It suppresses
execution, not the requirements captured by an active v2 definition set or a
plugin dependency. Those requirements remain unsatisfied while hooks are off;
turning the switch off cannot admit work without its required guard/context.
Legacy configurations without v2 requirements retain their existing behavior.

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
| effects | Explicit subset permitted for the event; empty declares no returned effects, but a success requirement can still make the handler controlling. |
| required | Optional boolean, default false; successful completion is required for the owning event under section 2.4. |
| require_context | Optional boolean, default false; success requires a nonempty accepted context contribution. Failure follows the event/required/dependency policy. Valid only with the context effect on a context-capable event. |
| match | Optional structured matcher; absent matches every occurrence of that event. |
| timeout_seconds | Finite positive execution timeout within host limits. |
| input | MCP-only JSON argument template, default empty object. |

Allowed effect names are deny, context, updated_input, child_limits,
continuation and stop_continuations. Command-specific fields are argv, env and
cwd; MCP-specific fields are server, tool and input. Type-inappropriate fields
are invalid. id, event, type and effects are required fields. match,
timeout_seconds, required and require_context are optional; input is optional
for MCP and forbidden for commands. Requiredness is normalized and included in
the reviewed definition digest, never inferred from output text.

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
hook. Authentication/setup happens through explicit connection flow. Section 3.2
defines the provisional authority for initialization before the first run.

Reject duplicate IDs, unknown fields, unsupported versions, illegal effect/event
combinations and malformed required handlers. Invalid required guards block
their dependent capabilities. Optional observations have no declared effects and
no controlling event policy, explicit required flag or active dependency
requirement. Context-producing handlers remain effectful even when failure is
optional; they never enter the lossy observation queue. require_context defines
valid output rather than changing its failure scope. Optional observation
failure is diagnostic, not authority to disable an unrelated component.
If a user v2 batch is rejected whole, retain bounded body-free metadata for
each inspected declaration's supported event and failure scope: explicit
required, selected event-policy control, optional or unresolved. An unresolved
shape prevents activation of that v2 definition set until repaired; it is not
silently downgraded to an optional handler. `require_context` alone remains an
output-success rule and never becomes explicit requiredness. A rejected
optional-only batch does not create a global run requirement. Active dependency
edges remain host-owned state outside this config record.

The pure phase classifier accepts host-resolved `dependency_required` (default
false), which is absent from definitions and their digest. `updated_input` wins;
otherwise deny, explicit required or an active dependency selects validation;
otherwise context selects context, and empty effects select observation.

`env` names use `[A-Za-z_][A-Za-z0-9_]*`. Values are literal strings or exactly
`{"variable": "DECLARED_NAME"}`; references require an owning declared-variable
context. NUL and reserved `PLUGIN_ROOT`/`PLUGIN_DATA` names are invalid. Host
values are set last. No shell or recursive expansion is implied.

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

The host-owned top level is closed: protocol_version=2, nonempty event_id/event/
timestamp/runtime_session_id, applicable run_id/parent_run_id/turn_id/workspace_id,
initiator, origin, optional plugin owner_installation_id/owner_component_id,
causal_chain_id, causal_depth and bounded data. Initial documented data keys are
SessionStart.reason (startup/resume/configuration_changed), UserPromptSubmit.prompt,
PreToolUse tool_name/tool_args/tool_id/provider/operation/definition_hash/
original_arguments/candidate_arguments, PostToolUse and PostToolUseFailure the
corresponding dispatch identity/status/failure code/result fields,
ApprovalRequested calls/session_active, SubagentStart child_task/tool_ids/
budget_caps/model/provider, SubagentStop child_run_id/status, PreCompact
candidate/reason, PostCompact reason/memory_id/summarized_prefix_digest, and
Stop.status. Other event data is empty. Future producers must amend
this projection before exposing further keys or template paths. Tool argument
and result objects are bounded untrusted JSON, never host identity.

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
`child_limits` is a closed object with at least one of duplicate-free `tool_ids`
(empty means no catalog tools) or nonempty `budget_caps`. Allowed caps are
max_steps, max_model_turns, max_wall_seconds, max_subagents,
max_subagent_result_chars, max_tool_result_chars, max_total_tokens,
max_tool_call_seconds and max_model_retries. Numeric booleans, nonfinite,
negative and unknown values fail; steps, model turns, wall seconds and
subagent-result characters are positive. The accepting owner intersects with
already contained child authority and separate runtime-tool gates. Zero means
unlimited for total tokens, tool-call seconds and tool-result characters only;
it cannot widen an existing finite cap. `continuation` is exactly one nonempty
UTF-8 `message` of at most 4 KiB; combined Stop settlement remains 8 KiB.
`stop_continuations` accepts only true. Neither effect supplies scheduler IDs.
Context lifetime is `turn`, or `runtime` on SessionStart only.
Nonzero exit, output overflow and invalid JSON are errors; exit code 2 is
not a new v2 protocol shortcut. Legacy/vendor exit meanings are translated
by the selected adapter before v2 result validation.

#### MCP result normalization

The MCP client/service boundary must retain the complete typed tool result,
including isError, structuredContent and content, until hook validation finishes.
Do not feed hooks a formatted display string or the current content-only client
projection. Transport/protocol failure, permission refusal, cancellation and an
MCP isError: true all fail the handler before any returned effects are considered,
even if the body contains a valid-looking pass decision. A present non-boolean
isError is invalid; absence follows the selected protocol profile's success
semantics. An uncertain call outcome is not converted into empty success.

For native v2 MCP hooks, normalize a successful tool result as follows:

1. If structuredContent is present, it must be a v2 result object. content may
   be absent or [], or contain exactly one text block encoding the same JSON
   object as a compatibility mirror. Reject mismatched or additional content;
   do not merge two representations or fall back after invalid structured data.
2. With no structuredContent, an explicitly successful result whose content is
   absent or [] means pass with no effects, like empty successful command stdout.
   It can satisfy a success-only requirement, never require_context: true.
3. Otherwise, require exactly one text content block containing one complete
   v2 JSON object. Allow surrounding whitespace, but no Markdown fences, prose,
   concatenated blocks or embedded-resource/image-derived effects.

Reject duplicate JSON keys, malformed objects and unsupported result shapes.
Command stdout and MCP text pass a strict UTF-8 raw gate capped at 16 KiB
before parsing. It rejects duplicate keys at any level, non-JSON constants,
trailing data and non-object values. Empty successful command output is a
separate no-effect success. Object validation cannot recover duplicate-key
provenance from an already decoded dict. Typed MCP protocol decoding, complete
result framing/size/depth and representation provenance are M1/M2/M3/H6 owner
requirements before H1 result normalization.
The 16 KiB MCP hook result cap covers the complete tool-result payload, including
both representations and metadata, before v2 normalization; do not truncate it.
The transport must also enforce finite framing/body/depth limits before parsing
the protocol envelope. A hook cap checked only after an unbounded receive is not
enforcement. Preserve ordinary non-hook MCP result behavior through a separate
display projection of the same typed result.

Generic tools returning prose are not implicitly hook protocols. A versioned,
reviewed vendor adapter may translate a documented result shape into v2; it must
preserve error status and bounded capture and cannot reinterpret an error as
pass. Show stricter shape limits as an adaptation. Neither discarded content
nor raw error bodies become ordinary logs, context or additional effects.

### 2.4 Required success and context

There are three sources of requiredness, with the stricter applicable rule
winning:

- Existing controlling event policy: selected PreToolUse transformations/guards
  and SubagentStart restrictions still fail closed, even when required is false.
- Explicit required: true: a user-configured or reviewed plugin handler must
  succeed for its owning event to continue. The plugin review
  shows that this can block the owning run/event, not just one package component.
- A plugin requires edge: success at the applicable event is a prerequisite for
  its declared dependent capabilities. This narrower scope does not become a
  blanket failure of unrelated capabilities. Dependency requirements apply even
  if the handler's own required field is false.

A successful pass with empty command stdout satisfies a success-only requirement;
guards may pass and transformers may make no change. It does not satisfy
require_context: true. That flag requires at least one non-whitespace context
block with the event's allowed lifetime, accepted in full within all budgets.
Malformed, missing, rejected or oversized required context is a failure. Merely
declaring the context effect does not make its output mandatory. Plugin edges
require success and also inherit the target handler's require_context contract.
require_context defines a successful result, not its failure scope; it does not
silently promote a dependency-only requirement into an owning-event requirement.

For standalone user hooks, required initialization uses required: true; required
context additionally sets require_context: true. No plugin dependency declaration
is needed. Nonmatching handlers do not create a failure. Disabled, invalid or
unavailable required handlers do:
the runtime must retain their requirements when it filters executable handlers.
Removing a requirement is an explicit definition/dependency change for a newly
reviewed/admitted set, not a side effect of a switch or a parse failure. An
already-admitted snapshot cannot lose its requirements halfway through work.

Required pre-event failure blocks the pending admission/action. Required
PostToolUse, PostToolUseFailure or PostCompact failure preserves the settled
result/committed compaction and fences the owner's next model input. A required
SubagentStop contribution similarly fences the active parent's next input while
preserving the child's settled result. With only a dependency-scoped requirement,
mark those dependents unready and refuse their next use; never silently omit
material from an explicit dependent invocation. Show a remediation or cancel
action. Do not replay the tool, compactor or hook automatically. If the owner
already settled, retain the diagnostic without reopening it or attaching effects
to a later run.

Install a pending checkpoint synchronously when a required post-event is
published, before completion is exposed to consumers that could admit the next
model request or declare the owner normally settled. Persisting/displaying the
tool result or compacted summary need not wait. The checkpoint covers the owning
event, or only affected capability use for a dependency-scoped requirement.
The latter still blocks model input that needs those capabilities or their
required instruction material; independently eligible work may continue.

Required handlers resolve the checkpoint only after successful validation and
atomic acceptance of their effects under the same admission coordination used
for the next model request. Failure/timeout leaves a failed checkpoint with
remediation; it never clears the pending requirement by omission. Key checkpoints
by event ID, owner and captured authority so a late result cannot release a
different event's barrier. No coordination lock is held while awaiting a hook.

For a failed dispatched tool, register requirements for both PostToolUse and
PostToolUseFailure before exposing completion; finishing the first event cannot
release the second event's requirement. SubagentStop installs the checkpoint in
the still-active parent at child settlement. A parent request already in flight
is unchanged, but its next model admission observes the pending checkpoint.
Normal owner settlement, root Stop and continuation admission cannot overtake
outstanding required post-events. User cancellation still seals admission
immediately and closes the owner; late results are discarded while host cleanup
retains resource ownership. An event whose owner already settled cannot create
a checkpoint in a later run.

ApprovalRequested, Stop, Interrupt and SessionEnd do not support explicit required
or require_context set true, or incoming required dependency edges. Reject these
declarations. Observations/continuation proposals at those boundaries cannot
veto host settlement, cleanup or already-completed work. Stop's existing
continuation-veto/error rules still apply. Optional observations alone use the
lossy notification queue; required handlers never do.

## 3. Event timing and effects

| Event | Exact boundary | Allowed v2 effects | Failure behavior |
| --- | --- | --- | --- |
| SessionStart | During provisional admission, before the first run of a live hook-runtime session | context with runtime lifetime; deny initialization | Explicit required failure blocks session admission; dependency-only failure blocks its dependents. Optional context failure is visible. |
| UserPromptSubmit | Explicit user submission after base input validation, before context preparation/admission | deny; turn context | Explicit denial blocks; optional failures remain fail-open as in ADR-148. Explicit v2 requirements block the owning admission; dependency-only failure blocks its dependents. |
| PreToolUse | Candidate prepared, before normal approval and dispatch | updated_input; deny; turn context | Fail closed for selected transformation/guard failures. |
| ApprovalRequested | A normal permission request is created | observation only | Diagnostic only; cannot approve, modify or answer for the user. |
| PostToolUse | Dispatched call settled, including ordinary execution-error results | turn context; observation | Preserve the result/effects. Optional failure is diagnostic; required failure fences subsequent input/dependent use under section 2.4. |
| PostToolUseFailure | After PostToolUse when a dispatched call reports execution failure | turn context; observation | Same required/optional checkpoint policy as PostToolUse; denied/not-dispatched calls do not emit it. |
| SubagentStart | Child draft has inherited restrictions, before child admission | deny; child_limits; child-turn context | Fail closed for selected guards/restrictions. |
| SubagentStop | Child result has settled | observation; context for the parent's next model step if still active | Required failure fences the active parent's next input/dependent use; no restart or output rewriting. Otherwise discard late context visibly. |
| PreCompact | Actual compaction candidate prepared, before compactor runs | context supplied to compaction input | Required context needed by the candidate aborts compaction on failure; never silently drops history. |
| PostCompact | New compacted context successfully committed | turn context for subsequent model input | Optional failure is diagnostic; required failure fences subsequent input/dependent use. No rollback or editing of committed summary. |
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
conversation, workspace binding and immutable active hook set. It is prepared
lazily during first-run admission, becomes live after required initialization
succeeds for the capabilities being admitted, and ends on explicit conversation
runtime closure, process shutdown or an idle context/hook-set replacement.

Changing tab focus does not create or end it. Durable conversation history is
not itself a live session. Reopening archived history does not run hooks until
a new execution admission is attempted. A new process creates a fresh live
session after recovery; there is no replay of missed SessionEnd events.

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

### 3.2 Provisional initialization and MCP prerequisites

Before SessionStart, perform base submission validation and resolve the actual
workspace/parent authority. Reserve admission under the existing run-capacity
and budget limits, pin the trusted component/hook snapshot, and allocate a
provisional session/run identity. This reservation is cancellable and revocable;
it is not an accepted root turn and does not advertise dependent capabilities.
It holds no lifecycle/permission lock or scarce execution slot while awaiting
setup/approval. No SessionStart event is emitted merely to inspect a package.

Resolve initialization dependencies against a private, prospective capability
view under those exact restrictions. A SessionStart MCP hook may call only an
already-connected dependency whose eligibility does not itself depend on that
unfinished initialization. Explicit connection/setup uses the ordinary reviewed
connection flow and never bypasses its own requirements. Missing connections
produce Needs configuration; SessionStart cannot launch/connect them itself.

The dependency graph includes component prerequisites and matching required
guards on initialization tool calls. Reject known cycles before dispatch,
including an initializer that needs a server gated on that same initializer;
retain causal-cycle checks for dynamically resolved invocations. Setup and
connection tests cannot become a back door around this graph. Authors must use
an independently eligible initializer dependency or revise the declarations.

Initialization calls use the reserved workspace/parent identity, normal tool
schema/profile/approval checks, fresh authority checks and ordinary process/run
leases. They get no borrowed approval stamps or temporary extra tool rights.
An update/disable can cancel the provisional reservation just as it can block
normal admission. Stage SessionStart context until its controlling requirements
succeed, then publish the live session and finish run admission.
Dependency-only failure leaves those components unready and discards their
dependent staged material; independently eligible work may still be admitted.
An explicit request for an unavailable dependent is refused, not silently altered.

On owning-event failure/cancellation, discard staged context and release the reservation
after owned work is settled or transferred to explicit cleanup-pending ownership.
Do not emit root Stop or replay initialization automatically. Already-performed
MCP/command effects remain recorded; initialization is not transactional rollback.
A setup change requiring different definitions creates a new admission attempt
with a freshly validated snapshot.

## 4. Matching and deterministic effects

Structured match keys are tool_id, provider, operation and reason. Values are
nonempty arrays of exact values or bounded case-sensitive glob patterns.
Keys combine with AND; values within a key combine with OR. Empty/unknown
keys or unavailable fields are invalid for that event. No shell expressions,
LLM matching or unbounded regular expressions.

SessionStart accepts only reason. PreToolUse/PostToolUse accept tool_id,
provider and operation. PostToolUseFailure additionally accepts reason. All
other events accept no matcher. Tool identity/provider are host-resolved catalog
identities; operation is a qualified canonical operation and failure reason is
a host-normalized code. An unavailable operation makes a matcher requiring it
unsupported without invalidating a tool-ID-only matcher. Missing required
producer identity/reason is invalid producer state, while a supported optional
occurrence value may genuinely nonmatch. ApprovalRequested is a batch.

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
3. Freeze final arguments. Run all matching non-transforming validators,
   including legacy deny guards, against those final arguments. A denial or
   controlling failure dominates staged context/pass results. No further
   transformation runs afterward.
4. Run remaining optional context-only handlers against the frozen arguments,
   in stable order. Stage their valid contributions; diagnose optional failures.
   Publish optional effect-free observations of that same final candidate to
   the bounded observation queue. They cannot supply effects or gate dispatch.
5. Accept staged effects under current authority, compute the final
   definition/argument identity and run the full existing permission/profile/
   workspace review chain.
6. Immediately before dispatch, recheck that approval, arguments, definitions,
   policy and owner generations still match.

Classify every valid declaration once from its reviewed definition, not from
the fields it happens to return on a particular call. Admit handlers lazily in
phase order under the existing reservation/execution budgets:

| Declaration | Phase and guarantee |
| --- | --- |
| updated_input declared, including with deny/context | Transformation phase only. Any denial concerns the candidate at that point; it is not a check of all later transformations. |
| No updated_input; deny declared, required: true, or an active dependency requirement | Final validation/completion phase. Deny guards validate the frozen arguments; success-only handlers establish only their declared completion/context requirement. |
| No updated_input/deny or completion requirement; context declared | Optional context phase, using frozen arguments. require_context can make its result invalid but does not itself widen failure scope. |
| Empty effects with no completion requirement | Optional observation queue, using frozen arguments. |

A transformer that returns no updated_input still stays in its declared phase.
Never rerun a mixed-effect handler as a final guard: the second execution could
repeat external effects. Native authors needing a final-argument constraint
declare a separate non-transforming deny guard and require that guard. A requires
edge to a transformer establishes only successful transformation-phase execution,
not final-argument validation. Inspection labels this distinction explicitly.
Vendor required-guard mappings cannot be satisfied by such a transformer alone;
require a separately qualified final guard/adaptation or leave that constrained
behavior unsupported. Do not invent a second invocation to claim equivalence.

Legacy guard-only batches retain their documented concurrent first-deny
behavior. Without new transformers their input is unchanged. Within a mixed
pipeline, final guards inspect the final candidate. Repeated transformation cycles
are forbidden; there is one finite pass. A transformer may deny on original
or intermediate input, and original arguments remain available for attributed
review. Context from a transformer remains attributed to that candidate; it is
never presented as a constraint validated against the final arguments.

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

Capture workspace/run owner identities and generations on admission; plugin
handlers also capture installation-wide authority generations. Validate before
command launch, MCP dispatch, result acceptance and scheduling. Disable here
fences the affected workspace immediately under admission synchronization;
Disable everywhere/uninstall fences all installation
scopes. Invalidate live ownership and begin host cancellation before awaiting
durable writes, trust unlock or mutation completion. Suppress the affected scope's
pending plugin events, including Stop, Interrupt and SessionEnd, from that live
fence onward. Close affected approvals/checkpoints/waiters through their existing
cancellation paths; a late result cannot release a checkpoint into runnable work.
Cancellation cannot turn a fenced hook back on.

Persistence failure keeps that live fence in place while host cleanup continues.
The plugin coordinator distinguishes a session-only block from committed disable
or uninstall; a failed write cannot establish a durable outcome or prove process
termination. Retry/reconciliation cannot silently reopen the live scope or release
unresolved process ownership. Durable revocation and file/data deletion follow
the companion plugin spec's separate commit and exact-root drain contracts.

Requests and callbacks retain their original workspace/run ownership even on a
shared MCP connection. Disabling A must preserve authorized B work, approvals and
callbacks. Share only equivalent reviewed execution/configuration/credential
authority and compatible session state, as the plugin spec requires. Cancel or
detach A's requests without killing a transport still serving B; discard A's late
effects while retaining uncertain outcomes and resource ownership. Terminate a
shared process only when all its owners are affected or have drained. Independently
configured user hooks/connections keep their own authority; they cannot revive
the fenced plugin scope.

Host cleanup owns process termination, waiter settlement and final resource
release. A cancelled waiter must not abandon cleanup or release a lease before
the owned child has settled. Queue admission is bounded and closes before
shutdown drains. Preserve the caller's cancellation/error when cleanup also
fails; record unresolved resources separately.

Track pending launch ownership before a process becomes publicly usable.
For exclusively owned processes, kill and reap the owned process group/tree on
timeout or cancellation using platform-qualified mechanisms; shared processes
follow the scoped request rule above. Deliberately escaped descendants remain
outside the guarantee. An unclean owner shutdown follows the companion plugin
spec's surviving-child recovery rule; lock acquisition does not establish
that the previous hook/MCP processes stopped.

User stop seals new work first. Still-authorized Interrupt handlers get a bounded
observation window afterward and cannot extend it; plugin disable/uninstall
suppresses affected handlers rather than waiting for them. The notification
deadline includes queueing and dispatch; on expiry, accept no further hook output
and start host cleanup under the scoped process/request rules above. The separate
post-kill reap allowance is host cleanup time, not an extension in which hooks
may run, prompt or submit effects. The maximum awaited teardown path is the
notification window plus the reap allowance (at most 3 + 5 seconds); unresolved
processes remain cleanup-pending after that, without claiming they stopped.
Cancellation/admission sealing does not wait for either window. SessionEnd is
best effort on graceful closure;
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
| Interrupt/SessionEnd notification | 1 s / 3 s wall time per event, including queue and execution | End notification/output acceptance at deadline; no new approval, connection or continuation; initiate scoped host cleanup. |
| Post-kill reap | 5 s maximum host cleanup allowance after notification/execution ends | Outside the handler/event window; preserve cleanup-pending ownership if unresolved. |
| Input envelope | 1 MiB UTF-8, depth 32, 16,384 JSON nodes | Required guard rejects the parent operation whole; optional observation drops with diagnostic. |
| Structured stdout / MCP tool-result payload | 16 KiB UTF-8 | Bound capture; MCP includes content, structuredContent and metadata before v2 normalization. Overflow fails the handler; protocol framing is independently bounded. |
| stderr retained | 4 KiB UTF-8 | Truncate diagnostic capture with marker; never parse as effects. |
| Context | 4 KiB/block; 16 KiB/event; companion 32 KiB/send aggregate | Reject blocks/effect batch whole when required; no silent constraint truncation. |
| Concurrent effectful executions | 4/runtime; 8/application across all v2 sessions | FIFO within a runtime, round-robin admission across eligible runtimes; cancellation releases reservations correctly. |
| Outstanding effectful handlers | 16/runtime; 64/application, including queued, active and suspended nested/approval continuations | Reserve one lifetime ticket per handler; refuse overflow under the event's required/optional failure policy; never dispatch an unguarded parent call. |
| Observation queue | 64 pending handler deliveries/runtime; 128/application; 4 active workers/runtime and 8/application | Each matching handler is one delivery. Drop newest optional delivery with a count; fair round-robin admission across runtimes; required handlers never enter this lossy queue. |
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

The application-wide v2 counters include provisional initialization, all live
sessions, child/late event work and teardown; creating another conversation
does not create another application allowance. They are independent of the
Console's adjustable root-run cap. Suspended required work retains a bounded
admission ticket without occupying an execution slot. Nested work needs its own
ticket; exhaustion fails the controlling event rather than bypassing guards or
waiting outside the bounds. Use the earlier of inherited and local deadlines.
Queueing/approval cannot reset those deadlines, and re-admission joins the fair
queue instead of jumping ahead of other runtimes.

Spawned local children consume their pool's application resource allowance until
reaped; cleanup-pending ownership retains that accounting. Repeated timeout or
session replacement cannot accumulate uncounted surviving processes. Settling
a cancelled waiter is distinct from releasing its owned process resources.
Remote outcomes can remain uncertain after local transport cleanup; these limits
bound host-owned work, not arbitrary remote side effects. Existing legacy pools
and timeout/output behavior remain separate; these are aggregate v2 guarantees,
not a retroactive change to ADR-148's contract.

If required initialization/context cannot be supplied, apply section 2.4's
owning-event or dependency-scoped failure. UserPromptSubmit remains fail-open
for optional errors, while explicit v2 requirements can block submission.
Observation errors cannot rewrite settled results.

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

Written-spec amendment scenarios are mandatory acceptance evidence:

- **Required success/context:** a standalone SessionStart required handler may
  succeed with empty stdout; the same handler with require_context refuses empty,
  whitespace-only or oversized context and accepts a valid bounded block. Cover
  the same distinction through plugin dependency edges. Master-off, disabled and
  invalid required handlers preserve the failure condition instead of vanishing
  from the executable list. Optional handler failure is a successful fail-open
  control where the event permits it.
- **Post-event checkpoint:** a tool side effect/result and a compacted summary
  each commit once, then their required context hook fails. Preserve those
  outcomes, block the next owning model input and do not replay either operation.
  Check narrower dependency-only failure, active-parent SubagentStop, late owner
  settlement, and rejection of required teardown/approval/Stop declarations.
- **Pending checkpoint race:** hold a required post-hook at a deterministic
  barrier after the tool/compaction/child outcome is visible, then attempt next
  model admission and normal root settlement. Neither can overtake an applicable
  pending checkpoint. Success atomically contributes context and releases it;
  failure/timeout keeps admission blocked. For an execution error, completing
  PostToolUse cannot release a still-pending PostToolUseFailure requirement.
  Test concurrent child settlement, already-in-flight parent requests,
  dependency-only independent work, cancellation, revocation and stale replies.
- **MCP result normalization:** an isError: true result containing a valid pass
  object fails before effects; so do transport failures and non-boolean error
  flags. Successful structured-only, single JSON text, exact mirrored and empty
  results exercise each accepted form. Mismatched mirrors, extra text/resource
  blocks, duplicate keys, prose, invalid structured data with valid fallback text,
  and oversized payload metadata fail without truncation or effects. Empty success
  fails require_context. Exercise the production typed client/service path on
  stdio and Streamable HTTP, with ordinary non-hook result display as a control.
- **PreToolUse phases:** transformer A checks/rewrites a path, transformer B
  changes it again, and a required final guard C refuses B's final path. C must
  see the final arguments and prevent dispatch; A cannot satisfy C's guarantee.
  With an allowed final path, dispatch receives that exact approved candidate.
  Exercise deny-plus-context guards, context-only handlers, required effect-free
  completion and optional observations. A mixed handler executes once even when
  it returns no replacement, and a vendor final-guard mapping to only that mixed
  handler remains unsupported. Context-only handlers observe the final candidate.
- **Initialization admission:** an explicitly connected independent MCP dependency
  succeeds through normal approval under a provisional workspace/run identity.
  A disconnected server launches nothing, a known initialization/guard cycle
  dispatches no tool, and denied/cancelled/revoked initialization publishes no
  dependent capability/context or root Stop. A successful control reaches normal
  run admission; no path can reuse a stamp from another run or bypass a guard.
- **Aggregate bounds:** exercise enough simultaneous/provisional/late sessions
  to exhaust application limits as well as per-runtime limits. Count actual
  active children/workers and pending tickets, prove FIFO/round-robin progress
  for an admitted quiet runtime, and verify cancellation, nested work and session
  replacement release or retain the correct reservations. A full optional queue
  cannot drop a required guard or permit its parent invocation.
- **Teardown timing:** with a controlled slow/unkillable-child simulation, seal
  admission immediately, cease notification/output acceptance by its deadline,
  and bound subsequent awaited host reaping separately. Check both confirmed
  exit and cleanup-pending outcomes; the latter remains counted and cannot
  advertise that the child stopped. Use real controlled child processes for
  the successful kill/reap control on each qualified platform.
- **Scoped shutdown with failed persistence:** hold/fail each plugin persistence
  boundary while A/B have pending hooks, approvals and post-event checkpoints.
  A's disable immediately fences output/admission and starts host cancellation;
  A's late success cannot release its checkpoint or start a continuation. B's
  authorized work still settles on a qualified shared MCP connection. A's
  uncertain request retains ownership without killing B's transport; global
  disable affects both. Successful durable retry never revives stale events,
  and failed persistence remains visibly session-only. Independently configured
  user handlers retain normal Stop/Interrupt behavior. Exercise the companion
  plugin spec's data-drain scenarios with idle and cleanup-pending hook/MCP
  processes; waiter settlement alone cannot prove root users have stopped.

Initial implementation targets include Agents/run_hooks.py, existing agent
dispatch/child lifecycle seams, Console runtime/submit/interrupt owners,
context compaction and scheduler admission. The existing
[MCP client](../../../tldw_chatbook/MCP/client.py) also needs its content-only
tool-result projection replaced by a typed service result before hooks can rely
on error/structured fields. Display consumers keep a separate compatible projection.
Their exact changes belong to
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

Written-spec review is complete. The [hook implementation plan](../plans/2026-09-15-expanded-hooks.md)
and [delivery plan](../plans/2026-09-15-managed-plugins-delivery.md) carry the
remaining work. This spec defines intended behavior and acceptance evidence;
no implementation or runtime test success is asserted here.

### H2 implemented owner interface

The execution foundation is `Agents/hooks_v2/{budgets,ownership,command_executor,engine}.py`.
The following APIs implement the accepted bounds without publishing the later
H3–H6 events or installing plugin adapters:

- `HookBudgetOwner()` binds the running application loop.
  `reserve(runtime_id, observation)` refuses overflow synchronously and returns
  a lifetime ticket. `await ticket.acquire()`, `ticket.suspend()` and
  `ticket.release()` separate execution capacity from lifetime custody.
  `snapshot(runtime_id=None)` exposes exact `execution`, `tickets`,
  `observations` (pending) and `workers` (active observation) counts. Release
  after actual process settlement; suspending a required nested/approval wait
  never releases its lifetime ticket. Optional observations cannot suspend for
  approval or nested work.
- `HookEngine(definitions, authority_check, budget_owner, *, process_owner,
  environment, host_environment, invalid_admissions, dependency_required,
  enabled)` keeps an immutable definition tuple and one injected owner.
  `authority_check(handler, event, stage)` runs outside admission locks at
  admission, launch and result acceptance. `environment(handler, event)` resolves
  declared reference names; `host_environment` sets `PLUGIN_ROOT`/`PLUGIN_DATA`
  last. Ambient reserved roots are removed first. The dependency callback is a
  synchronous host metadata accessor; active graph ownership stays with its host.
  `from_config(config, authority_check, budget_owner, **options)` retains H1's
  disabled and rejected-batch state. A rejected batch never launches any entry;
  explicit/event-control records retain their event scopes, while optional or
  unresolved entries do not become blanket requirements.
- `fire(event)` is the worker-thread facade; `await fire_async(event)` is the
  Console entry. `begin_event(event)` plus
  `fire_handler_async(execution, handler_id, event)`/`fire_handler(...)` supports
  H3's validated transformer chain on the same loop, budgets and process owner.
  Only data may change in the same execution scope. Unknown definitions,
  another engine's scope, changed host identity, concurrent use, expired or
  closed scopes refuse. `execution.close()` retires the caller-owned scope;
  the handle exposes read-only deadline/accounting/lifetime properties, cannot
  be publicly constructed or reassigned, and carries private engine-issued
  accounting without an unbounded registry of contexts. A caller cancellation
  discards effects while retained command work finishes cleanup.
- `HookEventOutcome` separates `accepted` `(handler_id, HookResult)` tuples,
  `failures` (fixed metadata codes and owning-event/dependency requiredness),
  `omissions` and `outstanding_cleanup`. `succeeded` is successful execution;
  `allowed` applies owning-event failure/deny scope. Required checkpoint release,
  aggregate context acceptance and downstream scheduling remain host actions.
  `notify(event)` admits only optional effect-free deliveries and returns an
  immediate admission boolean; overflow drops newest and increments the engine's
  metadata-only `notification_omissions` count. Fixed failure codes from admitted
  observations are counted in `notification_failures`; raw output is never
  retained there.
- `begin_close()` is the idempotent synchronous ordinary admission seal.
  `fire_teardown_async(event)`/`notify_teardown(event)` provide the still-authorized
  Interrupt/SessionEnd seam afterward. Their original event wall deadlines are
  capped by seal + 3 seconds and cannot reset on repeated calls. `close()` seals
  teardown admission and joins retained cleanup within seal + 8 seconds. Per-run
  user Stop must use H5's run cancellation, not terminally close the session
  engine. Revoked callbacks are refused by current authority even in teardown.
- `ConsoleRuntime.ensure_hooks_v2(session_id, definitions, authority_check,
  **owner_options)` lazily binds the app loop and shared budget independently
  of views. Existing snapshots reject replacement definitions. Session close
  seals only its engine; app disposal seals all before drain. `close_hooks_v2`
  joins shielded ownership. Exact-session hook custody must be settled before
  its fleet/wake fences can be released; a bounded close return is insufficient.
  `hooks_v2_cleanup_pending` prevents detaching an unresolved disposed owner. Actual H4 lifecycle producers supply scoped IDs and
  snapshots; no automatic SessionStart, Stop continuation or MCP dispatch is
  added here.

`HookProcessOwner.reserve_launch(event) -> str`,
`publish_process(token, provenance) -> None` and
`settle_process(token, confirmed) -> None` are plugin-independent. All root grants
and protected dirty checkpoint publication must precede reserve returning.
Publication receives non-secret process metadata; plugin composition must add
native restart identity and use the final F8 owner, rather than infer identity
from a PID. False settlement retains root/process ownership. The standalone
host owner retains the same in-memory custody without a plugin runtime.

Input is validated whole before JSON serialization. Capture drains both pipes,
retains at most 16 KiB original stdout and 4 KiB stderr, and records stderr
truncation as a separate marker; diagnostics do not expose raw bodies. Complete
stdout alone reaches H1 `decode_result`; nonzero exit and partial/invalid output
never supply effects. Launch tasks and actual transport handles outlive caller
cancellation. POSIX cleanup requires a child exit callback/return code and an
absent owned process group; transient signal errors do not substitute for that
proof. Retained records may be explicitly checked again with
`engine.processes.reap_pending()` after later host terminal evidence.

Qualification is controlled local Darwin processes, including a live/exited
leader, retained grandchildren, cancellation during launch/publication and
refused kill/settlement. Deliberately escaped descendants are outside the group
guarantee. R47 refuses Windows v2 commands before launch, root reservation or environment
access, with a fixed `unsupported_platform` outcome and released host reservation.
Linux POSIX behavior remains explicitly unqualified until its own controls run.
Future Windows support requires a qualified whole-tree terminal-proof backend
and real controls; legacy hooks remain unchanged. H6/M4 must likewise supply typed MCP outcomes and exact remote/local
ownership; this command foundation does not infer them.

H2 callback failures are protocol outcomes: `authority_check_failed` at admission
or final acceptance, and `dependency_check_failed` for an unavailable dependency
accessor (including while constructing another failure or considering optional
notification). Unknown dependency state remains dependency-required for the
owning capability coordinator; explicit/event-control failure scope is unchanged.
These outcomes contain no exception text. `CancelledError` retains its original
cancellation semantics and cannot release unsettled process custody.

### H3 integration contracts (R48–R49)

H3 consumes the exact app-owned immutable H2 session through a read-only runtime
accessor. It does not initialize H4 sessions. Engine-issued event scopes capture
host dependency status once for phase planning, execution and failure scope;
unknown dependency status remains required and cannot enter the lossy lane.

Post-event checkpoint queries accept host-resolved required handler IDs for the
next input/capability use. Explicit/event requirements always gate their owner;
dependency-only requirements gate only the selected dependent work. An empty
selection means positively independent work, never missing dependency resolution.
Unknown mappings refuse admission. Normal settlement joins every pending required
post-event, then retains each failure's original scope. I1 owns plugin graph
mapping; H3 does not create a graph or import Plugins into the hook engine.

`ToolResult.dispatch_state` is optional host provenance: `not_started`, `settled`,
or `uncertain`. Actual host permission/capacity/start gates publish `not_started`;
known returned/raised calls are `settled`; an unresolved worker after abandonment
or timeout is `uncertain`. Remote payload fields and error prose cannot establish
this state. Not-started results emit neither tool post event. Uncertain results
retain uncertainty in PostToolUse and never fabricate PostToolUseFailure or replay
settled tool work. Known errors install both distinct requirements before result
publication. Lower provider adapters must preserve honest host provenance.

H3 context uses the existing live `PluginContextText` carrier with separate
`HookContextOrigin` records (R50). User hooks have no invented installation.
Genuine package origins accompanying hook material remain linked attribution;
the same block is charged once. Copying and whole-block assembly preserve both
origin sets. The host-neutral final model-send check enforces hook4KiB/block,
16KiB/event, package8KiB/block and combined32KiB/send, including multimodal text.
H3 stages whole rendered contributions before accepting/releasing a checkpoint.
Only the final checked transport copy loses its live sidecars.

H3 candidate validation is offline (R51): `jsonschema`'s selected validator uses
`referencing.Registry()` with its no-retrieval default. Embedded/internal refs
remain supported; unavailable external/file refs fail with fixed metadata before
hook execution or permission review. No URI, payload, network or filesystem fetch
is authorized by a tool schema. Future adapters must supply reviewed local schema
resources rather than silently dropping unresolved constraints.

The concrete preparation entry is `prepare_tool(event, engine, *, definition)`;
`definition` is a host `ToolDefinitionSnapshot` containing immutable JSON schema,
name/ID/source and catalog/provider generation. Transformers and required/final
validators run once on one issued scope. The runtime retains that same scope
through the legacy final guard, then executes optional context and schedules
observations before permission review. A denied candidate closes its remaining
scope at run settlement without executing the optional tail. Dispatch rechecks
exact argument bytes and current definitions; existing argument repair cannot
change an approved candidate. Catalog dispatch checks again after repair/gates.
Live call-object ownership, including explicitly bound host reconstructions,
separates repeated or missing model call IDs.

`ToolHookRun` supplies `prepare_call`, `validate_dispatch`, `install_result`,
`admit_input` and `settle` to the existing loop/service. Completion requirements
are installed idempotently before definitive terminal callbacks, observational
callbacks (including inline skills), common result publication, next-model
admission and normal run persistence. Both known-failure events are installed
before either is allowed to complete. Terminal admission joins all pending
required events even when another has already failed, then reports failure in
its original scope. Neither failure nor uncertainty replays a tool.

R52 declares `jsonschema>=4.26,<5` and directly imported
`referencing>=0.37,<1` as core runtime requirements. The old dev-only jsonschema
declaration cannot satisfy a base install. These floors match the qualified
installed APIs; no missing-validator fallback or schema retrieval is permitted.
Offline wheel import qualification may reuse controlled existing dependency files;
it does not claim fresh full dependency resolution or broader platform support.


## Current-dev integration: standalone v2 consent (TASK-32679)

The Console next-Send review and canonical F9 Hooks settings include standalone
`hooks.handler` v2 definitions as well as legacy `hooks.hook` entries. V2 consent
uses the existing app-owned HookPermissions store and grant epochs; its fingerprint
covers the complete normalized closed-schema definition, including event, effects,
arguments, environment, matcher, required policy and timeout. Legacy fingerprints
and grants remain unchanged. A rejected v2 batch stays visible and cannot be
approved or partially activated. V2 definitions have the schema's master switch,
without inventing a per-handler enable field.

Runtime admission reads the canonical saved configuration, rather than an app's
possibly stale dictionary. Configured v2 engines capture exact grants. The existing
permission owner serializes actual subprocess creation against config/consent
changes; revocation and changed definitions fence staged effects and queued work.
Host-injected engines retain their explicit authority resolver and never become
standalone config grants. Lifecycle session replacement remains idle-only.


### H4 session and pre-run checkpoint ownership

The live Console hook session owns one checkpoint coordinator shared with its
provisional turn/operation and actual agent-run scopes. Host-assigned scope
identity is separate from optional actual run_id; manual operations do not create
fake agent runs. Applicable ancestor requirements, event currentness, effect
publication and input admission share the same coordination, with no external
await under its lock. Automatic compaction gates its same pending turn before
the later AgentService run is constructed. Manual summary commits install a
session-owned PostCompact checkpoint before completion publication, preserving
committed memory on failure. Manual execution may initialize a hook session after
real authority and capacity admission; previews/focus/planning cannot do so and
manual execution never emits a root chat Stop. A local visual projection without
a compactor model call or memory commit does not fabricate a PostCompact event.

H4 extends the closed data projection with actual producer fields:
SubagentStart child_task/tool_ids/budget_caps/model/provider; PreCompact candidate
(the actual auxiliary message array) and reason (manual/automatic); PostCompact
reason/memory_id/summarized_prefix_digest from the committed memory record.
Inherited model/provider are descriptive, not replacement authority. Child caps
use the existing closed nine-name numeric contract. Typed present fields and
unknown keys remain strictly validated within the existing raw envelope limits;
actual producers never substitute fabricated identities for unavailable fields.


### H4 delivered host API and operation behavior

`HookSessionLifecycle.reserve/initialize/publish/cancel` owns provisional
SessionStart. `open_scope`, `fire`, `install`, `wait`, `close_scope`, and `seal`
compose lifecycle work with the injected H3 `HookCheckpointStore`; `install`
is synchronous and publishes its pending requirement before scheduling external
execution. Fixed parent scopes carry ancestor requirements; tool events retain
H3's own exact-definition currentness callback. The ledger and checkpoint store
use the same reentrant condition lock. Synchronous currentness probes and effect
publication occur under that lock; external execution and waits for it do not.

`ConsoleRuntime.prepare_hooks_v2` is reached after actual Console validation and
an atomic reversible capacity reservation. Configuration/workspace/binding
changes replace the immutable live session only at idle admission. SessionStart
runtime contributions are untrusted user-role `PluginContextText` with genuine
`HookContextOrigin`, revalidated and emitted once per receiving model owner.
`SessionEnd` seals those owners before H2's retained cleanup join, including
viewless disposal. Existing H2 processes, deadlines and cleanup custody are not
replaced by a lifecycle-specific process supervisor.

SubagentStart receives the already restricted child draft before inline or fleet
admission. Tool identifiers narrow catalog and runtime tools. Budget caps
intersect inherited limits, preserving the existing zero-as-unlimited dimensions;
model/provider cannot be replaced. SubagentStop is installed after durable child
settlement and before parent completion publication. A retired parent receives no
new turn context; the host retains fixed late-context diagnostics and may still
schedule optional observations.

`CompactionHooks.before` appends required PreCompact material only to the actual
auxiliary candidate, including each actual focused-plan fallback candidate, then
checks attribution and prepared capacity. A failed hook aborts without compactor
or hook replay. The existing per-conversation operation serialization lock is
separate from the checkpoint lock and the actual memory commit critical section.
`committed` installs PostCompact immediately after a successful memory commit;
`finish` joins its execution. Automatic compaction fences the same turn's provider
admission. Manual compaction retains one handoff for the exact active memory
identity/revision/digest, branch, workspace and hook session; acceptance claims it
once after durable owner publication. Cancelled, stale or revoked effects cannot
attach to another turn. A required post failure preserves the memory and its
session fence. Hook contributions never become durable summary text.

Lifecycle data is closed and bounded by the existing H1 raw envelope limit.
Compaction `reason` is required and is `manual` or `automatic`. Present candidate
values must be arrays of message objects; present memory identifiers and prefix
digests are nonempty strings bounded to the existing record limits (200 and 256
characters). Child tool arrays are unique nonempty identifiers, and cap objects
accept only the nine existing cap names and numeric/zero semantics. Child task,
model and provider are strings when present. The live producers supply their
actual candidate/record/draft fields; optional absent model/provider and host
run IDs stay absent rather than becoming empty invented identities.


H4 closed scope IDs remain as identifier-only anti-rebind tombstones for their
hook-session owner's lifetime. Context bodies, delivery maps, parent/currentness
mappings and settled checkpoint state retire with their scope; pending cleanup
retains its original bounded custody. The tombstones do not cross sessions and
are never a durable or global registry. Their metadata grows with the number of
closed scopes in a long-lived session; this is not a constant-memory guarantee.
This preserves rejection of stale string-ID reuse without introducing a second
owner-handle protocol. Review actual lifecycle disposal and retention alongside
that tradeoff.


Current-dev H4 integration publishes the transaction's exact USER parent and
assistant parent with both live accepted/recovery owners. This fixes missing
ancestry at its common publication boundary and preserves the current batched
version projection; compaction does not restore per-message database reads on
every dispatch. Configured v2 lifecycle execution uses the ADR-197 exact grant
owner described above. Worker-refreshed source/epoch checks precede execution and
result parsing; checkpoint effect publication uses the same owner's cached
fences without disk I/O or config locks on the app loop. Process creation remains
inside the existing owner transaction on a worker, with transport custody on the
app loop. V2 settings reuse the existing Advanced Config editor and review modal;
no parallel guided schema or per-handler enable switch is introduced.

## H5 continuation admission and retention (TASK-32680)

The existing ConsolePromptQueueCoordinator alone chooses Stop follow-ups. A
host-issued continuation identity travels with queued custody, separately from
the existing `queued` dispatch origin: its initiator is `hook_continuation`, never
a fresh human UserPromptSubmit or child wake. It pins the accepted parent turn,
Stop event, chain start/count and inherited configuration. Only successful
accepted root settlement can issue it; retired operation/run scopes are never
rebound. Required post gates settle before transfer to the live session owner.

ChaChaNotes schema 74 adds `console_hook_continuation_receipts`, keyed uniquely
by `(parent_turn_id, stop_event_id)`. The existing acceptance transaction inserts
the receipt, user-role attributed untrusted input, assistant owner and active
dispatch checkpoint together. A duplicate rolls the competing acceptance back.
Receipt metadata names machine initiation and the parent assistant/conversation;
it contains no proposal body. Receipts survive terminal checkpoint deletion and
soft deletion, and cascade only with permanent parent/conversation removal.
The active checkpoint remains the sole uncertain-dispatch recovery owner; neither
a lost response nor restart replays Stop or automatically replays model/tools.
Ephemeral conversations retain only live-process deduplication.

Each settlement admits at most one combined proposal in declaration order, with
whole-message refusal above 4 KiB and whole-combination refusal above 8 KiB.
Three admitted continuations and 120 elapsed seconds bound a host-minted chain.
Exhausted parent budgets, queued human work, veto, current authority failure,
reviewed update drain, cancellation and closure discard proposals. Machine work
is never retained behind newly arrived foreground work. New turns keep normal
provider/tool authorization, inherited budgets, and the H3 context carrier.

Interrupt observation is installed only after immediate queue/run admission
sealing, once per cancellation identity, on the shared view-independent interrupt
host. H2's one-second Interrupt and three-second SessionEnd observation windows
are separate from retained process cleanup. Revoked callbacks are suppressed;
repeated cancellation neither releases cleanup custody nor resets deadlines, and
no hook can veto cleanup. The schema resource is
`tldw_chatbook/DB/migrations/chachanotes_v73_to_v74_hook_continuation_receipts.sql`;
its migration test is `Tests/DB/test_chachanotes_v74_hook_continuation_receipts_migration.py`.


### H5 admission point across the database worker (R56)

A coordinator-issued, body-free one-use gate is written through the existing
`ConsoleTransactionContribution` seam after messages/checkpoint/receipt insertion
and before transaction commit. Admitted human work and Stop synchronously
invalidate pending gates. Invalidation first rolls the entire transaction back;
consumption first establishes admission and later human work follows the normal
queue. Consumption never proves a commit and is never reset after failure or
uncertainty; the existing checkpoint/receipt owns reconciliation without replay.
The short gate lock covers only pending/invalidation/consumption. Authority,
reviewed plugin drain and deadline checks run outside it; no SQL, callbacks,
cleanup or event-loop waits occur under it. Its canonical fingerprint contains
only host-issued gate/session/entry/parent/Stop lineage. Ephemeral acceptance
consumes the same gate without claiming restart durability. Exact claimed-entry
cleanup releases live captures. The repository Library-policy validator remains
unchanged; the first-save continuation handoff accepts only identical full policy
values changing from new_session/no revision to the accepted durable revision 1.

The 120-second clock begins at the accepted root, and actual remaining AgentService
budgets include time spent waiting for Stop and queue admission. Event envelopes
use the existing `continuation` initiator; persisted custody/receipts use
`hook_continuation`, and scheduled roots retain `scheduled`. Hook messages carry
H3 hook attribution without invented package installation origins; the host's
untrusted-input label is separate from each contributed message's byte count.

A typed admission refusal permits only synchronous cleanup of the exact owned
preparation and transient echo. The coordinator acknowledges only that verified
cleanup epoch; unrelated context changes preserve the normal pause. Other errors
and uncertain acceptance retain ordinary recovery. User Stop seals its affected
turn and emits Interrupt once; SessionEnd belongs to graceful session disposal.
A later ordinary authorized turn can use the same session without reopening the
cancelled turn's admission.

## H5 pending Stop cancellation ownership (R57)

The accepted parent's original provider cancellation Event may be retained by the
existing queue coordinator only for that exact session, assistant and turn while
its Stop settlement is pending. Bind it while the accepted parent is still known;
provider-map removal does not revoke the coordinator's pending-settlement custody.
Retire the reference when that settlement finishes. No synthetic cancellation
identity, retired-scope rebind or later-turn borrowing is permitted.

User Stop first seals admission for that parent, sets its original Event and
notifies the shared interrupt host once for the exact parent. A child task owned
by that settlement runs only its Stop fire_async call. Cancel and join that child
through the existing per-event cancellation path: fire_handler_async stops the
delivery, closes that event execution and retains the delivery/process cleanup
owner until actual terminal proof. Never cancel the encompassing queue/root drain,
close the session or emit SessionEnd for this operation. Suppress CancelledError
only when this exact settlement's recorded user seal caused the child cancellation;
unrelated cancellation propagates. Repeated Stop neither resets deadlines nor
releases unresolved cleanup. Existing Interrupt observation and cleanup bounds
remain unchanged. A fresh authorized turn on the live session remains usable.

The composer exposes this pending Stop availability separately from generation
activity. Expanded and collapsed Stop remain reachable while the hook is pending;
Redirect and the Generating indicator still require an actual active generation.
