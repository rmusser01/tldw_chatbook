# ADR-163: Expanded Console hook runtime

Status: Accepted (2026-09-15) — written specification reviewed; implementation pending.
Date: 2026-09-15
Related Task: [TASK-32645](../tasks/task-32645%20-%20Design-managed-plugins-and-expanded-hook-runtime.md)
Supersedes: N/A; explicitly extends ADR-148's v1 scope while preserving legacy behavior.
Companion: [ADR-162](162-managed-agent-plugins.md)

## Decision

Extend the shared Console hook runtime with versioned events and structured
effects, including input proposals, narrowing child constraints, MCP-backed
handlers and bounded scheduler continuations. Preserve Chatbook's permission
authority, legacy hook behavior and explicit lifecycle ownership.

## Context

[ADR-148](148-console-run-hooks.md) provides six user-configured argv-command
events with deny-only permission effects and bounded execution. Plugin
interoperability benefits from additional session, subagent, compaction,
failure and interruption events. The user approved extending these hooks
as a shared foundation rather than keeping vendor hooks inert.

The expansion must not convert a hook response into permission bypass, recursively
invoke itself, block cancellation, or attach late output to another conversation.
Package discovery and shared execution need separate module boundaries.

## Contracts

1. Keep legacy user [[hooks.hook]] interpretation. Add explicit v2 handlers and
   a closed native plugin hook file. Do not infer a new protocol from stdout.
2. Add SessionStart/End, SubagentStart, PostToolUseFailure, PreCompact/PostCompact
   and Interrupt. Event boundaries belong to actual host runtime operations.
   Tab focus is not a session boundary.
3. A live hook session is tied to conversation/workspace and an immutable active
   hook set. Idle configuration replacement starts a new set explicitly;
   required initialization precedes dependent capability admission. Reserve a
   provisional session/run under normal authority and budgets; MCP initialization
   needs already-connected, independently eligible dependencies and rejects
   cycles. Failed initialization publishes no dependent capabilities or root Stop.
4. Hooks may deny, add bounded untrusted context, propose complete tool inputs,
   narrow child limits or request continuation only where the event allows it.
   Unknown/undeclared effects do not become authority. Explicit required and
   require_context fields distinguish successful completion from mandatory
   nonempty context, including standalone user handlers. Dependency edges retain
   their narrower scope; explicit required handlers control their owning event.
   Required post-events establish pending admission checkpoints before completion
   consumers can start the next model step or normal settlement. Validated effects
   and checkpoint release commit together; failure preserves settled results and
   blocks subsequent input/use. Disabling hooks cannot erase active requirements;
   required teardown, approval-observation and Stop dependencies are invalid.
5. Run input transformers in deterministic order, validate each proposal, freeze
   final arguments, run non-transforming validators and remaining optional context
   handlers, then perform full existing permission review. Mixed transform/deny
   handlers run once in the transformation phase and cannot establish a guard
   guarantee about final arguments. Required final constraints use distinct qualified
   guards; effect-free optional observers use their bounded queue. Fresh dispatch
   checks bind approval to the actual call.
6. MCP hooks use ordinary tool schemas, profiles, approval and cancellation.
   Causal-cycle detection never silently skips required guards. Observation of
   an approval cannot recursively create another approval request. Retain typed
   MCP results and reject tool/transport errors before interpreting effects;
   normalize only qualified structured, single-text or empty-success forms.
   Conflicting representations, unsupported shapes and overflow cannot become
   pass decisions or merged effects.
7. Stop continuation is one deduplicated scheduler proposal with chain/time
   limits and inherited budgets. User work, cancellation, revocation and update
   draining prevent stale automatic follow-up.
8. Seal the affected live installation/workspace/run scope and begin host
   cancellation before waiting for durable writes or trust unlock. Suppress
   affected plugin cleanup callbacks and stale checkpoint results immediately;
   persistence failure cannot reopen live admission or claim durable disable.
   Workspace disable preserves other scopes' authorized hooks and approvals;
   shared MCP requests retain scope ownership and cannot kill another authorized
   user's transport. Host cleanup owns process reaping, counted unresolved
   resources and surviving-child recovery, without claiming sandbox containment.
   File/data deletion follows ADR-162's durable fence and confirmed drain rules.
9. V2 has explicit input/output/time/concurrency/context limits and safe
   metadata-only ordinary diagnostics. Legacy output/logging remains its named
   contract; plugins are never silently placed into legacy raw-output logging.
   Per-runtime and application-wide v2 admission limits include provisional,
   child, late and cleanup work, with fair queues and counted surviving children.
   The teardown notification deadline and subsequent host-reaping allowance
   are distinct; neither delays immediate cancellation/admission sealing.
10. Runtime interfaces consume owned definitions and authority validators;
    they do not depend on marketplace discovery. Adapters qualify complete
    event/matcher/payload/output behavior and report stricter adaptations.
11. V2 definitions have a closed host-owned event projection and closed nested
    effect shapes. A resolved active dependency is supplied to phase selection
    by the host, not authored in a handler. Transformation takes precedence over
    validation, context and observation. A strict bounded raw JSON gate retains
    duplicate-key evidence for command/text results; already decoded objects
    receive shape validation only. Matcher availability comes from qualified
    producer identities, never names or untrusted result properties. Child cap
    proposals are intersected with current child authority by the accepting
    owner; zero denotes unlimited only for the specified token, tool-result and
    tool-call caps. PreCompact turn-lifetime context belongs to its pending
   compaction candidate and is discarded if that candidate aborts.
12. Whole-batch v2 rejection retains bounded event/scope metadata so invalid
    selected guards and explicit requirements cannot vanish. Invalid optional
    context remains optional: `require_context` checks accepted output, not
    failure scope. Unresolved malformed declarations refuse their v2 definition
    set's activation; they do not create a blanket requirement on unrelated
    legacy/user work. Active dependency state stays with the host owner.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Keep six events and reject all richer hooks | Misses the approved shared functionality and useful interop. |
| Treat vendor event names as direct aliases | Session timing, payloads, failures and output effects differ. |
| Permit hook allow to bypass permission review | Violates the existing permission floor and turns imported code into policy authority. |
| Execute vendor command strings through an implicit shell | Adds quoting/substitution execution without a reviewed argv contract. |
| Suppress all hooks for hook-origin MCP calls | Avoids recursion by bypassing required guards. |
| Run Stop follow-ups recursively | Loses scheduler ownership, user priority, deduplication and budgets. |
| Rewrite legacy config semantics | Breaks existing user hooks to introduce optional new functionality. |

## Consequences

- The companion spec defines an event/effect/failure matrix and exact budgets.
- Direct MCP transport and permission integration precede MCP-backed hooks.
- Vendor compatibility remains qualified per handler; prompt/agent handlers,
  editor-only events and permission-bypass semantics remain unsupported.
- Tests need real runtime entries, positive controls, causal-cycle cases and
  process-death evidence on each supported platform.
- ADR-148 remains the authoritative legacy contract. This ADR
  extends its previously deferred scope; it does not rewrite its historical
  behavior or automatically enable new hooks.

## Links

- [Expanded hook runtime design](../../Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md)
- [Managed plugins design](../../Docs/superpowers/specs/2026-09-15-managed-plugins-design.md)
- [Hook implementation plan](../../Docs/superpowers/plans/2026-09-15-expanded-hooks.md)

## H2 execution seam (TASK-32677, 2026-09-16)

One `HookBudgetOwner`, created lazily on the running Console application loop,
serves every immutable v2 engine from `ConsoleRuntime.ensure_hooks_v2`. Its
thread-safe reservations retain queued, active and suspended lifetimes;
`HookTicket.suspend()` frees only execution capacity and `acquire()` rejoins the
fair queue. Observations have separate pending and active counts. No per-call
loop or plugin import is introduced.

`HookEngine.begin_event(event)` issues a read-only execution handle with engine-controlled private accounting, the
original 180-second wall deadline and cumulative 60-second active allowance.
`fire_handler_async(scope, handler_id, event)` and agent-thread `fire_handler`
execute only an engine-owned definition with the same top-level event identity.
H3 may change validated event data between invocations; it owns transformer
ordering, argument validation, final permission review and atomic checkpoint/
effect acceptance. Closing the scope irreversibly forbids reuse. Public construction, transfer and
state reassignment are refused without retaining a new application registry. `fire` rejects calls from its owner loop; async producers
use `fire_async`. `from_config` preserves the master switch and H1's scoped
invalid-admission records rather than activating part of an invalid batch.

`begin_close()` synchronously seals ordinary engine admission and fixes the
teardown deadline once. `fire_teardown_async`/`notify_teardown` accept only
still-authorized Interrupt/SessionEnd during that window; `close()` closes this
last admission seam and shields its retained drain. Per-run Interrupt does not
require closing a session engine. H4/H5 own actual event publication and per-run
cancellation; an unrelated authorized runtime keeps its application allowance.
Console disposal retains cleanup through cancelled callers, permits teardown
until the fixed seal-plus-three deadline, then joins through seal-plus-eight.
An unresolved owner keeps the disposed runtime attached rather than being
mistaken for quiescence. Exact session closure also retains its fleet/wake
fences while that session engine has unresolved delivery or process custody;
a bounded close return alone cannot release them.

The plugin-independent `HookProcessOwner` reserve/publish/settle protocol is
adapted by plugin composition to F8's exact-root owner: all roots and dirty
checkpoint custody precede launch. Pending launch tasks, actual transports,
terminal callbacks and execution tickets remain owned across cancelled waiters
and late publication. PID metadata alone is not restart identity; the adapter
must record its platform-qualified identity. Only actual direct-child reap and
absence of its POSIX group permit terminal settlement here. Signal attempts,
pipe EOF and callback completion do not. A failed owner settlement retains the
record and counters for explicit reconciliation. Darwin local process groups
are qualified by controlled children; deliberately escaped descendants and
Windows tree-terminal proof remain outside this qualification. R47 refuses
Windows v2 commands with `unsupported_platform` before launch, root reservation
or environment access; no execution/lifetime ticket leaks. Linux POSIX behavior
remains unqualified until its own real controls run. Legacy hooks are unchanged.

Authority callback exceptions become fixed `authority_check_failed` outcomes at
admission and final acceptance. Dependency metadata failures become
`dependency_check_failed` and retain conservative dependency-required state;
this does not promote a dependency-only requirement into an owning-event veto.
Explicit/event-control requiredness remains intact. Cancellation propagates
through the retained cleanup path, never as a secret-bearing error message.

### H3 dependency admission and dispatch provenance (R48–R49)

Use host-resolved qualified handler IDs when checking dependent input readiness;
keep dependency-only failure separate from owning-event requirements. Empty means
positively independent, while unknown mapping refuses. Normal terminal admission
joins all pending required postevents before applying their failure scopes. I1
supplies admitted graph-to-handler mapping; Hooks does not own another graph.

Carry optional host `ToolResult.dispatch_state` (`not_started`, `settled`,
`uncertain`) from actual dispatch owners to H3's common result boundary. This
resolves existing ambiguous timeout/cancelled outcomes without inspecting prose.
Not-started gates emit no tool postevent; uncertain completion cannot establish a
known execution failure or authorize replay. Preserve permission owners, exact
approval identity, resource ownership and legacy observational callbacks.

R50 extends the existing live context carrier with separate hook origins and a
host-neutral mixed send-budget check. This preserves existing whole-block copy
and transformation behavior without a second carrier or fake package identity.
Hook blocks use hook limits even when genuinely package-attributed; linked dual
provenance is charged once. Required acceptance includes the whole rendered
contribution before checkpoint release, and final model transport validates the
actual combined send before stripping live sidecars.

H3 R51 uses the installed jsonschema validator with an explicit empty
`referencing.Registry()` so embedded refs resolve and missing external/file refs
refuse without retrieval. Diagnostics contain fixed metadata only. A schema is
not network/filesystem authority; future remote-schema adapters must provide
reviewed local resources, not discard constraints or enable automatic fetches.

H3 retains the original issued event scope across the legacy final guard:
transforms/final validators precede that guard, optional context/observations
follow its acceptance, and ordinary permission review remains authoritative.
The catalog snapshot and exact argument bytes are rechecked after existing
argument repair. Owned call objects and exact host reconstruction bindings keep
repeated/missing model call IDs independent. Idempotent post requirements precede
all service completion consumers (including definitive and inline-skill callbacks)
and common publication. Terminal settlement joins all pending required events
before reporting an existing scoped failure. H4/H5 use the shared context carrier;
I1 supplies the actual dependency selector and genuine package attribution.

R52 promotes jsonschema into core (`>=4.26,<5`) and declares the direct referencing
runtime dependency (`>=0.37,<1`), matching qualified installed APIs. Development
extras cannot supply a production validation boundary implicitly. The dev-only
duplicate is removed; unrelated pins/extras stay unchanged. Built-wheel metadata
and isolated offline imports qualify packaging against available dependency files;
fresh complete dependency resolution remains I7 distribution qualification.


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


### H4 clarification: session, provisional and manual owners

Automatic compaction precedes AgentService run construction, while explicit
manual summarization has no root agent run. One runtime-session checkpoint
coordinator therefore carries explicit host-owned subordinate scopes and is
shared with the tool pipeline. This preserves atomic currentness/context/input
admission without inventing durable runs or a competing gate. Manual operations
may lazily initialize the session only after actual validation, authority and
capacity admission; previews/focus do not, and manual operations never arm root
Stop. Required PostCompact is installed at actual successful memory commit and
fences relevant next input while preserving that commit.

Manual PostCompact context uses one bounded exact-memory handoff owned by the
still-live session: accept before manual completion, bind and consume once at
the first matching accepted turn, then apply turn lifetime. Capture session,
workspace, immutable hook set, branch and committed memory identity/revision.
Invalid submissions, stale/late output, cancellation or ownership changes cannot
consume or retarget it. Required failures remain fences; dropped material never
counts as delivered. The alternative of putting synthetic AgentRuns IDs on
manual actions misrepresents the existing runtime, while independent session and
run gates would lose atomic effect acceptance. A general cross-run context queue
is not introduced.

Actual H4 producers also extend the closed event projection with inherited child
task/tool IDs/budget caps/model/provider, pending auxiliary compaction
candidate/reason, and committed memory ID/prefix digest/reason. These are bounded
descriptive data, not permission or model-replacement effects. Local visual
projection alone has no committed-memory PostCompact boundary.


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

### F8 saved-root lifetime handoff (2026-09-16)

ADR-162's R18/R43 amendment owns exact-root grants and the separate protected
clean/dirty checkpoint. H2/H6 and M4 register all grants before launch/access,
retain them through idle connection and pending/active/reader lifetime, and report
actual terminal evidence through PluginRuntimeOwner. Cancelled waiters and request
completion do not release a surviving writer's grant. Graceful shutdown closes
admission before drain and publishes the final clean checkpoint only after joined
ownership settlement. No additional hook cleanup owner or permission gate is added.

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
`tldw_chatbook/DB/migrations/chachanotes_v74_to_v75_hook_continuation_receipts.sql`;
its migration test is `Tests/DB/test_chachanotes_v75_hook_continuation_receipts_migration.py`.


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

## Typed MCP transport evidence and shared execution (R58)

The external MCP typed entry runs through the existing local and unified control
services, preserving their governance, timeout, cancellation and single audit
owner. The legacy entry retains its explicit display projection. The five public
`MCPToolResult` fields carry protocol content, structured content, strict error
status, metadata and a separate sanitized transport failure. Private host evidence
retains the bounded, immutable original UTF-8 result-value bytes, duplicate-key
validation and dispatch state. This is the result span, including unknown members
and internal whitespace, not the JSON-RPC envelope or a reserialization. Existing
frame/result/depth limits remain independent of the hook result cap.

Decoded mappings/models without original transport evidence remain unqualified
for hook effects. Remote members cannot populate host evidence. Nested mutable
fields or model copies cannot borrow qualification for changed effects; hook
normalization derives its effects from the immutable qualified bytes or verifies
exact agreement first. In-process built-in application mappings retain their
legacy contract; they are not silently reinterpreted as wire protocol results.

One per-call host observation may carry dispatch truth across cancelled awaits:
not_started before proven dispatch, uncertain immediately before the actual write
attempt, settled on a validated matching terminal protocol response. Invalid or
lost responses and after-write cancellation retain uncertainty. An unacknowledged
bridge-future cancellation is not proof that dispatch never occurred or cannot
still occur. This observation owns no authority, process, lease or timeout. No
uncertain result authorizes replay or late effects. Audit receives the explicit
display projection and records tool-declared/transport failure honestly.

M1 implements R58 with `call_tool_result`, `execute_external_tool_result` and
`execute_hub_tool_result` through the existing owners. The raw reader retains
exact result spans for single/batch responses; the frozen result's private
mutation check stores a 32-byte SHA-256 rather than a second full encoded copy.
`MCPDispatchObservation` is carried through the existing await chain and has no
resource or permission ownership. The ordinary provider formats only after
classifying typed errors and preserving host dispatch state. Builtin and legacy
adapter mappings remain unqualified. H6 must cap and interpret qualified original
bytes and reject unqualified mappings; M2 must qualify its own HTTP raw decoder,
framing and dispatch evidence before using this contract.

The same host observation classifies the existing builtin delegate await after
its governance gates, without reinterpreting its mapping as MCP wire data.
Unobserved legacy-session failures remain uncertain rather than claiming a
pre-dispatch refusal.


### Bounded MCP bridge audit publication (R59)

The provider and existing unified execution service share one host-owned,
per-invocation atomic audit publication claim. A bridge Future timeout or
cancellation cannot establish that service audit did not start, nor prove a
possibly dispatched operation was blocked. Claim before attempted publication;
keep claim/capacity locks out of audit I/O and preserve sanitized uncertainty.

The unified service may perform bridge-fallback metadata publication on at most
one in-flight daemon thread per service, with no waiting queue and no caller
join. This covers failed submissions even when the target loop is closed, while
keeping provider completion bounded. It adds no execution/permission owner or
durable schema and retains no tool arguments/results. Saturation or thread-start/
write failure may lose a best-effort row; it cannot permit a duplicate publication,
change execution authority, infer remote completion or authorize replay. An
unscheduled fallback does not consume an otherwise available service publication.
Capacity remains held until the actual writer exits. A stalled filesystem write
may retain that one daemon writer and metadata until it returns or process exit.

### R70 — Actual provisional MCP admission

H6 builds its private prospective MCP invocation view through the real validated Console admission path before SessionStart, using the resolved configuration and existing workspace/parent, registry and reservation owners. Host injection is an extension seam; the normal caller must supply the context. The view remains unadvertised and grants no temporary permission or accepted root turn. Missing native dependency declarations remain unavailable until I1 supplies the graph. Later invocations use the actual AgentService run context and existing ToolHookRun guards and post-event settlement.

Initialization uses only independently eligible, already-connected capabilities. Enforce this narrowing at the actual standalone and owned connection/dispatch boundaries so an intervening disconnect refuses instead of reconnecting. Preserve exact current authority, request provenance and lifetime ownership during approval/nested suspension. A cancelled reacquisition waiter cannot release a lifetime ticket for unresolved work. This adds a narrow preparation interface and may add preparation work; it does not add a second permission, budget or lifecycle runtime.

### R71 — Internal hook operations and outer barriers

An MCP operation performing a hook uses a scoped operation checkpoint in the existing shared store. It retains an exact live parent and queries actual parent/ancestor pending and failed requirements for the tool's declared dependencies, including owning-event requirement IDs. Unknown mappings refuse. It runs ordinary tool guards and joins its own required post events before returning.

The outer event's generic next-input and terminal barriers remain installed on that outer owner; they cannot block the internal work needed to complete the same event. They still prevent normal input/settlement until all required outer work, including sibling post events, settles. A fresh unrelated checkpoint or an empty dependency list cannot manufacture readiness. Parent closure/revocation, static and dynamic cycles, actual resource custody and the original event budgets remain controlling. This narrow continuation query avoids self-wait while preserving dependency authority; it requires direct pending/failed ancestor, independent initializer and two-post-event controls.

### R72 — Nested MCP context and pending turn admission

An MCP call that performs a hook still runs ordinary PreToolUse/PostToolUse hooks. Their accepted context keeps its original event/handler attribution and turn lifetime even when their internal operation scope settles before the next input. Stage those contributions through the existing context/checkpoint owners until the containing event succeeds and its current authority is rechecked. Do not publish them directly into a live parent while an enclosing sibling can still deny that event; preserve the parent and sibling joins and complete per-event/shared-send limits.

During SessionStart, nested turn context belongs only to the exact pending submission that triggered initialization. Use its existing admission/lifecycle reservation and release the contribution only to that submission's accepted turn; do not create a root run early, promote it to runtime context, or transfer it to a later submission. SessionStart's own direct context keeps its documented runtime lifetime. Failed or replaced admission, enclosing denial, cancellation, revocation or closed parent discards the nested batch. Required context that cannot be delivered safely fails its controlling admission; omission cannot count as success.

For an already admitted operation, use the actual receiving input scope and the existing context carrier. Normal retirement of the nested request does not by itself retire accepted parent-bound context; parent currentness, original effect/definition authority and the containing acceptance still govern delivery. There is no second context runtime or generic cross-turn transfer. Qualify actual first-input delivery, no next-turn leak, cancelled/failed/replaced admission, sibling denial, required overflow, normal admitted nested context and unchanged direct SessionStart runtime contributions.

### R73 — Session capability views and bounded teardown

A pending submission's MCP view has the exact input lifetime defined by R72. A successful live session may also retain the same prepared registry/configuration ceiling as a private runtime view, with no input-context destination or accepted root identity. Retiring a pending binding cannot overwrite a newer binding, retain a failed admission, or transfer its turn context to the runtime view. Events with an actual run still require their exact run authority.

Seal first closes ordinary admission and starts the original teardown clock through the immediate engine cancellation fence, before entering lifecycle/checkpoint locks. Before retiring ordinary checkpoint/input scopes, the existing lifecycle/checkpoint owner may issue a narrow teardown scope from its already-prepared capability view and positively checked dependency state; ordinary scopes then close without reopening. It must retain the exact source identity/readiness provenance and refuse unknown, pending, failed or changed requirements; a closed or discarded requirement is never evidence of independence. Capturing this host state must not delay cancellation on external I/O. This is one bounded transfer within the existing owner, not a new runtime, restored closed scope, new permission or standalone empty dependency store.

Only an actual host-issued Interrupt/SessionEnd delivery can carry this scope across the seal. Its nested normal tool preparation, guards and post-event joins execute as continuations of that delivery in the same engine and retain its original notification deadline, causal limits, counters and cleanup allowance. A public event label or boolean cannot create that authority. Current tool/configuration/credential/scope permissions and already-connected checks still apply at the real dispatch and acceptance boundaries. Guards are enforced; no approval prompt, connect/reconnect, continuation turn, replay or deadline reset is introduced. A runtime-only view supplies no destination for nested turn context, so an undeliverable required contribution refuses under its ordinary controlling policy.

Retire this scope/view at the bounded notification boundary without confusing that retirement with actual request/process cleanup. Unresolved resource custody stays with its existing owners. Repeated close must neither issue new authority nor extend deadlines. Qualify actual Console close after a completed turn, preserved ordinary post-seal refusal, a denying nested guard, stale/dependency/context refusal, and real notification/cleanup timing. Best-effort omission remains correct whenever ordinary current authority or the exact original deadline no longer permits dispatch.


#### TASK-33648: saved standalone SessionEnd during host disposal

Host disposal closes ordinary Console and HookPermissions admission immediately.
An already-live lifecycle may retain its one host-issued, effect-free SessionEnd
notification through that closure. The exception belongs to its exact existing
engine-issued execution state, event identity, session and original notification
deadline; event labels, public teardown booleans, copied events and ordinary calls
cannot acquire it. Repeated sealing or disposal cannot issue another notification
or extend the window. Managed-plugin authority and ordinary MCP dispatch retain
their existing boundaries.

The standalone command still needs its original exact saved definition and grant.
At actual process creation, the existing permission owner reads canonical config
and local consent under their existing launch locks, compares the original
config/profile path and section identity, definition fingerprint and grant epoch,
and rechecks the authentic bounded delivery. Changed, revoked, reapproved,
disabled, locally sealed, refresh-pending, missing or unavailable state refuses.
This private teardown validation does not republish ordinary targets, reopen the
closed owner, create a grant, accept effects, prompt, reconnect or start a model
turn. Existing command launch, ticket, process and physical reaping custody remain
responsible through cancelled disposal callers and the unchanged cleanup allowance.

This is a narrow implementation of H2/R73 and
[ADR-197's standalone v2 consent](197-console-hook-configuration-review.md#current-dev-integration-standalone-v2-consent-task-32679),
recorded prospectively for
[TASK-33648](../tasks/task-33648%20-%20Preserve-authorized-SessionEnd-notification-during-Console-host-disposal.md).
No new permission owner, persistence schema, dependency or general post-disposal
authority is introduced. Qualification requires real saved/granted exact-close
and disposal controls, changed/revoked/forged negative controls and cancellation
with actual process, ticket and physical cleanup evidence.

### H6 integration and operational handoff

`MCPHookExecutor` consumes the ordinary `ToolCatalogRegistry`/`MCPToolProvider` path and the existing `ToolHookRun` preparation, guard and post-checkpoint owners. Original typed result capture is request-local and qualifies only the exact final returned `ToolResult`; display output grants no native effects. Current profile/persona/kill/schema/automatic-work and exact owned MCP authority remain in their existing owners, including checks after waits. Already-connected-only policy reaches both actual standalone and owned connection boundaries.

Console supplies its private prospective view from the validated submit reservation/configuration; AgentService supplies its actual run registry and dependency resolver. Independent supported standalone MCP is wired through the normal Console entry. Native managed graph/application composition remains I1: absent managed dependency declarations refuse. The same checkpoint store checks actual prerequisite states for internal operations and stages nested input context until its containing event accepts. Runtime and bounded teardown views have no input destination. Original provisional accounting labels preserve the same registry call-cap counters across those views.

Cancellation is not settlement. A hook ticket retires only after the local worker, actual bridge coroutine/lower result and every exact captured owned request prove completion. An uncertain standalone request with no later positive evidence retains cleanup-pending lifetime custody; it is never replayed. SessionEnd/Interrupt retain original deadlines and ordinary guards, cannot prompt/connect, and cannot acquire a later replacement view while queued. Platform qualification and exact evidence are recorded in the Task18 implementation report.


H6 Fix1 binds a private read-only currentness callback to the exact captured result and original call identity. The existing normal owner revalidates profile, definition, permission and owned mapping/scope/credential authority after required post-event settlement and after the final asynchronous hook-authority check, before native effects are accepted. The original one-time approval remains specific to that invocation; validation does not prompt, invoke, connect or recreate grants. Blocking owner validation remains off-loop under the same job/ticket and original deadline. A completed MCP request or cancelled waiter cannot settle a still-running validator.

### Current H3 approval projection integration

The normal provider may add its existing approval provenance to the captured result. Its single internal projection helper transfers the witness only from that exact captured object to the helper's approval-only dataclass replacement, under the capture lock. Unanswered metadata remains unanswered. Arbitrary replacements, late owned refusals, repeated capture or closed scopes cannot inherit the witness; result currentness still uses the original call authority. This fixes the six approved/session projection paths without changing permission or wire-result ownership.

The Console's hook currentness probe compares live workspace roots, project authority, permission profile and persona rules without rebuilding unrelated catalogs. The normal MCP provider reuses the service's exact live catalog resolver at hook dispatch and result acceptance; a vanished, stale, disconnected or changed definition refuses. Permission and kill-switch reads remain fresh at their normal boundaries. Custom host configuration providers retain their snapshot contract. This removes repeated catalog I/O from lifecycle polling without caching permission, restoring authority or extending the one/three-second notification deadlines.


I1 integrates selected owned hooks into the actual next-run configuration without
replacing the standalone scheduler. The native adapter reserves F2 command/root
custody before launch, publishes actual H2 process evidence, and releases only after
real terminal settlement. The existing checkpoint validates dependencies of the
actual tool definition and live typed material; unavailable graph nodes refuse the
affected operation while unrelated tools remain eligible. Manual preparation retains
an unchanged native configuration; changing workspace requires fresh configuration.


### I2 foreign hook qualification boundary

Pinned Codex/Cursor event correspondences are proposals, not HookHandlers. Native
payloads cannot currently reproduce full vendor cwd/model/permission/transcript,
input/output, success-only timing and timeout behavior. Every proposal records all
qualification axes; no incomplete mapping enters H2/H6. Shell syntax/substitution
and unqualified regex semantics never become argv/globs. Root/group/handler unknown
or required guard constraints are preserved and fence affected package activation;
known optional observers remain visible but unavailable. Ordinary user messages
remain possible without expanding blocked package material. No foreign allow result
can reach or bypass the native permission store. This documents the deliberately
unsupported runtime subset of TASK-32687, not original-host qualification.


### v75 receipt migration after the compaction v74 merge (2026-10-01)

PR #2939's v74 failure-reason migration is retained unchanged. Hook continuation
receipts migrate from v74 to v75, including genuine v73/v74 upgrade and atomic
rollback/reopen evidence. Current core and combined subscriptions recovery
catalogs are recaptured from actual constructors at v75; exact schema and version
validation remain required. The compaction retry latch and structural no-cost
fences run before the shared compact-once operation, which retains the same hook
owner. Required pre-hook failures record their content-free reason with no
summary call and do not claim a billed attempt. No consent, admission or cleanup
policy changes. Evidence: the PR #2946 integration report.
