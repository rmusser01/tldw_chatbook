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
