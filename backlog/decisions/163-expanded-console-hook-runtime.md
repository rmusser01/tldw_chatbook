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
