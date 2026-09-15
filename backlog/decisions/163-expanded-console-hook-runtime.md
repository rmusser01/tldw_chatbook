# ADR-163: Expanded Console hook runtime

Status: Proposed — approved capability scope; written specification awaiting review.
Date: 2026-09-15
Related Task: [TASK-32645](../tasks/task-32645%20-%20Design-managed-plugins-and-expanded-hook-runtime.md)
Supersedes: N/A while proposed; explicitly extends ADR-148's v1 scope on adoption.
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
   required initialization precedes dependent capability admission.
4. Hooks may deny, add bounded untrusted context, propose complete tool inputs,
   narrow child limits or request continuation only where the event allows it.
   Unknown/undeclared effects do not become authority.
5. Run input transformers in deterministic order, validate each proposal, freeze
   final arguments, run guards, then perform full existing permission review.
   Fresh dispatch checks bind approval to the actual call.
6. MCP hooks use ordinary tool schemas, profiles, approval and cancellation.
   Causal-cycle detection never silently skips required guards. Observation of
   an approval cannot recursively create another approval request.
7. Stop continuation is one deduplicated scheduler proposal with chain/time
   limits and inherited budgets. User work, cancellation, revocation and update
   draining prevent stale automatic follow-up.
8. Seal admission before cancellation/cleanup. Revoked plugin hooks do not
   receive cleanup callbacks. Host cleanup owns process reaping and uncertain
   surviving-child recovery, without claiming sandbox containment.
9. V2 has explicit input/output/time/concurrency/context limits and safe
   metadata-only ordinary diagnostics. Legacy output/logging remains its named
   contract; plugins are never silently placed into legacy raw-output logging.
10. Runtime interfaces consume owned definitions and authority validators;
    they do not depend on marketplace discovery. Adapters qualify complete
    event/matcher/payload/output behavior and report stricter adaptations.

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
- ADR-148 remains the authoritative legacy contract. Adoption of this ADR
  extends its previously deferred scope; it does not rewrite its historical
  behavior or automatically enable new hooks.

## Links

- [Expanded hook runtime design](../../Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md)
- [Managed plugins design](../../Docs/superpowers/specs/2026-09-15-managed-plugins-design.md)
