# ADR-154: Agent denial streak and completed-batch boundary

Status: Accepted
Date: 2026-09-12
Related task: TASK-18929
Related decisions: ADR-078 (structured agent tool outcome provenance), ADR-131 (durable budget accounting), ADR-134 and ADR-135 (fleet admission and automatic work)
Supersedes: N/A

## Context

Repeated denied tool requests can spend turns without progress. The existing ADR-078 `blocked` outcome intentionally includes approval and permission refusals alongside timeout, kill-switch, workspace-root and other policy blocks. That display contract cannot prove an explicit user Deny or configured permission Off. Existing review hooks flatten their decisions to strings, sometimes using the same refusal copy for answered and unresolved decisions. Inferring control from that text would misattribute denials and could let successful denial-looking payloads manipulate the loop.

The runtime also retains history only through its last coherent tool-batch boundary. Returning in the middle of a native multi-call batch drops that batch from retained history, while inventing skipped replies would obscure already-approved dispatch semantics. A denial guard therefore needs both precise authority and an explicit enforcement boundary.

Accepted ADR-078 is unchanged. This decision adds compatible control facts beside its existing broad outcome contract.

## Decision

1. Add optional bounded `approval_decision` metadata (`approved`, `denied`, or absent) to internal ToolResult as a keyword-only field. Existing `ok/content/error/outcome` positional construction and blocked presentation remain compatible.
2. Add `ToolReviewDecision(verdict, approval_decision)` as an explicit review value. Accept legacy string values alongside it. Resolve call-id before tool-name fallback exactly as today; normalize only the selected value. Legacy strings retain existing dispatch/refusal behavior but supply no new denial authority. Do not use a string subclass, error-text parsing or display classification to obtain authority.
3. Owners attach denied only for an explicit user Deny or configured permission Off. Defaulted unanswered denials, timeout, missing callback, unresolved gate errors, unavailable authority, root changes, cancellation and synthetic restore deferrals carry no denied fact. Owners may preserve approved provenance when approved execution subsequently fails. A fact never grants permission or replaces existing stamps/gates.
4. A runtime invocation starts its own denial counter at zero. A settled authoritative denial increments once. Success (`ok=True`), approval and every non-denial result reset it. Success is authoritative even when content resembles a denial. A later actual invocation denial supersedes an earlier approval; default dispatch eligibility is not an approval event.
5. `[agents] denial_circuit_breaker_limit` defaults to integer 3. Explicit integer 0 disables it. Invalid values fall back to 3. Resolve freshly per run and preserve the limit through child budgets and admitted/live intersections, treating zero as no ceiling.
6. Evaluate only the trailing streak after all calls in the already-reviewed batch settle and their replies are appended. An approved/successful tail resets the streak. A batch of four denials at limit3 stops with observed count4. Stop before another model call, without a summary request, and retain the complete coherent history including that batch.
7. Preserve existing cancellation and continuation-persistence failure precedence. Do not modify unrelated repeat guards. Replayed completed continuation results are not fresh decisions; synthetic restored_pending is not a denial. Newly invoked calls during restoration use the same authoritative rules.
8. Emit one STEP_ERROR and a run-local error log record naming the observed count, then RUN_STUCK. Reuse Console failure finalization to preserve partial text, emit a System explanation and allow retry. An optional additive terminal `RunOutcome.denial_count` identifies this outcome without parsing summary copy; it is not a persisted counter. Children retain their own terminal outcome/log and use existing completion delivery without marking siblings or the supervisor denial-stuck.

## Persistence and boundaries

The denial streak remains ephemeral. New/manual/resumed runtime invocations start at zero; no transcript-derived reconstruction or shared service/fleet counter exists. Existing steps JSON and RunBudget snapshot mechanisms support additive metadata; no SQLite column, schema migration, provider wire change, dependency or external service contract is needed. Durable accounting under ADR-131 and resource/admission ownership under ADR-134/135 remain unchanged.

## Alternatives considered

- Count every blocked result: rejected because blocked intentionally includes non-denial policy states under ADR-078.
- Parse canonical refusal strings: rejected because shared copy loses answer authority and successful content can collide with denial text.
- Return immediately at the threshold mid-batch: rejected because it discards the in-flight batch from coherent retained history or requires fabricated skipped results and new revocation semantics.
- Latch a threshold trip despite an approved batch tail: rejected because completed-batch trailing-streak semantics preserve reset-on-approval behavior and give an honest terminal count.
- Persist a shared streak across children or resumes: rejected because one child's refusal must not stop siblings and a new user retry must start fresh.
- Amend accepted ADR-078: rejected because accepted ADRs are immutable except nonsemantic corrections; this is a compatible extension decision.

## Verification requirements

Targeted tests must cover legacy string/structured review compatibility, authoritative explicit versus unanswered denials, permission-Off versus no_callback with shared copy, approved execution failures, successful denial-looking output, default3/explicit0/invalid values, child budget/intersection propagation, call-id granularity, mixed/all-denied batches, both native and fence paths, continuation/restore/cancellation precedence, complete retained replies, no additional model call, actual Console System row/partial preservation/retry, real run-log identity/count, and sibling/supervisor isolation.

Deterministic runtime tests do not alone establish Console or persistence behavior. Use the real controller/store, service/local JSONL and SQLite boundaries for those claims. No full sweep or live-provider validation is implied by this decision.

## Links

- [ADR-078 — Structured agent tool outcome provenance](078-structured-agent-tool-outcome-provenance.md)
- [TASK-18929 — Agent loop consecutive-denial circuit breaker](../tasks/task-18929%20-%20Agent-loop-consecutive-denial-circuit-breaker.md)
- [Design](../../Docs/superpowers/specs/2026-09-12-agent-denial-breaker-design.md)
- [Plan](../../Docs/superpowers/plans/2026-09-12-agent-denial-breaker.md)
