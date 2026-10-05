# ADR-220: Console decision coordination and compaction preflight ownership

Status: Accepted
Date: 2026-10-04
Related Task: TASK-34215.2
Related plan: [PR2995 review and merge](../../Docs/superpowers/plans/2026-10-03-console-pr2995-review-and-merge.md)
Extends: ADR-052, ADR-067, ADR-092, ADR-097 boot budgets and ADR-219
Amends: [Console interrupt-host design](../../Docs/superpowers/specs/2026-08-20-console-interrupt-host-design.md), section3.2

## Context

The integrated Console controller has33695 lines against its existing29367 ratchet. The store's pure fork projections were placed in their existing owner under ADR092 and now meet22344. Newly landed compaction/pre-dispatch behavior also brings ChatScreen to25239 lines/761 methods against25218/759. The source must retain both current dev and the approved creation/start contracts. Size guards stay fixed.

The existing interrupt host coordinates generic rounds but still receives a broad controller receiver. Its original design leaves kind-specific bridge behavior in the controller. Compaction planning/transactions have a resident owner, while controller preflight orchestration still couples to many live services and patch seams. This decision records the explicit contracts for those responsibilities before moving their implementations.

## Decision

### Human decision coordination

Extend the existing InterruptRoundHost in Chat/console_interrupt_rounds.py with kind-specific request/resolve bridges, pending decision projection, attention/remount behavior, permission-summary coordination and review-hook verdict construction. The exact selected88 controller members and17 module helpers are pinned by the plan's proof2/3 maps. Full implementation documentation moves with the actual owner. Retained controller and module APIs preserve their original signatures, defaults, decorators and sync/async shape through accurately documented forwarding functions.

Replace the production host's broad _seams receiver with explicitly named keyword-only live accessors and the one enumerated write-through decision counter. Shared mutable controller state stays in its current owner. Generic registries/payloads/lock remain native host ownership with existing controller alias identity. The separate chat-create lock, source records/observations, actor/current-primary and Close/revocation authority, grants, launch coordinator and durable acceptance remain controller-owned. No new lock, merged lock, generic reflected dependency resolver, proxy/mixin or snapshotted mutable callback is permitted.

Moved-to-moved paths continue to read the original controller/module patch route at invocation. The seven proof2-only private helpers have no repository consumer outside moved bodies and become direct same-owner calls; their full bodies/docs remain. This accepts normal internal-private relocation, not compatibility with arbitrary external private-name monkeypatches. Public/known patched/computed/unbound seams remain. The approval after_remount hook remains the actual bound ConsoleChatController._maybe_fire_permission_summary wrapper. Nullable setters remainNone when absent; controller replacement is visible through named getters. Legacy fake-seam tests use explicit test wiring with unchanged assertions, without restoring broad production receiver access.

Keep existing deadlines, arm-time event identities, non-reentrant locks/order and callback/thread handoffs. No SQLite read, arbitrary callback or token retirement moves into registry critical sections. ADR219's fresh phase-local row observation and atomic runtime/record/Close contract remains exact; the host gains no source or database authority.

### Compaction request preflight

Add ConsoleCompactionPreflight within the existing resident Chat/console_context_compaction.py. It owns the exact five selected implementations: compact_context_now, _apply_conversation_memory_preflight, _assess_context_compaction, _assess_request_capacity_only and _context_overflow_alert. The existing transaction service is unchanged. The controller retains wrappers and the compact maintenance decorator; the decorator executes once at the controller boundary.

The collaborator receives26 named live controller accessors and41 named live module-global accessors; it holds no controller receiver or independent mutable authority. Existing controller context accounting remains a live mapping, with the original item writes through that actual mapping. Store, repository, service, gateway, bridge and optional callback lookup retain their exact objects, callability/None checks, exception behavior and call order. Nested selected-method calls retain the original controller patch route. Full original docs and local imports follow the implementation.

Controller policy/admission/commit fences, failed-attempt latches, bounded auxiliary calls, memory selection and hooks preserve ADR052/054. The early/late compaction admission gate, hold/cancel/resume/answered/after-effects state, prepared/evidence freeze/capture/release, source/start and durable acceptance authority remain in place. The collaborator cannot choose new policy, skip safety windowing, admit a stale compaction result or consume future staged inputs.

### Two new recovery UI adapters

Place only the newly landed private _dispatch_console_trace_recovery and _console_trace_recovery_state implementations with ConsoleSessionController and route the center's two callbacks directly there. This follows [DESIGN.md section7](../../DESIGN.md#7-screen-decomposition) for the existing Session/wiring collaborators. DOM, IDs, CSS, focus and navigation behavior retain their existing implementation. Only one UI controller responsibility moves: the two new recovery adapters.

The region callbacks intentionally become Session-bound. Repository named/computed/borrowed consumers and identity checks were examined; no consumer other than the center's assignments was found for these new private names. Arbitrary external private receiver assertions remain a compatibility limit. The action still obtains actual live screen start-timer and UI-sync methods through named getters, preserves held-text/dispatch/composer/refill/return order, and uses the original dispatch/state module patch routes. No widget snapshot is retained. Two existing direct wiring fixtures receive only explicit reader arguments; original assertions remain.

### Governance and qualification

Original controller29367, screen25218/759, ready1033 and preimport557 gates cannot increase. Actual formatted projections are29336 controller and25194/759 screen; these are proposals, not behavior/boot qualification. Re-measure actual source and apply downward/slack rules. The enlarged interrupt and compaction owners receive truthful initial module-ratchet rows after actual measurement, without increasing any existing row.

Before extraction, characterize any uncovered existing action-time patch/nullability/order behavior with the original actual public wrapper or region callback. After extraction, qualify directly affected interrupt/confirmation/projection/summary/review-helper, compaction/preflight/hold and recovery UI owners, named late patch/nullability/alias/maintenance/callback-identity controls and exact diagnostic/worker/external-writer ownership. Carry unaffected feature/schema/provider and historical QA by exact source/AST maps. One actual final startup/public-navigation qualification follows both structures and latest-dev source integration. Independent scoped reviews and normal current-head Qodo/CI/PerfGuard/latest-dev merge gates remain required.

## Alternatives and consequences

Raising a cap would leave the growth ungoverned. Compressing documentation or wiring would hide responsibilities. Moving creation into its launch coordinator would combine distinct source/acceptance authority and does not fit alone. A generic controller proxy would preserve opaque coupling and violate named dependency governance. Keeping both new screen wrappers cannot meet the method cap.

Named live dependencies add explicit wiring and honest source length in the genuine owners. The store pure placement grows combined source132 lines; the new scopes likewise pursue ownership, not a claim of less total code. The projected headroom is narrow and final additions must be measured. Normal private relocation and changed two private callback receivers may require external private consumers to update. Known public/patch/lifetime/storage/permission contracts remain protected by explicit maps and targeted qualification.


## 2026-10-05 clarification: runtime accessors and annotations

Explicit live accessors cover values observed by executable runtime expressions. Postponed and local type annotations use the owner module's already imported types. The interrupt host's Any and Mapping getters have no runtime expression consumer; remove only those two getter parameters/assignments and matching controller/test wiring, replacing their17 annotation references with imported Any/Mapping. All actual runtime module observations retain their invocation-time controller patch routes. The host keeps120 named keyword-only bindings, including86 controller readers,33 runtime global readers and one enumerated write-through callback; generic/native registry and source/acceptance authority boundaries remain unchanged.
