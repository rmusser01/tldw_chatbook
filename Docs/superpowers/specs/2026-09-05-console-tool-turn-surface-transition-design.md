# Atomic trace transition after a completed tool turn

Date: 2026-09-05

Status: repair direction approved; written contract awaiting review.

Owner: TASK-31742, integration follow-up for PR2432.

ADR required: yes, an amendment to the existing trace-ledger ADR rather than a
second trace architecture.
ADR path: `backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md`.
Reason: this extends the typed cross-module admission/persistence contract, while
retaining the existing schema, immutable ledger and authorization boundaries.

## Problem and approved scope

A real saved Canvas turn completes through progressive discovery, tool loading,
creation and final response. Its next saved prompt cannot start Capture On.
The effective request surface contains the saved user followed by six tool
artifacts; the new request contains that user, the saved assistant and a new user.
The verified common prefix is one item, with six old items and two new items
remaining. Current admission permits replacement **or** append, not both.

A successful calculator turn reproduces the same failure. Ordinary two-turn
history and deliberate changed-history rejection still pass. These observations
come from the rebased feature tree, not an untouched-dev reproduction. Exact
commands and failure evidence are in `Docs/Canvas/V1_VERIFICATION.md`.

The user approved a focused shared trace repair before merging Canvas V1.
No Canvas privilege, tool approval, conversation storage, synchronization or V2
change is included. No migration or additional dependency is needed.

## Chosen approach and alternatives

Add one explicitly distinguished compound transition: replace the completed
run's bounded tool suffix with its verified saved assistant revision, then append
exactly the new saved user revision. Persist both operations and the dispatch
boundary atomically. Keep ordinary append, no-op and single-replacement behavior.

Two alternatives are rejected:

- Persist the replacement during preparation, then append during dispatch. This
  exposes an intermediate surface and complicates cancellation, retry and rollback.
- Accept arbitrary multiple-item replacements, reset the segment, or bypass
  capture. These weaken ownership or historical fidelity to avoid the actual gap.

This is not a general patch language. The new transition accepts exactly one
replacement and one append, in that order, under one verified predecessor.

## Admission contract

The special transition is eligible only for a new `AGENT_FIRST` request with an
unchanged saved-history prefix and a terminal tool-bearing predecessor run.
Existing paths retain their current supported shapes. The new shape requires:

1. The attached owner, conversation, segment and frozen disclosure policy agree
   with the prior run and incoming request. The prior run has a unique durable
   `AGENT_FIRST` origin and a latest `COMPLETE` terminal call at the current head.
2. The removed range is the active contiguous message-domain tool suffix of that
   same run, after its original request surface. Merely being an artifact is not
   sufficient. Range ownership must be proven from durable call boundaries and
   lineage, with the existing `MAX_SURFACE_REPLACEMENT_SPAN = 256` unchanged.
3. The terminal call's response link is a semantic revision with
   `verification_outcome="verified_equal"`. It identifies exactly the incoming
   saved assistant revision, in the same conversation. Resolve and compare the
   complete provider-visible envelope under the existing disclosure rules.
   An artifact response, missing response, or text-only resemblance is insufficient.
4. The second incoming descriptor is exactly the new saved user revision that
   owns this turn. No additional changed descriptors, omitted history, altered
   prefix, continuation-domain replacement, or extra append is admitted by this
   special path. Existing continuation behavior is not broadened or disabled.
5. The exact predecessor, range, final values and terminal-call witness are checked
   again at bind-and-dispatch. The new call's own reservation is expected; any
   other intervening reservation/call or changed surface invalidates the witness.

The response link is evidence, not authority. It cannot substitute for owner,
lineage, predecessor, range, policy or final-value verification. The proof must
also be reconstructable with a fresh factory from committed references; a warm
process cache is an optimization, not authorization.

## Component and transaction boundaries

`console_trace_runtime.py` selects the previous completed-run witness while
resolving the new saved turn. It does not rewrite conversation history or persist
an intermediate projection. Call reservation remains the existing separate,
content-free pre-dispatch operation.

`console_trace_final_values.py` gives admission and verified-delta objects an
explicit compound shape. A caller cannot enable general replacement-plus-items
by bypassing the current exclusive shape check. The preparation identity and
service-owned child capability bind the exact two structural slots, range and
previous-call witness to the final provider values. Provider bytes remain in the
existing bounded/sanitized value bundle, not copied into the admission metadata.

`console_trace_service.py` verifies the composed final projection, then persists
the replacement event/node, new user append, header and new call's
`dispatch_started` binding in the existing immediate transaction owned by
`bind_and_mark_dispatch`. The new call binds only to the final head. There is no
synthetic provider call for the intermediate projection.

Projection and descriptor roots must insert the replacement at the removed
range and append the user after it; neither may accidentally treat the last new
node as the replacement. Commit/rollback handling must keep capability caches
consistent with durable state, including retry after injected failures.

Existing node, replacement, event and response-link tables are sufficient.
Repository changes, if necessary, are bounded, parameterized witness queries;
they do not introduce a second run registry, schema or stored history array.
Historical calls retain their original heads, headers and reconstructed requests.

## Failure and resource behavior

If any proof is unavailable or stale, retain the existing blocked-capture UX:
Retry, explicit Send without capture, or Cancel. No automatic bypass or newly
permissive omission fallback is added for this compound transition.

Failure after either surface write rolls back both writes, the new header/binding
and dispatch transition. The earlier content-free reservation may remain under
the existing failure/recovery lifecycle; it must not look dispatched. Provider
entry is forbidden until the atomic dispatch transaction commits.

New durable metadata is constant-sized per transition: one bounded range, one
replacement reference, one appended reference and the existing call/header data.
Do not add transcript-sized ID lists, raw payload logs, hashes of ordinary text,
eager application imports, or relaxed startup/performance budgets.

## Verification required before merge

- Real calculator and progressive Canvas production-factory two-turn regressions
  pass. Canvas create/update retain separate turn/run ownership and exact parent
  revision linkage. Assert successful tool results, not just final synthetic prose.
- Read back every first-turn call before and after the transition: its original
  request is unchanged. The next call reconstructs exactly saved history plus the
  new user, with no duplicate/omitted message or misordered replacement node.
- Repeat with a fresh trace factory and across multiple completed tool turns.
  Existing ordinary-history, continuation and changed-history controls still pass.
- Reject wrong owners/policies, stale heads, intervening runs/reservations,
  incomplete terminal calls, artifact/missing/wrong response links, modified
  assistant envelopes, unproven artifact ownership, altered history, noncontiguous
  or oversized ranges, and extra incoming rewrites. No rejection enters transport.
- Inject failures after replacement, append and bind; assert durable rollback,
  unchanged historical reads and successful valid retry without stale capabilities.
- Verify constant per-turn structural growth and rerun the existing affected trace
  storage/performance gates without raising thresholds. Run targeted trace,
  gateway/controller, Canvas, startup and browser checks, not the full repository.
- Independent code review, current-base rebase, new Qodo feedback disposition and
  protected current-head CI remain mandatory before the requested merge. Start V2
  only after GitHub confirms that merge.

## Review state

Self-review checked scope, atomicity, two-phase reservation ordering, cold-cache
authority, projection ordering and rollback. No runtime implementation is included
in this design commit. The retained diagnostic tests intentionally remain failing
and uncommitted until the repair's test-first implementation cycle.
