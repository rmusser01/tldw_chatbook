# Atomic trace transition after a completed tool turn

Date: 2026-09-05

Status: repair direction approved; requested review corrections incorporated.

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

The special transition is eligible for a new `AGENT_FIRST` or ordinary `FRESH`
request with an unchanged saved-history prefix and a terminal tool-bearing
predecessor run. Turning agent mode off after a completed tool turn must not
prevent the next ordinary saved prompt from being captured. Eligibility is based
on the verified surface and ownership, not whether the next response uses tools.
Other routes, including `DIRECT_PREFILL`, `REGENERATE`, `EDIT`, `CONTINUE` and
fallback routes, gain no new compound-transition permission. Existing paths
retain their current supported shapes. The new shape requires:

1. The attached owner, conversation, segment and frozen disclosure policy agree
   with the prior run and incoming request. The prior run has a unique durable
   `AGENT_FIRST` origin and a latest `COMPLETE` terminal call at the current head,
   apart from the exact owned pre-dispatch reservation described below.
2. The removed range is the active contiguous message-domain tool suffix of that
   same run, after its original request surface. Merely being an artifact is not
   sufficient. Range ownership must be proven from durable call boundaries and
   lineage, with the existing `MAX_SURFACE_REPLACEMENT_SPAN = 256` unchanged.
3. The terminal call's response link is a semantic revision with
   `verification_outcome="verified_equal"`. It identifies exactly the incoming
   saved assistant revision, in the same conversation. Resolve and compare the
   complete provider-visible envelope under the existing disclosure rules.
   An artifact response, missing response, or text-only resemblance is insufficient.
4. The second descriptor in the changed suffix is exactly the new saved user
   revision that owns this turn. No additional changed descriptors, omitted history, altered
   prefix, continuation-domain replacement, or extra append is admitted by this
   special path. Existing continuation behavior is not broadened or disabled.
5. The exact predecessor, range, final values and terminal-call witness are checked
   again at bind-and-dispatch. The new call's own reservation is expected, including
   a verified reuse of that exact reservation on pre-dispatch Retry. Any unrelated
   intervening reservation/call or changed surface invalidates the witness.

The response link is evidence, not authority. It cannot substitute for owner,
lineage, predecessor, range, policy or final-value verification. The proof must
also be reconstructable with a fresh factory from committed references; a warm
process cache is an optimization, not authorization.

### Owned pre-dispatch Retry

Retry of a failed bind must not blindly call the factory to allocate another
reservation: the first reservation is already a durable ordered event. The
controller/gateway recovery handoff must identify the exact failed boundary and
its accepted-turn authority. Reuse that call's immutable logical identity and
idempotency key after reconstructing a fresh verified admission; never infer
ownership from equal text, matching conversation/turn IDs alone, or the newest
reservation found by a query.

Re-admission requires a durable read proving that the exact call is still
`RESERVED`, has no bound surface/header or dispatch/response timestamp, and has
no newer unrelated call boundary. Validate the same accepted user revision,
frozen request, route, destination and policy through the existing recovery
authority, then revalidate the prior completed run and unchanged predecessor.
Only this exact reservation is exempt from the latest-completed-call check.
Repeated failed binds reuse it; they must not accumulate a list of skipped calls
or extra call-boundary events.

This requires a narrow controller/gateway-to-factory recovery handoff, not a
generic caller-supplied bypass flag. The original service capability must be
invalidated or retired before replacement so that old and new admissions cannot
both dispatch. Cold recovery must reconstruct authority from the existing durable
accepted-turn recovery state; if that state cannot establish the exact call and
frozen request, retain the blocked recovery outcome instead of guessing.

A `NOT_DISPATCHED` call is terminal and cannot be revived by this rule. A call
with committed dispatch, response activity or unknown status is not an eligible
pre-dispatch retry. Unrelated or ambiguous reservations remain blockers; this
repair does not add a scan that skips all unsuccessful calls.

## Component and transaction boundaries

`console_trace_runtime.py` selects the previous completed-run witness while
resolving the new saved turn. It does not rewrite conversation history or persist
an intermediate projection. Call reservation remains the existing separate,
content-free pre-dispatch operation. The exact owned recovery path above reuses
that reservation rather than creating a second one.

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

An error before commit must roll back both surface writes, the new header/binding
and dispatch transition. The earlier content-free reservation remains available
for exact owned recovery. However, an exception alone is not proof of rollback:
connection-setting restoration or a wrapper can raise after the transaction has
committed. Provider entry is forbidden until the exact atomic dispatch outcome is
established.

### Commit-outcome reconciliation

On an uncertain write result, read back the exact reserved call identity and its
surface/header binding through a usable transaction-owning connection. Verify the
expected composed projection and predecessor relationship, not merely a terminal
status or a row with the same turn. Reconcile before exposing Retry/Cancel actions
or allowing a replacement admission to reach the provider:

| Durable outcome | Required handling |
| --- | --- |
| Rolled back: exact call remains unbound `RESERVED`, with the original surface unchanged | Discard tentative projection/capability state and permit the owned pre-dispatch Retry described above. |
| Committed: exact call and expected final surface/header are atomically bound with dispatch started | Preserve those records and reconcile the boundary's in-memory state. Do not append again, create another call, or mark it `NOT_DISPATCHED`. |
| Unavailable, inconsistent, or otherwise unknown | Fail closed; preserve the reservation and report the existing uncertain-delivery recovery state. Do not treat missing evidence as rollback or automatically redispatch. |

For a proven committed outcome, the original live invocation may proceed only
if it still owns the exact unconsumed gateway adapter-entry grant and can prove
adapter entry has not occurred. Consume that grant once. A new invocation or cold
recovery cannot infer non-delivery from the absence of a response; retain existing
explicit uncertain-delivery handling instead. A wrapper error after adapter entry
must not authorize a duplicate request. This does not promise exactly-once remote
delivery across crashes or network failures.

Post-commit cleanup errors must not hide the committed result or cause capture
cancellation to attempt an illegal `DISPATCH_STARTED` to `NOT_DISPATCHED`
transition. Cache reconciliation follows durable outcome: discard tentative state
on rollback, retain/rebuild the committed final head on success, and invalidate
uncertain capabilities until read-back establishes their owner and head.

New durable metadata is constant-sized per transition: one bounded range, one
replacement reference, one appended reference and the existing call/header data.
Do not add transcript-sized ID lists, raw payload logs, hashes of ordinary text,
eager application imports, or relaxed startup/performance budgets.

## Verification required before merge

- Real calculator and progressive Canvas production-factory two-turn regressions
  pass. Canvas create/update retain separate turn/run ownership and exact parent
  revision linkage. Assert successful tool results, not just final synthetic prose.
- Cover both `AGENT_FIRST` and `FRESH` as the next send, including disabling agent
  mode after the completed tool turn. Unsupported route/prefill combinations must
  remain rejected by the compound path without regressing existing route behavior.
- Read back every first-turn call before and after the transition: its original
  request is unchanged. The next call reconstructs exactly saved history plus the
  new user, with no duplicate/omitted message or misordered replacement node.
- Repeat with a fresh trace factory and across multiple completed tool turns.
  Existing ordinary-history, continuation and changed-history controls still pass.
- Reject wrong owners/policies, stale heads, unrelated intervening runs/reservations,
  incomplete terminal calls, artifact/missing/wrong response links, modified
  assistant envelopes, unproven artifact ownership, altered history, noncontiguous
  or oversized ranges, and extra incoming rewrites. No rejection enters transport.
- Inject failures after replacement, append and bind but before commit; assert durable rollback,
  unchanged historical reads and successful valid retry without stale capabilities.
- Drive Retry through the real controller/gateway/factory after failed binds,
  including repeated failures. Assert reuse of the exact reservation, no additional
  call-boundary events, successful valid recovery and rejection of a stale admission.
  Exercise foreign reservations, wrong accepted-turn authority, terminal
  `NOT_DISPATCHED` records and missing cold-recovery proof as negative controls.
- Inject an error after a successful commit, including connection-setting cleanup,
  and distinguish it from pre-commit rollback. Read back both surface operations and
  the exact dispatch binding; assert no duplicate writes, calls, adapter entries or
  illegal cancellation transition. Also inject reconciliation-read failure and an
  error after adapter entry: neither may automatically redispatch. Test the original
  live one-shot grant separately from a new/cold invocation with unknown delivery.
- Verify constant per-turn structural growth and rerun the existing affected trace
  storage/performance gates without raising thresholds. Run targeted trace,
  gateway/controller, Canvas, startup and browser checks, not the full repository.
- Independent code review, current-base rebase, new Qodo feedback disposition and
  protected current-head CI remain mandatory before the requested merge. Start V2
  only after GitHub confirms that merge.

## Review state

Self-review checked scope, atomicity, two-phase reservation ordering, cold-cache
authority, projection ordering and rollback. The requested review amendments add
exact-reservation retry ownership, committed/rolled-back/unknown outcome
reconciliation, actual controller/gateway recovery tests and next-send `FRESH`
coverage. They do not authorize arbitrary historical-call skipping, terminal-call
revival, routes beyond the two eligible next-send routes, or automatic
uncertain-delivery retries.

No runtime implementation is included in this design revision. The retained
diagnostic tests intentionally remain failing and uncommitted until the repair's
test-first implementation cycle. Detailed planning must include the recovery
handoff and reconciliation seams as well as the surface-delta types and persister.
