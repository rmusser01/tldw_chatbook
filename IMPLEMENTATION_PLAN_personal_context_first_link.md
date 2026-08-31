# TASK-24727 — Reconcile Personal Context on first link

## Stage 1: Reconciliation contract and RED tests

**Goal:** Pin read-only first-link planning, explicit collision/scope decisions,
write freezing, cancellation, interruption, and retry behavior.

**Success criteria:** Focused reconciliation, first-link, and modal tests fail
because the link service and review surface do not exist.

**Tests:** `Tests/Personal_Context/test_profile_reconciliation.py`,
`Tests/Sync_Interop/test_personal_context_first_link.py`, and
`Tests/UI/test_personal_context_link_modal.py`.

**Status:** Complete (Chatbook slice)

## Stage 2: Durable reconciliation and canonical rebind

**Goal:** Build encrypted reference/hash plans, persist explicit workspace
scope mappings, acquire a durable exact-snapshot write freeze, and journal
provisional-to-canonical identity adoption.

**Success criteria:** No canonical Personal Context content upload occurs before
approval; accepted decisions preserve canonical IDs/versions; cancel leaves both
content replicas unchanged; retry is idempotent and interrupted rebind resumes
safely. Conservative v1 blocks ordinary user/agent profile mutations with the
stable `personal_context_link_in_progress` reason while allowing reads; it does
not claim to journal and replay concurrent writes.

**Tests:** Reconciliation repository/service and concurrent mutation cases.

**Status:** Complete (Chatbook slice)

## Stage 3: Authenticated bootstrap, rebaseline, and Settings review

**Goal:** Consume the server bootstrap contract, replace provisional integrity
custody, recompute every Personal Context tag in a versioned full rebaseline,
and expose explicit user review in canonical F9 Settings.

**Success criteria:** Normal push/pull remains disabled until profile identity,
scope map, objects/versions, and cursor are confirmed; unlinked workspace
context stays unavailable to agents; cancellation and attention states are
clear and content-free.

**Tests:** First-link transport and production-shaped Textual modal tests.

**Status:** Complete (Chatbook slice)

**Cross-repository contract pin:** tldw_server commit `a92e12110d` preserves the
successful bootstrap shape and adds strict, content-free HTTP 409 attention for
schema range, quota shortfall, and purge-generation mismatch. Chatbook accepts
only those discriminated typed shapes; malformed or inconsistent bodies remain
generic failures and never enter the trusted Settings review surface.

## Stage 4: Verification and review

**Goal:** Complete targeted regressions, static/security gates, task evidence,
and independent review.

**Success criteria:** Scoped Personal Context, Sync_Interop, and UI tests plus
Ruff, compilation, Bandit, diff hygiene, and review pass; TASK-24727 is
complete.

**Status:** In Progress — Chatbook targeted tests pass (285 tests); Ruff,
compilation, CSS reproduction, both diff-hygiene checks, and the Bandit
high-severity gate pass. Controller-owned independent cross-repository review
remains before task closure.

### Independent-review remediation (2026-08-30)

- Added durable `reconciling` state, immutable bootstrap receipt data, a separate
  confirmed cursor, exact canonical head sets, and a dedicated complete-gated
  push/pull cycle with `include_own_changes=True`.
- Normal dispatch and pull now require the complete exact device/dataset/profile/
  key/purge/cursor binding. Sync-profile bootstrap merges existing generic Sync
  metadata and rejects dataset/device replacement.
- Rebuilt the reviewed outbox as an exact canonical materialization journal,
  including local-only history, explicit same-ID merge lineage, and remote-loser
  tombstones. Arbitrary user strings are no longer identity-rewritten.
- Added explicit `unlinked` workspace handling, random identities for `new`,
  one-to-one mapping enforcement, post-map collision fail-closed validation, and
  in-transaction authenticated snapshot/binding revalidation.
- Added secure persistent dataset-staging-key custody distinct from profile keys,
  restart loading without silent regeneration, and lazy production runtime wiring.
- Acquires a durable repository freeze before review and releases it on cancel,
  terminal attention, successful convergence, and expired-review recovery. The
  authenticated snapshot covers scope bindings and proposal hashes and is
  revalidated inside the apply write transaction.
- The dedicated first-link cycle drains the exact reviewed journal in negotiated
  bounded batches, calls server completion only for the immutable bootstrap
  receipt, confirms with `include_own_changes=True`, and opens ordinary Sync only
  after exact canonical heads and the final cursor match.
- Public Personal Context push and pull are fail closed. Only the exact reviewed
  reconciliation path and exact-complete LocalFirst path can reach private
  Personal Context transport wrappers.
- Exact preallocated `new` workspace scope IDs, device-only identity collisions,
  and mapping-created semantic collisions are shown or disabled before approval.
  Stale destination outbox copies are removed only for the exact pending Personal
  Context binding, including apply-crash recovery.

Bootstrap necessarily reserves content-free server control-plane scaffolding
(device, dataset, canonical authority/profile/key binding). "Planning is
read-only" and "cancel leaves both replicas unchanged" mean that no canonical
Personal Context record/proposal/scope/manifest content is uploaded or mutated;
cancel also releases the local freeze/staging state. This does not claim that the
approved bootstrap contract performs zero remote control-plane writes.

**Remediation status:** In Progress — targeted implementation is green; controller
review remains required before closure.

### Structured bootstrap attention integration (2026-08-30)

- Strictly parses the three server `a92e12110d` attention variants and rejects
  extra fields, wrong discriminators/error codes, coercion, and inconsistent
  content-free values.
- Carries only the typed attention object across the link-service boundary; raw
  error messages and response bodies are neither logged nor presented.
- Canonical F9 Settings renders exact schema bounds, required/server quota
  values and deficits, or expected/current purge generations in the incumbent
  protected modal. Approval remains disabled; retry and cancel remain available.
- Malformed 409 responses fall back to the existing content-safe generic
  notification. They cannot create review/freeze/link state.

**Integration status:** Complete in Chatbook commit `f42c173c55`; controller
review remains required before TASK-24727 closure.

### Final contract-quality remediation (2026-08-30)

- Bootstrap cursors are now immutable review receipts only. First-link push/pull
  starts from an existing ordinary transport cursor only when its persisted
  device and dataset binding match exactly; otherwise it performs a full pull.
  Planning and cancellation preserve the existing ordinary cursor and metadata.
- The entire source outbox dispatch/quarantine/ack boundary and every confirming
  inbound apply run under the exact durable `plan_id` reconciliation authority.
  Missing plan-scoped authority fails closed; cross-database crash replay remains
  idempotent under the production repositories.
- Key unwrap, secure staging, local apply, cleanup, and restart recovery now leave
  a supported content-free attention state and release the exact review freeze.
  Locked restart recovery can release only the matching content-free freeze even
  when the encrypted profile facade cannot compose.
- Replanning preserves already adopted, locally bound canonical workspace scope
  identities. Retained valid Undo artifacts rebind their schema identity fields
  and remain usable after canonical adoption.
- Capability negotiation allows only schema/quota readiness mismatches to reach
  the authenticated typed bootstrap-attention response. Every mixed or unrelated
  auth/device/integrity/transport blocker still fails before device registration.

**Remediation verification:** 322 broad affected tests pass across Personal
Context, Sync/API, and canonical Settings/UI groups. Ruff, compilation, CSS
reproduction, Bandit high-severity, and both diff-hygiene gates pass. TASK-24727
remains In Progress for controller review and cross-repository closure.

## ADR check

ADR required: no (existing)

ADR path: `backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md`

Reason: ADR-102 already governs reviewed first-link reconciliation, canonical
scope mappings, provisional identity adoption, key custody, and Sync gating.
