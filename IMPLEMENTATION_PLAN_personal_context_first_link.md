# TASK-24727 — Reconcile Personal Context on first link

## Stage 1: Reconciliation contract and RED tests

**Goal:** Pin read-only first-link planning, explicit collision/scope decisions,
write freezing, cancellation, interruption, and retry behavior.

**Success criteria:** Focused reconciliation, first-link, and modal tests fail
because the link service and review surface do not exist.

**Tests:** `Tests/Personal_Context/test_profile_reconciliation.py`,
`Tests/Sync_Interop/test_personal_context_first_link.py`, and
`Tests/UI/test_personal_context_link_modal.py`.

**Status:** Not Started

## Stage 2: Durable reconciliation and canonical rebind

**Goal:** Build encrypted reference/hash plans, persist explicit workspace
scope mappings, journal provisional-to-canonical identity adoption, and replay
concurrent local edits after convergence.

**Success criteria:** No upload occurs before approval; accepted decisions
preserve canonical IDs/versions; cancel leaves both replicas unchanged; retry
is idempotent and interrupted rebind resumes safely.

**Tests:** Reconciliation repository/service and concurrent mutation cases.

**Status:** Not Started

## Stage 3: Authenticated bootstrap, rebaseline, and Settings review

**Goal:** Consume the server bootstrap contract, replace provisional integrity
custody, recompute every Personal Context tag in a versioned full rebaseline,
and expose explicit user review in canonical F9 Settings.

**Success criteria:** Normal push/pull remains disabled until profile identity,
scope map, objects/versions, and cursor are confirmed; unlinked workspace
context stays unavailable to agents; cancellation and attention states are
clear and content-free.

**Tests:** First-link transport and production-shaped Textual modal tests.

**Status:** Not Started

## Stage 4: Verification and review

**Goal:** Complete targeted regressions, static/security gates, task evidence,
and independent review.

**Success criteria:** Scoped Personal Context, Sync_Interop, and UI tests plus
Ruff, compilation, Bandit, diff hygiene, and review pass; TASK-24727 is
complete.

**Status:** Not Started

## ADR check

ADR required: no (existing)

ADR path: `backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md`

Reason: ADR-102 already governs reviewed first-link reconciliation, canonical
scope mappings, provisional identity adoption, key custody, and Sync gating.
