---
id: TASK-24727
title: Reconcile Personal Context on first link
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-30 22:21'
updated_date: '2026-08-30 22:25'
labels:
  - personal-context
  - sync
  - security
dependencies:
  - TASK-24726
references:
  - >-
    backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md
documentation:
  - Docs/superpowers/plans/2026-08-28-personal-context-04-sync-multidevice.md
  - IMPLEMENTATION_PLAN_personal_context_first_link.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Perform a cancellable reviewed first-link reconciliation that adopts the server canonical profile identity, maps workspace scopes explicitly, replaces provisional integrity custody, and enables normal Sync only after confirmed convergence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 First-link planning is read-only, encrypted, and blocks upload until the user approves exact profile, scope, object, lineage, quota, and purge-generation outcomes.
- [ ] #2 Applying approved decisions adopts the server canonical profile ID, preserves canonical object IDs/versions, persists explicit workspace-scope mappings, and remains resumable after interruption.
- [ ] #3 The authenticated wrapped bootstrap replaces the provisional integrity key and completes a versioned full integrity rebaseline before ordinary push or pull is enabled.
- [ ] #4 The Settings review surface supports cancel, retry, collision review, unlinked workspace handling, and concurrent local mutation without exposing profile content in logs or Sync metadata.
- [ ] #5 Targeted reconciliation, first-link, and modal tests plus Ruff, compilation, Bandit, diff hygiene, and independent review pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add RED reconciliation, first-link transport, and production-shaped modal tests.
2. Implement encrypted read-only reconciliation plans, explicit scope mapping, write freeze, and resumable canonical rebind.
3. Consume authenticated wrapped bootstrap, replace provisional integrity custody, and run a versioned full rebaseline before normal Sync.
4. Add the F9 Settings review surface with cancel, retry, attention, collision, and unlinked-workspace behavior.
5. Run targeted tests, Ruff, compilation, Bandit, diff hygiene, independent review, and commit.

ADR required: no (existing)
ADR path: backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md
Reason: ADR-102 already governs reviewed first-link reconciliation, scope mappings, identity adoption, key custody, and Sync gating.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Targeted tests and verification recorded
- [ ] #3 Documentation updated
- [ ] #4 Static and security checks pass
- [ ] #5 Independent review completed
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes

The Chatbook first-link slice is implemented and remains In Progress pending
controller verification and cross-repository review. It uses the canonical
Personal Context records and versions directly; no Chatbook/server projection or
second authoritative profile was introduced.

- Added typed authenticated bootstrap/completion transport, secure RSA wrapping
  and staged integrity custody, distinct persistent dataset-staging-key custody,
  exact durable link receipts, and restart-safe runtime composition.
- Added a durable exact-snapshot review freeze. Conservative v1 permits reads but
  rejects ordinary user/agent profile mutations with
  `personal_context_link_in_progress` until cancel, terminal attention, or exact
  convergence releases it.
- Added schema-directed canonical identity rebind, full versioned integrity-tag
  rebaseline, exact first-link materialization journals, local lineage upload,
  same-ID merge lineage, remote-loser tombstones, proposal attention, and
  content-safe verification.
- Added explicit unlinked/new/mapped workspace outcomes, preallocated reviewed
  canonical scope IDs, one-to-one bindings, device-only privacy protection, and
  mapping-created collision prevention before approval.
- Added a dedicated bounded first-link push/pull cycle with negotiated batch size,
  include-own confirmation, separate bootstrap and confirmed cursors, exact-head
  verification, stale destination cleanup, and fail-closed public Personal
  Context push/pull. Ordinary LocalFirst transport requires the exact complete
  binding.
- Added canonical F9 Settings review, attention, retry, cancel, interrupted resume,
  and linked-state behavior with exact content-free identity/version/quota rows.

Bootstrap may reserve content-free server control-plane scaffolding required by
the approved server contract. Planning/cancel performs no canonical Personal
Context content upload or mutation; cancellation leaves both content replicas
unchanged and releases local freeze/staging.

Verification: the latest touched-scope run completed with 285 passed and 2
dependency warnings. Ruff, compilation, CSS reproduction, both diff-hygiene
checks, and the Bandit high-severity gate pass. The full Bandit report contains
no high-severity issue and retains the repository's known constant-identifier SQL
and low-severity exception/assert/subprocess findings. Exact commands and counts
are recorded in the SDD report. The full repository suite and live
server/keyring/TUI testing were not run, per scoped-verification policy. ADR-102
remains the governing decision.
