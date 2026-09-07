# Chat and Console Handoff Ownership Implementation Plan (TASK-645)

**Status:** Implemented and reconciled with current `dev`.

**Goal:** Replace raw Chat and Console pending fields with typed, memory-only,
revisioned single-slot handoffs that preserve the latest replacement and settle
only the exact claimed revision.

**Architecture:** `PendingHandoffStore` owns independent typed channels with one
in-flight claim and at most one latest replacement per channel. Producers stage
before navigation. Consumers acknowledge after a terminal result or successful
screen-local ownership transfer and release retryable failures or cancellation.
The store is memory-only and owner-thread-affine.

**Current production boundary:** TASK-577 retired `ChatTabContainer` before this
task was integrated. The registered Chat route is the native Console. No
ephemeral-tab creation or rollback contract remains. A generic Chat handoff
transfers into Console staged live work; a valid character handoff creates a
native character-bound Console session.

**Backlog:** [TASK-645](../../../backlog/tasks/task-645%20-%20Move-Chat-and-Console-handoffs-behind-revisioned-single-slot-ownership.md)

**Specification:** [Application Session State Ownership Design](../specs/2026-07-26-application-session-state-ownership-design.md)

**ADR required:** yes

**ADR path:** `backlog/decisions/033-application-session-state-ownership.md`

**Reason:** ADR-033 defines the cross-screen single-slot delivery,
replacement, settlement, privacy, and thread-affinity contract.

## Implementation

1. Add `PendingHandoffStore`, typed channel values, detached stage/claim
   snapshots, monotonic revisions, exact acknowledge/release, and owner-thread
   enforcement.
2. Migrate Chat context, Console live-work, Console prompt-insert, and Console
   provider producers to typed staging.
3. Transfer Console live work into screen-local state and acknowledge only
   after that transfer succeeds.
4. For native Chat delivery, acknowledge only after character-session creation
   or staged-live-work transfer; release the exact claim on failure or
   cancellation so a newer replacement survives.
5. For Console prompt insertion, acknowledge persistent setup rejection,
   release transient composer readiness and exceptions, and acknowledge after
   text lands.
6. Remove all raw application pending fields and add static ownership guards.

## Verification

- Test app-independent store/model behavior directly in
  `Tests/State/test_pending_handoff_store.py`.
- Test application behavior only through a normal production `TldwCli()`,
  `app.run_test()`, the registered production `ChatScreen`, and its real native
  Console widgets in `Tests/ProductionApp/`.
- Use async events for replacement, failure, and cancellation boundaries.
- Verify claim exclusivity, replacement survival, stale settlement rejection,
  owner-thread enforcement, structural detachment, memory-only storage, and
  metadata-only failure diagnostics.
- Run `Tests/test_application_state_ownership.py` to reject raw-field or owner
  bypass regressions.
- Include the production-app/state tests in the final integrated application
  state gate and run installed-distribution verification before merging.

Tests must not use a custom or simplified application, surrogate screen,
unbound `TldwCli` method, `SimpleNamespace`, `MagicMock`, or
`object.__new__(TldwCli)`.
